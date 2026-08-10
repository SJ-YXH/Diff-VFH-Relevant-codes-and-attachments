#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-speed 2D Diff-VFH-style controller for AirSim + PX4 + ROS2.

This node keeps the existing hw_insight velocity execution chain:
    /airsim_node/PX4/lidar/Lidar  -> this node
    /airsim_node/PX4/odom_local_ned -> this node
    this node -> /hw_insight/keyboard_velocity -> move_velocity -> PX4

The planner is intentionally planar. LiDAR points are projected to x-y,
Diff-VFH-style sector costs choose a safe heading, and an independent z-hold
loop keeps the UAV near target_altitude_ned.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, LaserScan
from std_msgs.msg import String, ColorRGBA
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray

try:
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:  # pragma: no cover
    pc2 = None

from hw_interface.msg import HWSimpleKeyboardInfo


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_to_yaw(q) -> float:
    # Standard yaw extraction. The odometry topic used here is expected to be local NED.
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class AirsimDiffVFHController(Node):
    def __init__(self) -> None:
        super().__init__('airsim_diff_vfh_controller')

        # Topics
        self.declare_parameter('lidar_topic', '/airsim_node/PX4/lidar/Lidar')
        self.declare_parameter('odom_topic', '/airsim_node/PX4/odom_local_ned')
        self.declare_parameter('cmd_topic', '/hw_insight/keyboard_velocity')
        self.declare_parameter('goal_topic', '/diff_vfh/goal_ned')
        self.declare_parameter('scan_topic', '/diff_vfh/scan')
        self.declare_parameter('status_topic', '/diff_vfh/status')
        self.declare_parameter('marker_topic', '/diff_vfh/markers')
        self.declare_parameter('path_topic', '/diff_vfh/path')

        # Frames and control mode
        self.declare_parameter('base_frame_id', 'PX4_lidar')
        self.declare_parameter('odom_frame_id', 'odom_local_ned')
        self.declare_parameter('enable_control', False)
        self.declare_parameter('publish_zero_when_disabled', True)
        self.declare_parameter('output_frame', 'ned')  # 'ned' or 'body'. Use 'ned' for move_velocity TrajectorySetpoint chain.
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('sensor_timeout', 0.7)
        self.declare_parameter('goal_timeout', 0.0)  # 0 disables timeout

        # LiDAR projection / LaserScan
        self.declare_parameter('scan_angle_min_deg', -180.0)
        self.declare_parameter('scan_angle_max_deg', 180.0)
        self.declare_parameter('n_sectors', 181)
        self.declare_parameter('range_min', 0.25)
        self.declare_parameter('range_max', 18.0)
        self.declare_parameter('height_min', -1.50)
        self.declare_parameter('height_max', 0.80)
        self.declare_parameter('use_height_filter', True)
        self.declare_parameter('downsample_step', 1)

        # High-speed planar VFH parameters
        self.declare_parameter('vehicle_radius', 0.45)
        self.declare_parameter('safety_margin', 0.55)
        self.declare_parameter('direction_window_deg', 12.0)
        self.declare_parameter('goal_weight', 1.55)
        self.declare_parameter('clearance_weight', 1.20)
        self.declare_parameter('inertia_weight', 0.35)
        self.declare_parameter('front_bias_weight', 0.10)
        self.declare_parameter('blocked_clearance', 0.15)
        self.declare_parameter('emergency_distance', 0.95)
        self.declare_parameter('hard_stop_distance', 1.25)
        self.declare_parameter('slow_distance', 4.0)
        self.declare_parameter('fast_clearance', 8.0)

        self.declare_parameter('cruise_speed', 4.2)
        self.declare_parameter('min_high_speed', 3.2)
        self.declare_parameter('max_command_xy', 6.0)
        self.declare_parameter('min_command_xy', 0.20)
        self.declare_parameter('goal_slow_radius', 5.0)
        self.declare_parameter('goal_tolerance', 1.0)
        self.declare_parameter('z_goal_tolerance', 0.25)
        self.declare_parameter('disable_on_arrival', True)

        # Deployment smoothing
        self.declare_parameter('max_xy_accel', 3.0)
        self.declare_parameter('max_xy_decel', 5.0)
        self.declare_parameter('command_filter_tau', 0.16)
        # yaw_rate_cmd is the fallback/manual yaw command.
        # For obstacle avoidance, auto_yaw_enable should usually be True so the
        # drone rotates toward the selected safe heading instead of sliding sideways.
        self.declare_parameter('yaw_rate_cmd', 0.0)
        self.declare_parameter('auto_yaw_enable', True)
        self.declare_parameter('yaw_kp', 1.20)
        self.declare_parameter('max_yaw_rate_cmd', 0.85)
        self.declare_parameter('yaw_deadband_deg', 3.0)
        self.declare_parameter('yaw_slowdown_start_deg', 20.0)
        self.declare_parameter('yaw_slowdown_stop_deg', 75.0)
        self.declare_parameter('min_yaw_alignment_speed_scale', 0.18)

        # Independent altitude hold. NED z=-2 means about 2 m above start.
        self.declare_parameter('target_altitude_ned', -2.0)
        self.declare_parameter('use_goal_z_as_target', True)
        self.declare_parameter('z_hold_kp', 0.65)
        self.declare_parameter('z_hold_kd', 0.25)
        self.declare_parameter('z_deadband', 0.05)
        self.declare_parameter('max_z_command', 0.45)
        # move_velocity.py sends TrajectorySetpoint.velocity directly to PX4.
        # PX4 local NED velocity uses negative z for upward flight.
        self.declare_parameter('z_command_sign', 1.0)

        # Initial goal is optional. GUI or ros2 topic pub may overwrite it.
        self.declare_parameter('default_goal_x', 18.0)
        self.declare_parameter('default_goal_y', 0.0)
        self.declare_parameter('default_goal_z', -2.0)
        self.declare_parameter('use_default_goal', False)

        # Automatic takeoff.  This keeps the launch command unchanged and avoids
        # using the GUI to control takeoff.  The controller waits for odometry,
        # sets a vertical goal above the current x-y position, and enables itself.
        self.declare_parameter('auto_takeoff_on_start', True)
        self.declare_parameter('auto_takeoff_height_m', 2.0)
        self.declare_parameter('auto_takeoff_delay_sec', 2.0)
        self.declare_parameter('auto_takeoff_wait_for_lidar', False)

        self._load_params()

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_default = QoSProfile(depth=10)

        self.lidar_sub = self.create_subscription(PointCloud2, self.lidar_topic, self.on_lidar, qos_sensor)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, qos_sensor)
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self.on_goal, qos_default)

        self.cmd_pub = self.create_publisher(HWSimpleKeyboardInfo, self.cmd_topic, 10)
        self.scan_pub = self.create_publisher(LaserScan, self.scan_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)

        self.enable_srv = self.create_service(SetBool, '/diff_vfh/enable', self.on_enable)

        self.odom: Optional[Odometry] = None
        self.last_odom_time = 0.0
        self.last_lidar_time = 0.0
        self.last_goal_time = 0.0
        self.goal: Optional[np.ndarray] = None
        if self.use_default_goal:
            self.goal = np.array([self.default_goal_x, self.default_goal_y, self.default_goal_z], dtype=np.float64)
            self.last_goal_time = time.time()

        self.ranges = np.full(self.n_sectors, self.range_max, dtype=np.float32)
        self.valid_scan = False
        self.last_cmd_xy = np.zeros(2, dtype=np.float64)
        self.last_heading_body = 0.0
        self.last_control_time = time.time()
        self.arrived_latched = False
        self.node_start_time = time.time()
        self.auto_takeoff_started = False
        self.path = Path()
        self.path.header.frame_id = self.odom_frame_id

        period = 1.0 / max(1.0, self.control_rate_hz)
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info(
            f'High-speed AirSim Diff-VFH controller started. lidar={self.lidar_topic}, '
            f'odom={self.odom_topic}, cmd={self.cmd_topic}, enable={self.enable_control}'
        )

    def _load_params(self) -> None:
        gp = lambda name: self.get_parameter(name).value
        self.lidar_topic = str(gp('lidar_topic'))
        self.odom_topic = str(gp('odom_topic'))
        self.cmd_topic = str(gp('cmd_topic'))
        self.goal_topic = str(gp('goal_topic'))
        self.scan_topic = str(gp('scan_topic'))
        self.status_topic = str(gp('status_topic'))
        self.marker_topic = str(gp('marker_topic'))
        self.path_topic = str(gp('path_topic'))
        self.base_frame_id = str(gp('base_frame_id'))
        self.odom_frame_id = str(gp('odom_frame_id'))
        self.enable_control = bool(gp('enable_control'))
        self.publish_zero_when_disabled = bool(gp('publish_zero_when_disabled'))
        self.output_frame = str(gp('output_frame')).strip().lower()
        self.control_rate_hz = float(gp('control_rate_hz'))
        self.sensor_timeout = float(gp('sensor_timeout'))
        self.goal_timeout = float(gp('goal_timeout'))

        self.scan_angle_min = math.radians(float(gp('scan_angle_min_deg')))
        self.scan_angle_max = math.radians(float(gp('scan_angle_max_deg')))
        self.n_sectors = int(gp('n_sectors'))
        self.range_min = float(gp('range_min'))
        self.range_max = float(gp('range_max'))
        self.height_min = float(gp('height_min'))
        self.height_max = float(gp('height_max'))
        self.use_height_filter = bool(gp('use_height_filter'))
        self.downsample_step = max(1, int(gp('downsample_step')))

        self.vehicle_radius = float(gp('vehicle_radius'))
        self.safety_margin = float(gp('safety_margin'))
        self.direction_window_deg = float(gp('direction_window_deg'))
        self.goal_weight = float(gp('goal_weight'))
        self.clearance_weight = float(gp('clearance_weight'))
        self.inertia_weight = float(gp('inertia_weight'))
        self.front_bias_weight = float(gp('front_bias_weight'))
        self.blocked_clearance = float(gp('blocked_clearance'))
        self.emergency_distance = float(gp('emergency_distance'))
        self.hard_stop_distance = float(gp('hard_stop_distance'))
        self.slow_distance = float(gp('slow_distance'))
        self.fast_clearance = float(gp('fast_clearance'))

        self.cruise_speed = float(gp('cruise_speed'))
        self.min_high_speed = float(gp('min_high_speed'))
        self.max_command_xy = float(gp('max_command_xy'))
        self.min_command_xy = float(gp('min_command_xy'))
        self.goal_slow_radius = float(gp('goal_slow_radius'))
        self.goal_tolerance = float(gp('goal_tolerance'))
        self.z_goal_tolerance = float(gp('z_goal_tolerance'))
        self.disable_on_arrival = bool(gp('disable_on_arrival'))

        self.max_xy_accel = float(gp('max_xy_accel'))
        self.max_xy_decel = float(gp('max_xy_decel'))
        self.command_filter_tau = float(gp('command_filter_tau'))
        self.yaw_rate_cmd = float(gp('yaw_rate_cmd'))
        self.auto_yaw_enable = bool(gp('auto_yaw_enable'))
        self.yaw_kp = float(gp('yaw_kp'))
        self.max_yaw_rate_cmd = float(gp('max_yaw_rate_cmd'))
        self.yaw_deadband_deg = float(gp('yaw_deadband_deg'))
        self.yaw_slowdown_start_deg = float(gp('yaw_slowdown_start_deg'))
        self.yaw_slowdown_stop_deg = float(gp('yaw_slowdown_stop_deg'))
        self.min_yaw_alignment_speed_scale = float(gp('min_yaw_alignment_speed_scale'))

        self.target_altitude_ned = float(gp('target_altitude_ned'))
        self.use_goal_z_as_target = bool(gp('use_goal_z_as_target'))
        self.z_hold_kp = float(gp('z_hold_kp'))
        self.z_hold_kd = float(gp('z_hold_kd'))
        self.z_deadband = float(gp('z_deadband'))
        self.max_z_command = float(gp('max_z_command'))
        self.z_command_sign = float(gp('z_command_sign'))

        self.default_goal_x = float(gp('default_goal_x'))
        self.default_goal_y = float(gp('default_goal_y'))
        self.default_goal_z = float(gp('default_goal_z'))
        self.use_default_goal = bool(gp('use_default_goal'))
        self.auto_takeoff_on_start = bool(gp('auto_takeoff_on_start'))
        self.auto_takeoff_height_m = float(gp('auto_takeoff_height_m'))
        self.auto_takeoff_delay_sec = float(gp('auto_takeoff_delay_sec'))
        self.auto_takeoff_wait_for_lidar = bool(gp('auto_takeoff_wait_for_lidar'))

    def try_start_auto_takeoff(self, now: float) -> None:
        """Set the initial takeoff goal automatically.

        This is deliberately implemented inside the controller instead of the
        Tk panel: after launch, the node waits for odometry, sets the goal to
        (current_x, current_y, -auto_takeoff_height_m), and enables control.
        The existing move_velocity node will then receive z velocity commands
        and perform the same PX4 offboard/arming logic as before.
        """
        if not self.auto_takeoff_on_start or self.auto_takeoff_started:
            return
        if self.odom is None:
            return
        if now - self.node_start_time < max(0.0, self.auto_takeoff_delay_sec):
            return
        if self.auto_takeoff_wait_for_lidar and (now - self.last_lidar_time) > self.sensor_timeout:
            return

        p = self.odom.pose.pose.position
        target_z = -abs(float(self.auto_takeoff_height_m))
        self.goal = np.array([float(p.x), float(p.y), target_z], dtype=np.float64)
        self.last_goal_time = now
        self.arrived_latched = False
        self.enable_control = True
        self.auto_takeoff_started = True
        self.get_logger().info(
            f'AUTO TAKEOFF: goal_ned=({self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.goal[2]:.2f}), control enabled'
        )
        self.publish_status(
            f'AUTO_TAKEOFF_GOAL x={self.goal[0]:.2f} y={self.goal[1]:.2f} z={self.goal[2]:.2f}'
        )

    def on_enable(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.enable_control = bool(request.data)
        self.arrived_latched = False
        self.node_start_time = time.time()
        self.auto_takeoff_started = False
        self.last_cmd_xy[:] = 0.0
        if not self.enable_control:
            self.publish_zero()
        response.success = True
        response.message = 'Diff-VFH control enabled' if self.enable_control else 'Diff-VFH control disabled'
        self.get_logger().info(response.message)
        return response

    def on_goal(self, msg: PoseStamped) -> None:
        self.goal = np.array([
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ], dtype=np.float64)
        self.last_goal_time = time.time()
        self.arrived_latched = False
        self.node_start_time = time.time()
        self.auto_takeoff_started = False
        self.get_logger().info(f'New goal_ned: x={self.goal[0]:.2f}, y={self.goal[1]:.2f}, z={self.goal[2]:.2f}')

    def on_odom(self, msg: Odometry) -> None:
        self.odom = msg
        self.last_odom_time = time.time()
        # Publish path at a reduced rate without a separate timer.
        if len(self.path.poses) == 0 or len(self.path.poses) % 5 == 0:
            self.path.header.stamp = self.get_clock().now().to_msg()
            pose = PoseStamped()
            pose.header = self.path.header
            pose.pose = msg.pose.pose
            self.path.poses.append(pose)
            if len(self.path.poses) > 1200:
                self.path.poses = self.path.poses[-1200:]
            self.path_pub.publish(self.path)

    def on_lidar(self, msg: PointCloud2) -> None:
        if pc2 is None:
            self.publish_status('WAIT_LIDAR: sensor_msgs_py.point_cloud2 is not available')
            return

        ranges = np.full(self.n_sectors, self.range_max, dtype=np.float32)
        angle_span = self.scan_angle_max - self.scan_angle_min
        if angle_span <= 1e-6:
            angle_span = 2.0 * math.pi

        count = 0
        accepted = 0
        fields = ('x', 'y', 'z')
        for p in pc2.read_points(msg, field_names=fields, skip_nans=True):
            count += 1
            if count % self.downsample_step != 0:
                continue
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if self.use_height_filter and (z < self.height_min or z > self.height_max):
                continue
            r = math.hypot(x, y)
            if r < self.range_min or r > self.range_max:
                continue
            a = wrap_pi(math.atan2(y, x))
            if a < self.scan_angle_min or a > self.scan_angle_max:
                continue
            idx = int((a - self.scan_angle_min) / angle_span * self.n_sectors)
            idx = max(0, min(self.n_sectors - 1, idx))
            if r < ranges[idx]:
                ranges[idx] = r
            accepted += 1

        # Fill isolated holes lightly to reduce steering jitter.
        if accepted > 0 and self.n_sectors >= 5:
            rr = ranges.copy()
            for i in range(self.n_sectors):
                window = ranges[max(0, i - 1):min(self.n_sectors, i + 2)]
                rr[i] = float(np.min(window))
            ranges = rr

        self.ranges = ranges
        self.valid_scan = accepted > 0
        self.last_lidar_time = time.time()
        self.publish_scan(msg.header.stamp)

    def publish_scan(self, stamp) -> None:
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = self.base_frame_id
        scan.angle_min = float(self.scan_angle_min)
        scan.angle_max = float(self.scan_angle_max)
        scan.angle_increment = float((self.scan_angle_max - self.scan_angle_min) / max(1, self.n_sectors - 1))
        scan.time_increment = 0.0
        scan.scan_time = 1.0 / max(1.0, self.control_rate_hz)
        scan.range_min = float(self.range_min)
        scan.range_max = float(self.range_max)
        scan.ranges = [float(x) for x in self.ranges]
        self.scan_pub.publish(scan)

    def current_pose_yaw(self) -> Optional[Tuple[np.ndarray, float, float]]:
        if self.odom is None:
            return None
        p = self.odom.pose.pose.position
        pos = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
        yaw = quat_to_yaw(self.odom.pose.pose.orientation)
        vz = float(self.odom.twist.twist.linear.z)
        return pos, yaw, vz

    def sector_angles(self) -> np.ndarray:
        return np.linspace(self.scan_angle_min, self.scan_angle_max, self.n_sectors, dtype=np.float64)

    def window_min_range(self, angle: float) -> float:
        angles = self.sector_angles()
        half = math.radians(self.direction_window_deg)
        diffs = np.abs(np.array([wrap_pi(float(a - angle)) for a in angles]))
        mask = diffs <= half
        if not np.any(mask):
            idx = int(np.argmin(diffs))
            return float(self.ranges[idx])
        return float(np.min(self.ranges[mask]))

    def compute_vfh_heading(self, goal_angle_body: float) -> Tuple[float, float, float]:
        angles = self.sector_angles()
        best_score = -1e9
        best_angle = goal_angle_body
        best_clearance_range = self.range_max

        safety_radius = self.vehicle_radius + self.safety_margin
        for a in angles:
            a = float(a)
            rmin = self.window_min_range(a)
            clearance = rmin - safety_radius
            if clearance < self.blocked_clearance:
                # Still keep as a last resort, but heavily penalize.
                block_penalty = 4.0 + 4.0 * (self.blocked_clearance - clearance)
            else:
                block_penalty = 0.0

            goal_err = abs(wrap_pi(a - goal_angle_body))
            inertia_err = abs(wrap_pi(a - self.last_heading_body))
            front_err = abs(wrap_pi(a))

            goal_score = 1.0 - goal_err / math.pi
            inertia_score = 1.0 - inertia_err / math.pi
            front_score = 1.0 - front_err / math.pi
            clear_score = clamp((rmin - self.hard_stop_distance) / max(1e-6, self.fast_clearance - self.hard_stop_distance), 0.0, 1.0)
            danger_penalty = 0.0
            if rmin < self.slow_distance:
                danger_penalty = (self.slow_distance - rmin) / max(1e-6, self.slow_distance - self.hard_stop_distance)

            score = (
                self.goal_weight * goal_score
                + self.clearance_weight * clear_score
                + self.inertia_weight * inertia_score
                + self.front_bias_weight * front_score
                - 1.65 * danger_penalty
                - block_penalty
            )
            if score > best_score:
                best_score = score
                best_angle = a
                best_clearance_range = rmin

        return best_angle, best_clearance_range, best_score

    def compute_speed(self, dist_to_goal: float, chosen_clearance_range: float) -> float:
        if chosen_clearance_range <= self.hard_stop_distance:
            speed = 0.0
        elif chosen_clearance_range < self.slow_distance:
            alpha = (chosen_clearance_range - self.hard_stop_distance) / max(1e-6, self.slow_distance - self.hard_stop_distance)
            speed = self.min_command_xy + alpha * (self.min_high_speed - self.min_command_xy)
        elif chosen_clearance_range < self.fast_clearance:
            alpha = (chosen_clearance_range - self.slow_distance) / max(1e-6, self.fast_clearance - self.slow_distance)
            speed = self.min_high_speed + alpha * (self.cruise_speed - self.min_high_speed)
        else:
            speed = max(self.cruise_speed, self.min_high_speed)

        speed = min(speed, self.max_command_xy)

        if dist_to_goal < self.goal_slow_radius:
            speed *= clamp(dist_to_goal / max(1e-6, self.goal_slow_radius), 0.0, 1.0)

        if dist_to_goal <= self.goal_tolerance:
            speed = 0.0

        return float(clamp(speed, 0.0, self.max_command_xy))

    def smooth_xy(self, desired_xy: np.ndarray, dt: float) -> np.ndarray:
        desired_xy = np.asarray(desired_xy, dtype=np.float64)
        prev = self.last_cmd_xy.copy()
        delta = desired_xy - prev
        delta_norm = float(np.linalg.norm(delta))
        prev_speed = float(np.linalg.norm(prev))
        des_speed = float(np.linalg.norm(desired_xy))
        accel_limit = self.max_xy_decel if des_speed < prev_speed else self.max_xy_accel
        max_delta = max(0.0, accel_limit * dt)
        if delta_norm > max_delta > 1e-9:
            delta *= max_delta / delta_norm
        limited = prev + delta

        tau = max(1e-6, self.command_filter_tau)
        alpha = clamp(dt / (tau + dt), 0.0, 1.0)
        out = (1.0 - alpha) * prev + alpha * limited

        speed = float(np.linalg.norm(out))
        if speed > self.max_command_xy:
            out *= self.max_command_xy / speed
        self.last_cmd_xy = out
        return out

    def compute_z_hold(self, pos: np.ndarray, vz: float) -> float:
        target_z = self.target_altitude_ned
        if self.use_goal_z_as_target and self.goal is not None and math.isfinite(float(self.goal[2])):
            target_z = float(self.goal[2])
        err = target_z - float(pos[2])
        if abs(err) < self.z_deadband:
            err = 0.0
        cmd = self.z_hold_kp * err - self.z_hold_kd * float(vz)
        return clamp(cmd, -self.max_z_command, self.max_z_command)

    def compute_yaw_rate_and_speed_scale(self, heading_body: float) -> Tuple[float, float]:
        """Command yaw toward the selected safe heading and slow down while misaligned.

        heading_body is the chosen VFH direction relative to the UAV body.
        If it is large, the drone should first rotate instead of translating
        sideways at high speed.  The HWSimpleKeyboardInfo.yaw field in the
        existing keyboard chain is normally used in about [-1, 1], so the
        default max_yaw_rate_cmd is intentionally below 1.0.
        """
        if not self.auto_yaw_enable:
            return float(self.yaw_rate_cmd), 1.0

        err = wrap_pi(float(heading_body))
        if abs(err) < math.radians(self.yaw_deadband_deg):
            yaw_rate = 0.0
        else:
            yaw_rate = clamp(self.yaw_kp * err, -self.max_yaw_rate_cmd, self.max_yaw_rate_cmd)

        abs_deg = abs(math.degrees(err))
        start = max(0.0, float(self.yaw_slowdown_start_deg))
        stop = max(start + 1e-6, float(self.yaw_slowdown_stop_deg))
        min_scale = clamp(float(self.min_yaw_alignment_speed_scale), 0.0, 1.0)

        if abs_deg <= start:
            speed_scale = 1.0
        elif abs_deg >= stop:
            speed_scale = min_scale
        else:
            alpha = (abs_deg - start) / (stop - start)
            speed_scale = 1.0 - alpha * (1.0 - min_scale)

        return float(yaw_rate), float(speed_scale)

    def publish_cmd(self, vx: float, vy: float, vz: float, yaw: float = 0.0) -> None:
        msg = HWSimpleKeyboardInfo()
        msg.x = float(vx)
        msg.y = float(vy)
        msg.z = float(self.z_command_sign * float(vz))
        msg.yaw = float(yaw)
        self.cmd_pub.publish(msg)

    def publish_zero(self) -> None:
        self.last_cmd_xy[:] = 0.0
        self.publish_cmd(0.0, 0.0, 0.0, 0.0)

    def publish_status(self, text: str) -> None:
        msg = String()
        msg.data = str(text)
        self.status_pub.publish(msg)

    def publish_markers(self, pos: np.ndarray, yaw: float, heading_body: float, speed: float, goal: np.ndarray) -> None:
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()

        # Delete previous marker set first to keep RViz clean.
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        goal_m = Marker()
        goal_m.header.frame_id = self.odom_frame_id
        goal_m.header.stamp = now
        goal_m.ns = 'diff_vfh_goal'
        goal_m.id = 1
        goal_m.type = Marker.SPHERE
        goal_m.action = Marker.ADD
        goal_m.pose.position.x = float(goal[0])
        goal_m.pose.position.y = float(goal[1])
        goal_m.pose.position.z = float(goal[2])
        goal_m.pose.orientation.w = 1.0
        goal_m.scale.x = goal_m.scale.y = goal_m.scale.z = 0.45
        goal_m.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.9)
        markers.markers.append(goal_m)

        arrow = Marker()
        arrow.header.frame_id = self.odom_frame_id
        arrow.header.stamp = now
        arrow.ns = 'diff_vfh_cmd'
        arrow.id = 2
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.scale.x = 0.12
        arrow.scale.y = 0.24
        arrow.scale.z = 0.24
        arrow.color = ColorRGBA(r=0.1, g=0.4, b=1.0, a=0.95)
        start = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        ned_angle = yaw + heading_body
        length = max(0.6, min(3.0, speed))
        end = Point(
            x=float(pos[0] + length * math.cos(ned_angle)),
            y=float(pos[1] + length * math.sin(ned_angle)),
            z=float(pos[2]),
        )
        arrow.points = [start, end]
        markers.markers.append(arrow)

        self.marker_pub.publish(markers)

    def control_loop(self) -> None:
        now = time.time()
        self.try_start_auto_takeoff(now)
        dt = clamp(now - self.last_control_time, 0.005, 0.2)
        self.last_control_time = now

        if not self.enable_control:
            if self.publish_zero_when_disabled:
                self.publish_zero()
            self.publish_status('DISABLED')
            return

        pose_yaw = self.current_pose_yaw()
        if pose_yaw is None or (now - self.last_odom_time) > self.sensor_timeout:
            self.publish_zero()
            self.publish_status('WAIT_ODOM')
            return
        pos, yaw, vz = pose_yaw

        if self.goal is None:
            self.publish_cmd(0.0, 0.0, self.compute_z_hold(pos, vz), 0.0)
            self.publish_status('WAIT_GOAL')
            return
        if self.goal_timeout > 0.0 and (now - self.last_goal_time) > self.goal_timeout:
            self.publish_cmd(0.0, 0.0, self.compute_z_hold(pos, vz), 0.0)
            self.publish_status('WAIT_FRESH_GOAL')
            return
        if (now - self.last_lidar_time) > self.sensor_timeout or not self.valid_scan:
            self.publish_cmd(0.0, 0.0, self.compute_z_hold(pos, vz), 0.0)
            self.publish_status('WAIT_LIDAR')
            return

        goal_vec = self.goal[:2] - pos[:2]
        dist = float(np.linalg.norm(goal_vec))
        goal_z = float(self.goal[2]) if self.goal is not None and math.isfinite(float(self.goal[2])) else self.target_altitude_ned
        z_err_to_goal = abs(goal_z - float(pos[2]))
        if dist <= self.goal_tolerance and z_err_to_goal <= self.z_goal_tolerance:
            zcmd = self.compute_z_hold(pos, vz)
            self.publish_cmd(0.0, 0.0, zcmd, 0.0)
            self.arrived_latched = True
            self.last_cmd_xy[:] = 0.0
            self.publish_status(f'ARRIVED dist={dist:.2f} zerr={z_err_to_goal:.2f} z={pos[2]:.2f}->{goal_z:.2f}')
            if self.disable_on_arrival:
                self.enable_control = False
            return

        goal_angle_ned = math.atan2(float(goal_vec[1]), float(goal_vec[0]))
        goal_angle_body = wrap_pi(goal_angle_ned - yaw)

        heading_body, clear_range, score = self.compute_vfh_heading(goal_angle_body)
        speed = self.compute_speed(dist, clear_range)

        # Important deployment fix:
        # rotate toward the selected safe corridor and reduce forward speed when
        # the body heading is far from that corridor.  This prevents the UAV from
        # looking/turning too slowly while still translating fast into obstacles.
        yaw_rate_cmd, yaw_speed_scale = self.compute_yaw_rate_and_speed_scale(heading_body)
        speed *= yaw_speed_scale

        # Emergency: if obstacle is too close in the selected corridor, stop planar motion.
        if clear_range < self.emergency_distance:
            speed = 0.0

        vx_body = speed * math.cos(heading_body)
        vy_body = speed * math.sin(heading_body)

        if self.output_frame == 'body':
            desired_xy = np.array([vx_body, vy_body], dtype=np.float64)
        else:
            vx_ned = vx_body * math.cos(yaw) - vy_body * math.sin(yaw)
            vy_ned = vx_body * math.sin(yaw) + vy_body * math.cos(yaw)
            desired_xy = np.array([vx_ned, vy_ned], dtype=np.float64)

        cmd_xy = self.smooth_xy(desired_xy, dt)
        zcmd = self.compute_z_hold(pos, vz)
        zcmd_msg = self.z_command_sign * zcmd
        self.publish_cmd(float(cmd_xy[0]), float(cmd_xy[1]), float(zcmd), yaw_rate_cmd)
        self.last_heading_body = float(heading_body)

        min_r = float(np.min(self.ranges)) if self.ranges.size else self.range_max
        self.publish_status(
            'AUTO_DIFF_VFH '
            f'dist={dist:.2f} min_r={min_r:.2f} dir_r={clear_range:.2f} '
            f'head_body_deg={math.degrees(heading_body):.1f} speed={speed:.2f} '
            f'cmd=({cmd_xy[0]:.2f},{cmd_xy[1]:.2f},{zcmd_msg:.2f},yaw={yaw_rate_cmd:.2f}) yaw_scale={yaw_speed_scale:.2f} '
            f'z={pos[2]:.2f}->{self.goal[2]:.2f} score={score:.2f}'
        )
        self.publish_markers(pos, yaw, heading_body, speed, self.goal)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = AirsimDiffVFHController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_zero()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
