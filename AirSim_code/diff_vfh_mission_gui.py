#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission GUI for AirSim + PX4 + ROS2 Diff-VFH deployment.

Important behavior:
- Takeoff is automatic inside airsim_diff_vfh_controller.
- The GUI no longer controls takeoff.
- After clicking "Send Goal and Enable", the GUI continuously republishes
  /diff_vfh/goal_ned at mission_goal_republish_hz.  This prevents the controller
  from falling back to the auto-takeoff/hold goal if a single goal message is
  missed or overwritten during timing transitions.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import SetBool

from hw_interface.msg import HWSimpleKeyboardInfo

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as exc:  # pragma: no cover
    tk = None
    ttk = None
    messagebox = None
    _TK_IMPORT_ERROR = exc
else:
    _TK_IMPORT_ERROR = None


class MissionGuiNode(Node):
    def __init__(self) -> None:
        super().__init__('diff_vfh_mission_gui')

        self.declare_parameter('goal_topic', '/diff_vfh/goal_ned')
        self.declare_parameter('enable_service', '/diff_vfh/enable')
        self.declare_parameter('odom_topic', '/airsim_node/PX4/odom_local_ned')
        self.declare_parameter('status_topic', '/diff_vfh/status')
        self.declare_parameter('cmd_topic', '/hw_insight/keyboard_velocity')

        self.declare_parameter('default_flight_height_m', 2.0)
        self.declare_parameter('default_goal_x', 20.0)
        self.declare_parameter('default_goal_y', 0.0)
        self.declare_parameter('mission_goal_republish_hz', 5.0)

        gp = lambda name: self.get_parameter(name).value
        self.goal_topic = str(gp('goal_topic'))
        self.enable_service = str(gp('enable_service'))
        self.odom_topic = str(gp('odom_topic'))
        self.status_topic = str(gp('status_topic'))
        self.cmd_topic = str(gp('cmd_topic'))
        self.default_flight_height_m = float(gp('default_flight_height_m'))
        self.default_goal_x = float(gp('default_goal_x'))
        self.default_goal_y = float(gp('default_goal_y'))
        self.mission_goal_republish_hz = max(0.2, float(gp('mission_goal_republish_hz')))

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic, 10)
        self.zero_cmd_pub = self.create_publisher(HWSimpleKeyboardInfo, self.cmd_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, qos_sensor)
        self.status_sub = self.create_subscription(String, self.status_topic, self.on_status, 10)
        self.cmd_sub = self.create_subscription(HWSimpleKeyboardInfo, self.cmd_topic, self.on_cmd, 10)

        self.enable_client = self.create_client(SetBool, self.enable_service)

        self.odom: Optional[Odometry] = None
        self.last_odom_time = 0.0
        self.status_text = 'WAIT'
        self.last_cmd = HWSimpleKeyboardInfo()

        self.active_goal_msg: Optional[PoseStamped] = None
        self.last_goal_label = 'AUTO TAKEOFF'
        self.last_goal_pub_time = 0.0
        self.goal_pub_count = 0

        # Persistent mission goal publisher.
        period = 1.0 / self.mission_goal_republish_hz
        self.goal_timer = self.create_timer(period, self.republish_active_goal)

    def on_odom(self, msg: Odometry) -> None:
        self.odom = msg
        self.last_odom_time = time.time()

    def on_status(self, msg: String) -> None:
        self.status_text = str(msg.data)

    def on_cmd(self, msg: HWSimpleKeyboardInfo) -> None:
        self.last_cmd = msg

    def current_xyz(self) -> Tuple[float, float, float]:
        if self.odom is None:
            return 0.0, 0.0, 0.0
        p = self.odom.pose.pose.position
        return float(p.x), float(p.y), float(p.z)

    def set_enable(self, enabled: bool) -> None:
        req = SetBool.Request()
        req.data = bool(enabled)

        if not self.enable_client.service_is_ready():
            self.enable_client.wait_for_service(timeout_sec=1.0)

        if self.enable_client.service_is_ready():
            future = self.enable_client.call_async(req)
            future.add_done_callback(lambda fut: self._enable_done(enabled, fut))
        else:
            self.get_logger().warn(f'Enable service is not ready: {self.enable_service}')

    def _enable_done(self, enabled: bool, fut) -> None:
        try:
            resp = fut.result()
            if resp is not None:
                self.get_logger().info(f'Controller enable={enabled}: {resp.message}')
        except Exception as exc:
            self.get_logger().warn(f'Enable service call failed: {exc}')

    def build_goal_msg(self, x: float, y: float, z_ned: float) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = 'odom_local_ned'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z_ned)
        msg.pose.orientation.w = 1.0
        return msg

    def publish_goal_once(self, msg: PoseStamped) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        self.goal_pub.publish(msg)
        self.last_goal_pub_time = time.time()
        self.goal_pub_count += 1

    def set_active_goal(self, x: float, y: float, z_ned: float, label: str) -> None:
        msg = self.build_goal_msg(x, y, z_ned)
        self.active_goal_msg = msg
        self.last_goal_label = f'{label}: ({x:.1f}, {y:.1f}, {z_ned:.1f})'
        self.goal_pub_count = 0

        # Burst-publish immediately so the controller receives it even if the
        # first message hits a startup/scheduling gap.
        for _ in range(5):
            self.publish_goal_once(msg)

        self.get_logger().info(
            f'Active goal set and latched for periodic publishing: {self.last_goal_label}, '
            f'rate={self.mission_goal_republish_hz:.1f} Hz'
        )

    def republish_active_goal(self) -> None:
        if self.active_goal_msg is None:
            return
        self.publish_goal_once(self.active_goal_msg)

    def stop_now(self) -> None:
        self.active_goal_msg = None
        self.set_enable(False)
        msg = HWSimpleKeyboardInfo()
        msg.x = 0.0
        msg.y = 0.0
        msg.z = 0.0
        msg.yaw = 0.0
        self.zero_cmd_pub.publish(msg)
        self.get_logger().info('STOP: controller disabled, active goal cleared, zero velocity published')


class MissionTkApp:
    def __init__(self, node: MissionGuiNode) -> None:
        if tk is None:
            raise RuntimeError(f'tkinter import failed: {_TK_IMPORT_ERROR}')

        self.node = node
        self.root = tk.Tk()
        self.root.title('Diff-VFH Mission')
        self.root.geometry('390x180')
        self.root.resizable(False, False)

        self.goal_x = tk.StringVar(value=f'{node.default_goal_x:.1f}')
        self.goal_y = tk.StringVar(value=f'{node.default_goal_y:.1f}')
        self.state_var = tk.StringVar(value='STATE: WAIT')
        self.takeoff_var = tk.StringVar(value=f'AUTO TAKEOFF: target height = {node.default_flight_height_m:.1f} m')
        self.pose_var = tk.StringVar(value='POSE: x=0.0, y=0.0, z=0.0')
        self.last_goal_var = tk.StringVar(value='LAST GOAL: AUTO TAKEOFF')
        self.pub_var = tk.StringVar(value='GOAL PUB: idle')

        self._build()
        self.root.after(150, self._tick)

    def _build(self) -> None:
        pad = {'padx': 6, 'pady': 3}

        ttk.Label(self.root, textvariable=self.takeoff_var).grid(row=0, column=0, columnspan=5, sticky='w', **pad)
        ttk.Label(self.root, textvariable=self.state_var).grid(row=1, column=0, columnspan=5, sticky='w', **pad)
        ttk.Label(self.root, textvariable=self.pose_var).grid(row=2, column=0, columnspan=5, sticky='w', **pad)
        ttk.Label(self.root, textvariable=self.last_goal_var).grid(row=3, column=0, columnspan=5, sticky='w', **pad)
        ttk.Label(self.root, textvariable=self.pub_var).grid(row=4, column=0, columnspan=5, sticky='w', **pad)

        ttk.Label(self.root, text='Goal X').grid(row=5, column=0, sticky='e', **pad)
        ttk.Entry(self.root, width=8, textvariable=self.goal_x).grid(row=5, column=1, sticky='w', **pad)
        ttk.Label(self.root, text='Goal Y').grid(row=5, column=2, sticky='e', **pad)
        ttk.Entry(self.root, width=8, textvariable=self.goal_y).grid(row=5, column=3, sticky='w', **pad)

        ttk.Button(self.root, text='Send Goal and Enable', command=self.send_goal).grid(row=6, column=0, columnspan=2, sticky='ew', **pad)
        ttk.Button(self.root, text='Hold Current Position', command=self.hold_current).grid(row=6, column=2, columnspan=2, sticky='ew', **pad)
        ttk.Button(self.root, text='STOP', command=self.stop).grid(row=6, column=4, sticky='ew', **pad)

        ttk.Label(
            self.root,
            text='After Send, /diff_vfh/goal_ned is republished continuously.',
            foreground='gray'
        ).grid(row=7, column=0, columnspan=5, sticky='w', **pad)

    def _flight_z_ned(self) -> float:
        return -abs(float(self.node.default_flight_height_m))

    def _safe_call(self, fn) -> None:
        try:
            fn()
        except Exception as exc:
            if messagebox is not None:
                messagebox.showerror('Diff-VFH Mission', str(exc))
            else:
                print(exc)

    def send_goal(self) -> None:
        self._safe_call(self._send_goal_impl)

    def _send_goal_impl(self) -> None:
        gx = float(self.goal_x.get())
        gy = float(self.goal_y.get())
        self.node.set_active_goal(gx, gy, self._flight_z_ned(), 'MISSION')
        self.node.set_enable(True)

    def hold_current(self) -> None:
        self._safe_call(self._hold_current_impl)

    def _hold_current_impl(self) -> None:
        x, y, _z = self.node.current_xyz()
        self.node.set_active_goal(x, y, self._flight_z_ned(), 'HOLD')
        self.node.set_enable(True)

    def stop(self) -> None:
        self.node.stop_now()

    def _tick(self) -> None:
        x, y, z = self.node.current_xyz()
        cmd = self.node.last_cmd
        self.takeoff_var.set(
            f'AUTO TAKEOFF: target z={self._flight_z_ned():.2f}  height={self.node.default_flight_height_m:.1f} m'
        )
        self.state_var.set(f'STATE: {self.node.status_text[:76]}')
        self.pose_var.set(
            f'POSE: x={x:.1f}, y={y:.1f}, z={z:.2f} | cmd=({float(cmd.x):.1f},{float(cmd.y):.1f},{float(cmd.z):.2f},yaw={float(cmd.yaw):.2f})'
        )
        self.last_goal_var.set(f'LAST GOAL: {self.node.last_goal_label}')
        if self.node.active_goal_msg is None:
            self.pub_var.set('GOAL PUB: idle')
        else:
            age = time.time() - self.node.last_goal_pub_time if self.node.last_goal_pub_time > 0 else 999.0
            self.pub_var.set(
                f'GOAL PUB: active {self.node.mission_goal_republish_hz:.1f} Hz, count={self.node.goal_pub_count}, age={age:.2f}s'
            )
        self.root.after(150, self._tick)

    def run(self) -> None:
        self.root.mainloop()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionGuiNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        app = MissionTkApp(node)
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_now()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
