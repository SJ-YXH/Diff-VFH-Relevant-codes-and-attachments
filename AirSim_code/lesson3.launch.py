#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import launch
import launch_ros.actions
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('hw_insight')
    diff_vfh_cfg = os.path.join(pkg_share, 'config', 'diff_vfh_airsim_auto_highspeed.yaml')

    depth_camera_relay_node = Node(
        package='topic_tools',
        executable='relay',
        name='depth_camera_relay',
        arguments=[
            '/airsim_node/PX4/CameraDepth/DepthPerspective/camera_info',
            '/airsim_node/PX4/CameraDepth/camera_info'
        ],
        output='screen'
    )

    image_camera_relay_node = Node(
        package='topic_tools',
        executable='relay',
        name='image_camera_relay',
        arguments=[
            '/airsim_node/PX4/CameraImage/Scene/camera_info',
            '/airsim_node/PX4/CameraImage/camera_info'
        ],
        output='screen'
    )

    hw_move_velocity_node = Node(
        package='hw_insight',
        executable='move_velocity',
        name='move_velocity',
        output='screen',
        parameters=[diff_vfh_cfg]
    )

    depth_rviz_path = os.path.join(pkg_share, 'rviz/depth_cloud.rviz')
    image_lidar_rviz_path = os.path.join(pkg_share, 'rviz/image_lidar.rviz')

    hw_rviz_depth_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='depth_rviz2',
        arguments=['-d', depth_rviz_path]
    )

    hw_rviz_image_lidar_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='image_lidar_rviz2',
        arguments=['-d', image_lidar_rviz_path]
    )

    airsim_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('airsim_ros_pkgs'), 'launch/airsim_node.launch.py')
        )
    )

    ld = LaunchDescription()
    ld.add_action(airsim_node_launch)
    ld.add_action(hw_move_velocity_node)
    ld.add_action(depth_camera_relay_node)
    ld.add_action(image_camera_relay_node)
    ld.add_action(hw_rviz_depth_node)
    ld.add_action(hw_rviz_image_lidar_node)
    return ld
