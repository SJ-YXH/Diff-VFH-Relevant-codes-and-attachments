#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('hw_insight')

    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'lesson3.launch.py')
        )
    )

    cfg = os.path.join(pkg_share, 'config', 'diff_vfh_airsim_auto_highspeed.yaml')

    controller = Node(
        package='hw_insight',
        executable='airsim_diff_vfh_controller',
        name='airsim_diff_vfh_controller',
        output='screen',
        parameters=[cfg],
    )

    mission_gui = Node(
        package='hw_insight',
        executable='diff_vfh_mission_gui',
        name='diff_vfh_mission_gui',
        output='screen',
        parameters=[cfg],
    )

    return LaunchDescription([
        base_launch,
        controller,
        mission_gui,
    ])
