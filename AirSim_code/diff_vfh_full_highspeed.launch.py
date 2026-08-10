#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('hw_insight')
    cfg = os.path.join(pkg_share, 'config', 'diff_vfh_airsim_auto_highspeed.yaml')

    start_agent = LaunchConfiguration('start_agent')
    start_px4 = LaunchConfiguration('start_px4')
    start_base = LaunchConfiguration('start_base')
    start_controller = LaunchConfiguration('start_controller')
    start_gui = LaunchConfiguration('start_gui')
    enable_control = LaunchConfiguration('enable_control')
    px4_dir = LaunchConfiguration('px4_dir')

    args = [
        # Default is false because most users already start these manually.
        # Starting a second MicroXRCEAgent causes: bind error port 8888 errno 98.
        # Starting a second PX4 causes: PX4 server already running for instance 0.
        DeclareLaunchArgument('start_agent', default_value='false'),
        DeclareLaunchArgument('start_px4', default_value='false'),

        DeclareLaunchArgument('start_base', default_value='true'),
        DeclareLaunchArgument('start_controller', default_value='true'),
        DeclareLaunchArgument('start_gui', default_value='true'),
        DeclareLaunchArgument('enable_control', default_value='false'),
        DeclareLaunchArgument('px4_dir', default_value=os.path.expanduser('~/px4v1.15.2')),
    ]

    agent = ExecuteProcess(
        cmd=['bash', '-lc', 'MicroXRCEAgent udp4 -p 8888'],
        output='screen',
        condition=IfCondition(start_agent),
    )

    px4 = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=['bash', '-lc', ['cd ', px4_dir, ' && make px4_sitl_default none_iris']],
                output='screen',
                condition=IfCondition(start_px4),
            )
        ],
    )

    base = TimerAction(
        period=1.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'lesson3.launch.py')),
                condition=IfCondition(start_base),
            )
        ],
    )

    controller = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='hw_insight',
                executable='airsim_diff_vfh_controller',
                name='airsim_diff_vfh_controller',
                output='screen',
                parameters=[cfg, {'enable_control': enable_control}],
                condition=IfCondition(start_controller),
            )
        ],
    )

    gui = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='hw_insight',
                executable='diff_vfh_mission_gui',
                name='diff_vfh_mission_gui',
                output='screen',
                parameters=[cfg],
                condition=IfCondition(start_gui),
            )
        ],
    )

    return LaunchDescription(args + [agent, px4, base, controller, gui])
