import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hw_insight'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hw',
    maintainer_email='toplaya@126.com',
    description='AirSim PX4 ROS2 control and Diff-VFH mission package',
    license='TODO',
    entry_points={
        'console_scripts': [
            'offboard = hw_insight.offboard:main',
            'decode_px4_fmu_out_vehicle_status = hw_insight.msg_px4_fmu_out_vehicle_status:main',
            'keyboard_position = hw_insight.keyboard_position:main',
            'move_position = hw_insight.move_position:main',
            'keyboard_velocity = hw_insight.keyboard_velocity:main',
            'move_velocity = hw_insight.move_velocity:main',
            'airsim_diff_vfh_controller = hw_insight.airsim_diff_vfh_controller:main',
            'diff_vfh_mission_gui = hw_insight.diff_vfh_mission_gui:main',
        ],
    },
)
