from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'yolo_object_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Ultimate YOLO 3D Detection Node',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ultimate_yolo_node = yolo_object_detector.ultimate_yolo_node:main',
            'websocket_bridge_node = yolo_object_detector.websocket_bridge_node:main',
            'simple_interaction_node = yolo_object_detector.simple_interaction_node:main',
            'task_coordinator_node = yolo_object_detector.task_coordinator_node:main',
            'pointcloud_grasp_planner = yolo_object_detector.pointcloud_grasp_planner:main',
        ],
    },
)
