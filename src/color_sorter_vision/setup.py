from setuptools import find_packages, setup

package_name = 'color_sorter_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sim',
    maintainer_email='local@host',
    description='HSV color detection, numpy image conversion (no cv_bridge).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'color_detector = color_sorter_vision.color_detector:main',
            'pick_and_place = color_sorter_vision.pick_and_place:main',
            'camera_viewer = color_sorter_vision.camera_viewer:main',
        ],
    },
)
