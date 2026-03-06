from setuptools import setup, find_packages

package_name = 'road_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    include_package_data=True,   # 🔴 REQUIRED
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sah3637s',
    maintainer_email='sah3637s@hs-coburg.de',
    description='ONNX-based road segmentation using DeepLabV3+',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'onnx_road_seg = road_segmentation.onnx_segment_node:main',
        ],
    },
)

