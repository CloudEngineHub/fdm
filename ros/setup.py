# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

# Minimum dependencies required prior to installation     "opencv-python",
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "tqdm",
    "torchvision",
    "torch",
    "torchmetrics",
    "pytictac",
    "omegaconf",
    "hydra-core",
    "rospkg",
    "pypose",
    "seaborn",
    "scipy",
]


if __name__ == "__main__":
    setup(
        name="fdm_navigation",
        author="Pascal Roth",
        version="1.0.0",
        description="A sampling-based planner that bases its sampling strategy on the FDM.",
        author_email="rothpa@ethz.ch",
        packages=find_packages(),
        include_package_data=True,
        package_data={"fdm_navigation": ["*/*.so"]},
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: BSD 3",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.8",
        ],
        license="BSD 3",
        install_requires=[INSTALL_REQUIRES],
        ext_modules=[],
        zip_safe=False,
    )
