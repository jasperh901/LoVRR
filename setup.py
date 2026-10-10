#!/usr/bin/env python3
"""
LoVRR: LoRA-based Video Reasoning and Refinement
Setup script for package installation
"""

from pathlib import Path

from setuptools import setup, find_packages

root = Path(__file__).parent
long_description = (root / "README.md").read_text(encoding="utf-8")
requirements = (root / "requirements.txt").read_text().splitlines()

setup(
    name="lovrr",
    version="1.0.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="LoRA-based Video Reasoning and Refinement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/LoVRR",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lovrr-serve=lovrr.serve:main",
            "lovrr-infer=lovrr.infer:main",
        ],
    },
)
