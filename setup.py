#!/usr/bin/env python3
"""
LoVRR: LoRA-based Video Reasoning and Refinement
Setup script for package installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    py_modules=["deepthinker_service", "deepthinker_client"],
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
    install_requires=[
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.45.2",
        "peft==0.14.0",
        "accelerate==1.2.1",
        "gradio>=5.0.0",
        "nncore>=0.4.5",
        "decord>=0.6.0",
        "numpy==1.26.4",
        "pandas>=2.2.0",
        "PyYAML>=6.0",
        "requests>=2.28.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "deepthinker": [
            "vllm==0.6.3.post1",
            "fastapi>=0.107.0,!=0.113.*,!=0.114.0",
            "uvicorn>=0.23.0",
            "qwen-vl-utils==0.0.4",
            "av>=12.0.0,<14.0.0",
            "pydantic==2.10.6",
        ],
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
            "deepthinker-serve=deepthinker_service:main",
        ],
    },
)
