#!/usr/bin/env python3
"""
Setup script for the Gemini PDF Fine-tuning Pipeline package.
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="gemini-pdf-finetuning",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A pipeline for fine-tuning Gemini models on PDF documentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gemini-pdf-finetuning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gemini-pdf-finetuning=src.main:app",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
)
