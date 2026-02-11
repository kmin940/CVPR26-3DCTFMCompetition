"""
CVPR 2026: Foundation Models for 3D Computed Tomography
Setup configuration for the challenge repository
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cvpr26-3dctfm",
    version="0.1.0",
    author="Sumin Kim",
    description="Foundation Models for 3D Computed Tomography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kmin940/CVPR26-3DCTFMCompetition",  # Update with actual repo URL
    packages=find_packages(include=["data_utils", "metrics"]),
    python_requires=">=3.12",
    install_requires=[
        # Core dependencies
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",

        # Medical imaging
        "SimpleITK>=2.1.0",
        "h5py>=3.1.0",

        # Metrics and visualization
        "torchmetrics>=0.11.0",
        "matplotlib>=3.3.0",

        # Logging and experiment tracking
        "wandb>=0.13.0",
        # Hugging Face
        "huggingface_hub>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "docker": [
            # Additional dependencies for Docker-based feature extraction
            "docker>=6.0.0",
        ],
    },
    scripts=[
        "run_linear_probe.py",
        "cvpr26_extract_feat_docker.py",
    ],
    entry_points={
        "console_scripts": [
            "cvpr26-linear-probe=run_linear_probe:main",
            "cvpr26-extract-feat=cvpr26_extract_feat_docker:main",
        ],
    },
)
