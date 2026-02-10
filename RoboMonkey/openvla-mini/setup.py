"""Setup configuration for openvla-mini package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if available
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="openvla-mini",
    version="0.1.0",
    description="OpenVLA Mini - Robot Manipulation Experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "torch",
        "tensorflow",
        "pillow",
        "transforms3d",
        "simpler_env",
        "imageio",
        "tqdm",
        "draccus",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
)
