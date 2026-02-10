"""Setup configuration for bridge_verifier package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if available
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="bridge-verifier",
    version="0.1.0",
    description="Bridge Verifier - Ensemble model for robot action verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    packages=["bridge_verifier", "bridge_verifier.ensemble_eval"],
    package_dir={"bridge_verifier": "."},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "open_clip_torch",
        "timm",
        "pillow",
        "tqdm",
        "ijson",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
)
