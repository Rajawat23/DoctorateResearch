"""
setup.py
--------
Install HEUQ as an editable package.

    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="heuq",
    version="1.0.0",
    description=(
        "Heterogeneous Ensemble framework for Uncertainty Quantification (HEUQ)"
    ),
    author="Akash Singh, Ashwin Ittoo, Pierre Ars, Eric Vandomme",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "scikit-learn>=1.0.0",
        "numpy",
        "pandas",
        "xgboost>=1.6.0",
        "catboost>=1.0.0",
        "scipy",
    ],
)
