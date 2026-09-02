from setuptools import setup, find_packages

setup(
    name="pssp",
    version="0.1.0",
    description="Protein Secondary Structure Prediction with Bi-RNN and Bi-LSTM",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black>=23.0", "ruff>=0.1"],
    },
)
