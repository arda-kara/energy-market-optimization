from setuptools import setup, find_packages

setup(
    name="energy_market_optimization",
    version="0.1.0",
    description="AI-driven energy distribution optimization using Reinforcement Learning",
    author="Arda",
    author_email="ardakara1881@hotmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "gymnasium>=0.26.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tensorboard>=2.6.0",
        "optuna>=2.10.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "PyYAML>=5.4.1",
    ],
    python_requires=">=3.8",
) 