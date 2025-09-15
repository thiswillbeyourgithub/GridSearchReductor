from setuptools import setup, find_packages
from setuptools.command.install import install

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="GridSearchReductor",
    version="0.3.1",
    description="Optimize hyperparameter search using Latin Hypercube Sampling principles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thiswillbeyourgithub/GridSearchReductor.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    license="GPLv3",
    keywords=[
        "latin-hypercube",
        "sampling",
        "grid-search",
        "hyperparameter-optimization",
        "machine-learning",
        "experiment-design",
        "parameter-tuning",
        "space-filling",
        "stratified-sampling",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy >= 1.24.0",
        "scikit-learn >= 1.3.0",
        "joblib",
    ],
)
