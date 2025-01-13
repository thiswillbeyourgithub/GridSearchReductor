
from setuptools import setup, find_packages
from setuptools.command.install import install

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="TaguchiGridSearchConverter",
    version="0.2.4",
    description="Optimize hyperparameter search using Taguchi array principles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thiswillbeyourgithub/TaguchiGridSearchConverter.git",
    packages=find_packages(),

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    license="GPLv3",
    keywords=["taguchi", "grid-search", "hyperparameter-optimization", "machine-learning", "experiment-design", "parameter-tuning"],
    python_requires=">=3.11",

    install_requires=[
        'numpy >= 1.24.0',
        'scikit-learn >= 1.3.0',
    ],
)
