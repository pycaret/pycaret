# Copyright (C) 2019-2020 Moez Ali <moez.ali@queensu.ca>
# License: MIT, moez.ali@queensu.ca

import time
from setuptools import setup, find_packages

nightly_version = "3.0.0"

def readme():
    with open("README.md") as f:
        README = f.read()
    return README


with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("requirements-optional.txt") as f:
    required_optional = f.read().splitlines()

with open("requirements-test.txt") as f:
    required_test = f.read().splitlines()

setup(
    name="pycaret-ts-alpha",
    version=str(nightly_version) + ".dev" + str(int(time.time())),
    description="PyCaret - An open source, low-code machine learning library in Python.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/pycaret/pycaret",
    author="Moez Ali",
    author_email="moez.ali@queensu.ca",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(include=["pycaret*"]),
    include_package_data=True,
    install_requires=required,
    extras_require={
        "analysis": required_optional[1:9],
        "models": required_optional[11:15],
        "tuners": required_optional[17:22],
        "mlops": required_optional[24:30],
        "nlp": required_optional[32:38],
        "full": required_optional,
    },
    tests_require=required_test,
    python_requires=">=3.7",
)
