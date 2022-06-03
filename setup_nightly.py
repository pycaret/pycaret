# Copyright (C) 2019-2020 Moez Ali <moez.ali@queensu.ca>
# License: MIT, moez.ali@queensu.ca

import time

from setuptools import find_packages, setup

nightly_version = "3.0.0"

nightly_readme = f"This is a nightly version of the [PyCaret](https://pypi.org/project/pycaret/) library, intended as a preview of the upcoming {nightly_version} version. It may contain unstable and untested code.\n"


def readme():
    with open("README.md") as f:
        README = f.read()
    return README


with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("requirements-optional.txt") as f:
    required_optional = f.read()

with open("requirements-test.txt") as f:
    required_test = f.read().splitlines()

setup(
    name="pycaret-nightly",
    version=str(nightly_version) + ".dev" + str(int(time.time())),
    description="Nightly version of PyCaret - An open source, low-code machine learning library in Python.",
    long_description=nightly_readme + readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/pycaret/pycaret-nightly",
    author="Moez Ali",
    author_email="moez.ali@queensu.ca",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
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
        "analysis": required_optional.split("\n\n")[0].splitlines(),
        "models": required_optional.split("\n\n")[1].splitlines(),
        "tuners": required_optional.split("\n\n")[2].splitlines(),
        "mlops": required_optional.split("\n\n")[3].splitlines(),
        "nlp": required_optional.split("\n\n")[4].splitlines(),
        "full": required_optional.splitlines(),
    },
    tests_require=required_test,
    python_requires=">=3.7",
)
