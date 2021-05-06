# Copyright (C) 2019-2020 Moez Ali <moez.ali@queensu.ca>
# License: MIT, moez.ali@queensu.ca

from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        README = f.read()
    return README


with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("requirements-optional.txt") as f:
    optional_required = f.read().splitlines()

setup(
    name="pycaret",
    version="2.3.1",
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
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    install_requires=required,
    extras_require={"full": optional_required,},
)
