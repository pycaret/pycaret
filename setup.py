# Copyright (C) 2019-2020 Moez Ali <moez.ali@queensu.ca>
# License: MIT, moez.ali@queensu.ca

from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="pycaret",
    version="1.0.1",
    description="An open source, low-code machine learning library in Python.",
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
    include_package_data=True,
    install_requires=required
)
