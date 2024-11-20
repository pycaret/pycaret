"""
This is the setup script for the PyCaret library.
"""


# Copyright (C) 2019-2024 PyCaret
# Author: Moez Ali (moez.ali@queensu.ca)
# Contributors (https://github.com/pycaret/pycaret/graphs/contributors)
# License: MIT

from setuptools import find_packages, setup


def readme():
    """
    Reads the contents of the README.md file and returns it as a string.
    """
    with open("README.md", encoding="utf8") as readme_file:
        readme_content = readme_file.read()
    return readme_content


with open("requirements.txt", encoding="utf8") as req_file:
    required = req_file.read().splitlines()

with open("requirements-optional.txt", encoding="utf8") as req_file:
    required_optional = req_file.read()

with open("requirements-test.txt", encoding="utf8") as req_file:
    required_test = req_file.read().splitlines()

with open("requirements-dev.txt", encoding="utf8") as req_file:
    required_dev = req_file.read().splitlines()


extras_require = {
    "analysis": required_optional.split("\n\n")[0].splitlines(),
    "models": required_optional.split("\n\n")[1].splitlines(),
    "tuners": required_optional.split("\n\n")[2].splitlines(),
    "mlops": required_optional.split("\n\n")[3].splitlines(),
    "parallel": required_optional.split("\n\n")[4].splitlines(),
    "test": required_test,
    "dev": required_dev,
}

extras_require["full"] = (
    extras_require["analysis"]
    + extras_require["models"]
    + extras_require["tuners"]
    + extras_require["mlops"]
    + extras_require["parallel"]
    + extras_require["test"]
    + extras_require["dev"]
)

setup(
    name="pycaret",
    version="3.4.0",
    description="PyCaret - An open source, low-code machine learning library in Python.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/pycaret/pycaret",
    author="Moez Ali",
    author_email="moez.ali@queensu.ca",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(include=["pycaret*"]),
    include_package_data=True,
    install_requires=required,
    extras_require=extras_require,
    tests_require=required_test,
    python_requires=">=3.9",
    project_urls={
        "Contributors": "https://github.com/pycaret/pycaret/graphs/contributors",
        "Documentation": "https://pycaret.gitbook.io/docs",
        "Source": "https://github.com/pycaret/pycaret",
    },
)
