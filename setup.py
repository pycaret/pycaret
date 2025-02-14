"""Setup script for the PyCaret library."""

# Copyright (C) 2019-2024 PyCaret
# Author: Moez Ali (moez.ali@queensu.ca)
# License: MIT

from typing import Dict, List
from setuptools import find_packages, setup

def read_file(filename: str) -> str:
    """Read and return contents of a file.
    
    Args:
        filename: Path to the file to read
        
    Returns:
        str: Contents of the file
    """
    with open(filename, encoding="utf8") as file:
        return file.read()

def get_requirements() -> Dict[str, List[str]]:
    """Get all package requirements.
    
    Returns:
        Dict containing main and optional requirements
    """
    required = read_file("requirements.txt").splitlines()
    required_optional = read_file("requirements-optional.txt")
    required_test = read_file("requirements-test.txt").splitlines()
    required_dev = read_file("requirements-dev.txt").splitlines()

    # Split optional requirements into categories
    optional_sections = required_optional.split("\n\n")
    extras_require = {
        "analysis": optional_sections[0].splitlines(),
        "models": optional_sections[1].splitlines(),
        "tuners": optional_sections[2].splitlines(),
        "mlops": optional_sections[3].splitlines(),
        "parallel": optional_sections[4].splitlines(),
        "test": required_test,
        "dev": required_dev,
    }

    # Combine all extras for 'full' installation
    extras_require["full"] = sum(
        [extras_require[key] for key in extras_require], 
        []
    )

    return {"required": required, "extras": extras_require}

# Main setup configuration
requirements = get_requirements()
setup(
    name="pycaret",
    version="3.4.0",
    description="PyCaret - An open source, low-code machine learning library in Python.",
    long_description=read_file("README.md"),
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
        "Operating System :: OS Independent",
    ],
    packages=find_packages(include=["pycaret*"]),
    include_package_data=True,
    install_requires=requirements["required"],
    extras_require=requirements["extras"],
    tests_require=requirements["extras"]["test"],
    python_requires=">=3.9",
    project_urls={
        "Contributors": "https://github.com/pycaret/pycaret/graphs/contributors",
        "Documentation": "https://pycaret.gitbook.io/docs", 
        "Source": "https://github.com/pycaret/pycaret",
    },
)