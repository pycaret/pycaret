# Contribution Guidelines

Thank you for choosing to contribute in PyCaret. There are a ton of great open-source projects out there, so we appreciate your interest in contributing to PyCaret. 

# Documentation
There is always a room for improvement in documentation. We welcome all the pull requests to fix typo / improve grammar or semantic structuring of documents. Here are few documents you can work on:

- [Official Documentation](https://github.com/pycaret/pycaret-docs) 
- [Tutorials](https://github.com/pycaret/pycaret/tree/master/tutorials)
- [Docstrings](https://pycaret.readthedocs.io/en/stable/)

# Open Issues
If you would like to help in working on open issues. Lookout for following tags: `good first issue`, `help wanted`, and `open for contribution`.


# Development setup
Follow [installation instructions](https://pycaret.readthedocs.io/en/latest/installation.html#installing-the-latest-release) to first create a virtual environment. Then, install the development version of the package:
```shell
pip install -e .[test]
```

We use [black](https://github.com/psf/black) and [isort](https://github.com/PyCQA/isort)
for code formatting. Make sure to run `isort pycaret` and `black pycaret`
from the home directory before creating the PR. Failing to do so can result
in a failed build, which would prevent the adoption of your code.


# Unit testing
Install development version of the package with additional extra dependencies required for unit testing:
```shell
pip install -e .[test]
python -m spacy download en_core_web_sm
```
We use [`pytest`](https://docs.pytest.org/en/latest/) for unit testing.

To run tests, except skipped ones (search for `@pytest.mark.skip` decorator over test functions), run:
```shell
pytest pycaret
```

# Documentation
We use [`sphinx`](https://www.sphinx-doc.org/) to build our documentation and [readthedocs](https://pycaret.readthedocs.io/en/latest/index.html) to host it. The source files can be found in [`docs/source/`](docs/source). The main configuration file for sphinx is [`conf.py`](docs/source/conf.py) and the main page is [`index.rst`](docs/source/index.rst).

To build the documentation locally, you need to install a few extra dependencies listed in
[`docs/source/requirements.txt`](docs/source/requirements.txt):
```shell
pip install -r docs/source/requirements.txt
```
To build the website locally, run:
```shell
sh make.sh
```
You can find the generated files in the `docs/build/` folder. To view the website, open `docs/build/index.html` with your preferred web browser.