Contribute
===========

Thank you for choosing to contribute to PyCaret. There are a ton of
great open-source projects out there, so we appreciate your interest in
contributing to PyCaret. It is an open-source, low-code machine learning
library in Python developed and open-sourced in April 2020 by Moez Ali
moez.ali@queensu.ca and is now maintained by awesome community members
just like you. In this documentation, we will cover a couple of ways you
can contribute to this project.

Documentation
-------------

There is always room for improvement in documentation. We welcome all
the pull requests to fix typo / improve grammar or semantic structuring
of documents. Here are a few documents you can work on:

-  Official Tutorials:
   https://github.com/pycaret/pycaret/tree/master/tutorials
-  README.md https://github.com/pycaret/pycaret/blob/master/README.md
-  Functional Documentation / Docstrings:
   https://github.com/pycaret/pycaret/tree/master/pycaret

Open Issues
-----------

If you would like to help in working on open issues. Look out for
following tags: ``good first issue`` ``help wanted``
``open for contribution``

Medium Writers
--------------

If you are interested or have already written a Medium story covering
``PyCaret``. You can submit your story in a ``markdown`` format. Submit
a PR to https://github.com/pycaret/pycaret/tree/master/resources. To
convert medium stories into ``markdown`` format please download this
chrome extension:
https://chrome.google.com/webstore/detail/export-to-markdown/dodkihcbgpjblncjahodbnlgkkflliim

Major Contribution
------------------

If you are willing to make a major contribution you can always lookout
for the active sprint under ``Projects`` and discuss the proposal with
sprint leader.

What we currently need help on?
-------------------------------
-  Improving unit-test cases and test coverage
   https://github.com/pycaret/pycaret/tree/master/tests
-  Refactor preprocessing pipeline to support GPU
-  Dask Integration

Development setup
-----------------
Follow `installation instructions <https://pycaret.readthedocs.io/en/latest/installation.html#installing-the-latest-release>`_ to first create a virtual environment. Then, install development version of the package:

.. code-block:: shell

    pip install -e .[test]

We use `pre-commit <https://pre-commit.com>`_ with `black <https://github.com/psf/black>`_ for code formatting. It runs automatically before you make a new commit. To set up pre-commit, follow these steps:

1. Install pre-commit:

.. code-block:: shell

    pip install pre-commit

2. Set up pre-commit:

.. code-block:: shell

    pre-commit install

Unit testing
------------
Install development version of the package with additional extra dependencies required for unit testing:

.. code-block:: shell

    pip install -e .[test]
    python -m spacy download en_core_web_sm

We use `pytest <https://docs.pytest.org/en/latest/>`_ for unit testing.

To run tests, except skipped ones (search for ``@pytest.mark.skip`` decorator over test functions), run:

.. code-block:: shell

    pytest pycaret

Documentation
-------------
We use `sphinx <https://www.sphinx-doc.org/>`_ to build our documentation and `readthedocs <https://pycaret.readthedocs.io/en/latest/index.html>`_ to host it. The source files can be found in ``docs/source/``. The main configuration file for sphinx is ``conf.py`` and the main page is ``index.rst``.

To build the documentation locally, you need to install a few extra dependencies listed in
``docs/source/requirements.txt``:

.. code-block:: shell

    pip install -r docs/source/requirements.txt

To build the website locally, run:

.. code-block:: shell

    sh make.sh

You can find the generated files in the ``docs/build/`` folder. To view the website, open  ``docs/build/index.html`` with your preferred web browser.
