![alt text](https://github.com/pycaret/pycaret/blob/master/pycaret2.png)

# PyCaret 2.0
[![Build Status](https://travis-ci.com/pycaret/pycaret.svg?branch=master)](https://travis-ci.com/pycaret/pycaret) [![Stability](https://img.shields.io/badge/stability-stable-green.svg)](https://img.shields.io/badge/stability-stable-green.svg) [![Documentation Status](https://readthedocs.org/projects/pip/badge/?version=stable)](http://pip.pypa.io/en/stable/?badge=stable) [![PyPI version](https://badge.fury.io/py/pycaret.svg)](https://badge.fury.io/py/pycaret) [![License](https://img.shields.io/pypi/l/ansicolortags.svg)](https://img.shields.io/pypi/l/ansicolortags.svg) [![Git count](http://hits.dwyl.com/pycaret/pycaret/pycaret.svg)](http://hits.dwyl.com/pycaret/pycaret/pycaret)

## What is PyCaret?
PyCaret is an open source `low-code` machine learning library in Python that aims to reduce the hypothesis to insights cycle time in a ML experiment. It enables data scientists to perform end-to-end experiments quickly and efficiently. In comparison with the other open source machine learning libraries, PyCaret is an alternate low-code library that can be used to perform complex machine learning tasks with only few lines of code. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as `scikit-learn`, `XGBoost`, `Microsoft LightGBM`, `spaCy` and many more. 

The design and simplicity of PyCaret is inspired by the emerging role of `citizen data scientists`, a term first used by Gartner. Citizen Data Scientists are `power users` who can perform both simple and moderately sophisticated analytical tasks that would previously have required more expertise. Seasoned data scientists are often difficult to find and expensive to hire but citizen data scientists can be an effective way to mitigate this gap and address data related challenges in business setting.

PyCaret is `simple`, `easy to use` and `deployment ready`. All the steps performed in a ML experiment can be reproduced using a pipeline that is automatically developed and orchestrated in PyCaret as you progress through the experiment. A `pipeline` can be saved in a binary file format that is transferable across environments.

For more information on PyCaret, please visit our official website https://www.pycaret.org

![alt text](https://github.com/pycaret/pycaret/blob/master/pycaret2-features.png)

## Current Release
PyCaret `2.0` is now available. See `2.0` release notes. The easiest way to install pycaret is using pip. 

```python
pip install pycaret==2.0
```

## Important Links
- 2.0 Release notes: https://github.com/pycaret/pycaret/releases/tag/2.0
- User Guide / Documentation: https://www.pycaret.org/guide
- Getting Started Tutorials: https://www.pycaret.org/tutorial
- Issue Logs: https://github.com/pycaret/pycaret/issues

## Who should use PyCaret?
PyCaret is an open source library that anybody can use. In our view the ideal target audience of PyCaret is: <br />

- Experienced Data Scientists who want to increase productivity.
- Citizen Data Scientists who prefer a low code machine learning solution.
- Students of Data Science.
- Data Science Professionals and Consultants involved in building Proof of Concept projects.

## Dependencies
Please read requirements.txt for list of requirements. They are automatically installed when pycaret is installed using pip. `shap`, `awscli` and `psutil` is removed from hard dependency in PyCaret 2.0. Must be installed separately, if needed.

## Note:
It supports Python 64 bit only.

## License

Copyright 2019-2020 Moez Ali <moez.ali@queensu.ca>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
© 2020 GitHub, Inc.
