![alt text](https://github.com/pycaret/pycaret/blob/master/pycaret2.2.png)

# PyCaret 2.2
![pytest on push](https://github.com/pycaret/pycaret/workflows/pytest%20on%20push/badge.svg) [![Documentation Status](https://readthedocs.org/projects/pip/badge/?version=stable)](http://pip.pypa.io/en/stable/?badge=stable) [![PyPI version](https://badge.fury.io/py/pycaret.svg)](https://badge.fury.io/py/pycaret) [![License](https://img.shields.io/pypi/l/ansicolortags.svg)](https://img.shields.io/pypi/l/ansicolortags.svg) [![Git count](http://hits.dwyl.com/pycaret/pycaret/pycaret.svg)](http://hits.dwyl.com/pycaret/pycaret/pycaret) [![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pycaretworkspace/shared_invite/zt-kdoe7hee-yvNANPHXPM9VtK7R6Npx4Q)

## What is PyCaret?
PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.

In comparison with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few words only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and many more. 

The design and simplicity of PyCaret is inspired by the emerging role of citizen data scientists, a term first used by Gartner. Citizen Data Scientists are power users who can perform both simple and moderately sophisticated analytical tasks that would previously have required more expertise. Seasoned data scientists are often difficult to find and expensive to hire but citizen data scientists can be an effective way to mitigate this gap and address data-related challenges in the business setting.

PyCaret is a great library which not only simplifies the machine learning tasks for citizen data scientists but also helps new startups to reduce the cost of investing in a team of data scientists. Therefore, this library has not only helped the citizen data scientists but has also helped individuals who want to start exploring the field of data science, having no prior knowledge in this field.

- Official Website: https://www.pycaret.org
- Documentation: https://pycaret.readthedocs.io/en/latest/

![alt text](https://github.com/pycaret/pycaret/blob/master/pycaret2-features.png)

## Current Release
PyCaret `2.2` is now available. See `2.2` release notes. The easiest way to install pycaret is using pip. 

```python
pip install pycaret
```

PyCaret's default installation is a slim version of pycaret which only installs hard dependencies that are listed in `requirements.txt`. To install the full version of pycaret, use the following command:

```python
pip install pycaret[full]
```

### Minor Release
- [December 22, 2020] `2.2.3` released fixing several bugs. Major compatibility issues of catboost, pyod (other impacts unknown as of now) with sklearn=0.24 (released on Dec 22, 2020). Temporary fix is requiring 0.23.2 specifically in the `requirements.txt`. [Click here](https://github.com/pycaret/pycaret/releases) to see release notes.
- [November 25, 2020] `2.2.2` released fixing several bugs. [Click here](https://github.com/pycaret/pycaret/releases) to see release notes.
- [November 9, 2020] `2.2.1` released fixing several bugs. [Click here](https://github.com/pycaret/pycaret/releases) to see release notes.

## PyCaret on GPU
PyCaret >= 2.2 provides the option to use GPU for select model training and hyperparameter tuning. There is no change in the use of the API, however, in some cases, additional libraries have to be installed as they are not installed with the default slim version or the full version. The following estimators can be trained on GPU.

- Extreme Gradient Boosting (requires no further installation)

- CatBoost (requires no further installation)

- Light Gradient Boosting Machine (requires GPU installation: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)

- Logistic Regression, Ridge Classifier, Random Forest, K Neighbors Classifier, K Neighbors Regressor, Support Vector Machine, Linear Regression, Ridge Regression, Lasso Regression (requires cuML >= 0.15 https://github.com/rapidsai/cuml)

If you are using Google Colab you can install Light Gradient Boosting Machine for GPU but first you have to uninstall LightGBM on CPU. Use the below command to do that:

```python
pip uninstall lightgbm -y

# install lightgbm GPU
pip install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"
```
CatBoost is only enabled on GPU when dataset has > 50,000 rows.

cuML >= 0.15 cannot be installed on Google Colab. Instead use blazingSQL (https://blazingsql.com/) which comes pre-installed with cuML 0.15. Use following command to install pycaret:

```python
# install pycaret on blazingSQL
!/opt/conda-environments/rapids-stable/bin/python -m pip install --upgrade pycaret
```

## Important Links
- Release notes: https://github.com/pycaret/pycaret/releases
- Docs: https://pycaret.readthedocs.io/en/latest/
- Tutorials: https://pycaret.readthedocs.io/en/latest/tutorials.html
- Example Notebooks: https://github.com/pycaret/pycaret/tree/master/examples
- Other Resources: https://github.com/pycaret/pycaret/tree/master/resources 
- Issue Logs: https://github.com/pycaret/pycaret/issues
- Contribute: https://pycaret.readthedocs.io/en/latest/contribute.html
- Join Slack Community: https://join.slack.com/t/pycaretworkspace/shared_invite/zt-kdoe7hee-yvNANPHXPM9VtK7R6Npx4Q  

## Who should use PyCaret?
PyCaret is an open source library that anybody can use. In our view the ideal target audience of PyCaret is: <br />

- Experienced Data Scientists who want to increase productivity.
- Citizen Data Scientists who prefer a low code machine learning solution.
- Data Science Students.
- Data Science Professionals who wants to build rapid prototypes.

## Current Contributors
<a href="https://github.com/pycaret/pycaret/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=pycaret/pycaret" />
</a>

Made with [contributors-img](https://contributors-img.web.app).

## License

Copyright 2021-2022 Moez Ali <moez.ali@queensu.ca>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
© 2021 GitHub, Inc.
