<div align="center">
  
<img src="logo.png" alt="drawing" width="200"/>

**An open-source, low-code machine learning library in Python 🚀**
  
<p align="center">
  <a href="https://www.pycaret.org">Official</a> •
  <a href="https://pycaret.readthedocs.io/en/latest/index.html">Docs</a> •
  <a href="https://pycaret.readthedocs.io/en/latest/installation.html">Install</a> •
  <a href="https://github.com/pycaret/pycaret/tree/master/tutorials">Tutorials</a> •
  <a href="https://github.com/pycaret/pycaret/discussions">Discussions</a> •
  <a href="https://pycaret.readthedocs.io/en/latest/contribute.html">Contribute</a> •
  <a href="https://github.com/pycaret/pycaret/tree/master/resources">Resources</a> •
  <a href="https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w">Slack</a>

</p>

[![Python](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://badge.fury.io/py/pycaret) 
![pytest on push](https://github.com/pycaret/pycaret/workflows/pytest%20on%20push/badge.svg) 
[![Documentation Status](https://readthedocs.org/projects/pip/badge/?version=stable)](http://pip.pypa.io/en/stable/?badge=stable) 
[![PyPI version](https://badge.fury.io/py/pycaret.svg)](https://badge.fury.io/py/pycaret) 
[![License](https://img.shields.io/pypi/l/ansicolortags.svg)](https://img.shields.io/pypi/l/ansicolortags.svg) 
<!-- [![Git count](http://hits.dwyl.com/pycaret/pycaret/pycaret.svg)](http://hits.dwyl.com/pycaret/pycaret/pycaret) -->
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w)

![alt text](https://github.com/pycaret/pycaret/blob/master/quick_start.gif)

<div align="left">
  
## What is PyCaret?
PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.

In comparison with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few words only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and many more.

The design and simplicity of PyCaret are inspired by the emerging role of citizen data scientists, a term first used by Gartner. Citizen Data Scientists are power users who can perform both simple and moderately sophisticated analytical tasks that would previously have required more expertise. Seasoned data scientists are often difficult to find and expensive to hire but citizen data scientists can be an effective way to mitigate this gap and address data-related challenges in the business setting.

## Installation

PyCaret's default installation only installs hard dependencies as listed in the [requirements.txt](requirements.txt) file. 

```python
pip install pycaret
```
To install the full version:

```python
pip install pycaret[full]
```

<div align="center">

## Supervised Workflow
  
  Classification           |  Regression
:-------------------------:|:-------------------------:
![](pycaret_classification.png)  | ![](pycaret_regression.png)

 ## Unsupervised Workflow
  
  Clustering               |  Anomaly Detection
:-------------------------:|:-------------------------:
![](pycaret_clustering.png)  |  ![](pycaret_anomaly.png)  
  
<div align="left">

## PyCaret ⚡NEW⚡ Time Series Module
  
PyCaret new time series module is now available in beta. Staying true to simplicity of PyCaret, it is consistent with our existing API and fully loaded with functionalities. Statistical testing, model training and selection (30+ algorithms), model analysis, automated hyperparameter tuning, experiment logging, deployment on cloud, and more. All of this with only few lines of code (just like the other modules of pycaret). If you would like to give it a try, checkout our official [quick start](https://github.com/pycaret/pycaret/blob/time_series/time_series_101.ipynb) notebook.
  
The module is still in beta. We are adding new functionalities every day and doing weekly pip releases. Please ensure to create a separate python environment to avoid dependency conflicts with main pycaret. The final release of this module will be merged with the main pycaret in next major release.
  
 ### Install Now 👇
  
 ```
 pip install pycaret-ts-alpha
 ```  
  


![alt text](pycaret_ts_quickdemo.gif)  

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

## Who should use PyCaret?
PyCaret is an open source library that anybody can use. In our view the ideal target audience of PyCaret is: <br />

- Experienced Data Scientists who want to increase productivity.
- Citizen Data Scientists who prefer a low code machine learning solution.
- Data Science Students.
- Data Science Professionals who want to build rapid prototypes.

## Contributors
<a href="https://github.com/pycaret/pycaret/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=pycaret/pycaret" />
</a>

Made with [contributors-img](https://contributors-img.web.app).
