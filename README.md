<div align="center">
  
<img src="docs/images/logo.png" alt="drawing" width="200"/>

## **An open-source, low-code machine learning library in Python** </br>
### :rocket: **PyCaret 3.0-rc is now available. `pip install --pre pycaret`**
  
<p align="center">
  <a href="https://www.pycaret.org">Official</a> •
  <a href="https://pycaret.gitbook.io/">Docs</a> •
  <a href="https://pycaret.gitbook.io/docs/get-started/installation">Install</a> •
  <a href="https://pycaret.gitbook.io/docs/get-started/tutorials">Tutorials</a> •
  <a href="https://pycaret.gitbook.io/docs/learn-pycaret/faqs">FAQs</a> •
  <a href="https://pycaret.gitbook.io/docs/learn-pycaret/cheat-sheet">Cheat sheet</a> •
  <a href="https://github.com/pycaret/pycaret/discussions">Discussions</a> •
  <a href="https://pycaret.readthedocs.io/en/latest/contribute.html">Contribute</a> •
  <a href="https://pycaret.gitbook.io/docs/learn-pycaret/official-blog">Blog</a> •
  <a href="https://www.linkedin.com/company/pycaret/">LinkedIn</a> • 
  <a href="https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g">YouTube</a> • 
  <a href="https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w">Slack</a>

</p>

[![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://badge.fury.io/py/pycaret) 
![pytest on push](https://github.com/pycaret/pycaret/workflows/pytest%20on%20push/badge.svg) 
[![Documentation Status](https://readthedocs.org/projects/pip/badge/?version=stable)](http://pip.pypa.io/en/stable/?badge=stable) 
[![PyPI version](https://badge.fury.io/py/pycaret.svg)](https://badge.fury.io/py/pycaret) 
[![License](https://img.shields.io/pypi/l/ansicolortags.svg)](https://img.shields.io/pypi/l/ansicolortags.svg) 
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w)

![alt text](docs/images/quick_start.gif)

<div align="left">
  
# Welcome to PyCaret
PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.

In comparison with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few lines only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, Hyperopt, Ray, and few more.

The design and simplicity of PyCaret are inspired by the emerging role of citizen data scientists, a term first used by Gartner. Citizen Data Scientists are power users who can perform both simple and moderately sophisticated analytical tasks that would previously have required more technical expertise. PyCaret was inspired by the caret library in R programming language.
 
# 🚀 Installation

## 🌐 Option 1: Install via PyPi
PyCaret is tested and supported on 64-bit systems with:
- Python 3.7, 3.8, 3.9, and 3.10
- Ubuntu 16.04 or later
- Windows 7 or later

You can install PyCaret with Python's pip package manager:

```python
pip install pycaret
```

PyCaret's default installation will not install all the optional dependencies automatically. Depending on the use case, you may be interested in one or more extras:
    
```python
# install analysis extras
pip install pycaret[analysis]

# models extras
pip install pycaret[models]

# install tuner extras
pip install pycaret[tuner]

# install mlops extras
pip install pycaret[mlops]

# install parallel extras
pip install pycaret[parallel]

# install test extras
pip install pycaret[test]

## 

# install multiple extras together
pip install pre pycaret[analysis,models]
```

Check out all [optional dependencies](https://github.com/pycaret/pycaret/blob/master/requirements-optional.txt). If you want to install everything including all the optional dependencies:

```python
# install full version
pip install pycaret[full]
```
## 📄 Option 2: Build from Source
Install the development version of the library directly from the source. The API may be unstable. It is not recommended for production use. 

```python
pip install git+https://github.com/pycaret/pycaret.git@master --upgrade
```

## 📦 Option 3: Docker
Docker creates virtual environments with containers that keep a PyCaret installation separate from the rest of the system. PyCaret docker comes pre-installed with a Jupyter notebook. It can share resources with its host machine (access directories, use the GPU, connect to the Internet, etc.). The PyCaret Docker images are always tested for the latest major releases.

```python
# default version
docker run -p 8888:8888 pycaret/slim

# full version
docker run -p 8888:8888 pycaret/full
```

## 💡 Example Workflows
<div align="center">

## **Classification**
  
  Functional API           |  OOP API
:-------------------------:|:-------------------------:
![](docs/images/classification_functional.png)  | ![](docs/images/classification_OOP.png)
 
## **Regression**
  
  Functional API           |  OOP API
:-------------------------:|:-------------------------:
![](docs/images/regression_functional.png)  | ![](docs/images/regression_OOP.png)

## **Time Series**
  
  Functional API           |  OOP API
:-------------------------:|:-------------------------:
![](docs/images/time_series_functional.png)  | ![](docs/images/time_series_OOP.png)

## **Clustering**
  
  Functional API           |  OOP API
:-------------------------:|:-------------------------:
![](docs/images/clustering_functional.png)  | ![](docs/images/clustering_OOP.png)

## **Anomaly Detection**
  
  Functional API           |  OOP API
:-------------------------:|:-------------------------:
![](docs/images/anomaly_functional.png)  | ![](docs/images/anomaly_OOP.png)

<div align="left">

# Who should use PyCaret?
PyCaret is an open source library that anybody can use. In our view the ideal target audience of PyCaret is: <br />

- Experienced Data Scientists who want to increase productivity.
- Citizen Data Scientists who prefer a low code machine learning solution.
- Data Science Professionals who want to build rapid prototypes.
- Data Science and Machine Learning students and enthusiasts.
  
# Training on GPUs
To train models on the GPU, simply pass use_gpu = True in the setup function. There is no change in the use of the API; however, in some cases, additional libraries have to be installed. The following models can be trained on GPUs:

- Extreme Gradient Boosting
- CatBoost
- Light Gradient Boosting Machine requires [GPU installation](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
- Logistic Regression, Ridge Classifier, Random Forest, K Neighbors Classifier, K Neighbors Regressor, Support Vector Machine, Linear Regression, Ridge Regression, Lasso Regression requires [cuML >= 0.15](https://github.com/rapidsai/cuml)

# PyCaret Intel sklearnex support
You can apply [Intel optimizations](https://github.com/intel/scikit-learn-intelex) for machine learning algorithms and speed up your workflow. To train models with Intel optimizations use `sklearnex` engine. There is no change in the use of the API, however, installation of Intel sklearnex is required:

```python
pip install scikit-learn-intelex
```

# Contributors
<a href="https://github.com/pycaret/pycaret/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=pycaret/pycaret" width = 500/>
</a>

# License
PyCaret is completely free and open-source and licensed under the [MIT](https://github.com/pycaret/pycaret/blob/master/LICENSE) license. 

# More Information

| Important Links              |                                                                |
| -------------------------- | -------------------------------------------------------------- |
| :star: **[Tutorials]**        | New to PyCaret? Checkout our official notebooks!            |
| :clipboard: **[Example Notebooks]** | Example notebooks created by community.               |
| :orange_book: **[Official Blog]** | Tutorials and articles by contributors.                      |
| :books: **[Documentation]**      | The detailed API docs of PyCaret                         |
| :tv: **[Video Tutorials]**            | Our video tutorial from various events.             |
| ✈️ **[Cheat sheet]**            | Cheat sheet for all functions across modules.             |
| :loudspeaker: **[Discussions]**        | Have questions? Engage with community and contributors.|
| :hammer_and_wrench: **[Changelog]**          | Changes and version history.                 |
| :deciduous_tree: **[Roadmap]**          | PyCaret's software and community development plan.|

[tutorials]: https://pycaret.gitbook.io/docs/get-started/tutorials
[Example notebooks]: https://github.com/pycaret/examples
[Official Blog]: https://pycaret.gitbook.io/docs/learn-pycaret/official-blog
[Documentation]: https://pycaret.gitbook.io
[video tutorials]: https://pycaret.gitbook.io/docs/learn-pycaret/videos
[Cheat sheet]: https://pycaret.gitbook.io/docs/learn-pycaret/cheat-sheet
[Discussions]: https://github.com/pycaret/pycaret/discussions
[changelog]: https://pycaret.gitbook.io/docs/get-started/release-notes
[roadmap]: https://github.com/pycaret/pycaret/issues/1756
