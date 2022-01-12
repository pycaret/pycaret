<div align="center">
  
<img src="docs/images/logo.png" alt="drawing" width="200"/>

**An open-source, low-code machine learning library in Python** </br>
:rocket: **Version 2.3.6 out now!** [Check out the release notes here](https://github.com/pycaret/pycaret/releases).
  
<p align="center">
  <a href="https://www.pycaret.org">Official</a> •
  <a href="https://pycaret.readthedocs.io/en/latest/index.html">Docs</a> •
  <a href="https://pycaret.readthedocs.io/en/latest/installation.html">Install</a> •
  <a href="https://github.com/pycaret/pycaret/tree/master/tutorials">Tutorials</a> •
  <a href="https://github.com/pycaret/pycaret/discussions">Discussions</a> •
  <a href="https://pycaret.readthedocs.io/en/latest/contribute.html">Contribute</a> •
  <a href="https://github.com/pycaret/pycaret/tree/master/resources">Resources</a> •
  <a href="https://moez-62905.medium.com/">Medium</a> •
  <a href="https://www.linkedin.com/company/pycaret/">LinkedIn</a> • 
  <a href="https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g">YouTube</a> • 
  <a href="https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w">Slack</a>

</p>

[![Python](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://badge.fury.io/py/pycaret) 
![pytest on push](https://github.com/pycaret/pycaret/workflows/pytest%20on%20push/badge.svg) 
[![Documentation Status](https://readthedocs.org/projects/pip/badge/?version=stable)](http://pip.pypa.io/en/stable/?badge=stable) 
[![PyPI version](https://badge.fury.io/py/pycaret.svg)](https://badge.fury.io/py/pycaret) 
[![License](https://img.shields.io/pypi/l/ansicolortags.svg)](https://img.shields.io/pypi/l/ansicolortags.svg) 
<!-- [![Git count](http://hits.dwyl.com/pycaret/pycaret/pycaret.svg)](http://hits.dwyl.com/pycaret/pycaret/pycaret) -->
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w)

![alt text](docs/images/quick_start.gif)

<div align="left">
  
## Welcome to PyCaret
PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.

In comparison with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few lines only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and few more.

The design and simplicity of PyCaret are inspired by the emerging role of citizen data scientists, a term first used by Gartner. Citizen Data Scientists are power users who can perform both simple and moderately sophisticated analytical tasks that would previously have required more technical expertise.

| Important Links              |                                                                |
| -------------------------- | -------------------------------------------------------------- |
| :star: **[Tutorials]**        | New to PyCaret? Checkout our official notebooks!            |
| :clipboard: **[Example Notebooks]** | Example notebooks created by community.               |
| :orange_book: **[Blog]** | Tutorials and articles by contributors.                      |
| :books: **[Documentation]**      | The detailed API docs of PyCaret                         |
| :tv: **[Video Tutorials]**            | Our video tutorial from various events.             |
| :loudspeaker: **[Discussions]**        | Have questions? Engage with community and contributors.|
| :hammer_and_wrench: **[Changelog]**          | Changes and version history.                 |
| :deciduous_tree: **[Roadmap]**          | PyCaret's software and community development plan.|
  
[tutorials]: https://github.com/pycaret/pycaret/tree/master/tutorials
[Example notebooks]: https://github.com/pycaret/pycaret/tree/master/examples
[Blog]: https://github.com/pycaret/pycaret/tree/master/resources
[Documentation]: https://pycaret.readthedocs.io/en/latest/index.html
[video tutorials]: https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g
[Discussions]: https://github.com/pycaret/pycaret/discussions
[changelog]: https://github.com/pycaret/pycaret/blob/master/CHANGELOG.md
[roadmap]: https://github.com/pycaret/pycaret/issues/1756
 
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
![](docs/images/pycaret_classification.png)  | ![](docs/images/pycaret_regression.png)

 ## Unsupervised Workflow
  
  Clustering               |  Anomaly Detection
:-------------------------:|:-------------------------:
![](docs/images/pycaret_clustering.png)  |  ![](docs/images/pycaret_anomaly.png)  
  
<div align="left">

## PyCaret ⚡NEW⚡ Time Series Module
  
PyCaret new time series module is now available in beta. Staying true to simplicity of PyCaret, it is consistent with our existing API and fully loaded with functionalities. Statistical testing, model training and selection (30+ algorithms), model analysis, automated hyperparameter tuning, experiment logging, deployment on cloud, and more. All of this with only few lines of code (just like the other modules of pycaret). If you would like to give it a try, checkout our official [quick start](https://nbviewer.org/github/pycaret/pycaret/blob/time_series_beta/time_series_101.ipynb) notebook.
  
:books: [Time Series Docs](https://pycaret.readthedocs.io/en/time_series/api/time_series.html)
  
:question: [Time Series FAQs](https://github.com/pycaret/pycaret/discussions/categories/faqs?discussions_q=category%3AFAQs+label%3Atime_series)
  
:rocket: [Features and Roadmap](https://github.com/pycaret/pycaret/issues/1648)
  
The module is still in beta. We are adding new functionalities every day and doing weekly pip releases. Please ensure to create a separate python environment to avoid dependency conflicts with main pycaret. The final release of this module will be merged with the main pycaret in next major release.
  
 ```
 pip install pycaret-ts-alpha
 ```  

![alt text](docs/images/pycaret_ts_quickdemo.gif)  

## Who should use PyCaret?
PyCaret is an open source library that anybody can use. In our view the ideal target audience of PyCaret is: <br />

- Experienced Data Scientists who want to increase productivity.
- Citizen Data Scientists who prefer a low code machine learning solution.
- Data Science Professionals who want to build rapid prototypes.
- Data Science and Machine Learning students and enthusiasts.
  
## PyCaret on GPU
With PyCaret >= 2.2, you can train models on GPU and speed up your workflow by 10x. To train models on GPU simply pass `use_gpu = True` in the setup function. There is no change in the use of the API, however, in some cases, additional libraries have to be installed as they are not installed with the default version or the full version. As of the latest release, the following models can be trained on GPU:

- Extreme Gradient Boosting (requires no further installation)
- CatBoost (requires no further installation)
- Light Gradient Boosting Machine requires [GPU installation](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
- Logistic Regression, Ridge Classifier, Random Forest, K Neighbors Classifier, K Neighbors Regressor, Support Vector Machine, Linear Regression, Ridge Regression, Lasso Regression requires [cuML >= 0.15](https://github.com/rapidsai/cuml)

## License
PyCaret is completely free and open-source and licensed under the [MIT](https://github.com/pycaret/pycaret/blob/master/LICENSE) license. 

## Contributors
<a href="https://github.com/pycaret/pycaret/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=pycaret/pycaret" width = 500/>
</a>
