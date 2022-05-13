# PyCaret for Machine Learning

> Automate ML workflows with PyCaret

Automate ML workflows with PyCaret
----------------------------------

![](https://miro.medium.com/max/1400/1*NHyb8AwNKSiGQuNkbBPgJA.png)

PyCaret — Caret but not Carrot ;)

Really? I can get this quick result with one line of code! Well, there come PyCaret the awsome an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that exponentially speeds up the experiment cycle and makes data scientist more productive.

PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and a few more.

![](https://miro.medium.com/max/1400/1*7052vV2a8m2gQwwbGMsMHw.png)

PyCaret Features (Image by Author)

In this article, we will discuss about:

1.  **Installing PyCaret**
2.  **Modules in PyCaret**
3.  **PyCaret ML use-case**

**pip**

Use pip to install the PyCaret library as follow:

pip install pyCaret

To install the full version:

pip install pycaret\[full\]

**conda-forge**

To install and its core dependencies you can use:

conda install -c conda-forge pycaret

In this section we cover how to use different modules in pycaret.We will see the code sample for following problem.

> **1\. Regression**
> 
> **2\. Classification**
> 
> **3\. Time series**
> 
> **4\. Clustering**
> 
> **5\. Anomaly Detection**

3.1 Regression Module in PyCaret
--------------------------------

PyCaret’s regression module [**pycaret.regression**](https://pycaret.readthedocs.io/en/latest/api/regression.html) is a supervised machine learning module used for predicting values or outcomes using various algorithms and techniques. It has over 25 algorithms and 10 plots to analyze the performance of the models. Be it ensembling, hyper-parameter tuning, or advanced tuning like stacking, PyCaret is your one-stop for all ML solutions.

You can train models on GPU in PyCaret and speed up your workflow by 10x. To train models on GPU simply pass use\_gpu = True in the setup function. There is no change in the use of the API, however, in some cases, additional libraries have to be installed as they are not installed with the default version or the full version.PyCaret regression module [**tutorial**](https://www.pycaret.org/tutorials/html/REG101.html) for more info.

Code sample : Regression with PyCaret

3.2 Classification Module in PyCaret
------------------------------------

The PyCaret classification module [**pycaret.classification**](https://pycaret.readthedocs.io/en/latest/api/classification.html) can be used for Binary or Multi-class classification problems. It has over 18 algorithms and 14 plots to analyze the performance of models. Be it hyper-parameter tuning, ensembling, or advanced techniques like stacking, PyCaret’s classification module has it all. PyCaret classiification module [**tutorial**](https://www.pycaret.org/tutorials/html/CLF101.html) for more info.

Code sample : Classification with PyCaret

3.3 Time Series Module in PyCaret
---------------------------------

As per the Pycaret officiial documentation, PyCaret’s new time series module is now available in beta. As like existing API it do have statistical testing, model training and selection (30+ algorithms), model analysis, automated hyperparameter tuning, experiment logging, deployment on cloud, and more.

To use the time series beta package you must install it in a separate conda environment in order to avoid dependency conflicts.

pip install pycaret-ts-alpha

Get the official code sample from the below link for forecast the number of airline passenger from the PyCaret team.

3.4 Clustering Module in PyCaret
--------------------------------

PyCaret’s clustering module [**pycaret.clustering**](https://pycaret.readthedocs.io/en/latest/api/clustering.html) is a an unsupervised machine learning module which performs the task of grouping a set of objects in such a way that those in the same group (called a cluster) are more similar to each other than to those in other groups.

PyCaret’s clustering module provides several pre-processing features that can be configured when initializing the setup through the `setup()` function. It has over 8 algorithms and several plots to analyze the results. PyCaret's clustering module also implements a unique function called `tune_model()` that allows you to tune the hyperparameters of a clustering model to optimize a supervised learning objective such as `AUC` for classification or `R2` for regression. PyCaret clustering module [**tutorial**](http://www.pycaret.org/tutorials/html/CLU101.html) for more info.

Code Sample : Clustering with PyCaret

3.5 Anomaly Detection module in PyCaret
---------------------------------------

PyCaret’s anomaly detection module [**pycaret.anomaly**](http://www.pycaret.org/tutorials/html/ANO101.html) is a an unsupervised machine learning module which performs the task of identifying rare items, events or observations which raise suspicions by differing significantly from the majority of the data.

PyCaret anomaly detection module provides several pre-processing features that can be configured when initializing the setup through `setup()` function. It has over 12 algorithms and few plots to analyze the results of anomaly detection. PyCaret's anomaly detection module also implements a unique function `tune_model()` that allows you to tune the hyperparameters of anomaly detection model to optimize the supervised learning objective such as `AUC` for classification or `R2` for regression.

Code Sample : Anomaly Detection with PyCaret

_PyCaret_ [**GitHub repo**](https://github.com/pycaret/pycaret) and [**documentation**](https://pycaret.gitbook.io/docs/) for more info.

From this article, I hope you discovered that PyCaret is a Python version of the popular and widely used caret machine learning package in R. PyCaret use to easily evaluate and compare standard machine learning models on a dataset. PyCaret use to easily tune the hyperparameters of a well-performing machine learning model.It does have setup() to setup the environment, create\_model() api to create the model, tune\_model() to tune the model, evaluate-medel() to analyze the best model, predict\_model() api to predict on new data and save\_model to save the best pipeline.

Try using it on different types of datasets you’ll truly grasp it’s utility the more you leverage it! It even supports model deployment on cloud services like AWS and that too with just one line of code.

> Thank You for Reading

**Do you have any questions?**Ask your questions in comment section and I will do my best to answer.


[Source](https://medium.com/@abonia/pycaret-for-machine-learning-3343fda42ad6)