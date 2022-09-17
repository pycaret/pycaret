
# Easy MLOps with PyCaret + MLflow

# A beginner-friendly, step-by-step tutorial on integrating MLOps in your Machine Learning experiments using PyCaret

![Photo by [Adi Goldstein](https://unsplash.com/@adigold1?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/7832/0*WSGn8A3YB42ALgSg)

# PyCaret

PyCaret is an open-source, low-code machine learning library and end-to-end model management tool built-in Python for automating machine learning workflows. It is known for its ease of use, simplicity, and ability to quickly and efficiently build and deploy end-to-end ML prototypes.

PyCaret is an alternate low-code library that can replace hundreds of code lines with few lines only. This makes the experiment cycle exponentially fast and efficient.

![PyCaret — An open-source, low-code machine learning library in Python](https://cdn-images-1.medium.com/max/2066/0*IOqb01w3mfxSYsRi.png)

To learn more about PyCaret, you can check out their [GitHub](https://www.github.com/pycaret/pycaret).

# MLflow

MLflow is an open-source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry. MLflow currently offers four components:

![MLflow is an open-source platform to manage ML lifecycle](https://cdn-images-1.medium.com/max/2852/1*EQ48xHBYlnqBKoas54URpQ.png)

To learn more about MLflow, you can check out [GitHub](https://github.com/mlflow/mlflow).

# Installing PyCaret

Installing PyCaret is very easy and takes only a few minutes. We strongly recommend using a virtual environment to avoid potential conflicts with other libraries.

PyCaret’s default installation is a slim version of pycaret that only installs hard dependencies [listed here](https://github.com/pycaret/pycaret/blob/master/requirements.txt).

    **# install slim version (default)
    **pip install pycaret

    **# install the full version**
    pip install pycaret[full]

When you install the full version of pycaret, all the optional dependencies as [listed here](https://github.com/pycaret/pycaret/blob/master/requirements-optional.txt) are also installed. MLflow is part of PyCaret’s dependency and hence does not need to be installed separately.

# 👉 Let’s get started

Before I talk about MLOps, let’s talk a little bit about the machine learning lifecycle at a high level:

![Machine Learning Life Cycle — Image by Author (Read from left-to-right)](https://cdn-images-1.medium.com/max/2580/1*qYCj8HXZ0CUD4DROjU9MAg.png)

* **Business Problem — **This is the first step of the machine learning workflow. It may take from few days to a few weeks to complete, depending on the use case and complexity of the problem. It is at this stage, data scientists meet with subject matter experts (SME’s) to gain an understanding of the problem, interview key stakeholders, collect information, and set the overall expectations of the project.

* **Data Sourcing & ETL — **Once the problem understanding is achieved, it then comes to using the information gained during interviews to source the data from the enterprise database.

* **Exploratory Data Analysis (EDA) — **Modeling hasn’t started yet. EDA is where you analyze the raw data. Your goal is to explore the data and assess the quality of the data, missing values, feature distribution, correlation, etc.

* **Data Preparation — **Now it’s time to prepare the data model training. This includes things like dividing data into a train and test set, imputing missing values, one-hot-encoding, target encoding, feature engineering, feature selection, etc.

* **Model Training & Selection — **This is the step everyone is excited about. This involves training a bunch of models, tuning hyperparameters, model ensembling, evaluating performance metrics, model analysis such as AUC, Confusion Matrix, Residuals, etc, and finally selecting one best model to be deployed in production for business use.

* **Deployment & Monitoring — **This is the final step which is mostly about MLOps. This includes things like packaging your final model, creating a docker image, writing the scoring script, and then making it all work together, and finally publish it as an API that can be used to obtain predictions on the new data coming through the pipeline.

The old way of doing all this is pretty cumbersome, long, and requires a lot of technical know-how and I possibly cannot cover it in one tutorial. However, in this tutorial, I will use PyCaret to demonstrate how easy it has become for a data scientist to do all this very efficiently. Before we get to the practical part, let’s talk a little bit more about MLOps.

# 👉 **What is MLOps?**

MLOps is an engineering discipline that aims to combine machine learning development i.e. experimentation (model training, hyperparameter tuning, model ensembling, model selection, etc.), normally performed by Data Scientist with ML engineering and operations in order to standardize and streamline the continuous delivery of machine learning models in production.

If you are an absolute beginner, chances are you have no idea what I am talking about. No worries. Let me give you a simple, non-technical definition:
>  *MLOps are bunch of technical engineering and operational tasks that allows your machine learning model to be used by other users and applications accross the organization. Basically, it’s a way through which your work i.e. *machine learning models *are published online, so other people can use them and satisfy some business objectives.*

This is a very toned-down definition of MLOps. In reality, it involved a little more work and benefits than this but it’s a good start for you if you are new to all this.

Now let’s follow the same workflow as shown in the diagram above to do a practical demo, make sure you have pycaret installed.

# 👉 Business Problem

For this tutorial, I will be using a very popular case study by Darden School of Business, published in [Harvard Business](https://hbsp.harvard.edu/product/UV0869-PDF-ENG). The case is regarding the story of two people who are going to be married in the future. The guy named *Greg *wanted to buy a ring to propose to a girl named *Sarah*. The problem is to find the ring Sarah will like, but after a suggestion from his close friend, Greg decides to buy a diamond stone instead so that Sarah can decide her choice. Greg then collects data of 6000 diamonds with their price and attributes like cut, color, shape, etc.

# 👉 Data

In this tutorial, I will be using a dataset from a very popular case study by the Darden School of Business, published in [Harvard Business](https://hbsp.harvard.edu/product/UV0869-PDF-ENG). The goal of this tutorial is to predict the diamond price based on its attributes like carat weight, cut, color, etc. You can download the dataset from [PyCaret’s repository](https://github.com/pycaret/pycaret/tree/master/datasets).

    **# load the dataset from pycaret
    **from pycaret.datasets import get_data
    data = get_data('diamond')

![Sample rows from data](https://cdn-images-1.medium.com/max/2000/0*rDRvbnmVe7vDGPVM.png)

# 👉 Exploratory Data Analysis

Let’s do some quick visualization to assess the relationship of independent features (weight, cut, color, clarity, etc.) with the target variable i.e. Price

    **# plot scatter carat_weight and Price**
    import plotly.express as px
    fig = px.scatter(x=data['Carat Weight'], y=data['Price'], 
                     facet_col = data['Cut'], opacity = 0.25, template = 'plotly_dark', trendline='ols',
                     trendline_color_override = 'red', title = 'SARAH GETS A DIAMOND - A CASE STUDY')
    fig.show()

![Sarah gets a diamond case study](https://cdn-images-1.medium.com/max/2328/0*Lo6mo1OUPjb-Cenm.png)

Let’s check the distribution of the target variable.

    **# plot histogram**
    fig = px.histogram(data, x=["Price"], template = 'plotly_dark', title = 'Histogram of Price')
    fig.show()

![](https://cdn-images-1.medium.com/max/2316/0*VjW1-hOpd7hVBtIj.png)

Notice that distribution of Price is right-skewed, we can quickly check to see if log transformation can make Price approximately normal to give fighting chance to algorithms that assume normality.

    import numpy as np

    **# create a copy of data**
    data_copy = data.copy()

    **# create a new feature Log_Price**
    data_copy['Log_Price'] = np.log(data['Price'])

    **# plot histogram**
    fig = px.histogram(data_copy, x=["Log_Price"], title = 'Histgram of Log Price', template = 'plotly_dark')
    fig.show()

![](https://cdn-images-1.medium.com/max/2322/0*KfvEU2c6f8LdjUPS.png)

This confirms our hypothesis. The transformation will help us to get away with skewness and make the target variable approximately normal. Based on this, we will transform the Price variable before training our models.

# 👉 Data Preparation

Common to all modules in PyCaret, the setup is the first and the only mandatory step in any machine learning experiment using PyCaret. This function takes care of all the data preparation required prior to training models. Besides performing some basic default processing tasks, PyCaret also offers a wide array of pre-processing features. To learn more about all the preprocessing functionalities in PyCaret, you can see this [link](https://pycaret.org/preprocessing/).

    **# initialize setup**
    from pycaret.regression import *
    s = setup(data, target = 'Price', transform_target = True, log_experiment = True, experiment_name = 'diamond')

![setup function in pycaret.regression module](https://cdn-images-1.medium.com/max/2736/0*dCiKXVXfpXyYQiYd.png)

When you initialize the setup function in PyCaret, it profiles the dataset and infers the data types for all input features. If all data types are correctly inferred, you can press enter to continue.

Notice that:

* I have passed log_experiment = True and experiment_name = 'diamond' , this will tell PyCaret to automatically log all the metrics, hyperparameters, and model artifacts behind the scene as you progress through the modeling phase. This is possible due to integration with [MLflow](https://www.mlflow.org).

* Also, I have used transform_target = True inside the setup. PyCaret will transform the Price variable behind the scene using box-cox transformation. It affects the distribution of data in a similar way as log transformation *(technically different)*. If you would like to learn more about box-cox transformations, you can refer to this [link](https://onlinestatbook.com/2/transformations/box-cox.html).

![Output from setup — truncated for display](https://cdn-images-1.medium.com/max/2000/0*b5w1YKkwK2G9n_YA.png)

# 👉 Model Training & Selection

Now that data is ready for modeling, let’s start the training process by using compare_models function. It will train all the algorithms available in the model library and evaluates multiple performance metrics using k-fold cross-validation.

    **# compare all models**
    best = compare_models()

![Output from compare_models](https://cdn-images-1.medium.com/max/2000/0*FZAGMj-lU-C_kxRl.png)

    **# check the residuals of trained model**
    plot_model(best, plot = 'residuals_interactive')

![Residuals and QQ-Plot of the best model](https://cdn-images-1.medium.com/max/2590/0*yOzbuZjSXY4s2v4Z.png)

    **# check feature importance**
    plot_model(best, plot = 'feature')

![](https://cdn-images-1.medium.com/max/2068/0*m8k8VaglnYOkNx5x.png)

# Finalize and Save Pipeline

Let’s now finalize the best model i.e. train the best model on the entire dataset including the test set and then save the pipeline as a pickle file.

    **# finalize the model**
    final_best = finalize_model(best)

    **# save model to disk
    **save_model(final_best, 'diamond-pipeline')

save_model function will save the entire pipeline (including the model) as a pickle file on your local disk. By default, it will save the file in the same folder as your Notebook or script is in but you can pass the complete path as well if you would like:

    save_model(final_best, 'c:/users/moez/models/diamond-pipeline'

# 👉 Deployment

Remember we passed log_experiment = True in the setup function along with experiment_name = 'diamond' . Let’s see the magic PyCaret has done with the help of MLflow behind the scene. To see the magic let’s initiate the MLflow server:

    **# within notebook (notice ! sign infront)
    **!mlflow ui

    **# on command line in the same folder
    **mlflow ui

Now open your browser and type “localhost:5000”. It will open a UI like this:

![https://localhost:5000](https://cdn-images-1.medium.com/max/3836/1*yZ4zThh0tnY0uW8SsLCpdw.png)

Each entry in the table above represents a training run resulting in a trained Pipeline and a bunch of metadata such as DateTime of a run, performance metrics, model hyperparameters, tags, etc. Let’s click on one of the models:

![Part I — CatBoost Regressor](https://cdn-images-1.medium.com/max/3776/1*TQEApDxCxDIoN6GWwvbBZw.png)

![Part II — CatBoost Regressor (continued)](https://cdn-images-1.medium.com/max/2438/1*RVC18B9Zk8rJkLp28jHkjA.png)

![Part II — CatBoost Regressor (continued)](https://cdn-images-1.medium.com/max/3392/1*1rsOUsPzyY3O0Djao2KlzQ.png)

Notice that you have an address path for the logged_model. This is the trained Pipeline with Catboost Regressor. You can read this Pipeline using the load_model function.

    **# load model**
    from pycaret.regression import load_model
    pipeline = load_model('C:/Users/moezs/mlruns/1/b8c10d259b294b28a3e233a9d2c209c0/artifacts/model/model')

    **# print pipeline
    **print(pipeline)

![Output from print(pipeline)](https://cdn-images-1.medium.com/max/2916/1*E0dUApG1JQnddUfVHmCYvg.png)

Let’s now use this Pipeline to generate predictions on the new data

    **# create a copy of data and drop Price
    **data2 = data.copy()
    data2.drop('Price', axis=1, inplace=True)

    **# generate predictions
    **from pycaret.regression import predict_model
    predictions = predict_model(pipeline, data=data2)
    predictions.head()

![Predictions generated from Pipeline](https://cdn-images-1.medium.com/max/2000/1*IVtFV6oRqcsgyNQTHbb3QA.png)

Woohoo! We now have inference from our trained Pipeline. Congrats, if this is your first one. Notice that all the transformations such as target transformation, one-hot-encoding, missing value imputation, etc. happened behind the scene automatically. You get a data frame with prediction in actual scale, and this is what you care about.

# Coming Soon!

What I have shown today is one out of many ways you can serve trained Pipelines from PyCaret in production with the help of MLflow. In the next tutorial, I plan to show how you can using MLflow native serving functionalities to register your models, version them and serve as an API.

There is no limit to what you can achieve using this lightweight workflow automation library in Python. If you find this useful, please do not forget to give us ⭐️ on our GitHub repository.

To hear more about PyCaret follow us on [LinkedIn](https://www.linkedin.com/company/pycaret/) and [Youtube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g).

Join us on our slack channel. Invite link [here](https://join.slack.com/t/pycaret/shared_invite/zt-p7aaexnl-EqdTfZ9U~mF0CwNcltffHg).

# You may also be interested in:

[Build your own AutoML in Power BI using PyCaret 2.0](https://towardsdatascience.com/build-your-own-automl-in-power-bi-using-pycaret-8291b64181d)
[Deploy Machine Learning Pipeline on Azure using Docker](https://towardsdatascience.com/deploy-machine-learning-pipeline-on-cloud-using-docker-container-bec64458dc01)
[Deploy Machine Learning Pipeline on Google Kubernetes Engine](https://towardsdatascience.com/deploy-machine-learning-model-on-google-kubernetes-engine-94daac85108b)
[Deploy Machine Learning Pipeline on AWS Fargate](https://towardsdatascience.com/deploy-machine-learning-pipeline-on-aws-fargate-eb6e1c50507)
[Build and deploy your first machine learning web app](https://towardsdatascience.com/build-and-deploy-your-first-machine-learning-web-app-e020db344a99)
[Deploy PyCaret and Streamlit app using AWS Fargate serverless](https://towardsdatascience.com/deploy-pycaret-and-streamlit-app-using-aws-fargate-serverless-infrastructure-8b7d7c0584c2)
[Build and deploy machine learning web app using PyCaret and Streamlit](https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104)
[Deploy Machine Learning App built using Streamlit and PyCaret on GKE](https://towardsdatascience.com/deploy-machine-learning-app-built-using-streamlit-and-pycaret-on-google-kubernetes-engine-fd7e393d99cb)

# Important Links

[Documentation](https://pycaret.readthedocs.io/en/latest/installation.html)
[Blog](https://medium.com/@moez_62905)
[GitHub](http://www.github.com/pycaret/pycaret)
[StackOverflow](https://stackoverflow.com/questions/tagged/pycaret)
[Install PyCaret
](https://pycaret.readthedocs.io/en/latest/installation.html)[Notebook Tutorials
](https://pycaret.readthedocs.io/en/latest/tutorials.html)[Contribute in PyCaret](https://pycaret.readthedocs.io/en/latest/contribute.html)

# Want to learn about a specific module?

Click on the links below to see the documentation and working examples.

[Classification
](https://pycaret.readthedocs.io/en/latest/api/classification.html)[Regression](https://pycaret.readthedocs.io/en/latest/api/regression.html)
[Clustering](https://pycaret.readthedocs.io/en/latest/api/clustering.html)
[Anomaly Detection](https://pycaret.readthedocs.io/en/latest/api/anomaly.html)
[Natural Language Processing
](https://pycaret.readthedocs.io/en/latest/api/nlp.html)[Association Rule Mining](https://pycaret.readthedocs.io/en/latest/api/arules.html)
