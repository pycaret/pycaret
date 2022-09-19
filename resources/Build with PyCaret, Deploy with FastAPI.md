
# Build with PyCaret, Deploy with FastAPI

# A step-by-step, beginner-friendly tutorial on how to build an end-to-end Machine Learning Pipeline with PyCaret and deploy it as an API.

![PyCaret — an open-source, low-code machine learning library in Python](https://cdn-images-1.medium.com/max/2332/1*X-TkH_enuSrM71f1zrbHSg.png)

# 👉 Introduction

This is a step-by-step, beginner-friendly tutorial on how to build an end-to-end Machine Learning Pipeline with [PyCaret](https://www.pycaret.org) and deploy it in production as a web API using [FastAPI](https://fastapi.tiangolo.com/).

# Learning Goals of this Tutorial

* Build an end-to-end machine learning pipeline using PyCaret

* What is a deployment and why do we deploy machine learning models

* Develop an API using FastAPI to generate predictions on unseen data

* Use Python to send a request to API for generating predictions programmatically.

This tutorial will cover the entire machine learning life cycle at a high level which is broken down into the following sections:

![PyCaret — Machine Learning High-Level Workflow](https://cdn-images-1.medium.com/max/2580/1*m_dqCv42rZmxqyFzQ7rpUQ.png)

# 💻 What tools we will use in this tutorial?

# 👉 PyCaret

PyCaret is an open-source, low-code machine learning library and end-to-end model management tool built-in Python for automating machine learning workflows. It is incredibly popular for its ease of use, simplicity, and ability to build and deploy end-to-end ML prototypes quickly and efficiently.

PyCaret is an alternate low-code library that can replace hundreds of code lines with few lines only. This makes the experiment cycle exponentially fast and efficient.

PyCaret is **simple and** **easy to use**. All the operations performed in PyCaret are sequentially stored in a **Pipeline** that is fully automated for **deployment. **Whether it’s imputing missing values, one-hot-encoding, transforming categorical data, feature engineering, or even hyperparameter tuning, PyCaret automates all of it.

To learn more about PyCaret, check out their [GitHub](https://www.github.com/pycaret/pycaret).

# 👉 FastAPI

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. The key features are:

* **Fast**: Very high performance, on par with **NodeJS** and **Go** (thanks to Starlette and Pydantic). [One of the fastest Python frameworks available](https://fastapi.tiangolo.com/#performance).

* **Fast to code**: Increase the speed to develop features by about 200% to 300%.

* **Easy**: Designed to be easy to use and learn. Less time reading docs.

To learn more about FastAPI, check out their [GitHub](https://github.com/tiangolo/fastapi).

![The workflow for PyCaret and FastAPI](https://cdn-images-1.medium.com/max/2318/1*cTds_qxWkAmBCovv0bmnIg.png)

# 👉 Installing PyCaret

Installing PyCaret is very easy and takes only a few minutes. We strongly recommend using a virtual environment to avoid potential conflicts with other libraries.

PyCaret’s default installation is a slim version of pycaret that only installs hard dependencies [listed here](https://github.com/pycaret/pycaret/blob/master/requirements.txt).

    **# install slim version (default)
    **pip install pycaret

    **# install the full version**
    pip install pycaret[full]

When you install the full version of pycaret, all the optional dependencies as [listed here](https://github.com/pycaret/pycaret/blob/master/requirements-optional.txt) are also installed.

# 👉 Installing FastAPI

You can install FastAPI from pip.

    pip install fastapi

# 👉 Business Problem

For this tutorial, I will be using a very popular case study by Darden School of Business, published in [Harvard Business](https://hbsp.harvard.edu/product/UV0869-PDF-ENG). The case is regarding the story of two people who are going to be married in the future. The guy named *Greg *wanted to buy a ring to propose to a girl named *Sarah*. The problem is to find the ring Sarah will like, but after a suggestion from his close friend, Greg decides to buy a diamond stone instead so that Sarah can decide her choice. Greg then collects data of 6000 diamonds with their price and attributes like cut, color, shape, etc.

# 👉 Data

In this tutorial, I will be using a dataset from a very popular case study by the Darden School of Business, published in [Harvard Business](https://hbsp.harvard.edu/product/UV0869-PDF-ENG). The goal of this tutorial is to predict the diamond price based on its attributes like carat weight, cut, color, etc. You can download the dataset from [PyCaret’s repository](https://github.com/pycaret/pycaret/tree/master/datasets).

    **# load the dataset from pycaret
    **from pycaret.datasets import get_data
    data = get_data('diamond')

![Sample rows from data](https://cdn-images-1.medium.com/max/2000/1*e-rJPY4j9GYciT-L-lZnWw.png)

# 👉 Exploratory Data Analysis

Let’s do some quick visualization to assess the relationship of independent features (weight, cut, color, clarity, etc.) with the target variable i.e. Price

    **# plot scatter carat_weight and Price**
    import plotly.express as px
    fig = px.scatter(x=data['Carat Weight'], y=data['Price'], 
                     facet_col = data['Cut'], opacity = 0.25, template = 'plotly_dark', trendline='ols',
                     trendline_color_override = 'red', title = 'SARAH GETS A DIAMOND - A CASE STUDY')
    fig.show()

![Sarah gets a diamond case study](https://cdn-images-1.medium.com/max/2328/1*8aLmTOhOB68KYLO05I6A3A.png)

Let’s check the distribution of the target variable.

    **# plot histogram**
    fig = px.histogram(data, x=["Price"], template = 'plotly_dark', title = 'Histogram of Price')
    fig.show()

![](https://cdn-images-1.medium.com/max/2316/1*wp_UPlON60zl43zZ7MwqPg.png)

Notice that distribution of Price is right-skewed, we can quickly check to see if log transformation can make Price approximately normal to give fighting chance to algorithms that assume normality.

    import numpy as np

    **# create a copy of data**
    data_copy = data.copy()

    **# create a new feature Log_Price**
    data_copy['Log_Price'] = np.log(data['Price'])

    **# plot histogram**
    fig = px.histogram(data_copy, x=["Log_Price"], title = 'Histgram of Log Price', template = 'plotly_dark')
    fig.show()

![](https://cdn-images-1.medium.com/max/2322/1*O6pYtQoZ7Pf07Xw-og0vFA.png)

This confirms our hypothesis. The transformation will help us to get away with skewness and make the target variable approximately normal. Based on this, we will transform the Price variable before training our models.

# 👉 Data Preparation

Common to all modules in PyCaret, the setup is the first and the only mandatory step in any machine learning experiment performed in PyCaret. This function takes care of all the data preparation required prior to training models. Besides performing some basic default processing tasks, PyCaret also offers a wide array of pre-processing features. To learn more about all the preprocessing functionalities in PyCaret, you can see this [link](https://pycaret.org/preprocessing/).

    **# init setup**
    from pycaret.regression import *
    s = setup(data, target = 'Price', transform_target = True)

![setup function in pycaret.regression module](https://cdn-images-1.medium.com/max/2418/1*Jw47qMeoH7ixupKG6xDNHA.png)

Whenever you initialize the setup function in PyCaret, it profiles the dataset and infers the data types for all input features. In this case, you can see except for Carat Weight all the other features are inferred as categorical, which is correct. You can press enter to continue.

Notice that I have used transform_target = True inside the setup. PyCaret will transform the Price variable behind the scene using box-cox transformation. It affects the distribution of data in a similar way as log transformation *(technically different)*. If you would like to learn more about box-cox transformations, you can refer to this [link](https://onlinestatbook.com/2/transformations/box-cox.html).

![Output from setup — truncated for display](https://cdn-images-1.medium.com/max/2000/1*Xm7xnZr1UjSTM_dTFwbv6g.png)

# 👉 Model Training & Selection

Now that data preparation is done, let’s start the training process by using compare_models functionality. This function trains all the algorithms available in the model library and evaluates multiple performance metrics using cross-validation.

    **# compare all models**
    best = compare_models()

![Output from compare_models](https://cdn-images-1.medium.com/max/2000/1*poaUGuUkOX-2kO3b5-Z_4g.png)

The best model based on *Mean Absolute Error (MAE)* is CatBoost Regressor. MAE using 10-fold cross-validation is $543 compared to the average diamond value of $11,600. This is less than 5%. Not bad for the efforts we have put in so far.

    **# check the residuals of trained model**
    plot_model(best, plot = 'residuals_interactive')

![Residuals and QQ-Plot of the best model](https://cdn-images-1.medium.com/max/2590/1*9XL2AJvT7ZoejJNgxmlU-g.png)

    **# check feature importance**
    plot_model(best, plot = 'feature')

![Feature Importance of best model](https://cdn-images-1.medium.com/max/2068/1*azpWQD8M5dPycAzWBGtNaQ.png)

# Finalize and Save Pipeline

Let’s now finalize the best model i.e. train the best model on the entire dataset including the test set and then save the pipeline as a pickle file.

    **# finalize the model**
    final_best = finalize_model(best)

    **# save model to disk
    **save_model(final_best, 'diamond-pipeline')

# 👉 Deployment
>  ***First, let’s understand why to deploy machine learning models?***

*The deployment of machine learning models is the process of making models available in production where web applications, enterprise software, and APIs can consume the trained model by providing new data points and generating predictions. Normally machine learning models are built so that they can be used to predict an outcome (binary value i.e. 1 or 0 for Classification, continuous values for Regression, labels for Clustering, etc. There are two broad ways of generating predictions (i) predict by batch; and (ii) predict in real-time. This tutorial will show how you can deploy your machine learning models as API to predict in real-time.*

Now that we understand why deployment is necessary and we have everything we need to create an API i.e. *Trained Model Pipeline as a pickle file*. Creating an API is extremely simple using FastAPI.

 <iframe src="https://medium.com/media/8fc8a66de48dafe353315369c286f2e4" frameborder=0></iframe>

The first few lines of the code are simple imports. Line 8 is initializing an app by calling FastAPI() . Line 11 is loading the trained model diamond-pipeline from your disk (Your script must be in the same folder as the file). Line 15–20 is defining a function called predict which will take the input and internally uses PyCaret’s predict_model function to generate predictions and return the value as a dictionary (Line 20).

You can then run this script by running the following command in your command prompt. You must be in the same directory as the python script and the model pickle file is, before executing this command.

    uvicorn main:app --reload

This will initialize an API service on your localhost. On your browser type [http://localhost:8000/docs](http://localhost:8000/docs) and it should show something like this:

![http://localhost:8000/docs](https://cdn-images-1.medium.com/max/3734/1*RWJbHRXV3YrMI5I4NdMBtA.png)

Click on green **POST **button and it will be open a form like this:

![[http://localhost:8000/docs](http://localhost:8000/docs)](https://cdn-images-1.medium.com/max/3682/1*NZlbBe8gAkq1CmqC_4_hDg.png)

Click on **“Try it out”** on the top right corner and fill in some values in the form and click on “Execute”. If you have done everything correctly, you will see this response:

![Response from FastAPI](https://cdn-images-1.medium.com/max/3660/1*1j3Pfu3G-63ic09mIkrwvA.png)

Notice that under the response body we have a prediction value of 5396 (this is based on values I entered in the form). This means that given all the attributes you entered, the predicted price of this diamond is $5,396.

This is great, this shows that our API is working. Now we can use the requests library in Python or any other language to connect to API and generate predictions. I have created the script shown below for that:

 <iframe src="https://medium.com/media/c526dbf645dbc59a175dc7d4cfc32788" frameborder=0></iframe>

Let’s see this function in action:

![get_prediction function created to make an API call](https://cdn-images-1.medium.com/max/2724/1*G83rpAEKbTeOEYXtl6_7qQ.png)

Notice that prediction is 5396 that's because I have used the same values here as I have used in the form above. (1.1, ‘Ideal’, ‘H’, ‘SII’, ‘VG’, ‘EX’, ‘GIA’)

I hope that you will appreciate the ease of use and simplicity in PyCaret and FastAPI. In less than 25 lines of code and few minutes of experimentation, I have trained and evaluated multiple models using PyCaret and deployed ML Pipeline using an API.

# Coming Soon!

Next week I will be writing a tutorial to advance deployment to the next level, I will introduce the concepts like Containerization and Dockers in my next tutorial. Please follow me on [Medium](https://medium.com/@moez-62905), [LinkedIn](https://www.linkedin.com/in/profile-moez/), and [Twitter](https://twitter.com/moezpycaretorg1) to get more updates.

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
