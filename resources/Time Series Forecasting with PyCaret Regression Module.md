
# Time Series Forecasting with PyCaret Regression Module

![Photo by [Lukas Blazek](https://unsplash.com/@goumbik?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/12288/0*6t7FzC-AdfDlA9LI)

# PyCaret

PyCaret is an open-source, low-code machine learning library and end-to-end model management tool built-in Python for automating machine learning workflows. It is incredibly popular for its ease of use, simplicity, and ability to build and deploy end-to-end ML prototypes quickly and efficiently.

PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few lines only. This makes the experiment cycle exponentially fast and efficient.

PyCaret is **simple and** **easy to use**. All the operations performed in PyCaret are sequentially stored in a **Pipeline** that is fully automated for **deployment. **Whether it's imputing missing values, one-hot-encoding, transforming categorical data, feature engineering, or even hyperparameter tuning, PyCaret automates all of it. To learn more about PyCaret, watch this 1-minute video.

 <iframe src="https://medium.com/media/d7cfbf3d4b3026ce42674e2a1e26af42" frameborder=0></iframe>

This tutorial assumes that you have some prior knowledge and experience with PyCaret. If you haven’t used it before, no problem — you can get a quick headstart through these tutorials:

* [PyCaret 2.2 is here — what’s new](https://towardsdatascience.com/pycaret-2-2-is-here-whats-new-ad7612ca63b)

* [Announcing PyCaret 2.0](https://towardsdatascience.com/announcing-pycaret-2-0-39c11014540e)

* [Five things you don’t know about PyCaret](https://towardsdatascience.com/5-things-you-dont-know-about-pycaret-528db0436eec)

# Installing PyCaret

Installing PyCaret is very easy and takes only a few minutes. We strongly recommend using a virtual environment to avoid potential conflicts with other libraries.

PyCaret’s default installation is a slim version of pycaret which only installs hard dependencies that are [listed here](https://github.com/pycaret/pycaret/blob/master/requirements.txt).

    **# install slim version (default)
    **pip install pycaret

    **# install the full version**
    pip install pycaret[full]

When you install the full version of pycaret, all the optional dependencies as [listed here](https://github.com/pycaret/pycaret/blob/master/requirements-optional.txt) are also installed.

# 👉 PyCaret Regression Module

PyCaret **Regression Module** is a supervised machine learning module used for estimating the relationships between a **dependent variable** (often called the ‘outcome variable’, or ‘target’) and one or more **independent variables** (often called ‘features’, or ‘predictors’).

The objective of regression is to predict continuous values such as sales amount, quantity, temperature, number of customers, etc. All modules in PyCaret provide many [pre-processing](https://www.pycaret.org/preprocessing) features to prepare the data for modeling through the [setup ](https://www.pycaret.org/setup)function. It has over 25 ready-to-use algorithms and [several plots](https://www.pycaret.org/plot-model) to analyze the performance of trained models.

# 👉 Time Series with PyCaret Regression Module

Time series forecasting can broadly be categorized into the following categories:

* **Classical / Statistical Models** — Moving Averages, Exponential smoothing, ARIMA, SARIMA, TBATS

* **Machine Learning **— Linear Regression, XGBoost, Random Forest, or any ML model with reduction methods

* **Deep Learning **— RNN, LSTM

This tutorial is focused on the second category i.e. *Machine Learning*.

PyCaret’s Regression module default settings are not ideal for time series data because it involves few data preparatory steps that are not valid for ordered data (*data with a sequence such as time series data*).

For example, the split of the dataset into train and test set is done randomly with shuffling. This wouldn’t make sense for time series data as you don’t want the recent dates to be included in the training set whereas historical dates are part of the test set.

Time-series data also requires a different kind of cross-validation since it needs to respect the order of dates. PyCaret regression module by default uses k-fold random cross-validation when evaluating models. The default cross-validation setting is not suitable for time-series data.

The following section in this tutorial will demonstrate how you can change default settings in PyCaret Regression Module easily to make it work for time series data.

# 👉 Dataset

For the purpose of this tutorial, I have used the US airline passengers dataset. You can download the dataset from [Kaggle](https://www.kaggle.com/chirag19/air-passengers).

    **# read csv file
    **import pandas as pd
    data = pd.read_csv('AirPassengers.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.head()

![Sample rows](https://cdn-images-1.medium.com/max/2000/1*9Gn9v-CWD3Eca3V92edtDw.png)

    **# create 12 month moving average
    **data['MA12'] = data['Passengers'].rolling(12).mean()

    **# plot the data and MA
    **import plotly.express as px
    fig = px.line(data, x="Date", y=["Passengers", "MA12"], template = 'plotly_dark')
    fig.show()

![US Airline Passenger Dataset Time Series Plot with Moving Average = 12](https://cdn-images-1.medium.com/max/2618/1*vXL7m7exxTid0kGcU9jGUA.png)

Since algorithms cannot directly deal with dates, let’s extract some simple features from dates such as month and year, and drop the original date column.

    **# extract month and year from dates**
    data['Month'] = [i.month for i in data['Date']]
    data['Year'] = [i.year for i in data['Date']]

    **# create a sequence of numbers
    **data['Series'] = np.arange(1,len(data)+1)

    **# drop unnecessary columns and re-arrange
    **data.drop(['Date', 'MA12'], axis=1, inplace=True)
    data = data[['Series', 'Year', 'Month', 'Passengers']] 

    **# check the head of the dataset**
    data.head()

![Sample rows after extracting features](https://cdn-images-1.medium.com/max/2000/1*T4xXVYhrfIyB35EDjl645A.png)

    **# split data into train-test set
    **train = data[data['Year'] < 1960]
    test = data[data['Year'] >= 1960]

    **# check shape
    **train.shape, test.shape
    >>> ((132, 4), (12, 4))

I have manually split the dataset before initializing the setup . An alternate would be to pass the entire dataset to PyCaret and let it handle the split, in which case you will have to pass data_split_shuffle = False in the setup function to avoid shuffling the dataset before the split.

# 👉 **Initialize Setup**

Now it’s time to initialize the setup function, where we will explicitly pass the training data, test data, and cross-validation strategy using the fold_strategy parameter.

    **# import the regression module**
    from pycaret.regression import *

    **# initialize setup**
    s = setup(data = train, test_data = test, target = 'Passengers', fold_strategy = 'timeseries', numeric_features = ['Year', 'Series'], fold = 3, transform_target = True, session_id = 123)

# 👉 **Train and Evaluate all Models**

    best = compare_models(sort = 'MAE')

![Results from compare_models](https://cdn-images-1.medium.com/max/2000/1*Mnqplw1KbJYm9fxyH-46Tg.png)

The best model based on cross-validated MAE is **Least Angle Regression **(MAE: 22.3). Let’s check the score on the test set.

    prediction_holdout = predict_model(best);

![Results from predict_model(best) function](https://cdn-images-1.medium.com/max/2000/1*Us888u-jaVQzasN8Kn3z6A.png)

MAE on the test set is 12% higher than the cross-validated MAE. Not so good, but we will work with it. Let’s plot the actual and predicted lines to visualize the fit.

    **# generate predictions on the original dataset**
    predictions = predict_model(best, data=data)

    **# add a date column in the dataset**
    predictions['Date'] = pd.date_range(start='1949-01-01', end = '1960-12-01', freq = 'MS')

    **# line plot**
    fig = px.line(predictions, x='Date', y=["Passengers", "Label"], template = 'plotly_dark')

    **# add a vertical rectange for test-set separation**
    fig.add_vrect(x0="1960-01-01", x1="1960-12-01", fillcolor="grey", opacity=0.25, line_width=0)

    fig.show()

![Actual and Predicted US airline passengers (1949–1960)](https://cdn-images-1.medium.com/max/2624/1*BlfRbXuxwcgvs-zrpK8C0Q.png)

The grey backdrop towards the end is the test period (i.e. 1960). Now let’s finalize the model i.e. train the best model i.e. *Least Angle Regression* on the entire dataset (this time, including the test set).

    final_best = finalize_model(best)

# 👉 Create a future scoring dataset

Now that we have trained our model on the entire dataset (1949 to 1960), let’s predict five years out in the future through 1964. To use our final model to generate future predictions, we first need to create a dataset consisting of the Month, Year, Series column on the future dates.

    future_dates = pd.date_range(start = '1961-01-01', end = '1965-01-01', freq = 'MS')

    future_df = pd.DataFrame()

    future_df['Month'] = [i.month for i in future_dates]
    future_df['Year'] = [i.year for i in future_dates]    
    future_df['Series'] = np.arange(145,(145+len(future_dates)))

    future_df.head()

![Sample rows from future_df](https://cdn-images-1.medium.com/max/2000/1*KD4G6VVmbuq-w_6088Vl8Q.png)

Now, let’s use the future_df to score and generate predictions.

    predictions_future = predict_model(final_best, data=future_df)
    predictions_future.head()

![Sample rows from predictions_future](https://cdn-images-1.medium.com/max/2000/1*5mqx2Qi2En2VPq9zNZG87g.png)

Let’s plot it.

    concat_df = pd.concat([data,predictions_future], axis=0)
    concat_df_i = pd.date_range(start='1949-01-01', end = '1965-01-01', freq = 'MS')
    concat_df.set_index(concat_df_i, inplace=True)

    fig = px.line(concat_df, x=concat_df.index, y=["Passengers", "Label"], template = 'plotly_dark')
    fig.show()

![Actual (1949–1960) and Predicted (1961–1964) US airline passengers](https://cdn-images-1.medium.com/max/2614/1*IjglwJEeU2hZMsjxM8yPbg.png)

Wasn’t that easy?

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
