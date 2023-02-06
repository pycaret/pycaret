
# Time Series 101 — For beginners

# A beginner-friendly introduction to Time Series Forecasting

![Photo by [Chris Liverani](https://unsplash.com/@chrisliverani?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/7262/0*AfqHPFyS5tc-9Amn)

# 👉 What is Time Series Data?

Time series data is data collected on the same subject at different points in time, such as **GDP of a country by year, a stock price of a particular company over a period of time, or your own heartbeat recorded at each second**, as a matter of fact, anything that you can capture continuously at different time-intervals is a time series data.

See below as an example of time series data, the chart below is the daily stock price of Tesla Inc. (Ticker Symbol: TSLA) for last year. The y-axis on the right-hand side is the value in US$ (The last point on the chart i.e. $701.91 is the latest stock price as of the writing of this article on April 12, 2021).

![Example of Time Series Data — Tesla Inc. (ticker symbol: TSLA) daily stock price 1Y interval.](https://cdn-images-1.medium.com/max/2874/1*5lgEl2_2wx3b7YEiROYPtA.png)

On the other hand, more conventional datasets such as customer information, product information, company information, etc. which store information at a single point in time are known as cross-sectional data.

See the example below of a dataset that tracks America’s best-selling electric cars in the first half of 2020. Notice that instead of tracking the cars sold over a period of time, the chart below tracks different cars such as Tesla, Chevy, and Nissan in the same time period.

![Source: [Forbes](https://www.forbes.com/sites/niallmccarthy/2020/08/13/americas-best-selling-electric-cars-in-the-first-half-of-2020-infographic/?sh=4d9c34856033)](https://cdn-images-1.medium.com/max/2000/1*pzKQuUCGJldCQcORZlLvXQ.jpeg)

It is not very hard to distinguish the difference between cross-sectional and time-series data as the objective of analysis for both datasets are widely different. For the first analysis, we were interested in tracking Tesla stock price over a period of time, whereas for the latter, we wanted to analyze different companies in the same time period i.e. first half of 2020.

However, a typical real-world dataset is likely to be a hybrid. Imagine a retailer like Walmart that sold thousand’s of products every day. If you analyze the sale by-product on a particular day, for example, if you want to find out what’s the number 1 selling item on Christmas eve, this will be a cross-sectional analysis. As opposed to, If you want to find out the sale of one particular item such as PS4 over a period of time (let’s say last 5 years), this now becomes a time-series analysis.

Precisely, the objective of the analysis for time-series and cross-sectional data is different and a real-world dataset is likely to be a hybrid of both time-series as well as cross-sectional data.

# 👉 What is Time Series Forecasting?

Time series forecasting is exactly what it sounds like i.e. predicting the future unknown values. However, unlike sci-fi movies, it’s a little less thrilling in the real world. It involves the collection of historical data, preparing it for algorithms to consume (the algorithm is simply put the maths that goes behind the scene), and then predict the future values based on patterns learned from the historical data.

Can you think of a reason why would companies or anybody be interested in forecasting future values for any time series (GDP, monthly sales, inventory, unemployment, global temperatures, etc.). Let me give you some business perspective:

* A retailer may be interested in predicting future sales at an SKU level for planning and budgeting.

* A small merchant may be interested in forecasting sales by store, so it can schedule the right resources (more people during busy periods and vice versa).

* A software giant like Google may be interested in knowing the busiest hour of the day or busiest day of the week so that it can schedule server resources accordingly.

* The health department may be interested in predicting the cumulative COVID vaccination administered so that it can know the point of consolidation where herd immunity is expected to kick in.

# 👉 Time Series Forecasting Methods

Time series forecasting can broadly be categorized into the following categories:

* **Classical / Statistical Models** — Moving Averages, Exponential smoothing, ARIMA, SARIMA, TBATS

* **Machine Learning **— Linear Regression, XGBoost, Random Forest, or any ML model with reduction methods

* **Deep Learning **— RNN, LSTM

This tutorial is focused on forecasting time series using ***Machine Learning***. For this tutorial, I will use the Regression Module of an open-source, low-code machine library in Python called [PyCaret](https://www.pycaret.org). If you haven’t used PyCaret before, you can get quickly started [here](https://www.pycaret.org/guide). Although, you don’t require any prior knowledge of PyCaret to follow along with this tutorial.

# 👉 PyCaret Regression Module

PyCaret **Regression Module** is a supervised machine learning module used for estimating the relationships between a **dependent variable** (often called the ‘outcome variable’, or ‘target’) and one or more **independent variables** (often called ‘features’, or ‘predictors’).

The objective of regression is to predict continuous values such as sales amount, quantity, temperature, number of customers, etc. All modules in PyCaret provide many [pre-processing](https://www.pycaret.org/preprocessing) features to prepare the data for modeling through the [setup ](https://www.pycaret.org/setup)function. It has over 25 ready-to-use algorithms and [several plots](https://www.pycaret.org/plot-model) to analyze the performance of trained models.

# 👉 Dataset

For this tutorial, I have used the US airline passengers dataset. You can download the dataset from [Kaggle](https://www.kaggle.com/chirag19/air-passengers). This dataset provides monthly totals of US airline passengers from 1949 to 1960.

    **# read csv file
    **import pandas as pd
    data = pd.read_csv('AirPassengers.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.head()

![Sample rows](https://cdn-images-1.medium.com/max/2000/0*1-RkItCLsIyv3HpN.png)

    **# create 12 month moving average
    **data['MA12'] = data['Passengers'].rolling(12).mean()

    **# plot the data and MA
    **import plotly.express as px
    fig = px.line(data, x="Date", y=["Passengers", "MA12"], template = 'plotly_dark')
    fig.show()

![US Airline Passenger Dataset Time Series Plot with Moving Average = 12](https://cdn-images-1.medium.com/max/2618/0*iwkq5c6sZM63PU9u.png)

Since machine learning algorithms cannot directly deal with dates, let’s extract some simple features from dates such as month and year, and drop the original date column.

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

![Sample rows after extracting features](https://cdn-images-1.medium.com/max/2000/0*h8TaSF1bi9-Y9Psa.png)

    **# split data into train-test set
    **train = data[data['Year'] < 1960]
    test = data[data['Year'] >= 1960]

    **# check shape
    **train.shape, test.shape
    >>> ((132, 4), (12, 4))

I have manually split the dataset before initializing the setup . An alternate would be to pass the entire dataset to PyCaret and let it handle the split, in which case you will have to pass data_split_shuffle = False in the setup function to avoid shuffling the dataset before the split.

# 👉 Initialize Setup

Now it’s time to initialize the setup function, where we will explicitly pass the training data, test data, and cross-validation strategy using the fold_strategy parameter.

    **# import the regression module**
    from pycaret.regression import *

    **# initialize setup**
    s = setup(data = train, test_data = test, target = 'Passengers', fold_strategy = 'timeseries', numeric_features = ['Year', 'Series'], fold = 3, transform_target = True, session_id = 123)

# 👉 Train and Evaluate all Models

    best = compare_models(sort = 'MAE')

![Results from compare_models](https://cdn-images-1.medium.com/max/2000/1*Z7VBrEv1Sh5z532cNy6_qQ.png)

The best model based on cross-validated MAE is **Least Angle Regression **(MAE: 22.3). Let’s check the score on the test set.

    prediction_holdout = predict_model(best);

![Results from predict_model(best) function](https://cdn-images-1.medium.com/max/2000/0*O0gKIfX126Z0Ni-B.png)

MAE on the test set is 12% higher than the cross-validated MAE. Not so good, but we will work with it. Let’s plot the actual and predicted lines to visualize the fit.

    **# generate predictions on the original dataset**
    predictions = predict_model(best, data=data)

    **# add a date column in the dataset**
    predictions['Date'] = pd.date_range(start='1949-01-01', end = '1960-12-01', freq = 'MS')

    **# line plot**
    fig = px.line(predictions, x='Date', y=["Passengers", "prediction_label"], template = 'plotly_dark')

    **# add a vertical rectange for test-set separation**
    fig.add_vrect(x0="1960-01-01", x1="1960-12-01", fillcolor="grey", opacity=0.25, line_width=0)fig.show()

![Actual and Predicted US airline passengers (1949–1960)](https://cdn-images-1.medium.com/max/2624/0*_1esCD8Bx6MZ14Mj.png)

The grey backdrop towards the end is the test period (i.e. 1960). Now let’s finalize the model i.e. train the best model i.e. *Least Angle Regression* on the entire dataset (this time, including the test set).

    final_best = finalize_model(best)

# 👉 Create a future scoring dataset

Now that we have trained our model on the entire dataset (1949 to 1960), let’s predict five years out in the future through 1964. To use our final model to generate future predictions, we first need to create a dataset consisting of the Month, Year, Series column on the future dates.

    future_dates = pd.date_range(start = '1961-01-01', end = '1965-01-01', freq = 'MS')

    future_df = pd.DataFrame()

    future_df['Month'] = [i.month for i in future_dates]
    future_df['Year'] = [i.year for i in future_dates]    
    future_df['Series'] = np.arange(145 (145+len(future_dates)))future_df.head()

![Sample rows from future_df](https://cdn-images-1.medium.com/max/2000/0*N26BSbKSR9u3k7hv.png)

Now, let’s use the future_df to score and generate predictions.

    predictions_future = predict_model(final_best, data=future_df)
    predictions_future.head()

![Sample rows from predictions_future](https://cdn-images-1.medium.com/max/2000/0*c97sliOBqExx6Hs_.png)

# **👉 Plot the actual data and predictions**

    concat_df = pd.concat([data,predictions_future], axis=0)
    concat_df_i = pd.date_range(start='1949-01-01', end = '1965-01-01', freq = 'MS')
    concat_df.set_index(concat_df_i, inplace=True)fig = 

    px.line(concat_df, x=concat_df.index, y=["Passengers", "prediction_label"], template = 'plotly_dark')
    fig.show()

![Actual (1949–1960) and Predicted (1961–1964) US airline passengers](https://cdn-images-1.medium.com/max/2614/0*x8TJJUO--Unxheeg.png)

I hope you find this tutorial easy. If you think you are ready for the next level, you can check out my advanced time-series tutorial on [Multiple Time Series Forecasting with PyCaret](https://towardsdatascience.com/multiple-time-series-forecasting-with-pycaret-bc0a779a22fe).

# Coming Soon!

I will soon be writing a tutorial on unsupervised anomaly detection on time-series data using [PyCaret Anomaly Detection Module](https://pycaret.readthedocs.io/en/latest/api/anomaly.html). If you would like to get more updates, you can follow me on [Medium](https://medium.com/@moez-62905), [LinkedIn](https://www.linkedin.com/in/profile-moez/), and [Twitter](https://twitter.com/moezpycaretorg1).

There is no limit to what you can achieve using this lightweight workflow automation library in Python. If you find this useful, please do not forget to give ⭐️ on our GitHub repository.

To learn more about PyCaret follow us on [LinkedIn](https://www.linkedin.com/company/pycaret/) and [Youtube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g).

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
