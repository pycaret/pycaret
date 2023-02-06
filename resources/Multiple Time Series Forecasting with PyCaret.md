
# Multiple Time Series Forecasting with PyCaret

# A step-by-step tutorial to forecast multiple time series with PyCaret

![PyCaret — An open-source, low-code machine learning library in Python](https://cdn-images-1.medium.com/max/2000/1*c8mBuCW7nP0KGhwXQC98Eg.png)

# PyCaret

PyCaret is an open-source, low-code machine learning library and end-to-end model management tool built-in Python for automating machine learning workflows. It is incredibly popular for its ease of use, simplicity, and ability to build and deploy end-to-end ML prototypes quickly and efficiently.

PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few lines only. This makes the experiment cycle exponentially fast and efficient.

PyCaret is **simple and** **easy to use**. All the operations performed in PyCaret are sequentially stored in a **Pipeline** that is fully automated for **deployment. **Whether it’s imputing missing values, one-hot-encoding, transforming categorical data, feature engineering, or even hyperparameter tuning, PyCaret automates all of it.

This tutorial assumes that you have some prior knowledge and experience with PyCaret. If you haven’t used it before, no problem — you can get a quick headstart through these tutorials:

* [PyCaret 2.2 is here — what’s new](https://towardsdatascience.com/pycaret-2-2-is-here-whats-new-ad7612ca63b)

* [Announcing PyCaret 2.0](https://towardsdatascience.com/announcing-pycaret-2-0-39c11014540e)

* [Five things you don’t know about PyCaret](https://towardsdatascience.com/5-things-you-dont-know-about-pycaret-528db0436eec)

# **RECAP**

In my [last tutorial](https://towardsdatascience.com/time-series-forecasting-with-pycaret-regression-module-237b703a0c63), I have demonstrated how you can use PyCaret to forecast time-series data using Machine Learning through [PyCaret Regression Module](https://pycaret.readthedocs.io/en/latest/api/regression.html). If you haven’t read that yet, you can read [Time Series Forecasting with PyCaret Regression Module](https://towardsdatascience.com/time-series-forecasting-with-pycaret-regression-module-237b703a0c63) tutorial before continuing with this one, as this tutorial builds upon some important concepts covered in the last tutorial.

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

# 👉 Dataset

For this tutorial, I will show the end-to-end implementation of multiple time-series data forecasting, including both the training as well as predicting future values.

I have used the [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only) dataset from Kaggle. This dataset has 10 different stores and each store has 50 items, i.e. total of 500 daily level time series data for five years (2013–2017).

![Sample Dataset](https://cdn-images-1.medium.com/max/2000/1*VY7MljIxivAiYWSMmAjN6g.png)

# 👉 Load and prepare the data

    **# read the csv file
    **import pandas as pd
    data = pd.read_csv('train.csv')
    data['date'] = pd.to_datetime(data['date'])

    **# combine store and item column as time_series**
    data['store'] = ['store_' + str(i) for i in data['store']]
    data['item'] = ['item_' + str(i) for i in data['item']]
    data['time_series'] = data[['store', 'item']].apply(lambda x: '_'.join(x), axis=1)
    data.drop(['store', 'item'], axis=1, inplace=True)

    **# extract features from date**
    data['month'] = [i.month for i in data['date']]
    data['year'] = [i.year for i in data['date']]
    data['day_of_week'] = [i.dayofweek for i in data['date']]
    data['day_of_year'] = [i.dayofyear for i in data['date']]

    data.head()

![Samples rows from data](https://cdn-images-1.medium.com/max/2000/1*D3PBqLf-PsnGdTn7AjGmWw.png)

    **# check the unique time_series**
    data['time_series'].nunique()
    >>> 500

# 👉 Visualize time-series

    **# plot multiple time series with moving avgs in a loop**

    import plotly.express as px

    for i in data['time_series'].unique():
        subset = data[data['time_series'] == i]
        subset['moving_average'] = subset['sales'].rolling(30).mean()
        fig = px.line(subset, x="date", y=["sales","moving_average"], title = i, template = 'plotly_dark')
        fig.show()

![store_1_item_1 time series and 30-day moving average](https://cdn-images-1.medium.com/max/2608/1*BlE7UEMFRCV6kbEK1oWTEA.png)

![store_2_item_1 time series and 30-day moving average](https://cdn-images-1.medium.com/max/2616/1*Vhc8EP7IbA-_qdwyuENHaQ.png)

# 👉 Start the training process

Now that we have the data ready, let’s start the training loop. Notice that verbose = False in all functions to avoid printing results on the console while training.

The code below is a loop around time_series column we created during the data preparatory step. There are a total of 150 time series (10 stores x 50 items).

Line 10 below is filtering the dataset for time_series variable. The first part inside the loop is initializing the setup function, followed by compare_models to find the best model. Line 24–26 captures the results and appends the performance metrics of the best model in a list called all_results . The last part of the code uses the finalize_model function to retrain the best model on the entire dataset including the 5% left in the test set and saves the entire pipeline including the model as a pickle file.

 <iframe src="https://medium.com/media/c57291cd250844ea000a9ad0c1febfb3" frameborder=0></iframe>

We can now create a data frame from all_results list. It will display the best model selected for each time series.

    concat_results = pd.concat(all_results,axis=0)
    concat_results.head()

![sample_rows from concat_results](https://cdn-images-1.medium.com/max/2000/1*qgu9jP86L2gaZHi-TvM0SA.png)

# Training Process 👇

![Training process](https://cdn-images-1.medium.com/max/2560/1*SwI6InjXRuB-TlQwsKZZRQ.gif)

# 👉 Generate predictions using trained models

Now that we have trained models, let’s use them to generate predictions, but first, we need to create the dataset for scoring (X variables).

    **# create a date range from 2013 to 2019**
    all_dates = pd.date_range(start='2013-01-01', end = '2019-12-31', freq = 'D')

    **# create empty dataframe**
    score_df = pd.DataFrame()

    **# add columns to dataset**
    score_df['date'] = all_dates
    score_df['month'] = [i.month for i in score_df['date']]
    score_df['year'] = [i.year for i in score_df['date']]
    score_df['day_of_week'] = [i.dayofweek for i in score_df['date']]
    score_df['day_of_year'] = [i.dayofyear for i in score_df['date']]

    score_df.head()

![sample rows from score_df dataset](https://cdn-images-1.medium.com/max/2000/1*Dycm3JuIt0gsGPwPc4_8fw.png)

Now let’s create a loop to load the trained pipelines and use the predict_model function to generate prediction labels.

    from pycaret.regression import load_model, predict_model

    all_score_df = []

    for i in tqdm(data['time_series'].unique()):
        l = load_model('trained_models/' + str(i), verbose=False)
        p = predict_model(l, data=score_df)
        p['time_series'] = i
        all_score_df.append(p)

    concat_df = pd.concat(all_score_df, axis=0)
    concat_df.head()

![samples rows from concat_df](https://cdn-images-1.medium.com/max/2000/1*2l9A8jStBQEAULk9fqMObA.png)

We will now join the dataand concat_df .

    final_df = pd.merge(concat_df, data, how = 'left', left_on=['date', 'time_series'], right_on = ['date', 'time_series'])
    final_df.head()

![sample rows from final_df](https://cdn-images-1.medium.com/max/2490/1*TN9d9WssEBkn-GFSYVgNWg.png)

We can now create a loop to see all plots.

    for i in final_df['time_series'].unique()[:5]:
        sub_df = final_df[final_df['time_series'] == i]
        
        import plotly.express as px
        fig = px.line(sub_df, x="date", y=['sales', 'Label'], title=i, template = 'plotly_dark')
        fig.show()

![store_1_item_1 actual sales and predicted labels](https://cdn-images-1.medium.com/max/2604/1*tvjEor2VvHVFgumGgEtfKQ.png)

![store_2_item_1 actual sales and predicted labels](https://cdn-images-1.medium.com/max/2598/1*99cyEOPNr97uDMLKrnNqig.png)

I hope that you will appreciate the ease of use and simplicity in PyCaret. In less than 50 lines of code and one hour of experimentation, I have trained over 10,000 models (25 estimators x 500 time series) and productionalized 500 best models to generate predictions.

# Coming Soon!

Next week I will be writing a tutorial on unsupervised anomaly detection on time-series data using [PyCaret Anomaly Detection Module](https://pycaret.readthedocs.io/en/latest/api/anomaly.html). Please follow me on [Medium](https://medium.com/@moez-62905), [LinkedIn](https://www.linkedin.com/in/profile-moez/), and [Twitter](https://twitter.com/moezpycaretorg1) to get more updates.

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
