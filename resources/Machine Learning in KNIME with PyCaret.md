
# Machine Learning in KNIME with PyCaret

# A step-by-step guide on training and scoring machine learning models in KNIME using PyCaret

![[PyCaret](https://www.pycaret.org) is an open-source Python library and [KNIME](https://www.knime.com) is an open-source data analytics platform](https://cdn-images-1.medium.com/max/2000/1*GCzo1_0f0E9HyK9jm7B2-w.png)

# PyCaret

[PyCaret](https://www.pycaret.org) is an open-source, low-code machine learning library and end-to-end model management tool built-in Python for automating machine learning workflows. Its ease of use, simplicity, and ability to quickly and efficiently build and deploy end-to-end machine learning pipelines will amaze you.

PyCaret is an alternate low-code library that can replace hundreds of lines of code with few lines only. This makes the experiment cycle exponentially fast and efficient.

PyCaret is **simple and easy to use. **All the operations performed in PyCaret are sequentially stored in a **Pipeline** that is fully automated for **deployment. **Whether it’s imputing missing values, one-hot-encoding, transforming categorical data, feature engineering, or even hyperparameter tuning, PyCaret automates all of it. To learn more about PyCaret, watch this 1-minute video.

 <iframe src="https://medium.com/media/7f23b1284ac52708e0987513c1107a79" frameborder=0></iframe>

# KNIME

[KNIME Analytics Platform](https://www.knime.com/knime-analytics-platform) is open-source software for creating data science. Intuitive, open, and continuously integrating new developments, KNIME makes understanding data and designing data science workflows and reusable components accessible to everyone.

KNIME Analytics platform is one of the most popular open-source platforms used in data science to automate the data science process. KNIME has thousands of nodes in the node repository which allows you to drag and drop the nodes into the KNIME workbench. A collection of interrelated nodes creates a workflow that can be executed locally as well as can be executed in the KNIME web portal after deploying the workflow into the KNIME server.

![KNIME Analytics Platform — Creating Data Science](https://cdn-images-1.medium.com/max/2000/0*ct-Ux9jTTyDYyYHZ)

# Installation

For this tutorial, you will need two things. The first one being the KNIME Analytics Platform which is a desktop software that you can download from [here](https://www.knime.com/downloads). Second, you need Python.

The easiest way to get started with Python is to download Anaconda Distribution. To download, [click here](https://www.anaconda.com/distribution/).

Once you have both the KNIME Analytics Platform and Python installed, you need to create a separate conda environment in which we will install PyCaret. Open the Anaconda prompt and run the following commands:

    ***# create a conda environment*
    **conda create --name knimeenv python=3.6
    
    ***# activate environment*
    **conda activate knimeenv
    
    ***# install pycaret*
    **pip install pycaret

Now open the KNIME Analytics Platform and go to File → Install KNIME Extensions → KNIME & Extensions → and select KNIME Python Extension and install it.

Once installation completes, go to File → Preferences → KNIME → Python and select your Python 3 environment. Notice that in my case the name of the environment is “powerbi”. If you have followed the commands above, the name of the environment is “knimeenv”.

![Python setup in KNIME Analytics Platform](https://cdn-images-1.medium.com/max/2000/1*KmNfJY16OzVldEbgfXh8kQ.png)

# 👉We are ready now

Click on “New KNIME Workflow” and a blank canvas will open.

![KNIME New Workflow](https://cdn-images-1.medium.com/max/3830/1*TdQQ1wfEMH487wd3zz9OJg.png)

On the left-hand side, there are tools that you can drag and drop on the canvas and execute the workflow by connecting each component to one another. All the actions in the repository on the left side are called *Nodes*.

# **Dataset**

For this tutorial, I am using a regression dataset from PyCaret’s repository called ‘insurance’. You can download the data from [here](https://github.com/pycaret/pycaret/blob/master/datasets/insurance.csv).

![Sample Dataset](https://cdn-images-1.medium.com/max/2000/1*mpP1hqC9HQ37WGQmdZAoFQ.png)

I will create two separate workflows. First one for model training and selection and the second one for scoring the new data using the trained pipeline.

# 👉 **Model Training & Selection**

Let’s first read the CSV file from the **CSV Reader** node followed by a **Python Script. **Inside the Python script execute the following code:

    **# init setup, prepare data**
    from pycaret.regression import *
    s = setup(input_table_1, target = 'charges', silent=True)

    **# model training and selection
    **best = compare_models()

    **# store the results, print and save**
    output_table_1 = pull()
    output_table_1.to_csv('c:/users/moezs/pycaret-demo-knime/results.csv', index = False)

    **# finalize best model and save**
    best_final = finalize_model(best)
    save_model(best_final, 'c:/users/moezs/pycaret-demo-knime/pipeline')

This script is importing the regression module from pycaret, then initializing the setup function which automatically handles train_test_split and all the data preparation tasks such as missing value imputation, scaling, feature engineering, etc. compare_models trains and evaluates all the estimators using kfold cross-validation and returns the best model. pull function calls the model performance metric as a Dataframe which is then saved as results.csv on a local drive. Finally, save_model saves the entire transformation pipeline and model as a pickle file.

![Training Workflow](https://cdn-images-1.medium.com/max/2000/1*dgzEEn15t8NmEsKKKd9sBA.png)

When you successfully execute this workflow, you will generate pipeline.pkl and results.csv file in the defined folder.

![](https://cdn-images-1.medium.com/max/2000/1*d1rh9V4BApHEqNwXOR796A.png)

This is what results.csv contains:

![](https://cdn-images-1.medium.com/max/2000/1*8iQmxMyNmXW4NS5lzpOdWA.png)

These are the cross-validated metrics for all the models. The best model, in this case, is ***Gradient Boosting Regressor***.

# 👉 Model Scoring

We can now use our pipeline.pkl to score on the new dataset. Since I do not have a separate dataset for ‘insurance.csv’, what I will do is drop the target column from the same file, just to demonstrate.

![Scoring Workflow](https://cdn-images-1.medium.com/max/2000/1*JK5Dmk1_I7u7qa7Zrskv_A.png)

I have used the **Column Filter** node to remove the target column i.e. charges . In the Python script execute the following code:

    **# load pipeline
    **from pycaret.regression import load_model, predict_model
    pipeline = load_model('c:/users/moezs/pycaret-demo-knime/pipeline')

    **# generate predictions and save to csv**
    output_table_1 = predict_model(pipeline, data = input_table_1)
    output_table_1.to_csv('c:/users/moezs/pycaret-demo-knime/predictions.csv', index=False)

When you successfully execute this workflow, it will generate predictions.csv.

![predictions.csv](https://cdn-images-1.medium.com/max/2000/1*0NjNrFay0-93xe0pje8j_g.png)

I hope that you will appreciate the ease of use and simplicity in PyCaret. When used within an analytics platform like KNIME, it can save you many hours of coding and then maintaining that code in production. With less than 10 lines of code, I have trained and evaluated multiple models using PyCaret and deployed an ML Pipeline KNIME.

# Coming Soon!

Next week I will take a deep dive and focus on more advanced functionalities of PyCaret that you can use within KNIME to enhance your machine learning workflows. If you would like to be notified automatically, you can follow me on [Medium](https://medium.com/@moez-62905), [LinkedIn](https://www.linkedin.com/in/profile-moez/), and [Twitter](https://twitter.com/moezpycaretorg1).

![Image by Author](https://cdn-images-1.medium.com/max/NaN/1*-Ul7wtRGqybl3eBm58ELcA.png)

![PyCaret — Image by Author](https://cdn-images-1.medium.com/max/NaN/1*WSZ6hqiO_B3u5ReftUCGSA.png)

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

# More PyCaret related tutorials:
[**Easy MLOps with PyCaret + MLflow**
*A beginner-friendly, step-by-step tutorial on integrating MLOps in your Machine Learning experiments using PyCaret*towardsdatascience.com](https://towardsdatascience.com/easy-mlops-with-pycaret-mlflow-7fbcbf1e38c6)
[**Write and train your own custom machine learning models using PyCaret**
towardsdatascience.com](https://towardsdatascience.com/write-and-train-your-own-custom-machine-learning-models-using-pycaret-8fa76237374e)
[**Build with PyCaret, Deploy with FastAPI**
*A step-by-step, beginner-friendly tutorial on how to build an end-to-end Machine Learning Pipeline with PyCaret and…*towardsdatascience.com](https://towardsdatascience.com/build-with-pycaret-deploy-with-fastapi-333c710dc786)
[**Time Series Anomaly Detection with PyCaret**
*A step-by-step tutorial on unsupervised anomaly detection for time series data using PyCaret*towardsdatascience.com](https://towardsdatascience.com/time-series-anomaly-detection-with-pycaret-706a6e2b2427)
[**Supercharge your Machine Learning Experiments with PyCaret and Gradio**
*A step-by-step tutorial to develop and interact with machine learning pipelines rapidly*towardsdatascience.com](https://towardsdatascience.com/supercharge-your-machine-learning-experiments-with-pycaret-and-gradio-5932c61f80d9)
[**Multiple Time Series Forecasting with PyCaret**
*A step-by-step tutorial on forecasting multiple time series using PyCaret*towardsdatascience.com](https://towardsdatascience.com/multiple-time-series-forecasting-with-pycaret-bc0a779a22fe)
[**Time Series Forecasting with PyCaret Regression Module**
*A step-by-step tutorial for time-series forecasting using PyCaret*towardsdatascience.com](https://towardsdatascience.com/time-series-forecasting-with-pycaret-regression-module-237b703a0c63)
[**5 things you are doing wrong in PyCaret**
*From the Creator of PyCaret*towardsdatascience.com](https://towardsdatascience.com/5-things-you-are-doing-wrong-in-pycaret-e01981575d2a)
[**GitHub is the best AutoML you will ever need**
*A step-by-step tutorial to build AutoML using PyCaret 2.0*towardsdatascience.com](https://towardsdatascience.com/github-is-the-best-automl-you-will-ever-need-5331f671f105)
[**Build your own AutoML in Power BI using PyCaret**
*A step-by-step tutorial to build AutoML solution in Power BI*towardsdatascience.com](https://towardsdatascience.com/build-your-own-automl-in-power-bi-using-pycaret-8291b64181d)
[**Deploy PyCaret and Streamlit app using AWS Fargate — serverless infrastructure**
*A step-by-step tutorial to containerize machine learning app and deploy it using AWS Fargate.*towardsdatascience.com](https://towardsdatascience.com/deploy-pycaret-and-streamlit-app-using-aws-fargate-serverless-infrastructure-8b7d7c0584c2)
[**Deploy Machine Learning App built using Streamlit and PyCaret on Google Kubernetes Engine**
*A step-by-step beginner’s guide to containerize and deploy a Streamlit app on Google Kubernetes Engine*towardsdatascience.com](https://towardsdatascience.com/deploy-machine-learning-app-built-using-streamlit-and-pycaret-on-google-kubernetes-engine-fd7e393d99cb)
[**Build and deploy machine learning web app using PyCaret and Streamlit**
*A beginner’s guide to deploying a machine learning app on Heroku PaaS*towardsdatascience.com](https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104)
[**Deploy Machine Learning Pipeline on AWS Fargate**
*A beginner’s guide to containerize and deploy machine learning pipeline serverless on AWS Fargate*towardsdatascience.com](https://towardsdatascience.com/deploy-machine-learning-pipeline-on-aws-fargate-eb6e1c50507)
[**Topic Modeling in Power BI using PyCaret**
*A step-by-step tutorial for implementing Topic Model in Power BI*towardsdatascience.com](https://towardsdatascience.com/topic-modeling-in-power-bi-using-pycaret-54422b4e36d6)
[**Deploy Machine Learning Pipeline on Google Kubernetes Engine**
*A beginner’s guide to containerize and deploy machine learning pipeline on Google Kubernetes Engine*towardsdatascience.com](https://towardsdatascience.com/deploy-machine-learning-model-on-google-kubernetes-engine-94daac85108b)
[**How to implement Clustering in Power BI using PyCaret**
*A step-by-step tutorial for implementing Clustering in Power BI*towardsdatascience.com](https://towardsdatascience.com/how-to-implement-clustering-in-power-bi-using-pycaret-4b5e34b1405b)
[**Build your first Anomaly Detector in Power BI using PyCaret**
*A step-by-step tutorial for implementing anomaly detection in Power BI*towardsdatascience.com](https://towardsdatascience.com/build-your-first-anomaly-detector-in-power-bi-using-pycaret-2b41b363244e)
[**Deploy Machine Learning Pipeline on the cloud using Docker Container**
*A beginner’s guide to deploy machine learning pipelines on the cloud using PyCaret, Flask, Docker Container, and Azure Web…*towardsdatascience.com](https://towardsdatascience.com/deploy-machine-learning-pipeline-on-cloud-using-docker-container-bec64458dc01)
[**Build and deploy your first machine learning web app**
*A beginner’s guide to train and deploy machine learning pipelines in Python using PyCaret*towardsdatascience.com](https://towardsdatascience.com/build-and-deploy-your-first-machine-learning-web-app-e020db344a99)
[**Machine Learning in Power BI using PyCaret**
*A step-by-step tutorial for implementing machine learning in Power BI within minutes*towardsdatascience.com](https://towardsdatascience.com/machine-learning-in-power-bi-using-pycaret-34307f09394a)
