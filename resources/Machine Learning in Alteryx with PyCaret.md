
# Machine Learning in Alteryx with PyCaret

# A step-by-step tutorial on training and deploying machine learning models in Alteryx Designer using PyCaret

![](https://cdn-images-1.medium.com/max/2000/1*T6OjmWCOMcsm8wi0xQcjeQ.jpeg)

# Introduction

In this tutorial, I will show you how you can train and deploy machine learning pipelines in a very popular ETL tool [Alteryx](https://www.alteryx.com) using [PyCaret](https://www.pycaret.org) — an open-source, low-code machine learning library in Python. The Learning Goals of this tutorial are:

👉 What is PyCaret and how to get started?

👉 What is Alteryx Designer and how to set it up?

👉 Train end-to-end machine learning pipeline in Alteryx Designer including data preparation such as missing value imputation, one-hot-encoding, scaling, transformations, etc.

👉 Deploy trained pipeline and generate inference during ETL.

# PyCaret

[PyCaret](https://www.pycaret.org/) is an open-source, low-code machine learning library and end-to-end model management tool built-in Python for automating machine learning workflows. PyCaret is known for its ease of use, simplicity, and ability to quickly and efficiently build and deploy end-to-end machine learning pipelines. To learn more about PyCaret, check out their [GitHub](https://www.github.com/pycaret/pycaret).

# Alteryx Designer

[Alteryx Designer](https://www.alteryx.com/products/alteryx-platform/alteryx-designer) is a proprietary tool developed by [**Alteryx](https://www.alteryx.com)** and is used for automating every step of analytics, including data preparation, blending, reporting, predictive analytics, and data science. You can access any data source, file, application, or data type, and experience the simplicity and power of a self-service platform with 260+ drag-and-drop building blocks. You can download the one-month free trial version of Alteryx Designer from [here](https://www.alteryx.com/designer-trial/alteryx-free-trial).

![[https://www.alteryx.com](https://www.alteryx.com)](https://cdn-images-1.medium.com/max/3648/1*OeDHEH-vFx2u3nF69Wu3DQ.png)

# Tutorial Pre-Requisites:

For this tutorial, you will need two things. The first one being the Alteryx Designer which is a desktop software that you can download from [here](https://www.alteryx.com/designer-trial/alteryx-free-trial). Second, you need Python. The easiest way to get Python is to download Anaconda Distribution. To download that, [click here](https://www.anaconda.com/distribution/).

# 👉We are ready now

Open Alteryx Designer and click on File → New Workflow

![New Workflow in Alteryx Designer](https://cdn-images-1.medium.com/max/3818/1*O7on438FoX76Ou9vjFDGpw.png)

On the top, there are tools that you can drag and drop on the canvas and execute the workflow by connecting each component to one another.

# Dataset

For this tutorial, I am using a regression dataset from PyCaret’s repository called ***insurance***. You can download the data from [here](https://github.com/pycaret/pycaret/blob/master/datasets/insurance.csv).

![Sample Dataset](https://cdn-images-1.medium.com/max/2000/0*_5ZOcQ4IBD55ADn6.png)

I will create two separate Alteryx workflows. First one for **model training and selection** and the second one for **scoring the new data** using the trained pipeline.

# 👉 Model Training & Selection

Let’s first read the CSV file from the **Input Data **tool followed by a **Python Script. **Inside the Python script execute the following code:

    **# install pycaret
    **from ayx import Package
    Package.installPackages('pycaret')

    **# read data from input data tool**
    from ayx import Alteryx
    data = Alteryx.read("#1")

    **# init setup, prepare data**
    from pycaret.regression import *
    s = setup(data, target = 'charges', silent=True)

    **# model training and selection
    **best = compare_models()

    **# store the results, print and save**
    results = pull()
    results.to_csv('c:/users/moezs/pycaret-demo-alteryx/results.csv', index = False)
    Alteryx.write(results, 1)

    **# finalize best model and save**
    best_final = finalize_model(best)
    save_model(best_final, 'c:/users/moezs/pycaret-demo-alteryx/pipeline')

This script is importing the regression module from pycaret, then initializing the setup function which automatically handles train_test_split and all the data preparation tasks such as missing value imputation, scaling, feature engineering, etc. compare_models trains and evaluates all the estimators using kfold cross-validation and returns the best model.

pull function calls the model performance metric as a Dataframe which is then saved as results.csv on a drive and also written to the first anchor of Python tool in Alteryx (so that you can view results on screen).

Finally, save_model saves the entire transformation pipeline including the best model as a pickle file.

![Training Workflow](https://cdn-images-1.medium.com/max/3836/1*2qny4Iy7SNePSpT7fZWSuw.png)

When you successfully execute this workflow, you will generate pipeline.pkl and results.csv file. You can see the output of the best models and their cross-validated metrics on-screen as well.

![](https://cdn-images-1.medium.com/max/2000/1*Vc6Pr88a6cxVfxxUGSp9yg.png)

This is what results.csv contains:

![](https://cdn-images-1.medium.com/max/2000/0*u9dRI79LDdDOrvw5.png)

These are the cross-validated metrics for all the models. The best model, in this case, is ***Gradient Boosting Regressor***.

# 👉 Model Scoring

We can now use our pipeline.pkl to score on the new dataset. Since I do not have a separate dataset for ***insurance.csv ***without the label***, ***what I will do is drop the target column i.e. ***charges**,* and then generate predictions using the trained pipeline.

![Scoring Workflow](https://cdn-images-1.medium.com/max/3830/1*ZVEhi6EdcXg_dKINWisR0g.png)

I have used the **Select Tool **to remove the target column i.e. charges . In the Python script execute the following code:

    **# read data from the input tool**
    from ayx import Alteryx**
    **data = Alteryx.read("#1")

    **# load pipeline
    **from pycaret.regression import load_model, predict_model
    pipeline = load_model('c:/users/moezs/pycaret-demo-alteryx/pipeline')

    **# generate predictions and save to csv
    **predictions = predict_model(pipeline, data)
    predictions.to_csv('c:/users/moezs/pycaret-demo-alteryx/predictions.csv', index=False)

    **# display in alteryx
    **Alteryx.write(predictions, 1)

When you successfully execute this workflow, it will generate predictions.csv.

![predictions.csv](https://cdn-images-1.medium.com/max/2000/0*v6pthOCcVwNMww9S.png)

# Coming Soon!

Next week I will take a deep dive and focus on more advanced functionalities of PyCaret that you can use within Alteryx to enhance your machine learning workflows. If you would like to be notified automatically, you can follow me on [Medium](https://medium.com/@moez-62905), [LinkedIn](https://www.linkedin.com/in/profile-moez/), and [Twitter](https://twitter.com/moezpycaretorg1).

![PyCaret — Image by Author](https://cdn-images-1.medium.com/max/2412/0*PLdJpNCTXdttEn8W.png)

![PyCaret — Image by Author](https://cdn-images-1.medium.com/max/2402/0*IvqhUYDstXqz55eF.png)

There is no limit to what you can achieve using this lightweight workflow automation library in Python. If you find this useful, please do not forget to give us ⭐️ on our GitHub repository.

To hear more about PyCaret follow us on [LinkedIn](https://www.linkedin.com/company/pycaret/) and [Youtube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g).

Join us on our slack channel. Invite link [here](https://join.slack.com/t/pycaret/shared_invite/zt-p7aaexnl-EqdTfZ9U~mF0CwNcltffHg).

# Important Links

[Documentation](https://pycaret.readthedocs.io/en/latest/installation.html)
[Blog](https://medium.com/@moez_62905)
[GitHub](http://www.github.com/pycaret/pycaret)
[StackOverflow](https://stackoverflow.com/questions/tagged/pycaret)
[Install PyCaret
](https://pycaret.readthedocs.io/en/latest/installation.html)[Notebook Tutorials
](https://pycaret.readthedocs.io/en/latest/tutorials.html)[Contribute in PyCaret](https://pycaret.readthedocs.io/en/latest/contribute.html)

# More PyCaret related tutorials:
[**Machine Learning in KNIME with PyCaret**
*A step-by-step guide on training and deploying end-to-end machine learning pipelines in KNIME using PyCaret*towardsdatascience.com](https://towardsdatascience.com/machine-learning-in-knime-with-pycaret-420346e133e2)
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
