
# Deploy PyCaret Models on Edge Devices with ONNX Runtime

# A step-by-step tutorial on how to convert ML models trained using PyCaret to ONNX for high-performance scoring (CPU or GPU)

![Photo by [Austin Distel](https://unsplash.com/@austindistel?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/12668/0*X79bEMfw0xAW7nKT)

# Introduction

In this tutorial, I will show you how you can train machine learning models using [PyCaret](https://www.pycaret.org/) — an open-source, low-code machine learning library in Python—and convert them in ONNX format for deployment on an edge device or any other non-Python environment. For example, you can train machine learning models using PyCaret in Python and deploy them in R, Java, or C. The Learning Goals of this tutorial are:

👉 What is PyCaret and how to get started?

👉 What are different types of model formats (pickle, onnx, pmml, etc.)

👉 What is ONNX (*pronounced as ONEX*) and what are its benefits?

👉 Train machine learning model using PyCaret and convert it in ONNX for deployment on edge.

# PyCaret

[PyCaret](https://www.pycaret.org/) is an open-source, low-code machine learning library and end-to-end model management tool built-in Python for automating machine learning workflows. PyCaret is known for its ease of use, simplicity, and ability to quickly and efficiently build and deploy end-to-end machine learning pipelines. To learn more about PyCaret, check out their [GitHub](https://www.github.com/pycaret/pycaret).

**Features:**

![PyCaret — An open-source, low-code machine learning library in Python](https://cdn-images-1.medium.com/max/2084/1*sESpLOGhMa2U1FsFdxxzIQ.png)

# skl2onnx

[skl2onnx](https://github.com/onnx/sklearn-onnx) is an open-source project that converts scikit-learn models to ONNX. Once in the ONNX format, you can use tools like ONNX Runtime for high-performance scoring. This project was started by the engineers and data scientists at Microsoft in 2017. To learn more about this project, check out their [GitHub](https://github.com/onnx/sklearn-onnx).

# Install

You will need to install the following libraries for this tutorial. The installation will take only a few minutes.

    **# install pycaret
    **pip install pycaret

    **# install skl2onnx
    **pip install skl2onnx

    **# install onnxruntime
    **pip install onnxruntime

# Different Model Formats

Before I introduce ONNX and the benefits, let’s see what are the different model formats available today for deployment.

# 👉**Pickle**

This is the most common format and default way of saving model objects into files for many Python libraries including PyCaret. [Pickle](https://docs.python.org/3/library/pickle.html) converts a Python object to a bitstream and allows it to be stored to disk and reloaded at a later time. It provides a good format to store machine learning models provided that the inference applications are also built-in python.

# 👉PMML

Predictive model markup language (PMML) is another format for machine learning models, relatively less common than Pickle. PMML has been around since 1997 and so has a large footprint of applications leveraging the format. Applications such as SAP **and PEGA CRM are able to leverage certain versions of the PMML. There are open-source libraries available that can convert scikit-learn models (PyCaret) to PMML. The biggest drawback of the PMML format is that it doesn’t support all machine learning models.

# 👉ONNX

[ONNX](https://github.com/onnx), the Open Neural Network Exchange Format is an open format that supports the storing and porting of machine learning models across libraries and languages. This means that you can train your machine learning model using any framework in any language and then convert it into ONNX that can be used to generate inference in any environment (be it Java, C, .Net, Android, etc.). This language-agnostic capability of ONNX makes it really powerful compared to the other formats (For example You cannot use a model saved as a Pickle file in any other language than Python).

# What is ONNX?

[ONNX](https://onnx.ai/) is an open format to represent both deep learning and traditional models. With ONNX, AI developers can more easily move models between state-of-the-art tools and choose the combination that is best for them. ONNX is developed and supported by a community of partners such as Microsoft, Facebook, and AWS.

ONNX is widely supported and can be found in many frameworks, tools, and hardware. Enabling interoperability between different frameworks and streamlining the path from research to production helps increase the speed of innovation in the AI community. ONNX helps to solve the challenge of hardware dependency related to AI models and enables deploying the same AI models to several HW accelerated targets.

***Source: Microsoft***

![[https://microsoft.github.io/ai-at-edge/docs/onnx/](https://microsoft.github.io/ai-at-edge/docs/onnx/)](https://cdn-images-1.medium.com/max/2000/0*9WvPLwTrLDynzQGM.PNG)

There are many excellent machine learning libraries in various languages — PyTorch, TensorFlow, scikit-learn, PyCaret, etc. The idea is that you can train a model with any tool, language, or framework and then deploy it using another language or application for inference and prediction. For example, let’s say you have a web application built with .Net, an Android app, or even an edge device and you want to integrate your machine learning model predictions into those downstream systems. You can do that by converting your model into ONNX format. *You cannot do that with Pickle or PMML format.*

# **Key Benefits:**

# 👉 Interoperability

Develop in your preferred framework without worrying about downstream inferencing implications. ONNX enables you to use your preferred framework with your chosen inference engine.

# 👉Hardware Access

ONNX makes it easier to access hardware optimizations. Use ONNX-compatible runtimes and libraries designed to maximize performance across hardware. This means that you can even use ONNX models on GPU for inference if latency is something you care about.

![Compatibility vs. Interoperability](https://cdn-images-1.medium.com/max/2000/0*CNFZ8AKtAPwDYki3.png)

# 👉Let's Get Started

# Dataset

For this tutorial, I am using a regression dataset from PyCaret’s repository called ***insurance***. You can download the data from [here](https://github.com/pycaret/pycaret/blob/master/datasets/insurance.csv).

![Sample Dataset](https://cdn-images-1.medium.com/max/2000/0*AlNXvwqZitdNOLUJ.png)

    **# loading dataset
    **from pycaret.datasets import get_data
    data = get_data('insurance')

    **# initialize setup / data preparation
    **from pycaret.regression import *
    s = setup(data, target = 'charges')

![Output from the setup function (compressed for display purpose)](https://cdn-images-1.medium.com/max/2000/1*wRI5YKWljqvtzKHNnc4osQ.png)

# 👉 Model Training & Selection

Now that data is ready for modeling, let’s start the training process by using compare_models function. It will train all the algorithms available in the model library and evaluates multiple performance metrics using k-fold cross-validation.

    **# compare all models**
    best = compare_models()

![Output from compare_models](https://cdn-images-1.medium.com/max/2000/1*7aZp9Tt2oPIyw6xbdzlnLQ.png)

Based on cross-validation metrics the best model is ***Gradient Boosting Regressor. ***You can save the model as a Pickle file with the save_model function.

    **# save model to drive
    **save_model(best, 'c:/users/models/insurance')

This will save the model in a Pickle format.

# 👉 Generate Predictions using Pickle format

You can load the saved model back in the Python environment with the load_model function and generate inference using predict_model function.

    **# load the model
    **from pycaret.regression import load_model
    loaded_model = load_model('c:/users/models/insurance')

    **# generate predictions / inference
    **from pycaret.regression import predict_model
    pred = predict_model(loaded_model, data=data) # new data

![Predictions generated on the test set](https://cdn-images-1.medium.com/max/2000/1*vjO887TVlqS9H2utp2rY9A.png)

# 👉 ONNX Conversion

So far what we have seen is saving and loading trained models in Pickle format (which is the default format for PyCaret). However, using the skl2onnx library we can convert the model in ONNX:

    **# convert best model to onnx**
    from skl2onnx import to_onnx
    X_sample = get_config('X_train')[:1]
    model_onnx = to_onnx(best, X_sample.to_numpy())

We can also save the model_onnx to local drive:

    **# save the model to drive**
    with open("c:/users/models/insurance.onnx", "wb") as f:
        f.write(model_onnx.SerializeToString())

Now to generate the inference from the insurance.onnx we will use onnxruntime library in Python (just to demonstrate the point). Essentially you can now use this insurance.onnx in any other platform or environment.

    **# generate inference on onnx**
    from onnxruntime import InferenceSession
    sess = InferenceSession(model_onnx.SerializeToString())
    X_test = get_config('X_test').to_numpy()
    predictions_onnx = sess.run(None, {'X': X_test})[0]

    **# print predictions_onnx
    **print(predictions_onnx)

![predictions_onnx](https://cdn-images-1.medium.com/max/2000/1*GnEcu24SeB7y2--rYT0qyA.png)

Notice that the output from predictions_onnx is a numpy array compared to the pandas DataFrame when we have used predict_model function from PyCaret but if you match the values, the numbers are all same (*with ONNX sometimes you will find minor differences beyond the 4th decimal point — very rarely*).
>  **Mission Accomplished!**

# Coming Soon!

Next week I will take a deep dive into ONNX conversions and talk about how to convert the entire machine learning pipelines (*including imputers and transformers*) into ONNX. If you would like to be notified automatically, you can follow me on [Medium](https://medium.com/@moez-62905), [LinkedIn](https://www.linkedin.com/in/profile-moez/), and [Twitter](https://twitter.com/moezpycaretorg1).

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
[**Machine Learning in Alteryx with PyCaret**
*A step-by-step tutorial on training and deploying machine learning models in Alteryx Designer using PyCaret*towardsdatascience.com](https://towardsdatascience.com/machine-learning-in-alteryx-with-pycaret-fafd52e2d4a)
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
