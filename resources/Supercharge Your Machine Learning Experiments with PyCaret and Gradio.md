
# Supercharge Your Machine Learning Experiments with PyCaret and Gradio

# A step-by-step tutorial to develop and interact with machine learning pipelines rapidly

![Photo by [Hunter Harritt](https://unsplash.com/@hharritt?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/10944/0*izwo6BPsV7Ru4b6r)

# 👉 Introduction

This tutorial is a step-by-step, beginner-friendly explanation of how you can integrate [PyCaret](https://www.pycaret.org) and [Gradio](https://www.gradio.app/), the two powerful open-source libraries in Python, and supercharge your machine learning experimentation within minutes.

This tutorial is a “hello world” example, I have used [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) from UCI, which is a multiclassification problem where the goal is to predict the class of iris plants. The code given in this example can be reproduced on any other dataset, without any major modifications.

# 👉 PyCaret

PyCaret is an open-source, low-code machine learning library and end-to-end model management tool built-in Python for automating machine learning workflows. It is incredibly popular for its ease of use, simplicity, and ability to build and deploy end-to-end ML prototypes quickly and efficiently.

PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few lines only. This makes the experiment cycle exponentially fast and efficient.

PyCaret is **simple and** **easy to use**. All the operations performed in PyCaret are sequentially stored in a **Pipeline** that is fully automated for **deployment. **Whether it’s imputing missing values, one-hot-encoding, transforming categorical data, feature engineering, or even hyperparameter tuning, PyCaret automates all of it.

To learn more about PyCaret, check out their [GitHub](https://www.github.com/pycaret/pycaret).

# 👉 Gradio

Gradio is an open-source Python library for creating customizable UI components around your machine learning models. Gradio makes it easy for you to “play around” with your model in your browser by dragging and dropping in your own images, pasting your own text, recording your own voice, etc., and seeing what the model outputs.

Gradio is useful for:

* Creating quick demos around your trained ML pipelines

* Getting live feedback on model performance

* Debugging your model interactively during development

To learn more about Gradio, check out their [GitHub](https://github.com/gradio-app/gradio).

![The workflow for PyCaret and Gradio](https://cdn-images-1.medium.com/max/2000/1*CLPbvtAvxkI5MbnPFE59sQ.png)

# 👉 Installing PyCaret

Installing PyCaret is very easy and takes only a few minutes. We strongly recommend using a virtual environment to avoid potential conflicts with other libraries.

PyCaret’s default installation is a slim version of pycaret which only installs hard dependencies that are [listed here](https://github.com/pycaret/pycaret/blob/master/requirements.txt).

    **# install slim version (default)
    **pip install pycaret

    **# install the full version**
    pip install pycaret[full]

When you install the full version of pycaret, all the optional dependencies as [listed here](https://github.com/pycaret/pycaret/blob/master/requirements-optional.txt) are also installed.

# 👉 Installing Gradio

You can install gradio from pip.

    pip install gradio

# 👉 Let’s get started

    **# load the iris dataset from pycaret repo**
    from pycaret.datasets import get_data
    data = get_data('iris')

![Sample rows from iris dataset](https://cdn-images-1.medium.com/max/2000/1*qttXFQnZ3atRv_qb9FtVTw.png)

# 👉 Initialize Setup

    **# initialize setup**
    from pycaret.classification import *
    s = setup(data, target = 'species', session_id = 123)

![](https://cdn-images-1.medium.com/max/2444/1*m5Sgz4IGqGEKNjbar6hGfg.png)

Whenever you initialize the setup function in PyCaret, it profiles the dataset and infers the data types for all input features. In this case, you can see all the four features (*sepal_length, sepal_width, petal_length, and petal_width*) are identified correctly as Numeric datatype. You can press enter to continue.

![Output from setup — truncated for display](https://cdn-images-1.medium.com/max/2000/1*MNchQT8Y7E_Lsg-66CVSFg.png)

Common to all modules in PyCaret, the setup function is the first and the only mandatory step to start any machine learning experiment in PyCaret. Besides performing some basic processing tasks by default, PyCaret also offers a wide array of pre-processing features such as [scaling and transformation](https://pycaret.org/normalization/), [feature engineering](https://pycaret.org/feature-interaction/), [feature selection](https://pycaret.org/feature-importance/), and several key data preparatory steps such as [one-hot-encoding](https://pycaret.org/one-hot-encoding/), [missing values imputation](https://pycaret.org/missing-values/), [over-sampling/under-sampling](https://pycaret.org/fix-imbalance/), etc. To learn more about all the preprocessing functionalities in PyCaret, you can see this [link](https://pycaret.org/preprocessing/).

![[https://pycaret.org/preprocessing/](https://pycaret.org/preprocessing/)](https://cdn-images-1.medium.com/max/2242/1*7AOrLPzJWLFH90asByQqsg.png)

# 👉 Compare Models

This is the first step we recommend in the workflow of *any* supervised experiment in PyCaret. This function trains all the available models in the model library using default hyperparameters and evaluates performance metrics using cross-validation.

The output of this function is a table showing the mean cross-validated scores for all the models. The number of folds can be defined using the foldparameter (default = 10 folds). The table is sorted (highest to lowest) by the metric of choice which can be defined using the sortparameter (default = ‘Accuracy’).

    best = compare_models(n_select = 15)
    compare_model_results = pull()

n_select parameter in the setup function controls the return of trained models. In this case, I am setting it to 15, meaning return the top 15 models as a list. pull function in the second line stores the output of compare_models as pd.DataFrame .

![Output from compare_models](https://cdn-images-1.medium.com/max/2060/1*Qu62jca8TpZLkhZgUq1uFA.png)

    len(best)
    >>> 15

    print(best[:5])

![Output from print(best[:5])](https://cdn-images-1.medium.com/max/2000/1*_H72UEY5AQlYnQswyZ0xhQ.png)

# 👉 Gradio

Now that we are done with the modeling process, let’s create a simple UI using Gradio to interact with our models. I will do it in two parts, first I will create a function that will use PyCaret’s predict_model functionality to generate and return predictions and the second part will be feeding that function into Gradio and designing a simple input form for interactivity.

# **Part I — Creating an internal function**

The first two lines of the code take the input features and convert them into pandas DataFrame. Line 7 is creating a unique list of model names displayed in the compare_models output (this will be used as a dropdown in the UI). Line 8 selects the best model based on the index value of the list (which will be passed in through UI) and Line 9 uses the predict_model functionality of PyCaret to score the dataset.

 <iframe src="https://medium.com/media/1e071798476a9bdc5bb029813507a225" frameborder=0></iframe>

# Part II — Creating a UI with Gradio

Line 3 in the code below creates a dropdown for model names, Line 4–7 creates a slider for each of the input features and I have set the default value to the mean of each feature. Line 9 initiates a UI (in the notebook as well as on your local host so you can view it in the browser).

 <iframe src="https://medium.com/media/a646d56ac3ec71b2df9dc9c9ad0709d1" frameborder=0></iframe>

![Output from running Gradio interface](https://cdn-images-1.medium.com/max/2910/1*zVe2L4L8fqDL4zIN75rFwQ.png)

You can see this quick video here to see how easy it is to interact with your pipelines and query your models without writing hundreds of lines of code or developing a full-fledged front-end.

 <iframe src="https://medium.com/media/e71d354d5edd17200ab087bee87d9608" frameborder=0></iframe>

I hope that you will appreciate the ease of use and simplicity in PyCaret and Gradio. In less than 25 lines of code and few minutes of experimentation, I have trained and evaluated multiple models using PyCaret and developed a lightweight UI to interact with models in the Notebook.

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
