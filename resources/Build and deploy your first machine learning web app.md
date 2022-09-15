# Build and deploy your first machine learning web app

# A beginner’s guide to train and deploy machine learning pipelines in Python using PyCaret

# by Moez Ali

![](https://cdn-images-1.medium.com/max/2000/1*NWklye0cNThqH_cTImozlA.png)


In our [last post](https://towardsdatascience.com/machine-learning-in-power-bi-using-pycaret-34307f09394a) we demonstrated how to train and deploy machine learning models in Power BI using [PyCaret](https://www.pycaret.org/). If you haven’t heard about PyCaret before, please read our [announcement](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46) to get a quick start.

In this tutorial we will use PyCaret to develop a **machine learning pipeline,** that will include preprocessing transformations and a regression model to predict patient hospitalization charges based on demographic and basic patient health risk metrics such as age, BMI, smoking status etc.

# 👉 What you will learn in this tutorial

* What is a deployment and why do we deploy machine learning models.

* Develop a machine learning pipeline and train models using PyCaret.

* Build a simple web app using a Python framework called ‘Flask’.

* Deploy a web app on ‘Heroku’ and see your model in action.

# 💻 What tools we will use in this tutorial?

# PyCaret

[PyCaret](https://www.pycaret.org/) is an open source, low-code machine learning library in Python to train and deploy machine learning pipelines and models in production. PyCaret can be installed easily using pip.

    # for Jupyter notebook on your local computer
    pip install **pycaret**

    # for azure notebooks and google colab
    !pip install **pycaret**

# Flask

[Flask](https://flask.palletsprojects.com/en/1.1.x/) is a framework that allows you to build web applications. A web application can be a commercial website, a blog, e-commerce system, or an application that generates predictions from data provided in real-time using trained models. If you don’t have Flask installed, you can use pip to install it.

    # install flask
    pip install **Flask**

# GitHub

[GitHub](https://www.github.com/) is a cloud-based service that is used to host, manage and control code. Imagine you are working in a large team where multiple people (sometime hundreds of them) are making changes. PyCaret is itself an example of an open-source project where hundreds of community developers are continuously contributing to source code. If you haven’t used GitHub before, you can [sign up](https://github.com/join) for a free account.

# Heroku

[Heroku](https://www.heroku.com/) is a platform as a service (PaaS) that enables the deployment of web apps based on a managed container system, with integrated data services and a powerful ecosystem. In simple words, this will allow you to take the application from your local machine to the cloud so that anybody can access it using a Web URL. In this tutorial we have chosen Heroku for deployment as it provides free resource hours when you [sign up](https://signup.heroku.com/) for new account.

![Machine Learning Workflow (from Training to Deployment on PaaS)](https://cdn-images-1.medium.com/max/2000/1*GCRVoOwIKL_AhmrwOtQwaA.png)

# Why Deploy Machine Learning Models?

The deployment of machine learning models is the process of making models available in production where web applications, enterprise software and APIs can consume the trained model by providing new data points and generating predictions.

Normally machine learning models are built so that they can be used to predict an outcome (binary value i.e. 1 or 0 for [Classification](https://www.pycaret.org/classification), continuous values for [Regression](https://www.pycaret.org/regression), labels for [Clustering](https://www.pycaret.org/clustering) etc. There are two broad ways of generating predictions (i) predict by batch; and (ii) predict in real-time. In our [last tutorial](https://towardsdatascience.com/machine-learning-in-power-bi-using-pycaret-34307f09394a) we demonstrated how to deploy machine learning model in Power BI and predict by batch. In this tutorial we will see how to deploy a machine learning model to predict in real-time.

# Business Problem

An insurance company wants to improve its cash flow forecasting by better predicting patient charges using demographic and basic patient health risk metrics at the time of hospitalization.

![](https://cdn-images-1.medium.com/max/2000/0*10e8RTwI5t0Wi8fg.png)

*([data source](https://www.kaggle.com/mirichoi0218/insurance#insurance.csv))*

# Objective

To build a web application where demographic and health information of a patient is entered in a web form to predict charges.

# Tasks

* Train and validate models and develop a machine learning pipeline for deployment.

* Build a basic HTML front-end with an input form for independent variables (age, sex, bmi, children, smoker, region).

* Build a back-end of the web application using a Flask Framework.

* Deploy the web app on Heroku. Once deployed, it will become publicly available and can be accessed via Web URL.

# 👉 Task 1 — Model Training and Validation

Training and model validation are performed in Integrated Development Environment (IDE) or Notebook either on your local machine or on cloud. In this tutorial we will use PyCaret in Jupyter Notebook to develop machine learning pipeline and train regression models. If you haven’t used PyCaret before, [click here](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46) to learn more about PyCaret or see [Getting Started Tutorials](https://www.pycaret.org/tutorial) on our [website](https://www.pycaret.org/).

In this tutorial, we have performed two experiments. The first experiment is performed with default preprocessing settings in PyCaret (missing value imputation, categorical encoding etc). The second experiment has some additional preprocessing tasks such as scaling and normalization, automatic feature engineering and binning continuous data into intervals. See the setup example for the second experiment:

    # Experiment No. 2

    from **pycaret.regression** import *****

    r2 = **setup**(data, target = 'charges', session_id = 123,
               normalize = True,
               polynomial_features = True, trigonometry_features = True,
               feature_interaction=True, 
               bin_numeric_features= ['age', 'bmi'])

![Comparison of information grid for both experiments](https://cdn-images-1.medium.com/max/2000/0*lA_5MECr5Onj0nRS.png)

The magic happens with only a few lines of code. Notice that in **Experiment 2** the transformed dataset has 62 features for training derived from only 7 features in the original dataset. All of the new features are the result of transformations and automatic feature engineering in PyCaret.

![Columns in dataset after transformation](https://cdn-images-1.medium.com/max/2000/0*c6jeng5IXupSzJtE.png)

Sample code for model training and validation in PyCaret:

    # Model Training and Validation 
    lr = **create_model**('lr')

![10 Fold cross-validation of Linear Regression Model(s)](https://cdn-images-1.medium.com/max/2276/0*qGM8XwQdUpU5YK2z.png)

Notice the impact of transformations and automatic feature engineering. The R2 has increased by 10% with very little effort. We can compare the **residual plot** of linear regression model for both experiments and observe the impact of transformations and feature engineering on the **heteroskedasticity **of model.

    # plot residuals of trained model**
    plot_model**(lr, plot = 'residuals')

![Residual Plot of Linear Regression Model(s)](https://cdn-images-1.medium.com/max/2876/0*FJDkF4CzHtwfGlbg.png)

Machine learning is an *iterative *process. Number of iterations and techniques used within are dependent on how critical the task is and what the impact will be if predictions are wrong. The severity and impact of a machine learning model to predict a patient outcome in real-time in the ICU of a hospital is far more than a model built to predict customer churn.

In this tutorial, we have performed only two iterations and the linear regression model from the second experiment will be used for deployment. At this stage, however, the model is still only an object within notebook. To save it as a file that can be transferred to and consumed by other applications, run the following code:

    # save transformation pipeline and model 
    **save_model**(lr, model_name = 'c:/*username*/ins/deployment_28042020')

When you save a model in PyCaret, the entire transformation pipeline based on the configuration defined in the **setup() **function is created . All inter-dependencies are orchestrated automatically. See the pipeline and model stored in the ‘deployment_28042020’ variable:

![Pipeline created using PyCaret](https://cdn-images-1.medium.com/max/2000/0*ZY7Ep3vsER-6G7ug.png)

We have finished our first task of training and selecting a model for deployment. The final machine learning pipeline and linear regression model is now saved as a file in the local drive under the location defined in the **save_model() **function. (In this example: c:/*username*/ins/deployment_28042020.pkl).

# 👉 Task 2 — Building Web Application

Now that our machine learning pipeline and model are ready we will start building a web application that can connect to them and generate predictions on new data in real-time. There are two parts of this application:

* Front-end (designed using HTML)

* Back-end (developed using Flask in Python)

# Front-end of Web Application

Generally, the front-end of web applications are built using HTML which is not the focus of this article. We have used a simple HTML template and a CSS style sheet to design an input form. Here’s the HTML snippet of the front-end page of our web application.

![Code snippet from home.html file](https://cdn-images-1.medium.com/max/2388/1*t0a5Ev7oFJf73oIY8lkHZQ.png)

You don’t need to be an expert in HTML to build simple applications. There are numerous free platforms that provide HTML and CSS templates as well as enable building beautiful HTML pages quickly by using a drag and drop interface.

**CSS Style Sheet
**CSS (also known as Cascading Style Sheets) describes how HTML elements are displayed on a screen. It is an efficient way of controlling the layout of your application. Style sheets contain information such as background color, font size and color, margins etc. They are saved externally as a .css file and is linked to HTML but including 1 line of code.

![Code snippet from home.html file](https://cdn-images-1.medium.com/max/2392/1*2CecMn4-O-slFc6Tf1BK1w.png)

# Back-end of Web Application

The back-end of a web application is developed using a Flask framework. For beginner’s it is intuitive to consider Flask as a library that you can import just like any other library in Python. See the sample code snippet of our back-end written using a Flask framework in Python.

![Code snippet from app.py file](https://cdn-images-1.medium.com/max/2444/0*2KpsfiiecB5mCtab.png)

If you remember from the Step 1 above we have finalized linear regression model that was trained on 62 features that were automatically engineered by PyCaret. However, the front-end of our web application has an input form that collects only the six features i.e. age, sex, bmi, children, smoker, region.

How do we transform 6 features of a new data point in real-time into 62 features on which model was trained? With a sequence of transformations applied during model training, coding becomes increasingly complex and time-taking task.

In PyCaret all transformations such as categorical encoding, scaling, missing value imputation, feature engineering and even feature selection are automatically executed in real-time before generating predictions.
> # *Imagine the amount of code you would have had to write to apply all the transformations in strict sequence before you could even use your model for predictions. In practice, when you think of machine learning, you should think about the entire ML pipeline and not just the model.*

**Testing App
**One final step before we publish the application on Heroku is to test the web app locally. Open Anaconda Prompt and navigate to folder where **‘app.py’** is saved on your computer. Run the python file with below code:

    python **app.py**

![Output in Anaconda Prompt when app.py is executed](https://cdn-images-1.medium.com/max/2204/0*NvcdEyGUoUWWoJKZ.png)

Once executed, copy the URL into a browser and it should open a web application hosted on your local machine (127.0.0.1). Try entering test values to see if the predict function is working. In the example below, the expected bill for a 19 year old female smoker with no children in the southwest is $20,900.

![Web application opened on local machine](https://cdn-images-1.medium.com/max/3780/0*GBP1kfSwpBzstWzI.png)

Congratulations! you have now built your first machine learning app. Now it’s time to take this application from your local machine into the cloud so other people can use it with a Web URL.

# 👉 Task 3 — Deploy the Web App on Heroku

Now that the model is trained, the machine learning pipeline is ready, and the application is tested on our local machine, we are ready to start our deployment on Heroku. There are couple of ways to upload your application source code onto Heroku. The simplest way is to link a GitHub repository to your Heroku account.

If you would like to follow along you can fork this [repository](https://github.com/pycaret/deployment-heroku) from GitHub. If you don’t know how to fork a repo, please [read this](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) official GitHub tutorial.

![[https://www.github.com/pycaret/deployment-heroku](https://www.github.com/pycaret/deployment-heroku)](https://cdn-images-1.medium.com/max/2524/0*GPOio0x0TnQFr8r8.png)

By now you are familiar with all the files in repository shown above except for two files i.e. ‘**requirements.txt’** and ‘**Procfile’.**

![requirements.txt](https://cdn-images-1.medium.com/max/2440/0*XHMUGa90vc6csKfO.png)

**requirements.txt **file is a text file containing the names of the python packages required to execute the application. If these packages are not installed in the environment application is running, it will fail.

![Procfile](https://cdn-images-1.medium.com/max/2470/0*NFbZtaStgESRUIol.png)

**Procfile **is simply one line of code that provides startup instructions to web server that indicate which file should be executed first when somebody logs into the application. In this example the name of our application file is ‘**app.py’ **and the name of the application is also ‘**app’**. *(hence app:app)*

Once all the files are uploaded onto the GitHub repository, we are now ready to start deployment on Heroku. Follow the steps below:

**Step 1 — Sign up on heroku.com and click on ‘Create new app’**

![Heroku Dashboard](https://cdn-images-1.medium.com/max/3108/0*MXhm58jzLVET5Xa4.png)

**Step 2 — Enter App name and region**

![Heroku — Create new app](https://cdn-images-1.medium.com/max/2000/0*8Lu1Fc9A7iGnVJCm.png)

**Step 3 — Connect to your GitHub repository where code is hosted**

![Heroku — Connect to GitHub](https://cdn-images-1.medium.com/max/3092/0*VyAQvI2kDr2SsXYz.png)

**Step 4 — Deploy branch**

![Heroku — Deploy Branch](https://cdn-images-1.medium.com/max/3022/0*A5Tg_Qt5cZ6aLl92.png)

**Step 5 — Wait 5–10 minutes and BOOM**

![Heroku — Successful deployment](https://cdn-images-1.medium.com/max/3078/0*TFPIem6Q5k6DKszI.png)

App is published to URL: [https://pycaret-insurance.herokuapp.com/](https://pycaret-insurance.herokuapp.com/)

![[https://pycaret-insurance.herokuapp.com/](https://pycaret-insurance.herokuapp.com/)](https://cdn-images-1.medium.com/max/3772/0*Fr199orKNCkPMBSf.png)

There is one last thing to see before we end the tutorial.

So far we have built and deployed a web application that works with our machine learning pipeline. Now imagine that you already have an enterprise application in which you want to integrate predictions from your model. What you need is a web service where you can make an API call with input data points and get the predictions back. To achieve this we have created the ***predict_api*** function in our **‘app.py’** file. See the code snippet:

![Code snippet from app.py file (back-end of web app)](https://cdn-images-1.medium.com/max/2000/0*JvwXvC3bBpKaPTE_.png)

Here’s how you can use this web service in Python using the requests library:

    import **requests**url = 'https://pycaret-insurance.herokuapp.com/predict_api'pred = **requests.post(**url,json={'age':55, 'sex':'male', 'bmi':59, 'children':1, 'smoker':'male', 'region':'northwest'})**print**(pred.json())

![Make a request to a published web service to generate predictions in a Notebook](https://cdn-images-1.medium.com/max/2474/0*a9T8yMRXwymlccdr.png)

# Next Tutorial

In the next tutorial for deploying machine learning pipelines, we will dive deeper into deploying machine learning pipelines using docker containers. We will demonstrate how to easily deploy and run containerized machine learning applications on Linux.

Follow our [LinkedIn](https://www.linkedin.com/company/pycaret/) and subscribe to our [Youtube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g) channel to learn more about PyCaret.

# Important Links

[User Guide / Documentation](https://www.pycaret.org/guide)
[GitHub Repository
](https://www.github.com/pycaret/pycaret)[Install PyCaret](https://www.pycaret.org/install)
[Notebook Tutorials](https://www.pycaret.org/tutorial)
[Contribute in PyCaret](https://www.pycaret.org/contribute)

# Want to learn about a specific module?

As of the first release 1.0.0, PyCaret has the following modules available for use. Click on the links below to see the documentation and working examples in Python.

[Classification](https://www.pycaret.org/classification)
[Regression
](https://www.pycaret.org/regression)[Clustering](https://www.pycaret.org/clustering)
[Anomaly Detection
](https://www.pycaret.org/anomaly-detection)[Natural Language Processing](https://www.pycaret.org/nlp)
[Association Rule Mining](https://www.pycaret.org/association-rules)

# Also see:

PyCaret getting started tutorials in Notebook:

[Clustering](https://www.pycaret.org/clu101)
[Anomaly Detection](https://www.pycaret.org/anom101)
[Natural Language Processing](https://www.pycaret.org/nlp101)
[Association Rule Mining](https://www.pycaret.org/arul101)
[Regression](https://www.pycaret.org/reg101)
[Classification](https://www.pycaret.org/clf101)

# What’s in the development pipeline?

We are actively working on improving PyCaret. Our future development pipeline includes a new **Time Series Forecasting **module, integration with **TensorFlow, **and major improvements on the scalability of PyCaret. If you would like to share your feedback and help us improve further, you may [fill this form](https://www.pycaret.org/feedback) on the website or leave a comment on our [GitHub ](https://www.github.com/pycaret/)or [LinkedIn](https://www.linkedin.com/company/pycaret/) page.

# Would you like to contribute?

PyCaret is an open source project. Everybody is welcome to contribute. If you would like contribute, please feel free to work on [open issues](https://github.com/pycaret/pycaret/issues). Pull requests are accepted with unit tests on dev-1.0.1 branch.

Please give us ⭐️ on our [GitHub repo](https://www.github.com/pycaret/pycaret) if you like PyCaret.

Medium : [https://medium.com/@moez_62905/](https://medium.com/@moez_62905/machine-learning-in-power-bi-using-pycaret-34307f09394a)

LinkedIn : [https://www.linkedin.com/in/profile-moez/](https://www.linkedin.com/in/profile-moez/)

Twitter : [https://twitter.com/moezpycaretorg1](https://twitter.com/moezpycaretorg1)
