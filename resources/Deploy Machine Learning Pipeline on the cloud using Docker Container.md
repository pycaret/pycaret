
# Deploy Machine Learning Pipeline on the cloud using Docker Container

# by Moez Ali

![](https://cdn-images-1.medium.com/max/2000/1*N3IRs4nRw4vcMt_AHmbkPA.png)

# **RECAP**

In our [last post](https://towardsdatascience.com/build-and-deploy-your-first-machine-learning-web-app-e020db344a99), we demonstrated how to develop a machine learning pipeline and deploy it as a web app using PyCaret and Flask framework in Python. If you haven’t heard about PyCaret before, please read this [announcement](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46) to learn more.

In this tutorial, we will use the same machine learning pipeline and Flask app that we built and deployed previously. This time we will demonstrate how to deploy a machine learning pipeline as a web app using the [Microsoft Azure Web App Service](https://azure.microsoft.com/en-us/services/app-service/web/).

In order to deploy a machine learning pipeline on Microsoft Azure, we will have to containerize our pipeline in a software called **“Docker”**. If you don’t know what does containerize means, *no problem* — this tutorial is all about that.

# 👉 Learning Goals of this Tutorial

* What is a container? What is Docker? and why do we need it?

* Build a Docker file on your local computer and publish it into [Azure Container Registry (ACR)](https://azure.microsoft.com/en-us/services/container-registry/).

* Deploy a web service on Azure using the container we uploaded into ACR.

* See a web app in action that uses a trained machine learning pipeline to predict on new data points in real-time.

In our last post, we covered the basics of model deployment and why it is needed. If you would like to learn more about model deployment, [click here](https://towardsdatascience.com/build-and-deploy-your-first-machine-learning-web-app-e020db344a99) to read our last article.

This tutorial will cover the entire workflow of building a container locally to pushing it onto Azure Container Registry and then deploying our pre-trained machine learning pipeline and Flask app onto Azure Web Services.

![WORKFLOW: Create an image → Build container locally → Push to ACR → Deploy app on cloud](https://cdn-images-1.medium.com/max/2512/1*4McqTG9jDQvl_t-omPkEuA.png)

# 💻 Toolbox for this tutorial

# PyCaret

[PyCaret](https://www.pycaret.org/) is an open source, low-code machine learning library in Python that is used to train and deploy machine learning pipelines and models into production. PyCaret can be installed easily using pip.

    pip install **pycaret**

# Flask

[Flask](https://flask.palletsprojects.com/en/1.1.x/) is a framework that allows you to build web applications. A web application can be a commercial website, blog, e-commerce system, or an application that generates predictions from data provided in real-time using trained models. If you don’t have Flask installed, you can use pip to install it.

# **Docker**

[Docker](https://www.docker.com/)** **is a tool designed to make it easier to create, deploy, and run applications by using containers. Containers are used to package up an application with all of its necessary components, such as libraries and other dependencies, and ship it all out as one package. If you haven’t used docker before, this tutorial also covers the installation of docker on Windows 10.

# **Microsoft Azure**

[Microsoft Azure](https://azure.microsoft.com/en-ca/overview/what-is-azure/) is a set of cloud services that is used to build, manage and deploy applications on a massive and global network. Other cloud services that are often used for deploying ML pipelines are [Amazon Web Services (AWS)](https://aws.amazon.com/), [Google Cloud](https://cloud.google.com), [IBM Cloud](https://www.ibm.com/cloud) and [Alibaba Cloud](https://www.alibabacloud.com/). We will cover most of them in our future tutorials.

If you haven’t used Microsoft Azure before you can [sign up](https://azure.microsoft.com/en-ca/free/search/?&ef_id=EAIaIQobChMIm8Onqp6i6QIViY7ICh2QVA2jEAAYASAAEgK9FvD_BwE:G:s&OCID=AID2000061_SEM_EAIaIQobChMIm8Onqp6i6QIViY7ICh2QVA2jEAAYASAAEgK9FvD_BwE:G:s&dclid=CK6R8aueoukCFVbJyAoduGYLcQ) for a free account here. When you sign up for the first time you get a free credit for the first 30 days. You can utilize that credit in building your own web app by following this tutorial.

# What is a Container and why do we need it?

Have you ever had the problem where your python code (*or any other code*) works fine on your computer but when your friend tries to run the exact same code, it doesn’t work? If your friend is repeating the exact same steps, they should get the same results right? The one-word answer to this is ***the environment*. **Your friend’s Python environment is different than yours.

What does an environment include? → Python (*or any other language you have used*) and all the libraries and dependencies with the exact versions using which application was built and tested.

If we can somehow create an environment that we can transfer to other machines (for example: your friend’s computer or a cloud service provider like Microsoft Azure), we can reproduce the results anywhere. Hence, **a** **container **is a type of software that packages up an application and all its dependencies so the application runs reliably from one computing environment to another.
> # “Think about **containers, **when you think about containers.”

![[https://www.freepik.com/free-photos-vectors/cargo-ship](https://www.freepik.com/free-photos-vectors/cargo-ship)](https://cdn-images-1.medium.com/max/2000/1*SlzsvRhA71oFOhAjE1Hs0A.jpeg)

This is the most intuitive way to understand containers in data science. **They are just like containers on a ship **where the goal is to isolate the *contents *of one container from the others so they don’t get mixed up. This is exactly what containers are used for in data science.

Now that we understand the metaphor behind containers, let’s look at alternate options for creating an isolated environment for our application. One simple alternative is to have a separate machine for each of your applications.

(1 machine = 1 application = no conflict = everything is good)

Using a separate machine is straight forward but it doesn’t outweigh the benefits of using containers since maintaining multiple machines for each application is expensive, a nightmare-to-maintain and hard-to-scale. In short, it’s not practical in many real-life scenarios.

Another alternate for creating an isolated environment are **virtual machines. **Containers are again preferable here because they require fewer resources, are very portable, and are faster to spin up.

![Virtual Machines vs. Containers](https://cdn-images-1.medium.com/max/3840/1*snINBI0HUIYa0BWKyCO3Xg.jpeg)

Can you spot the difference between Virtual Machines and Containers? When you use containers, you do not require guest operating systems. Imagine 10 applications running on a virtual machine. This would require 10 guest operating systems compared to none required when you use containers.

# I understand containers but what is Docker?

Docker is a company that provides software (also called Docker) that allows users to build, run and manage containers. While Docker’s container are the most common, there are other less famous *alternatives* such as [LXD](https://linuxcontainers.org/lxd/introduction/) and [LXC](https://linuxcontainers.org/) that provides container solution.

In this tutorial, we will use **Docker Desktop for Windows **to create a container that we will publish on Azure Container Registry. We will then deploy a web app using that container.

![](https://cdn-images-1.medium.com/max/2000/1*EJx9QN4ENSPKZuz51rC39w.png)

# Docker Image vs. Docker Container

What is the difference between a docker image and a docker container? This is by far the most common question asked so let’s clear this right away. There are many technical definitions available, however, it is intuitive to think about a docker image as a mold based on which container is created. An image is essentially a snapshot of container.

If you prefer a slightly more technical definition then consider this: Docker images become containers at runtime when they run on a Docker Engine.

# **Breaking the hype:**

At the end of the day, docker is just a file with a few lines of instructions that are saved under your project folder with the name ***“Dockerfile”***.

Another way to think about docker file is that they are like recipes you have invented in your own kitchen. When you share those recipes with somebody else and they follow the exact same instructions, they are able to produce the same dish. Similarly, you can share your docker file with other people, who can then create images and run containers based on that docker file.

Now that you understand containers, docker and why we should use them, let’s quickly set the business context.

# Setting the Business Context

An insurance company wants to improve its cash flow forecasting by better predicting patient charges using demographic and basic patient health risk metrics at the time of hospitalization.

![](https://cdn-images-1.medium.com/max/2000/1*qM1HiWZ_uigwdcZ0_ZL6yA.png)

*([data source](https://www.kaggle.com/mirichoi0218/insurance#insurance.csv))*

# Objective

To build and deploy a web application where the demographic and health information of a patient is entered into a web-based form which then outputs a predicted charge amount.

# Tasks

* Train and develop a machine learning pipeline for deployment.

* Build a web app using Flask framework. It will use the trained ML pipeline to generate predictions on new data points in real-time.

* Create a docker image and container.

* Publish the container onto Azure Container Registry (ACR).

* Deploy the web app in the container by publishing onto ACR. Once deployed, it will become publicly available and can be accessed via a Web URL.

Since we have already covered the first two tasks in our last tutorial, we will quickly recap them and focus on the remaining tasks in the list above. If you are interested in learning more about developing machine learning pipeline in Python using PyCaret and building a web app using Flask framework, you can read our [last tutorial](https://towardsdatascience.com/build-and-deploy-your-first-machine-learning-web-app-e020db344a99).

# 👉 Develop Machine Learning Pipeline

We are using PyCaret in Python for training and developing a machine learning pipeline which will be used as part of our web app. The Machine Learning Pipeline can be developed in an Integrated Development Environment (IDE) or Notebook. We have used a notebook to run the below code:

 <iframe src="https://medium.com/media/ab682376970c812cbe30e75c0ae3a370" frameborder=0></iframe>

When you save a model in PyCaret, the entire transformation pipeline based on the configuration defined in the **setup() **function is created . All inter-dependencies are orchestrated automatically. See the pipeline and model stored in the ‘deployment_28042020’ variable:

![Machine Learning Pipeline created using PyCaret](https://cdn-images-1.medium.com/max/2000/1*NWoHVWJzO7i7gIvrlBnIiQ.png)

# 👉 Build Web Application

This tutorial is not focused on building a Flask application. It is only discussed here for completeness. Now that our machine learning pipeline is ready we need a web application that can connect to our trained pipeline to generate predictions on new data points in real-time. We have created the web application using Flask framework in Python. There are two parts of this application:

* Front-end (designed using HTML)

* Back-end (developed using Flask)

This is how our web application looks:

![Web application opened on local machine](https://cdn-images-1.medium.com/max/2800/1*EU4Cp9w1YHS2om8Mmfqh2g.png)

If you would like to see this web app in action, [click here](https://pycaret-insurance.herokuapp.com/) to open a deployed web app on Heroku (*It may take few minutes to open*).

If you haven’t followed along, no problem. You can simply fork this [repository](https://github.com/pycaret/deployment-heroku) from GitHub. If you don’t know how to fork a repo, please [read this](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) official GitHub tutorial. This is how your project folder should look at this point:

![[https://github.com/pycaret/deployment-heroku](https://github.com/pycaret/deployment-heroku)](https://cdn-images-1.medium.com/max/2524/1*D2LzCUWv5au7AI6dsgGjUA.png)

Now that we have a fully functional web application, we can start the process of containerizing the app using Docker.

# 10-steps to deploy a ML pipeline in docker container:

# 👉 **Step 1 — Install Docker Desktop for Windows**

You can use Docker Desktop on Mac as well as Windows. Depending on your operating system, you can download the Docker Desktop from [this link](https://docs.docker.com/docker-for-windows/install/). We will be using Docker Desktop for Windows in this tutorial.

![[https://hub.docker.com/editions/community/docker-ce-desktop-windows/](https://hub.docker.com/editions/community/docker-ce-desktop-windows/)](https://cdn-images-1.medium.com/max/2692/1*jVBJIDIUyw9UJbzpv2zbpQ.png)

The easiest way to check if the installation was successful is by opening the command prompt and typing in ‘docker’. It should print the help menu.

![Command prompt](https://cdn-images-1.medium.com/max/2200/1*5XYrNYDi6XlLrmIO4ZNHdQ.png)

# 👉 **Step 2 — Install Kitematic**

Kitematic is an intuitive graphical user interface (GUI) for running Docker containers on Windows or Mac. You can download Kitematic from [Docker’s GitHub repository](https://github.com/docker/kitematic/releases).

![[https://github.com/docker/kitematic/releases](https://github.com/docker/kitematic/releases)](https://cdn-images-1.medium.com/max/2508/1*Tl5M7tNVH8smsnkaihxfpA.png)

Once downloaded, simply unzip the file into the desired location.

# 👉 Step 3 — Create a Dockerfile

The first step of creating a Docker image is to create a Dockerfile. A Dockerfile is just a file with a set of instructions. The Dockerfile for this project looks like this:

 <iframe src="https://medium.com/media/44a998312ca7131c53f8b7eaef9a47e3" frameborder=0></iframe>

Dockerfile is case-sensitive and must be in the project folder with the other project files. A Dockerfile has no extension and can be created using any editor. We have used [Visual Studio Code](https://code.visualstudio.com/) to create it.

# 👉 Step 4— Create Azure Container Registry

If you don’t have a Microsoft Azure account or haven’t used it before, you can [sign up](https://azure.microsoft.com/en-ca/free/search/?&ef_id=EAIaIQobChMIm8Onqp6i6QIViY7ICh2QVA2jEAAYASAAEgK9FvD_BwE:G:s&OCID=AID2000061_SEM_EAIaIQobChMIm8Onqp6i6QIViY7ICh2QVA2jEAAYASAAEgK9FvD_BwE:G:s&dclid=CK6R8aueoukCFVbJyAoduGYLcQ) for free. When you sign up for the first time you get a free credit for the first 30 days. You can utilize that credit to build and deploy a web app on Azure. Once you sign up, follow these steps:

* Login on [https://portal.azure.com](https://portal.azure.com).

* Click on Create a Resource.

* Search for Container Registry and click on Create.

* Select Subscription, Resource group and Registry name (in our case: **pycaret.azurecr.io** is our registry name)

![[https://portal.azure.com](https://portal.azure.com) → Sign in → Create a Resource → Container Registry](https://cdn-images-1.medium.com/max/2560/1*InmsXcD7yfbeaMMzobwIJQ.png)

# 👉 Step 5— Build Docker Image

Once a registry is created in Azure portal, the first step is to build a docker image using command line. Navigate to the project folder and execute the following code.

    docker build -t pycaret.azurecr.io/pycaret-insurance:latest . 

![Building docker image using anaconda prompt](https://cdn-images-1.medium.com/max/2566/1*6cPcluJCHV8cpgziPXcGzw.png)

* **pycaret.azurecr.io** is the name of the registry that you get when you create a resource on Azure portal.

* **pycaret-insurance** is the name of the image and **latest **is the tag. This can be anything you want.

# 👉 Step 6— Run container from docker image

Now that the image is created we will run a container locally and test the application before we push it to Azure Container Registry. To run the container locally execute the following code:

    docker run -d -p 5000:5000 pycaret.azurecr.io/pycaret-insurance

Once this command is successfully executed it will return an ID of the container created.

![Running docker container locally](https://cdn-images-1.medium.com/max/2566/1*9g7OQNUA_8zLekDdWa3LHQ.png)

# 👉 Step 7 — Test container on your local machine

Open Kitematic and you should be able to see an application up and running.

![Kitematic — A GUI for managing containers on Mac and Windows OS](https://cdn-images-1.medium.com/max/2690/1*CyJZ98AI5q7HbRa__-KWfg.png)

You can see the app in action by going to localhost:5000 in your internet browser. It should open up a web app.

![Application running on local container (localhost:5000)](https://cdn-images-1.medium.com/max/3824/1*wtDSmSt3Nsh1qQP7DC_kBg.png)

Make sure that once you are done with this, you stop the app using Kitematic, otherwise, it will continue to utilize resources on your computer.

# 👉 Step 8— Authenticate Azure Credentials

One final step before you can upload the container onto ACR is to authenticate azure credentials on your local machine. Execute the following code in the command line to do that:

    docker login pycaret.azurecr.io

You will be prompted for a Username and password. The username is the name of your registry (in this example username is “pycaret”). You can find your password under the Access keys of the Azure Container Registry resource you created.

![portal.azure.com → Azure Container Registry → Access keys](https://cdn-images-1.medium.com/max/3792/1*5pEA3466EIedSiPhe9CGcQ.png)

# 👉 Step 9— Push Container onto Azure Container Registry

Now that you have authenticated to ACR, you can push the container you have created to ACR by executing the following code:

    docker push pycaret.azurecr.io/pycaret-insurance:latest

Depending on the size of the container, the push command may take some time to transfer the container to the cloud.

# 👉 Step 10— Create a Azure Web App and see your model in action

To create a web app on Azure, follow these steps:

* Login on [https://portal.azure.com](https://portal.azure.com).

* Click on Create a Resource.

* Search for Web App and click on Create.

* Link your ACR image that you pushed in (step 9 above) to your app.

![portal.azure.com → Web App → Create → Basics](https://cdn-images-1.medium.com/max/2032/1*_4aEC8X867ybKhGrIl-L6A.png)

![portal.azure.com → Web App → Create → Docker](https://cdn-images-1.medium.com/max/2170/1*kcZWeLbntnrUKRWTE7EzkQ.png)

**BOOM!! The app is now up and running on Azure Web Services.**

![https://pycaret-insurance2.azurewebsites.net](https://cdn-images-1.medium.com/max/3812/1*zElHKEUtI_7NiEW6C5Z7dw.png)

**Note:** By the time this story is published, the app from [https://pycaret-insurance2.azurewebsites.net](https://pycaret-insurance2.azurewebsites.net) will be removed to restrict resource consumption.

[**Link to GitHub Repository for this tutorial.](https://github.com/pycaret/pycaret-deployment-azure)**

[**Link to GitHub Repository for Heroku Deployment.](https://www.github.com/pycaret/deployment-heroku) *(without docker)***

# Next Tutorial

In the next tutorial for deploying machine learning pipelines, we will dive deeper into deploying machine learning pipelines using the Kubernetes Service on Google Cloud and Microsoft Azure.

Follow our [LinkedIn](https://www.linkedin.com/company/pycaret/) and subscribe to our [Youtube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g) channel to learn more about PyCaret.

# Important Links

[User Guide / Documentation](https://www.pycaret.org/guide)
[GitHub Repository
](https://www.github.com/pycaret/pycaret)[Install PyCaret](https://www.pycaret.org/install)
[Notebook Tutorials](https://www.pycaret.org/tutorial)
[Contribute in PyCaret](https://www.pycaret.org/contribute)

# PyCaret 1.0.1 is coming!

We have received overwhelming support and feedback from the community. We are actively working on improving PyCaret and preparing for our next release. **PyCaret 1.0.1 will be bigger and better**. If you would like to share your feedback and help us improve further, you may [fill this form](https://www.pycaret.org/feedback) on the website or leave a comment on our [GitHub ](https://www.github.com/pycaret/)or [LinkedIn](https://www.linkedin.com/company/pycaret/) page.

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

# Would you like to contribute?

PyCaret is an open source project. Everybody is welcome to contribute. If you would like contribute, please feel free to work on [open issues](https://github.com/pycaret/pycaret/issues). Pull requests are accepted with unit tests on dev-1.0.1 branch.

Please give us ⭐️ on our [GitHub repo](https://www.github.com/pycaret/pycaret) if you like PyCaret.

Medium : [https://medium.com/@moez_62905/](https://medium.com/@moez_62905/machine-learning-in-power-bi-using-pycaret-34307f09394a)

LinkedIn : [https://www.linkedin.com/in/profile-moez/](https://www.linkedin.com/in/profile-moez/)

Twitter : [https://twitter.com/moezpycaretorg1](https://twitter.com/moezpycaretorg1)
