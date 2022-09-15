
# Deploy Machine Learning Pipeline on Google Kubernetes Engine

# by Moez Ali

![A step-by-step beginner’s guide to containerize and deploy ML pipeline on Google Kubernetes Engine](https://cdn-images-1.medium.com/max/2000/1*P-JjI7MXq6UJV9Xab-B9qg.png)

# RECAP

In our [last post](https://towardsdatascience.com/deploy-machine-learning-pipeline-on-cloud-using-docker-container-bec64458dc01) on deploying a machine learning pipeline in the cloud, we demonstrated how to develop a machine learning pipeline in PyCaret, containerize it with Docker and serve as a web app using Microsoft Azure Web App Services. If you haven’t heard about PyCaret before, please read this [announcement](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46) to learn more.

In this tutorial, we will use the same machine learning pipeline and Flask app that we built and deployed previously. This time we will demonstrate how to containerize and deploy a machine learning pipeline on Google Kubernetes Engine.

# 👉 Learning Goals of this Tutorial

* Learn what is a Container, what is Docker, what is Kubernetes, and what is Google Kubernetes Engine?

* Build a Docker image and upload it on Google Container Registry (GCR).

* Create clusters and deploy a machine learning pipeline with a Flask app as a web service.

* See a web app in action that uses a trained machine learning pipeline to predict new data points in real-time.

Previously we demonstrated [how to deploy a ML pipeline on Heroku PaaS](https://towardsdatascience.com/build-and-deploy-your-first-machine-learning-web-app-e020db344a99) and [how to deploy a ML pipeline on Azure Web Services with a Docker container.](https://towardsdatascience.com/deploy-machine-learning-pipeline-on-cloud-using-docker-container-bec64458dc01)

This tutorial will cover the entire workflow starting from building a docker image, uploading it onto Google Container Registry and then deploying the pre-trained machine learning pipeline and Flask app onto Google Kubernetes Engine (GKE).

# 💻 Toolbox for this tutorial

# PyCaret

[PyCaret](https://www.pycaret.org/) is an open source, low-code machine learning library in Python that is used to train and deploy machine learning pipelines and models into production. PyCaret can be installed easily using pip.

    pip install pycaret

# Flask

[Flask](https://flask.palletsprojects.com/en/1.1.x/) is a framework that allows you to build web applications. A web application can be a commercial website, blog, e-commerce system, or an application that generates predictions from data provided in real-time using trained models. If you don’t have Flask installed, you can use pip to install it.

# Google Cloud Platform

Google Cloud Platform (GCP), offered by Google, is a suite of cloud computing services that runs on the same infrastructure that Google uses internally for its end-user products, such as Google Search, Gmail and YouTube. If you do not have an account with GCP, you can sign-up [here](https://console.cloud.google.com/getting-started). If you are signing up for the first time you will get free credits for 1 year.

# Let’s get started.

Before we get into Kubernetes, let’s understand what a container is and why we would need one?

![[https://www.freepik.com/free-photos-vectors/cargo-ship](https://www.freepik.com/free-photos-vectors/cargo-ship)](https://cdn-images-1.medium.com/max/2000/1*SlzsvRhA71oFOhAjE1Hs0A.jpeg)

Have you ever had the problem where your code works fine on your computer but when a friend tries to run the exact same code, it doesn’t work? If your friend is repeating the exact same steps, he or she should get the same results, right? The one-word answer to this is ***the environment*. **Your friend’s environment is different than yours.

What does an environment include? → Programing language such as Python and all the libraries and dependencies with the exact versions using which application was built and tested.

If we can create an environment that we can transfer to other machines (for example: your friend’s computer or a cloud service provider like Google Cloud Platform), we can reproduce the results anywhere. Hence, ***a ****container ***is a type of software that packages up an application and all its dependencies so the application runs reliably from one computing environment to another.
>  What’s Docker then?

![](https://cdn-images-1.medium.com/max/2000/1*EJx9QN4ENSPKZuz51rC39w.png)

Docker is a company that provides software (also called Docker) that allows users to build, run and manage containers. While Docker’s container are the most common, there are other less famous *alternatives* such as [LXD](https://linuxcontainers.org/lxd/introduction/) and [LXC](https://linuxcontainers.org/) that provides container solution.

Now that you understand containers and docker specifically, let’s understand what Kubernetes is all about.

# What is Kubernetes?

Kubernetes is a powerful open-source system developed by Google back in 2014, for managing containerized applications. In simple words, Kubernetes ****is a system for running and coordinating containerized applications across a cluster of machines. It is a platform designed to completely manage the life cycle of containerized applications.

![Photo by [chuttersnap](https://unsplash.com/@chuttersnap?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/23216/0*2ZayMwt1Un8-9ZFA)

# Features

✔️ **Load Balancing: **Automatically distributes the load between containers.

✔️ **Scaling: **Automatically scale up or down by adding or removing containers when demand changes such as peak hours, weekends and holidays.

✔️ **Storage: **Keeps storage consistent with multiple instances of an application.

✔️ **Self-healing** Automatically restarts containers that fail and kills containers that don’t respond to your user-defined health check.

✔️ **Automated Rollouts **you can automate Kubernetes to create new containers for your deployment, remove existing containers and adopt all of their resources to the new container.

# Why do you need Kubernetes if you have Docker?

Imagine a scenario where you have to run multiple docker containers on multiple machines to support an enterprise level ML application with varied workloads during day and night. As simple as it may sound, it is a lot of work to do manually.

You need to start the right containers at the right time, figure out how they can talk to each other, handle storage considerations, and deal with failed containers or hardware. This is the problem Kubernetes is solving by allowing large numbers of containers to work together in harmony, reducing the operational burden.
> # It’s a mistake to compare **Docker with Kubernetes. **These are two different technologies. Docker is a software that allows you to containerize applications while Kubernetes is a container management system that allows to create, scale and monitor hundreds and thousands of containers.

In the lifecycle of any application, Docker is used for packaging the application at the time of deployment, while kubernetes is used for rest of the life for managing the application.

![Lifecycle of an application deployed through Kubernetes / Docker](https://cdn-images-1.medium.com/max/3200/1*dBJjxZrfdMppXhdwjZLX6w.png)

# What is Google Kubernetes Engine?

Google Kubernetes Engine is implementation of *Google’s open source Kubernetes* on Google Cloud Platform. Simple!

Other popular alternatives to GKE are [Amazon ECS](https://aws.amazon.com/ecs/) and [Microsoft Azure Kubernetes Service](https://azure.microsoft.com/en-us/services/kubernetes-service/).

# One final time, do you understand this?

* **A Container **is a type of software that packages up an application and all its dependencies so the application runs reliably from one computing environment to another.

* **Docker **is a software used for building and managing containers.

* **Kubernetes **is an open-source system for managing containerized applications in a clustered environment.

* **Google Kubernetes Engine** is an implementation of the open source Kubernetes framework on Google Cloud Platform.

In this tutorial we will use Google Kubernetes Engine. In order to follow along, you must have a Google Cloud Platform account. [Click here](https://console.cloud.google.com/getting-started) to sign-up for free.

# Setting the Business Context

An insurance company wants to improve its cash flow forecasting by better predicting patient charges using demographic and basic patient health risk metrics at the time of hospitalization.

![](https://cdn-images-1.medium.com/max/2000/1*qM1HiWZ_uigwdcZ0_ZL6yA.png)

*([data source](https://www.kaggle.com/mirichoi0218/insurance#insurance.csv))*

# Objective

To build and deploy a web application where the demographic and health information of a patient is entered into a web-based form which then outputs a predicted charge amount.

# Tasks

* Train and develop a machine learning pipeline for deployment.

* Build a web app using a Flask framework. It will use the trained ML pipeline to generate predictions on new data points in real-time.

* Build a docker image and upload a container onto Google Container Registry (GCR).

* Create clusters and deploy the app on Google Kubernetes Engine.

Since we have already covered the first two tasks in our initial tutorial, we will quickly recap them and then focus on the remaining items in the list above. If you are interested in learning more about developing a machine learning pipeline in Python using PyCaret and building a web app using a Flask framework, please read [this tutorial](https://towardsdatascience.com/build-and-deploy-your-first-machine-learning-web-app-e020db344a99).

# 👉 Develop a Machine Learning Pipeline

We are using PyCaret in Python for training and developing a machine learning pipeline which will be used as part of our web app. The Machine Learning Pipeline can be developed in an Integrated Development Environment (IDE) or Notebook. We have used a notebook to run the below code:

 <iframe src="https://medium.com/media/ab682376970c812cbe30e75c0ae3a370" frameborder=0></iframe>

When you save a model in PyCaret, the entire transformation pipeline based on the configuration defined in the **setup() **function is created . All inter-dependencies are orchestrated automatically. See the pipeline and model stored in the ‘deployment_28042020’ variable:

![Machine Learning Pipeline created using PyCaret](https://cdn-images-1.medium.com/max/2000/1*P7EXfIxqZZGrpeLgDdk1vQ.png)

# 👉 Build a Web Application

This tutorial is not focused on building a Flask application. It is only discussed here for completeness. Now that our machine learning pipeline is ready we need a web application that can connect to our trained pipeline to generate predictions on new data points in real-time. We have created the web application using Flask framework in Python. There are two parts of this application:

* Front-end (designed using HTML)

* Back-end (developed using Flask)

This is how our web application looks:

![Web application on local machine](https://cdn-images-1.medium.com/max/3780/1*tc_6S8NztYKB85rPUJd1uQ.png)

If you haven’t followed along so far, no problem. You can simply fork this [repository](https://github.com/pycaret/pycaret-deployment-google) from GitHub. This is how your project folder should look at this point:

![[https://www.github.com/pycaret/pycaret-deployment-google](https://www.github.com/pycaret/pycaret-deployment-google)](https://cdn-images-1.medium.com/max/3796/1*CcId22jB-BMCen8o1hWNdQ.png)

Now that we have a fully functional web application, we can start the process of containerizing and deploying the app on Google Kubernetes Engine.

# 10-steps to deploy a ML pipeline on Google Kubernetes Engine:

# 👉 Step 1 — Create a new project in GCP Console

Sign-in to your GCP console and go to Manage Resources

![Google Cloud Platform Console → Manage Resources](https://cdn-images-1.medium.com/max/3832/1*OS16COUUns7uBnpyUxH9-w.png)

Click on **Create New Project**

![Google Cloud Platform Console → Manage Resources → Create New Project](https://cdn-images-1.medium.com/max/3834/1*QJz8fITeJJWP44yPm2v4vQ.png)

# 👉 Step 2 — Import Project Code

Click the **Activate Cloud Shell **button at the top of the console window to open the Cloud Shell.

![Google Cloud Platform (Project Info Page)](https://cdn-images-1.medium.com/max/3834/1*Mbcd4RlkCcz98Pbf4KSUAA.png)

Execute the following code in Cloud Shell to clone the GitHub repository used in this tutorial.

    git clone https://github.com/pycaret/pycaret-deployment-google.git

![git clone [https://github.com/pycaret/pycaret-deployment-google.git](https://github.com/pycaret/pycaret-deployment-google.git)](https://cdn-images-1.medium.com/max/3838/1*g_RQ30jDG4UsyS84mh-qrw.png)

# 👉 Step 3— Set Project ID Environment Variable

Execute the following code to set the PROJECT_ID environment variable.

    export PROJECT_ID=**pycaret-kubernetes-demo**

*pycaret-kubernetes-demo* is the name of the project we chose in step 1 above.

# 👉 Step 4— Build the docker image

Build the docker image of the application and tag it for uploading by executing the following code:

    docker build -t gcr.io/${PROJECT_ID}/insurance-app:v1 .

![Message returned when docker build is successful](https://cdn-images-1.medium.com/max/3834/1*Zo7_W7pG6JhFvHbzyQeEsA.png)

You can check the available images by running the following code:

    docker images

![Output of “docker images” command on Cloud Shell](https://cdn-images-1.medium.com/max/3834/1*0paobe_W8tmdCF1xhX4BgA.png)

# 👉 Step 5— Upload the container image

 1. Authenticate to [Container Registry](https://cloud.google.com/container-registry) (you need to run this only once):

    gcloud auth configure-docker

2. Execute the following code to upload the docker image to Google Container Registry:

    docker push gcr.io/${PROJECT_ID}/insurance-app:v1

# 👉 Step 6— Create Cluster

Now that the container is uploaded, you need a cluster to run the container. A cluster consists of a pool of Compute Engine VM instances, running Kubernetes.

 1. Set your project ID and Compute Engine zone options for the gcloud tool:

    gcloud config set project $PROJECT_ID 
    gcloud config set compute/zone **us-central1**

2. Create a cluster by executing the following code:

    gcloud container clusters create **insurance-cluster** --num-nodes=2

![Google Cloud Platform → Kubernetes Engine → Clusters](https://cdn-images-1.medium.com/max/3832/1*l2sHrv5nuFjDKiyAtjYapQ.png)

# 👉 Step 7— Deploy Application

To deploy and manage applications on a GKE cluster, you must communicate with the Kubernetes cluster management system. Execute the following command to deploy the application:

    kubectl create deployment insurance-app --image=gcr.io/${PROJECT_ID}/insurance-app:v1

![Output returned on creating deployment through kubectl](https://cdn-images-1.medium.com/max/3836/1*p0_A6PZnfYJ4mnttM7lzzA.png)

# 👉 Step 8— Expose your application to the internet

By default, the containers you run on GKE are not accessible from the internet because they do not have external IP addresses. Execute the following code to expose the application to the internet:

    kubectl expose deployment insurance-app --type=LoadBalancer --port 80 --target-port 8080

# 👉 Step 9— Check Service

Execute the following code to get the status of the service. **EXTERNAL-IP** is the web address you can use in browser to view the published app.

    kubectl get service

![Cloud Shell → kubectl get service](https://cdn-images-1.medium.com/max/3832/1*aRWl7frtmvPYaYjAoloFgQ.png)

👉 Step 10— See the app in action on [http://34.71.77.61:8080](http://34.71.77.61:8080)

![Final app uploaded on [http://34.71.77.61:8080](http://34.71.77.61:8080)](https://cdn-images-1.medium.com/max/3838/1*bKuZiYSPdE8T5SLKXx5B_Q.png)

**Note:** By the time this story is published, the app will be removed from the public address to restrict resource consumption.

[Link to GitHub Repository for this tutorial](https://www.github.com/pycaret/pycaret-deployment-google)

[Link to GitHub Repository for Microsoft Azure Deployment](https://www.github.com/pycaret/pycaret-azure-deployment)

[Link to GitHub Repository for Heroku Deployment](https://www.github.com/pycaret/deployment-heroku)

# PyCaret 1.0.1 is coming!

We have received overwhelming support and feedback from the community. We are actively working on improving PyCaret and preparing for our next release. **PyCaret 1.0.1 will be bigger and better**. If you would like to share your feedback and help us improve further, you may [fill this form](https://www.pycaret.org/feedback) on the website or leave a comment on our [GitHub ](https://www.github.com/pycaret/)or [LinkedIn](https://www.linkedin.com/company/pycaret/) page.

Follow our [LinkedIn](https://www.linkedin.com/company/pycaret/) and subscribe to our [YouTube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g) channel to learn more about PyCaret.

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
