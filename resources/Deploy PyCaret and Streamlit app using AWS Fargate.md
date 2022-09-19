
# Deploy PyCaret and Streamlit app using AWS Fargate — serverless infrastructure

# by Moez Ali

![A step-by-step beginner’s guide to containerize and deploy ML pipeline serverless on AWS Fargate](https://cdn-images-1.medium.com/max/2000/1*QznGlPsGrGQS4DadTunLXw.png)

# RECAP

In our [last post](https://towardsdatascience.com/deploy-machine-learning-app-built-using-streamlit-and-pycaret-on-google-kubernetes-engine-fd7e393d99cb), we demonstrated how to develop a machine learning pipeline using PyCaret and serve it as a Streamlit web application deployed onto Google Kubernetes Engine. If you haven’t heard about PyCaret before, you can read this [announcement](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46) to get started.

In this tutorial, we will use the same web app and machine learning pipeline that we had built previously and demonstrate how to deploy it using AWS Fargate which is a serverless compute for containers.

By the end of this tutorial, you will be able to build and host a fully functional containerized web app on AWS without provisioning any server infrastructure.

![Web Application](https://cdn-images-1.medium.com/max/2800/1*TesAmfCyanOeMEPiYxInUg.png)

# 👉 Learning Goals of this Tutorial

* What is a Container? What is Docker? What is Kubernetes?

* What is Amazon Elastic Container Service (ECS), AWS Fargate and serverless deployment?

* Build and push a Docker image onto Amazon Elastic Container Registry.

* Deploy web app using serverless infrastructure i.e. AWS Fargate.

This tutorial will cover the entire workflow starting from building a docker image locally, uploading it onto Amazon Elastic Container Registry, creating a cluster and then defining and executing task using AWS-managed infrastructure.

In the past, we have covered deployment on other cloud platforms such as Azure and Google. If you are interested in learning more about those, you can read the following tutorials:

* [Deploy Streamlit app onto Google Kubernetes Engine](https://towardsdatascience.com/deploy-machine-learning-app-built-using-streamlit-and-pycaret-on-google-kubernetes-engine-fd7e393d99cb)

* [Build and deploy machine learning web app using PyCaret and Streamlit](https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104)

* [Deploy Machine Learning Pipeline on AWS Fargate](https://towardsdatascience.com/deploy-machine-learning-pipeline-on-aws-fargate-eb6e1c50507)

* [Deploy Machine Learning Pipeline on Google Kubernetes Engine](https://towardsdatascience.com/deploy-machine-learning-model-on-google-kubernetes-engine-94daac85108b)

* [Deploy Machine Learning Pipeline on AWS Web Service](https://towardsdatascience.com/deploy-machine-learning-pipeline-on-cloud-using-docker-container-bec64458dc01)

* [Build and deploy your first machine learning web app on Heroku PaaS](https://towardsdatascience.com/build-and-deploy-your-first-machine-learning-web-app-e020db344a99)

# 💻 Toolbox for this tutorial

# PyCaret

[PyCaret](https://www.pycaret.org/) is an open source, low-code machine learning library in Python that is used to train and deploy machine learning pipelines and models into production. PyCaret can be installed easily using pip.

    pip install pycaret

# Streamlit

[Streamlit](https://www.streamlit.io/) is an open-source Python library that makes it easy to build beautiful custom web-apps for machine learning and data science. Streamlit can be installed easily using pip.

    pip install streamlit

# Docker Toolbox for Windows 10 Home

[Docker](https://www.docker.com/)** **is a tool designed to make it easier to create, deploy, and run applications by using containers. Containers are used to package up an application with all of its necessary components, such as libraries and other dependencies, and ship it all out as one package. If you haven’t used docker before, this tutorial also covers the installation of Docker Toolbox (legacy) on **Windows 10 Home**. In the [previous tutorial](https://towardsdatascience.com/deploy-machine-learning-pipeline-on-cloud-using-docker-container-bec64458dc01) we covered how to install Docker Desktop on **Windows 10 Pro edition**.

# Amazon Web Services (AWS)

Amazon Web Services (AWS) is a comprehensive and broadly adopted cloud platform, offered by Amazon. It has over 175 fully-featured services from data centers globally. If you haven’t used AWS before, you can [sign-up](https://aws.amazon.com/) for a free account.

# ✔️Let’s get started…..

# What is a Container?

Before we get into implementation using AWS Fargate, let’s understand what a container is and why we would need one?

![[https://www.freepik.com/free-photos-vectors/cargo-ship](https://www.freepik.com/free-photos-vectors/cargo-ship)](https://cdn-images-1.medium.com/max/2000/1*SlzsvRhA71oFOhAjE1Hs0A.jpeg)

Have you ever had the problem where your code works fine on your computer but when a friend tries to run the exact same code, it doesn’t work? If your friend is repeating the exact same steps, they should get the same results, right? The one-word answer to this is ***the environment*. **Your friend’s environment is different than yours.

What does an environment include? → The programing language such as Python and all the libraries and dependencies with the exact versions using which application was built and tested.

If we can create an environment that we can transfer to other machines (for example: your friend’s computer or a cloud service provider like Google Cloud Platform), we can reproduce the results anywhere. Hence, ***a ****container ***is a type of software that packages up an application and all its dependencies so the application runs reliably from one computing environment to another.

# What is Docker?

Docker is a company that provides software (also called **Docker**) that allows users to build, run and manage containers. While Docker’s container are the most common, there are other less famous *alternatives* such as [LXD](https://linuxcontainers.org/lxd/introduction/) and [LXC](https://linuxcontainers.org/).

![](https://cdn-images-1.medium.com/max/2000/1*EJx9QN4ENSPKZuz51rC39w.png)

Now that you theoretically understand what a container is and how Docker is used to containerize applications, let’s imagine a scenario where you have to run multiple containers across a fleet of machines to support an enterprise level machine learning application with varied workloads during day and night. This is pretty common for real-life and as simple as it may sound, it is a lot of work to do manually.

You need to start the right containers at the right time, figure out how they can talk to each other, handle storage considerations, deal with failed containers or hardware and million other things!

This entire process of managing hundreds and thousands of containers to keep the application up and running is known as **container orchestration**. Don’t get caught up in the technical details yet.

At this point, you must recognize that managing real-life applications require more than one container and managing all of the infrastructure to keep containers up and running is cumbersome, manual and an administrative burden.

This brings us to **Kubernetes**.

# What is Kubernetes?

Kubernetes is an open-source system developed by Google in 2014 for managing containerized applications. In simple words, Kubernetes ****is a system for running and coordinating containerized applications across a cluster of machines.

![Photo by [chuttersnap](https://unsplash.com/@chuttersnap?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/14720/0*vscKcwTh1qNmKv3s)

While Kubernetes is an open-source system developed by Google, almost all major cloud service providers offer Kubernetes as a Managed Service. For example: **Amazon Elastic Kubernetes Service (EKS) **offered by Amazon**, Google Kubernetes Engine (GKE) **offered by Google**, **and **Azure Kubernetes Service (AKS) **offered by Microsoft.

So far we have discussed and understood:

✔️ A ***container***

✔️ Docker

✔️ Kubernetes

Before introducing AWS Fargate, there is only one thing left to discuss and that is Amazon’s own container orchestration service **Amazon Elastic Container Service (ECS).**

# AWS Elastic Container Service (ECS)

Amazon Elastic Container Service (Amazon ECS) is Amazon’s home-grown container orchestration platform. The idea behind ECS is similar to Kubernetes *(both of them are orchestration services)*.

ECS is an AWS-native service, meaning that it is only possible to use on AWS infrastructure. On the other hand, **EKS** is based on Kubernetes, an open-source project which is available to users running on multi-cloud (AWS, GCP, Azure) and even On-Premise.

Amazon also offers a Kubernetes based container orchestration service known as **Amazon Elastic Kubernetes Service (Amazon EKS). **Even though the purpose of ECS and EKS is pretty similar i.e. *orchestrating containerized applications*, there are quite a few differences in pricing, compatibility and security. There is no best answer and the choice of solution depends on the use-case.

Irrespective of whichever container orchestration service you are using (ECS or EKS), there are two ways you can implement the underlying infrastructure:

 1. Manually manage the cluster and underlying infrastructure such as Virtual Machines / Servers / (also known as EC2 instances).

 2. Serverless — Absolutely no need to manage anything. Just upload the container and that’s it. ← **This is AWS Fargate.**

![Amazon ECS underlying infrastructure](https://cdn-images-1.medium.com/max/2798/1*k4famzZ1w2Ee5XMHRo1Ggw.png)

# AWS Fargate — serverless compute for containers

AWS Fargate is a serverless compute engine for containers that works with both Amazon Elastic Container Service (ECS) and Amazon Elastic Kubernetes Service (EKS). Fargate makes it easy for you to focus on building your applications. Fargate removes the need to provision and manage servers, lets you specify and pay for resources per application, and improves security through application isolation by design.

Fargate allocates the right amount of compute, eliminating the need to choose instances and scale cluster capacity. You only pay for the resources required to run your containers, so there is no over-provisioning and paying for additional servers.

![How AWS Fargate works — [https://aws.amazon.com/fargate/](https://aws.amazon.com/fargate/)](https://cdn-images-1.medium.com/max/4668/1*WWQBLhVao-FN_FCrnkPhQg.png)

There is no best answer as to which approach is better. The choice between going serverless or manually managing an EC2 cluster depends on the use-case. Some pointers that can assist with this choice include:

**ECS EC2 (Manual Approach)**

* You are all-in on AWS.

* You have a dedicated Ops team in place to manage AWS resources.

* You have an existing footprint on AWS i.e. you are already managing EC2 instances

**AWS Fargate**

* You do not have huge Ops team to manage AWS resources.

* You do not want operational responsibility or want to reduce it.

* Your application is stateless *(A stateless app is an application that does not save client data generated in one session for use in the next session with that client)*.

# Setting the Business Context

An insurance company wants to improve its cash flow forecasting by better predicting patient charges using demographic and basic patient health risk metrics at the time of hospitalization.

![](https://cdn-images-1.medium.com/max/2000/1*qM1HiWZ_uigwdcZ0_ZL6yA.png)

*([data source](https://www.kaggle.com/mirichoi0218/insurance#insurance.csv))*

# Objective

To build and deploy a web application where the demographic and health information of a patient is entered into a web-based form which then outputs a predicted charge amount.

# Tasks

* Train, validate and develop a machine learning pipeline using PyCaret.

* Build a front-end web application with two functionalities: (i) online prediction and (ii) batch prediction.

* Create a Dockerfile

* Create and execute a task to deploy the app using AWS Fargate serverless infrastructure.

Since we have already covered the first two tasks in our [last tutorial](https://towardsdatascience.com/deploy-machine-learning-app-built-using-streamlit-and-pycaret-on-google-kubernetes-engine-fd7e393d99cb), we will quickly recap them and then focus on the remaining items in the list above. If you are interested in learning more about developing a machine learning pipeline in Python using PyCaret and building a web app using a Streamlit framework, please read [this tutorial](https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104).

# 👉 Task 1 — Model Training and Validation

We are using PyCaret in Python for training and developing a machine learning pipeline which will be used as part of our web app. The Machine Learning Pipeline can be developed in an Integrated Development Environment (IDE) or Notebook. We have used a notebook to run the below code:

 <iframe src="https://medium.com/media/1b574df07e2e91e6ff9b0ea404b1981e" frameborder=0></iframe>

When you save a model in PyCaret, the entire transformation pipeline based on the configuration defined in the **setup() **function is created . All inter-dependencies are orchestrated automatically. See the pipeline and model stored in the ‘deployment_28042020’ variable:

![Machine Learning Pipeline created using PyCaret](https://cdn-images-1.medium.com/max/2000/1*P7EXfIxqZZGrpeLgDdk1vQ.png)

# 👉 Task 2 — Build a front-end web application

Now that our machine learning pipeline and model are ready to start building a front-end web application that can generate predictions on new datapoints. This application will support ‘Online’ as well as ‘Batch’ predictions through a csv file upload. Let’s breakdown the application code into three main parts:

# Header / Layout

This section imports libraries, loads the trained model and creates a basic layout with a logo on top, a jpg image and a dropdown menu on the sidebar to toggle between ‘Online’ and ‘Batch’ prediction.

![app.py — code snippet part 1](https://cdn-images-1.medium.com/max/2268/1*xAnCZ1N_BNoPW7FoA-NXrA.png)

# Online Predictions

This section deals with the initial app function, Online one-by-one predictions. We are using streamlit widgets such as *number input, text input, drop down menu and checkbox* to collect the datapoints used to train the model such as Age, Sex, BMI, Children, Smoker, Region.

![app.py — code snippet part 2](https://cdn-images-1.medium.com/max/2408/1*eFeq1wINsUUnvLJfuL-GOA.png)

# Batch Predictions

Predictions by batch is the second layer of the app’s functionality. The **file_uploader** widget in streamlit is used to upload a csv file and then called the native **predict_model() **function from PyCaret to generate predictions that are displayed using streamlit’s write() function.

![app.py — code snippet part 3](https://cdn-images-1.medium.com/max/2410/1*u-g2iLy_gV7hom71PM3CEA.png)

**Testing App
**One final step before we deploy the application on AWS Fargate is to test the app locally. Open Anaconda Prompt and navigate to your project folder and execute the following code:

    streamlit run app.py

![Streamlit application testing — Online Prediction](https://cdn-images-1.medium.com/max/2800/1*TesAmfCyanOeMEPiYxInUg.png)

# 👉 Task 3 — Create a Dockerfile

To containerize our application for deployment we need a docker image that becomes a container at runtime. A docker image is created using a Dockerfile. A Dockerfile is just a file with a set of instructions. The Dockerfile for this project looks like this:

 <iframe src="https://medium.com/media/49f930956226e898d453265dab390382" frameborder=0></iframe>

The last part of this Dockerfile (starting at line 23) is Streamlit specific. Dockerfile is case-sensitive and must be in the project folder with the other project files.

# 👉 Task 4–Deploy on AWS Fargate:

Follow these simple 9 steps to deploy app on AWS Fargate:

# 👉 Step 1 — Install Docker Toolbox (for Windows 10 Home)

In order to build a docker image locally, you will need Docker installed on your computer. If you are using Windows 10 64-bit: Pro, Enterprise, or Education (Build 15063 or later) you can download Docker Desktop from [DockerHub](https://hub.docker.com/editions/community/docker-ce-desktop-windows/).

However, if you are using Windows 10 Home, you would need to install the last release of legacy Docker Toolbox (v19.03.1) from [Dockers GitHub page](https://github.com/docker/toolbox/releases).

![[https://github.com/docker/toolbox/releases](https://github.com/docker/toolbox/releases)](https://cdn-images-1.medium.com/max/2000/1*wn3zVxR0d5rZFDkvhEHi1Q.png)

Download and Run **DockerToolbox-19.03.1.exe** file.

The easiest way to check if the installation was successful is by opening the command prompt and typing in ‘docker’. It should print the help menu.

![Anaconda Prompt to check docker](https://cdn-images-1.medium.com/max/2198/1*f5l4Tds3EOTFSPx6CT5M7w.png)

# 👉 Step 2 — Create a Repository in Elastic Container Registry (ECR)

**(a) Login to your AWS console and search for Elastic Container Registry:**

![AWS Console](https://cdn-images-1.medium.com/max/2000/1*XCvjm_Ho1CiaNg59y3MPiw.png)

**(b) Create a new repository:**

![Create New Repository on Amazon Elastic Container Registry](https://cdn-images-1.medium.com/max/3822/1*alFdHEfwYrdZ5J9d14gGgA.png)

![Create Repository](https://cdn-images-1.medium.com/max/3822/1*BeVF99WdFAPApWLS83SJ3Q.png)

Click on “Create Repository”.

**(c) Click on “View push commands”:**

![Push commands for pycaret-streamlit-aws repository](https://cdn-images-1.medium.com/max/2000/1*WC-0ShGuB0MB6LgTq07B0Q.png)

# 👉 Step 3— Execute push commands

Navigate to your project folder using Anaconda Prompt and execute the commands you have copied in the step above. You must be in the folder where the Dockerfile and the rest of your code reside before executing these commands.

These commands are for building docker image and then uploading it on AWS ECR.

# 👉 Step 4 — Check your uploaded image

Click on the repository you created and you will see an image URI of the uploaded image in the step above. Copy the image URI (it would be needed in step 6 below).

![](https://cdn-images-1.medium.com/max/3828/1*VuYsEXDoSmmHlEFfYgOAhg.png)

# 👉 Step 5 — Create and Configure a Cluster

**(a) Click on “Clusters” on left-side menu:**

![Create Cluster — Step 1](https://cdn-images-1.medium.com/max/3834/1*eGOSlysIcdpDZi9GnPAhHw.png)

**(b) Select “Networking only” and click Next step:**

![Select Networking Only Template](https://cdn-images-1.medium.com/max/2000/1*a0VectBKdBhmZC_My5OylQ.png)

**(c) Configure Cluster (Enter cluster name) and click on Create:**

![Configure Cluster](https://cdn-images-1.medium.com/max/3780/1*6AMEaRIr4Rz1qt_ZmhDy4Q.png)

Click on “Create”.

**(d) Cluster Created:**

![Cluster Created](https://cdn-images-1.medium.com/max/3824/1*1UfMJt807V92-jc6Z9ZlfQ.png)

# 👉 Step 6 — Create a new Task definition

A **task** definition is required to run Docker containers in Amazon ECS. Some of the parameters you can specify in a **task** definition include: The Docker image to use with each container in your **task**. How much CPU and memory to use with each **task** or each container within a **task**.

**(a) Click on “Create new task definition”:**

![Create a new task definition](https://cdn-images-1.medium.com/max/3820/1*6ET40juZ2owkA1xdDOsDHg.png)

**(b) Select “FARGATE” as launch type:**

![Select Launch Type Compatibility](https://cdn-images-1.medium.com/max/3822/1*1Ebz8wmfSisxcrultB86nQ.png)

**(c) Fill in the details:**

![Configure Task and container definitions (part 1)](https://cdn-images-1.medium.com/max/2000/1*JqrJPuts4QpVBUK2pKFPpg.png)

![Configure Task and container definitions (part 2)](https://cdn-images-1.medium.com/max/2000/1*SoM892EIZ2NpSzUCUg10rA.png)

**(d) Click on “Add Containers” and fill in the details:**

![Adding Container in task definitions](https://cdn-images-1.medium.com/max/2508/1*Kt9zGo0kk4bAUyrWhedU4Q.png)

Click “Create Task” on the bottom right.

![](https://cdn-images-1.medium.com/max/3828/1*DZpHXH5iy3daszNT4RYoEQ.png)

# 👉 Step 7— Execute Task Definition

In last step we created a task that will start the container. Now we will execute the task by clicking **“Run Task”** under Actions.

![](https://cdn-images-1.medium.com/max/3836/1*nuUekT3eyCeDRoeZlTXk_Q.png)

**(a) Click on “Switch to launch type” to change the type to Fargate:**

![](https://cdn-images-1.medium.com/max/3850/1*_TMuygT58eKgMJQStWCwQw.png)

**(b) Select the VPC and Subnet from the dropdown:**

![](https://cdn-images-1.medium.com/max/3812/1*w7uipHeBNz9RhaBFsN85Bw.png)

Click on “Run Task” on bottom right.

# 👉 Step 8— Allow inbound port 8501 from Network settings

One last step before we can see our application in action on Public IP address is to allow port 8501 (used by streamlit) by creating a new rule. In order to do that, follow these steps:

**(a) Click on Task**

![](https://cdn-images-1.medium.com/max/3834/1*lZh9LgN8vgctY3Xa_aeMrg.png)

**(b) Click on ENI Id:**

![](https://cdn-images-1.medium.com/max/3832/1*K1L_vExR8-2q-6b020voPQ.png)

**(c) Click on Security groups**

![](https://cdn-images-1.medium.com/max/3822/1*vPhVnBMZTXqBQj0ntWx5WA.png)

**(d) Scroll down and click on “Edit inbound rules”**

![](https://cdn-images-1.medium.com/max/3828/1*nWb74Ex5UWs-yJOZs5Ecew.png)

**(e) Add a Custom TCP rule of port 8501**

![](https://cdn-images-1.medium.com/max/3826/1*uqgV_Fr5NPGzWzwQ5LHAxw.png)

# 👉 Congratulations! You have published your app serverless on AWS Fargate. Use public IP address with port 8501 to access the application.

![App published on 99.79.189.46:8501](https://cdn-images-1.medium.com/max/3834/1*q9GXNH-YCL2vT7Q-Uj9clQ.png)

**Note:** By the time this story is published, the app will be removed from the public address to restrict resource consumption.

[Link to GitHub Repository for this tutorial](https://www.github.com/pycaret/pycaret-streamlit-aws)

[Link to GitHub Repository for Google Kubernetes Deployment](https://towardsdatascience.com/deploy-machine-learning-app-built-using-streamlit-and-pycaret-on-google-kubernetes-engine-fd7e393d99cb)

[Link to GitHub Repository for Heroku Deployment](https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104)

# PyCaret 2.0.0 is coming!

We have received overwhelming support and feedback from the community. We are actively working on improving PyCaret and preparing for our next release. **PyCaret 2.0.0 will be bigger and better**. If you would like to share your feedback and help us improve further, you may [fill this form](https://www.pycaret.org/feedback) on the website or leave a comment on our [GitHub ](https://www.github.com/pycaret/)or [LinkedIn](https://www.linkedin.com/company/pycaret/) page.

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

[Classification](https://www.pycaret.org/clf101)
[Regression](https://www.pycaret.org/reg101)
[Clustering](https://www.pycaret.org/clu101)
[Anomaly Detection](https://www.pycaret.org/anom101)
[Natural Language Processing](https://www.pycaret.org/nlp101)
[Association Rule Mining](https://www.pycaret.org/arul101)

# Would you like to contribute?

PyCaret is an open source project. Everybody is welcome to contribute. If you would like to contribute, please feel free to work on [open issues](https://github.com/pycaret/pycaret/issues). Pull requests are accepted with unit tests on dev-1.0.1 branch.

Please give us ⭐️ on our [GitHub repo](https://www.github.com/pycaret/pycaret) if you like PyCaret.

Medium: [https://medium.com/@moez_62905/](https://medium.com/@moez_62905/machine-learning-in-power-bi-using-pycaret-34307f09394a)

LinkedIn: [https://www.linkedin.com/in/profile-moez/](https://www.linkedin.com/in/profile-moez/)

Twitter: [https://twitter.com/moezpycaretorg1](https://twitter.com/moezpycaretorg1)
