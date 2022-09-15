
# How to implement Clustering in Power BI using PyCaret

# by Moez Ali

![Clustering Dashboard in Power BI](https://cdn-images-1.medium.com/max/2632/1*sUeqYcENVII1RlyYA_-Uxg.png)

In our [last post](https://towardsdatascience.com/build-your-first-anomaly-detector-in-power-bi-using-pycaret-2b41b363244e), we demonstrated how to build an anomaly detector in Power BI by integrating it with PyCaret, thus allowing analysts and data scientists to add a layer of machine learning to their reports and dashboards without any additional license costs.

In this post, we will see how we can implement Clustering Analysis in Power BI using PyCaret. If you haven’t heard about PyCaret before, please read this [announcement](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46) to learn more.

# Learning Goals of this Tutorial

* What is Clustering? Types of Clustering.

* Train and implement an unsupervised Clustering model in Power BI.

* Analyze results and visualize information in a dashboard.

* How to deploy the Clustering model in Power BI production?

# Before we start

If you have used Python before, it is likely that you already have Anaconda Distribution installed on your computer. If not, [click here](https://www.anaconda.com/distribution/) to download Anaconda Distribution with Python 3.7 or greater.

![[https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)](https://cdn-images-1.medium.com/max/2612/1*sMceDxpwFVHDtdFi528jEg.png)

# Setting up the Environment

Before we start using PyCaret’s machine learning capabilities in Power BI we have to create a virtual environment and install pycaret. It’s a three-step process:

[✅](https://fsymbols.com/signs/tick/) **Step 1 — Create an anaconda environment**

Open **Anaconda Prompt **from start menu and execute the following code:

    conda create --name **myenv** python=3.7

[✅](https://fsymbols.com/signs/tick/) **Step 2 — Install PyCaret**

Execute the following code in Anaconda Prompt:

    pip install pycaret

Installation may take 15–20 minutes. If you are having issues with installation, please see our [GitHub](https://www.github.com/pycaret/pycaret) page for known issues and resolutions.

[✅](https://fsymbols.com/signs/tick/)**Step 3 — Set Python Directory in Power BI**

The virtual environment created must be linked with Power BI. This can be done using Global Settings in Power BI Desktop (File → Options → Global → Python scripting). Anaconda Environment by default is installed under:

C:\Users\***username***\AppData\Local\Continuum\anaconda3\envs\myenv

![File → Options → Global → Python scripting](https://cdn-images-1.medium.com/max/2000/1*zQMKuyEk8LGrOPE-NByjrg.png)

# What is Clustering?

Clustering is a technique that groups data points with similar characteristics. These groupings are useful for exploring data, identifying patterns and analyzing a subset of data. Organising data into clusters helps in identify underlying structures in the data and finds applications across many industries. Some common business use cases for clustering are:

✔ Customer segmentation for the purpose of marketing.

✔ Customer purchasing behavior analysis for promotions and discounts.

✔ Identifying geo-clusters in an epidemic outbreak such as COVID-19.

# Types of Clustering

Given the subjective nature of clustering tasks, there are various algorithms that suit different types of problems. Each algorithm has its own rules and the mathematics behind how clusters are calculated.

This tutorial is about implementing a clustering analysis in Power BI using a Python library called PyCaret. Discussion of the specific algorithmic details and mathematics behind these algorithms are out-of-scope for this tutorial.

![Ghosal A., Nandy A., Das A.K., Goswami S., Panday M. (2020) A Short Review on Different Clustering Techniques and Their Applications.](https://cdn-images-1.medium.com/max/2726/1*2eQuIebjtTMJot27bWXgCQ.png)

In this tutorial we will use a K-Means algorithm which is one of the simplest and most popular unsupervised machine learning algorithms. If you would like to learn more about K-Means, you can read [this paper](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html).

# Setting the Business Context

In this tutorial we will use the current health expenditure dataset from the World Health Organization’s*** ***Global Health Expenditure database. The dataset contains health expenditure as a % of National GDP for over 200 countries from year 2000 through 2017.

Our objective is to find patterns and groups in this data by using a K-Means clustering algorithm.

[Source Data](https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS)

![Sample Data points](https://cdn-images-1.medium.com/max/2366/1*E1z19x_qa7rko1FZpAw61Q.png)

# 👉 Let’s get started

Now that you have set up the Anaconda Environment, installed PyCaret, understand the basics of Clustering Analysis and have the business context for this tutorial, let’s get started.

# 1. Get Data

The first step is importing the dataset into Power BI Desktop. You can load the data using a web connector. (Power BI Desktop → Get Data → From Web).

![Power BI Desktop → Get Data → Other → Web](https://cdn-images-1.medium.com/max/3842/1*JZ3MwRe8rJXp5e0ac7lamw.png)

Link to csv file: 
[https://github.com/pycaret/powerbi-clustering/blob/master/clustering.csv](https://github.com/pycaret/powerbi-clustering/blob/master/clustering.csv)

# 2. Model Training

To train a clustering model in Power BI we will have to execute a Python script in Power Query Editor (Power Query Editor → Transform → Run python script). Run the following code as a Python script:

    from **pycaret.clustering** import *
    dataset = **get_clusters**(dataset, num_clusters=5, ignore_features=['Country'])

![Power Query Editor (Transform → Run python script)](https://cdn-images-1.medium.com/max/2000/1*SK0XxzF9XZlwtGH1786OUQ.png)

We have ignored the ‘*Country*’ column in the dataset using the **ignore_features** parameter. There could be many reasons for which you might not want to use certain columns for training a machine learning algorithm.

PyCaret allows you to hide instead of drop unneeded columns from a dataset as you might require those columns for later analysis. For example, in this case we don’t want to use ‘Country’ for training an algorithm and hence we have passed it under **ignore_features.**

There are over 8 ready-to-use clustering algorithms available in PyCaret.

![](https://cdn-images-1.medium.com/max/2632/1*ihezKFr61Vrgu7E-0-JA5g.png)

By default, PyCaret trains a **K-Means Clustering model **with 4 clusters. Default values can be changed easily:

* To change the model type use the ***model ***parameter within **get_clusters()**.

* To change the cluster number, use the ***num_clusters ***parameter.

See the example code for **K-Modes Clustering** with 6 clusters.

    from **pycaret.clustering **import *
    dataset = **get_clusters**(dataset, model='kmodes', num_clusters=6, ignore_features=['Country'])

**Output:**

![Clustering Results (after execution of Python code)](https://cdn-images-1.medium.com/max/2000/1*RCYtFO6XDGI2-qbZdYeMfQ.png)

![Final Output (after clicking on Table)](https://cdn-images-1.medium.com/max/3848/1*a6mAzuXC8Ta6gRyolaF5uA.png)

A new column which contains the cluster label is attached to the original dataset. All the year columns are then *unpivoted *to normalize the data so it can be used for visualization in Power BI.

Here’s how the final output looks like in Power BI.

![Results in Power BI Desktop (after applying query)](https://cdn-images-1.medium.com/max/2564/1*oy_X3VIdVPS32qQxkOeehw.png)

# 3. Dashboard

Once you have cluster labels in Power BI, here’s an example of how you can visualize it in dashboard to generate insights:

![Summary page of Dashboard](https://cdn-images-1.medium.com/max/2632/1*sUeqYcENVII1RlyYA_-Uxg.png)

![Details page of Dashboard](https://cdn-images-1.medium.com/max/2632/1*1ck--1zR_hRPqREKDC7ztg.png)

You can download the PBIX file and the data set from our [GitHub](https://github.com/pycaret/powerbi-clustering).

# 👉 Implementing Clustering in Production

What has been demonstrated above was one simple way to implement Clustering in Power BI. However, it is important to note that the method shown above trains the clustering model every time the Power BI dataset is refreshed. This may be a problem for two reasons:

* When the model is re-trained with new data, the cluster labels may change (eg: some data points that were labeled as Cluster 1 earlier might be labelled as Cluster 2 when re-trained)

* You don’t want to spend hours of time everyday re-training the model.

A more productive way to implement clustering in Power BI is to use a pre-trained model for generating cluster labels instead of re-training the model every time.

# Training Model before-hand

You can use any Integrated Development Environment (IDE)or Notebook for training machine learning models. In this example, we have used Visual Studio Code to train a clustering model.

![Model Training in Visual Studio Code](https://cdn-images-1.medium.com/max/2000/1*5roevyCmjxWthy0bYyf4ow.png)

A trained model is then saved as a pickle file and imported into Power Query for generating cluster labels.

![Clustering Pipeline saved as a pickle file](https://cdn-images-1.medium.com/max/2000/1*XxknQxv_O_Cx1WJ4kzwPkQ.png)

If you would like to learn more about implementing Clustering Analysis in Jupyter notebook using PyCaret, watch this 2 minute video tutorial:

 <iframe src="https://medium.com/media/ac70d2254314877ee7e9e524e1f2b1bf" frameborder=0></iframe>

# Using the pre-trained model

Execute the below code as a Python script to generate labels from the pre-trained model.

    from **pycaret.clustering **import *
    dataset = **predict_model**('c:/.../clustering_deployment_20052020, data = dataset)

The output of this will be the same as the one we saw above. The difference is that when you use a pre-trained model, the label is generated on a new dataset using the same model instead of re-training the model.

# Making it work on Power BI Service

Once you’ve uploaded the .pbix file to the Power BI service, a couple more steps are necessary to enable seamless integration of the machine learning pipeline into your data pipeline. These include:

* **Enable scheduled refresh for the dataset** — to enable a scheduled refresh for the workbook that contains your dataset with Python scripts, see [Configuring scheduled refresh](https://docs.microsoft.com/en-us/power-bi/connect-data/refresh-scheduled-refresh), which also includes information about **Personal Gateway**.

* **Install the Personal Gateway** — you need a **Personal Gateway** installed on the machine where the file is located, and where Python is installed; the Power BI service must have access to that Python environment. You can get more information on how to [install and configure Personal Gateway](https://docs.microsoft.com/en-us/power-bi/connect-data/service-gateway-personal-mode).

If you are Interested in learning more about Clustering Analysis, checkout our [Notebook Tutorial](https://www.pycaret.org/clu101).

# PyCaret 1.0.1 is coming!

We have received overwhelming support and feedback from the community. We are actively working on improving PyCaret and preparing for our next release. **PyCaret 1.0.1 will be bigger and better**. If you would like to share your feedback and help us improve further, you may [fill this form](https://www.pycaret.org/feedback) on the website or leave a comment on our [GitHub ](https://www.github.com/pycaret/)or [LinkedIn](https://www.linkedin.com/company/pycaret/) page.

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

# Would you like to contribute?

PyCaret is an open source project. Everybody is welcome to contribute. If you would like to contribute, please feel free to work on [open issues](https://github.com/pycaret/pycaret/issues). Pull requests are accepted with unit tests on dev-1.0.1 branch.

Please give us ⭐️ on our [GitHub repo](https://www.github.com/pycaret/pycaret) if you like PyCaret.

Medium : [https://medium.com/@moez_62905/](https://medium.com/@moez_62905/machine-learning-in-power-bi-using-pycaret-34307f09394a)

LinkedIn : [https://www.linkedin.com/in/profile-moez/](https://www.linkedin.com/in/profile-moez/)

Twitter : [https://twitter.com/moezpycaretorg1](https://twitter.com/moezpycaretorg1)
