
# Topic Modeling in Power BI using PyCaret

# by Moez Ali

![NLP Dashboard in Power BI](https://cdn-images-1.medium.com/max/2624/1*SyZczsDz5Pf-4Srfj_p8vQ.png)

In our [last post](https://towardsdatascience.com/how-to-implement-clustering-in-power-bi-using-pycaret-4b5e34b1405b), we demonstrated how to implement clustering analysis in Power BI by integrating it with PyCaret, thus allowing analysts and data scientists to add a layer of machine learning to their reports and dashboards without any additional license costs.

In this post, we will see how we can implement topic modeling in Power BI using PyCaret. If you haven’t heard about PyCaret before, please read this [announcement](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46) to learn more.

# Learning Goals of this Tutorial

* What is Natural Language Processing?

* What is Topic Modeling?

* Train and implement a Latent Dirichlet Allocation model in Power BI.

* Analyze results and visualize information in a dashboard.

# Before we start

If you have used Python before, it is likely that you already have Anaconda Distribution installed on your computer. If not, [click here](https://www.anaconda.com/distribution/) to download Anaconda Distribution with Python 3.7 or greater.

![[https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)](https://cdn-images-1.medium.com/max/2612/1*sMceDxpwFVHDtdFi528jEg.png)

# Setting up the Environment

Before we start using PyCaret’s machine learning capabilities in Power BI we have to create a virtual environment and install pycaret. It’s a four-step process:

[✅](https://fsymbols.com/signs/tick/) **Step 1 — Create an anaconda environment**

Open **Anaconda Prompt **from start menu and execute the following code:

    conda create --name **powerbi **python=3.7

*“powerbi” is the name of environment we have chosen. You can keep whatever name you would like.*

[✅](https://fsymbols.com/signs/tick/) **Step 2 — Install PyCaret**

Execute the following code in Anaconda Prompt:

    pip install **pycaret**

Installation may take 15–20 minutes. If you are having issues with installation, please see our [GitHub](https://www.github.com/pycaret/pycaret) page for known issues and resolutions.

[✅](https://fsymbols.com/signs/tick/)**Step 3 — Set Python Directory in Power BI**

The virtual environment created must be linked with Power BI. This can be done using Global Settings in Power BI Desktop (File → Options → Global → Python scripting). Anaconda Environment by default is installed under:

C:\Users\***username***\Anaconda3\envs\

![File → Options → Global → Python scripting](https://cdn-images-1.medium.com/max/2000/1*3qTuOM-N6ekhoiQmDpHgXg.png)

[✅](https://fsymbols.com/signs/tick/)**Step 4 — Install Language Model**

In order to perform NLP tasks you must download language model by executing following code in your Anaconda Prompt.

First activate your conda environment in Anaconda Prompt:

    conda activate **powerbi**

Download English Language Model:

    python -m spacy download en_core_web_sm
    python -m textblob.download_corpora

![python -m spacy download en_core_web_sm](https://cdn-images-1.medium.com/max/3840/1*savPqt23x7nBcK76-0MBxw.png)

![python -m textblob.download_corpora](https://cdn-images-1.medium.com/max/3838/1*NYaSehQvRp9ANsEpC_GPQw.png)

# What is Natural Language Processing?

Natural language processing (NLP) is a subfield of computer science and artificial intelligence that is concerned with the interactions between computers and human languages. In particular, NLP covers broad range of techniques on how to program computers to process and analyze large amounts of natural language data.

NLP-powered software helps us in our daily lives in various ways and it is likely that you have been using it without even knowing. Some examples are:

* **Personal assistants**: Siri, Cortana, Alexa.

* **Auto-complete**: In search engines (*e.g:* Google, Bing, Baidu, Yahoo).

* **Spell checking**: Almost everywhere, in your browser, your IDE (*e.g:* Visual Studio), desktop apps (*e.g:* Microsoft Word).

* **Machine Translation**: Google Translate.

* **Document Summarization Software: **Text compactor, Autosummarizer.

![Source: [https://clevertap.com/blog/natural-language-processing](https://clevertap.com/blog/natural-language-processing/)](https://cdn-images-1.medium.com/max/2800/1*IEuGZY5vaWoVnTqQpoZUvQ.jpeg)

Topic Modeling is a type of statistical model used for discovering abstract topics in text data. It is one of many practical applications within NLP.

# What is Topic Modeling?

A topic model is a type of statistical model that falls under unsupervised machine learning and is used for discovering abstract topics in text data. The goal of topic modeling is to automatically find the topics / themes in a set of documents.

Some common use-cases for topic modeling are:

* **Summarizing** large text data by classifying documents into topics (*the idea is pretty similar to clustering*).

* **Exploratory Data Analysis **to gain understanding of data such as customer feedback forms, amazon reviews, survey results etc.

* **Feature Engineering **creating features for supervised machine learning experiments such as classification or regression

There are several algorithms used for topic modeling. Some common ones are Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA), and Non-Negative Matrix Factorization (NMF). Each algorithm has its own mathematical details which will not be covered in this tutorial. We will implement a Latent Dirichlet Allocation (LDA) model in Power BI using PyCaret’s NLP module.

If you are interested in learning the technical details of the LDA algorithm, you can read [this paper](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf).

![Source : [https://springerplus.springeropen.com/articles/10.1186/s40064-016-3252-8](https://springerplus.springeropen.com/articles/10.1186/s40064-016-3252-8)](https://cdn-images-1.medium.com/max/2000/1*DYbV9YMI94QsUeRiiJyrSg.png)

# **Text preprocessing for Topic Modeling**

In order to get meaningful results from topic modeling text data must be processed before feeding it to the algorithm. This is common with almost all NLP tasks. The preprocessing of text is different from the classical preprocessing techniques often used in machine learning when dealing with structured data (data in rows and columns).

PyCaret automatically preprocess text data by applying over 15 techniques such as **stop word removal**, **tokenization**, **lemmatization**, **bi-gram/tri-gram extraction etc**. If you would like to learn more about all the text preprocessing features available in PyCaret, [click here](https://www.pycaret.org/nlp).

# Setting the Business Context

Kiva is an international non-profit founded in 2005 in San Francisco. Its mission is to expand financial access to underserved communities in order to help them thrive.

![Source: [https://www.kiva.org/about](https://www.kiva.org/about)](https://cdn-images-1.medium.com/max/2124/1*U4zzTYo6MoCk6PxuZl3FBw.png)

In this tutorial we will use the open dataset from Kiva which contains loan information on 6,818 approved loan applicants. The dataset includes information such as loan amount, country, gender and some text data which is the application submitted by the borrower.

![Sample Data points](https://cdn-images-1.medium.com/max/3194/1*jnQvTmQHhWpOSAgSMqaspg.png)

Our objective is to analyze the text data in the ‘*en*’ column to find abstract topics and then use them to evaluate the effect of certain topics (or certain types of loans) on the default rate.

# 👉 Let’s get started

Now that you have set up the Anaconda Environment, understand topic modeling and have the business context for this tutorial, let’s get started.

# 1. Get Data

The first step is importing the dataset into Power BI Desktop. You can load the data using a web connector. (Power BI Desktop → Get Data → From Web).

![Power BI Desktop → Get Data → Other → Web](https://cdn-images-1.medium.com/max/3828/1*lGqJEUm2lVDcYNDdGNUfbw.png)

Link to csv file:
[https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/kiva.csv](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/kiva.csv)

# 2. Model Training

To train a topic model in Power BI we will have to execute a Python script in Power Query Editor (Power Query Editor → Transform → Run python script). Run the following code as a Python script:

    from **pycaret.nlp **import *
    dataset = **get_topics**(dataset, text='en')

![Power Query Editor (Transform → Run python script)](https://cdn-images-1.medium.com/max/2000/1*EwC-QI4m6DORCtdakOAPmQ.png)

There are 5 ready-to-use topic models available in PyCaret.

![](https://cdn-images-1.medium.com/max/2000/1*LszI1w45K6i5pOBJ0ZmndA.png)

By default, PyCaret trains a **Latent Dirichlet Allocation (LDA) **model with 4 topics. Default values can be changed easily:

* To change the model type use the ***model ***parameter within **get_topics()**.

* To change the number of topics, use the ***num_topics ***parameter.

See the example code for a **Non-Negative Matrix Factorization** model with 6 topics.

    from **pycaret.nlp **import *
    dataset = **get_topics**(dataset, text='en', model='nmf', num_topics=6)

**Output:**

![Topic Modeling Results (after execution of Python code)](https://cdn-images-1.medium.com/max/3834/1*DY70gtEWPMy5BPiuKwWPPA.png)

![Final Output (after clicking on Table)](https://cdn-images-1.medium.com/max/3840/1*lbTGdPoqZQkYejsl01D4dQ.png)

New columns containing topic weights are attached to the original dataset. Here’s how the final output looks like in Power BI once you apply the query.

![Results in Power BI Desktop (after applying query)](https://cdn-images-1.medium.com/max/3844/1*btTSFxgmmEV8e7-Nw133mw.png)

# 3. Dashboard

Once you have topic weights in Power BI, here’s an example of how you can visualize it in dashboard to generate insights:

![Summary page of Dashboard](https://cdn-images-1.medium.com/max/2624/1*SyZczsDz5Pf-4Srfj_p8vQ.png)

![Details page of Dashboard](https://cdn-images-1.medium.com/max/2660/1*SVY-1iq0qXmh_Dl8D3rl0w.png)

You can download the PBIX file and the data set from our [GitHub](https://github.com/pycaret/powerbi-nlp).

If you would like to learn more about implementing Topic Modeling in Jupyter notebook using PyCaret, watch this 2 minute video tutorial:

 <iframe src="https://medium.com/media/75ec7a7299cd663bd63aa14ba8716025" frameborder=0></iframe>

If you are Interested in learning more about Topic Modeling, you can also checkout our NLP 101 [Notebook Tutorial](https://www.pycaret.org/nlp101) for beginners.

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
