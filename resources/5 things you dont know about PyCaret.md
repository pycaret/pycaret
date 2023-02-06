
# 5 things you don’t know about PyCaret
# by Moez Ali

![From the author of PyCaret](https://cdn-images-1.medium.com/max/2000/1*1HEakzOhZRd21FfAT3TyZw.png)

# PyCaret

PyCaret is an open source machine learning library in Python to train and deploy supervised and unsupervised machine learning models in a **low-code** environment. It is known for its ease of use and efficiency.

In comparison with the other open source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with a few words only.

If you haven’t used PyCaret before or would like to learn more, a good place to start is [here](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46).
> # “After talking to many data scientists who use PyCaret on a daily basis, I have shortlisted 5 features of PyCaret that are lesser known but they extremely powerful.” — Moez Ali

# 👉You can tune “n parameter” in unsupervised experiments

In unsupervised machine learning the “n parameter” i.e. the number of clusters for clustering experiments, the fraction of the outliers in anomaly detection, and the number of topics in topic modeling, is of fundamental importance.

When the eventual objective of the experiment is to predict an outcome (classification or regression) using the results from the unsupervised experiments, then the tune_model() function in the **pycaret.clustering **module**, **the **pycaret.anomaly **module**,** and the **pycaret.nlp **module ****comes in very handy.

To understand this, let’s see an example using the “[Kiva](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/kiva.csv)” dataset.

![](https://cdn-images-1.medium.com/max/2000/1*-161ThHHhI7lMVHuY4jbsA.png)

This is a micro-banking loan dataset where each row represents a borrower with their relevant information. Column ‘en’ captures the loan application text of each borrower, and the column ‘status’ represents whether the borrower defaulted or not (default = 1 or no default = 0).

You can use **tune_model **function in **pycaret.nlp **to optimize **num_topics **parameter based on the target variable of supervised experiment (i.e. predicting the optimum number of topics required to improve the prediction of the final target variable). You can define the model for training using **estimator** parameter (‘xgboost’ in this case). This function returns a trained topic model and a visual showing supervised metrics at each iteration.

 <iframe src="https://medium.com/media/7adc412481d38c5d11fc89f4196671a3" frameborder=0></iframe>

![](https://cdn-images-1.medium.com/max/2314/1*RIOVzRCYsA-r-c1Iy7x_5w.png)

# 👉You can improve results from hyperparameter tuning by increasing “n_iter”

The **tune_model **function in the **pycaret.classification **module and the **pycaret.regression** module employs random grid search over pre-defined grid search for hyper-parameter tuning. Here the default number of iterations is set to 10.

Results from **tune_model **may not necessarily be an improvement on the results from the base models created using **create_model. **Since the grid search is random, you can increase the **n_iter **parameter to improve the performance. See example below:

 <iframe src="https://medium.com/media/009f98cee1bc5a231fc1342e08d406b3" frameborder=0></iframe>

![](https://cdn-images-1.medium.com/max/2000/1*LRu2R2f4rXYkOrWVC6ul5A.png)

# 👉You can programmatically define data types in the setup function

When you initialize the **setup **function**, **you will be asked to confirm data types through a user input. More often when you run the scripts as a part of workflows or execute it as remote kernels (for e.g. Kaggle Notebooks), then in such case, it is required to provide the data types programmatically rather than through the user input box.

See example below using “[insurance](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/insurance.csv)” dataset.

![](https://cdn-images-1.medium.com/max/2000/1*q2WFe3JgZ1SxSkiuuvonKQ.png)

 <iframe src="https://medium.com/media/a96c059e33ee57e12df796357fe19044" frameborder=0></iframe>

the **silent** parameter is set to True to avoid input, **categorical_features **parameter takes the name of categorical columns as string, and **numeric_features **parameter takes the name of numeric columns as a string.

# 👉You can ignore certain columns for model building

On many occasions, you have features in dataset that you do not necessarily want to remove but want to ignore for training a machine learning model. A good example would be a clustering problem where you want to ignore certain features during cluster creation but later you need those columns for analysis of cluster labels. In such cases, you can use the **ignore_features **parameter within the **setup **to ignore such features.

In the example below, we will perform a clustering experiment and we want to ignore **‘Country Name’** and **‘Indicator Name’**.

![](https://cdn-images-1.medium.com/max/2000/1*0xcKweKh77A-vgzb5u5_mw.png)

 <iframe src="https://medium.com/media/87c6f8d873c53b758b3ec6e2a588f20e" frameborder=0></iframe>

# 👉You can optimize the probability threshold % in binary classification

In classification problems, the cost of **false positives** is almost never the same as the cost of **false negatives**. As such, if you are optimizing a solution for a business problem where **Type 1** and **Type 2** errors have a different impact, you can optimize your classifier for a probability threshold value to optimize the custom loss function simply by defining the cost of true positives, true negatives, false positives and false negatives separately. By default, all classifiers have a threshold of 0.5.

See example below using “[credit](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/credit.csv)” dataset.

 <iframe src="https://medium.com/media/9476e7c2d508b2711631020ceebe583f" frameborder=0></iframe>

![](https://cdn-images-1.medium.com/max/2000/1*oCsUyp91pSJSDdzi-ho6QA.png)

You can then pass **0.2 **as **probability_threshold **parameter in **predict_model **function to use 0.2 as a threshold for classifying positive class. See example below:

 <iframe src="https://medium.com/media/7670ed065b5f318524592e8b84bdbf54" frameborder=0></iframe>

# PyCaret 2.0.0 is coming!

We have received overwhelming support and feedback from the data science community. We are actively working on improving PyCaret and preparing for our next release. **PyCaret 2.0.0 will be bigger and better**. If you would like to share your feedback and help us improve further, you may [fill this form](https://www.pycaret.org/feedback) on the website or leave a comment on our [GitHub ](https://www.github.com/pycaret/)or [LinkedIn](https://www.linkedin.com/company/pycaret/) page.

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
