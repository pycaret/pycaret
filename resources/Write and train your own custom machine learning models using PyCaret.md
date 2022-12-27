
# Write and train your own custom machine learning models using PyCaret

# A step-by-step, beginner-friendly tutorial on how to write and train custom machine learning models in PyCaret

![Photo by [Rob Lambert](https://unsplash.com/@roblambertjr?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/10368/0*dERI0tdhD_Yay4OU)

# PyCaret

PyCaret is an open-source, low-code machine learning library and end-to-end model management tool built-in Python for automating machine learning workflows. It is incredibly popular for its ease of use, simplicity, and ability to quickly and efficiently build and deploy end-to-end ML prototypes.

PyCaret is an alternate low-code library that can replace hundreds of code lines with few lines only. This makes the experiment cycle exponentially fast and efficient.

PyCaret is **simple and** **easy to use**. All the operations performed in PyCaret are sequentially stored in a **Pipeline** that is fully automated for **deployment. **Whether it’s imputing missing values, one-hot-encoding, transforming categorical data, feature engineering, or even hyperparameter tuning, PyCaret automates all of it.

This tutorial assumes that you have some prior knowledge and experience with PyCaret. If you haven’t used it before, no problem — you can get a quick headstart through these tutorials:

* [PyCaret 2.2 is here — what’s new](https://towardsdatascience.com/pycaret-2-2-is-here-whats-new-ad7612ca63b)

* [Announcing PyCaret 2.0](https://towardsdatascience.com/announcing-pycaret-2-0-39c11014540e)

* [Five things you don’t know about PyCaret](https://towardsdatascience.com/5-things-you-dont-know-about-pycaret-528db0436eec)

# Installing PyCaret

Installing PyCaret is very easy and takes only a few minutes. We strongly recommend using a virtual environment to avoid potential conflicts with other libraries.

PyCaret’s default installation is a slim version of pycaret that only installs hard dependencies [listed here](https://github.com/pycaret/pycaret/blob/master/requirements.txt).

    **# install slim version (default)
    **pip install pycaret

    **# install the full version**
    pip install pycaret[full]

When you install the full version of pycaret, all the optional dependencies as [listed here](https://github.com/pycaret/pycaret/blob/master/requirements-optional.txt) are also installed.

# 👉 Let’s get started

Before we start talking about custom model training, let’s see a quick demo of how PyCaret works with out-of-the-box models. I will be using the ‘insurance’ dataset available on [PyCaret’s Repository](https://github.com/pycaret/pycaret/tree/master/datasets). The goal of this dataset is to predict patient charges based on some attributes.

# 👉 **Dataset**

    **# read data from pycaret repo
    **from pycaret.datasets import get_data
    data = get_data('insurance')

![Sample rows from insurance dataset](https://cdn-images-1.medium.com/max/2000/1*h4UQIEtUUmsP2ybJlsFKPA.png)

# 👉 **Data Preparation**

Common to all modules in PyCaret, the setup is the first and the only mandatory step in any machine learning experiment performed in PyCaret. This function takes care of all the data preparation required before training models. Besides performing some basic default processing tasks, PyCaret also offers a wide array of pre-processing features. To learn more about all the preprocessing functionalities in PyCaret, you can see this [link](https://pycaret.org/preprocessing/).

    **# initialize setup
    **from pycaret.regression import *
    s = setup(data, target = 'charges')

![setup function in pycaret.regression module](https://cdn-images-1.medium.com/max/2736/1*s5bhY1IN1jOM11JDuwFaZw.png)

Whenever you initialize the setup function in PyCaret, it profiles the dataset and infers the data types for all input features. If all data types are correctly inferred, you can press enter to continue.

![Output from setup — truncated for display](https://cdn-images-1.medium.com/max/2000/1*Y9kIg0BfRfzG1WdZm6MnbQ.png)

# 👉 Available Models

To check the list of all models available for training, you can use the function called models . It displays a table with model ID, name, and the reference of the actual estimator.

    **# check all the available models
    **models()

![Output from models() — Output truncated for display purpose](https://cdn-images-1.medium.com/max/2000/1*JVVVA2aUyVBJ9SyQjEC13A.png)

# 👉 Model Training & Selection

The most used function for training any model in PyCaret is create_model . It takes an ID for the estimator you want to train.

    **# train decision tree
    **dt = create_model('dt')

![Output from create_model(‘dt’)](https://cdn-images-1.medium.com/max/2000/1*-txdluhP3Jl27ZUvoMzdow.png)

The output shows the 10-fold cross-validated metrics with mean and standard deviation. The output from this function is a trained model object, which is essentially a scikit-learn object.

    print(dt)

![Output from print(dt)](https://cdn-images-1.medium.com/max/2000/1*7z-sMZdcNVamc1U-PFzVXg.png)

To train multiple models in a loop, you can write a simple list comprehension:

    **# train multiple models**
    multiple_models = [create_model(i) for i in ['dt', 'lr', 'xgboost']]

    **# check multiple_models
    **type(multiple_models), len(multiple_models)
    >>> (list, 3)

    print(multiple_models)

![Output from print(multiple_models)](https://cdn-images-1.medium.com/max/2734/1*OO_I-wbH7h4PseoHOQ36Vg.png)

If you want to train all the models available in the library instead of the few selected you can use PyCaret’s compare_models function instead of writing your own loop (*the results will be the same though*).

    **# compare all models**
    best_model = compare_models()

![Output from the compare_models function](https://cdn-images-1.medium.com/max/2000/1*9XtwGLvLDmJ5ro2fq67HLQ.png)

compare_models returns the output which shows the cross-validated metrics for all models. According to this output, Gradient Boosting Regressor is the best model with $2,702 in Mean Absolute Error ****(MAE) ****using 10-fold cross-validation on the train set.

    **# check the best model**
    print(best_model)

![Output from the print(best_model)](https://cdn-images-1.medium.com/max/2000/1*pQSXoclIKREi-2U2uG3jhQ.png)

The metrics shown in the above grid is cross-validation scores, to check the score of the best_modelon hold-out set:

    **# predict on hold-out
    **pred_holdout = predict_model(best_model)

![Output from the predict_model(best_model) function](https://cdn-images-1.medium.com/max/2000/1*G1l1bG1f_Tzoeo7X_ieixw.png)

To generate predictions on the unseen dataset you can use the same predict_model function but just pass an extra parameter data :

    **# create copy of data drop target column**
    data2 = data.copy()
    data2.drop('charges', axis=1, inplace=True)

    **# generate predictions
    **predictions = predict_model(best_model, data = data2)

![Output from predict_model(best_model, data = data2)](https://cdn-images-1.medium.com/max/2000/1*jch_dJNscn_i2vNfgWpV5g.png)

# 👉 Writing and Training Custom Model

So far what we have seen is training and model selection for all the available models in PyCaret. However, the way PyCaret works for custom models is exactly the same. As long as, your estimator is compatible with sklearn API style, it will work the same way. Let’s see few examples.

Before I show you how to write your own custom class, I will first demonstrate how you can work with custom non-sklearn models (models that are not available in sklearn or pycaret’s base library).

# 👉 **GPLearn Models**

While Genetic Programming (GP) can be used to perform a [very wide variety of tasks](http://www.genetic-programming.org/combined.php), gplearn is purposefully constrained to solving symbolic regression problems.

Symbolic regression is a machine learning technique that aims to identify an underlying mathematical expression that best describes a relationship. It begins by building a population of naive random formulas to represent a relationship between known independent variables and their dependent variable targets to predict new data. Each successive generation of programs is then evolved from the one that came before it by selecting the fittest individuals from the population to undergo genetic operations.

To use models from gplearn you will have to first install it:

    **# install gplearn
    **pip install gplearn

Now you can simply import the untrained model and pass it in the create_model function:

    **# import untrained estimator**
    from gplearn.genetic import SymbolicRegressor
    sc = SymbolicRegressor()

    **# train using create_model
    **sc_trained = create_model(sc)

![Output from create_model(sc_trained)](https://cdn-images-1.medium.com/max/2000/1*WjHjXcM_Q4w7zuM_nfzVng.png)

    print(sc_trained)

![Output from print(sc_trained)](https://cdn-images-1.medium.com/max/2000/1*1glIGPLohn6bxElYSmgtuQ.png)

You can also check the hold-out score for this:

    **# check hold-out score
    **pred_holdout_sc = predict_model(sc_trained)

![Output from predict_model(sc_trained)](https://cdn-images-1.medium.com/max/2000/1*EoTHf4G1wm8Zh0xqScS_Gg.png)

# 👉 NGBoost Models

ngboost is a Python library that implements Natural Gradient Boosting, as described in [“NGBoost: Natural Gradient Boosting for Probabilistic Prediction”](https://stanfordmlgroup.github.io/projects/ngboost/). It is built on top of [Scikit-Learn](https://scikit-learn.org/stable/) and is designed to be scalable and modular with respect to the choice of proper scoring rule, distribution, and base learner. A didactic introduction to the methodology underlying NGBoost is available in this [slide deck](https://drive.google.com/file/d/183BWFAdFms81MKy6hSku8qI97OwS_JH_/view?usp=sharing).

To use models from ngboost, you will have to first install ngboost:

    **# install ngboost**
    pip install ngboost

Once installed, you can import the untrained estimator from the ngboost library and use create_model to train and evaluate the model:

    **# import untrained estimator**
    from ngboost import NGBRegressor
    ng = NGBRegressor()

    **# train using create_model
    **ng_trained = create_model(ng)

![Output from create_model(ng)](https://cdn-images-1.medium.com/max/2000/1*6GjuFIPOBN4f19Qj7YPMlw.png)

    print(ng_trained)

![Output from print(ng_trained)](https://cdn-images-1.medium.com/max/2000/1*SSZoHUK4NnLE2Ri_Uy985Q.png)

# 👉 Writing Custom Class

The above two examples gplearn and ngboost are custom models for pycaret as they are not available in the default library but you can use them just like you can use any other out-of-the-box models. However, there may be a use-case that involves writing your own algorithm (i.e. maths behind the algorithm), in which case you can inherit the base class from sklearn and write your own maths.

Let’s create a naive estimator which learns the mean value of target variable during fit stage and predicts the same mean value for all new data points, irrespective of X input (*probably not useful in real life, but just to make demonstrate the functionality*).

    **# create custom estimator**
    import numpy as np**
    **from sklearn.base import BaseEstimator

    class MyOwnModel(BaseEstimator):
        
        def __init__(self):
            self.mean = 0
            
        def fit(self, X, y):
            self.mean = y.mean()
            return self
        
        def predict(self, X):
            return np.array(X.shape[0]*[self.mean])

Now let’s use this estimator for training:

    **# import MyOwnModel class**
    mom = MyOwnModel()

    **# train using create_model
    **mom_trained = create_model(mom)

![Output from create_model(mom)](https://cdn-images-1.medium.com/max/2000/1*ElBm8PRRTYgkCQ7J6tsf_g.png)

    **# generate predictions on data**
    predictions = predict_model(mom_trained, data=data)

![Output from predict_model(mom, data=data)](https://cdn-images-1.medium.com/max/2000/1*SE8aSw-Rhj41PYzRQxHWQw.png)

Notice that Label column which is essentially the prediction is the same number $13,225 for all the rows, that’s because we created this algorithm in such a way, that learns from the mean of train set and predict the same value (just to keep things simple).

I hope that you will appreciate the ease of use and simplicity in PyCaret. In just a few lines, you can perform end-to-end machine learning experiments and write your own algorithms without adjusting any native code.

# Coming Soon!

Next week I will be writing a tutorial to advance this tutorial. We will write a more complex algorithm instead of just a mean prediction. I will introduce some complex concepts in the next tutorial. Please follow me on [Medium](https://medium.com/@moez-62905), [LinkedIn](https://www.linkedin.com/in/profile-moez/), and [Twitter](https://twitter.com/moezpycaretorg1) to get more updates.

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
