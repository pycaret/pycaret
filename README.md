## pycaret
pycaret is the free software and open source machine learning library for python programming language. It is built around several popular machine learning libraries in python. Its primary objective is to reduce the cycle time of hypothesis to insights by providing an easy to use high level unified API. pycaret's vision is to become defacto standard for teaching machine learning and data science. Our strength is in our easy to use unified interface for both supervised and unsupervised machine learning problems. It saves time and effort that citizen data scientists, students and researchers spent on coding or learning to code using different interfaces, so that now they can focus on business problem and value creation. 

## Key Features
* Ease of Use
* Focus on Business Problem
* 10x efficient
* Collaboration
* Business Ready
* Cloud Ready

## Current Release
The current release is beta 0.0.4 (as of 23/12/2019). A full release for public is targetted to be available by 31/12/2020.

## Installation

#### Dependencies
Please read requirements.txt for list of requirements. They are automatically installed when pycaret is installed using pip.

#### User Installation
The easiest way to install pycaret is using pip.

```python
pip install pycaret
```

## Quick Start
As of beta 0.0.4 classification, regression and nlp modules are available. Future release will be include Anamoly Detection, Association Rules, Clustering, Recommender System and Time Series.

### Classification

Getting data from pycaret repository

```python
from pycaret.datasets import get_data
juice = get_data('juice')
```

Initializing the pycaret environment setup

```python
exp1 = setup(juice, 'Purchase')
```

Creating a simple logistic regression (includes fitting, CV and metric evaluation)
```python
lr = create_model('lr')
```

List of available estimators:

Estimator                   Abbreviated String     Original Implementation 
---------                   ------------------     -----------------------
Logistic Regression         'lr'                   linear_model.LogisticRegression
K Nearest Neighbour         'knn'                  neighbors.KNeighborsClassifier
Naives Bayes                'nb'                   naive_bayes.GaussianNB
Decision Tree               'dt'                   tree.DecisionTreeClassifier
SVM (Linear)                'svm'                  linear_model.SGDClassifier
SVM (RBF)                   'rbfsvm'               svm.SVC
Gaussian Process            'gpc'                  gaussian_process.GPC
Multi Level Perceptron      'mlp'                  neural_network.MLPClassifier
Ridge Classifier            'ridge'                linear_model.RidgeClassifier
Random Forest               'rf'                   ensemble.RandomForestClassifier
Quadratic Disc. Analysis    'qda'                  discriminant_analysis.QDA
AdaBoost                    'ada'                  ensemble.AdaBoostClassifier
Gradient Boosting           'gbc'                  ensemble.GradientBoostingClassifier
Linear Disc. Analysis       'lda'                  discriminant_analysis.LDA
Extra Trees Classifier      'et'                   ensemble.ExtraTreesClassifier
Extreme Gradient Boosting   'xgboost'              xgboost.readthedocs.io
Light Gradient Boosting     'lightgbm'             github.com/microsoft/LightGBM

Tuning a model using GridSearchCV with pre-defined grids.
```python
tuned_lr = tune_model('lr')
```
Ensembling trained model
```python
dt = create_model('dt')
dt_bagging = ensemble_model('dt', method='Bagging')
dt_boosting = ensemble_model('dt', method='Boosting')
```

Creating a voting classifier
```python
voting_clf1 = blend_models() #creates voting classifier for entire library

#create voting classifier for specific models
lr = create_model('lr')
svm = create_model('svm')
mlp = create_model('mlp')
xgboost = create_model('xgboost')

voting_clf2 = blend_models( [lr, svm, mlp, xgboost] )
```

Creating a voting classifier
```python
voting_clf1 = blend_models() #creates voting classifier for entire library

#create voting classifier for specific models
lr = create_model('lr')
svm = create_model('svm')
mlp = create_model('mlp')
xgboost = create_model('xgboost')

voting_clf2 = blend_models( [lr, svm, mlp, xgboost] )
```

Stacking Models in one layer
```python
#create individual classifiers
lr = create_model('lr')
svm = create_model('svm')
mlp = create_model('mlp')
xgboost = create_model('xgboost')

stacker = stack_models( [lr,svm,mlp], meta_model = xgboost )
```

Stacking Models in Multiple layers
```python
#create individual classifiers
lr = create_model('lr')
svm = create_model('svm')
mlp = create_model('mlp')
gbc = create_model('gbc')
nb = create_model('nb')
lightgbm = create_model('lightgbm')
knn = create_model('knn')
xgboost = create_model('xgboost')

stacknet = create_stacknet( [ [lr,svm,mlp], [gbc, nb], [lightgbm, knn] ], meta_model = xgboost )
#meta model by default is Logistic Regression
```

Plot Models
```python
lr = create_model('lr')
plot_model(lr, plot='auc')
```
List of available plots:

Name                        Abbreviated String     Original Implementation 
---------                   ------------------     -----------------------
Area Under the Curve         'auc'                 .. / rocauc.html
Discrimination Threshold     'threshold'           .. / threshold.html
Precision Recall Curve       'pr'                  .. / prcurve.html
Confusion Matrix             'confusion_matrix'    .. / confusion_matrix.html
Class Prediction Error       'error'               .. / class_prediction_error.html
Classification Report        'class_report'        .. / classification_report.html
Decision Boundary            'boundary'            .. / boundaries.html
Recursive Feat. Selection    'rfe'                 .. / rfecv.html
Learning Curve               'learning'            .. / learning_curve.html
Manifold Learning            'manifold'            .. / manifold.html
Calibration Curve            'calibration'         .. / calibration_curve.html
Validation Curve             'vc'                  .. / validation_curve.html
Dimension Learning           'dimension'           .. / radviz.html
Feature Importance           'feature'                   N/A 
Model Hyperparameter         'parameter'                 N/A 

Saving Model for Deployment
```python
lr = create_model('lr')
save_model(lr, 'lr_23122019')
```
Saving Entire Experiment Pipeline
```python
save_experiment('expname1')
```
Loading Model / Experiment
```python
m = load_model('lr_23122019')
e = load_experiment('expname1')
```
AutoML
```python
aml1 = automl()
```
## Documentation
Documentation work is in progress. They will be uploaded on our website http://www.pycaret.org as soon as they are available. (Target Availability : 21/01/2020)

## Contributions
Contributions are most welcome. To make contribution please reach out moez.ali@queensu.ca
