## PyCaret
PyCaret is end-to-end open source machine learning library for python programming language. Its primary objective is to reduce the cycle time of hypothesis to insights by providing an easy to use high level unified API. PyCaret's vision is to become defacto standard for teaching machine learning and data science. Our strength is in our easy to use unified interface for both supervised and unsupervised learning. It saves time and effort that citizen data scientists, students and researchers spent on coding or learning to code using different interfaces, so that now they can focus on business problem.

## Current Release
The current release is beta 0.0.20 (as of 20/01/2020). A full release is targetted in the first week of February 2020.

## Features Currently Available
As per beta 0.0.20 following modules are generally available:
* pycaret.datasets <br/>
* pycaret.classification (binary and multiclass) <br/>
* pycaret.regression <br/>
* pycaret.nlp <br/>
* pycaret.arules <br/>
* pycaret.anamoly <br/>
* pycaret.clustering <br/>

## Future Release
Full public release is targetted to be released in first week of Feb 2020.

## Installation

#### Dependencies
Please read requirements.txt for list of requirements. They are automatically installed when pycaret is installed using pip.

#### User Installation
The easiest way to install pycaret is using pip.

```python
pip install pycaret
```

## Quick Start
As of beta 0.0.20 classification, regression, nlp, arules, anomaly and clustering modules are available.

### Classification / Regression

Getting data from pycaret repository

```python
from pycaret.datasets import get_data
juice = get_data('juice') #classification dataset
```

1. Initializing the pycaret environment setup

```python
from pycaret.classification import * #for classification
from pycaret.regression import * #for regression
exp1 = setup(juice, 'Purchase')
```

2. Creating a simple logistic regression (includes fitting, CV and metric evaluation)
```python
lr = create_model('lr')
```

List of available estimators:

Logistic Regression (lr) <br/>
K Nearest Neighbour (knn) <br/>
Naive Bayes (nb) <br/>
Decision Tree (dt) <br/>
Support Vector Machine - Linear (svm) <br/>
SVM Radial Function (rbfsvm) <br/>
Gaussian Process Classifier (gpc) <br/>
Multi Level Perceptron (mlp) <br/>
Ridge Classifier (ridge) <br/>
Random Forest (rf) <br/>
Quadtratic Discriminant Analysis (qda) <br/>
Adaboost (ada) <br/>
Gradient Boosting Classifier (gbc) <br/>
Linear Discriminant Analysis (lda) <br/>
Extra Trees Classifier (et) <br/>
Extreme Gradient Boosting - xgboost (xgboost) <br/>
Light Gradient Boosting - Microsoft LightGBM (lightgbm) <br/>

3. Compare all models at once
```python
compare_models()
```

4. Tuning a model using pre-built search grids.
```python
tuned_xgb = tune_model('xgboost')
```

4. Ensembling Model
```python
dt = create_model('dt')
dt_bagging = ensemble_model(dt, method='Bagging')
dt_boosting = ensemble_model(dt, method='Boosting')
```

5. Creating a voting classifier
```python
voting_all = blend_models() #creates voting classifier for entire library

#create voting classifier for specific models
lr = create_model('lr')
svm = create_model('svm')
mlp = create_model('mlp')
xgboost = create_model('xgboost')

voting_clf2 = blend_models( [ lr, svm, mlp, xgboost ] )
```

6. Stacking Models in Single Layer
```python
#create individual classifiers
lr = create_model('lr')
svm = create_model('svm')
mlp = create_model('mlp')
xgboost = create_model('xgboost')

stacker = stack_models( [lr,svm,mlp], meta_model = xgboost )
```

7. Stacking Models in Multiple Layers
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

8. Plot Models
```python
lr = create_model('lr')
plot_model(lr, plot='auc')
```
List of available plots:

Area Under the Curve (auc) <br/>
Discrimination Threshold (threshold) <br/>
Precision Recall Curve (pr) <br/>
Confusion Matrix (confusion_matrix) <br/>
Class Prediction Error (error) <br/>
Classification Report (class_report) <br/>
Decision Boundary (boundary) <br/>
Recursive Feature Selection (rfe) <br/>
Learning Curve (learning) <br/>
Manifold Learning (manifold) <br/>
Calibration Curve (calibration) <br/>
Validation Curve (vc) <br/>
Dimension Learning (dimension) <br/>
Feature Importance (feature) <br/>
Model Hyperparameter (parameter) <br/>

9. Evaluate Model
```python
lr = create_model('lr')
evaluate_model(lr) #displays user interface for interactive plotting
```

10. Interpret Tree Based Models
```python
xgboost = create_model('xgboost')
interpret_model(xgboost)
```

11. Saving Model for Deployment
```python
lr = create_model('lr')
save_model(lr, 'lr_23122019')
```

12. Saving Entire Experiment Pipeline
```python
save_experiment('expname1')
```

13. Loading Model / Experiment
```python
m = load_model('lr_23122019')
e = load_experiment('expname1')
```


## Getting Started Tutorials
Tutorials are work in progress. Will be uploaded on our git page by 25/01/2020.

## Documentation
Documentation work is in progress. They will be uploaded on our website http://www.pycaret.org as soon as they are available. (Target Availability : 21/01/2020)

## Contributions
Contributions are most welcome. To make contribution please reach out moez.ali@queensu.ca

## License

Copyright 2020 PyCaret / Copyright 2020 Moez Ali

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
© 2019 GitHub, Inc.