## Welcome to PyCaret!
PyCaret is a free software and an open source low-code machine learning library for supervised and unsupervised machine learning techniques in Python programming language. Its primary objective is to reduce the cycle time from hypothesis to insights and make data scientists more productive in their experiments. It does so by providing a high-level API which is sophisticated yet easy to use and consistent across all modules. PyCaret enables data scientists and analysts to perform iterative end-to-end data science experiments in very efficient way allowing them to reach the conclusions faster. Through the use of its high-level low-code API, the amount of time spent in coding experiments reduce drastically, allowing business to restructure their machine learning workflows and re-evaluate the value chain of data science projects. PyCaret is essentially a python wrapper around several machine learning frameworks and libraries such as scikit-learn, XGBoost, Microsoft LightGBM, spaCy to name a few.

## Current Release
The current release is beta 0.0.43 (as of 11/02/2020). A full public release is expected by end of Feb 2020. 

## Who should use PyCaret?
PyCaret is free and open source library which is easy to install and can be setup either locally or on any cloud service within minutes. As such there is no limitation of use however, in our opinion following are the ideal target audience: <br />

- Citizen data scientists who wants to easily implement data science projects in a low-code environment.
- Data scientists who wants to increase their productivity.
- Consultants / Data Science teams who wants to deliver Proof of Concept (POC) quickly and efficiently.
- Students learning or planning to learn data science skills with no prior background in computer science.
- Data Science Professionals and members of academic community who are involved in teaching machine learning.
- Data Scientists and members of data science community who compete on platforms like Kaggle.
- Small to mid-size companies looking to implement data science projects without committing significant resources.
- Students, academicians, researchers and data science professionals seeking to combine simplicity of R language with power of Python.

## Installation

#### Dependencies
Please read requirements.txt for list of requirements. They are automatically installed when pycaret is installed using pip.

#### User Installation
The easiest way to install pycaret is using pip.

```python
pip install pycaret
```

## PyCaret is really low code
PyCaret is not only simple and easy to use but it is also easy to maintain. It allows data scientist to perform end-to-end experiment and enhances its ability to perform simple to complex tasks without the need to write and maintain extra lines of codes. By removing the hindrance of coding, we allow data scientists to be more creative and focused on business problems.


## PyCaret's sophisticated Pipeline
PyCaret is simple and easy to use but its functionalities are beyond basic. The architect of PyCaret is deployment ready which means as you perform the experiment, all the steps are automatically saved in a pipeline which can be deployed into production with ease. 

Though PyCaret is natively used in Notebook environment (either local or cloud), it is built with a production ready architect. Experiments performed in PyCaret can be easily saved in a binary file format which is transferrable across environment. The saved experiment consists of entire preprocessing pipeline, model object and any other output created during experiment. Experiments can then later be loaded in the same or different environment to either continue with modeling or consuming the trained model in production

When a model is constructed using PyCaret, it becomes part of pipeline. A pipeline is an object in PyCaret that contains sequence of tasks and subtasks that are independently executable as a standalone machine learning experiment in production or encapsulated as part of any workflow that supports python. Example of tasks performed in a typical machine learning pipeline are:

**Data Preparation:** Imputation of missing values and categorical encoding.

**Transformation:** Feature Scaling and non-linear transformations, ordinal and cardinal encoding and target transformation.

**Feature Engineering:** Extraction from the existing features, creating new features using polynomial/trigonometric combinations and feature interactions of numeric search space, statistical extraction using group features, feature creation using unsupervised methods such as clustering.

**Feature Selection:** Feature selection using various wrapper-based techniques, removing features with high multi-collinearity and low statistical variances.

**Dimensionality Reduction:** Dimensionality reduction using PCA and t-SNE, combining rare categories using unsupervised techniques.

**Model Training:** Model training and cross validation, Model ensembling and stacking, hyper-parameter tuning, validation and model selection.

**Deployment:** Model deployment and consumption on cloud.


PyCaret automatically orchestrates all of the dependencies between pipeline steps. Once a pipeline is constructed, it can be transferred to another environment to run on a different hardware to perform tasks at scale. When an experiment is initialized in PyCaret, a pseudo-random seed is generated and distributed to all processes and sub-processes in PyCaret. This allows for reproducibility in future. By organizing an experiment using pipeline, PyCaret supports the computer science imperative of modularization i.e. each component should do only one thing. Modularity is vital in successfully deploying data science projects. Not only this helps you to stay organized experiment but also without pipeline, it is difficult to manage the entire machine learning experiment and requires a lot of technical expertise and resources to manage deployment. 

## PyCaret is seemlessly integrated

PyCaret and its Machine Learning capabilities can be seamlessly integrated within any other environment that supports python integration such as Microsoft Power BI, Tableau, Alteryx, Informatica and KNIME to name a few. This gives immense power to users of these platforms who can now integrate PyCaret in their existing workflows to add layer of Machine Learning in their BI applications absolutely free and with ease. 

PyCaret also has rich magic functions for unsupervised learning. Magic functions are essentially shortcuts (one-word code) that can be executed within the existing ETL pipeline. This will allow business analysts and domain experts to integrate machine learning and implement sophisticated techniques be it Density-Based Spatial Clustering for segment analysis or Isolation Forest for outlier detection. This will enable analysts to leverage the power of advanced analytics and machine learning in their comfort zone without needing to write hundreds of lines of code.

## Reproducibility using PyCaret

Reproducing the entire experiment with the same results is often a challenge. This is due to nature of randomization involved in machine learning. Many preprocessing steps and algorithms are dependent on randomized component which is controlled through pseudo-random number generator. Many libraries including sci-kit learn does not have its own random generator. A global random state can be set in an environment, However, it is prone to modification by other code during execution. Thus, the only way to achieve replicability is the pass random state instance in every function. For a typical machine learning experiment with iteration, this could be well over fifty (50) places to define. PyCaret's architect has solved this problem by back tracing all the functions and distributing the unified random state instance globally. When environment is initialized in PyCaret it generates a pseudo-random number which is also displayed in the grid printed after setup is complete. You can also pass the same number in setup as a parameter to reproduce the exact same results in any other environment as well.

## PyCaret Deployment abilities

PyCaret also supports model deployment on cloud for consuming in production. It does so in a very simplistic way. This capability in PyCaret reduces the reliance of data science practitioners on data engineers to some extent, allowing them to focus on high scale systems. As of the first public release, only AWS S3 containers are supported for deployment. This feature is in preview for now and only support’s batch predictions. We are working to improve the functionality and include support for other cloud providers such as Microsoft Azure and Google.

## PyCaret Runtime Visibility
Often when performing experiment, one of the challenge or frustration is anticipating how long a particular model will take in training. Waiting sometimes may result in hours or days. Having visibility of remaining time of experiment as your code progresses through is a big feature PyCaret has to offer (only for Notebook users – since it uses HTML). Having run time visibility of the code may help data scientists / citizen data scientists to anticipate the total run time based on initial estimation shown by PyCaret. This may lead to decision making process to run or not run certain models which otherwise would have not been known until training is completed.


## License

Copyright 2019-2020 Moez Ali <moez.ali@queensu.ca>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
© 2020 GitHub, Inc.