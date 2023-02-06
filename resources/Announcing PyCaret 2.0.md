
# Announcing PyCaret 2.0

# by Moez Ali

![https://www.pycaret.org](https://cdn-images-1.medium.com/max/2126/1*oT-VYfpNDeKJ1L9vkpESdw.png)

We are excited to announce the second release of PyCaret today.

PyCaret is an open source, **low-code** machine learning library in Python that automates machine learning workflow. It is an end-to-end machine learning and model management tool that speeds up machine learning experiment cycle and makes you more productive.

In comparison with the other open source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few words only. This makes experiments exponentially fast and efficient.

See detailed [release notes](https://github.com/pycaret/pycaret/releases/tag/2.0) for PyCaret 2.0.

# **Why use PyCaret?**

![PyCaret 2.0 Features](https://cdn-images-1.medium.com/max/2066/1*wT0m1kx8WjY_P7hrM6KDbA.png)

# Installing PyCaret 2.0

Installing PyCaret is very easy and takes only a few minutes. We strongly recommend using virtual environment to avoid potential conflict with other libraries. See the following example code to create a ***conda*** ***environment ***and install pycaret within that conda environment:

    **# create a conda environment **
    conda create --name yourenvname python=3.6  

    **# activate environment **
    conda activate yourenvname  

    **# install pycaret **
    pip install **pycaret==2.0  **

    **# create notebook kernel linked with the conda environment python -m **ipykernel install --user --name yourenvname --display-name "display-name"

If you are using Azure notebooks or Google Colab, run the following code to install PyCaret.

    !pip install **pycaret==2.0**

All hard dependencies are automatically installed when you install PyCaret using pip. [Click here](https://github.com/pycaret/pycaret/blob/master/requirements.txt) to see the complete list of dependencies.

# 👉 Getting Started with PyCaret 2.0

The first step of any machine learning experiment in PyCaret is to set up an environment by importing the relevant module and initialize the **setup function **by passing dataframe and name of the target variable. See example code:

 <iframe src="https://medium.com/media/bbe8bbbe5bd7be704d960a3c230ce804" frameborder=0></iframe>

**Sample Output:**

![Output is truncated](https://cdn-images-1.medium.com/max/2000/1*di8zOe7rN7kWHO8t-6C6hg.png)

All the preprocessing transformations are applied within **setup function. **PyCaret provides over 20 different pre-processing transformation that can be defined within setup function. [Click here](https://www.pycaret.org/preprocessing) to learn more about PyCaret’s preprocessing abilities.

![https://www.pycaret.org/preprocessing](https://cdn-images-1.medium.com/max/2000/1*EcfstE4cOIhazduR4iUX0w.png)

# 👉 **Compare Models**

This is the first step we recommend in any supervised machine learning task. This function trains all the models in the model library using default hyperparameters and evaluates performance metrics using cross validation. It returns the trained model object class. The evaluation metrics used are:

* **For Classification: **Accuracy, AUC, Recall, Precision, F1, Kappa, MCC

* **For Regression: **MAE, MSE, RMSE, R2, RMSLE, MAPE

Here are few ways you can use **compare_models** function:

 <iframe src="https://medium.com/media/779d56a284937e68237a19836a3e3239" frameborder=0></iframe>

**Sample Output:**

![Sample output from compare_models function](https://cdn-images-1.medium.com/max/2000/1*gUjKx0cQbpaWl226CzAdyw.png)

# 👉 **Create Model**

Create Model function trains a model using default hyperparameters and evaluates performance metrics using cross validation. This function is base to almost all other functions in PyCaret. It returns the trained model object class. Here are few ways you can use this function:

 <iframe src="https://medium.com/media/cb264ea67c8515266f65308882e3f668" frameborder=0></iframe>

**Sample Output:**

![Sample output from create_model function](https://cdn-images-1.medium.com/max/2000/1*NDwHzljCyqQpkH55ogHzTA.png)

To learn more about **create model** function, [click here](https://www.pycaret.org/create-model).

# 👉 Tune Model

Tune Model function tunes the hyperparameter of the model passed as an estimator. It uses Random grid search with pre-defined tuning grids that are fully customizable. Here are few ways you can use this function:

 <iframe src="https://medium.com/media/83249af10ce9201391d94e59befa9e8b" frameborder=0></iframe>

To learn more about **tune model** function, [click here](https://www.pycaret.org/tune-model).

# 👉 Ensemble Model

There are few functions available to ensemble base learners. **ensemble_model**, **blend_models **and **stack_models **are three of them. Here are few ways you can use this function:

 <iframe src="https://medium.com/media/cf9d78c2fadb78827e3f900fc2263b3a" frameborder=0></iframe>

To learn more about ensemble models in PyCaret, [click here](https://www.pycaret.org/ensemble-model).

# 👉 Predict Model

As the name suggests, this function is used for inference / prediction. Here is how you can use it:

 <iframe src="https://medium.com/media/57c73e65c64b95d8decf64566a5d9d5d" frameborder=0></iframe>

# 👉 Plot Model

Plot Model function is used to evaluate performance of the trained machine learning model. Here is an example:

 <iframe src="https://medium.com/media/e8303e8f0a3cc89fc600a59ff9bf032d" frameborder=0></iframe>

![Sample output from plot_model function](https://cdn-images-1.medium.com/max/2000/1*Do2ho2O_fg8w62VPuwVm7g.png)

[Click here](https://www.pycaret.org/plot-model) to learn more about different visualization in PyCaret.

Alternatively, you can use **evaluate_model **function to see plots *via *the user interface within notebook.

![evaluate_model function in PyCaret](https://cdn-images-1.medium.com/max/2560/1*AGsJlbX6bhCOG2r_uhvkKg.gif)

# 👉 Util functions

PyCaret 2.0 includes several new util functions that comes handy when managing your machine learning experiments with PyCaret. Some of them are shown below:

 <iframe src="https://medium.com/media/425caeb91cf6c710ea6c2ef31dcb9313" frameborder=0></iframe>

To see all new functions implemented in PyCaret 2.0, See [release notes](https://github.com/pycaret/pycaret/releases/tag/2.0).

# 👉 Experiment Logging

PyCaret 2.0 embeds MLflow tracking component as a backend API and UI for logging parameters, code versions, metrics, and output files when running your machine learning code and for later visualizing the results. Here is how you can log your experiment in PyCaret.

 <iframe src="https://medium.com/media/d7aa1f47a2054147a3836dba599878a9" frameborder=0></iframe>

**Output (on localhost:5000)**

![https://localhost:5000](https://cdn-images-1.medium.com/max/3788/1*Z_08utFByIK_9nsps3rctA.png)

# 👉 Putting it all together — Create your own AutoML software

Using all the functions, let’s create a simple command line software that will train multiple models with default parameters, tune hyperparameters of top candidate models, try different ensembling techniques and returns / saves the best model. Here is the command line script:

 <iframe src="https://medium.com/media/5a7a5bd674b74c3fa2e92658aa5abbd1" frameborder=0></iframe>

This script will dynamically select and saves the best model. In just few lines of code you have developed your own Auto ML software with a full fledged logging system and even a UI presenting beautiful leaderboard.

There is no limit to what you can achieve using the light weight workflow automation library in Python. If you find this useful, please do not forget to give us ⭐️ on our github repo if you like PyCaret.

To hear more about PyCaret follow us on [LinkedIn](https://www.linkedin.com/company/pycaret/) and [Youtube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g).

# Important Links

[Release Notes for PyCaret 2.0](https://github.com/pycaret/pycaret/releases/tag/2.0)
[User Guide / Documentation](https://www.pycaret.org/guide)[
](https://github.com/pycaret/pycaret/releases/tag/2.0)[Github](http://www.github.com/pycaret/pycaret)
[Install PyCaret](https://www.pycaret.org/install)
[Notebook Tutorials](https://www.pycaret.org/tutorial)
[Contribute in PyCaret](https://www.pycaret.org/contribute)

# Want to learn about a specific module?

Click on the links below to see the documentation and working examples.

[Classification](https://www.pycaret.org/classification)
[Regression
](https://www.pycaret.org/regression)[Clustering](https://www.pycaret.org/clustering)
[Anomaly Detection
](https://www.pycaret.org/anomaly-detection)[Natural Language Processing](https://www.pycaret.org/nlp)
[Association Rule Mining](https://www.pycaret.org/association-rules)
