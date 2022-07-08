<p align="center">
<img src="https://i.ibb.co/c2ZRy10/pycaret-bentoml-repo-logo.png" width="700" />
</p>

<h4 align="center">
    <a href="https://pycaret.org/" target="blank">PyCaret</a> • <a href="https://www.bentoml.com/" target="blank">BentoML</a> • <a href="https://docs.pytest.org/" target="blank">PyTest</a> • <a href="https://dvc.org/" target="blank">DVC</a>
</h4>

<h3 align="center">
    <b>
        A data science template for projects built on top of PyCaret <br> 
        for ML modeling and BentoML for production-ready model serving.
    </b>
</h3>

Just run:
```
$ bentoml serve service:svc
```

And get:
<p align="center">
<img src="https://i.ibb.co/HgQyfZr/bentoml-uber-fare.png" width="900" />
</p>

## Project Description
This project is highly inspired by the <a href="https://github.com/cookiecutter/cookiecutter" target="blank">Cookiecutter</a> data science template project. Cookiecutter is a standard choice for many data scientist when it comes to structure an end-to-end data science project. I've been using this library for more than a year now, and I've started to deal with some "problems" (totally personal point of view based on my own job requirements) when I need to deploy a model product of my data science modeling.

At least for me, and based on my job requirements, Cookiecutter has a lot of extra directories I don't need at all and have never used on my modelings. In this modification I got rid of these folders, leaving only the most critical ones for a data science modeling project that will generate a trained pipeline that need to go to a production server.

Unlike in a Cookiecutter project, by default I'm using a YAML config file to manage all the pipeline and model information as a sort of "single source of truth". By leveraging a YAML config file to set lists of predictors, paths, model hyperparameters and etc we can avoid hard-coding a lot of this stuff.

**Author**: Arthur G.

## Main Project Dependencies
+ dvc == 2.12.1"
+ pytest == 7.1.2"
+ bentoml == 1.0.0rc3"
+ pycaret[full] == 2.3.10"

## Project Structure
------------
    ├── data
    │   ├── finalized        <- Data ready for modeling and retraining in production.
    │   ├── processed        <- Intermediate data that has been transformed.
    │   └── raw              <- The original, immutable data dump.
    │
    ├── conf                 <- Folder to store the main config.yaml file
    │   ├── __init__.py      <- Makes conf a Python module
    │   ├── config_core.py   <- Scripts to make config.yaml data importable. 
    │   └── config.yaml      <- Config file to store data, pipeline and model's related information.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── src                       <- Source code for use in this project.
    │   ├── __init__.py           <- Makes src a Python module
    │   ├── data_processing.py    <- Scripts to run the data processing define in the notebooks.
    │   └── train_pipeline.py     <- Scripts to train/retrain the model pipeline.
    │
    ├── tests                          <- Code for test data proc and train scripts.
    │   ├── __init__.py                <- Makes tests a Python module
    │   ├── test_data_processing.py    <- Test data processing script.
    │   └── test_train_pipeline.py     <- Test training script.
    │
    ├── bentofile.yaml     <- File used to build a deployable Bento API.
    ├── Pipfile            <- The requirements file for reproducing the environment
    ├── Makefile           <- Makefile with commands like `make install` or `make activate`
    ├── LICENSE            <- The specified LICENSE for the project.
    ├── README.md          <- The top-level README for developers using this project.
    └── service.py         <- File to run the BentoML server and make the trained model available.
--------

## Main Features
+ Research and production in one code base.
+ Just run one file to generate production-ready API with BentoML CLI.
+ Use Pipenv to ease the project dependencies management.
+ Manage code files as well as data and model files on the same project using DVC.

## Run The Project
You can start by running one of these two commands:
```
$ make install
# or
$ pipenv install
```

And all the dependencies will be installed within an environment created by pipenv (I expect you to have pipenv installed). To activate the environment and start working on the notebooks or with BentoML, you can run:
```
$ make activate
#or
$ pipenv shell
```

Run Pytest to check if the data processing and model retraining scripts are working properly (and yes, they are, at least on my machine):
```
$ pytest
```

For now, the last step is to run the BentoML server with an estimation service (that comes from the trained pipeline in the models folder) as a REST API.:
```
$ bentoml serve service:svc
```

The **service** keyword in the command shown above reference the *service.py* file.