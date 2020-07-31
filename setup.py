<<<<<<< HEAD
# Copyright (C) 2019-2020 Moez Ali <moez.ali@queensu.ca>
# License: MIT, moez.ali@queensu.ca

from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setup(
    name="pycaret",
    version="1.0.0",
    description="An open source, low-code machine learning library in Python",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/pycaret/pycaret",
    author="Moez Ali",
    author_email="moez.ali@queensu.ca",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["pycaret"],
    include_package_data=True,
    install_requires=["pandas", "numpy", "seaborn", "matplotlib", "IPython", "joblib", 
                     "scikit-learn==0.22", "shap==0.32.1", "ipywidgets", "yellowbrick==1.0.1", "xgboost==0.90",
                     "wordcloud", "textblob", "plotly==4.4.1", "cufflinks==0.17.0", "umap-learn",
                     "lightgbm==2.3.1", "pyLDAvis", "gensim", "spacy", "nltk", "mlxtend",
                     "pyod", "catboost==0.20.2", "pandas-profiling==2.3.0", "kmodes==0.10.1",
                     "datefinder==0.7.0", "datetime", "DateTime==4.3", "awscli"]
)
=======
# Copyright (C) 2019-2020 Moez Ali <moez.ali@queensu.ca>
# License: MIT, moez.ali@queensu.ca

from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="pycaret",
    version="2.0",
    description="PyCaret - An open source, low-code machine learning library in Python.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/pycaret/pycaret",
    author="Moez Ali",
    author_email="moez.ali@queensu.ca",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["pycaret"],
    include_package_data=True,
    install_requires=required
)
>>>>>>> dev-1.0.1
