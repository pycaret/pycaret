# Copyright (C) 2019-2020 Moez Ali <moez.ali@queensu.ca>
# License: MIT, moez.ali@queensu.ca

"""
  
PyCaret - An end-to-end open source machine learning library

Module Author:
--------------
pycaret.anomaly, Moez Ali <moez.ali@queensu.ca>
pycaret.classification, Moez Ali <moez.ali@queensu.ca>
pycaret.clustering, Moez Ali <moez.ali@queensu.ca>
pycaret.datasets, Moez Ali <moez.ali@queensu.ca>
pycaret.nlp, Moez Ali <moez.ali@queensu.ca>
pycaret.preprocess, Fahad Akbar <M.akbar@queensu.ca>
pycaret.regression, Moez Ali <moez.ali@queensu.ca>

"""

from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README


setup(
    name="pycaret",
    version="0.0.43",
    description="A Python package for supervised and unsupervised machine learning.",
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
                     "datefinder==0.7.0", "datetime", "DateTime==4.3", "tqdm==4.36.1", "awscli"]
)