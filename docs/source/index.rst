PyCaret 
=======

.. image:: https://raw.githubusercontent.com/pycaret/pycaret/master/docs/images/logo.png

PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.

In comparison with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few words only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as `scikit-learn <https://scikit-learn.org/stable/>`_, `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_, `LightGBM <https://github.com/microsoft/LightGBM>`_, `CatBoost <https://catboost.ai/>`_, `spaCy <https://spacy.io/>`_, `Optuna <https://github.com/optuna/optuna>`_, `Hyperopt <https://github.com/hyperopt/hyperopt>`_, `Ray <https://github.com/ray-project/ray/tree/master/python/ray/tune>`_, and many more. 

The design and simplicity of PyCaret is inspired by the emerging role of citizen data scientists, a term first used by Gartner. Citizen Data Scientists are power users who can perform both simple and moderately sophisticated analytical tasks that would previously have required more expertise. Seasoned data scientists are often difficult to find and expensive to hire but citizen data scientists can be an effective way to mitigate this gap and address data-related challenges in the business setting.

**Citing PyCaret**\ :

If you’re citing PyCaret in research or scientific paper, please cite this page as the resource. PyCaret’s first stable release 1.0.0 was made publicly available in April 2020. 

pycaret.org. PyCaret, April 2020. URL https://pycaret.org/. PyCaret version 1.0.0.

A formatted version of the citation would look like this::

    @Manual{PyCaret,
      author  = {Moez Ali},
      title   = {PyCaret: An open source, low-code machine learning library in Python},
      year    = {2020},
      month   = {April},
      note    = {PyCaret version 1.0.0},
      url     = {https://www.pycaret.org}
    }

We are appreciated that PyCaret has been increasingly referred and cited in scientific works. See all citations `here <https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=pycaret&btnG=>`_.


**Key Links and Resources**\ :

* `Release Notes <https://github.com/pycaret/pycaret/releases>`_
* `Example Notebooks <https://github.com/pycaret/pycaret/tree/master/examples>`_
* `Blog Posts <https://github.com/pycaret/pycaret/tree/master/resources>`_
* `LinkedIn <https://www.linkedin.com/company/pycaret/>`_
* `YouTube <https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g>`_
* `Contribute <https://github.com/pycaret/pycaret/blob/master/CONTRIBUTING.md>`_
* `More about PyCaret <https://pycaret.gitbook.io/docs/learn-pycaret/official-blog>`_


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   self
   installation
   tutorials
   contribute
   modules

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api/classification
   api/regression
   api/time_series
   api/clustering
   api/anomaly
   api/nlp
   api/arules
   api/datasets