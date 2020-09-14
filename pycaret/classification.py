# Module: Classification
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT
# Release: PyCaret 2.2
# Last modified : 26/08/2020

from pycaret.internal.tune_sklearn_patches import (
    get_tune_sklearn_tunegridsearchcv,
    get_tune_sklearn_tunesearchcv,
)
from pycaret.internal.utils import (
    color_df,
    normalize_custom_transformers,
    make_internal_pipeline,
    nullcontext,
    supports_partial_fit,
    true_warm_start,
    estimator_pipeline,
)
from pycaret.internal.logging import get_logger
from pycaret.internal.plotting import show_yellowbrick_plot
from pycaret.internal.Display import Display
from pycaret.internal.distributions import *
from pycaret.internal.validation import *
from pycaret.containers.models.classification import (
    get_all_model_containers,
    LogisticRegressionClassifierContainer,
)
from pycaret.containers.metrics.classification import (
    get_all_metric_containers,
    ClassificationMetricContainer,
)
import pandas as pd
import numpy as np
import os
import sys
import datetime
import time
import random
import gc
from copy import deepcopy
from sklearn.base import clone
from typing import List, Tuple, Any, Union
import warnings
from IPython.utils import io
import traceback

warnings.filterwarnings("ignore")


def setup(
    data: pd.DataFrame,
    target: str,
    train_size: float = 0.7,
    test_data: Optional[pd.DataFrame] = None,
    preprocess: bool = True,
    sampling: bool = False,
    sample_estimator: Optional[Any] = None,
    categorical_features: Optional[List[str]] = None,
    categorical_imputation: str = "constant",
    ordinal_features: Optional[Dict[str, list]] = None,
    high_cardinality_features: Optional[List[str]] = None,
    high_cardinality_method: str = "frequency",
    numeric_features: Optional[List[str]] = None,
    numeric_imputation: str = "mean",  # method 'zero' added in pycaret==2.1
    date_features: Optional[List[str]] = None,
    ignore_features: Optional[List[str]] = None,
    normalize: bool = False,
    normalize_method: str = "zscore",
    transformation: bool = False,
    transformation_method: str = "yeo-johnson",
    handle_unknown_categorical: bool = True,
    unknown_categorical_method: str = "least_frequent",
    pca: bool = False,
    pca_method: str = "linear",
    pca_components: Optional[float] = None,
    ignore_low_variance: bool = False,
    combine_rare_levels: bool = False,
    rare_level_threshold: float = 0.10,
    bin_numeric_features: Optional[List[str]] = None,
    remove_outliers: bool = False,
    outliers_threshold: float = 0.05,
    remove_multicollinearity: bool = False,
    multicollinearity_threshold: float = 0.9,
    remove_perfect_collinearity: bool = False,
    create_clusters: bool = False,
    cluster_iter: int = 20,
    polynomial_features: bool = False,
    polynomial_degree: int = 2,
    trigonometry_features: bool = False,
    polynomial_threshold: float = 0.1,
    group_features: Optional[List[str]] = None,
    group_names: Optional[List[str]] = None,
    feature_selection: bool = False,
    feature_selection_threshold: float = 0.8,
    feature_selection_method: str = "classic",  # boruta algorithm added in pycaret==2.1
    feature_interaction: bool = False,
    feature_ratio: bool = False,
    interaction_threshold: float = 0.01,
    fix_imbalance: bool = False,
    fix_imbalance_method: Optional[Any] = None,
    data_split_shuffle: bool = True,
    data_split_stratify: Union[bool, List[str]] = False,  # added in pycaret==2.2
    fold_strategy: Union[str, Any] = "stratifiedkfold",  # added in pycaret==2.2
    fold: int = 10,  # added in pycaret==2.2
    fold_shuffle: bool = False,
    fold_groups: Optional[Union[str, pd.DataFrame]] = None,
    n_jobs: Optional[int] = -1,
    use_gpu: bool = False,  # added in pycaret==2.1
    custom_pipeline: Union[
        Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]
    ] = None,
    html: bool = True,
    session_id: Optional[int] = None,
    log_experiment: bool = False,
    experiment_name: Optional[str] = None,
    log_plots: bool = False,
    log_profile: bool = False,
    log_data: bool = False,
    silent: bool = False,
    verbose: bool = True,
    profile: bool = False,
    display: Optional[Display] = None,
):

    """
    This function initializes the environment in pycaret and creates the transformation
    pipeline to prepare the data for modeling and deployment. setup() must called before
    executing any other function in pycaret. It takes two mandatory parameters:
    data and name of the target column.
    
    All other parameters are optional.

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')

    'juice' is a pandas.DataFrame  and 'Purchase' is the name of target column.

    Parameters
    ----------
    data : pandas.DataFrame
        Shape (n_samples, n_features) where n_samples is the number of samples and 
        n_features is the number of features.

    target: str
        Name of the target column to be passed in as a string. The target variable can 
        be either binary or multiclass.

    train_size: float, default = 0.7
        Size of the training set. By default, 70% of the data will be used for training 
        and validation. The remaining data will be used for a test / hold-out set.

    test_data: pandas.DataFrame, default = None
        When a dataframe is passed to this parameter it will be used as the test set, instead
        of using a portion of the data to create one.

    preprocess: bool, default = True
        If False, will not do any preprocessing on the data aside from mandatory steps, and
        ignore all other parameters related to preprocessing (including 'ordinal_features' and 
        'high_cardinality_features' params), aside from 'custom_pipeline'.

    sampling: bool, default = False
        When the sample size exceeds 25,000 samples, pycaret will build a base estimator
        at various sample sizes from the original dataset. This will return a performance 
        plot of AUC, Accuracy, Recall, Precision, Kappa and F1 values at various sample 
        levels, that will assist in deciding the preferred sample size for modeling. 
        The desired sample size must then be entered for training and validation in the 
        pycaret environment. When sample_size entered is less than 1, the remaining dataset 
        (1 - sample) is used for fitting the model only when finalize_model() is called.
    
    sample_estimator: object, default = None
        If None, Logistic Regression is used by default.
    
    categorical_features: list, default = None
        If the inferred data types are not correct, categorical_features can be used to
        overwrite the inferred type. If when running setup the type of 'column1' is
        inferred as numeric instead of categorical, then this parameter can be used 
        to overwrite the type by passing categorical_features = ['column1'].
    
    categorical_imputation: str, default = 'constant'
        If missing values are found in categorical features, they will be imputed with
        a constant 'not_available' value. The other available option is 'mode' which 
        imputes the missing value using most frequent value in the training dataset. 
    
    ordinal_features: dictionary, default = None
        When the data contains ordinal features, they must be encoded differently using 
        the ordinal_features param. If the data has a categorical variable with values
        of 'low', 'medium', 'high' and it is known that low < medium < high, then it can 
        be passed as ordinal_features = { 'column_name' : ['low', 'medium', 'high'] }. 
        The list sequence must be in increasing order from lowest to highest.
    
    high_cardinality_features: list, default = None
        When the data containts features with high cardinality, they can be compressed
        into fewer levels by passing them as a list of column names with high cardinality.
        Features are compressed using method defined in high_cardinality_method param.
    
    high_cardinality_method: str, default = 'frequency'
        When method set to 'frequency' it will replace the original value of feature
        with the frequency distribution and convert the feature into numeric. Other
        available method is 'clustering' which performs the clustering on statistical
        attribute of data and replaces the original value of feature with cluster label.
        The number of clusters is determined using a combination of Calinski-Harabasz and 
        Silhouette criterion. 
          
    numeric_features: list, default = None
        If the inferred data types are not correct, numeric_features can be used to
        overwrite the inferred type. If when running setup the type of 'column1' is 
        inferred as a categorical instead of numeric, then this parameter can be used 
        to overwrite by passing numeric_features = ['column1'].    

    numeric_imputation: str, default = 'mean'
        If missing values are found in numeric features, they will be imputed with the 
        mean value of the feature. The other available options are 'median' which imputes 
        the value using the median value in the training dataset and 'zero' which
        replaces missing values with zeroes.

    date_features: str, default = None
        If the data has a DateTime column that is not automatically detected when running
        setup, this parameter can be used by passing date_features = 'date_column_name'. 
        It can work with multiple date columns. Date columns are not used in modeling. 
        Instead, feature extraction is performed and date columns are dropped from the 
        dataset. If the date column includes a time stamp, features related to time will 
        also be extracted.

    ignore_features: str, default = None
        If any feature should be ignored for modeling, it can be passed to the param
        ignore_features. The ID and DateTime columns when inferred, are automatically 
        set to ignore for modeling. 

    normalize: bool, default = False
        When set to True, the feature space is transformed using the normalized_method
        param. Generally, linear algorithms perform better with normalized data however, 
        the results may vary and it is advised to run multiple experiments to evaluate
        the benefit of normalization.

    normalize_method: str, default = 'zscore'
        Defines the method to be used for normalization. By default, normalize method
        is set to 'zscore'. The standard zscore is calculated as z = (x - u) / s. The
        other available options are:

        'minmax'    : scales and translates each feature individually such that it is in 
                    the range of 0 - 1.
        
        'maxabs'    : scales and translates each feature individually such that the maximal 
                    absolute value of each feature will be 1.0. It does not shift/center 
                    the data, and thus does not destroy any sparsity.
        
        'robust'    : scales and translates each feature according to the Interquartile 
                    range. When the dataset contains outliers, robust scaler often gives 
                    better results.
    
    transformation: bool, default = False
        When set to True, a power transformation is applied to make the data more normal /
        Gaussian-like. This is useful for modeling issues related to heteroscedasticity or 
        other situations where normality is desired. The optimal parameter for stabilizing 
        variance and minimizing skewness is estimated through maximum likelihood.

    transformation_method: str, default = 'yeo-johnson'
        Defines the method for transformation. By default, the transformation method is set
        to 'yeo-johnson'. The other available option is 'quantile' transformation. Both 
        the transformation transforms the feature set to follow a Gaussian-like or normal
        distribution. Note that the quantile transformer is non-linear and may distort 
        linear correlations between variables measured at the same scale.
    
    handle_unknown_categorical: bool, default = True
        When set to True, unknown categorical levels in new / unseen data are replaced by
        the most or least frequent level as learned in the training data. The method is 
        defined under the unknown_categorical_method param.

    unknown_categorical_method: str, default = 'least_frequent'
        Method used to replace unknown categorical levels in unseen data. Method can be
        set to 'least_frequent' or 'most_frequent'.

    pca: bool, default = False
        When set to True, dimensionality reduction is applied to project the data into 
        a lower dimensional space using the method defined in pca_method param. In 
        supervised learning pca is generally performed when dealing with high feature
        space and memory is a constraint. Note that not all datasets can be decomposed
        efficiently using a linear PCA technique and that applying PCA may result in loss 
        of information. As such, it is advised to run multiple experiments with different 
        pca_methods to evaluate the impact. 

    pca_method: str, default = 'linear'
        The 'linear' method performs Linear dimensionality reduction using Singular Value 
        Decomposition. The other available options are:
        
        kernel      : dimensionality reduction through the use of RVF kernel.  
        
        incremental : replacement for 'linear' pca when the dataset to be decomposed is 
                    too large to fit in memory

    pca_components: int/float, default = 0.99
        Number of components to keep. if pca_components is a float, it is treated as a 
        target percentage for information retention. When pca_components is an integer
        it is treated as the number of features to be kept. pca_components must be strictly
        less than the original number of features in the dataset.

    ignore_low_variance: bool, default = False
        When set to True, all categorical features with insignificant variances are 
        removed from the dataset. The variance is calculated using the ratio of unique 
        values to the number of samples, and the ratio of the most common value to the 
        frequency of the second most common value.
    
    combine_rare_levels: bool, default = False
        When set to True, all levels in categorical features below the threshold defined 
        in rare_level_threshold param are combined together as a single level. There must 
        be atleast two levels under the threshold for this to take effect. 
        rare_level_threshold represents the percentile distribution of level frequency. 
        Generally, this technique is applied to limit a sparse matrix caused by high 
        numbers of levels in categorical features. 
    
    rare_level_threshold: float, default = 0.1
        Percentile distribution below which rare categories are combined. Only comes into
        effect when combine_rare_levels is set to True.
    
    bin_numeric_features: list, default = None
        When a list of numeric features is passed they are transformed into categorical
        features using KMeans, where values in each bin have the same nearest center of a 
        1D k-means cluster. The number of clusters are determined based on the 'sturges' 
        method. It is only optimal for gaussian data and underestimates the number of bins 
        for large non-gaussian datasets.

    remove_outliers: bool, default = False
        When set to True, outliers from the training data are removed using PCA linear
        dimensionality reduction using the Singular Value Decomposition technique.

    outliers_threshold: float, default = 0.05
        The percentage / proportion of outliers in the dataset can be defined using
        the outliers_threshold param. By default, 0.05 is used which means 0.025 of the 
        values on each side of the distribution's tail are dropped from training data.

    remove_multicollinearity: bool, default = False
        When set to True, the variables with inter-correlations higher than the threshold
        defined under the multicollinearity_threshold param are dropped. When two features
        are highly correlated with each other, the feature that is less correlated with 
        the target variable is dropped. 

    multicollinearity_threshold: float, default = 0.9
        Threshold used for dropping the correlated features. Only comes into effect when 
        remove_multicollinearity is set to True.
    
    remove_perfect_collinearity: bool, default = False
        When set to True, perfect collinearity (features with correlation = 1) is removed
        from the dataset, When two features are 100% correlated, one of it is randomly 
        dropped from the dataset.

    create_clusters: bool, default = False
        When set to True, an additional feature is created where each instance is assigned
        to a cluster. The number of clusters is determined using a combination of 
        Calinski-Harabasz and Silhouette criterion. 

    cluster_iter: int, default = 20
        Number of iterations used to create a cluster. Each iteration represents cluster 
        size. Only comes into effect when create_clusters param is set to True.

    polynomial_features: bool, default = False
        When set to True, new features are created based on all polynomial combinations 
        that exist within the numeric features in a dataset to the degree defined in 
        polynomial_degree param. 

    polynomial_degree: int, default = 2pca_method_pass
        Degree of polynomial features. For example, if an input sample is two dimensional 
        and of the form [a, b], the polynomial features with degree = 2 are: 
        [1, a, b, a^2, ab, b^2].

    trigonometry_features: bool, default = False
        When set to True, new features are created based on all trigonometric combinations 
        that exist within the numeric features in a dataset to the degree defined in the
        polynomial_degree param.

    polynomial_threshold: float, default = 0.1
        This is used to compress a sparse matrix of polynomial and trigonometric features.
        Polynomial and trigonometric features whose feature importance based on the 
        combination of Random Forest, AdaBoost and Linear correlation falls within the 
        percentile of the defined threshold are kept in the dataset. Remaining features 
        are dropped before further processing.

    group_features: list or list of list, default = None
        When a dataset contains features that have related characteristics, group_features
        param can be used for statistical feature extraction. For example, if a dataset has 
        numeric features that are related with each other (i.e 'Col1', 'Col2', 'Col3'), a 
        list containing the column names can be passed under group_features to extract 
        statistical information such as the mean, median, mode and standard deviation.
    
    group_names: list, default = None
        When group_features is passed, a name of the group can be passed into the 
        group_names param as a list containing strings. The length of a group_names 
        list must equal to the length  of group_features. When the length doesn't 
        match or the name is not passed, new features are sequentially named such as 
        group_1, group_2 etc.
    
    feature_selection: bool, default = False
        When set to True, a subset of features are selected using a combination of various
        permutation importance techniques including Random Forest, Adaboost and Linear 
        correlation with target variable. The size of the subset is dependent on the 
        feature_selection_param. Generally, this is used to constrain the feature space 
        in order to improve efficiency in modeling. When polynomial_features and 
        feature_interaction  are used, it is highly recommended to define the 
        feature_selection_threshold param with a lower value. Feature selection algorithm
        by default is 'classic' but could be 'boruta', which will lead PyCaret to create
        use the Boruta selection algorithm.

    feature_selection_threshold: float, default = 0.8
        Threshold used for feature selection (including newly created polynomial features).
        A higher value will result in a higher feature space. It is recommended to do 
        multiple trials with different values of feature_selection_threshold specially in 
        cases where polynomial_features and feature_interaction are used. Setting a very 
        low value may be efficient but could result in under-fitting.
    
    feature_selection_method: str, default = 'classic'
        Can be either 'classic' or 'boruta'. Selects the algorithm responsible for
        choosing a subset of features. For the 'classic' selection method, PyCaret will 
        use various permutation importance techniques. For the 'boruta' algorithm, PyCaret
        will create an instance of boosted trees model, which will iterate with permutation 
        over all features and choose the best ones based on the distributions of feature 
        importance.
    
    feature_interaction: bool, default = False 
        When set to True, it will create new features by interacting (a * b) for all 
        numeric variables in the dataset including polynomial and trigonometric features 
        (if created). This feature is not scalable and may not work as expected on datasets
        with large feature space.
    
    feature_ratio: bool, default = False
        When set to True, it will create new features by calculating the ratios (a / b) 
        of all numeric variables in the dataset. This feature is not scalable and may not 
        work as expected on datasets with large feature space.
    
    interaction_threshold: bool, default = 0.01
        Similar to polynomial_threshold, It is used to compress a sparse matrix of newly 
        created features through interaction. Features whose importance based on the 
        combination  of  Random Forest, AdaBoost and Linear correlation falls within the 
        percentile of the  defined threshold are kept in the dataset. Remaining features 
        are dropped before further processing.
    
    fix_imbalance: bool, default = False
        When dataset has unequal distribution of target class it can be fixed using
        fix_imbalance parameter. When set to True, SMOTE (Synthetic Minority Over-sampling 
        Technique) is applied by default to create synthetic datapoints for minority class.

    fix_imbalance_method: obj, default = None
        When fix_imbalance is set to True and fix_imbalance_method is None, 'smote' is 
        applied by default to oversample minority class during cross validation. This 
        parameter accepts any module from 'imblearn' that supports 'fit_resample' method.

    data_split_shuffle: bool, default = True
        If set to False, prevents shuffling of rows when splitting data.

    data_split_stratify: bool or list, default = False
        Whether to stratify when splitting data.
        If True, will stratify by the target column. If False, will not stratify.
        If list is passed, will stratify by the columns with the names in the list.
        Requires data_split_shuffle to be set to True.

    fold_strategy: str or scikit-learn compatible CV generator object, default = 'stratifiedkfold'
        Choice of cross validation strategy. Possible values are:

        * 'kfold' for KFold CV,
        * 'stratifiedkfold' for Stratified KFold CV,
        * 'groupkfold' for Group KFold CV,
        * 'timeseries' for TimeSeriesSplit CV,
        * a custom CV generator object compatible with scikit-learn.

    fold: integer, default = 10
        Number of folds to be used in cross validation. Must be at least 2.
        Ignored if fold_strategy is an object.

    fold_shuffle: bool, default = False
        If set to False, prevents shuffling of rows when using cross validation. Only applicable for
        'kfold' and 'stratifiedkfold' fold_strategy. Ignored if fold_strategy is an object.
    
    fold_groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).

    n_jobs: int, default = -1
        The number of jobs to run in parallel (for functions that supports parallel 
        processing) -1 means using all processors. To run all functions on single 
        processor set n_jobs to None.

    use_gpu: str or bool, default = False
        If set to 'Force', will try to use GPU with all algorithms that support it,
        and raise exceptions if they are unavailable.
        If set to True, will use GPU with algorithms that support it, and fall
        back to CPU if they are unavailable.
        If set to False, will only use CPU.

        GPU enabled algorithms:
        
        - CatBoost
        - XGBoost
        - Logistic Regression, Ridge, SVM, SVC - requires cuML >= 0.15 to be installed.
          https://github.com/rapidsai/cuml

    custom_pipeline: transformer or list of transformers or tuple
    (str, transformer) or list of tuples (str, transformer), default = None
        If set, will append the passed transformers (including Pipelines) to the PyCaret
        preprocessing Pipeline applied after train-test split during model fitting.
        This Pipeline is applied on each CV fold separately and on the final fit.
        The transformers will be applied before PyCaret transformers (eg. SMOTE).

    html: bool, default = True
        If set to False, prevents runtime display of monitor. This must be set to False
        when using environment that doesnt support HTML.

    session_id: int, default = None
        If None, a random seed is generated and returned in the Information grid. The 
        unique number is then distributed as a seed in all functions used during the 
        experiment. This can be used for later reproducibility of the entire experiment.

    log_experiment: bool, default = False
        When set to True, all metrics and parameters are logged on MLFlow server.

    experiment_name: str, default = None
        Name of experiment for logging. When set to None, 'clf' is by default used as 
        alias for the experiment name.

    log_plots: bool, default = False
        When set to True, specific plots are logged in MLflow as a png file. By default,
        it is set to False. 

    log_profile: bool, default = False
        When set to True, data profile is also logged on MLflow as a html file. 
        By default, it is set to False. 

    log_data: bool, default = False
        When set to True, train and test dataset are logged as csv. 

    silent: bool, default = False
        When set to True, confirmation of data types is not required. All preprocessing 
        will be performed assuming automatically inferred data types. Not recommended 
        for direct use except for established pipelines.
    
    verbose: bool, default = True
        Information grid is not printed when verbose is set to False.

    profile: bool, default = False
        If set to true, a data profile for Exploratory Data Analysis will be displayed 
        in an interactive HTML report. 

    Warnings
    --------
    - Some GPU models require conversion from float64 to float32,
      which may result in loss of precision. It should not be an issue in majority of cases.
      Models impacted:

        * cuml.ensemble.RandomForestClassifier

    Returns
    -------
    info_grid
        Information grid is printed.

    environment
        This function returns various outputs that are stored in variables
        as tuples. They are used by other functions in pycaret.
      
       
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    warnings.filterwarnings("ignore")

    from pycaret.utils import __version__

    ver = __version__

    # create logger
    global logger

    logger = get_logger()

    logger.info("PyCaret Classification Module")
    logger.info(f"version {ver}")
    logger.info("Initializing setup()")
    logger.info(f"setup({function_params_str})")

    # logging environment and libraries
    logger.info("Checking environment")

    from platform import python_version, platform, python_build, machine

    logger.info(f"python_version: {python_version()}")
    logger.info(f"python_build: {python_build()}")
    logger.info(f"machine: {machine()}")
    logger.info(f"platform: {platform()}")

    try:
        import psutil

        logger.info(f"Memory: {psutil.virtual_memory()}")
        logger.info(f"Physical Core: {psutil.cpu_count(logical=False)}")
        logger.info(f"Logical Core: {psutil.cpu_count(logical=True)}")
    except:
        logger.warning(
            "cannot find psutil installation. memory not traceable. Install psutil using pip to enable memory logging."
        )

    logger.info("Checking libraries")

    try:
        from pandas import __version__

        logger.info(f"pd=={__version__}")
    except:
        logger.warning("pandas not found")

    try:
        from numpy import __version__

        logger.info(f"numpy=={__version__}")
    except:
        logger.warning("numpy not found")

    try:
        from sklearn import __version__

        logger.info(f"sklearn=={__version__}")
    except:
        logger.warning("sklearn not found")

    try:
        from xgboost import __version__

        logger.info(f"xgboost=={__version__}")
    except:
        logger.warning("xgboost not found")

    try:
        from lightgbm import __version__

        logger.info(f"lightgbm=={__version__}")
    except:
        logger.warning("lightgbm not found")

    try:
        from catboost import __version__

        logger.info(f"catboost=={__version__}")
    except:
        logger.warning("catboost not found")

    try:
        from mlflow.version import VERSION

        warnings.filterwarnings("ignore")
        logger.info(f"mlflow=={VERSION}")
    except:
        logger.warning("mlflow not found")

    # run_time
    runtime_start = time.time()

    logger.info("Checking Exceptions")

    all_cols = list(data.columns)
    all_cols.remove(target)

    # checking data type
    if hasattr(data, "shape") is False:
        raise TypeError("data passed must be of type pandas.DataFrame")

    # checking train size parameter
    if type(train_size) is not float:
        raise TypeError("train_size parameter only accepts float value.")

    # checking sampling parameter
    if type(sampling) is not bool:
        raise TypeError("sampling parameter only accepts True or False.")

    # checking sampling parameter
    if target not in data.columns:
        raise ValueError("Target parameter doesnt exist in the data provided.")

    # checking session_id
    if session_id is not None:
        if type(session_id) is not int:
            raise TypeError("session_id parameter must be an integer.")

    # checking sampling parameter
    if type(profile) is not bool:
        raise TypeError("profile parameter only accepts True or False.")

    # checking normalize parameter
    if type(normalize) is not bool:
        raise TypeError("normalize parameter only accepts True or False.")

    # checking transformation parameter
    if type(transformation) is not bool:
        raise TypeError("transformation parameter only accepts True or False.")

    # checking categorical imputation
    allowed_categorical_imputation = ["constant", "mode"]
    if categorical_imputation not in allowed_categorical_imputation:
        raise ValueError(
            "categorical_imputation param only accepts 'constant' or 'mode'"
        )

    # ordinal_features
    if ordinal_features is not None:
        if type(ordinal_features) is not dict:
            raise TypeError(
                "ordinal_features must be of type dictionary with column name as key and ordered values as list."
            )

    # ordinal features check
    if ordinal_features is not None:
        data_cols = data.columns.drop(target)
        ord_keys = ordinal_features.keys()

        for i in ord_keys:
            if i not in data_cols:
                raise ValueError(
                    "Column name passed as a key in ordinal_features param doesnt exist."
                )

        for k in ord_keys:
            if data[k].nunique() != len(ordinal_features[k]):
                raise ValueError(
                    "Levels passed in ordinal_features param doesnt match with levels in data."
                )

        for i in ord_keys:
            value_in_keys = ordinal_features.get(i)
            value_in_data = list(data[i].unique().astype(str))
            for j in value_in_keys:
                if j not in value_in_data:
                    raise ValueError(
                        f"Column name '{i}' doesn't contain any level named '{j}'."
                    )

    # high_cardinality_features
    if high_cardinality_features is not None:
        if type(high_cardinality_features) is not list:
            raise TypeError(
                "high_cardinality_features param only accepts name of columns as a list."
            )

    if high_cardinality_features is not None:
        data_cols = data.columns.drop(target)
        for i in high_cardinality_features:
            if i not in data_cols:
                raise ValueError(
                    "Column type forced is either target column or doesn't exist in the dataset."
                )

    # stratify
    if data_split_stratify:
        if (
            type(data_split_stratify) is not list
            and type(data_split_stratify) is not bool
        ):
            raise TypeError(
                "data_split_stratify param only accepts a bool or a list of strings."
            )

        if not data_split_shuffle:
            raise TypeError(
                "data_split_stratify param requires data_split_shuffle to be set to True."
            )

    # high_cardinality_methods
    high_cardinality_allowed_methods = ["frequency", "clustering"]
    if high_cardinality_method not in high_cardinality_allowed_methods:
        raise ValueError(
            "high_cardinality_method param only accepts 'frequency' or 'clustering'"
        )

    # checking numeric imputation
    allowed_numeric_imputation = ["mean", "median", "zero"]
    if numeric_imputation not in allowed_numeric_imputation:
        raise ValueError(
            f"numeric_imputation param only accepts {', '.join(allowed_numeric_imputation)}."
        )

    # checking normalize method
    allowed_normalize_method = ["zscore", "minmax", "maxabs", "robust"]
    if normalize_method not in allowed_normalize_method:
        raise ValueError(
            f"normalize_method param only accepts {', '.join(allowed_normalize_method)}."
        )

    # checking transformation method
    allowed_transformation_method = ["yeo-johnson", "quantile"]
    if transformation_method not in allowed_transformation_method:
        raise ValueError(
            f"transformation_method param only accepts {', '.join(allowed_transformation_method)}."
        )

    # handle unknown categorical
    if type(handle_unknown_categorical) is not bool:
        raise TypeError(
            "handle_unknown_categorical parameter only accepts True or False."
        )

    # unknown categorical method
    unknown_categorical_method_available = ["least_frequent", "most_frequent"]

    if unknown_categorical_method not in unknown_categorical_method_available:
        raise TypeError(
            f"unknown_categorical_method only accepts {', '.join(unknown_categorical_method_available)}."
        )

    # check pca
    if type(pca) is not bool:
        raise TypeError("PCA parameter only accepts True or False.")

    # pca method check
    allowed_pca_methods = ["linear", "kernel", "incremental"]
    if pca_method not in allowed_pca_methods:
        raise ValueError(
            f"pca method param only accepts {', '.join(allowed_pca_methods)}."
        )

    # pca components check
    if pca is True:
        if pca_method != "linear":
            if pca_components is not None:
                if (type(pca_components)) is not int:
                    raise TypeError(
                        "pca_components parameter must be integer when pca_method is not 'linear'."
                    )

    # pca components check 2
    if pca is True:
        if pca_method != "linear":
            if pca_components is not None:
                if pca_components > len(data.columns) - 1:
                    raise TypeError(
                        "pca_components parameter cannot be greater than original features space."
                    )

    # pca components check 3
    if pca is True:
        if pca_method == "linear":
            if pca_components is not None:
                if type(pca_components) is not float:
                    if pca_components > len(data.columns) - 1:
                        raise TypeError(
                            "pca_components parameter cannot be greater than original features space or float between 0 - 1."
                        )

    # check ignore_low_variance
    if type(ignore_low_variance) is not bool:
        raise TypeError("ignore_low_variance parameter only accepts True or False.")

    # check ignore_low_variance
    if type(combine_rare_levels) is not bool:
        raise TypeError("combine_rare_levels parameter only accepts True or False.")

    # check rare_level_threshold
    if type(rare_level_threshold) is not float:
        raise TypeError("rare_level_threshold must be a float between 0 and 1.")

    # bin numeric features
    if bin_numeric_features is not None:
        for i in bin_numeric_features:
            if i not in all_cols:
                raise ValueError(
                    "Column type forced is either target column or doesn't exist in the dataset."
                )

    # remove_outliers
    if type(remove_outliers) is not bool:
        raise TypeError("remove_outliers parameter only accepts True or False.")

    # outliers_threshold
    if type(outliers_threshold) is not float:
        raise TypeError("outliers_threshold must be a float between 0 and 1.")

    # remove_multicollinearity
    if type(remove_multicollinearity) is not bool:
        raise TypeError(
            "remove_multicollinearity parameter only accepts True or False."
        )

    # multicollinearity_threshold
    if type(multicollinearity_threshold) is not float:
        raise TypeError("multicollinearity_threshold must be a float between 0 and 1.")

    # create_clusters
    if type(create_clusters) is not bool:
        raise TypeError("create_clusters parameter only accepts True or False.")

    # cluster_iter
    if type(cluster_iter) is not int:
        raise TypeError("cluster_iter must be a integer greater than 1.")

    # polynomial_features
    if type(polynomial_features) is not bool:
        raise TypeError("polynomial_features only accepts True or False.")

    # polynomial_degree
    if type(polynomial_degree) is not int:
        raise TypeError("polynomial_degree must be an integer.")

    # polynomial_features
    if type(trigonometry_features) is not bool:
        raise TypeError("trigonometry_features only accepts True or False.")

    # polynomial threshold
    if type(polynomial_threshold) is not float:
        raise TypeError("polynomial_threshold must be a float between 0 and 1.")

    # group features
    if group_features is not None:
        if type(group_features) is not list:
            raise TypeError("group_features must be of type list.")

    if group_names is not None:
        if type(group_names) is not list:
            raise TypeError("group_names must be of type list.")

    # cannot drop target
    if ignore_features is not None:
        if target in ignore_features:
            raise ValueError("cannot drop target column.")

    # feature_selection
    if type(feature_selection) is not bool:
        raise TypeError("feature_selection only accepts True or False.")

    # feature_selection_threshold
    if type(feature_selection_threshold) is not float:
        raise TypeError("feature_selection_threshold must be a float between 0 and 1.")

    # feature_selection_method
    feature_selection_methods = ["boruta", "classic"]
    if feature_selection_method not in feature_selection_methods:
        raise TypeError(
            f"feature_selection_method must be one of {', '.join(feature_selection_methods)}"
        )

    # feature_interaction
    if type(feature_interaction) is not bool:
        raise TypeError("feature_interaction only accepts True or False.")

    # feature_ratio
    if type(feature_ratio) is not bool:
        raise TypeError("feature_ratio only accepts True or False.")

    # interaction_threshold
    if type(interaction_threshold) is not float:
        raise TypeError("interaction_threshold must be a float between 0 and 1.")

    # categorical
    if categorical_features is not None:
        for i in categorical_features:
            if i not in all_cols:
                raise ValueError(
                    "Column type forced is either target column or doesn't exist in the dataset."
                )

    # numeric
    if numeric_features is not None:
        for i in numeric_features:
            if i not in all_cols:
                raise ValueError(
                    "Column type forced is either target column or doesn't exist in the dataset."
                )

    # date features
    if date_features is not None:
        for i in date_features:
            if i not in all_cols:
                raise ValueError(
                    "Column type forced is either target column or doesn't exist in the dataset."
                )

    # drop features
    if ignore_features is not None:
        for i in ignore_features:
            if i not in all_cols:
                raise ValueError(
                    "Feature ignored is either target column or doesn't exist in the dataset."
                )

    # log_experiment
    if type(log_experiment) is not bool:
        raise TypeError("log_experiment parameter only accepts True or False.")

    # log_profile
    if type(log_profile) is not bool:
        raise TypeError("log_profile parameter only accepts True or False.")

    # experiment_name
    if experiment_name is not None:
        if type(experiment_name) is not str:
            raise TypeError("experiment_name parameter must be str if not None.")

    # silent
    if type(silent) is not bool:
        raise TypeError("silent parameter only accepts True or False.")

    # remove_perfect_collinearity
    if type(remove_perfect_collinearity) is not bool:
        raise TypeError(
            "remove_perfect_collinearity parameter only accepts True or False."
        )

    # html
    if type(html) is not bool:
        raise TypeError("html parameter only accepts True or False.")

    # use_gpu
    if use_gpu != "Force" and type(use_gpu) is not bool:
        raise TypeError("use_gpu parameter only accepts 'Force', True or False.")

    # data_split_shuffle
    if type(data_split_shuffle) is not bool:
        raise TypeError("data_split_shuffle parameter only accepts True or False.")

    possible_fold_strategy = ["kfold", "stratifiedkfold", "groupkfold", "timeseries"]
    if not (
        fold_strategy in possible_fold_strategy
        or is_sklearn_cv_generator(fold_strategy)
    ):
        raise TypeError(
            f"fold_strategy parameter must be either a scikit-learn compatible CV generator object or one of {', '.join(possible_fold_strategy)}."
        )

    if fold_strategy == "groupkfold" and (fold_groups is None or len(fold_groups) == 0):
        raise ValueError(
            "'groupkfold' fold_strategy requires 'fold_groups' to be a non-empty array-like object."
        )

    if isinstance(fold_groups, str):
        if fold_groups not in all_cols:
            raise ValueError(
                f"Column {fold_groups} used for fold_groups is not present in the dataset."
            )

    # checking fold parameter
    if type(fold) is not int:
        raise TypeError("fold parameter only accepts integer value.")

    # fold_shuffle
    if type(fold_shuffle) is not bool:
        raise TypeError("fold_shuffle parameter only accepts True or False.")

    # log_plots
    if type(log_plots) is not bool:
        raise TypeError("log_plots parameter only accepts True or False.")

    # log_data
    if type(log_data) is not bool:
        raise TypeError("log_data parameter only accepts True or False.")

    # log_profile
    if type(log_profile) is not bool:
        raise TypeError("log_profile parameter only accepts True or False.")

    # fix_imbalance
    if type(fix_imbalance) is not bool:
        raise TypeError("fix_imbalance parameter only accepts True or False.")

    # fix_imbalance_method
    if fix_imbalance:
        if fix_imbalance_method is not None:
            if hasattr(fix_imbalance_method, "fit_sample"):
                pass
            else:
                raise TypeError(
                    "fix_imbalance_method must contain resampler with fit_sample method."
                )

    # pandas option
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.max_rows", 500)

    # generate USI for mlflow tracking
    import secrets

    # declaring global variables to be accessed by other functions
    logger.info("Declaring global variables")
    global USI, html_param, X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, experiment__, fold_shuffle_param, n_jobs_param, gpu_n_jobs_param, create_model_container, master_model_container, display_container, exp_name_log, logging_param, log_plots_param, fix_imbalance_param, fix_imbalance_method_param, data_before_preprocess, target_param, gpu_param, all_models, _all_models_internal, all_metrics, _internal_pipeline, stratify_param, fold_generator, fold_param, fold_groups_param

    USI = secrets.token_hex(nbytes=2)
    logger.info(f"USI: {USI}")

    global pycaret_globals
    pycaret_globals = {
        "pycaret_globals",
        "USI",
        "html_param",
        "X",
        "y",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "seed",
        "prep_pipe",
        "experiment__",
        "fold_shuffle_param",
        "n_jobs_param",
        "gpu_n_jobs_param",
        "create_model_container",
        "master_model_container",
        "display_container",
        "exp_name_log",
        "logging_param",
        "log_plots_param",
        "fix_imbalance_param",
        "fix_imbalance_method_param",
        "data_before_preprocess",
        "target_param",
        "gpu_param",
        "all_models",
        "_all_models_internal",
        "all_metrics",
        "_internal_pipeline",
        "stratify_param",
        "fold_generator",
        "fold_param",
        "fold_groups_param",
    }

    logger.info(f"pycaret_globals: {pycaret_globals}")

    # create html_param
    html_param = html

    # silent parameter to also set sampling to False
    if silent:
        sampling = False

    logger.info("Preparing display monitor")

    if not display:
        # progress bar
        max_steps = 3
        if sampling:
            max_steps += 10

        progress_args = {"max": max_steps}
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
        ]
        display = Display(
            verbose=verbose,
            html_param=html_param,
            progress_args=progress_args,
            monitor_rows=monitor_rows,
        )

        display.display_progress()
        display.display_monitor()

    logger.info("Importing libraries")

    # general dependencies

    from sklearn.model_selection import train_test_split

    # setting sklearn config to print all parameters including default
    import sklearn

    sklearn.set_config(print_changed_only=False)

    # define highlight function for function grid to display
    def highlight_max(s):
        is_max = s == True
        return ["background-color: lightgreen" if v else "" for v in is_max]

    # cufflinks
    import cufflinks as cf

    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    logger.info("Copying data for preprocessing")

    # copy original data for pandas profiler
    data_before_preprocess = data.copy()

    # generate seed to be used globally
    seed = random.randint(150, 9000) if session_id is None else session_id

    np.random.seed(seed)

    _internal_pipeline = []

    """
    preprocessing starts here
    """

    display.update_monitor(1, "Preparing Data for Modeling")
    display.display_monitor()

    # define parameters for preprocessor

    logger.info("Declaring preprocessing parameters")

    # categorical features
    cat_features_pass = categorical_features or []

    # numeric features
    numeric_features_pass = numeric_features or []

    # drop features
    ignore_features_pass = ignore_features or []

    # date features
    date_features_pass = date_features or []

    # categorical imputation strategy
    cat_dict = {"constant": "not_available", "mode": "most frequent"}
    categorical_imputation_pass = cat_dict[categorical_imputation]

    # transformation method strategy
    trans_dict = {"yeo-johnson": "yj", "quantile": "quantile"}
    trans_method_pass = trans_dict[transformation_method]

    # pass method
    pca_dict = {
        "linear": "pca_liner",
        "kernel": "pca_kernal",
        "incremental": "incremental",
        "pls": "pls",
    }
    pca_method_pass = pca_dict[pca_method]

    # pca components
    if pca is True:
        if pca_components is None:
            if pca_method == "linear":
                pca_components_pass = 0.99
            else:
                pca_components_pass = int((len(data.columns) - 1) * 0.5)

        else:
            pca_components_pass = pca_components

    else:
        pca_components_pass = 0.99

    apply_binning_pass = False if bin_numeric_features is None else True
    features_to_bin_pass = bin_numeric_features or []

    # trignometry
    trigonometry_features_pass = ["sin", "cos", "tan"] if trigonometry_features else []

    # group features
    # =============#

    # apply grouping
    apply_grouping_pass = True if group_features is not None else False

    # group features listing
    if apply_grouping_pass is True:

        if type(group_features[0]) is str:
            group_features_pass = []
            group_features_pass.append(group_features)
        else:
            group_features_pass = group_features

    else:

        group_features_pass = [[]]

    # group names
    if apply_grouping_pass is True:

        if (group_names is None) or (len(group_names) != len(group_features_pass)):
            group_names_pass = list(np.arange(len(group_features_pass)))
            group_names_pass = [f"group_{i}" for i in group_names_pass]

        else:
            group_names_pass = group_names

    else:
        group_names_pass = []

    # feature interactions

    apply_feature_interactions_pass = (
        True if feature_interaction or feature_ratio else False
    )

    interactions_to_apply_pass = []

    if feature_interaction:
        interactions_to_apply_pass.append("multiply")

    if feature_ratio:
        interactions_to_apply_pass.append("divide")

    # unknown categorical
    unkn_dict = {"least_frequent": "least frequent", "most_frequent": "most frequent"}
    unknown_categorical_method_pass = unkn_dict[unknown_categorical_method]

    # ordinal_features
    apply_ordinal_encoding_pass = True if ordinal_features is not None else False

    ordinal_columns_and_categories_pass = (
        ordinal_features if apply_ordinal_encoding_pass else {}
    )

    apply_cardinality_reduction_pass = (
        True if high_cardinality_features is not None else False
    )

    hi_card_dict = {"frequency": "count", "clustering": "cluster"}
    cardinal_method_pass = hi_card_dict[high_cardinality_method]

    cardinal_features_pass = (
        high_cardinality_features if apply_cardinality_reduction_pass else []
    )

    display_dtypes_pass = False if silent else True

    # creating variables to be used later in the function
    train_data = data
    X_before_preprocess = data.drop(target, axis=1)
    y_before_preprocess = data[target]

    logger.info("Importing preprocessing module")

    # import library
    import pycaret.preprocess

    logger.info("Creating preprocessing pipeline")

    prep_pipe = pycaret.preprocess.Preprocess_Path_One(
        train_data=data,
        target_variable=target,
        categorical_features=cat_features_pass,
        apply_ordinal_encoding=apply_ordinal_encoding_pass,
        ordinal_columns_and_categories=ordinal_columns_and_categories_pass,
        apply_cardinality_reduction=apply_cardinality_reduction_pass,
        cardinal_method=cardinal_method_pass,
        cardinal_features=cardinal_features_pass,
        numerical_features=numeric_features_pass,
        time_features=date_features_pass,
        features_todrop=ignore_features_pass,
        numeric_imputation_strategy=numeric_imputation,
        categorical_imputation_strategy=categorical_imputation_pass,
        scale_data=normalize,
        scaling_method=normalize_method,
        Power_transform_data=transformation,
        Power_transform_method=trans_method_pass,
        apply_untrained_levels_treatment=handle_unknown_categorical,
        untrained_levels_treatment_method=unknown_categorical_method_pass,
        apply_pca=pca,
        pca_method=pca_method_pass,
        pca_variance_retained_or_number_of_components=pca_components_pass,
        apply_zero_nearZero_variance=ignore_low_variance,
        club_rare_levels=combine_rare_levels,
        rara_level_threshold_percentage=rare_level_threshold,
        apply_binning=apply_binning_pass,
        features_to_binn=features_to_bin_pass,
        remove_outliers=remove_outliers,
        outlier_contamination_percentage=outliers_threshold,
        outlier_methods=["pca"],
        remove_multicollinearity=remove_multicollinearity,
        maximum_correlation_between_features=multicollinearity_threshold,
        remove_perfect_collinearity=remove_perfect_collinearity,
        cluster_entire_data=create_clusters,
        range_of_clusters_to_try=cluster_iter,
        apply_polynomial_trigonometry_features=polynomial_features,
        max_polynomial=polynomial_degree,
        trigonometry_calculations=trigonometry_features_pass,
        top_poly_trig_features_to_select_percentage=polynomial_threshold,
        apply_grouping=apply_grouping_pass,
        features_to_group_ListofList=group_features_pass,
        group_name=group_names_pass,
        apply_feature_selection=feature_selection,
        feature_selection_top_features_percentage=feature_selection_threshold,
        feature_selection_method=feature_selection_method,
        apply_feature_interactions=apply_feature_interactions_pass,
        feature_interactions_to_apply=interactions_to_apply_pass,
        feature_interactions_top_features_to_select_percentage=interaction_threshold,
        display_types=display_dtypes_pass,  # this is for inferred input box
        target_transformation=False,  # not needed for classification
        random_state=seed,
    )

    dtypes = prep_pipe.named_steps["dtypes"]

    display.move_progress()
    logger.info("Preprocessing pipeline created successfully")

    try:
        res_type = ["quit", "Quit", "exit", "EXIT", "q", "Q", "e", "E", "QUIT", "Exit"]
        res = dtypes.response

        if res in res_type:
            sys.exit(
                "(Process Exit): setup has been interupted with user command 'quit'. setup must rerun."
            )

    except:
        logger.error(
            "(Process Exit): setup has been interupted with user command 'quit'. setup must rerun."
        )

    if not preprocess:
        prep_pipe.steps = prep_pipe.steps[:1]

    """
    preprocessing ends here
    """

    # reset pandas option
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")

    logger.info("Creating global containers")

    # create an empty list for pickling later.
    experiment__ = []

    # CV params
    fold_param = fold

    if fold_groups is not None and isinstance(fold_groups, str):
        fold_groups_param = X_before_preprocess[fold_groups]
    else:
        fold_groups_param = fold_groups
    fold_shuffle_param = fold_shuffle

    from sklearn.model_selection import (
        StratifiedKFold,
        KFold,
        GroupKFold,
        TimeSeriesSplit,
    )

    if fold_strategy == "kfold":
        fold_generator = KFold(
            fold_param, random_state=seed, shuffle=fold_shuffle_param
        )
    elif fold_strategy == "stratifiedkfold":
        fold_generator = StratifiedKFold(
            fold_param, random_state=seed, shuffle=fold_shuffle_param
        )
    elif fold_strategy == "groupkfold":
        fold_generator = GroupKFold(fold_param)
    elif fold_strategy == "timeseries":
        fold_generator = TimeSeriesSplit(fold_param)
    else:
        fold_generator = fold_strategy

    # create n_jobs_param
    n_jobs_param = n_jobs

    cuml_version = None
    if use_gpu:
        try:
            from cuml import __version__

            cuml_version = __version__
            logger.info(f"cuml=={cuml_version}")

            cuml_version = cuml_version.split(".")
            cuml_version = (int(cuml_version[0]), int(cuml_version[1]))
        except:
            logger.warning(f"cuML not found")

        if not cuml_version >= (0, 15):
            message = f"cuML is outdated or not found. Required version is >=0.15, got {__version__}"
            if use_gpu == "Force":
                raise ImportError(message)
            else:
                logger.warning(message)

    # create gpu_n_jobs_param
    gpu_n_jobs_param = n_jobs if not use_gpu else 1

    # create create_model_container
    create_model_container = []

    # create master_model_container
    master_model_container = []

    # create display container
    display_container = []

    # create logging parameter
    logging_param = log_experiment

    # create exp_name_log param incase logging is False
    exp_name_log = "no_logging"

    # create an empty log_plots_param
    if log_plots:
        log_plots_param = True
    else:
        log_plots_param = False

    # add custom transformers to prep pipe
    if custom_pipeline:
        custom_steps = normalize_custom_transformers(custom_pipeline)
        _internal_pipeline.extend(custom_steps)

    # create a fix_imbalance_param and fix_imbalance_method_param
    fix_imbalance_param = fix_imbalance and preprocess
    fix_imbalance_method_param = fix_imbalance_method

    if fix_imbalance_method_param is None:
        fix_imbalance_model_name = "SMOTE"

    if fix_imbalance_param:
        if fix_imbalance_method_param is None:
            import six

            sys.modules["sklearn.externals.six"] = six
            from imblearn.over_sampling import SMOTE

            fix_imbalance_resampler = SMOTE(random_state=seed)
        else:
            fix_imbalance_model_name = str(fix_imbalance_method_param).split("(")[0]
            fix_imbalance_resampler = fix_imbalance_method_param
        _internal_pipeline.append(("fix_imbalance", fix_imbalance_resampler))

    prep_pipe.steps.extend(
        [step for step in _internal_pipeline if hasattr(step, "transform")]
    )

    _internal_pipeline = make_internal_pipeline(_internal_pipeline)

    logger.info(f"Internal pipeline: {_internal_pipeline}")

    # create target_param var
    target_param = target

    # create gpu_param var
    gpu_param = use_gpu

    # create stratify_param var
    stratify_param = data_split_stratify

    # determining target type
    if _is_multiclass():
        target_type = "Multiclass"
    else:
        target_type = "Binary"

    lr = LogisticRegressionClassifierContainer(globals())

    # sample estimator
    if sample_estimator is None:
        model = lr
    else:
        model = sample_estimator

    display.move_progress()

    if sampling and data.shape[0] > 25000:  # change this back to 25000

        X_train, X_test, y_train, y_test = _sample_data(
            model, seed, train_size, data_split_shuffle, display
        )

    else:
        display.update_monitor(1, "Splitting Data")
        display.display_monitor()

        _stratify_columns = _get_columns_to_stratify_by(
            X_before_preprocess, y_before_preprocess, stratify_param, target
        )

        if test_data is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_before_preprocess,
                y_before_preprocess,
                test_size=1 - train_size,
                stratify=_stratify_columns,
                random_state=seed,
                shuffle=data_split_shuffle,
            )
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

        train_data = prep_pipe.fit_transform(train_data)
        # workaround to also transform target
        dtypes.final_training_columns.append(target)
        test_data = prep_pipe.transform(test_data)

        X_train = train_data.drop(target, axis=1)
        y_train = train_data[target]

        X_test = test_data.drop(target, axis=1)
        y_test = test_data[target]

        if fold_groups_param is not None:
            fold_groups_param = fold_groups_param[
                fold_groups_param.index.isin(X_train.index)
            ]

        display.move_progress()

    data = prep_pipe.transform(data)
    X = data.drop(target, axis=1)
    y = data[target]
    try:
        dtypes.final_training_columns.remove(target)
    except:
        pass

    all_models = models(force_regenerate=True, raise_errors=True)
    _all_models_internal = models(
        internal=True, force_regenerate=True, raise_errors=True
    )
    all_metrics = get_metrics()

    """
    Final display Starts
    """
    logger.info("Creating grid variables")

    if hasattr(dtypes, "replacement"):
        label_encoded = dtypes.replacement
        label_encoded = str(label_encoded).replace("'", "")
        label_encoded = str(label_encoded).replace("{", "")
        label_encoded = str(label_encoded).replace("}", "")

    else:
        label_encoded = "None"

    # generate values for grid show
    missing_values = data_before_preprocess.isna().sum().sum()
    missing_flag = True if missing_values > 0 else False

    normalize_grid = normalize_method if normalize else "None"

    transformation_grid = transformation_method if transformation else "None"

    pca_method_grid = pca_method if pca else "None"

    pca_components_grid = pca_components_pass if pca else "None"

    rare_level_threshold_grid = rare_level_threshold if combine_rare_levels else "None"

    numeric_bin_grid = False if bin_numeric_features is None else True

    outliers_threshold_grid = outliers_threshold if remove_outliers else None

    multicollinearity_threshold_grid = (
        multicollinearity_threshold if remove_multicollinearity else None
    )

    cluster_iter_grid = cluster_iter if create_clusters else None

    polynomial_degree_grid = polynomial_degree if polynomial_features else None

    polynomial_threshold_grid = (
        polynomial_threshold if polynomial_features or trigonometry_features else None
    )

    feature_selection_threshold_grid = (
        feature_selection_threshold if feature_selection else None
    )

    interaction_threshold_grid = (
        interaction_threshold if feature_interaction or feature_ratio else None
    )

    ordinal_features_grid = False if ordinal_features is None else True

    unknown_categorical_method_grid = (
        unknown_categorical_method if handle_unknown_categorical else None
    )

    group_features_grid = False if group_features is None else True

    high_cardinality_features_grid = (
        False if high_cardinality_features is None else True
    )

    high_cardinality_method_grid = (
        high_cardinality_method if high_cardinality_features_grid else None
    )

    learned_types = dtypes.learent_dtypes
    learned_types.drop(target, inplace=True)

    float_type = 0
    cat_type = 0

    for i in dtypes.learent_dtypes:
        if "float" in str(i):
            float_type += 1
        elif "object" in str(i):
            cat_type += 1
        elif "int" in str(i):
            float_type += 1

    if profile:
        print("Setup Succesfully Completed! Loading Profile Now... Please Wait!")
    else:
        if verbose:
            print("Setup Succesfully Completed!")

    functions = pd.DataFrame(
        [
            ["session_id ", seed],
            ["Target Type ", target_type],
            ["Label Encoded ", label_encoded],
            ["Original Data ", data_before_preprocess.shape],
            ["Missing Values ", missing_flag],
            ["Numeric Features ", str(float_type)],
            ["Categorical Features ", str(cat_type)],
        ]
        + (
            [
                ["Ordinal Features ", ordinal_features_grid],
                ["High Cardinality Features ", high_cardinality_features_grid],
                ["High Cardinality Method ", high_cardinality_method_grid],
            ]
            if preprocess
            else []
        )
        + [
            [
                "Sampled Data ",
                f"({X_train.shape[0] + X_test.shape[0]}, {data_before_preprocess.shape[1]})",
            ],
            ["Transformed Train Set ", X_train.shape],
            ["Transformed Test Set ", X_test.shape],
        ]
        + (
            [
                ["Numeric Imputer ", numeric_imputation],
                ["Categorical Imputer ", categorical_imputation],
                ["Unknown Categoricals Handling ", unknown_categorical_method_grid],
                ["Normalize ", normalize],
                ["Normalize Method ", normalize_grid],
                ["Transformation ", transformation],
                ["Transformation Method ", transformation_grid],
                ["PCA ", pca],
                ["PCA Method ", pca_method_grid],
                ["PCA Components ", pca_components_grid],
                ["Ignore Low Variance ", ignore_low_variance],
                ["Combine Rare Levels ", combine_rare_levels],
                ["Rare Level Threshold ", rare_level_threshold_grid],
                ["Numeric Binning ", numeric_bin_grid],
                ["Remove Outliers ", remove_outliers],
                ["Outliers Threshold ", outliers_threshold_grid],
                ["Remove Multicollinearity ", remove_multicollinearity],
                ["Multicollinearity Threshold ", multicollinearity_threshold_grid],
                ["Clustering ", create_clusters],
                ["Clustering Iteration ", cluster_iter_grid],
                ["Polynomial Features ", polynomial_features],
                ["Polynomial Degree ", polynomial_degree_grid],
                ["Trignometry Features ", trigonometry_features],
                ["Polynomial Threshold ", polynomial_threshold_grid],
                ["Group Features ", group_features_grid],
                ["Feature Selection ", feature_selection],
                ["Features Selection Threshold ", feature_selection_threshold_grid],
                ["Feature Interaction ", feature_interaction],
                ["Feature Ratio ", feature_ratio],
                ["Interaction Threshold ", interaction_threshold_grid],
                ["Fix Imbalance ", fix_imbalance_param],
                ["Fix Imbalance Method ", fix_imbalance_model_name],
            ]
            if preprocess
            else []
        )
        + [
            ["CV Generator ", type(fold_generator).__name__],
            ["CV Folds ", fold_param],
        ],
        columns=["Description", "Value"],
    )

    functions_ = functions.style.apply(highlight_max)

    display.display(functions_, clear=True)

    if profile:
        try:
            import pandas_profiling

            pf = pandas_profiling.ProfileReport(data_before_preprocess)
            display.display(pf, clear=True)
        except:
            print(
                "Data Profiler Failed. No output to show, please continue with Modeling."
            )
            logger.error(
                "Data Profiler Failed. No output to show, please continue with Modeling."
            )

    """
    Final display Ends
    """

    # log into experiment
    experiment__.append(("Classification Setup Config", functions))
    experiment__.append(("X_training Set", X_train))
    experiment__.append(("y_training Set", y_train))
    experiment__.append(("X_test Set", X_test))
    experiment__.append(("y_test Set", y_test))
    experiment__.append(("Transformation Pipeline", prep_pipe))

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    if logging_param:

        logger.info("Logging experiment in MLFlow")

        import mlflow

        if experiment_name is None:
            exp_name_ = "clf-default-name"
        else:
            exp_name_ = experiment_name

        URI = secrets.token_hex(nbytes=4)
        exp_name_log = exp_name_

        try:
            mlflow.create_experiment(exp_name_log)
        except:
            pass

        # mlflow logging
        mlflow.set_experiment(exp_name_log)

        run_name_ = f"Session Initialized {USI}"

        with mlflow.start_run(run_name=run_name_) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            k = functions.copy()
            k.set_index("Description", drop=True, inplace=True)
            kdict = k.to_dict()
            params = kdict.get("Value")
            mlflow.log_params(params)

            # set tag of compare_models
            mlflow.set_tag("Source", "setup")

            import secrets

            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log the transformation pipeline
            logger.info(
                "SubProcess save_model() called =================================="
            )
            save_model(prep_pipe, "Transformation Pipeline", verbose=False)
            logger.info(
                "SubProcess save_model() end =================================="
            )
            mlflow.log_artifact("Transformation Pipeline.pkl")
            os.remove("Transformation Pipeline.pkl")

            # Log pandas profile
            if log_profile:
                import pandas_profiling

                pf = pandas_profiling.ProfileReport(data_before_preprocess)
                pf.to_file("Data Profile.html")
                mlflow.log_artifact("Data Profile.html")
                os.remove("Data Profile.html")
                display.display(functions_, clear=True)

            # Log training and testing set
            if log_data:
                X_train.join(y_train).to_csv("Train.csv")
                X_test.join(y_test).to_csv("Test.csv")
                mlflow.log_artifact("Train.csv")
                mlflow.log_artifact("Test.csv")
                os.remove("Train.csv")
                os.remove("Test.csv")

    logger.info(f"create_model_container: {len(create_model_container)}")
    logger.info(f"master_model_container: {len(master_model_container)}")
    logger.info(f"display_container: {len(display_container)}")

    logger.info(str(prep_pipe))
    logger.info("setup() succesfully completed......................................")

    gc.collect()

    return (
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test,
        seed,
        prep_pipe,
        experiment__,
        fold_shuffle_param,
        n_jobs_param,
        html_param,
        create_model_container,
        master_model_container,
        display_container,
        exp_name_log,
        logging_param,
        log_plots_param,
        USI,
        fix_imbalance_param,
        fix_imbalance_method_param,
        logger,
        data_before_preprocess,
        target_param,
        gpu_param,
        gpu_n_jobs_param,
        stratify_param,
        fold_generator,
        fold_param,
        fold_groups_param,
    )


def compare_models(
    include: Optional[
        List[Union[str, Any]]
    ] = None,  # changed whitelist to include in pycaret==2.1
    exclude: Optional[List[str]] = None,  # changed blacklist to exclude in pycaret==2.1
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    sort: str = "Accuracy",
    n_select: int = 1,
    budget_time: Optional[float] = None,  # added in pycaret==2.1.0
    turbo: bool = True,
    errors: str = "ignore",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    display: Optional[Display] = None,
) -> List[Any]:

    """
    This function train all the models available in the model library and scores them 
    using Cross Validation. The output prints a score grid with Accuracy, 
    AUC, Recall, Precision, F1, Kappa and MCC (averaged across folds).
    
    This function returns all of the models compared, sorted by the value of the selected metric.

    When turbo is set to True ('rbfsvm', 'gpc' and 'mlp') are excluded due to longer
    training time. By default turbo param is set to True.

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> best_model = compare_models() 

    This will return the averaged score grid of all the models except 'rbfsvm', 'gpc' 
    and 'mlp'. When turbo param is set to False, all models including 'rbfsvm', 'gpc' 
    and 'mlp' are used but this may result in longer training time.
    
    >>> best_model = compare_models( exclude = [ 'knn', 'gbc' ] , turbo = False) 

    This will return a comparison of all models except K Nearest Neighbour and
    Gradient Boosting Classifier.
    
    >>> best_model = compare_models( exclude = [ 'knn', 'gbc' ] , turbo = True) 

    This will return comparison of all models except K Nearest Neighbour, 
    Gradient Boosting Classifier, SVM (RBF), Gaussian Process Classifier and
    Multi Level Perceptron.
        

    >>> tuned_model = tune_model(create_model('lr'))
    >>> best_model = compare_models( include = [ 'lr', tuned_model ]) 

    This will compare a tuned Linear Regression model with an untuned one.

    Parameters
    ----------
    exclude: list of strings, default = None
        In order to omit certain models from the comparison model ID's can be passed as 
        a list of strings in exclude param. 

    include: list of strings or objects, default = None
        In order to run only certain models for the comparison, the model ID's can be 
        passed as a list of strings in include param. The list can also include estimator
        objects to be compared.

    fold: integer or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.
  
    cross_validation: bool, default = True
        When cross_validation set to False fold parameter is ignored and models are trained
        on entire training dataset, returning metrics calculated using the train (holdout) set.

    sort: str, default = 'Accuracy'
        The scoring measure specified is used for sorting the average score grid
        Other options are 'AUC', 'Recall', 'Precision', 'F1', 'Kappa' and 'MCC'.

    n_select: int, default = 1
        Number of top_n models to return. use negative argument for bottom selection.
        for example, n_select = -3 means bottom 3 models.

    budget_time: int or float, default = None
        If not 0 or None, will terminate execution of the function after budget_time 
        minutes have passed and return results up to that point.

    turbo: bool, default = True
        When turbo is set to True, it excludes estimators that have longer
        training time.

    errors: str, default = 'ignore'
        If 'ignore', will suppress model exceptions and continue.
        If 'raise', will allow exceptions to be raised.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model. The parameters will be applied to all models,
        therefore it is recommended to set errors parameter to 'ignore'.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.
    
    Returns
    -------
    score_grid
        A table containing the scores of the model across the kfolds. 
        Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are also returned.

    list
        List of fitted model objects that were compared.

    Warnings
    --------
    - compare_models() though attractive, might be time consuming with large 
      datasets. By default turbo is set to True, which excludes models that
      have longer training times. Changing turbo parameter to False may result 
      in very high training times with datasets where number of samples exceed 
      10,000.

    - If target variable is multiclass (more than 2 classes), AUC will be 
      returned as zero (0.0)

    - If cross_validation param is set to False, no models will be logged with MLFlow.

    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing compare_models()")
    logger.info(f"compare_models({function_params_str})")

    logger.info("Checking exceptions")

    if not fit_kwargs:
        fit_kwargs = {}

    # checking error for exclude (string)
    available_estimators = all_models.index

    if exclude != None:
        for i in exclude:
            if i not in available_estimators:
                raise ValueError(
                    f"Estimator Not Available {i}. Please see docstring for list of available estimators."
                )

    if include != None:
        for i in include:
            if isinstance(i, str):
                if i not in available_estimators:
                    raise ValueError(
                        f"Estimator {i} Not Available. Please see docstring for list of available estimators."
                    )
            elif not hasattr(i, "fit"):
                raise ValueError(
                    f"Estimator {i} does not have the required fit() method."
                )

    # include and exclude together check
    if include is not None and exclude is not None:
        raise TypeError(
            "Cannot use exclude parameter when include is used to compare models."
        )

    # checking fold parameter
    if fold is not None and not (type(fold) is int or is_sklearn_cv_generator(fold)):
        raise TypeError(
            "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
        )

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    # checking budget_time parameter
    if budget_time and type(budget_time) is not int and type(budget_time) is not float:
        raise TypeError("budget_time parameter only accepts integer or float values.")

    # checking sort parameter
    if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
        sort = _get_metric(sort)
        if sort is None:
            raise ValueError(
                f"Sort method not supported. See docstring for list of available parameters."
            )

    # checking errors parameter
    possible_errors = ["ignore", "raise"]
    if errors not in possible_errors:
        raise ValueError(
            f"errors parameter must be one of: {', '.join(possible_errors)}."
        )

    # checking optimize parameter for multiclass
    if _is_multiclass():
        if not sort["Multiclass"]:
            raise TypeError(
                f"{sort} metric not supported for multiclass problems. See docstring for list of other optimization parameters."
            )

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    fold = _get_cv_splitter(fold)

    groups = _get_groups(groups)

    pd.set_option("display.max_columns", 500)

    logger.info("Preparing display monitor")

    len_mod = len(all_models[all_models["Turbo"] == True]) if turbo else len(all_models)

    if include:
        len_mod = len(include)
    elif exclude:
        len_mod -= len(exclude)

    if not display:
        progress_args = {"max": (4 * len_mod) + 4 + len_mod}
        master_display_columns = (
            ["Model"] + all_metrics["Display Name"].to_list() + ["TT (Sec)"]
        )
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
            ["Estimator", ". . . . . . . . . . . . . . . . . .", "Compiling Library"],
        ]
        display = Display(
            verbose=verbose,
            html_param=html_param,
            progress_args=progress_args,
            master_display_columns=master_display_columns,
            monitor_rows=monitor_rows,
        )

        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    np.random.seed(seed)

    display.move_progress()

    # defining sort parameter (making Precision equivalent to Prec. )

    if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
        sort = sort["Display Name"]
    else:
        sort = "TT (Sec)"

    sort_ascending = sort == "TT (Sec)"

    """
    MONITOR UPDATE STARTS
    """

    display.update_monitor(1, "Loading Estimator")
    display.display_monitor()

    """
    MONITOR UPDATE ENDS
    """

    if include:
        model_library = include
    else:
        if turbo:
            model_library = models()
            model_library = list(model_library[model_library["Turbo"] == True].index)
        else:
            model_library = list(models().index)
        if exclude:
            model_library = [x for x in model_library if x not in exclude]

    display.move_progress()

    # create URI (before loop)
    import secrets

    URI = secrets.token_hex(nbytes=4)

    master_display = None

    total_runtime_start = time.time()
    total_runtime = 0
    over_time_budget = False
    if budget_time and budget_time > 0:
        logger.info(f"Time budget is {budget_time} minutes")

    for i, model in enumerate(model_library):

        model_name = _get_model_name(model)

        if isinstance(model, str):
            logger.info(f"Initializing {model_name}")
        else:
            logger.info(f"Initializing custom model {model_name}")

        # run_time
        runtime_start = time.time()
        total_runtime += (runtime_start - total_runtime_start) / 60
        logger.info(f"Total runtime is {total_runtime} minutes")
        over_time_budget = (
            budget_time and budget_time > 0 and total_runtime > budget_time
        )
        if over_time_budget:
            logger.info(
                f"Total runtime {total_runtime} is over time budget by {total_runtime - budget_time}, breaking loop"
            )
            break
        total_runtime_start = runtime_start

        display.move_progress()

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(2, model_name)
        display.display_monitor()

        """
        MONITOR UPDATE ENDS
        """
        display.replace_master_display(None)

        logger.info(
            "SubProcess create_model() called =================================="
        )
        if errors == "raise":
            model, model_fit_time = _create_model(
                estimator=model,
                system=False,
                verbose=False,
                display=display,
                fold=fold,
                round=round,
                cross_validation=cross_validation,
                fit_kwargs=fit_kwargs,
                groups=groups,
                refit=False,
            )
            model_results = pull(pop=True)
        else:
            try:
                model, model_fit_time = _create_model(
                    estimator=model,
                    system=False,
                    verbose=False,
                    display=display,
                    fold=fold,
                    round=round,
                    cross_validation=cross_validation,
                    fit_kwargs=fit_kwargs,
                    groups=groups,
                    refit=False,
                )
                model_results = pull(pop=True)
                assert np.sum(model_results.iloc[0]) != 0.0
            except:
                logger.error(
                    f"create_model() for {model} raised an exception or returned all 0.0, trying without fit_kwargs:"
                )
                logger.error(traceback.format_exc())
                try:
                    model, model_fit_time = _create_model(
                        estimator=model,
                        system=False,
                        verbose=False,
                        display=display,
                        fold=fold,
                        round=round,
                        cross_validation=cross_validation,
                        groups=groups,
                        refit=False,
                    )
                    model_results = pull(pop=True)
                except:
                    logger.error(f"create_model() for {model} raised an exception:")
                    logger.error(traceback.format_exc())
                continue
        logger.info("SubProcess create_model() end ==================================")

        if model is None:
            over_time_budget = True
            logger.info(f"Time budged exceeded in create_model(), breaking loop")
            break

        logger.info("Creating metrics dataframe")
        if cross_validation:
            compare_models_ = pd.DataFrame(model_results.loc["Mean"]).T
        else:
            compare_models_ = pd.DataFrame(model_results.iloc[0]).T
        compare_models_.insert(len(compare_models_.columns), "TT (Sec)", model_fit_time)
        compare_models_.insert(0, "Model", model_name)
        compare_models_.insert(0, "Object", [model])
        compare_models_.insert(0, "index", [i])
        compare_models_.set_index("index", drop=True, inplace=True)
        if master_display is None:
            master_display = compare_models_
        else:
            master_display = pd.concat(
                [master_display, compare_models_], ignore_index=True
            )
        master_display = master_display.round(round)
        master_display = master_display.sort_values(by=sort, ascending=sort_ascending)
        master_display.reset_index(drop=True, inplace=True)

        master_display_ = master_display.drop("Object", axis=1).style.set_precision(
            round
        )
        master_display_ = master_display_.set_properties(**{"text-align": "left"})
        master_display_ = master_display_.set_table_styles(
            [dict(selector="th", props=[("text-align", "left")])]
        )

        display.replace_master_display(master_display_)

        display.display_master_display()
        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

    display.move_progress()

    def highlight_max(s):
        to_highlight = s == s.max()
        return ["background-color: yellow" if v else "" for v in to_highlight]

    def highlight_cols(s):
        color = "lightgrey"
        return f"background-color: {color}"

    compare_models_ = (
        master_display.drop("Object", axis=1)
        .style.apply(highlight_max, subset=master_display.columns[2:],)
        .applymap(highlight_cols, subset=["TT (Sec)"])
    )
    compare_models_ = compare_models_.set_precision(round)
    compare_models_ = compare_models_.set_properties(**{"text-align": "left"})
    compare_models_ = compare_models_.set_table_styles(
        [dict(selector="th", props=[("text-align", "left")])]
    )

    display.move_progress()

    display.update_monitor(1, "Compiling Final Models")
    display.display_monitor()

    sorted_models = master_display["Object"].to_list()

    if n_select < 0:
        sorted_models = sorted_models[n_select:]
    else:
        sorted_models = sorted_models[:n_select]

    def create_model_and_update_display(model, i, fold, round, fit_kwargs, groups):
        display.update_monitor(2, _get_model_name(model))
        display.display_monitor()
        model, model_fit_time = _create_model(
            estimator=model,
            system=False,
            verbose=False,
            fold=fold,
            round=round,
            cross_validation=False,
            predict=False,
            fit_kwargs=fit_kwargs,
            groups=groups,
        )

        if logging_param and cross_validation:

            avgs_dict_log = {
                k: v
                for k, v in compare_models_.data.drop(
                    ["Object", "Model", "TT (Sec)"], errors="ignore", axis=1
                )
                .iloc[i]
                .items()
            }

            try:
                _mlflow_log_model(
                    model=model,
                    model_results=model_results,
                    score_dict=avgs_dict_log,
                    source="compare_models",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=prep_pipe,
                    log_plots=False,
                    URI=URI,
                    display=display,
                )
            except:
                logger.error(f"_mlflow_log_model() for {model} raised an exception:")
                logger.error(traceback.format_exc())
        return model

    sorted_models = [
        create_model_and_update_display(model, i, fold, round, fit_kwargs, groups)
        for i, model in enumerate(sorted_models)
    ]

    if len(sorted_models) == 1:
        sorted_models = sorted_models[0]

    display.display(compare_models_, clear=True)

    pd.reset_option("display.max_columns")

    # store in display container
    display_container.append(compare_models_.data)

    logger.info(f"create_model_container: {len(create_model_container)}")
    logger.info(f"master_model_container: {len(master_model_container)}")
    logger.info(f"display_container: {len(display_container)}")

    logger.info(str(sorted_models))
    logger.info(
        "compare_models() succesfully completed......................................"
    )

    return sorted_models


def create_model(
    estimator,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    display: Optional[Display] = None,  # added in pycaret==2.2.0
    **kwargs,
) -> Any:
    """  
    This function creates a model and scores it using Cross Validation. 
    The output prints a score grid that shows Accuracy, AUC, Recall, Precision, 
    F1, Kappa and MCC by fold (default = 10 Fold). 

    This function returns a trained model object. 

    setup() function must be called before using create_model()

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')

    This will create a trained Logistic Regression model.

    Parameters
    ----------
    estimator : str / object, default = None
        Enter ID of the estimators available in model library or pass an untrained model 
        object consistent with fit / predict API to train and evaluate model. All 
        estimators support binary or multiclass problem. List of estimators in model 
        library (ID - Name):

        * 'lr' - Logistic Regression             
        * 'knn' - K Nearest Neighbour            
        * 'nb' - Naive Bayes             
        * 'dt' - Decision Tree Classifier                   
        * 'svm' - SVM - Linear Kernel	            
        * 'rbfsvm' - SVM - Radial Kernel               
        * 'gpc' - Gaussian Process Classifier                  
        * 'mlp' - Multi Level Perceptron                  
        * 'ridge' - Ridge Classifier                
        * 'rf' - Random Forest Classifier                   
        * 'qda' - Quadratic Discriminant Analysis                  
        * 'ada' - Ada Boost Classifier                 
        * 'gbc' - Gradient Boosting Classifier                  
        * 'lda' - Linear Discriminant Analysis                  
        * 'et' - Extra Trees Classifier                   
        * 'xgboost' - Extreme Gradient Boosting              
        * 'lightgbm' - Light Gradient Boosting              
        * 'catboost' - CatBoost Classifier             

    fold: integer or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    cross_validation: bool, default = True
        When cross_validation set to False fold parameter is ignored and model is trained
        on entire training dataset, returning metrics calculated using the train (holdout) set.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.

    **kwargs: 
        Additional keyword arguments to pass to the estimator.

    Returns
    -------
    score_grid
        A table containing the scores of the model across the kfolds. 
        Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are highlighted in yellow.

    model
        trained model object

    Warnings
    --------
    - 'svm' and 'ridge' doesn't support predict_proba method. As such, AUC will be
      returned as zero (0.0)
     
    - If target variable is multiclass (more than 2 classes), AUC will be returned 
      as zero (0.0)

    - 'rbfsvm' and 'gpc' uses non-linear kernel and hence the fit time complexity is 
      more than quadratic. These estimators are hard to scale on datasets with more 
      than 10,000 samples.

    - If cross_validation param is set to False, model will not be logged with MLFlow.

    """
    return _create_model(
        estimator,
        fold=fold,
        round=round,
        cross_validation=cross_validation,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
        display=display,
        **kwargs,
    )


def _create_model(
    estimator,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    predict: bool = True,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    refit: bool = True,
    verbose: bool = True,
    system: bool = True,
    X_train_data: Optional[pd.DataFrame] = None,  # added in pycaret==2.2.0
    y_train_data: Optional[pd.DataFrame] = None,  # added in pycaret==2.2.0
    display: Optional[Display] = None,  # added in pycaret==2.2.0
    **kwargs,
) -> Any:

    """  
    This is an internal version of the create_model function.

    This function creates a model and scores it using Cross Validation. 
    The output prints a score grid that shows Accuracy, AUC, Recall, Precision, 
    F1, Kappa and MCC by fold (default = 10 Fold). 

    This function returns a trained model object. 

    setup() function must be called before using create_model()

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')

    This will create a trained Logistic Regression model.

    Parameters
    ----------
    estimator : str / object, default = None
        Enter ID of the estimators available in model library or pass an untrained model 
        object consistent with fit / predict API to train and evaluate model. All 
        estimators support binary or multiclass problem. List of estimators in model 
        library (ID - Name):

        * 'lr' - Logistic Regression             
        * 'knn' - K Nearest Neighbour            
        * 'nb' - Naive Bayes             
        * 'dt' - Decision Tree Classifier                   
        * 'svm' - SVM - Linear Kernel	            
        * 'rbfsvm' - SVM - Radial Kernel               
        * 'gpc' - Gaussian Process Classifier                  
        * 'mlp' - Multi Level Perceptron                  
        * 'ridge' - Ridge Classifier                
        * 'rf' - Random Forest Classifier                   
        * 'qda' - Quadratic Discriminant Analysis                  
        * 'ada' - Ada Boost Classifier                 
        * 'gbc' - Gradient Boosting Classifier                  
        * 'lda' - Linear Discriminant Analysis                  
        * 'et' - Extra Trees Classifier                   
        * 'xgboost' - Extreme Gradient Boosting              
        * 'lightgbm' - Light Gradient Boosting              
        * 'catboost' - CatBoost Classifier             

    fold: integer or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    cross_validation: bool, default = True
        When cross_validation set to False fold parameter is ignored and model is trained
        on entire training dataset.

    predict: bool, default = True
        Whether to predict model on holdout if cross_validation == False.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    refit: bool, default = True
        Whether to refit the model on the entire dataset after CV. Ignored if cross_validation == False.

    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.

    system: bool, default = True
        Must remain True all times. Only to be changed by internal functions.
        If False, method will return a tuple of model and the model fit time.

    X_train_data: pandas.DataFrame, default = None
        If not None, will use this dataframe as training features.
        Intended to be only changed by internal functions.

    y_train_data: pandas.DataFrame, default = None
        If not None, will use this dataframe as training target.
        Intended to be only changed by internal functions.

    **kwargs: 
        Additional keyword arguments to pass to the estimator.

    Returns
    -------
    score_grid
        A table containing the scores of the model across the kfolds. 
        Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are highlighted in yellow.

    model
        trained model object

    Warnings
    --------
    - 'svm' and 'ridge' doesn't support predict_proba method. As such, AUC will be
      returned as zero (0.0)
     
    - If target variable is multiclass (more than 2 classes), AUC will be returned 
      as zero (0.0)

    - 'rbfsvm' and 'gpc' uses non-linear kernel and hence the fit time complexity is 
      more than quadratic. These estimators are hard to scale on datasets with more 
      than 10,000 samples.

    - If cross_validation param is set to False, model will not be logged with MLFlow.

    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing create_model()")
    logger.info(f"create_model({function_params_str})")

    logger.info("Checking exceptions")

    # run_time
    runtime_start = time.time()

    available_estimators = set(_all_models_internal.index)

    if not fit_kwargs:
        fit_kwargs = {}

    # only raise exception of estimator is of type string.
    if isinstance(estimator, str):
        if estimator not in available_estimators:
            raise ValueError(
                f"Estimator {estimator} not available. Please see docstring for list of available estimators."
            )
    elif not hasattr(estimator, "fit"):
        raise ValueError(
            f"Estimator {estimator} does not have the required fit() method."
        )

    # checking fold parameter
    if fold is not None and not (type(fold) is int or is_sklearn_cv_generator(fold)):
        raise TypeError(
            "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
        )

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    # checking verbose parameter
    if type(verbose) is not bool:
        raise TypeError("Verbose parameter can only take argument as True or False.")

    # checking system parameter
    if type(system) is not bool:
        raise TypeError("System parameter can only take argument as True or False.")

    # checking cross_validation parameter
    if type(cross_validation) is not bool:
        raise TypeError(
            "cross_validation parameter can only take argument as True or False."
        )

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    groups = _get_groups(groups)

    if not display:
        progress_args = {"max": 4}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
        ]
        display = Display(
            verbose=verbose,
            html_param=html_param,
            progress_args=progress_args,
            master_display_columns=master_display_columns,
            monitor_rows=monitor_rows,
        )
        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    logger.info("Importing libraries")

    # general dependencies

    np.random.seed(seed)

    logger.info("Copying training dataset")

    # Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy() if X_train_data is None else X_train_data.copy()
    data_y = y_train.copy() if y_train_data is None else y_train_data.copy()

    # reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)

    display.move_progress()

    logger.info("Defining folds")

    # cross validation setup starts here
    cv = _get_cv_splitter(fold)

    logger.info("Declaring metric variables")

    """
    MONITOR UPDATE STARTS
    """
    display.update_monitor(1, "Selecting Estimator")
    display.display_monitor()
    """
    MONITOR UPDATE ENDS
    """

    logger.info("Importing untrained model")

    if isinstance(estimator, str) and estimator in available_estimators:
        model_definition = _all_models_internal.loc[estimator]
        model_args = model_definition["Args"]
        model_args = {**model_args, **kwargs}
        model = model_definition["Class"](**model_args)
        full_name = model_definition["Name"]
    else:
        logger.info("Declaring custom model")

        model = clone(estimator)
        model.set_params(**kwargs)

        full_name = _get_model_name(model)

    logger.info(f"{full_name} Imported succesfully")

    display.move_progress()

    """
    MONITOR UPDATE STARTS
    """
    if not cross_validation:
        display.update_monitor(1, f"Fitting {str(full_name)}")
    else:
        display.update_monitor(1, "Initializing CV")

    display.display_monitor()
    """
    MONITOR UPDATE ENDS
    """

    if not cross_validation:

        with estimator_pipeline(_internal_pipeline, model) as pipeline_with_model:
            fit_kwargs = _get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
            logger.info("Cross validation set to False")

            logger.info("Fitting Model")
            model_fit_start = time.time()
            with io.capture_output():
                pipeline_with_model.fit(data_X, data_y, **fit_kwargs)
            model_fit_end = time.time()

            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

            display.move_progress()

            if predict:
                predict_model(pipeline_with_model, verbose=False)
                model_results = pull(pop=True).drop("Model", axis=1)

                display_container.append(model_results)

                display.display(
                    model_results, clear=system, override=False if not system else None
                )

                logger.info(f"display_container: {len(display_container)}")

        display.move_progress()

        logger.info(str(model))
        logger.info(
            "create_models() succesfully completed......................................"
        )

        gc.collect()

        if not system:
            return (model, model_fit_time)
        return model

    """
    MONITOR UPDATE STARTS
    """
    display.update_monitor(1, f"Fitting {_get_cv_n_folds(fold)} Folds")
    display.display_monitor()
    """
    MONITOR UPDATE ENDS
    """

    from sklearn.model_selection import cross_validate

    metrics_dict = dict(zip(all_metrics.index, all_metrics["Scorer"]))

    logger.info("Starting cross validation")

    n_jobs = gpu_n_jobs_param
    from sklearn.gaussian_process import GaussianProcessClassifier

    # special case to prevent running out of memory
    if isinstance(model, GaussianProcessClassifier):
        n_jobs = 1

    with estimator_pipeline(_internal_pipeline, model) as pipeline_with_model:
        fit_kwargs = _get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
        logger.info(f"Cross validating with n_jobs={n_jobs}")

        model_fit_start = time.time()
        scores = cross_validate(
            pipeline_with_model,
            data_X,
            data_y,
            cv=cv,
            groups=groups,
            scoring=metrics_dict,
            fit_params=fit_kwargs,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score=0,
        )
        model_fit_end = time.time()
        model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

        score_dict = dict(
            zip([f"test_{x}" for x in all_metrics.index], all_metrics["Display Name"])
        )
        score_dict = {v: scores[k] for k, v in score_dict.items()}

        logger.info("Calculating mean and std")

        avgs_dict = {k: [np.mean(v), np.std(v)] for k, v in score_dict.items()}

        display.move_progress()

        logger.info("Creating metrics dataframe")

        model_results = pd.DataFrame(score_dict)
        model_avgs = pd.DataFrame(avgs_dict, index=["Mean", "SD"],)

        model_results = model_results.append(model_avgs)
        model_results = model_results.round(round)

        # yellow the mean
        model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
        model_results = model_results.set_precision(round)

        # refitting the model on complete X_train, y_train
        display.update_monitor(1, "Finalizing Model")
        display.display_monitor()

        if refit:
            model_fit_start = time.time()
            logger.info("Finalizing model")
            with io.capture_output():
                pipeline_with_model.fit(data_X, data_y, **fit_kwargs)
            model_fit_end = time.time()

            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
        else:
            model_fit_time /= _get_cv_n_folds(cv)

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # mlflow logging
        if logging_param and system and refit:

            avgs_dict_log = avgs_dict.copy()
            avgs_dict_log = {k: v[0] for k, v in avgs_dict_log.items()}

            try:
                _mlflow_log_model(
                    model=pipeline_with_model,
                    model_results=model_results,
                    score_dict=avgs_dict_log,
                    source="create_model",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=prep_pipe,
                    log_plots=log_plots_param,
                    display=display,
                )
            except:
                logger.error(
                    f"_mlflow_log_model() for {pipeline_with_model} raised an exception:"
                )
                logger.error(traceback.format_exc())

        display.move_progress()

    logger.info("Uploading results into container")

    # storing results in create_model_container
    create_model_container.append(model_results.data)
    display_container.append(model_results.data)

    # storing results in master_model_container
    logger.info("Uploading model into container now")
    master_model_container.append(model)

    display.display(model_results, clear=system, override=False if not system else None)

    logger.info(f"create_model_container: {len(create_model_container)}")
    logger.info(f"master_model_container: {len(master_model_container)}")
    logger.info(f"display_container: {len(display_container)}")

    logger.info(str(model))
    logger.info(
        "create_model() succesfully completed......................................"
    )
    gc.collect()

    if not system:
        return (model, model_fit_time)

    return model


def tune_model(
    estimator,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    n_iter: int = 10,
    custom_grid: Optional[Union[Dict[str, list], Any]] = None,
    optimize: str = "Accuracy",
    custom_scorer=None,  # added in pycaret==2.1 - depreciated
    search_library: str = "scikit-learn",
    search_algorithm: Optional[str] = None,
    early_stopping: Any = "asha",
    early_stopping_max_iters: int = 10,
    choose_better: bool = False,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    display: Optional[Display] = None,
    **kwargs,
) -> Any:

    """
    This function tunes the hyperparameters of a model and scores it using Cross Validation.
    The output prints a score grid that shows Accuracy, AUC, Recall
    Precision, F1, Kappa and MCC by fold (by default = 10 Folds).

    This function returns a trained model object.

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> xgboost = create_model('xgboost')
    >>> tuned_xgboost = tune_model(xgboost) 

    This will tune the hyperparameters of Extreme Gradient Boosting Classifier.


    Parameters
    ----------
    estimator : object, default = None

    fold: integer or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    n_iter: integer, default = 10
        Number of iterations within the Random Grid Search. For every iteration, 
        the model randomly selects one value from the pre-defined grid of 
        hyperparameters.

    custom_grid: dictionary, default = None
        To use custom hyperparameters for tuning pass a dictionary with parameter name
        and values to be iterated. When set to None it uses pre-defined tuning grid.
        Custom grids must be in a format supported by the chosen search library.

    optimize: str, default = 'Accuracy'
        Measure used to select the best model through hyperparameter tuning.
        Can be either a string representing a metric or a custom scorer object
        created using sklearn.make_scorer. 

    custom_scorer: object, default = None
        Will be eventually depreciated.
        custom_scorer can be passed to tune hyperparameters of the model. It must be
        created using sklearn.make_scorer. 

    search_library: str, default = 'scikit-learn'
        The search library used to tune hyperparameters.
        Possible values:

        - 'scikit-learn' - default, requires no further installation
        - 'scikit-optimize' - scikit-optimize. ``pip install scikit-optimize`` https://scikit-optimize.github.io/stable/
        - 'tune-sklearn' - Ray Tune scikit API. Does not support GPU models.
          ``pip install tune-sklearn ray[tune]`` https://github.com/ray-project/tune-sklearn
        - 'optuna' - Optuna. ``pip install optuna`` https://optuna.org/

    search_algorithm: str, default = None
        The search algorithm to be used for finding the best hyperparameters.
        Selection of search algorithms depends on the search_library parameter.
        Some search algorithms require additional libraries to be installed.
        If None, will use search library-specific default algorith.
        'scikit-learn' possible values:

        - 'random' - random grid search (default)
        - 'grid' - grid search

        'scikit-optimize' possible values:

        - 'bayesian' - Bayesian search (default)

        'tune-sklearn' possible values:

        - 'random' - random grid search (default)
        - 'grid' - grid search
        - 'bayesian' - Bayesian search using scikit-optimize
          ``pip install scikit-optimize``
        - 'hyperopt' - Tree-structured Parzen Estimator search using Hyperopt 
          ``pip install hyperopt``
        - 'bohb' - Bayesian search using HpBandSter 
          ``pip install hpbandster ConfigSpace``

        'optuna' possible values:

        - 'random' - randomized search
        - 'tpe' - Tree-structured Parzen Estimator search (default)

    early_stopping: bool or str or object, default = 'asha'
        Use early stopping to stop fitting to a hyperparameter configuration 
        if it performs poorly. Ignored if search_library is ``scikit-learn``, or
        if the estimator doesn't have partial_fit attribute.
        If False or None, early stopping will not be used.
        Can be either an object accepted by the search library or one of the
        following:

        - 'asha' for Asynchronous Successive Halving Algorithm
        - 'hyperband' for Hyperband
        - 'median' for median stopping rule
        - If False or None, early stopping will not be used.

        More info for Optuna - https://optuna.readthedocs.io/en/stable/reference/pruners.html
        More info for Ray Tune (tune-sklearn) - https://docs.ray.io/en/master/tune/api_docs/schedulers.html

    early_stopping_max_iters: int, default = 10
        Maximum number of epochs to run for each sampled configuration.
        Ignored if early_stopping is False or None.

    choose_better: bool, default = False
        When set to set to True, base estimator is returned when the performance doesn't 
        improve by tune_model. This gurantees the returned object would perform atleast 
        equivalent to base estimator created using create_model or model returned by 
        compare_models.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the tuner.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.

    **kwargs: 
        Additional keyword arguments to pass to the optimizer.

    Returns
    -------
    score_grid
        A table containing the scores of the model across the kfolds. 
        Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are also returned.

    model
        Trained and tuned model object. 

    Notes
    -----

    - If a StackingClassifier is passed, the hyperparameters of the meta model (final_estimator)
      will be tuned.
    
    - If a VotingClassifier is passed, the weights will be tuned.

    Warnings
    --------

    - Using 'Grid' search algorithm with default parameter grids may result in very
      long computation.


    """
    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing tune_model()")
    logger.info(f"tune_model({function_params_str})")

    logger.info("Checking exceptions")

    # run_time
    runtime_start = time.time()

    if not fit_kwargs:
        fit_kwargs = {}

    # checking estimator if string
    if type(estimator) is str:
        raise TypeError(
            "The behavior of tune_model in version 1.0.1 is changed. Please pass trained model object."
        )

    # Check for estimator
    if not hasattr(estimator, "fit"):
        raise ValueError(
            f"Estimator {estimator} does not have the required fit() method."
        )

    # checking fold parameter
    if fold is not None and not (type(fold) is int or is_sklearn_cv_generator(fold)):
        raise TypeError(
            "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
        )

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    # checking n_iter parameter
    if type(n_iter) is not int:
        raise TypeError("n_iter parameter only accepts integer value.")

    # checking early_stopping parameter
    possible_early_stopping = ["asha", "Hyperband", "Median"]
    if (
        isinstance(early_stopping, str)
        and early_stopping not in possible_early_stopping
    ):
        raise TypeError(
            f"early_stopping parameter must be one of {', '.join(possible_early_stopping)}"
        )

    # checking early_stopping_max_iters parameter
    if type(early_stopping_max_iters) is not int:
        raise TypeError(
            "early_stopping_max_iters parameter only accepts integer value."
        )

    # checking search_library parameter
    possible_search_libraries = [
        "scikit-learn",
        "scikit-optimize",
        "tune-sklearn",
        "optuna",
    ]
    search_library = search_library.lower()
    if search_library not in possible_search_libraries:
        raise ValueError(
            f"search_library parameter must be one of {', '.join(possible_search_libraries)}"
        )

    if search_library == "scikit-optimize":
        try:
            import skopt
        except ImportError:
            raise ImportError(
                "'scikit-optimize' requires scikit-optimize package to be installed. Do: pip install scikit-optimize"
            )

        if not search_algorithm:
            search_algorithm = "bayesian"

        possible_search_algorithms = ["bayesian"]
        if search_algorithm not in possible_search_algorithms:
            raise ValueError(
                f"For 'scikit-optimize' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
            )

    elif search_library == "tune-sklearn":
        try:
            import tune_sklearn
        except ImportError:
            raise ImportError(
                "'tune-sklearn' requires tune_sklearn package to be installed. Do: pip install tune-sklearn ray[tune]"
            )

        if not search_algorithm:
            search_algorithm = "random"

        possible_search_algorithms = ["random", "grid", "bayesian", "hyperopt", "bohb"]
        if search_algorithm not in possible_search_algorithms:
            raise ValueError(
                f"For 'tune-sklearn' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
            )

        if search_algorithm == "bohb":
            try:
                from ray.tune.suggest.bohb import TuneBOHB
                from ray.tune.schedulers import HyperBandForBOHB
                import ConfigSpace as CS
                import hpbandster
            except ImportError:
                raise ImportError(
                    "It appears that either HpBandSter or ConfigSpace is not installed. Do: pip install hpbandster ConfigSpace"
                )
        elif search_algorithm == "hyperopt":
            try:
                from ray.tune.suggest.hyperopt import HyperOptSearch
                from hyperopt import hp
            except ImportError:
                raise ImportError(
                    "It appears that hyperopt is not installed. Do: pip install hyperopt"
                )
        elif search_algorithm == "bayesian":
            try:
                import skopt
            except ImportError:
                raise ImportError(
                    "It appears that scikit-optimize is not installed. Do: pip install scikit-optimize"
                )

    elif search_library == "optuna":
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "'optuna' requires optuna package to be installed. Do: pip install optuna"
            )

        if not search_algorithm:
            search_algorithm = "tpe"

        possible_search_algorithms = ["random", "tpe"]
        if search_algorithm not in possible_search_algorithms:
            raise ValueError(
                f"For 'optuna' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
            )
    else:
        if not search_algorithm:
            search_algorithm = "random"

        possible_search_algorithms = ["random", "grid"]
        if search_algorithm not in possible_search_algorithms:
            raise ValueError(
                f"For 'scikit-learn' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
            )

    if custom_scorer is not None:
        optimize = custom_scorer
        warnings.warn(
            "custom_scorer parameter will be depreciated, use optimize instead",
            DeprecationWarning,
            stacklevel=2,
        )

    if isinstance(optimize, str):
        # checking optimize parameter
        optimize = _get_metric(optimize)
        if optimize is None:
            raise ValueError(
                "Optimize method not supported. See docstring for list of available parameters."
            )

        # checking optimize parameter for multiclass
        if _is_multiclass():
            if not optimize["Multiclass"]:
                raise TypeError(
                    "Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
                )
    else:
        logger.info(f"optimize set to user defined function {optimize}")

    # checking verbose parameter
    if type(verbose) is not bool:
        raise TypeError("Verbose parameter can only take argument as True or False.")

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    fold = _get_cv_splitter(fold)

    groups = _get_groups(groups)

    if not display:
        progress_args = {"max": 3 + 4}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
        ]
        display = Display(
            verbose=verbose,
            html_param=html_param,
            progress_args=progress_args,
            master_display_columns=master_display_columns,
            monitor_rows=monitor_rows,
        )

        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    # ignore warnings

    warnings.filterwarnings("ignore")

    logger.info("Importing libraries")

    # general dependencies

    import logging

    np.random.seed(seed)

    logger.info("Copying training dataset")
    # Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()

    # reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)

    display.move_progress()

    # setting optimize parameter

    compare_dimension = optimize["Display Name"]
    optimize = optimize["Scorer"]

    # convert trained estimator into string name for grids

    logger.info("Checking base model")

    model = clone(estimator)
    is_stacked_model = False

    base_estimator = model

    if hasattr(base_estimator, "final_estimator"):
        logger.info("Model is stacked, using the definition of the meta-model")
        is_stacked_model = True
        base_estimator = base_estimator.final_estimator

    estimator_id = _get_model_id(base_estimator)

    estimator_definition = _all_models_internal.loc[estimator_id]
    estimator_name = estimator_definition["Name"]
    logger.info(f"Base model : {estimator_name}")

    if search_library == "tune-sklearn" and estimator_definition["GPU Enabled"]:
        raise ValueError("tune-sklearn not supported for GPU enabled models.")

    display.move_progress()

    logger.info("Declaring metric variables")

    """
    MONITOR UPDATE STARTS
    """

    display.update_monitor(1, "Searching Hyperparameters")
    display.display_monitor()

    """
    MONITOR UPDATE ENDS
    """

    logger.info("Defining Hyperparameters")

    from sklearn.ensemble import VotingClassifier

    if custom_grid is not None:
        param_grid = custom_grid
    elif search_library == "scikit-learn" or (
        search_library == "tune-sklearn"
        and (search_algorithm == "grid" or search_algorithm == "random")
    ):
        param_grid = estimator_definition["Tune Grid"]
        if isinstance(base_estimator, VotingClassifier):
            # special case to handle VotingClassifier, as weights need to be
            # generated dynamically
            param_grid = {
                f"weight_{i}": np.arange(0.1, 1, 0.1)
                for i, e in enumerate(base_estimator.estimators)
            }
    else:
        param_grid = estimator_definition["Tune Distributions"]

        if isinstance(base_estimator, VotingClassifier):
            # special case to handle VotingClassifier, as weights need to be
            # generated dynamically
            param_grid = {
                f"weight_{i}": UniformDistribution(0.000000001, 1)
                for i, e in enumerate(base_estimator.estimators)
            }

    if not param_grid:
        raise ValueError(
            "parameter grid for tuning is empty. If passing custom_grid, make sure that it is not empty. If not passing custom_grid, the passed estimator does not have a built-in tuning grid."
        )

    suffixes = []

    if is_stacked_model:
        logger.info("Stacked model passed, will tune meta model hyperparameters")
        suffixes.append("final_estimator")

    gc.collect()

    with estimator_pipeline(_internal_pipeline, model) as pipeline_with_model:
        fit_kwargs = _get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)

        suffixes.append("actual_estimator")

        suffixes = "__".join(reversed(suffixes))

        param_grid = {f"{suffixes}__{k}": v for k, v in param_grid.items()}

        search_kwargs = {**estimator_definition["Tune Args"], **kwargs}

        logger.info(f"param_grid: {param_grid}")

        def _can_early_stop(estimator, consider_warm_start, consider_xgboost, params):
            """
            From https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_basesearch.py.
            
            Helper method to determine if it is possible to do early stopping.
            Only sklearn estimators with ``partial_fit`` or ``warm_start`` can be early
            stopped. warm_start works by picking up training from the previous
            call to ``fit``.
            
            Returns
            -------
                bool
                    if the estimator can early stop
            """

            from sklearn.tree import BaseDecisionTree
            from sklearn.ensemble import BaseEnsemble

            can_partial_fit = supports_partial_fit(estimator, params=params)

            if consider_warm_start:
                is_not_tree_subclass = not issubclass(type(estimator), BaseDecisionTree)
                is_ensemble_subclass = issubclass(type(estimator), BaseEnsemble)
                can_warm_start = hasattr(estimator, "warm_start") and (
                    (
                        hasattr(estimator, "max_iter")
                        and is_not_tree_subclass
                        and not is_ensemble_subclass
                    )
                    or (is_ensemble_subclass and hasattr(estimator, "n_estimators"))
                )
            else:
                can_warm_start = False

            if consider_xgboost:
                from xgboost.sklearn import XGBModel

                is_xgboost = isinstance(estimator, XGBModel)
            else:
                is_xgboost = False

            logger.info(
                f"can_partial_fit: {can_partial_fit}, can_warm_start: {can_warm_start}, is_xgboost: {is_xgboost}"
            )

            return can_partial_fit or can_warm_start or is_xgboost

        n_jobs = gpu_n_jobs_param

        from sklearn.gaussian_process import GaussianProcessClassifier

        # special case to prevent running out of memory
        if isinstance(pipeline_with_model.steps[-1][1], GaussianProcessClassifier):
            n_jobs = 1

        logger.info(f"Tuning with n_jobs={n_jobs}")

        if search_library == "optuna":
            # suppress output
            logging.getLogger("optuna").setLevel(logging.ERROR)

            pruner_translator = {
                "asha": optuna.pruners.SuccessiveHalvingPruner(),
                "hyperband": optuna.pruners.HyperbandPruner(),
                "median": optuna.pruners.MedianPruner(),
                False: optuna.pruners.NopPruner(),
                None: optuna.pruners.NopPruner(),
            }
            pruner = early_stopping
            if pruner in pruner_translator:
                pruner = pruner_translator[early_stopping]

            sampler_translator = {
                "tpe": optuna.samplers.TPESampler(seed=seed),
                "random": optuna.samplers.RandomSampler(seed=seed),
            }
            sampler = sampler_translator[search_algorithm]

            if custom_grid is None:
                param_grid = get_optuna_distributions(param_grid)

            logger.info(f"param_grid: {param_grid}")

            study = optuna.create_study(
                direction="maximize", sampler=sampler, pruner=pruner
            )

            logger.info("Initializing optuna.integration.OptunaSearchCV")
            model_grid = optuna.integration.OptunaSearchCV(
                estimator=pipeline_with_model,
                param_distributions=param_grid,
                cv=fold,
                enable_pruning=early_stopping
                and _can_early_stop(base_estimator, False, False, param_grid),
                max_iter=early_stopping_max_iters,
                n_jobs=n_jobs,
                n_trials=n_iter,
                random_state=seed,
                scoring=optimize,
                study=study,
                refit=False,
                verbose=0,
                **search_kwargs,
            )

        elif search_library == "tune-sklearn":
            early_stopping_translator = {
                "asha": "ASHAScheduler",
                "hyperband": "HyperBandScheduler",
                "median": "MedianStoppingRule",
            }
            if early_stopping in early_stopping_translator:
                early_stopping = early_stopping_translator[early_stopping]

            can_early_stop = early_stopping and _can_early_stop(
                base_estimator, True, True, param_grid
            )

            if not can_early_stop and search_algorithm == "bohb":
                raise ValueError(
                    "'bohb' requires early_stopping = True and the estimator to support partial_fit or have warm_start param set to True."
                )

            # if n_jobs is None:
            # enable Ray local mode - otherwise the performance is terrible
            n_jobs = 1

            TuneSearchCV = get_tune_sklearn_tunesearchcv()
            TuneGridSearchCV = get_tune_sklearn_tunegridsearchcv()

            with true_warm_start(
                pipeline_with_model
            ) if can_early_stop else nullcontext():
                if search_algorithm == "grid":

                    logger.info("Initializing tune_sklearn.TuneGridSearchCV")
                    model_grid = TuneGridSearchCV(
                        estimator=pipeline_with_model,
                        param_grid=param_grid,
                        early_stopping=can_early_stop,
                        scoring=optimize,
                        cv=fold,
                        max_iters=early_stopping_max_iters,
                        n_jobs=n_jobs,
                        use_gpu=gpu_param,
                        refit=True,
                        verbose=0,
                        # pipeline_detection=False,
                        **search_kwargs,
                    )
                elif search_algorithm == "hyperopt":
                    if custom_grid is None:
                        param_grid = get_hyperopt_distributions(param_grid)
                    logger.info(f"param_grid: {param_grid}")
                    logger.info("Initializing tune_sklearn.TuneSearchCV, hyperopt")
                    model_grid = TuneSearchCV(
                        estimator=pipeline_with_model,
                        search_optimization="hyperopt",
                        param_distributions=param_grid,
                        n_trials=n_iter,
                        early_stopping=can_early_stop,
                        scoring=optimize,
                        cv=fold,
                        random_state=seed,
                        max_iters=early_stopping_max_iters,
                        n_jobs=n_jobs,
                        use_gpu=gpu_param,
                        refit=True,
                        verbose=0,
                        # pipeline_detection=False,
                        **search_kwargs,
                    )
                elif search_algorithm == "bayesian":
                    if custom_grid is None:
                        param_grid = get_skopt_distributions(param_grid)
                    logger.info(f"param_grid: {param_grid}")
                    logger.info("Initializing tune_sklearn.TuneSearchCV, bayesian")
                    model_grid = TuneSearchCV(
                        estimator=pipeline_with_model,
                        search_optimization="bayesian",
                        param_distributions=param_grid,
                        n_trials=n_iter,
                        early_stopping=can_early_stop,
                        scoring=optimize,
                        cv=fold,
                        random_state=seed,
                        max_iters=early_stopping_max_iters,
                        n_jobs=n_jobs,
                        use_gpu=gpu_param,
                        refit=True,
                        verbose=0,
                        # pipeline_detection=False,
                        **search_kwargs,
                    )
                elif search_algorithm == "bohb":
                    if custom_grid is None:
                        param_grid = get_CS_distributions(param_grid)
                    logger.info(f"param_grid: {param_grid}")
                    logger.info("Initializing tune_sklearn.TuneSearchCV, bohb")
                    model_grid = TuneSearchCV(
                        estimator=pipeline_with_model,
                        search_optimization="bohb",
                        param_distributions=param_grid,
                        n_trials=n_iter,
                        early_stopping=can_early_stop,
                        scoring=optimize,
                        cv=fold,
                        random_state=seed,
                        max_iters=early_stopping_max_iters,
                        n_jobs=n_jobs,
                        use_gpu=gpu_param,
                        refit=True,
                        verbose=0,
                        # pipeline_detection=False,
                        **search_kwargs,
                    )
                else:
                    logger.info("Initializing tune_sklearn.TuneSearchCV, random")
                    model_grid = TuneSearchCV(
                        estimator=pipeline_with_model,
                        param_distributions=param_grid,
                        early_stopping=can_early_stop,
                        n_trials=n_iter,
                        scoring=optimize,
                        cv=fold,
                        random_state=seed,
                        max_iters=early_stopping_max_iters,
                        n_jobs=n_jobs,
                        use_gpu=gpu_param,
                        refit=True,
                        verbose=0,
                        # pipeline_detection=False,
                        **search_kwargs,
                    )

        elif search_library == "scikit-optimize":
            import skopt

            if custom_grid is None:
                param_grid = get_skopt_distributions(param_grid)
            logger.info(f"param_grid: {param_grid}")

            logger.info("Initializing skopt.BayesSearchCV")
            model_grid = skopt.BayesSearchCV(
                estimator=pipeline_with_model,
                search_spaces=param_grid,
                scoring=optimize,
                n_iter=n_iter,
                cv=fold,
                random_state=seed,
                refit=False,
                n_jobs=n_jobs,
                verbose=0,
                **search_kwargs,
            )
        else:
            if search_algorithm == "grid":
                from sklearn.model_selection import GridSearchCV

                logger.info("Initializing GridSearchCV")
                model_grid = GridSearchCV(
                    estimator=pipeline_with_model,
                    param_grid=param_grid,
                    scoring=optimize,
                    cv=fold,
                    refit=False,
                    n_jobs=n_jobs,
                    **search_kwargs,
                )
            else:
                from sklearn.model_selection import RandomizedSearchCV

                logger.info("Initializing RandomizedSearchCV")
                model_grid = RandomizedSearchCV(
                    estimator=pipeline_with_model,
                    param_distributions=param_grid,
                    scoring=optimize,
                    n_iter=n_iter,
                    cv=fold,
                    random_state=seed,
                    refit=False,
                    n_jobs=n_jobs,
                    **search_kwargs,
                )

        # with io.capture_output():
        model_grid.fit(X_train, y_train, groups=groups, **fit_kwargs)
        best_params = model_grid.best_params_
        logger.info(f"best_params: {best_params}")
        # 18 chars - actual_estimator__
        best_params = {k[18:]: v for k, v in best_params.items()}
        cv_results = None
        try:
            cv_results = model_grid.cv_results_
        except:
            logger.warning("Couldn't get cv_results from model_grid.")

    display.move_progress()

    logger.info("Random search completed")

    logger.info("SubProcess create_model() called ==================================")
    best_model, model_fit_time = _create_model(
        estimator=model,
        system=False,
        display=display,
        fold=fold,
        round=round,
        groups=groups,
        fit_kwargs=fit_kwargs,
        **best_params,
    )
    model_results = pull()
    logger.info("SubProcess create_model() end ==================================")

    if choose_better:
        best_model = _choose_better(
            model,
            [best_model],
            compare_dimension,
            fold,
            new_results_list=[model_results],
            groups=groups,
            fit_kwargs=fit_kwargs,
            display=display,
        )

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # mlflow logging
    if logging_param:

        avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

        try:
            _mlflow_log_model(
                model=best_model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="tune_model",
                runtime=runtime,
                model_fit_time=model_fit_time,
                _prep_pipe=prep_pipe,
                log_plots=log_plots_param,
                tune_cv_results=cv_results,
                display=display,
            )
        except:
            logger.error(f"_mlflow_log_model() for {best_model} raised an exception:")
            logger.error(traceback.format_exc())

    model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
    model_results = model_results.set_precision(round)
    # display.display(model_results, clear=True)

    logger.info(f"create_model_container: {len(create_model_container)}")
    logger.info(f"master_model_container: {len(master_model_container)}")
    logger.info(f"display_container: {len(display_container)}")

    logger.info(str(best_model))
    logger.info(
        "tune_model() succesfully completed......................................"
    )

    gc.collect()
    return best_model


def ensemble_model(
    estimator,
    method: str = "Bagging",
    fold: Optional[Union[int, Any]] = None,
    n_estimators: int = 10,
    round: int = 4,
    choose_better: bool = False,
    optimize: str = "Accuracy",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    display: Optional[Display] = None,  # added in pycaret==2.2.0
) -> Any:
    """
    This function ensembles the trained base estimator using the method defined in 
    'method' param (default = 'Bagging'). The output prints a score grid that shows 
    Accuracy, AUC, Recall, Precision, F1, Kappa and MCC by fold (default = 10 Fold). 

    This function returns a trained model object.  

    Model must be created using create_model() or tune_model().

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> dt = create_model('dt')
    >>> ensembled_dt = ensemble_model(dt)

    This will return an ensembled Decision Tree model using 'Bagging'.
    
    Parameters
    ----------
    estimator : object, default = None

    method: str, default = 'Bagging'
        Bagging method will create an ensemble meta-estimator that fits base 
        classifiers each on random subsets of the original dataset. The other
        available method is 'Boosting' which will create a meta-estimators by
        fitting a classifier on the original dataset and then fits additional 
        copies of the classifier on the same dataset but where the weights of 
        incorrectly classified instances are adjusted such that subsequent 
        classifiers focus more on difficult cases.
    
    fold: integer or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.
    
    n_estimators: integer, default = 10
        The number of base estimators in the ensemble.
        In case of perfect fit, the learning procedure is stopped early.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.

    choose_better: bool, default = False
        When set to set to True, base estimator is returned when the metric doesn't 
        improve by ensemble_model. This gurantees the returned object would perform 
        atleast equivalent to base estimator created using create_model or model 
        returned by compare_models.

    optimize: str, default = 'Accuracy'
        Only used when choose_better is set to True. optimize parameter is used
        to compare emsembled model with base estimator. Values accepted in 
        optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 
        'Kappa', 'MCC'.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.

    Returns
    -------
    score_grid
        A table containing the scores of the model across the kfolds. 
        Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are also returned.

    model
        Trained ensembled model object.

    Warnings
    --------  
    - If target variable is multiclass (more than 2 classes), AUC will be returned 
      as zero (0.0).
        
    
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing ensemble_model()")
    logger.info(f"ensemble_model({function_params_str})")

    logger.info("Checking exceptions")

    # run_time
    runtime_start = time.time()

    if not fit_kwargs:
        fit_kwargs = {}

    # Check for estimator
    if not hasattr(estimator, "fit"):
        raise ValueError(
            f"Estimator {estimator} does not have the required fit() method."
        )

    # Check for allowed method
    available_method = ["Bagging", "Boosting"]
    if method not in available_method:
        raise ValueError(
            "Method parameter only accepts two values 'Bagging' or 'Boosting'."
        )

    # check boosting conflict
    if method == "Boosting":

        boosting_model_definition = _all_models_internal.loc["ada"]

        check_model = estimator

        try:
            check_model = boosting_model_definition["Class"](
                check_model,
                n_estimators=n_estimators,
                **boosting_model_definition["Args"],
            )
            with io.capture_output():
                check_model.fit(X_train, y_train)
        except:
            raise TypeError(
                "Estimator not supported for the Boosting method. Change the estimator or method to 'Bagging'."
            )

    # checking fold parameter
    if fold is not None and not (type(fold) is int or is_sklearn_cv_generator(fold)):
        raise TypeError(
            "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
        )

    # checking n_estimators parameter
    if type(n_estimators) is not int:
        raise TypeError("n_estimators parameter only accepts integer value.")

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    # checking verbose parameter
    if type(verbose) is not bool:
        raise TypeError("Verbose parameter can only take argument as True or False.")

    # checking optimize parameter
    optimize = _get_metric(optimize)
    if optimize is None:
        raise ValueError(
            f"Optimize method not supported. See docstring for list of available parameters."
        )

    # checking optimize parameter for multiclass
    if _is_multiclass():
        if not optimize["Multiclass"]:
            raise TypeError(
                f"Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
            )

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    fold = _get_cv_splitter(fold)

    groups = _get_groups(groups)

    if not display:
        progress_args = {"max": 2 + 4}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
        ]
        display = Display(
            verbose=verbose,
            html_param=html_param,
            progress_args=progress_args,
            master_display_columns=master_display_columns,
            monitor_rows=monitor_rows,
        )

        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    logger.info("Importing libraries")

    np.random.seed(seed)

    logger.info("Copying training dataset")

    # Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()

    # reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)

    display.move_progress()

    # setting optimize parameter

    compare_dimension = optimize["Display Name"]
    optimize = optimize["Scorer"]

    logger.info("Checking base model")

    _estimator_ = estimator

    estimator_id = _get_model_id(estimator)

    estimator_definition = _all_models_internal.loc[estimator_id]
    estimator_name = estimator_definition["Name"]
    logger.info(f"Base model : {estimator_name}")

    """
    MONITOR UPDATE STARTS
    """

    display.update_monitor(1, "Selecting Estimator")
    display.display_monitor()

    """
    MONITOR UPDATE ENDS
    """

    model = _estimator_

    logger.info("Importing untrained ensembler")

    if method == "Bagging":
        logger.info("Ensemble method set to Bagging")
        bagging_model_definition = _all_models_internal.loc["Bagging"]

        model = bagging_model_definition["Class"](
            model,
            bootstrap=True,
            n_estimators=n_estimators,
            **bagging_model_definition["Args"],
        )

    else:
        logger.info("Ensemble method set to Boosting")
        boosting_model_definition = _all_models_internal.loc["ada"]
        model = boosting_model_definition["Class"](
            model, n_estimators=n_estimators, **boosting_model_definition["Args"]
        )

    display.move_progress()

    logger.info("SubProcess create_model() called ==================================")
    model, model_fit_time = _create_model(
        estimator=model,
        system=False,
        display=display,
        fold=fold,
        round=round,
        fit_kwargs=fit_kwargs,
        groups=groups,
    )
    best_model = model
    model_results = pull()
    logger.info("SubProcess create_model() end ==================================")

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # mlflow logging
    if logging_param:

        avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

        try:
            _mlflow_log_model(
                model=best_model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="ensemble_model",
                runtime=runtime,
                model_fit_time=model_fit_time,
                _prep_pipe=prep_pipe,
                log_plots=log_plots_param,
                display=display,
            )
        except:
            logger.error(f"_mlflow_log_model() for {best_model} raised an exception:")
            logger.error(traceback.format_exc())

    if choose_better:
        model = _choose_better(
            _estimator_,
            [best_model],
            compare_dimension,
            fold,
            new_results_list=[model_results],
            groups=groups,
            fit_kwargs=fit_kwargs,
            display=display,
        )

    model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
    model_results = model_results.set_precision(round)
    display.display(model_results, clear=True)

    logger.info(f"create_model_container: {len(create_model_container)}")
    logger.info(f"master_model_container: {len(master_model_container)}")
    logger.info(f"display_container: {len(display_container)}")

    logger.info(str(model))
    logger.info(
        "ensemble_model() succesfully completed......................................"
    )

    gc.collect()
    return model


def blend_models(
    estimator_list: list,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    choose_better: bool = False,
    optimize: str = "Accuracy",
    method: str = "auto",
    weights: Optional[List[float]] = None,  # added in pycaret==2.2.0
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    display: Optional[Display] = None,  # added in pycaret==2.2.0
) -> Any:

    """
    This function creates a Soft Voting / Majority Rule classifier for all the 
    estimators in the model library (excluding the few when turbo is True) or 
    for specific trained estimators passed as a list in estimator_list param.
    It scores it using Cross Validation. The output prints a score
    grid that shows Accuracy, AUC, Recall, Precision, F1, Kappa and MCC by 
    fold (default CV = 10 Folds).

    This function returns a trained model object.

    Example
    -------
    >>> lr = create_model('lr')
    >>> rf = create_model('rf')
    >>> knn = create_model('knn')
    >>> blend_three = blend_models(estimator_list = [lr,rf,knn])

    This will create a VotingClassifier of lr, rf and knn.

    Parameters
    ----------
    estimator_list : list of objects

    fold: integer or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.

    choose_better: bool, default = False
        When set to set to True, base estimator is returned when the metric doesn't 
        improve by ensemble_model. This gurantees the returned object would perform 
        atleast equivalent to base estimator created using create_model or model 
        returned by compare_models.

    optimize: str, default = 'Accuracy'
        Only used when choose_better is set to True. optimize parameter is used
        to compare emsembled model with base estimator. Values accepted in 
        optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 
        'Kappa', 'MCC'.

    method: str, default = 'auto'
        'hard' uses predicted class labels for majority rule voting. 'soft', predicts 
        the class label based on the argmax of the sums of the predicted probabilities, 
        which is recommended for an ensemble of well-calibrated classifiers. Default value,
        'auto', will try to use 'soft' and fall back to 'hard' if the former is not supported.

    weights: list, default = None
        Sequence of weights (float or int) to weight the occurrences of predicted class labels (hard voting)
        or class probabilities before averaging (soft voting). Uses uniform weights if None.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.

    Returns
    -------
    score_grid
        A table containing the scores of the model across the kfolds. 
        Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are also returned.

    model
        Trained Voting Classifier model object. 

    Warnings
    --------
    - When passing estimator_list with method set to 'soft'. All the models in the
      estimator_list must support predict_proba function. 'svm' and 'ridge' doesnt
      support the predict_proba and hence an exception will be raised.
      
    - When estimator_list is set to 'All' and method is forced to 'soft', estimators
      that doesnt support the predict_proba function will be dropped from the estimator
      list.
          
    - If target variable is multiclass (more than 2 classes), AUC will be returned as
      zero (0.0).
        
       
  
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing blend_models()")
    logger.info(f"blend_models({function_params_str})")

    logger.info("Checking exceptions")

    # run_time
    runtime_start = time.time()

    if not fit_kwargs:
        fit_kwargs = {}

    # checking method parameter
    available_method = ["auto", "soft", "hard"]
    if method not in available_method:
        raise ValueError(
            "Method parameter only accepts 'auto', 'soft' or 'hard' as a parameter. See Docstring for details."
        )

    # checking error for estimator_list
    for i in estimator_list:
        if not hasattr(i, "fit"):
            raise ValueError(f"Estimator {i} does not have the required fit() method.")

        # checking method param with estimator list
        if method != "hard":

            for i in estimator_list:
                if not hasattr(i, "predict_proba"):
                    if method != "auto":
                        raise TypeError(
                            f"Estimator list contains estimator {i} that doesn't support probabilities and method is forced to 'soft'. Either change the method or drop the estimator."
                        )
                    else:
                        logger.info(
                            f"Estimator {i} doesn't support probabilities, falling back to 'hard'."
                        )
                        method = "hard"
                        break

            if method == "auto":
                method = "soft"

    # checking fold parameter
    if fold is not None and not (type(fold) is int or is_sklearn_cv_generator(fold)):
        raise TypeError(
            "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
        )

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    if weights is not None:
        num_estimators = len(estimator_list)
        # checking weights parameter
        if len(weights) != num_estimators:
            raise ValueError(
                "weights parameter must have the same length as the estimator_list."
            )
        if not all((isinstance(x, int) or isinstance(x, float)) for x in weights):
            raise TypeError("weights must contain only ints or floats.")

    # checking verbose parameter
    if type(verbose) is not bool:
        raise TypeError("Verbose parameter can only take argument as True or False.")

    # checking optimize parameter
    optimize = _get_metric(optimize)
    if optimize is None:
        raise ValueError(
            f"Optimize method not supported. See docstring for list of available parameters."
        )

    # checking optimize parameter for multiclass
    if _is_multiclass():
        if not optimize["Multiclass"]:
            raise TypeError(
                f"Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
            )

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    fold = _get_cv_splitter(fold)

    groups = _get_groups(groups)

    if not display:
        progress_args = {"max": 2 + 4}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
        ]
        display = Display(
            verbose=verbose,
            html_param=html_param,
            progress_args=progress_args,
            master_display_columns=master_display_columns,
            monitor_rows=monitor_rows,
        )
        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    logger.info("Importing libraries")

    np.random.seed(seed)

    logger.info("Copying training dataset")

    # Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()

    # reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)

    # setting optimize parameter
    compare_dimension = optimize["Display Name"]
    optimize = optimize["Scorer"]

    display.move_progress()

    """
    MONITOR UPDATE STARTS
    """

    display.update_monitor(1, "Compiling Estimators")
    display.display_monitor()

    """
    MONITOR UPDATE ENDS
    """

    logger.info("Getting model names")
    estimator_dict = {}
    for x in estimator_list:
        name = _get_model_id(x)
        suffix = 1
        original_name = name
        while name in estimator_dict:
            name = f"{original_name}_{suffix}"
            suffix += 1
        estimator_dict[name] = x

    estimator_list = list(estimator_dict.items())

    votingclassifier_model_definition = _all_models_internal.loc["Voting"]
    try:
        model = votingclassifier_model_definition["Class"](
            estimators=estimator_list, voting=method, n_jobs=gpu_n_jobs_param
        )
        logger.info("n_jobs multiple passed")
    except:
        logger.info("n_jobs multiple failed")
        model = votingclassifier_model_definition["Class"](
            estimators=estimator_list, voting=method, weights=weights
        )

    display.move_progress()

    logger.info("SubProcess create_model() called ==================================")
    model, model_fit_time = _create_model(
        estimator=model,
        system=False,
        display=display,
        fold=fold,
        round=round,
        fit_kwargs=fit_kwargs,
        groups=groups,
    )
    model_results = pull()
    logger.info("SubProcess create_model() end ==================================")

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # mlflow logging
    if logging_param:

        avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

        try:
            _mlflow_log_model(
                model=model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="blend_models",
                runtime=runtime,
                model_fit_time=model_fit_time,
                _prep_pipe=prep_pipe,
                log_plots=log_plots_param,
                display=display,
            )
        except:
            logger.error(f"_mlflow_log_model() for {model} raised an exception:")
            logger.error(traceback.format_exc())

    if choose_better:
        model = _choose_better(
            model,
            estimator_list,
            compare_dimension,
            fold,
            model_results=model_results,
            groups=groups,
            fit_kwargs=fit_kwargs,
            display=display,
        )

    model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
    model_results = model_results.set_precision(round)
    display.display(model_results, clear=True)

    logger.info(f"create_model_container: {len(create_model_container)}")
    logger.info(f"master_model_container: {len(master_model_container)}")
    logger.info(f"display_container: {len(display_container)}")

    logger.info(str(model))
    logger.info(
        "blend_models() succesfully completed......................................"
    )

    gc.collect()
    return model


def stack_models(
    estimator_list: list,
    meta_model=None,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    method: str = "auto",
    restack: bool = True,
    choose_better: bool = False,
    optimize: str = "Accuracy",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    display: Optional[Display] = None,
) -> Any:

    """
    This function trains a meta model and scores it using Cross Validation.
    The predictions from the base level models as passed in the estimator_list param 
    are used as input features for the meta model. The restacking parameter controls
    the ability to expose raw features to the meta model when set to True
    (default = False).

    The output prints the score grid that shows Accuracy, AUC, Recall, Precision, 
    F1, Kappa and MCC by fold (default = 10 Folds). 
    
    This function returns a trained model object. 

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> dt = create_model('dt')
    >>> rf = create_model('rf')
    >>> ada = create_model('ada')
    >>> ridge = create_model('ridge')
    >>> knn = create_model('knn')
    >>> stacked_models = stack_models(estimator_list=[dt,rf,ada,ridge,knn])

    This will create a meta model that will use the predictions of all the 
    models provided in estimator_list param. By default, the meta model is 
    Logistic Regression but can be changed with meta_model param.

    Parameters
    ----------
    estimator_list : list of objects

    meta_model : object, default = None
        If set to None, Logistic Regression is used as a meta model.

    fold: integer or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.

    method: string, default = 'auto'
        - if ‘auto’, it will try to invoke, for each estimator, 'predict_proba', 
        'decision_function' or 'predict' in that order.
        - otherwise, one of 'predict_proba', 'decision_function' or 'predict'. 
        If the method is not implemented by the estimator, it will raise an error.

    restack: bool, default = True
        When restack is set to True, raw data will be exposed to meta model when
        making predictions, otherwise when False, only the predicted label or
        probabilities is passed to meta model when making final predictions.

    choose_better: bool, default = False
        When set to set to True, base estimator is returned when the metric doesn't 
        improve by ensemble_model. This gurantees the returned object would perform 
        atleast equivalent to base estimator created using create_model or model 
        returned by compare_models.

    optimize: str, default = 'Accuracy'
        Only used when choose_better is set to True. optimize parameter is used
        to compare emsembled model with base estimator. Values accepted in 
        optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 
        'Kappa', 'MCC'.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.

    Returns
    -------
    score_grid
        A table containing the scores of the model across the kfolds. 
        Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are also returned.

    model
        Trained model object.

    Warnings
    --------
    -  If target variable is multiclass (more than 2 classes), AUC will be returned 
       as zero (0.0).

    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing stack_models()")
    logger.info(f"stack_models({function_params_str})")

    logger.info("Checking exceptions")

    # run_time
    runtime_start = time.time()

    if not fit_kwargs:
        fit_kwargs = {}

    # checking error for estimator_list
    for i in estimator_list:
        if not hasattr(i, "fit"):
            raise ValueError(f"Estimator {i} does not have the required fit() method.")

    # checking meta model
    if meta_model is not None:
        if not hasattr(meta_model, "fit"):
            raise ValueError(
                f"Meta Model {meta_model} does not have the required fit() method."
            )

    # checking fold parameter
    if fold is not None and not (type(fold) is int or is_sklearn_cv_generator(fold)):
        raise TypeError(
            "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
        )

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    # checking method parameter
    available_method = ["auto", "predict_proba", "decision_function", "predict"]
    if method not in available_method:
        raise ValueError(
            "Method parameter not acceptable. It only accepts 'auto', 'predict_proba', 'decision_function', 'predict'."
        )

    # checking restack parameter
    if type(restack) is not bool:
        raise TypeError("Restack parameter can only take argument as True or False.")

    # checking verbose parameter
    if type(verbose) is not bool:
        raise TypeError("Verbose parameter can only take argument as True or False.")

    # checking optimize parameter
    optimize = _get_metric(optimize)
    if optimize is None:
        raise ValueError(
            f"Optimize method not supported. See docstring for list of available parameters."
        )

    # checking optimize parameter for multiclass
    if _is_multiclass():
        if not optimize["Multiclass"]:
            raise TypeError(
                f"Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
            )

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    fold = _get_cv_splitter(fold)

    groups = _get_groups(groups)

    logger.info("Defining meta model")
    # Defining meta model.
    if meta_model == None:
        estimator = "lr"
        meta_model_definition = _all_models_internal.loc[estimator]
        meta_model_args = meta_model_definition["Args"]
        meta_model = meta_model_definition["Class"](**meta_model_args)
    else:
        meta_model = clone(meta_model)

    if not display:
        progress_args = {"max": 2 + 4}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
        ]
        display = Display(
            verbose=verbose,
            html_param=html_param,
            progress_args=progress_args,
            master_display_columns=master_display_columns,
            monitor_rows=monitor_rows,
        )
        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    np.random.seed(seed)

    logger.info("Copying training dataset")
    # Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()

    # reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)

    # setting optimize parameter
    compare_dimension = optimize["Display Name"]
    optimize = optimize["Scorer"]

    display.move_progress()

    """
    MONITOR UPDATE STARTS
    """

    display.update_monitor(1, "Compiling Estimators")
    display.display_monitor()

    """
    MONITOR UPDATE ENDS
    """

    logger.info("Getting model names")
    estimator_dict = {}
    for x in estimator_list:
        name = _get_model_id(x)
        suffix = 1
        original_name = name
        while name in estimator_dict:
            name = f"{original_name}_{suffix}"
            suffix += 1
        estimator_dict[name] = x

    estimator_list = list(estimator_dict.items())

    logger.info(estimator_list)
    logger.info("Creating StackingClassifier()")

    stackingclassifier_model_definition = _all_models_internal.loc["Stacking"]
    model = stackingclassifier_model_definition["Class"](
        estimators=estimator_list,
        final_estimator=meta_model,
        cv=fold,
        stack_method=method,
        n_jobs=gpu_n_jobs_param,
        passthrough=restack,
    )

    display.move_progress()

    logger.info("SubProcess create_model() called ==================================")
    model, model_fit_time = _create_model(
        estimator=model,
        system=False,
        display=display,
        fold=fold,
        round=round,
        fit_kwargs=fit_kwargs,
        groups=groups,
    )
    model_results = pull()
    logger.info("SubProcess create_model() end ==================================")

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # mlflow logging
    if logging_param:

        avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

        try:
            _mlflow_log_model(
                model=model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="stack_models",
                runtime=runtime,
                model_fit_time=model_fit_time,
                _prep_pipe=prep_pipe,
                log_plots=log_plots_param,
                display=display,
            )
        except:
            logger.error(f"_mlflow_log_model() for {model} raised an exception:")
            logger.error(traceback.format_exc())

    if choose_better:
        model = _choose_better(
            model,
            estimator_list,
            compare_dimension,
            fold,
            model_results=model_results,
            groups=groups,
            fit_kwargs=fit_kwargs,
            display=display,
        )

    model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
    model_results = model_results.set_precision(round)
    display.display(model_results, clear=True)

    logger.info(f"create_model_container: {len(create_model_container)}")
    logger.info(f"master_model_container: {len(master_model_container)}")
    logger.info(f"display_container: {len(display_container)}")

    logger.info(str(model))
    logger.info(
        "stack_models() succesfully completed......................................"
    )

    gc.collect()
    return model


def plot_model(
    estimator,
    plot: str = "auc",
    scale: float = 1,  # added in pycaret==2.1.0
    save: bool = False,
    fold: Optional[Union[int, Any]] = None,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    system: bool = True,
    display: Optional[Display] = None,  # added in pycaret==2.2.0
):

    """
    This function takes a trained model object and returns a plot based on the
    test / hold-out set. The process may require the model to be re-trained in
    certain cases. See list of plots supported below. 
    
    Model must be created using create_model() or tune_model().

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> plot_model(lr)

    This will return an AUC plot of a trained Logistic Regression model.

    Parameters
    ----------
    estimator : object, default = none
        A trained model object should be passed as an estimator. 

    plot : str, default = auc
        Enter abbreviation of type of plot. The current list of plots supported are (Plot - Name):

        * 'auc' - Area Under the Curve                 
        * 'threshold' - Discrimination Threshold           
        * 'pr' - Precision Recall Curve                  
        * 'confusion_matrix' - Confusion Matrix    
        * 'error' - Class Prediction Error                
        * 'class_report' - Classification Report        
        * 'boundary' - Decision Boundary            
        * 'rfe' - Recursive Feature Selection                 
        * 'learning' - Learning Curve             
        * 'manifold' - Manifold Learning            
        * 'calibration' - Calibration Curve         
        * 'vc' - Validation Curve                  
        * 'dimension' - Dimension Learning           
        * 'feature' - Feature Importance              
        * 'parameter' - Model Hyperparameter
        * 'lift' - Lift Curve
        * 'gain' - Gain Chart

    scale: float, default = 1
        The resolution scale of the figure.

    save: bool, default = False
        When set to True, Plot is saved as a 'png' file in current working directory.

    fold: integer or scikit-learn compatible CV generator, default = None
        Controls cross-validation used in certain plots. If None, will use the CV generator
        defined in setup(). If integer, will use KFold CV with that many folds.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    verbose: bool, default = True
        Progress bar not shown when verbose set to False. 

    system: bool, default = True
        Must remain True all times. Only to be changed by internal functions.

    Returns
    -------
    Visual_Plot
        Prints the visual plot. 

    Warnings
    --------
    -  'svm' and 'ridge' doesn't support the predict_proba method. As such, AUC and 
        calibration plots are not available for these estimators.
       
    -   When the 'max_features' parameter of a trained model object is not equal to 
        the number of samples in training set, the 'rfe' plot is not available.
              
    -   'calibration', 'threshold', 'manifold' and 'rfe' plots are not available for
         multiclass problems.
                

    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing plot_model()")
    logger.info(f"plot_model({function_params_str})")

    logger.info("Checking exceptions")

    if not fit_kwargs:
        fit_kwargs = {}

    # checking plots (string)
    available_plots = [
        ("Hyperparameters", "parameter"),
        ("AUC", "auc"),
        ("Confusion Matrix", "confusion_matrix"),
        ("Threshold", "threshold"),
        ("Precision Recall", "pr"),
        ("Error", "error"),
        ("Class Report", "class_report"),
        ("Feature Selection", "rfe"),
        ("Learning Curve", "learning"),
        ("Manifold Learning", "manifold"),
        ("Calibration Curve", "calibration"),
        ("Validation Curve", "vc"),
        ("Dimensions", "dimension"),
        ("Feature Importance", "feature"),
        ("Decision Boundary", "boundary"),
        ("Lift Chart", "lift"),
        ("Gain Chart", "gain"),
    ]
    available_plots = {k: v for v, k in available_plots}

    if plot not in available_plots:
        raise ValueError(
            "Plot Not Available. Please see docstring for list of available Plots."
        )

    # multiclass plot exceptions:
    multiclass_not_available = ["calibration", "threshold", "manifold", "rfe"]
    if _is_multiclass():
        if plot in multiclass_not_available:
            raise ValueError(
                "Plot Not Available for multiclass problems. Please see docstring for list of available Plots."
            )

    # exception for CatBoost
    if "CatBoostClassifier" in str(type(estimator)):
        raise ValueError(
            "CatBoost estimator is not compatible with plot_model function, try using Catboost with interpret_model instead."
        )

    # checking for auc plot
    if not hasattr(estimator, "predict_proba") and plot == "auc":
        raise TypeError(
            "AUC plot not available for estimators with no predict_proba attribute."
        )

    # checking for auc plot
    if not hasattr(estimator, "predict_proba") and plot == "auc":
        raise TypeError(
            "AUC plot not available for estimators with no predict_proba attribute."
        )

    # checking for calibration plot
    if not hasattr(estimator, "predict_proba") and plot == "calibration":
        raise TypeError(
            "Calibration plot not available for estimators with no predict_proba attribute."
        )

    # checking for rfe
    if (
        hasattr(estimator, "max_features")
        and plot == "rfe"
        and estimator.max_features_ != X_train.shape[1]
    ):
        raise TypeError(
            "RFE plot not available when max_features parameter is not set to None."
        )

    # checking for feature plot
    if (
        not (hasattr(estimator, "coef_") or hasattr(estimator, "feature_importances_"))
        and plot == "feature"
    ):
        raise TypeError(
            "Feature Importance plot not available for estimators that doesnt support coef_ or feature_importances_ attribute."
        )

    # checking fold parameter
    if fold is not None and not (type(fold) is int or is_sklearn_cv_generator(fold)):
        raise TypeError(
            "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
        )

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    fold = _get_cv_splitter(fold)

    groups = _get_groups(groups)

    if not display:
        progress_args = {"max": 5}
        display = Display(
            verbose=verbose, html_param=html_param, progress_args=progress_args
        )
        display.display_progress()

    logger.info("Preloading libraries")
    # pre-load libraries
    import matplotlib.pyplot as plt

    np.random.seed(seed)

    display.move_progress()

    # defining estimator as model locally
    model = estimator

    display.move_progress()

    # plots used for logging (controlled through plots_log_param)
    # AUC, #Confusion Matrix and #Feature Importance

    logger.info("Copying training dataset")

    # Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()

    # reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)

    logger.info("Copying test dataset")

    # Storing X_train and y_train in data_X and data_y parameter
    test_X = X_test.copy()
    test_y = y_test.copy()

    # reset index
    test_X.reset_index(drop=True, inplace=True)
    test_y.reset_index(drop=True, inplace=True)

    logger.info(f"Plot type: {plot}")
    plot_name = available_plots[plot]
    display.move_progress()

    model_name = _get_model_name(model)
    with estimator_pipeline(_internal_pipeline, model) as pipeline_with_model:
        fit_kwargs = _get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)

        _base_dpi = 100

        cv = _get_cv_splitter(fold)

        class MatplotlibDefaultDPI(object):
            def __init__(self, base_dpi: float = 100, scale_to_set: float = 1):
                try:
                    self.default_skplt_dpit = skplt.metrics.plt.rcParams["figure.dpi"]
                    skplt.metrics.plt.rcParams["figure.dpi"] = base_dpi * scale_to_set
                except:
                    pass

            def __enter__(self) -> None:
                return None

            def __exit__(self, type, value, traceback):
                try:
                    skplt.metrics.plt.rcParams["figure.dpi"] = self.default_skplt_dpit
                except:
                    pass

        if plot == "auc":

            from yellowbrick.classifier import ROCAUC

            visualizer = ROCAUC(pipeline_with_model)
            show_yellowbrick_plot(
                visualizer=visualizer,
                X_train=data_X,
                y_train=data_y,
                X_test=test_X,
                y_test=test_y,
                name=plot_name,
                scale=scale,
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
            )

        elif plot == "threshold":

            from yellowbrick.classifier import DiscriminationThreshold

            visualizer = DiscriminationThreshold(pipeline_with_model, random_state=seed)
            show_yellowbrick_plot(
                visualizer=visualizer,
                X_train=data_X,
                y_train=data_y,
                X_test=test_X,
                y_test=test_y,
                name=plot_name,
                scale=scale,
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
            )

        elif plot == "pr":

            from yellowbrick.classifier import PrecisionRecallCurve

            visualizer = PrecisionRecallCurve(pipeline_with_model, random_state=seed)
            show_yellowbrick_plot(
                visualizer=visualizer,
                X_train=data_X,
                y_train=data_y,
                X_test=test_X,
                y_test=test_y,
                name=plot_name,
                scale=scale,
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
            )

        elif plot == "confusion_matrix":

            from yellowbrick.classifier import ConfusionMatrix

            visualizer = ConfusionMatrix(
                pipeline_with_model, random_state=seed, fontsize=15, cmap="Greens"
            )
            show_yellowbrick_plot(
                visualizer=visualizer,
                X_train=data_X,
                y_train=data_y,
                X_test=test_X,
                y_test=test_y,
                name=plot_name,
                scale=scale,
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
            )

        elif plot == "error":

            from yellowbrick.classifier import ClassPredictionError

            visualizer = ClassPredictionError(pipeline_with_model, random_state=seed)
            show_yellowbrick_plot(
                visualizer=visualizer,
                X_train=data_X,
                y_train=data_y,
                X_test=test_X,
                y_test=test_y,
                name=plot_name,
                scale=scale,
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
            )

        elif plot == "class_report":

            from yellowbrick.classifier import ClassificationReport

            visualizer = ClassificationReport(
                pipeline_with_model, random_state=seed, support=True
            )
            show_yellowbrick_plot(
                visualizer=visualizer,
                X_train=data_X,
                y_train=data_y,
                X_test=test_X,
                y_test=test_y,
                name=plot_name,
                scale=scale,
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
            )

        elif plot == "boundary":

            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from yellowbrick.contrib.classifier import DecisionViz

            data_X_transformed = data_X.select_dtypes(include="float64")
            test_X_transformed = test_X.select_dtypes(include="float64")
            logger.info("Fitting StandardScaler()")
            data_X_transformed = StandardScaler().fit_transform(data_X_transformed)
            test_X_transformed = StandardScaler().fit_transform(test_X_transformed)
            pca = PCA(n_components=2, random_state=seed)
            logger.info("Fitting PCA()")
            data_X_transformed = pca.fit_transform(data_X_transformed)
            test_X_transformed = pca.fit_transform(test_X_transformed)

            data_y_transformed = np.array(data_y)
            test_y_transformed = np.array(test_y)

            viz_ = DecisionViz(pipeline_with_model)
            show_yellowbrick_plot(
                visualizer=viz_,
                X_train=data_X_transformed,
                y_train=data_y_transformed,
                X_test=test_X_transformed,
                y_test=test_y_transformed,
                name=plot_name,
                scale=scale,
                handle_test="draw",
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
                features=["Feature One", "Feature Two"],
                classes=["A", "B"],
            )

        elif plot == "rfe":

            from yellowbrick.model_selection import RFECV

            visualizer = RFECV(pipeline_with_model, cv=cv)
            show_yellowbrick_plot(
                visualizer=visualizer,
                X_train=data_X,
                y_train=data_y,
                X_test=test_X,
                y_test=test_y,
                handle_test="",
                name=plot_name,
                scale=scale,
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
            )

        elif plot == "learning":

            from yellowbrick.model_selection import LearningCurve

            sizes = np.linspace(0.3, 1.0, 10)
            visualizer = LearningCurve(
                pipeline_with_model,
                cv=cv,
                train_sizes=sizes,
                n_jobs=gpu_n_jobs_param,
                random_state=seed,
            )
            show_yellowbrick_plot(
                visualizer=visualizer,
                X_train=data_X,
                y_train=data_y,
                X_test=test_X,
                y_test=test_y,
                handle_test="",
                name=plot_name,
                scale=scale,
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
            )

        elif plot == "lift":

            import scikitplot as skplt

            display.move_progress()
            logger.info("Generating predictions / predict_proba on X_test")
            y_test__ = pipeline_with_model.predict(X_test)
            predict_proba__ = pipeline_with_model.predict_proba(X_test)
            display.move_progress()
            display.move_progress()
            display.clear_output()
            with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                fig = skplt.metrics.plot_lift_curve(
                    y_test__, predict_proba__, figsize=(10, 6)
                )
                if save:
                    logger.info(f"Saving '{plot_name}.png' in current active directory")
                    plt.savefig(f"{plot_name}.png")
                    if not system:
                        plt.close()
                else:
                    plt.show()

            logger.info("Visual Rendered Successfully")

        elif plot == "gain":

            import scikitplot as skplt

            display.move_progress()
            logger.info("Generating predictions / predict_proba on X_test")
            y_test__ = pipeline_with_model.predict(X_test)
            predict_proba__ = pipeline_with_model.predict_proba(X_test)
            display.move_progress()
            display.move_progress()
            display.clear_output()
            with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                fig = skplt.metrics.plot_cumulative_gain(
                    y_test__, predict_proba__, figsize=(10, 6)
                )
                if save:
                    logger.info(f"Saving '{plot_name}.png' in current active directory")
                    plt.savefig(f"{plot_name}.png")
                    if not system:
                        plt.close()
                else:
                    plt.show()

            logger.info("Visual Rendered Successfully")

        elif plot == "manifold":

            from yellowbrick.features import Manifold

            data_X_transformed = data_X.select_dtypes(include="float64")
            visualizer = Manifold(manifold="tsne", random_state=seed)
            show_yellowbrick_plot(
                visualizer=visualizer,
                X_train=data_X_transformed,
                y_train=data_y,
                X_test=test_X,
                y_test=test_y,
                handle_train="fit_transform",
                handle_test="",
                name=plot_name,
                scale=scale,
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
            )

        elif plot == "calibration":

            from sklearn.calibration import calibration_curve

            plt.figure(figsize=(7, 6), dpi=_base_dpi * scale)
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

            ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            display.move_progress()
            logger.info("Scoring test/hold-out set")
            prob_pos = pipeline_with_model.predict_proba(test_X)[:, 1]
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            fraction_of_positives, mean_predicted_value = calibration_curve(
                test_y, prob_pos, n_bins=10
            )
            display.move_progress()
            ax1.plot(
                mean_predicted_value,
                fraction_of_positives,
                "s-",
                label=f"{model_name}",
            )

            ax1.set_ylabel("Fraction of positives")
            ax1.set_ylim([0, 1])
            ax1.set_xlim([0, 1])
            ax1.legend(loc="lower right")
            ax1.set_title("Calibration plots  (reliability curve)")
            ax1.set_facecolor("white")
            ax1.grid(b=True, color="grey", linewidth=0.5, linestyle="-")
            plt.tight_layout()
            display.move_progress()
            display.clear_output()
            if save:
                logger.info(f"Saving '{plot_name}.png' in current active directory")
                plt.savefig(f"{plot_name}.png")
                if not system:
                    plt.close()
            else:
                plt.show()

            logger.info("Visual Rendered Successfully")

        elif plot == "vc":

            logger.info("Determining param_name")

            # SGD Classifier
            if model_name == "SGDClassifier":
                param_name = "l1_ratio"
                param_range = np.arange(0, 1, 0.01)

            elif model_name == "LinearDiscriminantAnalysis":
                raise ValueError(
                    "Shrinkage Parameter not supported in Validation Curve Plot."
                )

            # tree based models
            elif hasattr(pipeline_with_model, "actual_estimator__max_depth"):
                param_name = "actual_estimator__max_depth"
                param_range = np.arange(1, 11)

            # knn
            elif hasattr(pipeline_with_model, "actual_estimator__n_neighbors"):
                param_name = "actual_estimator__n_neighbors"
                param_range = np.arange(1, 11)

            # MLP / Ridge
            elif hasattr(pipeline_with_model, "actual_estimator__alpha"):
                param_name = "actual_estimator__alpha"
                param_range = np.arange(0, 1, 0.1)

            # Logistic Regression
            elif hasattr(pipeline_with_model, "actual_estimator__C"):
                param_name = "actual_estimator__C"
                param_range = np.arange(1, 11)

            # Bagging / Boosting
            elif hasattr(pipeline_with_model, "actual_estimator__n_estimators"):
                param_name = "actual_estimator__n_estimators"
                param_range = np.arange(1, 100, 10)

            # Bagging / Boosting / gbc / ada /
            elif hasattr(pipeline_with_model, "actual_estimator__n_estimators"):
                param_name = "actual_estimator__n_estimators"
                param_range = np.arange(1, 100, 10)

            # Naive Bayes
            elif hasattr(pipeline_with_model, "actual_estimator__var_smoothing"):
                param_name = "actual_estimator__var_smoothing"
                param_range = np.arange(0.1, 1, 0.01)

            # QDA
            elif hasattr(pipeline_with_model, "actual_estimator__reg_param"):
                param_name = "actual_estimator__reg_param"
                param_range = np.arange(0, 1, 0.1)

            # GPC
            elif hasattr(pipeline_with_model, "actual_estimator__max_iter_predict"):
                param_name = "actual_estimator__max_iter_predict"
                param_range = np.arange(100, 1000, 100)

            else:
                display.clear_output()
                raise TypeError(
                    "Plot not supported for this estimator. Try different estimator."
                )

            logger.info(f"param_name: {param_name}")

            display.move_progress()

            from yellowbrick.model_selection import ValidationCurve

            viz = ValidationCurve(
                pipeline_with_model,
                param_name=param_name,
                param_range=param_range,
                cv=cv,
                random_state=seed,
            )
            show_yellowbrick_plot(
                visualizer=viz,
                X_train=data_X,
                y_train=data_y,
                X_test=test_X,
                y_test=test_y,
                handle_train="fit",
                handle_test="",
                name=plot_name,
                scale=scale,
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
            )

        elif plot == "dimension":

            from yellowbrick.features import RadViz
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            data_X_transformed = data_X.select_dtypes(include="float64")
            logger.info("Fitting StandardScaler()")
            data_X_transformed = StandardScaler().fit_transform(data_X_transformed)
            data_y_transformed = np.array(data_y)

            features = min(round(len(data_X.columns) * 0.3, 0), 5)
            features = int(features)

            pca = PCA(n_components=features, random_state=seed)
            logger.info("Fitting PCA()")
            data_X_transformed = pca.fit_transform(data_X_transformed)
            display.move_progress()
            classes = data_y.unique().tolist()
            visualizer = RadViz(classes=classes, alpha=0.25)

            show_yellowbrick_plot(
                visualizer=visualizer,
                X_train=data_X_transformed,
                y_train=data_y_transformed,
                X_test=test_X,
                y_test=test_y,
                handle_train="fit_transform",
                handle_test="",
                name=plot_name,
                scale=scale,
                save=save,
                fit_kwargs=fit_kwargs,
                groups=groups,
                system=system,
                display=display,
            )

        elif plot == "feature":
            temp_model = pipeline_with_model
            if hasattr(pipeline_with_model, "steps"):
                temp_model = pipeline_with_model.steps[-1][1]
            if hasattr(temp_model, "coef_"):
                variables = abs(temp_model.coef_[0])
            else:
                logger.warning("No coef_ found. Trying feature_importances_")
                variables = abs(temp_model.feature_importances_)
            coef_df = pd.DataFrame({"Variable": data_X.columns, "Value": variables})
            sorted_df = (
                coef_df.sort_values(by="Value", ascending=False)
                .head(10)
                .sort_values(by="Value")
            )
            my_range = range(1, len(sorted_df.index) + 1)
            display.move_progress()
            plt.figure(figsize=(8, 5), dpi=_base_dpi * scale)
            plt.hlines(y=my_range, xmin=0, xmax=sorted_df["Value"], color="skyblue")
            plt.plot(sorted_df["Value"], my_range, "o")
            display.move_progress()
            plt.yticks(my_range, sorted_df["Variable"])
            plt.title("Feature Importance Plot")
            plt.xlabel("Variable Importance")
            plt.ylabel("Features")
            display.move_progress()
            display.clear_output()
            if save:
                logger.info(f"Saving '{plot_name}.png' in current active directory")
                plt.savefig(f"{plot_name}.png")
                if not system:
                    plt.close()
            else:
                plt.show()

            logger.info("Visual Rendered Successfully")

        elif plot == "parameter":

            param_df = pd.DataFrame.from_dict(
                estimator.get_params(estimator), orient="index", columns=["Parameters"]
            )
            display.display(param_df, clear=True)
            logger.info("Visual Rendered Successfully")

        try:
            plt.close()
        except:
            pass

    gc.collect()

    logger.info(
        "plot_model() succesfully completed......................................"
    )


def evaluate_model(
    estimator,
    fold: Optional[Union[int, Any]] = None,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
):

    """
    This function displays a user interface for all of the available plots for 
    a given estimator. It internally uses the plot_model() function. 
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> evaluate_model(lr)
    
    This will display the User Interface for all of the plots for a given
    estimator.

    Parameters
    ----------
    estimator : object, default = none
        A trained model object should be passed as an estimator. 

    fold: integer or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    Returns
    -------
    User_Interface
        Displays the user interface for plotting.

    """

    from ipywidgets import widgets
    from ipywidgets.widgets import interact, fixed

    if not fit_kwargs:
        fit_kwargs = {}

    a = widgets.ToggleButtons(
        options=[
            ("Hyperparameters", "parameter"),
            ("AUC", "auc"),
            ("Confusion Matrix", "confusion_matrix"),
            ("Threshold", "threshold"),
            ("Precision Recall", "pr"),
            ("Error", "error"),
            ("Class Report", "class_report"),
            ("Feature Selection", "rfe"),
            ("Learning Curve", "learning"),
            ("Manifold Learning", "manifold"),
            ("Calibration Curve", "calibration"),
            ("Validation Curve", "vc"),
            ("Dimensions", "dimension"),
            ("Feature Importance", "feature"),
            ("Decision Boundary", "boundary"),
            ("Lift Chart", "lift"),
            ("Gain Chart", "gain"),
        ],
        description="Plot Type:",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        icons=[""],
    )

    fold = _get_cv_splitter(fold)

    groups = _get_groups(groups)

    d = interact(
        plot_model,
        estimator=fixed(estimator),
        plot=a,
        save=fixed(False),
        verbose=fixed(True),
        scale=fixed(1),
        fold=fixed(fold),
        fit_kwargs=fixed(fit_kwargs),
        groups=fixed(groups),
        system=fixed(True),
        display=fixed(None),
    )


def interpret_model(
    estimator,
    plot: str = "summary",
    feature: Optional[str] = None,
    observation: Optional[int] = None,
    **kwargs,  # added in pycaret==2.1
):

    """
    This function takes a trained model object and returns an interpretation plot 
    based on the test / hold-out set. It only supports tree based algorithms. 

    This function is implemented based on the SHAP (SHapley Additive exPlanations),
    which is a unified approach to explain the output of any machine learning model. 
    SHAP connects game theory with local explanations.

    For more information : https://shap.readthedocs.io/en/latest/

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> dt = create_model('dt')
    >>> interpret_model(dt)

    This will return a summary interpretation plot of Decision Tree model.

    Parameters
    ----------
    estimator : object, default = none
        A trained tree based model object should be passed as an estimator. 

    plot : str, default = 'summary'
        Other available options are 'correlation' and 'reason'.

    feature: str, default = None
        This parameter is only needed when plot = 'correlation'. By default feature is 
        set to None which means the first column of the dataset will be used as a 
        variable. A feature parameter must be passed to change this.

    observation: integer, default = None
        This parameter only comes into effect when plot is set to 'reason'. If no 
        observation number is provided, it will return an analysis of all observations 
        with the option to select the feature on x and y axes through drop down 
        interactivity. For analysis at the sample level, an observation parameter must
        be passed with the index value of the observation in test / hold-out set. 

    **kwargs: 
        Additional keyword arguments to pass to the plot.

    Returns
    -------
    Visual_Plot
        Returns the visual plot.
        Returns the interactive JS plot when plot = 'reason'.

    Warnings
    -------- 
    - interpret_model doesn't support multiclass problems.

    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing interpret_model()")
    logger.info(f"interpret_model({function_params_str})")

    logger.info("Checking exceptions")

    # checking if shap available
    try:
        import shap
    except:
        logger.error(
            "shap library not found. pip install shap to use interpret_model function."
        )
        raise ImportError(
            "shap library not found. pip install shap to use interpret_model function."
        )

    # allowed models
    model_id = _get_model_id(estimator)

    shap_models = _all_models_internal[_all_models_internal["SHAP"] != False]
    shap_models_ids = set(shap_models.index)

    if model_id not in shap_models_ids:
        raise TypeError(
            f"This function only supports tree based models for binary classification: {', '.join(shap_models['Name'].to_list())}."
        )

    # plot type
    allowed_types = ["summary", "correlation", "reason"]
    if plot not in allowed_types:
        raise ValueError(
            "type parameter only accepts 'summary', 'correlation' or 'reason'."
        )

    """
    Error Checking Ends here
    
    """

    logger.info("Importing libraries")
    # general dependencies

    np.random.seed(seed)

    # storing estimator in model variable
    model = estimator

    # defining type of classifier
    shap_models_type1 = set(shap_models[shap_models["SHAP"] == "type1"].index)
    shap_models_type2 = set(shap_models[shap_models["SHAP"] == "type2"].index)

    logger.info(f"plot type: {plot}")

    shap_plot = None

    if plot == "summary":

        logger.info("Creating TreeExplainer")
        explainer = shap.TreeExplainer(model)
        logger.info("Compiling shap values")
        shap_values = explainer.shap_values(X_test)
        shap_plot = shap.summary_plot(shap_values, X_test, **kwargs)

    elif plot == "correlation":

        if feature == None:

            logger.warning(
                f"No feature passed. Default value of feature used for correlation plot: {X_test.columns[0]}"
            )
            dependence = X_test.columns[0]

        else:

            logger.warning(
                f"feature value passed. Feature used for correlation plot: {X_test.columns[0]}"
            )
            dependence = feature

        logger.info("Creating TreeExplainer")
        explainer = shap.TreeExplainer(model)
        logger.info("Compiling shap values")
        shap_values = explainer.shap_values(X_test)

        if model_id in shap_models_type1:
            logger.info("model type detected: type 1")
            shap.dependence_plot(dependence, shap_values[1], X_test, **kwargs)
        elif model_id in shap_models_type2:
            logger.info("model type detected: type 2")
            shap.dependence_plot(dependence, shap_values, X_test, **kwargs)

    elif plot == "reason":

        if model_id in shap_models_type1:
            logger.info("model type detected: type 1")

            logger.info("Creating TreeExplainer")
            explainer = shap.TreeExplainer(model)
            logger.info("Compiling shap values")

            if observation is None:
                logger.warning(
                    "Observation set to None. Model agnostic plot will be rendered."
                )
                shap_values = explainer.shap_values(X_test)
                shap.initjs()
                shap_plot = shap.force_plot(
                    explainer.expected_value[1], shap_values[1], X_test, **kwargs
                )

            else:
                row_to_show = observation
                data_for_prediction = X_test.iloc[row_to_show]

                if model_id == "lightgbm":
                    logger.info("model type detected: LGBMClassifier")
                    shap_values = explainer.shap_values(X_test)
                    shap.initjs()
                    shap_plot = shap.force_plot(
                        explainer.expected_value[1],
                        shap_values[0][row_to_show],
                        data_for_prediction,
                        **kwargs,
                    )

                else:
                    logger.info("model type detected: Unknown")

                    shap_values = explainer.shap_values(data_for_prediction)
                    shap.initjs()
                    shap_plot = shap.force_plot(
                        explainer.expected_value[1],
                        shap_values[1],
                        data_for_prediction,
                        **kwargs,
                    )

        elif model_id in shap_models_type2:
            logger.info("model type detected: type 2")

            logger.info("Creating TreeExplainer")
            explainer = shap.TreeExplainer(model)
            logger.info("Compiling shap values")
            shap_values = explainer.shap_values(X_test)
            shap.initjs()

            if observation is None:
                logger.warning(
                    "Observation set to None. Model agnostic plot will be rendered."
                )

                shap_plot = shap.force_plot(
                    explainer.expected_value, shap_values, X_test, **kwargs
                )

            else:

                row_to_show = observation
                data_for_prediction = X_test.iloc[row_to_show]

                shap_plot = shap.force_plot(
                    explainer.expected_value,
                    shap_values[row_to_show, :],
                    X_test.iloc[row_to_show, :],
                    **kwargs,
                )

    logger.info("Visual Rendered Successfully")

    logger.info(
        "interpret_model() succesfully completed......................................"
    )

    gc.collect()
    return shap_plot


def calibrate_model(
    estimator,
    method: str = "sigmoid",
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    display: Optional[Display] = None,  # added in pycaret==2.2.0
) -> Any:

    """
    This function takes the input of trained estimator and performs probability 
    calibration with sigmoid or isotonic regression. The output prints a score 
    grid that shows Accuracy, AUC, Recall, Precision, F1, Kappa and MCC by fold 
    (default = 10 Fold). The ouput of the original estimator and the calibrated 
    estimator (created using this function) might not differ much. In order 
    to see the calibration differences, use 'calibration' plot in plot_model to 
    see the difference before and after.

    This function returns a trained model object. 

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> dt_boosted = create_model('dt', ensemble = True, method = 'Boosting')
    >>> calibrated_dt = calibrate_model(dt_boosted)

    This will return Calibrated Boosted Decision Tree Model.

    Parameters
    ----------
    estimator : object
    
    method : str, default = 'sigmoid'
        The method to use for calibration. Can be 'sigmoid' which corresponds to Platt's 
        method or 'isotonic' which is a non-parametric approach. It is not advised to use
        isotonic calibration with too few calibration samples

    fold: integer or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.

    Returns
    -------
    score_grid
        A table containing the scores of the model across the kfolds. 
        Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are also returned.

    model
        trained and calibrated model object.

    Warnings
    --------
    - Avoid isotonic calibration with too few calibration samples (<1000) since it 
      tends to overfit.
      
    - calibration plot not available for multiclass problems.
      
  
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing calibrate_model()")
    logger.info(f"calibrate_model({function_params_str})")

    logger.info("Checking exceptions")

    # run_time
    runtime_start = time.time()

    if not fit_kwargs:
        fit_kwargs = {}

    # checking fold parameter
    if fold is not None and not (type(fold) is int or is_sklearn_cv_generator(fold)):
        raise TypeError(
            "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
        )

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    # checking verbose parameter
    if type(verbose) is not bool:
        raise TypeError("Verbose parameter can only take argument as True or False.")

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    fold = _get_cv_splitter(fold)

    groups = _get_groups(groups)

    logger.info("Preloading libraries")

    # pre-load libraries

    logger.info("Preparing display monitor")

    if not display:
        progress_args = {"max": 2 + 4}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
        ]
        display = Display(
            verbose=verbose,
            html_param=html_param,
            progress_args=progress_args,
            master_display_columns=master_display_columns,
            monitor_rows=monitor_rows,
        )

        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    np.random.seed(seed)

    logger.info("Getting model name")

    full_name = _get_model_name(estimator)

    logger.info(f"Base model : {full_name}")

    """
    MONITOR UPDATE STARTS
    """

    display.update_monitor(1, "Selecting Estimator")
    display.display_monitor()

    """
    MONITOR UPDATE ENDS
    """

    # calibrating estimator

    logger.info("Importing untrained CalibratedClassifierCV")

    calibrated_model_definition = _all_models_internal.loc["CalibratedCV"]
    model = calibrated_model_definition["Class"](
        base_estimator=estimator,
        method=method,
        cv=fold,
        **calibrated_model_definition["Args"],
    )

    display.move_progress()

    logger.info("SubProcess create_model() called ==================================")
    model, model_fit_time = _create_model(
        estimator=model,
        system=False,
        display=display,
        fold=fold,
        round=round,
        fit_kwargs=fit_kwargs,
        groups=groups,
    )
    model_results = pull()
    logger.info("SubProcess create_model() end ==================================")

    model_results = model_results.round(round)

    display.move_progress()

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # mlflow logging
    if logging_param:

        avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

        try:
            _mlflow_log_model(
                model=model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="calibrate_models",
                runtime=runtime,
                model_fit_time=model_fit_time,
                _prep_pipe=prep_pipe,
                log_plots=log_plots_param,
                display=display,
            )
        except:
            logger.error(f"_mlflow_log_model() for {model} raised an exception:")
            logger.error(traceback.format_exc())

    model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
    model_results = model_results.set_precision(round)
    display.display(model_results, clear=True)

    logger.info(f"create_model_container: {len(create_model_container)}")
    logger.info(f"master_model_container: {len(master_model_container)}")
    logger.info(f"display_container: {len(display_container)}")

    logger.info(str(model))
    logger.info(
        "calibrate_model() succesfully completed......................................"
    )

    gc.collect()
    return model


def optimize_threshold(
    estimator,
    true_positive: int = 0,
    true_negative: int = 0,
    false_positive: int = 0,
    false_negative: int = 0,
):

    """
    This function optimizes probability threshold for a trained model using custom cost
    function that can be defined using combination of True Positives, True Negatives,
    False Positives (also known as Type I error), and False Negatives (Type II error).
    
    This function returns a plot of optimized cost as a function of probability 
    threshold between 0 to 100. 

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> optimize_threshold(lr, true_negative = 10, false_negative = -100)

    This will return a plot of optimized cost as a function of probability threshold.

    Parameters
    ----------
    estimator : object
        A trained model object should be passed as an estimator. 
    
    true_positive : int, default = 0
        Cost function or returns when prediction is true positive.  
    
    true_negative : int, default = 0
        Cost function or returns when prediction is true negative.
    
    false_positive : int, default = 0
        Cost function or returns when prediction is false positive.    
    
    false_negative : int, default = 0
        Cost function or returns when prediction is false negative.       
    
    
    Returns
    -------
    Visual_Plot
        Prints the visual plot. 

    Warnings
    --------
    - This function is not supported for multiclass problems.
      
       
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing optimize_threshold()")
    logger.info(f"optimize_threshold({function_params_str})")

    logger.info("Importing libraries")

    # import libraries

    import plotly.express as px
    from IPython.display import clear_output

    np.random.seed(seed)

    # cufflinks
    import cufflinks as cf

    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    """
    ERROR HANDLING STARTS HERE
    """

    logger.info("Checking exceptions")

    # exception 1 for multi-class
    if _is_multiclass():
        raise TypeError(
            "optimize_threshold() cannot be used when target is multi-class."
        )

    # check predict_proba value
    if type(estimator) is not list:
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(
                "Estimator doesn't support predict_proba function and cannot be used in optimize_threshold()."
            )

    # check cost function type
    allowed_types = [int, float]

    if type(true_positive) not in allowed_types:
        raise TypeError("true_positive parameter only accepts float or integer value.")

    if type(true_negative) not in allowed_types:
        raise TypeError("true_negative parameter only accepts float or integer value.")

    if type(false_positive) not in allowed_types:
        raise TypeError("false_positive parameter only accepts float or integer value.")

    if type(false_negative) not in allowed_types:
        raise TypeError("false_negative parameter only accepts float or integer value.")

    """
    ERROR HANDLING ENDS HERE
    """

    # define model as estimator
    model = estimator

    model_name = _get_model_name(model)

    # generate predictions and store actual on y_test in numpy array
    actual = np.array(y_test)

    predicted = model.predict_proba(X_test)
    predicted = predicted[:, 1]

    """
    internal function to calculate loss starts here
    """

    logger.info("Defining loss function")

    def calculate_loss(
        actual,
        predicted,
        tp_cost=true_positive,
        tn_cost=true_negative,
        fp_cost=false_positive,
        fn_cost=false_negative,
    ):

        # true positives
        tp = predicted + actual
        tp = np.where(tp == 2, 1, 0)
        tp = tp.sum()

        # true negative
        tn = predicted + actual
        tn = np.where(tn == 0, 1, 0)
        tn = tn.sum()

        # false positive
        fp = (predicted > actual).astype(int)
        fp = np.where(fp == 1, 1, 0)
        fp = fp.sum()

        # false negative
        fn = (predicted < actual).astype(int)
        fn = np.where(fn == 1, 1, 0)
        fn = fn.sum()

        total_cost = (tp_cost * tp) + (tn_cost * tn) + (fp_cost * fp) + (fn_cost * fn)

        return total_cost

    """
    internal function to calculate loss ends here
    """

    grid = np.arange(0, 1, 0.01)

    # loop starts here

    cost = []
    # global optimize_results

    logger.info("Iteration starts at 0")

    for i in grid:

        pred_prob = (predicted >= i).astype(int)
        cost.append(calculate_loss(actual, pred_prob))

    optimize_results = pd.DataFrame(
        {"Probability Threshold": grid, "Cost Function": cost}
    )
    fig = px.line(
        optimize_results,
        x="Probability Threshold",
        y="Cost Function",
        line_shape="linear",
    )
    fig.update_layout(plot_bgcolor="rgb(245,245,245)")
    title = f"{model_name} Probability Threshold Optimization"

    # calculate vertical line
    y0 = optimize_results["Cost Function"].min()
    y1 = optimize_results["Cost Function"].max()
    x0 = optimize_results.sort_values(by="Cost Function", ascending=False).iloc[0][0]
    x1 = x0

    t = x0.round(2)

    fig.add_shape(
        dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="red", width=2))
    )
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.45,
            "xanchor": "center",
            "yanchor": "top",
        }
    )
    logger.info("Figure ready for render")
    fig.show()
    print(f"Optimized Probability Threshold: {t} | Optimized Cost Function: {y1}")
    logger.info(
        "optimize_threshold() succesfully completed......................................"
    )


def predict_model(
    estimator,
    data: Optional[pd.DataFrame] = None,
    probability_threshold: Optional[float] = None,
    encoded_labels: bool = False,  # added in pycaret==2.1.0
    round: int = 4,  # added in pycaret==2.2.0
    verbose: bool = True,
    display: Optional[Display] = None,  # added in pycaret==2.2.0
) -> pd.DataFrame:

    """
    This function is used to predict label and probability score on the new dataset
    using a trained estimator. New unseen data can be passed to data param as pandas 
    Dataframe. If data is not passed, the test / hold-out set separated at the time of 
    setup() is used to generate predictions. 
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> lr_predictions_holdout = predict_model(lr)
        
    Parameters
    ----------
    estimator : object, default = none
        A trained model object / pipeline should be passed as an estimator. 
     
    data : pandas.DataFrame
        Shape (n_samples, n_features) where n_samples is the number of samples 
        and n_features is the number of features. All features used during training 
        must be present in the new dataset.
    
    probability_threshold : float, default = None
        Threshold used to convert probability values into binary outcome. By default 
        the probability threshold for all binary classifiers is 0.5 (50%). This can be 
        changed using probability_threshold param.

    encoded_labels: Boolean, default = False
        If True, will return labels encoded as an integer.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    verbose: bool, default = True
        Holdout score grid is not printed when verbose is set to False.

    Returns
    -------
    Predictions
        Predictions (Label and Score) column attached to the original dataset
        and returned as pandas dataframe.

    score_grid
        A table containing the scoring metrics on hold-out / test set.

    Warnings
    --------
    - The behavior of the predict_model is changed in version 2.1 without backward compatibility.
    As such, the pipelines trained using the version (<= 2.0), may not work for inference 
    with version >= 2.1. You can either retrain your models with a newer version or downgrade
    the version for inference.
    
    
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing predict_model()")
    logger.info(f"predict_model({function_params_str})")

    logger.info("Checking exceptions")

    """
    exception checking starts here
    """

    if probability_threshold is not None:
        # probability_threshold allowed types
        allowed_types = [int, float]
        if type(probability_threshold) not in allowed_types:
            raise TypeError(
                "probability_threshold parameter only accepts value between 0 to 1."
            )

        if probability_threshold > 1:
            raise TypeError(
                "probability_threshold parameter only accepts value between 0 to 1."
            )

        if probability_threshold < 0:
            raise TypeError(
                "probability_threshold parameter only accepts value between 0 to 1."
            )

    """
    exception checking ends here
    """

    logger.info("Preloading libraries")

    # general dependencies
    from sklearn import metrics

    np.random.seed(seed)

    if not display:
        display = Display(verbose=verbose, html_param=html_param,)

    dtypes = None

    # dataset
    if data is None:

        if is_sklearn_pipeline(estimator):
            estimator = estimator.steps[-1][1]

        X_test_ = X_test.copy()
        y_test_ = y_test.copy()

        dtypes = prep_pipe.named_steps["dtypes"]

        X_test_.reset_index(drop=True, inplace=True)
        y_test_.reset_index(drop=True, inplace=True)

    else:

        if hasattr(estimator, "steps") and hasattr(estimator, "predict"):
            dtypes = estimator.named_steps["dtypes"]
        else:
            try:
                dtypes = prep_pipe.named_steps["dtypes"]
                estimator_ = deepcopy(prep_pipe)
                estimator_.steps.append(["trained model", estimator])
                estimator = estimator_
                del estimator_

            except:
                raise ValueError("Pipeline not found")

        X_test_ = data.copy()

    # function to replace encoded labels with their original values
    # will not run if categorical_labels is false
    def replace_lables_in_column(label_column):
        if dtypes and hasattr(dtypes, "replacement"):
            replacement_mapper = {int(v): k for k, v in dtypes.replacement.items()}
            label_column.replace(replacement_mapper, inplace=True)

    # model name
    full_name = _get_model_name(estimator)

    # prediction starts here

    pred_ = estimator.predict(X_test_)

    try:
        pred_prob = estimator.predict_proba(X_test_)

        if len(pred_prob[0]) > 2:
            p_counter = 0
            d = []
            for i in range(0, len(pred_prob)):
                d.append(pred_prob[i][pred_[p_counter]])
                p_counter += 1

            pred_prob = d

        else:
            pred_prob = pred_prob[:, 1]

    except:
        pred_prob = None

    if probability_threshold is not None and pred_prob is not None:
        try:
            pred_ = (pred_prob >= probability_threshold).astype(int)
        except:
            pass

    if pred_prob is None:
        pred_prob = pred_

    df_score = None

    if data is None:
        metrics = _calculate_metrics(y_test_, pred_, pred_prob)
        df_score = pd.DataFrame(metrics, index=[0])
        df_score.insert(0, "Model", full_name)
        df_score = df_score.round(round)
        display.display(df_score.style.set_precision(round), clear=False)

    label = pd.DataFrame(pred_)
    label.columns = ["Label"]
    label["Label"] = label["Label"].astype(int)
    if not encoded_labels:
        replace_lables_in_column(label["Label"])

    if data is None:
        if not encoded_labels:
            replace_lables_in_column(y_test_)
        X_test_ = pd.concat([X_test_, y_test_, label], axis=1)
    else:
        X_test_.insert(len(X_test_.columns), "Label", label["Label"].to_list())

    if pred_prob is not None:
        try:
            score = pd.DataFrame(pred_prob)
            score.columns = ["Score"]
            score = score.round(round)
            X_test_ = pd.concat([X_test_, score], axis=1)
        except:
            pass

    # store predictions on hold-out in display_container
    if df_score is not None:
        display_container.append(df_score)

    gc.collect()
    return X_test_


def finalize_model(
    estimator,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    display=None,
) -> Any:  # added in pycaret==2.2.0

    """
    This function fits the estimator onto the complete dataset passed during the
    setup() stage. The purpose of this function is to prepare for final model
    deployment after experimentation. 
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> final_lr = finalize_model(lr)
    
    This will return the final model object fitted to complete dataset. 

    Parameters
    ----------
    estimator : object, default = none
        A trained model object should be passed as an estimator. 

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    Returns
    -------
    model
        Trained model object fitted on complete dataset.

    Warnings
    --------
    - If the model returned by finalize_model(), is used on predict_model() without 
      passing a new unseen dataset, then the information grid printed is misleading 
      as the model is trained on the complete dataset including test / hold-out sample. 
      Once finalize_model() is used, the model is considered ready for deployment and
      should be used on new unseens dataset only.
       
         
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing finalize_model()")
    logger.info(f"finalize_model({function_params_str})")

    # run_time
    runtime_start = time.time()

    if not fit_kwargs:
        fit_kwargs = {}

    groups = _get_groups(groups)

    if not display:
        display = Display(verbose=False, html_param=html_param,)

    np.random.seed(seed)

    logger.info(f"Finalizing {estimator}")
    display.clear_output()
    model_final, model_fit_time = _create_model(
        estimator=estimator,
        verbose=False,
        system=False,
        X_train_data=X,
        y_train_data=y,
        fit_kwargs=fit_kwargs,
        groups=groups,
    )
    model_results = pull(pop=True)

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # mlflow logging
    if logging_param:

        avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

        try:
            _mlflow_log_model(
                model=model_final,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="finalize_model",
                runtime=runtime,
                model_fit_time=model_fit_time,
                _prep_pipe=prep_pipe,
                log_plots=log_plots_param,
                display=display,
            )
        except:
            logger.error(f"_mlflow_log_model() for {model_final} raised an exception:")
            logger.error(traceback.format_exc())

    model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
    model_results = model_results.set_precision(round)
    display.display(model_results, clear=True)

    logger.info(f"create_model_container: {len(create_model_container)}")
    logger.info(f"master_model_container: {len(master_model_container)}")
    logger.info(f"display_container: {len(display_container)}")

    logger.info(str(model_final))
    logger.info(
        "finalize_model() succesfully completed......................................"
    )

    gc.collect()
    return model_final


def deploy_model(
    model,
    model_name: str,
    authentication: dict,
    platform: str = "aws",  # added gcp and azure support in pycaret==2.1
):

    """
    (In Preview)

    This function deploys the transformation pipeline and trained model object for
    production use. The platform of deployment can be defined under the platform
    param along with the applicable authentication tokens which are passed as a
    dictionary to the authentication param.
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> deploy_model(model = lr, model_name = 'deploy_lr', platform = 'aws', authentication = {'bucket' : 'pycaret-test'})
    
    This will deploy the model on an AWS S3 account under bucket 'pycaret-test'
    
    Notes
    -----
    For AWS users:
    Before deploying a model to an AWS S3 ('aws'), environment variables must be 
    configured using the command line interface. To configure AWS env. variables, 
    type aws configure in your python command line. The following information is
    required which can be generated using the Identity and Access Management (IAM) 
    portal of your amazon console account:

    - AWS Access Key ID
    - AWS Secret Key Access
    - Default Region Name (can be seen under Global settings on your AWS console)
    - Default output format (must be left blank)

    For GCP users:
    --------------
    Before deploying a model to Google Cloud Platform (GCP), project must be created 
    either using command line or GCP console. Once project is created, you must create 
    a service account and download the service account key as a JSON file, which is 
    then used to set environment variable. 

    https://cloud.google.com/docs/authentication/production

    - Google Cloud Project
    - Service Account Authetication

    For Azure users:
    ---------------
    Before deploying a model to Microsoft's Azure (Azure), environment variables
    for connection string must be set. In order to get connection string, user has
    to create account of Azure. Once it is done, create a Storage account. In the settings
    section of storage account, user can get the connection string.

    Read below link for more details.
    https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json

    - Azure Storage Account

    Parameters
    ----------
    model : object
        A trained model object should be passed as an estimator. 
    
    model_name : str
        Name of model to be passed as a str.
    
    authentication : dict
        Dictionary of applicable authentication tokens.

        When platform = 'aws':
        {'bucket' : 'Name of Bucket on S3'}

        When platform = 'gcp':
        {'project': 'gcp_pycaret', 'bucket' : 'pycaret-test'}

        When platform = 'azure':
        {'container': 'pycaret-test'}
    
    platform: str, default = 'aws'
        Name of platform for deployment. Current available options are: 'aws', 'gcp' and 'azure'

    Returns
    -------
    Success_Message
    
    Warnings
    --------
    - This function uses file storage services to deploy the model on cloud platform. 
      As such, this is efficient for batch-use. Where the production objective is to 
      obtain prediction at an instance level, this may not be the efficient choice as 
      it transmits the binary pickle file between your local python environment and
      the platform. 
    
    """
    import pycaret.internal.persistence

    return pycaret.internal.persistence.deploy_model(
        model, model_name, authentication, platform, prep_pipe
    )


def create_webservice(model, model_endopoint, api_key=True, pydantic_payload=None):
    """
    (In Preview)

    This function deploys the transformation pipeline and trained model object as api. Rest api base on FastAPI and could run on localhost, it uses
   the model name as a path to POST endpoint. The endpoint can be protected by api key generated by pycaret and return for the user.
    Create_webservice uses pydantic style input/output model.
    Parameters
    ----------
    model : object
        A trained model object should be passed as an estimator.

    model_endopoint : string
        Name of model to be passed as a string.

    api_key: bool, default = True
        Security for API, if True Pycaret generates api key and print in console,
        else user can post data without header but it not safe if application will
        expose external.

    pydantic_payload: pydantic.main.ModelMetaclass, default = None
        Pycaret allows us to automatically generate a schema for the input model,
        thanks to which can prevent incorrect requests. User can generate own pydantic model and use it as an input model.

    Returns
    -------
    Dictionary with api_key: FastAPI application class which is ready to run with console
    if api_key is set to False, the dictionary key is set to 'Not_exist'.

    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing create_service()")
    logger.info(f"create_service({function_params_str})")

    try:
        from fastapi import FastAPI
    except ImportError:
        logger.error(
            "fastapi library not found. pip install fastapi to use create_service function."
        )
        raise ImportError(
            "fastapi library not found. pip install fastapi to use create_service function."
        )
    # try initialize predict before add it to endpoint (cold start)
    try:
        _ = predict_model(estimator=model, verbose=False)
    except:
        raise ValueError(
            "Cannot predict on cold start check probability_threshold or model"
        )

    # check pydantic style
    try:
        from pydantic import create_model, BaseModel, Json
        from pydantic.main import ModelMetaclass
    except ImportError:
        logger.error(
            "pydantic library not found. pip install fastapi to use create_service function."
        )
        ImportError(
            "pydantic library not found. pip install fastapi to use create_service function."
        )
    if pydantic_payload is not None:
        assert isinstance(
            pydantic_payload, ModelMetaclass
        ), "pydantic_payload must be ModelMetaClass type"
    else:
        # automatically create pydantic payload model
        import json
        from typing import Optional

        print(
            "You are using an automatic data validation model it could fail in some cases"
        )
        print(
            "To be sure the model works properly create pydantic model in your own (pydantic_payload)"
        )
        fields = {
            name: (Optional[type(t)], ...)
            for name, t in json.loads(
                data_before_preprocess.drop(columns=[target_param])
                .convert_dtypes()
                .sample(1)
                .to_json(orient="records")
            )[0].items()
        }
        print(fields)
        pydantic_payload = create_model("DefaultModel", __base__=BaseModel, **fields)
        logger.info(
            "Generated json schema: {}".format(pydantic_payload.schema_json(indent=2))
        )

    # generate apikey
    import secrets
    from typing import Optional
    from fastapi.security.api_key import APIKeyHeader
    from fastapi import HTTPException, Security

    api_key_handler = APIKeyHeader(name="token", auto_error=False)
    if api_key:
        # generate key and log into console
        key = secrets.token_urlsafe(30)

        def validate_request(header: Optional[str] = Security(api_key_handler)):
            if header is None:
                raise HTTPException(status_code=400, detail="No api key", headers={})
            if not secrets.compare_digest(header, str(key)):
                raise HTTPException(
                    status_code=401, detail="Unauthorized request", headers={}
                )
            return True

    else:
        key = "Not_exist"
        print("API will be working without security")

        def validate_request(header: Optional[str] = Security(api_key_handler)):
            return True

    # validate request functionality
    validation = validate_request

    # creating response model
    from typing import Any, Optional
    from pycaret.utils import __version__

    class PredictionResult(BaseModel):
        prediction: Any
        author: str = "pycaret"
        lib_version: str = __version__()
        input_data: pydantic_payload
        processed_input_data: Json = None
        time_utc: Optional[str] = None

        class Config:
            schema_extra = {
                "example": {
                    "prediction": 1,
                    "autohor": "pycaret",
                    "lib_version": "2.0.0",
                    "input_data": pydantic_payload,
                    "processed_input_data": {"col1": 1, "col2": "string"},
                    "time": "2020-09-10 20:00",
                }
            }

    app = FastAPI(
        title="REST API for ML prediction created by Pycaret",
        description="This is the REST API for the ML model generated"
        "by the Pycaret library: https://pycaret.org. "
        "All endpoints should run asynchronously, please validate"
        "the Pydantic model and read api documentation. "
        "In case of trouble, please add issuesto github: https://github.com/pycaret/pycaret/issues",
        version="pycaret: {}".format(__version__()),
        externalDocs={"Pycaret": "https://pycaret.org/"},
    )

    # import additionals from fastAPI
    import pandas as pd
    import time
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi import Depends
    from fastapi.encoders import jsonable_encoder

    # enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/predict/{}".format(model_endopoint))
    def post_predict(
        block_data: pydantic_payload, authenticated: bool = Depends(validation)
    ):
        # encode input data
        try:
            encoded_data_df = pd.DataFrame(jsonable_encoder(block_data), index=[0])
        except Exception as e:
            raise HTTPException(status_code=404, detail="Wrong json format")
        # predict values
        unseen_predictions = predict_model(model, data=encoded_data_df)
        # change format to dictionary to be sure that python types used
        unseen_predictions = unseen_predictions.to_dict(orient="records")[0]
        label = unseen_predictions["Label"]
        del unseen_predictions["Label"]

        # creating return object
        predict_schema = PredictionResult(
            prediction=label,
            input_data=block_data,
            processed_input_data=json.dumps(unseen_predictions),
            time_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time())),
        )
        return predict_schema

    return {key: app}


def save_model(model, model_name: str, model_only: bool = False, verbose: bool = True):

    """
    This function saves the transformation pipeline and trained model object 
    into the current active directory as a pickle file for later use. 
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> save_model(lr, 'lr_model_23122019')
    
    This will save the transformation pipeline and model as a binary pickle
    file in the current active directory. 

    Parameters
    ----------
    model : object, default = none
        A trained model object should be passed as an estimator. 
    
    model_name : str, default = none
        Name of pickle file to be passed as a string.
    
    model_only : bool, default = False
        When set to True, only trained model object is saved and all the 
        transformations are ignored.

    verbose: bool, default = True
        Success message is not printed when verbose is set to False.

    Returns
    -------
    Success_Message
    
         
    """

    import pycaret.internal.persistence

    return pycaret.internal.persistence.save_model(
        model, model_name, prep_pipe if model_only else None, verbose
    )


def load_model(
    model_name,
    platform: Optional[str] = None,
    authentication: Optional[Dict[str, str]] = None,
    verbose: bool = True,
):

    """
    This function loads a previously saved transformation pipeline and model 
    from the current active directory into the current python environment. 
    Load object must be a pickle file.
    
    Example
    -------
    >>> saved_lr = load_model('lr_model_23122019')
    
    This will load the previously saved model in saved_lr variable. The file 
    must be in the current directory.

    Parameters
    ----------
    model_name : str, default = none
        Name of pickle file to be passed as a string.
      
    platform: str, default = None
        Name of platform, if loading model from cloud. Current available options are:
        'aws', 'gcp' and 'azure'.
    
    authentication : dict
        dictionary of applicable authentication tokens.

        When platform = 'aws':
        {'bucket' : 'Name of Bucket on S3'}

        When platform = 'gcp':
        {'project': 'gcp_pycaret', 'bucket' : 'pycaret-test'}

        When platform = 'azure':
        {'container': 'pycaret-test'}
    
    verbose: bool, default = True
        Success message is not printed when verbose is set to False.

    Returns
    -------
    Model Object

    """

    import pycaret.internal.persistence

    return pycaret.internal.persistence.load_model(
        model_name, platform, authentication, verbose
    )


def automl(optimize: str = "Accuracy", use_holdout: bool = False) -> Any:

    """
    This function returns the best model out of all models created in 
    current active environment based on metric defined in optimize parameter. 

    Parameters
    ----------
    optimize : str, default = 'Accuracy'
        Other values you can pass in optimize param are 'AUC', 'Recall', 'Precision',
        'F1', 'Kappa', and 'MCC'.

    use_holdout: bool, default = False
        When set to True, metrics are evaluated on holdout set instead of CV.

    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing automl()")
    logger.info(f"automl({function_params_str})")

    # checking optimize parameter
    optimize = _get_metric(optimize)
    if optimize is None:
        raise ValueError(
            f"Optimize method not supported. See docstring for list of available parameters."
        )

    # checking optimize parameter for multiclass
    if _is_multiclass():
        if not optimize["Multiclass"]:
            raise TypeError(
                f"Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
            )

    compare_dimension = optimize["Display Name"]
    optimize = optimize["Scorer"]

    scorer = []

    if use_holdout:
        logger.info("Model Selection Basis : Holdout set")
        for i in master_model_container:
            pred_holdout = predict_model(i, verbose=False)
            p = pull(pop=True)
            p = p[compare_dimension][0]
            scorer.append(p)

    else:
        logger.info("Model Selection Basis : CV Results on Training set")
        for i in create_model_container:
            r = i[compare_dimension][-2:][0]
            scorer.append(r)

    # returning better model
    index_scorer = scorer.index(max(scorer))

    automl_result = master_model_container[index_scorer]

    logger.info("SubProcess finalize_model() called ==================================")
    automl_finalized = finalize_model(automl_result)
    logger.info("SubProcess finalize_model() end ==================================")

    logger.info(str(automl_finalized))
    logger.info("automl() succesfully completed......................................")

    return automl_finalized


def pull(pop=False) -> pd.DataFrame:  # added in pycaret==2.2.0
    """
    Returns latest displayed table.

    Parameters
    ----------
    pop : bool, default = False
        If true, will pop (remove) the returned dataframe from the
        display container.

    Returns
    -------
    pandas.DataFrame
        Equivalent to get_config('display_container')[-1]

    """
    return display_container.pop(-1) if pop else display_container[-1]


def models(
    type: Optional[str] = None,
    internal: bool = False,
    force_regenerate: bool = False,
    raise_errors: bool = True,
) -> pd.DataFrame:

    """
    Returns table of models available in model library.

    Example
    -------
    >>> all_models = models()

    This will return pandas dataframe with all available 
    models and their metadata.

    Parameters
    ----------
    type : str, default = None
        - linear : filters and only return linear models
        - tree : filters and only return tree based models
        - ensemble : filters and only return ensemble models
    
    internal: bool, default = False
        If True, will return extra columns and rows used internally.

    force_regenerate: bool, default = False
        If True, will force the DataFrame to be regenerated,
        instead of using a cached version.

    raise_errors: bool, default = True
        If False, will suppress all exceptions, ignoring models
        that couldn't be created.

    Returns
    -------
    pandas.DataFrame

    """

    def filter_model_df_by_type(df):
        model_type = {
            "linear": ["lr", "ridge", "svm"],
            "tree": ["dt"],
            "ensemble": ["rf", "et", "gbc", "xgboost", "lightgbm", "catboost", "ada"],
        }

        # Check if type is valid
        if type not in list(model_type) + [None]:
            raise ValueError(
                f"type param only accepts {', '.join(list(model_type) + str(None))}."
            )
        return df[df.index.isin(model_type.get(type, df.index))]

    if not force_regenerate:
        try:
            if internal:
                return filter_model_df_by_type(_all_models_internal)
            else:
                return filter_model_df_by_type(all_models)
        except:
            pass

    logger.info(f"gpu_param set to {gpu_param}")

    model_containers = get_all_model_containers(globals(), raise_errors)
    rows = [
        v.get_dict(internal)
        for k, v in model_containers.items()
        if (internal or not v.is_special)
    ]

    df = pd.DataFrame(rows)
    df.set_index("ID", inplace=True, drop=True)

    return filter_model_df_by_type(df)


def get_metrics(
    force_regenerate: bool = False,
    reset: bool = False,
    include_custom: bool = True,
    raise_errors: bool = True,
) -> pd.DataFrame:
    """
    Returns table of metrics available.

    Example
    -------
    >>> metrics = get_metrics()

    This will return pandas dataframe with all available 
    metrics and their metadata.

    Parameters
    ----------
    force_regenerate: bool, default = False
        If True, will return a regenerated DataFrame,
        instead of using a cached version.
    reset: bool, default = False
        If True, will reset all changes made using add_metric() and get_metric().
    include_custom: bool, default = True
        Whether to include user added (custom) metrics or not.
    raise_errors: bool, default = True
        If False, will suppress all exceptions, ignoring models
        that couldn't be created.

    Returns
    -------
    pandas.DataFrame

    """

    if reset and not "all_metrics" in globals():
        raise ValueError("setup() needs to be ran first.")

    global all_metrics

    if not force_regenerate and not reset:
        try:
            if not include_custom:
                return all_metrics[all_metrics["Custom"] == False]
            return all_metrics
        except:
            pass

    np.random.seed(seed)

    metric_containers = get_all_metric_containers(globals(), raise_errors)
    rows = [v.get_dict() for k, v in metric_containers.items()]

    df = pd.DataFrame(rows)
    df.set_index("ID", inplace=True, drop=True)

    if not include_custom:
        df = df[df["Custom"] == False]

    if reset:
        all_metrics = df
    return df


def _get_metric(name_or_id: str):
    """
    Gets a metric from get_metrics() by name or index.
    """
    metrics = get_metrics()
    metric = None
    try:
        metric = metrics.loc[name_or_id]
        return metric
    except:
        pass

    try:
        metric = metrics[metrics["Name"] == name_or_id].iloc[0]
        return metric
    except:
        pass

    return metric


def add_metric(
    id: str,
    name: str,
    score_func: type,
    target: str = "pred",
    args: dict = None,
    multiclass: bool = True,
) -> pd.Series:
    """
    Adds a custom metric to be used in all functions.

    Parameters
    ----------
    id: str
        Unique id for the metric.

    name: str
        Display name of the metric.

    score_func: type
        Score function (or loss function) with signature score_func(y, y_pred, **kwargs).

    target: str, default = 'pred'
        The target of the score function.
        - 'pred' for the prediction table
        - 'pred_proba' for pred_proba
        - 'threshold' for decision_function or predict_proba

    args: dict, default = {}
        Arguments to be passed to score function.

    multiclass: bool, default = True
        Whether the metric supports multiclass problems.

    Returns
    -------
    pandas.Series
        The created row as Series.

    """

    if not args:
        args = {}

    if not "all_metrics" in globals():
        raise ValueError("setup() needs to be ran first.")

    global all_metrics

    if id in all_metrics.index:
        raise ValueError("id already present in metrics dataframe.")

    new_metric = ClassificationMetricContainer(
        id=id,
        name=name,
        score_func=score_func,
        target=target,
        args=args,
        display_name=name,
        is_multiclass=bool(multiclass),
        is_custom=True,
    )

    new_metric = new_metric.get_dict()

    new_metric = pd.Series(new_metric, name=id.replace(" ", "_")).drop("ID")

    all_metrics = all_metrics.append(new_metric)
    return all_metrics.iloc[-1]


def remove_metric(name_or_id: str):
    """
    Removes a metric used in all functions.

    Parameters
    ----------
    name_or_id: str
        Display name or ID of the metric.

    """
    if not "all_metrics" in globals():
        raise ValueError("setup() needs to be ran first.")

    try:
        all_metrics.drop(name_or_id, axis=0, inplace=True)
        return
    except:
        pass

    try:
        all_metrics.drop(
            all_metrics[all_metrics["Name"] == name_or_id].index, axis=0, inplace=True
        )
        return
    except:
        pass

    raise ValueError(
        f"No row with 'Display Name' or 'ID' (index) {name_or_id} present in the metrics dataframe."
    )


def get_logs(experiment_name: Optional[str] = None, save: bool = False) -> pd.DataFrame:

    """
    Returns a table with experiment logs consisting
    run details, parameter, metrics and tags. 

    Example
    -------
    >>> logs = get_logs()

    This will return pandas dataframe.

    Parameters
    ----------
    experiment_name : str, default = None
        When set to None current active run is used.

    save : bool, default = False
        When set to True, csv file is saved in current directory.

    Returns
    -------
    pandas.DataFrame

    """

    if experiment_name is None:
        exp_name_log_ = exp_name_log
    else:
        exp_name_log_ = experiment_name

    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    if client.get_experiment_by_name(exp_name_log_) is None:
        raise ValueError(
            "No active run found. Check logging parameter in setup or to get logs for inactive run pass experiment_name."
        )

    exp_id = client.get_experiment_by_name(exp_name_log_).experiment_id
    runs = mlflow.search_runs(exp_id)

    if save:
        file_name = f"{exp_name_log_}_logs.csv"
        runs.to_csv(file_name, index=False)

    return runs


def get_config(variable: str):

    """
    This function is used to access global environment variables.
    Following variables can be accessed:

    - X: Transformed dataset (X)
    - y: Transformed dataset (y)  
    - X_train: Transformed train dataset (X)
    - X_test: Transformed test/holdout dataset (X)
    - y_train: Transformed train dataset (y)
    - y_test: Transformed test/holdout dataset (y)
    - seed: random state set through session_id
    - prep_pipe: Transformation pipeline configured through setup
    - fold_shuffle_param: shuffle parameter used in Kfolds
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - create_model_container: results grid storage container
    - master_model_container: model storage container
    - display_container: results display container
    - exp_name_log: Name of experiment set through setup
    - logging_param: log_experiment param set through setup
    - log_plots_param: log_plots param set through setup
    - USI: Unique session ID parameter set through setup
    - fix_imbalance_param: fix_imbalance param set through setup
    - fix_imbalance_method_param: fix_imbalance_method param set through setup
    - data_before_preprocess: data before preprocessing
    - target_param: name of target variable
    - gpu_param: use_gpu param configured through setup

    Example
    -------
    >>> X_train = get_config('X_train') 

    This will return X_train transformed dataset.

    Returns
    -------
    variable

    """

    import pycaret.internal.utils

    return pycaret.internal.utils.get_config(variable, globals())


def set_config(variable: str, value):

    """
    This function is used to reset global environment variables.
    Following variables can be accessed:

    - X: Transformed dataset (X)
    - y: Transformed dataset (y)  
    - X_train: Transformed train dataset (X)
    - X_test: Transformed test/holdout dataset (X)
    - y_train: Transformed train dataset (y)
    - y_test: Transformed test/holdout dataset (y)
    - seed: random state set through session_id
    - prep_pipe: Transformation pipeline configured through setup
    - fold_shuffle_param: shuffle parameter used in Kfolds
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - create_model_container: results grid storage container
    - master_model_container: model storage container
    - display_container: results display container
    - exp_name_log: Name of experiment set through setup
    - logging_param: log_experiment param set through setup
    - log_plots_param: log_plots param set through setup
    - USI: Unique session ID parameter set through setup
    - fix_imbalance_param: fix_imbalance param set through setup
    - fix_imbalance_method_param: fix_imbalance_method param set through setup
    - data_before_preprocess: data before preprocessing

    Example
    -------
    >>> set_config('seed', 123) 

    This will set the global seed to '123'.

    """

    import pycaret.internal.utils

    return pycaret.internal.utils.set_config(variable, value, globals())


def save_config(file_name: str):

    """
    This function is used to save all enviroment variables to file,
    allowing to later resume modeling without rerunning setup().

    Example
    -------
    >>> save_config('myvars.pkl') 

    This will save all enviroment variables to 'myvars.pkl'.

    """

    import pycaret.internal.utils

    return pycaret.internal.utils.save_config(file_name, globals())


def load_config(file_name: str):

    """
    This function is used to load enviroment variables from file created with save_config(),
    allowing to later resume modeling without rerunning setup().


    Example
    -------
    >>> load_config('myvars.pkl') 

    This will load all enviroment variables from 'myvars.pkl'.

    """

    import pycaret.internal.utils

    return pycaret.internal.utils.load_config(file_name, globals())


def _choose_better(
    model,
    new_estimator_list: list,
    compare_dimension: str,
    fold: int,
    model_results=None,
    new_results_list: Optional[list] = None,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    display: Optional[Display] = None,
):
    """
    When choose_better is set to True, optimize metric in scoregrid is
    compared with base model created using create_model so that the
    functions return the model with better score only. This will ensure 
    model performance is at least equivalent to what is seen in compare_models 
    """

    if new_results_list and len(new_results_list) != len(new_estimator_list):
        raise ValueError(
            "new_results_list and new_estimator_list must have the same length"
        )

    logger = get_logger()
    logger.info("choose_better activated")
    display.update_monitor(1, "Compiling Final Results")
    display.display_monitor()

    scorer = []

    if not fit_kwargs:
        fit_kwargs = {}

    if model_results is None:
        logger.info(
            "SubProcess create_model() called =================================="
        )
        _create_model(
            model,
            verbose=False,
            system=False,
            fold=fold,
            fit_kwargs=fit_kwargs,
            groups=groups,
        )
        logger.info("SubProcess create_model() end ==================================")
        model_results = pull(pop=True)

    model_results = model_results.loc["Mean"][compare_dimension]
    logger.info(f"Base model {model} result for {compare_dimension} is {model_results}")

    scorer.append(model_results)

    base_models_ = []
    for i, new_estimator in enumerate(new_estimator_list):
        if isinstance(new_estimator, tuple):
            new_estimator = new_estimator[1]
        if new_results_list:
            m = new_estimator
            s = new_results_list[i].loc["Mean"][compare_dimension]
        else:
            logger.info(
                "SubProcess create_model() called =================================="
            )
            m, _ = _create_model(
                new_estimator,
                verbose=False,
                system=False,
                fold=fold,
                fit_kwargs=fit_kwargs,
                groups=groups,
            )
            logger.info(
                "SubProcess create_model() end =================================="
            )
            s = pull(pop=True).loc["Mean"][compare_dimension]
        logger.info(f"{new_estimator} result for {compare_dimension} is {s}")
        scorer.append(s)
        base_models_.append(m)

    index_scorer = scorer.index(max(scorer))

    if index_scorer != 0:
        model = base_models_[index_scorer - 1]
    logger.info(f"{model} is best model")

    logger.info("choose_better completed")
    return model


def _sample_data(
    model, seed: int, train_size: float, data_split_shuffle: bool, display: Display
):
    """
    Method to sample data.
    """

    from sklearn.model_selection import train_test_split
    import plotly.express as px

    np.random.seed(seed)

    logger = get_logger()

    logger.info("Sampling dataset")

    split_perc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    split_perc_text = [
        "10%",
        "20%",
        "30%",
        "40%",
        "50%",
        "60%",
        "70%",
        "80%",
        "90%",
        "100%",
    ]
    split_perc_tt = split_perc.copy()
    split_perc_tt_total = []

    score_dict = {metric: np.empty((0, 0)) for metric in all_metrics["Display Name"]}

    counter = 0

    for i in split_perc:

        display.move_progress()

        t0 = time.time()

        """
        MONITOR UPDATE STARTS
        """

        perc_text = split_perc_text[counter]
        display.update_monitor(1, f"Fitting Model on {perc_text} sample")
        display.display_monitor()

        """
        MONITOR UPDATE ENDS
        """

        _stratify_columns = _get_columns_to_stratify_by(
            X, y, stratify_param, target_param
        )

        X_, _, y_, _ = train_test_split(
            X,
            y,
            test_size=1 - i,
            stratify=_stratify_columns,
            random_state=seed,
            shuffle=data_split_shuffle,
        )

        _stratify_columns = _get_columns_to_stratify_by(
            X_, y_, stratify_param, target_param
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_,
            y_,
            test_size=1 - train_size,
            stratify=_stratify_columns,
            random_state=seed,
            shuffle=data_split_shuffle,
        )

        with io.capture_output():
            model.fit(X_train, y_train)
        pred_ = model.predict(X_test)
        try:
            pred_prob = model.predict_proba(X_test)[:, 1]
        except:
            logger.warning("model has no predict_proba attribute.")
            pred_prob = 0

        score_dict = _calculate_metrics(y_test, pred_, pred_prob)

        t1 = time.time()

        """
        Time calculation begins
        """

        tt = t1 - t0
        total_tt = tt / i
        split_perc_tt.pop(0)

        for remain in split_perc_tt:
            ss = total_tt * remain
            split_perc_tt_total.append(ss)

        """
        Time calculation Ends
        """

        split_perc_tt_total = []
        counter += 1

    model_results = []
    for i in split_perc:
        for metric_name, metric in score_dict.items():
            row = (i, metric[i], metric_name)
            model_results.append(row)

    model_results = pd.DataFrame(
        model_results, columns=["Sample", "Metric", "Metric Name"]
    )
    fig = px.line(
        model_results,
        x="Sample",
        y="Metric",
        color="Metric Name",
        line_shape="linear",
        range_y=[0, 1],
    )
    fig.update_layout(plot_bgcolor="rgb(245,245,245)")
    title = f"{_get_model_name(model)} Metrics and Sample %"
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.45,
            "xanchor": "center",
            "yanchor": "top",
        }
    )
    fig.show()

    display.update_monitor(1, "Waiting for input")
    display.display_monitor()

    print(
        "Please Enter the sample % of data you would like to use for modeling. Example: Enter 0.3 for 30%."
    )
    print("Press Enter if you would like to use 100% of the data.")

    sample_size = input("Sample Size: ")

    _stratify_columns = _get_columns_to_stratify_by(X, y, stratify_param, target_param)

    if sample_size == "" or sample_size == "1":

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=1 - train_size,
            stratify=_stratify_columns,
            random_state=seed,
            shuffle=data_split_shuffle,
        )

    else:

        sample_n = float(sample_size)
        X_selected, _, y_selected, _ = train_test_split(
            X,
            y,
            test_size=1 - sample_n,
            stratify=_stratify_columns,
            random_state=seed,
            shuffle=data_split_shuffle,
        )

        _stratify_columns = _get_columns_to_stratify_by(
            X_selected, y_selected, stratify_param, target_param
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected,
            y_selected,
            test_size=1 - train_size,
            stratify=_stratify_columns,
            random_state=seed,
            shuffle=data_split_shuffle,
        )

    return X_train, X_test, y_train, y_test


def _is_multiclass() -> bool:
    """
    Method to check if the problem is multiclass.
    """
    try:
        return y.value_counts().count() > 2
    except:
        return False


def _get_model_id(e) -> str:
    """
    Get model id.
    """
    import pycaret.internal.utils

    return pycaret.internal.utils.get_model_id(e, models(internal=True))


def _get_model_name(e, deep: bool = True) -> str:
    """
    Get model name.
    """
    import pycaret.internal.utils

    return pycaret.internal.utils.get_model_name(e, models(internal=True), deep=deep)


def _is_special_model(e) -> bool:
    """
    Is the model special (eg. VotingClassifier).
    """
    import pycaret.internal.utils

    return pycaret.internal.utils.is_special_model(e, models(internal=True))


def _calculate_metrics(
    ytest, pred_, pred_prob: float, weights: Optional[list] = None,
) -> dict:
    """
    Calculate all metrics in get_metrics().
    """
    from pycaret.internal.utils import calculate_metrics

    return calculate_metrics(
        metrics=get_metrics(),
        ytest=ytest,
        pred_=pred_,
        pred_proba=pred_prob,
        weights=weights,
    )


def _mlflow_log_model(
    model,
    model_results,
    score_dict: dict,
    source: str,
    runtime: float,
    model_fit_time: float,
    _prep_pipe,
    log_plots: bool = False,
    tune_cv_results=None,
    URI=None,
    display: Optional[Display] = None,
):
    logger = get_logger()

    logger.info("Creating MLFlow logs")

    # Creating Logs message monitor
    if display:
        display.update_monitor(1, "Creating Logs")
        display.display_monitor()

    # import mlflow
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment(exp_name_log)

    full_name = _get_model_name(model)
    logger.info(f"Model: {full_name}")

    with mlflow.start_run(run_name=full_name) as run:

        # Get active run to log as tag
        RunID = mlflow.active_run().info.run_id

        # Log model parameters
        if hasattr(model, "named_steps") and "actual_estimator" in model.named_steps:
            params = model.named_steps["actual_estimator"]
        else:
            params = model

        if hasattr(params, "get_all_params"):
            params = params.get_all_params()
        else:
            params = params.get_params()

        for i in list(params):
            v = params.get(i)
            if len(str(v)) > 250:
                params.pop(i)

        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(score_dict)

        # set tag of compare_models
        mlflow.set_tag("Source", source)

        if not URI:
            import secrets

            URI = secrets.token_hex(nbytes=4)
        mlflow.set_tag("URI", URI)
        mlflow.set_tag("USI", USI)
        mlflow.set_tag("Run Time", runtime)
        mlflow.set_tag("Run ID", RunID)

        # Log training time in seconds
        mlflow.log_metric("TT", model_fit_time)

        # Log the CV results as model_results.html artifact
        try:
            model_results.data.to_html("Results.html", col_space=65, justify="left")
        except:
            model_results.to_html("Results.html", col_space=65, justify="left")
        mlflow.log_artifact("Results.html")
        os.remove("Results.html")

        # Generate hold-out predictions and save as html
        holdout = predict_model(model, verbose=False)
        holdout_score = pull(pop=True)
        del holdout
        holdout_score.to_html("Holdout.html", col_space=65, justify="left")
        mlflow.log_artifact("Holdout.html")
        os.remove("Holdout.html")

        # Log AUC and Confusion Matrix plot

        if log_plots:

            logger.info(
                "SubProcess plot_model() called =================================="
            )

            try:
                plot_model(model, plot="auc", verbose=False, save=True, system=False)
                mlflow.log_artifact("AUC.png")
                os.remove("AUC.png")
            except:
                pass

            try:
                plot_model(
                    model,
                    plot="confusion_matrix",
                    verbose=False,
                    save=True,
                    system=False,
                )
                mlflow.log_artifact("Confusion Matrix.png")
                os.remove("Confusion Matrix.png")
            except:
                pass

            try:
                plot_model(
                    model, plot="feature", verbose=False, save=True, system=False
                )
                mlflow.log_artifact("Feature Importance.png")
                os.remove("Feature Importance.png")
            except:
                pass

            logger.info(
                "SubProcess plot_model() end =================================="
            )

        # Log hyperparameter tuning grid
        if tune_cv_results:
            d1 = tune_cv_results.get("params")
            dd = pd.DataFrame.from_dict(d1)
            dd["Score"] = tune_cv_results.get("mean_test_score")
            dd.to_html("Iterations.html", col_space=75, justify="left")
            mlflow.log_artifact("Iterations.html")
            os.remove("Iterations.html")

        # get default conda env
        from mlflow.sklearn import get_default_conda_env

        default_conda_env = get_default_conda_env()
        default_conda_env["name"] = f"{exp_name_log}-env"
        default_conda_env.get("dependencies").pop(-3)
        dependencies = default_conda_env.get("dependencies")[-1]
        from pycaret.utils import __version__

        dep = f"pycaret=={__version__}"
        dependencies["pip"] = [dep]

        # define model signature
        from mlflow.models.signature import infer_signature

        try:
            signature = infer_signature(
                data_before_preprocess.drop([target_param], axis=1)
            )
        except:
            logger.warning("Couldn't infer MLFlow signature.")
            signature = None
        input_example = (
            data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()
        )

        # log model as sklearn flavor
        prep_pipe_temp = deepcopy(_prep_pipe)
        prep_pipe_temp.steps.append(["trained model", model])
        mlflow.sklearn.log_model(
            prep_pipe_temp,
            "model",
            conda_env=default_conda_env,
            signature=signature,
            input_example=input_example,
        )
        del prep_pipe_temp


def _get_columns_to_stratify_by(
    X: pd.DataFrame, y: pd.DataFrame, stratify: Union[bool, List[str]], target: str
) -> pd.DataFrame:
    if not stratify:
        stratify = None
    else:
        if isinstance(stratify, list):
            data = pd.concat([X, y], axis=1)
            if not all(col in data.columns for col in stratify):
                raise ValueError("Column to stratify by does not exist in the dataset.")
            stratify = data[stratify]
        else:
            stratify = y
    return stratify


def _get_cv_splitter(fold):
    import pycaret.internal.utils

    return pycaret.internal.utils.get_cv_splitter(
        fold,
        default=fold_generator,
        seed=seed,
        shuffle=fold_shuffle_param,
        int_default="stratifiedkfold",
    )


def _get_cv_n_folds(fold):
    import pycaret.internal.utils

    return pycaret.internal.utils.get_cv_n_folds(fold, default_folds=fold_param)


def _get_pipeline_fit_kwargs(pipeline, fit_kwargs: dict) -> dict:
    try:
        model_step = pipeline.steps[-1]
    except:
        return fit_kwargs

    if any(k.startswith(f"{model_step[0]}__") for k in fit_kwargs.keys()):
        return fit_kwargs

    return {f"{model_step[0]}__{k}": v for k, v in fit_kwargs.items()}


def _get_groups(groups):
    import pycaret.internal.utils

    return pycaret.internal.utils.get_groups(groups, X_train, fold_groups_param)
