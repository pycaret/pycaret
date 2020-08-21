# Module: Classification
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT
# Release: PyCaret 2.1x
# Last modified : 12/08/2020

from pycaret.internal.utils import color_df
from pycaret.internal.logging import get_logger
from pycaret.internal.plotting import show_yellowbrick_plot
from pycaret.internal.Display import Display
import pandas
from typing import List, Set, Dict, Tuple, Optional, Any


def setup(
    data: pandas.DataFrame,
    target: str,
    train_size: float = 0.7,
    sampling: bool = True,
    sample_estimator=None,
    categorical_features: List[str] = None,
    categorical_imputation: str = "constant",
    ordinal_features: dict = None,
    high_cardinality_features: List[str] = None,
    high_cardinality_method: str = "frequency",
    numeric_features: List[str] = None,
    numeric_imputation: str = "mean",
    date_features: List[str] = None,
    ignore_features: List[str] = None,
    normalize: bool = False,
    normalize_method: str = "zscore",
    transformation: bool = False,
    transformation_method: str = "yeo-johnson",
    handle_unknown_categorical: bool = True,
    unknown_categorical_method: str = "least_frequent",
    pca: bool = False,
    pca_method: str = "linear",
    pca_components: float = None,
    ignore_low_variance: bool = False,
    combine_rare_levels: bool = False,
    rare_level_threshold: float = 0.10,
    bin_numeric_features: list = None,
    remove_outliers: bool = False,
    outliers_threshold: float = 0.05,
    remove_multicollinearity: bool = False,
    multicollinearity_threshold: float = 0.9,
    remove_perfect_collinearity: bool = False,  # added in pycaret==2.0.0
    create_clusters: bool = False,
    cluster_iter: int = 20,
    polynomial_features: bool = False,
    polynomial_degree: int = 2,
    trigonometry_features: bool = False,
    polynomial_threshold: float = 0.1,
    group_features: List[str] = None,
    group_names: List[str] = None,
    feature_selection: bool = False,
    feature_selection_threshold: float = 0.8,
    feature_selection_method: str = "classic",
    feature_interaction: bool = False,
    feature_ratio: bool = False,
    interaction_threshold: float = 0.01,
    fix_imbalance: bool = False,  # added in pycaret==2.0.0
    fix_imbalance_method: Any = None,  # added in pycaret==2.0.0
    data_split_shuffle: bool = True,  # added in pycaret==2.0.0
    folds_shuffle: bool = False,  # added in pycaret==2.0.0
    n_jobs: int = -1,  # added in pycaret==2.0.0
    use_gpu: bool = False,  # added in pycaret==2.1
    html: bool = True,  # added in pycaret==2.0.0
    session_id: int = None,
    log_experiment: bool = False,  # added in pycaret==2.0.0
    experiment_name: str = None,  # added in pycaret==2.0.0
    log_plots: bool = False,  # added in pycaret==2.0.0
    log_profile: bool = False,  # added in pycaret==2.0.0
    log_data: bool = False,  # added in pycaret==2.0.0
    silent: bool = False,
    verbose: bool = True,  # added in pycaret==2.0.0
    profile: bool = False,
    display: Display = None,
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
        Shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.

    target: string
        Name of the target column to be passed in as a string. The target variable could 
        be binary or multiclass. In case of a multiclass target, all estimators are wrapped
        with a OneVsRest classifier.

    train_size: float, default = 0.7
        Size of the training set. By default, 70% of the data will be used for training 
        and validation. The remaining data will be used for a test / hold-out set.

    sampling: bool, default = True
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
    
    categorical_imputation: string, default = 'constant'
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
    
    high_cardinality_method: string, default = 'frequency'
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

    numeric_imputation: string, default = 'mean'
        If missing values are found in numeric features, they will be imputed with the 
        mean value of the feature. The other available option is 'median' which imputes 
        the value using the median value in the training dataset. 
    
    date_features: string, default = None
        If the data has a DateTime column that is not automatically detected when running
        setup, this parameter can be used by passing date_features = 'date_column_name'. 
        It can work with multiple date columns. Date columns are not used in modeling. 
        Instead, feature extraction is performed and date columns are dropped from the 
        dataset. If the date column includes a time stamp, features related to time will 
        also be extracted.
    
    ignore_features: string, default = None
        If any feature should be ignored for modeling, it can be passed to the param
        ignore_features. The ID and DateTime columns when inferred, are automatically 
        set to ignore for modeling. 
    
    normalize: bool, default = False
        When set to True, the feature space is transformed using the normalized_method
        param. Generally, linear algorithms perform better with normalized data however, 
        the results may vary and it is advised to run multiple experiments to evaluate
        the benefit of normalization.
    
    normalize_method: string, default = 'zscore'
        Defines the method to be used for normalization. By default, normalize method
        is set to 'zscore'. The standard zscore is calculated as z = (x - u) / s. The
        other available options are:
    
        'minmax'    : scales and translates each feature individually such that it is in 
                    the range of 0 - 1.
        
        'maxabs'    : scales and translates each feature individually such that the maximal 
                    absolute value of each feature will be 1.0. It does not shift/center 
                    the data, and thus does not destroy any sparsity.
        
        'robust'    : scales and translates each feature according to the Interquartile range.
                    When the dataset contains outliers, robust scaler often gives better
                    results.
    
    transformation: bool, default = False
        When set to True, a power transformation is applied to make the data more normal /
        Gaussian-like. This is useful for modeling issues related to heteroscedasticity or 
        other situations where normality is desired. The optimal parameter for stabilizing 
        variance and minimizing skewness is estimated through maximum likelihood.
    
    transformation_method: string, default = 'yeo-johnson'
        Defines the method for transformation. By default, the transformation method is set
        to 'yeo-johnson'. The other available option is 'quantile' transformation. Both 
        the transformation transforms the feature set to follow a Gaussian-like or normal
        distribution. Note that the quantile transformer is non-linear and may distort linear 
        correlations between variables measured at the same scale.
    
    handle_unknown_categorical: bool, default = True
        When set to True, unknown categorical levels in new / unseen data are replaced by
        the most or least frequent level as learned in the training data. The method is 
        defined under the unknown_categorical_method param.
    
    unknown_categorical_method: string, default = 'least_frequent'
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

    pca_method: string, default = 'linear'
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
        When set to True, all categorical features with statistically insignificant variances 
        are removed from the dataset. The variance is calculated using the ratio of unique 
        values to the number of samples, and the ratio of the most common value to the 
        frequency of the second most common value.
    
    combine_rare_levels: bool, default = False
        When set to True, all levels in categorical features below the threshold defined 
        in rare_level_threshold param are combined together as a single level. There must be 
        atleast two levels under the threshold for this to take effect. rare_level_threshold
        represents the percentile distribution of level frequency. Generally, this technique 
        is applied to limit a sparse matrix caused by high numbers of levels in categorical 
        features. 
    
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
        When a dataset contains features that have related characteristics, the group_features
        param can be used for statistical feature extraction. For example, if a dataset has 
        numeric features that are related with each other (i.e 'Col1', 'Col2', 'Col3'), a list 
        containing the column names can be passed under group_features to extract statistical 
        information such as the mean, median, mode and standard deviation.
    
    group_names: list, default = None
        When group_features is passed, a name of the group can be passed into the group_names 
        param as a list containing strings. The length of a group_names list must equal to the 
        length  of group_features. When the length doesn't match or the name is not passed, new 
        features are sequentially named such as group_1, group_2 etc.
    
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
        A higher value will result in a higher feature space. It is recommended to do multiple
        trials with different values of feature_selection_threshold specially in cases where 
        polynomial_features and feature_interaction are used. Setting a very low value may be 
        efficient but could result in under-fitting.
    
    feature_selection_method: str, default = 'classic'
        Can be either 'classic' or 'boruta'. Selects the algorithm responsible for
        choosing a subset of features. For the 'classic' selection method, PyCaret will use various
        permutation importance techniques. For the 'boruta' algorithm, PyCaret will create 
        an instance of boosted trees model, which will iterate with permutation over all
        features and choose the best ones based on the distributions of feature importance.
        More in: https://pdfs.semanticscholar.org/85a8/b1d9c52f9f795fda7e12376e751526953f38.pdf%3E

    feature_interaction: bool, default = False 
        When set to True, it will create new features by interacting (a * b) for all numeric 
        variables in the dataset including polynomial and trigonometric features (if created). 
        This feature is not scalable and may not work as expected on datasets with large 
        feature space.
    
    feature_ratio: bool, default = False
        When set to True, it will create new features by calculating the ratios (a / b) of all 
        numeric variables in the dataset. This feature is not scalable and may not work as 
        expected on datasets with large feature space.
    
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
        When fix_imbalance is set to True and fix_imbalance_method is None, 'smote' is applied 
        by default to oversample minority class during cross validation. This parameter
        accepts any module from 'imblearn' that supports 'fit_resample' method.

    data_split_shuffle: bool, default = True
        If set to False, prevents shuffling of rows when splitting data.

    folds_shuffle: bool, default = False
        If set to False, prevents shuffling of rows when using cross validation.

    n_jobs: int, default = -1
        The number of jobs to run in parallel (for functions that supports parallel 
        processing) -1 means using all processors. To run all functions on single processor 
        set n_jobs to None.

    use_gpu: bool, default = False
        If set to True, algorithms that supports gpu are trained using gpu.

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
        When set to True, data profile is also logged on MLflow as a html file. By default,
        it is set to False. 

    log_data: bool, default = False
        When set to True, train and test dataset are logged as csv. 

    silent: bool, default = False
        When set to True, confirmation of data types is not required. All preprocessing will 
        be performed assuming automatically inferred data types. Not recommended for direct use 
        except for established pipelines.
    
    verbose: Boolean, default = True
        Information grid is not printed when verbose is set to False.

    profile: bool, default = False
        If set to true, a data profile for Exploratory Data Analysis will be displayed 
        in an interactive HTML report. 
    
    Returns
    -------
    info_grid
        Information grid is printed.

    environment
        This function returns various outputs that are stored in variables
        as tuples. They are used by other functions in pycaret.
      
       
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    # exception checking
    import sys

    from pycaret.utils import __version__

    ver = __version__

    # create logger
    global logger

    logger = get_logger()

    logger.info("PyCaret Classification Module")
    logger.info(f"version {ver}")
    logger.info("Initializing setup()")
    logger.info(f"setup({function_params_str})")

    # generate USI for mlflow tracking
    import secrets

    global USI
    USI = secrets.token_hex(nbytes=2)
    logger.info(f"USI: {USI}")

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
            "cannot find psutil installation. memory not traceable. Install psutil using pip to enable memory logging. "
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
        import warnings

        warnings.filterwarnings("ignore")
        logger.info(f"mlflow=={VERSION}")
    except:
        logger.warning("mlflow not found")

    # run_time
    import datetime, time

    runtime_start = time.time()

    logger.info("Checking Exceptions")

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
            "categorical_imputation param only accepts 'constant' or 'mode' "
        )

    # ordinal_features
    if ordinal_features is not None:
        if type(ordinal_features) is not dict:
            raise TypeError(
                "ordinal_features must be of type dictionary with column name as key and ordered values as list. "
            )

    # ordinal features check
    if ordinal_features is not None:
        data_cols = data.columns
        data_cols = data_cols.drop(target)
        ord_keys = ordinal_features.keys()

        for i in ord_keys:
            if i not in data_cols:
                raise ValueError(
                    "Column name passed as a key in ordinal_features param doesnt exist. "
                )

        for k in ord_keys:
            if data[k].nunique() != len(ordinal_features[k]):
                raise ValueError(
                    "Levels passed in ordinal_features param doesnt match with levels in data. "
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
                "high_cardinality_features param only accepts name of columns as a list. "
            )

    if high_cardinality_features is not None:
        data_cols = data.columns
        data_cols = data_cols.drop(target)
        for i in high_cardinality_features:
            if i not in data_cols:
                raise ValueError(
                    "Column type forced is either target column or doesn't exist in the dataset."
                )

    # high_cardinality_methods
    high_cardinality_allowed_methods = ["frequency", "clustering"]
    if high_cardinality_method not in high_cardinality_allowed_methods:
        raise ValueError(
            "high_cardinality_method param only accepts 'frequency' or 'clustering' "
        )

    # checking numeric imputation
    allowed_numeric_imputation = ["mean", "median"]
    if numeric_imputation not in allowed_numeric_imputation:
        raise ValueError("numeric_imputation param only accepts 'mean' or 'median' ")

    # checking normalize method
    allowed_normalize_method = ["zscore", "minmax", "maxabs", "robust"]
    if normalize_method not in allowed_normalize_method:
        raise ValueError(
            "normalize_method param only accepts 'zscore', 'minxmax', 'maxabs' or 'robust'. "
        )

    # checking transformation method
    allowed_transformation_method = ["yeo-johnson", "quantile"]
    if transformation_method not in allowed_transformation_method:
        raise ValueError(
            "transformation_method param only accepts 'yeo-johnson' or 'quantile'. "
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
            "unknown_categorical_method only accepts 'least_frequent' or 'most_frequent'."
        )

    # check pca
    if type(pca) is not bool:
        raise TypeError("PCA parameter only accepts True or False.")

    # pca method check
    allowed_pca_methods = ["linear", "kernel", "incremental"]
    if pca_method not in allowed_pca_methods:
        raise ValueError(
            "pca method param only accepts 'linear', 'kernel', or 'incremental'. "
        )

    # pca components check
    if pca is True:
        if pca_method != "linear":
            if pca_components is not None:
                if (type(pca_components)) is not int:
                    raise TypeError(
                        "pca_components parameter must be integer when pca_method is not 'linear'. "
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
        raise TypeError("rare_level_threshold must be a float between 0 and 1. ")

    # bin numeric features
    if bin_numeric_features is not None:
        all_cols = list(data.columns)
        all_cols.remove(target)

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
        raise TypeError("outliers_threshold must be a float between 0 and 1. ")

    # remove_multicollinearity
    if type(remove_multicollinearity) is not bool:
        raise TypeError(
            "remove_multicollinearity parameter only accepts True or False."
        )

    # multicollinearity_threshold
    if type(multicollinearity_threshold) is not float:
        raise TypeError("multicollinearity_threshold must be a float between 0 and 1. ")

    # create_clusters
    if type(create_clusters) is not bool:
        raise TypeError("create_clusters parameter only accepts True or False.")

    # cluster_iter
    if type(cluster_iter) is not int:
        raise TypeError("cluster_iter must be a integer greater than 1. ")

    # polynomial_features
    if type(polynomial_features) is not bool:
        raise TypeError("polynomial_features only accepts True or False. ")

    # polynomial_degree
    if type(polynomial_degree) is not int:
        raise TypeError("polynomial_degree must be an integer. ")

    # polynomial_features
    if type(trigonometry_features) is not bool:
        raise TypeError("trigonometry_features only accepts True or False. ")

    # polynomial threshold
    if type(polynomial_threshold) is not float:
        raise TypeError("polynomial_threshold must be a float between 0 and 1. ")

    # group features
    if group_features is not None:
        if type(group_features) is not list:
            raise TypeError("group_features must be of type list. ")

    if group_names is not None:
        if type(group_names) is not list:
            raise TypeError("group_names must be of type list. ")

    # cannot drop target
    if ignore_features is not None:
        if target in ignore_features:
            raise ValueError("cannot drop target column. ")

    # feature_selection
    if type(feature_selection) is not bool:
        raise TypeError("feature_selection only accepts True or False. ")

    # feature_selection_threshold
    if type(feature_selection_threshold) is not float:
        raise TypeError("feature_selection_threshold must be a float between 0 and 1. ")

    # feature_selection_method
    feature_selection_methods = ["boruta", "classic"]
    if feature_selection_method not in feature_selection_methods:
        raise TypeError(
            f"feature_selection_method must be one of {', '.join(feature_selection_methods)}"
        )

    # feature_interaction
    if type(feature_interaction) is not bool:
        raise TypeError("feature_interaction only accepts True or False. ")

    # feature_ratio
    if type(feature_ratio) is not bool:
        raise TypeError("feature_ratio only accepts True or False. ")

    # interaction_threshold
    if type(interaction_threshold) is not float:
        raise TypeError("interaction_threshold must be a float between 0 and 1. ")

    # forced type check
    all_cols = list(data.columns)
    all_cols.remove(target)

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
        raise TypeError("log_experiment parameter only accepts True or False. ")

    # log_profile
    if type(log_profile) is not bool:
        raise TypeError("log_profile parameter only accepts True or False. ")

    # experiment_name
    if experiment_name is not None:
        if type(experiment_name) is not str:
            raise TypeError("experiment_name parameter must be string if not None. ")

    # silent
    if type(silent) is not bool:
        raise TypeError("silent parameter only accepts True or False. ")

    # remove_perfect_collinearity
    if type(remove_perfect_collinearity) is not bool:
        raise TypeError(
            "remove_perfect_collinearity parameter only accepts True or False."
        )

    # html
    if type(html) is not bool:
        raise TypeError("html parameter only accepts True or False.")

    # use_gpu
    if type(use_gpu) is not bool:
        raise TypeError("use_gpu parameter only accepts True or False.")

    # folds_shuffle
    if type(folds_shuffle) is not bool:
        raise TypeError("folds_shuffle parameter only accepts True or False.")

    # data_split_shuffle
    if type(data_split_shuffle) is not bool:
        raise TypeError("data_split_shuffle parameter only accepts True or False.")

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

    logger.info("Preloading libraries")

    # pre-load libraries
    import pandas as pd
    import os

    # pandas option
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.max_rows", 500)

    # global html_param
    global html_param

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
            ["ETC", ". . . . . . . . . . . . . . . . . .", "Calculating ETC"],
        ]
        display = Display(
            verbose, html_param, progress_args, monitor_rows, logger=logger,
        )

        display.display_progress()
        display.display_monitor()

    logger.info("Importing libraries")

    # general dependencies
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import random
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px

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

    # declaring global variables to be accessed by other functions
    logger.info("Declaring global variables")
    global X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, experiment__, folds_shuffle_param, n_jobs_param, create_model_container, master_model_container, display_container, exp_name_log, logging_param, log_plots_param, fix_imbalance_param, fix_imbalance_method_param, data_before_preprocess, target_param, gpu_param, all_models, _all_models_internal, all_metrics

    logger.info("Copying data for preprocessing")

    # copy original data for pandas profiler
    data_before_preprocess = data.copy()

    # generate seed to be used globally
    if session_id is None:
        seed = random.randint(150, 9000)
    else:
        seed = session_id

    np.random.seed(seed)

    """
    preprocessing starts here
    """

    display.update_monitor(1, "Preparing Data for Modeling")
    display.display_monitor()

    # define parameters for preprocessor

    logger.info("Declaring preprocessing parameters")

    # categorical features
    if categorical_features is None:
        cat_features_pass = []
    else:
        cat_features_pass = categorical_features

    # numeric features
    if numeric_features is None:
        numeric_features_pass = []
    else:
        numeric_features_pass = numeric_features

    # drop features
    if ignore_features is None:
        ignore_features_pass = []
    else:
        ignore_features_pass = ignore_features

    # date features
    if date_features is None:
        date_features_pass = []
    else:
        date_features_pass = date_features

    # categorical imputation strategy
    if categorical_imputation == "constant":
        categorical_imputation_pass = "not_available"
    elif categorical_imputation == "mode":
        categorical_imputation_pass = "most frequent"

    # transformation method strategy
    if transformation_method == "yeo-johnson":
        trans_method_pass = "yj"
    elif transformation_method == "quantile":
        trans_method_pass = "quantile"

    # pass method
    if pca_method == "linear":
        pca_method_pass = "pca_liner"

    elif pca_method == "kernel":
        pca_method_pass = "pca_kernal"

    elif pca_method == "incremental":
        pca_method_pass = "incremental"

    elif pca_method == "pls":
        pca_method_pass = "pls"

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

    if bin_numeric_features is None:
        apply_binning_pass = False
        features_to_bin_pass = []

    else:
        apply_binning_pass = True
        features_to_bin_pass = bin_numeric_features

    # trignometry
    if trigonometry_features is False:
        trigonometry_features_pass = []
    else:
        trigonometry_features_pass = ["sin", "cos", "tan"]

    # group features
    # =============#

    # apply grouping
    if group_features is not None:
        apply_grouping_pass = True
    else:
        apply_grouping_pass = False

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

    if feature_interaction or feature_ratio:
        apply_feature_interactions_pass = True
    else:
        apply_feature_interactions_pass = False

    interactions_to_apply_pass = []

    if feature_interaction:
        interactions_to_apply_pass.append("multiply")

    if feature_ratio:
        interactions_to_apply_pass.append("divide")

    # unknown categorical
    if unknown_categorical_method == "least_frequent":
        unknown_categorical_method_pass = "least frequent"
    elif unknown_categorical_method == "most_frequent":
        unknown_categorical_method_pass = "most frequent"

    # ordinal_features
    if ordinal_features is not None:
        apply_ordinal_encoding_pass = True
    else:
        apply_ordinal_encoding_pass = False

    if apply_ordinal_encoding_pass is True:
        ordinal_columns_and_categories_pass = ordinal_features
    else:
        ordinal_columns_and_categories_pass = {}

    if high_cardinality_features is not None:
        apply_cardinality_reduction_pass = True
    else:
        apply_cardinality_reduction_pass = False

    if high_cardinality_method == "frequency":
        cardinal_method_pass = "count"
    elif high_cardinality_method == "clustering":
        cardinal_method_pass = "cluster"

    if apply_cardinality_reduction_pass:
        cardinal_features_pass = high_cardinality_features
    else:
        cardinal_features_pass = []

    if silent:
        display_dtypes_pass = False
    else:
        display_dtypes_pass = True

    logger.info("Importing preprocessing module")

    # import library
    import pycaret.preprocess as preprocess

    logger.info("Creating preprocessing pipeline")

    data = preprocess.Preprocess_Path_One(
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

    display.move_progress()
    logger.info("Preprocessing pipeline created successfully")

    if hasattr(preprocess.dtypes, "replacement"):
        label_encoded = preprocess.dtypes.replacement
        label_encoded = str(label_encoded).replace("'", "")
        label_encoded = str(label_encoded).replace("{", "")
        label_encoded = str(label_encoded).replace("}", "")

    else:
        label_encoded = "None"

    try:
        res_type = ["quit", "Quit", "exit", "EXIT", "q", "Q", "e", "E", "QUIT", "Exit"]
        res = preprocess.dtypes.response

        if res in res_type:
            sys.exit(
                "(Process Exit): setup has been interupted with user command 'quit'. setup must rerun."
            )

    except:
        logger.error(
            "(Process Exit): setup has been interupted with user command 'quit'. setup must rerun."
        )

    # save prep pipe
    prep_pipe = preprocess.pipe

    logger.info("Creating grid variables")

    # generate values for grid show
    missing_values = data_before_preprocess.isna().sum().sum()
    if missing_values > 0:
        missing_flag = True
    else:
        missing_flag = False

    if normalize is True:
        normalize_grid = normalize_method
    else:
        normalize_grid = "None"

    if transformation is True:
        transformation_grid = transformation_method
    else:
        transformation_grid = "None"

    if pca is True:
        pca_method_grid = pca_method
    else:
        pca_method_grid = "None"

    if pca is True:
        pca_components_grid = pca_components_pass
    else:
        pca_components_grid = "None"

    if combine_rare_levels:
        rare_level_threshold_grid = rare_level_threshold
    else:
        rare_level_threshold_grid = "None"

    if bin_numeric_features is None:
        numeric_bin_grid = False
    else:
        numeric_bin_grid = True

    if remove_outliers is False:
        outliers_threshold_grid = None
    else:
        outliers_threshold_grid = outliers_threshold

    if remove_multicollinearity is False:
        multicollinearity_threshold_grid = None
    else:
        multicollinearity_threshold_grid = multicollinearity_threshold

    if create_clusters is False:
        cluster_iter_grid = None
    else:
        cluster_iter_grid = cluster_iter

    if polynomial_features:
        polynomial_degree_grid = polynomial_degree
    else:
        polynomial_degree_grid = None

    if polynomial_features or trigonometry_features:
        polynomial_threshold_grid = polynomial_threshold
    else:
        polynomial_threshold_grid = None

    if feature_selection:
        feature_selection_threshold_grid = feature_selection_threshold
    else:
        feature_selection_threshold_grid = None

    if feature_interaction or feature_ratio:
        interaction_threshold_grid = interaction_threshold
    else:
        interaction_threshold_grid = None

    if ordinal_features is not None:
        ordinal_features_grid = True
    else:
        ordinal_features_grid = False

    if handle_unknown_categorical:
        unknown_categorical_method_grid = unknown_categorical_method
    else:
        unknown_categorical_method_grid = None

    if group_features is not None:
        group_features_grid = True
    else:
        group_features_grid = False

    if high_cardinality_features is not None:
        high_cardinality_features_grid = True
    else:
        high_cardinality_features_grid = False

    if high_cardinality_features_grid:
        high_cardinality_method_grid = high_cardinality_method
    else:
        high_cardinality_method_grid = None

    learned_types = preprocess.dtypes.learent_dtypes
    learned_types.drop(target, inplace=True)

    float_type = 0
    cat_type = 0

    for i in preprocess.dtypes.learent_dtypes:
        if "float" in str(i):
            float_type += 1
        elif "object" in str(i):
            cat_type += 1
        elif "int" in str(i):
            float_type += 1

    """
    preprocessing ends here
    """

    # reset pandas option
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")

    logger.info("Creating global containers")

    # create an empty list for pickling later.
    experiment__ = []

    # create folds_shuffle_param
    folds_shuffle_param = folds_shuffle

    # create n_jobs_param
    n_jobs_param = n_jobs

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

    # create a fix_imbalance_param and fix_imbalance_method_param
    fix_imbalance_param = fix_imbalance
    fix_imbalance_method_param = fix_imbalance_method

    if fix_imbalance_method_param is None:
        fix_imbalance_model_name = "SMOTE"
    else:
        fix_imbalance_model_name = str(fix_imbalance_method_param).split("(")[0]

    # create target_param var
    target_param = target

    # create gpu_param var
    gpu_param = use_gpu

    # sample estimator
    if sample_estimator is None:
        model = LogisticRegression()
    else:
        model = sample_estimator

    model_name = str(model).split("(")[0]
    if "CatBoostClassifier" in model_name:
        model_name = "CatBoostClassifier"

    # creating variables to be used later in the function
    X = data.drop(target, axis=1)
    y = data[target]

    # determining target type
    if _is_multiclass():
        target_type = "Multiclass"
    else:
        target_type = "Binary"

    display.move_progress()

    if sampling is True and data.shape[0] > 25000:  # change this back to 25000

        X_train, X_test, y_train, y_test = _sample_data(
            model, seed, train_size, data_split_shuffle, display
        )

    else:
        display.update_monitor(1, "Splitting Data")
        display.display_monitor()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=1 - train_size,
            stratify=y,
            random_state=seed,
            shuffle=data_split_shuffle,
        )
        display.move_progress()

    """
    Final display Starts
    """
    if profile:
        print("Setup Succesfully Completed! Loading Profile Now... Please Wait!")
    else:
        if verbose:
            print("Setup Succesfully Completed!")

    functions = pd.DataFrame(
        [
            ["session_id", seed],
            ["Target Type", target_type],
            ["Label Encoded", label_encoded],
            ["Original Data", data_before_preprocess.shape],
            ["Missing Values ", missing_flag],
            ["Numeric Features ", str(float_type)],
            ["Categorical Features ", str(cat_type)],
            ["Ordinal Features ", ordinal_features_grid],
            ["High Cardinality Features ", high_cardinality_features_grid],
            ["High Cardinality Method ", high_cardinality_method_grid],
            [
                "Sampled Data",
                f"({X_train.shape[0] + X_test.shape[0]}, {data_before_preprocess.shape[1]})",
            ],
            ["Transformed Train Set", X_train.shape],
            ["Transformed Test Set", X_test.shape],
            ["Numeric Imputer ", numeric_imputation],
            ["Categorical Imputer ", categorical_imputation],
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
            ["Fix Imbalance", fix_imbalance_param],
            ["Fix Imbalance Method", fix_imbalance_model_name],
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

    all_models = models(force_regenerate=True)
    _all_models_internal = models(internal=True, force_regenerate=True)
    all_metrics = get_metrics()

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    if logging_param:

        logger.info("Logging experiment in MLFlow")

        import mlflow
        from pathlib import Path

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
        folds_shuffle_param,
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
    )


def compare_models(
    exclude: List[str] = None,
    include: list = None,  # added in pycaret==2.0.0
    fold: int = 10,
    round: int = 4,
    sort: str = "Accuracy",
    n_select: int = 1,  # added in pycaret==2.0.0
    budget_time: float = 0,  # added in pycaret==2.1.0
    turbo: bool = True,
    verbose: bool = True,
    display: Display = None,
):  # added in pycaret==2.0.0

    """
    This function train all the models available in the model library and scores them 
    using Stratified Cross Validation. The output prints a score grid with Accuracy, 
    AUC, Recall, Precision, F1, Kappa and MCC (averaged accross folds), determined by
    fold parameter.
    
    This function returns the best model based on metric defined in sort parameter. 
    
    To select top N models, use n_select parameter that is set to 1 by default.
    Where n_select parameter > 1, it will return a list of trained model objects.

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

    fold: integer, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.
  
    sort: string, default = 'Accuracy'
        The scoring measure specified is used for sorting the average score grid
        Other options are 'AUC', 'Recall', 'Precision', 'F1', 'Kappa' and 'MCC'.

    n_select: int, default = 1
        Number of top_n models to return. use negative argument for bottom selection.
        for example, n_select = -3 means bottom 3 models.

    budget_time: int or float, default = 0
        If set above 0, will terminate execution of the function after budget_time minutes have
        passed and return results up to that point.

    turbo: Boolean, default = True
        When turbo is set to True, it excludes estimators that have longer
        training time.

    verbose: Boolean, default = True
        Score grid is not printed when verbose is set to False.
    
    Returns
    -------
    score_grid
        A table containing the scores of the model across the kfolds. 
        Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are also returned.

    Warnings
    --------
    - compare_models() though attractive, might be time consuming with large 
      datasets. By default turbo is set to True, which excludes models that
      have longer training times. Changing turbo parameter to False may result 
      in very high training times with datasets where number of samples exceed 
      10,000.
      
    - If target variable is multiclass (more than 2 classes), AUC will be 
      returned as zero (0.0)      
             
    
    """

    """
    
    ERROR HANDLING STARTS HERE
    
    """
    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing compare_models()")
    logger.info(f"compare_models({function_params_str})")

    logger.info("Checking exceptions")

    # exception checking
    import sys

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
    if type(fold) is not int:
        raise TypeError("Fold parameter only accepts integer value.")

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    # checking n_select parameter
    if type(n_select) is not int:
        raise TypeError("n_select parameter only accepts integer value.")

    # checking budget_time parameter
    if type(budget_time) is not int and type(budget_time) is not float:
        raise TypeError("budget_time parameter only accepts integer or float values.")

    # checking sort parameter
    sort = _get_metric(sort)
    if sort is None:
        raise ValueError(
            f"Sort method not supported. See docstring for list of available parameters."
        )

    # checking optimize parameter for multiclass
    if _is_multiclass():
        if not all_metrics[all_metrics["Name"] == sort].iloc[0]["Multiclass"]:
            raise TypeError(
                f"{sort} metric not supported for multiclass problems. See docstring for list of other optimization parameters."
            )

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    logger.info("Preloading libraries")

    # pre-load libraries
    import pandas as pd
    import datetime, time

    pd.set_option("display.max_columns", 500)

    logger.info("Preparing display monitor")

    len_mod = len(all_models[all_models["Turbo"] == True]) if turbo else len(all_models)

    if include:
        len_mod = len(include)
    elif exclude:
        len_mod -= len(exclude)

    if not display:
        progress_args = {"max": ((fold + 4) * len_mod) + 4 + len_mod}
        master_display_columns = ["Model"] + all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
            ["Estimator", ". . . . . . . . . . . . . . . . . .", "Compiling Library"],
            ["ETC", ". . . . . . . . . . . . . . . . . .", "Calculating ETC"],
        ]
        display = Display(
            verbose,
            html_param,
            progress_args,
            master_display_columns,
            monitor_rows,
            logger=logger,
        )

        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    # general dependencies
    import numpy as np
    import random
    from sklearn import metrics
    import pandas.io.formats.style

    np.random.seed(seed)

    logger.info("Copying training dataset")
    # defining X_train and y_train as data_X and data_y
    data_X = X_train
    data_y = y_train

    display.move_progress()

    # defining sort parameter (making Precision equivalent to Prec. )

    sort = sort["Display Name"]

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

        if (
            not hasattr(model, "estimators")
            and not isinstance(model, str)
            and _is_multiclass()
        ):
            model_name = _get_model_name(model.estimator)
        else:
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
        display.update_monitor(3, "Calculating ETC")
        display.display_monitor()

        """
        MONITOR UPDATE ENDS
        """
        display.replace_master_display(None)

        logger.info(
            "SubProcess create_model() called =================================="
        )
        model = create_model(
            estimator=model,
            system=False,
            verbose=False,
            display=display,
            fold=fold,
            round=round,
            budget_time=budget_time - total_runtime
            if budget_time and budget_time > 0
            else 0,
        )
        model_results = pull(pop=True)
        logger.info("SubProcess create_model() end ==================================")

        if not model:
            over_time_budget = True
            logger.info(f"Time budged exceeded in create_model(), breaking loop")
            break

        logger.info("Creating metrics dataframe")
        compare_models_ = pd.DataFrame(model_results.loc["Mean"]).T
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
        master_display = master_display.sort_values(by=sort, ascending=False)
        # master_display.reset_index(drop=True, inplace=True)

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

        """
        MLflow logging starts here
        """

        if logging_param:

            logger.info("Creating MLFlow logs")

            import mlflow
            from pathlib import Path
            import os

            run_name = model_name

            with mlflow.start_run(run_name=run_name) as run:

                # Get active run to log as tag
                RunID = mlflow.active_run().info.run_id

                params = model.get_params()

                for i in list(params):
                    v = params.get(i)
                    if len(str(v)) > 250:
                        params.pop(i)

                mlflow.log_params(params)

                # set tag of compare_models
                mlflow.set_tag("Source", "compare_models")
                mlflow.set_tag("URI", URI)
                mlflow.set_tag("USI", USI)
                mlflow.set_tag("Run Time", runtime)
                mlflow.set_tag("Run ID", RunID)

                # Log top model metrics
                mlflow.log_metrics(
                    {
                        k: v
                        for k, v in master_display.drop(
                            ["Object", "Model", "TT (Sec)"], axis=1
                        )
                        .iloc[0]
                        .items()
                    }
                )

                # Log model and transformation pipeline
                from copy import deepcopy

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

                signature = infer_signature(
                    data_before_preprocess.drop([target_param], axis=1)
                )
                input_example = (
                    data_before_preprocess.drop([target_param], axis=1)
                    .iloc[0]
                    .to_dict()
                )

                # log model as sklearn flavor
                prep_pipe_temp = deepcopy(prep_pipe)
                prep_pipe_temp.steps.append(["trained model", model])
                mlflow.sklearn.log_model(
                    prep_pipe_temp,
                    "model",
                    conda_env=default_conda_env,
                    signature=signature,
                    input_example=input_example,
                )
                del prep_pipe_temp

    display.move_progress()

    def highlight_max(s):
        to_highlight = s == s.max()
        return ["background-color: yellow" if v else "" for v in to_highlight]

    def highlight_cols(s):
        color = "lightgrey"
        return f"background-color: {color}"

    if _is_multiclass():

        compare_models_ = (
            master_display.drop("Object", axis=1)
            .style.apply(highlight_max, subset=master_display.columns[2:],)
            .applymap(highlight_cols, subset=["TT (Sec)"])
        )
    else:

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

    display.update_monitor(1, "Compiling Final Model")
    display.update_monitor(3, "Almost Finished")
    display.display_monitor()

    sorted_models = master_display["Object"].to_list()
    n_select = n_select if n_select <= len(sorted_models) else len(sorted_models)
    if n_select < 0:
        sorted_models = sorted_models[n_select:]
    else:
        sorted_models = sorted_models[:n_select]

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
    estimator=None,
    ensemble: bool = False,
    method: str = None,
    fold: int = 10,
    round: int = 4,
    cross_validation: bool = True,  # added in pycaret==2.0.0
    budget_time: float = 0,
    verbose: bool = True,
    system: bool = True,  # added in pycaret==2.0.0
    return_fit_time: bool = False,  # added in pycaret==2.2.0
    X_train_data: pandas.DataFrame = None,  # added in pycaret==2.2.0
    Y_train_data: pandas.DataFrame = None,  # added in pycaret==2.2.0
    display: Display = None,  # added in pycaret==2.2.0
    **kwargs,
):  # added in pycaret==2.0.0

    """  
    This function creates a model and scores it using Stratified Cross Validation. 
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
    estimator : string / object, default = None
        Enter ID of the estimators available in model library or pass an untrained model 
        object consistent with fit / predict API to train and evaluate model. All estimators 
        support binary or multiclass problem. List of estimators in model library (ID - Name):

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

    ensemble: Boolean, default = False
        True would result in an ensemble of estimator using the method parameter defined. 

    method: String, 'Bagging' or 'Boosting', default = None.
        method must be defined when ensemble is set to True. Default method is set to None. 

    fold: integer, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    budget_time: int or float, default = 0
        If set above 0, will terminate execution of the function after budget_time minutes have
        passed.

    cross_validation: bool, default = True
        When cross_validation set to False fold parameter is ignored and model is trained
        on entire training dataset. No metric evaluation is returned. 

    verbose: Boolean, default = True
        Score grid is not printed when verbose is set to False.

    system: Boolean, default = True
        Must remain True all times. Only to be changed by internal functions.
    
    return_fit_time: Boolean, default = False
        If True, will return a tuple of the model and its fit time.
        Only to be changed by internal functions.

    X_train_data: pandas.DataFrame, default = None
        If not None, will use this dataframe as training features.
        Intended to be only changed by internal functions.

    Y_train_data: pandas.DataFrame, default = None
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

    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing create_model()")
    logger.info(f"create_model({function_params_str})")

    logger.info("Checking exceptions")

    # exception checking
    import sys

    # run_time
    import datetime, time

    runtime_start = time.time()

    available_estimators = set(_all_models_internal.index)

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

    # checking error for ensemble:
    if type(ensemble) is not bool:
        raise TypeError("Ensemble parameter can only take argument as True or False.")

    # checking error for method:

    # 1 Check When method given and ensemble is not set to True.
    if ensemble is False and method is not None:
        raise TypeError(
            "Method parameter only accepts value when ensemble is set to True."
        )

    # 2 Check when ensemble is set to True and method is not passed.
    if ensemble is True and method is None:
        raise TypeError(
            "Method parameter missing. Pass method = 'Bagging' or 'Boosting'."
        )

    # 3 Check when ensemble is set to True and method is passed but not allowed.
    available_method = ["Bagging", "Boosting"]
    if ensemble is True and method not in available_method:
        raise ValueError(
            "Method parameter only accepts two values 'Bagging' or 'Boosting'."
        )

    # checking fold parameter
    if type(fold) is not int:
        raise TypeError("Fold parameter only accepts integer value.")

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    # checking budget_time parameter
    if type(budget_time) is not int and type(budget_time) is not float:
        raise TypeError("budget_time parameter only accepts integer or float values.")

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

    # checking boosting conflict with estimators
    if (
        ensemble is True
        and method == "Boosting"
        and (
            (
                isinstance(estimator, str)
                and not _all_models_internal.loc[estimator]["Boosting Supported"]
            )
            or not (
                hasattr(estimator, "class_weights")
                or hasattr(estimator, "predict_proba")
            )
        )
    ):
        raise TypeError(
            "Estimator does not provide class_weights or predict_proba function and hence not supported for the Boosting method. Change the estimator or method to 'Bagging'."
        )

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    logger.info("Preloading libraries")

    # pre-load libraries
    import pandas as pd

    if not display:
        progress_args = {"max": fold + 4}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
            ["ETC", ". . . . . . . . . . . . . . . . . .", "Calculating ETC"],
        ]
        display = Display(
            verbose,
            html_param,
            progress_args,
            master_display_columns,
            monitor_rows,
            logger=logger,
        )
        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    logger.info("Importing libraries")

    # general dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    np.random.seed(seed)

    logger.info("Copying training dataset")

    # Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy() if X_train_data is None else X_train_data
    data_y = y_train.copy() if Y_train_data is None else Y_train_data

    # reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)

    display.move_progress()

    logger.info("Defining folds")

    # cross validation setup starts here
    kf = StratifiedKFold(fold, random_state=seed, shuffle=folds_shuffle_param)

    logger.info("Declaring metric variables")

    score_dict = {metric: np.empty((0, 0)) for metric in all_metrics["Display Name"]}

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

        if _is_one_vs_rest(model):
            full_name = _get_model_name(model.estimator)
        else:
            full_name = _get_model_name(model)

    logger.info(f"{full_name} Imported succesfully")

    display.move_progress()

    # checking method when ensemble is set to True.

    logger.info("Checking ensemble method")

    if method == "Bagging":
        logger.info("Ensemble method set to Bagging")
        bagging_model_definition = _all_models_internal.loc["Bagging"]

        model = bagging_model_definition["Class"](
            model, bootstrap=True, n_estimators=10, **bagging_model_definition["Args"]
        )

    elif method == "Boosting":
        logger.info("Ensemble method set to Boosting")
        boosting_model_definition = _all_models_internal.loc["ada"]
        model = boosting_model_definition["Class"](
            model, n_estimators=10, **boosting_model_definition["Args"]
        )

    onevsrest_model_definition = _all_models_internal.loc["OneVsRest"]
    # multiclass checking
    if _is_multiclass() and not _is_special_model(model):
        logger.info("Target variable is Multiclass. OneVsRestClassifier activated")

        model = onevsrest_model_definition["Class"](
            model, **onevsrest_model_definition["Args"]
        )

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

    total_runtime_start = time.time()
    total_runtime = 0
    over_time_budget = False
    if budget_time and budget_time > 0:
        logger.info(f"Time budget is {budget_time} minutes")

    if not cross_validation:

        logger.info("Cross validation set to False")

        logger.info("Fitting Model")
        model.fit(data_X, data_y)

        display.display("", clear=True)

        logger.info(f"create_model_container: {len(create_model_container)}")
        logger.info(f"master_model_container: {len(master_model_container)}")
        logger.info(f"display_container: {len(display_container)}")

        logger.info(str(model))
        logger.info(
            "create_models() succesfully completed......................................"
        )

        return model

    fold_num = 1

    for train_i, test_i in kf.split(data_X, data_y):

        logger.info(f"Initializing Fold {fold_num}")

        t0 = time.time()
        total_runtime += (t0 - total_runtime_start) / 60
        logger.info(f"Total runtime is {total_runtime} minutes")
        over_time_budget = (
            budget_time and budget_time > 0 and total_runtime > budget_time
        )
        if over_time_budget:
            logger.info(
                f"Total runtime {total_runtime} is over time budget by {total_runtime - budget_time}, terminating function"
            )
            return None
        total_runtime_start = t0

        """
        MONITOR UPDATE STARTS
        """
        display.update_monitor(1, f"Fitting Fold {str(fold_num)} of {str(fold)}")
        display.display_monitor()
        """
        MONITOR UPDATE ENDS
        """

        Xtrain, Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain, ytest = data_y.iloc[train_i], data_y.iloc[test_i]
        # time just for fitting
        time_start = time.time()

        if fix_imbalance_param:
            Xtrain, ytrain = _fix_imbalance(Xtrain, ytrain, fix_imbalance_method_param)

        logger.info("Fitting Model")
        model.fit(Xtrain, ytrain)
        logger.info("Evaluating Metrics")

        if hasattr(model, "predict_proba"):
            pred_prob = model.predict_proba(Xtest)
            pred_prob = pred_prob[:, 1]
        else:
            logger.warning(
                "model has no predict_proba attribute. pred_prob set to 0.00"
            )
            pred_prob = 0.00

        pred_ = model.predict(Xtest)

        _calculate_metrics(ytest, pred_, pred_prob, score_dict)

        logger.info("Compiling Metrics")
        time_end = time.time()
        training_time = time_end - time_start

        if "TT (Sec)" in score_dict:
            score_dict["TT (Sec)"] = np.append(score_dict["TT (Sec)"], training_time)

        display.move_progress()

        """
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        """

        fold_results = pd.DataFrame({k: [v[-1]] for k, v in score_dict.items()}).round(
            round
        )
        display.append_to_master_display(fold_results,)
        fold_results = []

        """
        TIME CALCULATION SUB-SECTION STARTS HERE
        """
        t1 = time.time()

        tt = (t1 - t0) * (fold - fold_num) / 60
        tt = np.around(tt, 2)

        if tt < 1:
            tt = str(np.around((tt * 60), 2))
            ETC = f"{tt} Seconds Remaining"

        else:
            tt = str(tt)
            ETC = f"{tt} Minutes Remaining"

        """
        MONITOR UPDATE STARTS
        """
        display.update_monitor(-1, ETC)
        display.display_monitor()
        """
        MONITOR UPDATE ENDS
        """

        fold_num += 1

        """
        TIME CALCULATION ENDS HERE
        """
        display.display_master_display()

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
    display.update_monitor(2, "Almost Finished")
    display.display_monitor()

    model_fit_start = time.time()
    logger.info("Finalizing model")
    model.fit(data_X, data_y)
    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # mlflow logging
    if logging_param and system:

        logger.info("Creating MLFlow logs")

        # Creating Logs message monitor
        display.update_monitor(1, "Creating Logs")
        display.update_monitor(2, "Almost Finished")
        display.display_monitor()

        # import mlflow
        import mlflow
        import mlflow.sklearn
        from pathlib import Path
        import os

        mlflow.set_experiment(exp_name_log)

        with mlflow.start_run(run_name=full_name) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            # Log model parameters
            params = model.get_params()

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            mlflow.log_params(params)

            # Log metrics
            avgs_dict_log = avgs_dict.copy()
            avgs_dict_log.pop("TT (Sec)")
            mlflow.log_metrics({k: v[0] for k, v in avgs_dict_log.items()})

            # set tag of compare_models
            mlflow.set_tag("Source", "create_model")

            import secrets

            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time in seconds
            mlflow.log_metric("TT", model_fit_time)

            # Log the CV results as model_results.html artifact
            model_results.data.to_html("Results.html", col_space=65, justify="left")
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

            if log_plots_param:

                logger.info(
                    "SubProcess plot_model() called =================================="
                )

                try:
                    plot_model(
                        model, plot="auc", verbose=False, save=True, system=False
                    )
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

            # Log model and transformation pipeline
            from copy import deepcopy

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

            signature = infer_signature(
                data_before_preprocess.drop([target_param], axis=1)
            )
            input_example = (
                data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()
            )

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(["trained model", model])
            mlflow.sklearn.log_model(
                prep_pipe_temp,
                "model",
                conda_env=default_conda_env,
                signature=signature,
                input_example=input_example,
            )
            del prep_pipe_temp

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

    if return_fit_time:
        return (model, model_fit_time)

    return model


def tune_model(
    estimator=None,
    fold: int = 10,
    round: int = 4,
    n_iter: int = 10,
    custom_grid: dict = None,  # added in pycaret==2.0.0
    optimize: str = "Accuracy",
    custom_scorer=None,  # added in pycaret==2.1 - depreciated
    choose_better: bool = False,  # added in pycaret==2.0.0
    verbose: bool = True,
    display: Display = None,
    **kwargs,
):

    """
    This function tunes the hyperparameters of a model and scores it using Stratified 
    Cross Validation. The output prints a score grid that shows Accuracy, AUC, Recall
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

    fold: integer, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    n_iter: integer, default = 10
        Number of iterations within the Random Grid Search. For every iteration, 
        the model randomly selects one value from the pre-defined grid of hyperparameters.

    custom_grid: dictionary, default = None
        To use custom hyperparameters for tuning pass a dictionary with parameter name
        and values to be iterated. When set to None it uses pre-defined tuning grid.  

    optimize: string, default = 'Accuracy'
        Measure used to select the best model through hyperparameter tuning.
        Can be either a string representing a metric or a custom scorer object
        created using sklearn.make_scorer. 

    custom_scorer: object, default = None
        Will be eventually depreciated.
        custom_scorer can be passed to tune hyperparameters of the model. It must be
        created using sklearn.make_scorer. 

    choose_better: Boolean, default = False
        When set to set to True, base estimator is returned when the performance doesn't 
        improve by tune_model. This gurantees the returned object would perform atleast 
        equivalent to base estimator created using create_model or model returned by 
        compare_models.

    verbose: Boolean, default = True
        Score grid is not printed when verbose is set to False.

    Returns
    -------
    score_grid
        A table containing the scores of the model across the kfolds. 
        Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are also returned.

    model
        Trained and tuned model object. 

    Warnings
    --------
   
    - If target variable is multiclass (more than 2 classes), optimize param 'AUC' is 
      not acceptable.
      
    - If target variable is multiclass (more than 2 classes), AUC will be returned as
      zero (0.0)
        
          
    
    """
    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing tune_model()")
    logger.info(f"tune_model({function_params_str})")

    logger.info("Checking exceptions")

    # exception checking
    import sys
    import warnings

    # run_time
    import datetime, time

    runtime_start = time.time()

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

    # restrict VotingClassifier
    if hasattr(estimator, "voting"):
        raise TypeError("VotingClassifier not allowed under tune_model().")

    # checking fold parameter
    if type(fold) is not int:
        raise TypeError("Fold parameter only accepts integer value.")

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    # checking n_iter parameter
    if type(n_iter) is not int:
        raise TypeError("n_iter parameter only accepts integer value.")

    if custom_scorer is not None:
        optimize = custom_scorer
        warnings.warn(
            f"custom_scorer parameter will be depreciated, use optimize instead",
            DeprecationWarning,
            stacklevel=2,
        )

    if isinstance(optimize, str):
        # checking optimize parameter
        optimize = _get_metric(optimize)
        if optimize is None:
            raise ValueError(
                f"Optimize method not supported. See docstring for list of available parameters."
            )

        # checking optimize parameter for multiclass
        if _is_multiclass():
            if not all_metrics[all_metrics["Name"] == optimize].iloc[0]["Multiclass"]:
                raise TypeError(
                    f"Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
                )
    else:
        logger.info(f"optimize set to user defined function {optimize}")

    if type(n_iter) is not int:
        raise TypeError("n_iter parameter only accepts integer value.")

    # checking verbose parameter
    if type(verbose) is not bool:
        raise TypeError("Verbose parameter can only take argument as True or False.")

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    logger.info("Preloading libraries")

    # pre-load libraries
    import pandas as pd

    if not display:
        progress_args = {"max": fold + 3 + 4}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
            ["ETC", ". . . . . . . . . . . . . . . . . .", "Calculating ETC"],
        ]
        display = Display(
            verbose,
            html_param,
            progress_args,
            master_display_columns,
            monitor_rows,
            logger=logger,
        )

        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    # ignore warnings

    warnings.filterwarnings("ignore")

    logger.info("Importing libraries")
    # general dependencies
    import random
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.base import clone

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

    _estimator_ = clone(estimator)
    is_stacked_model = False

    if hasattr(estimator, "final_estimator"):
        logger.info("Model is stacked, using the definition of the meta-model")
        is_stacked_model = True
        estimator_id = _get_model_id(_estimator_.final_estimator)
    elif _is_one_vs_rest(estimator):
        estimator_id = _get_model_id(_estimator_.estimator)
        _estimator_ = _estimator_.estimator
    else:
        estimator_id = _get_model_id(_estimator_)

    estimator_definition = _all_models_internal.loc[estimator_id]
    estimator_name = estimator_definition["Name"]
    logger.info(f"Base model : {estimator_name}")

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
    logger.info("Initializing RandomizedSearchCV")

    if custom_grid is not None:
        param_grid = custom_grid
    else:
        param_grid = estimator_definition["Tune Grid"]

    if not param_grid:
        raise ValueError(
            "parameter grid for tuning is empty. If passing custom_grid, make sure that it is not empty. If not passing custom_grid, the passed estimator does not have a built-in tuning grid."
        )

    if is_stacked_model:
        param_grid = {f"final_estimator__{k}": v for k, v in param_grid.items()}

    search_kwargs = {**estimator_definition["Tune Args"], **kwargs}

    model_grid = RandomizedSearchCV(
        estimator=_estimator_,
        param_distributions=param_grid,
        scoring=optimize,
        n_iter=n_iter,
        cv=fold,
        random_state=seed,
        n_jobs=n_jobs_param,
        **search_kwargs,
    )

    model_grid.fit(X_train, y_train)
    model = model_grid.best_estimator_
    best_model = model_grid.best_estimator_

    display.move_progress()

    logger.info("Random search completed")

    # multiclass checking
    if _is_multiclass() and not is_stacked_model:
        onevsrest_model_definition = _all_models_internal.loc["OneVsRest"]
        model = onevsrest_model_definition["Class"](
            model, **onevsrest_model_definition["Args"]
        )
        best_model = model

    logger.info("SubProcess create_model() called ==================================")
    model, model_fit_time = create_model(
        estimator=model,
        system=False,
        return_fit_time=True,
        display=display,
        fold=fold,
        round=round,
    )
    best_model = model
    model_results = pull()
    logger.info("SubProcess create_model() end ==================================")

    if choose_better:
        model = _choose_better(
            _estimator_,
            [best_model],
            compare_dimension,
            fold,
            new_results_list=[model_results],
            display=display,
        )

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # mlflow logging
    if logging_param:

        logger.info("Creating MLFlow logs")

        # Creating Logs message monitor
        display.update_monitor(1, "Creating Logs")
        display.update_monitor(2, "Almost Finished")
        display.display_monitor()

        import mlflow
        from pathlib import Path
        import os

        mlflow.set_experiment(exp_name_log)
        full_name = estimator_name

        with mlflow.start_run(run_name=full_name) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            # Log model parameters
            params = model.get_params()

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            mlflow.log_params(params)

            mlflow.log_metrics(
                {
                    k: v
                    for k, v in model_results.drop("TT (Sec)", axis=1, errors="ignore")
                    .loc["Mean"]
                    .items()
                }
            )

            # set tag of compare_models
            mlflow.set_tag("Source", "tune_model")

            import secrets

            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time in seconds
            mlflow.log_metric("TT", model_fit_time)

            # Log the CV results as model_results.html artifact
            model_results.to_html("Results.html", col_space=65, justify="left")
            mlflow.log_artifact("Results.html")
            os.remove("Results.html")

            # Generate hold-out predictions and save as html
            holdout = predict_model(best_model, verbose=False)
            holdout_score = pull(pop=True)
            del holdout
            holdout_score.to_html("Holdout.html", col_space=65, justify="left")
            mlflow.log_artifact("Holdout.html")
            os.remove("Holdout.html")

            # Log AUC and Confusion Matrix plot
            if log_plots_param:

                logger.info(
                    "SubProcess plot_model() called =================================="
                )

                try:
                    plot_model(
                        model, plot="auc", verbose=False, save=True, system=False
                    )
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
            d1 = model_grid.cv_results_.get("params")
            dd = pd.DataFrame.from_dict(d1)
            dd["Score"] = model_grid.cv_results_.get("mean_test_score")
            dd.to_html("Iterations.html", col_space=75, justify="left")
            mlflow.log_artifact("Iterations.html")
            os.remove("Iterations.html")

            # Log model and transformation pipeline
            from copy import deepcopy

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

            signature = infer_signature(
                data_before_preprocess.drop([target_param], axis=1)
            )
            input_example = (
                data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()
            )

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(["trained model", model])
            mlflow.sklearn.log_model(
                prep_pipe_temp,
                "model",
                conda_env=default_conda_env,
                signature=signature,
                input_example=input_example,
            )
            del prep_pipe_temp

    model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
    model_results = model_results.set_precision(round)
    display.display(model_results, clear=True)

    logger.info(f"create_model_container: {len(create_model_container)}")
    logger.info(f"master_model_container: {len(master_model_container)}")
    logger.info(f"display_container: {len(display_container)}")

    logger.info(str(best_model))
    logger.info(
        "tune_model() succesfully completed......................................"
    )

    return best_model


def ensemble_model(
    estimator,
    method: str = "Bagging",
    fold: int = 10,
    n_estimators: int = 10,
    round: int = 4,
    choose_better: bool = False,  # added in pycaret==2.0.0
    optimize: str = "Accuracy",  # added in pycaret==2.0.0
    verbose: bool = True,
    display: Display = None,  # added in pycaret==2.2.0
):
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

    method: String, default = 'Bagging'
        Bagging method will create an ensemble meta-estimator that fits base 
        classifiers each on random subsets of the original dataset. The other
        available method is 'Boosting' which will create a meta-estimators by
        fitting a classifier on the original dataset and then fits additional 
        copies of the classifier on the same dataset but where the weights of 
        incorrectly classified instances are adjusted such that subsequent 
        classifiers focus more on difficult cases.
    
    fold: integer, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2.
    
    n_estimators: integer, default = 10
        The number of base estimators in the ensemble.
        In case of perfect fit, the learning procedure is stopped early.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.

    choose_better: Boolean, default = False
        When set to set to True, base estimator is returned when the metric doesn't 
        improve by ensemble_model. This gurantees the returned object would perform 
        atleast equivalent to base estimator created using create_model or model 
        returned by compare_models.

    optimize: string, default = 'Accuracy'
        Only used when choose_better is set to True. optimize parameter is used
        to compare emsembled model with base estimator. Values accepted in 
        optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 
        'Kappa', 'MCC'.

    verbose: Boolean, default = True
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

    # exception checking
    import sys

    # run_time
    import datetime, time

    runtime_start = time.time()

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
            if _is_one_vs_rest(check_model):
                check_model = check_model.estimator
                check_model = boosting_model_definition["Class"](
                    check_model,
                    n_estimators=n_estimators,
                    **boosting_model_definition["Args"],
                )
                onevsrest_model_definition = _all_models_internal.loc["OneVsRest"]
                check_model = onevsrest_model_definition["Class"](
                    model, **onevsrest_model_definition["Args"]
                )

            else:
                check_model = boosting_model_definition["Class"](
                    check_model,
                    n_estimators=n_estimators,
                    **boosting_model_definition["Args"],
                )

            check_model.fit(X_train, y_train)
        except:
            raise TypeError(
                "Estimator does not provide class_weights or predict_proba function and hence not supported for the Boosting method. Change the estimator or method to 'Bagging'."
            )

    # checking fold parameter
    if type(fold) is not int:
        raise TypeError("Fold parameter only accepts integer value.")

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
        if not all_metrics[all_metrics["Name"] == optimize].iloc[0]["Multiclass"]:
            raise TypeError(
                f"Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
            )
        if hasattr(estimator, "estimators") and any(
            _is_one_vs_rest(model) for name, model in estimator.estimators
        ):
            raise TypeError(
                f"Ensembling of VotingClassifier() and StackingClassifier() is not supported for multiclass problems."
            )

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    logger.info("Preloading libraries")

    # pre-load libraries
    import pandas as pd

    if not display:
        progress_args = {"max": fold + 2 + 4}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
            ["ETC", ". . . . . . . . . . . . . . . . . .", "Calculating ETC"],
        ]
        display = Display(
            verbose,
            html_param,
            progress_args,
            master_display_columns,
            monitor_rows,
            logger=logger,
        )

        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    logger.info("Importing libraries")

    # dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold

    np.random.seed(seed)

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

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

    if _is_one_vs_rest(estimator):
        estimator_id = _get_model_id(estimator.estimator)
    else:
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
    if _is_one_vs_rest(model):
        model = model.estimator

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

    onevsrest_model_definition = _all_models_internal.loc["OneVsRest"]
    # multiclass checking
    if _is_multiclass() and not _is_special_model(model):
        logger.info("Target variable is Multiclass. OneVsRestClassifier activated")

        model = onevsrest_model_definition["Class"](
            model, **onevsrest_model_definition["Args"]
        )

    display.move_progress()

    logger.info("SubProcess create_model() called ==================================")
    model, model_fit_time = create_model(
        estimator=model,
        system=False,
        return_fit_time=True,
        display=display,
        fold=fold,
        round=round,
    )
    best_model = model
    model_results = pull()
    logger.info("SubProcess create_model() end ==================================")

    if choose_better:
        model = _choose_better(
            _estimator_,
            [best_model],
            compare_dimension,
            fold,
            new_results_list=[model_results],
            display=display,
        )

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    if logging_param:

        logger.info("Creating MLFlow logs")

        # Creating Logs message monitor
        display.update_monitor(1, "Creating Logs")
        display.update_monitor(2, "Almost Finished")
        display.display_monitor()

        import mlflow
        from pathlib import Path
        import os

        mlflow.set_experiment(exp_name_log)
        full_name = estimator_name

        with mlflow.start_run(run_name=full_name) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            params = model.get_params()

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    k: v
                    for k, v in model_results.drop("TT (Sec)", axis=1, errors="ignore")
                    .loc["Mean"]
                    .items()
                }
            )

            # set tag of compare_models
            mlflow.set_tag("Source", "ensemble_model")

            import secrets

            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time in seconds
            mlflow.log_metric("TT", model_fit_time)

            # Generate hold-out predictions and save as html
            holdout = predict_model(model, verbose=False)
            holdout_score = pull(pop=True)
            del holdout
            holdout_score.to_html("Holdout.html", col_space=65, justify="left")
            mlflow.log_artifact("Holdout.html")
            os.remove("Holdout.html")

            # Log AUC and Confusion Matrix plot
            if log_plots_param:

                logger.info(
                    "SubProcess plot_model() called =================================="
                )

                try:
                    plot_model(
                        model, plot="auc", verbose=False, save=True, system=False
                    )
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

            # Log the CV results as model_results.html artifact
            model_results.to_html("Results.html", col_space=65, justify="left")
            mlflow.log_artifact("Results.html")
            os.remove("Results.html")

            # Log model and transformation pipeline
            from copy import deepcopy

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

            signature = infer_signature(
                data_before_preprocess.drop([target_param], axis=1)
            )
            input_example = (
                data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()
            )

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(["trained model", model])
            mlflow.sklearn.log_model(
                prep_pipe_temp,
                "model",
                conda_env=default_conda_env,
                signature=signature,
                input_example=input_example,
            )
            del prep_pipe_temp

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

    return model


def blend_models(
    estimator_list="All",
    fold: int = 10,
    round: int = 4,
    choose_better: bool = False,  # added in pycaret==2.0.0
    optimize: str = "Accuracy",  # added in pycaret==2.0.0
    method: str = "hard",
    weights: list = None,  # added in pycaret==2.2.0
    turbo: bool = True,
    verbose: bool = True,
    display: Display = None,  # added in pycaret==2.2.0
):

    """
    This function creates a Soft Voting / Majority Rule classifier for all the 
    estimators in the model library (excluding the few when turbo is True) or 
    for specific trained estimators passed as a list in estimator_list param.
    It scores it using Stratified Cross Validation. The output prints a score
    grid that shows Accuracy,  AUC, Recall, Precision, F1, Kappa and MCC by 
    fold (default CV = 10 Folds). 

    This function returns a trained model object.  

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> blend_all = blend_models() 

    This will create a VotingClassifier for all models in the model library 
    except for 'rbfsvm', 'gpc' and 'mlp'.

    For specific models, you can use:

    >>> lr = create_model('lr')
    >>> rf = create_model('rf')
    >>> knn = create_model('knn')
    >>> blend_three = blend_models(estimator_list = [lr,rf,knn])

    This will create a VotingClassifier of lr, rf and knn.

    Parameters
    ----------
    estimator_list : string ('All') or list of object, default = 'All'

    fold: integer, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.

    choose_better: Boolean, default = False
        When set to set to True, base estimator is returned when the metric doesn't 
        improve by ensemble_model. This gurantees the returned object would perform 
        atleast equivalent to base estimator created using create_model or model 
        returned by compare_models.

    optimize: string, default = 'Accuracy'
        Only used when choose_better is set to True. optimize parameter is used
        to compare emsembled model with base estimator. Values accepted in 
        optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 
        'Kappa', 'MCC'.

    method: string, default = 'hard'
        'hard' uses predicted class labels for majority rule voting.'soft', predicts 
        the class label based on the argmax of the sums of the predicted probabilities, 
        which is recommended for an ensemble of well-calibrated classifiers. 

    weights: list, default = None
        Sequence of weights (float or int) to weight the occurrences of predicted class labels (hard voting)
        or class probabilities before averaging (soft voting). Uses uniform weights if None.

    turbo: Boolean, default = True
        When turbo is set to True, it excludes estimator that uses Radial Kernel.

    verbose: Boolean, default = True
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
      
    - CatBoost Classifier not supported in blend_models().
    
    - If target variable is multiclass (more than 2 classes), AUC will be returned as
      zero (0.0).
        
       
  
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing blend_models()")
    logger.info(f"blend_models({function_params_str})")

    logger.info("Checking exceptions")

    # exception checking
    import sys

    # run_time
    import datetime, time

    runtime_start = time.time()

    # checking error for estimator_list (string)

    if estimator_list != "All":
        if type(estimator_list) is not list:
            raise ValueError(
                "estimator_list parameter only accepts 'All' as string or list of trained models."
            )

        for i in estimator_list:
            if not hasattr(i, "fit"):
                raise ValueError(
                    f"Estimator {i} does not have the required fit() method."
                )

        # checking method param with estimator list
        if method == "soft":

            check = 0

            for i in estimator_list:
                if not hasattr(i, "predict_proba"):
                    raise TypeError(
                        "Estimator list contains estimator that doesnt support probabilities and method is forced to soft. Either change the method or drop the estimator."
                    )

    # checking fold parameter
    if type(fold) is not int:
        raise TypeError("Fold parameter only accepts integer value.")

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    # checking method parameter
    available_method = ["soft", "hard"]
    if method not in available_method:
        raise ValueError(
            "Method parameter only accepts 'soft' or 'hard' as a parameter. See Docstring for details."
        )

    # checking verbose parameter
    if type(turbo) is not bool:
        raise TypeError("Turbo parameter can only take argument as True or False.")

    if weights is not None:
        if isinstance(estimator_list, list):
            num_estimators = len(estimator_list)
        else:
            num_estimators = models(internal=True)
            num_estimators = num_estimators[num_estimators["Special"] == False]
            if turbo:
                num_estimators = num_estimators[num_estimators["Turbo"] == True]
            num_estimators = len(num_estimators)
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
        if not all_metrics[all_metrics["Name"] == optimize].iloc[0]["Multiclass"]:
            raise TypeError(
                f"Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
            )

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    logger.info("Preloading libraries")
    # pre-load libraries
    import pandas as pd

    # estimator_list_flag
    all_flag = estimator_list == "All"

    if not display:
        all_models_offset = len(models()) if all_flag else 0
        progress_args = {"max": fold + 2 + 4 + all_models_offset}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
            ["ETC", ". . . . . . . . . . . . . . . . . .", "Calculating ETC"],
        ]
        display = Display(
            verbose,
            html_param,
            progress_args,
            master_display_columns,
            monitor_rows,
            logger=logger,
        )
        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    logger.info("Importing libraries")
    # general dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.ensemble import VotingClassifier
    import re

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

    if all_flag:
        models_to_check = models(internal=True)
        models_to_check = models_to_check[models_to_check["Special"] == False]
        if method == "soft":
            models_to_check = models_to_check[models_to_check["Soft Voting"] == True]
        if turbo:
            models_to_check = models_to_check[models_to_check["Turbo"] == True]
        estimator_list_str = models_to_check.index.to_list()
        estimator_list = []
        for model in estimator_list_str:
            model_name = _get_model_name(model)
            logger.info(
                "SubProcess create_model() called =================================="
            )
            model = create_model(
                estimator=model, system=False, verbose=False, fold=fold, round=round,
            )
            # re-instate display_constainer state
            pull(pop=True)
            logger.info(
                "SubProcess create_model() end =================================="
            )
            estimator_list.append((model_name, model))
            display.move_progress()
    else:
        model_names = []
        model_names_counter = {}
        for x in estimator_list:
            if _is_one_vs_rest(x):
                name = _get_model_name(x.estimator)
            else:
                name = _get_model_name(x)
            if name in model_names_counter:
                model_names_counter[name] += 1
                name += f"_{model_names_counter[name]-1}"
            else:
                model_names_counter[name] = 1
            model_names.append(name)

        estimator_list = list(zip(model_names, estimator_list))

    # if _is_multiclass():
    #    estimator_list = [x.estimator for x in estimator_list]

    votingclassifier_model_definition = _all_models_internal.loc["Voting"]
    try:
        model = votingclassifier_model_definition["Class"](
            estimators=estimator_list, voting=method, n_jobs=n_jobs_param
        )
        logger.info("n_jobs multiple passed")
    except:
        logger.info("n_jobs multiple failed")
        model = votingclassifier_model_definition["Class"](
            estimators=estimator_list, voting=method, weights=weights
        )

    display.move_progress()

    logger.info("SubProcess create_model() called ==================================")
    model, model_fit_time = create_model(
        estimator=model,
        system=False,
        return_fit_time=True,
        display=display,
        fold=fold,
        round=round,
    )
    model_results = pull()
    logger.info("SubProcess create_model() end ==================================")

    if choose_better and not all_flag:
        model = _choose_better(
            model,
            estimator_list,
            compare_dimension,
            fold,
            model_results=model_results,
            display=display,
        )

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    if logging_param:

        logger.info("Creating MLFlow logs")

        # Creating Logs message monitor
        display.update_monitor(1, "Creating Logs")
        display.update_monitor(2, "Almost Finished")
        display.display_monitor()

        import mlflow
        from pathlib import Path
        import os

        with mlflow.start_run(run_name="Voting Classifier") as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            mlflow.log_metrics(
                {
                    k: v
                    for k, v in model_results.drop("TT (Sec)", axis=1, errors="ignore")
                    .loc["Mean"]
                    .items()
                }
            )

            # Generate hold-out predictions and save as html
            holdout = predict_model(model, verbose=False)
            holdout_score = pull(pop=True)
            del holdout
            holdout_score.to_html("Holdout.html", col_space=65, justify="left")
            mlflow.log_artifact("Holdout.html")
            os.remove("Holdout.html")

            # set tag of compare_models
            mlflow.set_tag("Source", "blend_models")

            import secrets

            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time of compare_models
            mlflow.log_metric("TT", model_fit_time)

            # Log AUC and Confusion Matrix plot
            if log_plots_param:

                logger.info(
                    "SubProcess plot_model() called =================================="
                )

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

                logger.info(
                    "SubProcess plot_model() end =================================="
                )

            # Log the CV results as model_results.html artifact
            model_results.to_html("Results.html", col_space=65, justify="left")
            mlflow.log_artifact("Results.html")
            os.remove("Results.html")

            # Log model and transformation pipeline
            from copy import deepcopy

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

            signature = infer_signature(
                data_before_preprocess.drop([target_param], axis=1)
            )
            input_example = (
                data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()
            )

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(["trained model", model])
            mlflow.sklearn.log_model(
                prep_pipe_temp,
                "model",
                conda_env=default_conda_env,
                signature=signature,
                input_example=input_example,
            )
            del prep_pipe_temp

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

    return model


def stack_models(
    estimator_list: list,
    meta_model=None,
    fold: int = 10,
    round: int = 4,
    method: str = "auto",
    restack: bool = True,
    choose_better: bool = False,  # added in pycaret==2.0.0
    optimize: str = "Accuracy",  # added in pycaret==2.0.0
    verbose: bool = True,
    display: Display = None,
):

    """
    This function trains a meta model and scores it using Stratified Cross Validation.
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

    fold: integer, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.

    method: string, default = 'auto'
        - if ‘auto’, it will try to invoke, for each estimator, 'predict_proba', 'decision_function' or 'predict' in that order.
        - otherwise, one of 'predict_proba', 'decision_function' or 'predict'. If the method is not implemented by the estimator, it will raise an error.

    restack: Boolean, default = True
        When restack is set to True, raw data will be exposed to meta model when
        making predictions, otherwise when False, only the predicted label or
        probabilities is passed to meta model when making final predictions.

    choose_better: Boolean, default = False
        When set to set to True, base estimator is returned when the metric doesn't 
        improve by ensemble_model. This gurantees the returned object would perform 
        atleast equivalent to base estimator created using create_model or model 
        returned by compare_models.

    optimize: string, default = 'Accuracy'
        Only used when choose_better is set to True. optimize parameter is used
        to compare emsembled model with base estimator. Values accepted in 
        optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 
        'Kappa', 'MCC'.
    
    verbose: Boolean, default = True
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
    -  If target variable is multiclass (more than 2 classes), AUC will be returned as zero (0.0).

    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing stack_models()")
    logger.info(f"stack_models({function_params_str})")

    logger.info("Checking exceptions")

    # exception checking
    import sys

    # run_time
    import datetime, time

    runtime_start = time.time()

    # checking error for estimator_list
    for i in estimator_list:
        if not hasattr(i, "fit"):
            raise ValueError(f"Estimator {i} does not have the required fit() method.")

    # checking meta model
    if meta_model is not None:
        if not hasattr(meta_model, "fit"):
            raise ValueError(f"Meta Model {i} does not have the required fit() method.")

    # checking fold parameter
    if type(fold) is not int:
        raise TypeError("Fold parameter only accepts integer value.")

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
        if not all_metrics[all_metrics["Name"] == optimize].iloc[0]["Multiclass"]:
            raise TypeError(
                f"Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
            )

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    logger.info("Preloading libraries")
    # pre-load libraries
    import pandas as pd
    from copy import deepcopy
    from sklearn.base import clone
    from sklearn.ensemble import StackingClassifier

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

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
        progress_args = {"max": fold + 2 + 4}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
            ["ETC", ". . . . . . . . . . . . . . . . . .", "Calculating ETC"],
        ]
        display = Display(
            verbose,
            html_param,
            progress_args,
            master_display_columns,
            monitor_rows,
            logger=logger,
        )
        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    logger.info("Importing libraries")
    # dependencies
    import numpy as np
    from sklearn import metrics

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
    model_names = []
    model_names_counter = {}
    for x in estimator_list:
        if _is_one_vs_rest(x):
            name = _get_model_name(x.estimator)
        else:
            name = _get_model_name(x)
        if name in model_names_counter:
            model_names_counter[name] += 1
            name += f"_{model_names_counter[name]-1}"
        else:
            model_names_counter[name] = 1
        model_names.append(name)

    estimator_list = list(zip(model_names, estimator_list))

    logger.info("Creating StackingClassifier()")

    stackingclassifier_model_definition = _all_models_internal.loc["Stacking"]
    model = stackingclassifier_model_definition["Class"](
        estimators=estimator_list,
        final_estimator=meta_model,
        cv=fold,
        stack_method=method,
        n_jobs=n_jobs_param,
        passthrough=restack,
    )

    display.move_progress()

    logger.info("SubProcess create_model() called ==================================")
    model, model_fit_time = create_model(
        estimator=model,
        system=False,
        return_fit_time=True,
        display=display,
        fold=fold,
        round=round,
    )
    model_results = pull()
    logger.info("SubProcess create_model() end ==================================")

    if choose_better:
        model = _choose_better(
            model,
            estimator_list,
            compare_dimension,
            fold,
            model_results=model_results,
            display=display,
        )

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    if logging_param:

        logger.info("Creating MLFlow logs")

        import mlflow
        from pathlib import Path
        import os

        # Creating Logs message monitor
        display.update_monitor(1, "Creating Logs")
        display.update_monitor(2, "Almost Finished")
        display.display_monitor()

        with mlflow.start_run(run_name="Stacking Classifier") as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            params = model.get_params()

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            try:
                mlflow.log_params(params)
            except:
                pass

            mlflow.log_metrics(
                {
                    k: v
                    for k, v in model_results.drop("TT (Sec)", axis=1, errors="ignore")
                    .loc["Mean"]
                    .items()
                }
            )

            # set tag of stack_models
            mlflow.set_tag("Source", "stack_models")

            import secrets

            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time of compare_models
            mlflow.log_metric("TT", model_fit_time)

            # Log the CV results as model_results.html artifact
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
            if log_plots_param:

                logger.info(
                    "SubProcess plot_model() called =================================="
                )

                try:
                    plot_model(
                        model, plot="auc", verbose=False, save=True, system=False
                    )
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

            # Log model and transformation pipeline
            from copy import deepcopy

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

            signature = infer_signature(
                data_before_preprocess.drop([target_param], axis=1)
            )
            input_example = (
                data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()
            )

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(["trained model", model])
            mlflow.sklearn.log_model(
                prep_pipe_temp,
                "model",
                conda_env=default_conda_env,
                signature=signature,
                input_example=input_example,
            )
            del prep_pipe_temp

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

    return model


def plot_model(
    estimator,
    plot: str = "auc",
    scale=1,  # added in pycaret 2.1.0
    save: bool = False,  # added in pycaret 2.0.0
    verbose: bool = True,  # added in pycaret 2.0.0
    system: bool = True,
    display: Display = None,  # added in pycaret==2.2.0
):  # added in pycaret 2.0.0

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

    plot : string, default = auc
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

    scale: float, default = 1
        The resolution scale of the figure.

    save: Boolean, default = False
        When set to True, Plot is saved as a 'png' file in current working directory.

    verbose: Boolean, default = True
        Progress bar not shown when verbose set to False. 

    system: Boolean, default = True
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

    # exception checking
    import sys

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
    ]
    available_plots = {k: v for v, k in available_plots}
    print(available_plots)

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

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    if not display:
        progress_args = {"max": 5}
        display = Display(
            verbose, html_param, progress_args, None, None, logger=logger,
        )
        display.display_progress()

    logger.info("Preloading libraries")
    # pre-load libraries
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    np.random.seed(seed)

    display.move_progress()

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    # defining estimator as model locally
    model = estimator

    display.move_progress()

    # plots used for logging (controlled through plots_log_param)
    # AUC, #Confusion Matrix and #Feature Importance

    logger.info(f"Plot type: {plot}")
    plot_name = available_plots[plot]
    display.move_progress()

    if plot == "auc":

        from yellowbrick.classifier import ROCAUC

        visualizer = ROCAUC(model)
        show_yellowbrick_plot(
            visualizer=visualizer,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            name=plot_name,
            scale=scale,
            save=save,
            system=system,
            logger=logger,
            display=display,
        )

    elif plot == "threshold":

        from yellowbrick.classifier import DiscriminationThreshold

        visualizer = DiscriminationThreshold(model, random_state=seed)
        show_yellowbrick_plot(
            visualizer=visualizer,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            name=plot_name,
            scale=scale,
            save=save,
            system=system,
            logger=logger,
            display=display,
        )

    elif plot == "pr":

        from yellowbrick.classifier import PrecisionRecallCurve

        visualizer = PrecisionRecallCurve(model, random_state=seed)
        show_yellowbrick_plot(
            visualizer=visualizer,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            name=plot_name,
            scale=scale,
            save=save,
            system=system,
            logger=logger,
            display=display,
        )

    elif plot == "confusion_matrix":

        from yellowbrick.classifier import ConfusionMatrix

        visualizer = ConfusionMatrix(
            model, random_state=seed, fontsize=15, cmap="Greens"
        )
        show_yellowbrick_plot(
            visualizer=visualizer,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            name=plot_name,
            scale=scale,
            save=save,
            system=system,
            logger=logger,
            display=display,
        )

    elif plot == "error":

        from yellowbrick.classifier import ClassPredictionError

        visualizer = ClassPredictionError(model, random_state=seed)
        show_yellowbrick_plot(
            visualizer=visualizer,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            name=plot_name,
            scale=scale,
            save=save,
            system=system,
            logger=logger,
            display=display,
        )

    elif plot == "class_report":

        from yellowbrick.classifier import ClassificationReport

        visualizer = ClassificationReport(model, random_state=seed, support=True)
        show_yellowbrick_plot(
            visualizer=visualizer,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            name=plot_name,
            scale=scale,
            save=save,
            system=system,
            logger=logger,
            display=display,
        )

    elif plot == "boundary":

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from yellowbrick.contrib.classifier import DecisionViz
        from copy import deepcopy

        model2 = deepcopy(estimator)

        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()
        X_train_transformed = X_train_transformed.select_dtypes(include="float64")
        X_test_transformed = X_test_transformed.select_dtypes(include="float64")
        logger.info("Fitting StandardScaler()")
        X_train_transformed = StandardScaler().fit_transform(X_train_transformed)
        X_test_transformed = StandardScaler().fit_transform(X_test_transformed)
        pca = PCA(n_components=2, random_state=seed)
        logger.info("Fitting PCA()")
        X_train_transformed = pca.fit_transform(X_train_transformed)
        X_test_transformed = pca.fit_transform(X_test_transformed)

        y_train_transformed = y_train.copy()
        y_test_transformed = y_test.copy()
        y_train_transformed = np.array(y_train_transformed)
        y_test_transformed = np.array(y_test_transformed)

        viz_ = DecisionViz(model2)
        show_yellowbrick_plot(
            visualizer=viz_,
            X_train=X_train_transformed,
            y_train=y_train_transformed,
            X_test=X_test_transformed,
            y_test=y_test_transformed,
            name=plot_name,
            scale=scale,
            handle_train="draw",
            save=save,
            system=system,
            logger=logger,
            display=display,
            features=["Feature One", "Feature Two"],
            classes=["A", "B"],
        )

    elif plot == "rfe":

        from yellowbrick.model_selection import RFECV

        visualizer = RFECV(model, cv=10)
        show_yellowbrick_plot(
            visualizer=visualizer,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            handle_test="",
            name=plot_name,
            scale=scale,
            save=save,
            system=system,
            logger=logger,
            display=display,
        )

    elif plot == "learning":

        from yellowbrick.model_selection import LearningCurve

        sizes = np.linspace(0.3, 1.0, 10)
        visualizer = LearningCurve(
            model, cv=10, train_sizes=sizes, n_jobs=n_jobs_param, random_state=seed
        )
        show_yellowbrick_plot(
            visualizer=visualizer,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            handle_test="",
            name=plot_name,
            scale=scale,
            save=save,
            system=system,
            logger=logger,
            display=display,
        )

    elif plot == "manifold":

        from yellowbrick.features import Manifold

        X_train_transformed = X_train.select_dtypes(include="float64")
        visualizer = Manifold(manifold="tsne", random_state=seed)
        show_yellowbrick_plot(
            visualizer=visualizer,
            X_train=X_train_transformed,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            handle_train="fit_transform",
            handle_test="",
            name=plot_name,
            scale=scale,
            save=save,
            system=system,
            logger=logger,
            display=display,
        )

    elif plot == "calibration":

        from sklearn.calibration import calibration_curve

        model_name = str(model).split("(")[0]

        plt.figure(figsize=(7, 6), dpi=100 * scale)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        display.move_progress()
        logger.info("Scoring test/hold-out set")
        prob_pos = model.predict_proba(X_test)[:, 1]
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob_pos, n_bins=10
        )
        display.move_progress()
        ax1.plot(
            mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name}",
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

        model_name = str(model).split("(")[0]

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
        elif hasattr(model, "max_depth"):
            param_name = "max_depth"
            param_range = np.arange(1, 11)

        # knn
        elif hasattr(model, "n_neighbors"):
            param_name = "n_neighbors"
            param_range = np.arange(1, 11)

        # MLP / Ridge
        elif hasattr(model, "alpha"):
            param_name = "alpha"
            param_range = np.arange(0, 1, 0.1)

        # Logistic Regression
        elif hasattr(model, "C"):
            param_name = "C"
            param_range = np.arange(1, 11)

        # Bagging / Boosting
        elif hasattr(model, "n_estimators"):
            param_name = "n_estimators"
            param_range = np.arange(1, 100, 10)

        # Bagging / Boosting / gbc / ada /
        elif hasattr(model, "n_estimators"):
            param_name = "n_estimators"
            param_range = np.arange(1, 100, 10)

        # Naive Bayes
        elif hasattr(model, "var_smoothing"):
            param_name = "var_smoothing"
            param_range = np.arange(0.1, 1, 0.01)

        # QDA
        elif hasattr(model, "reg_param"):
            param_name = "reg_param"
            param_range = np.arange(0, 1, 0.1)

        # GPC
        elif hasattr(model, "max_iter_predict"):
            param_name = "max_iter_predict"
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
            model,
            param_name=param_name,
            param_range=param_range,
            cv=10,
            random_state=seed,
        )
        show_yellowbrick_plot(
            visualizer=viz,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            handle_train="fit",
            handle_test="",
            name=plot_name,
            scale=scale,
            save=save,
            system=system,
            logger=logger,
            display=display,
        )

    elif plot == "dimension":

        from yellowbrick.features import RadViz
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        X_train_transformed = X_train.select_dtypes(include="float64")
        logger.info("Fitting StandardScaler()")
        X_train_transformed = StandardScaler().fit_transform(X_train_transformed)
        y_train_transformed = np.array(y_train)

        features = min(round(len(X_train.columns) * 0.3, 0), 5)
        features = int(features)

        pca = PCA(n_components=features, random_state=seed)
        logger.info("Fitting PCA()")
        X_train_transformed = pca.fit_transform(X_train_transformed)
        display.move_progress()
        classes = y_train.unique().tolist()
        visualizer = RadViz(classes=classes, alpha=0.25)

        show_yellowbrick_plot(
            visualizer=visualizer,
            X_train=X_train_transformed,
            y_train=y_train_transformed,
            X_test=X_test,
            y_test=y_test,
            handle_train="fit_transform",
            handle_test="",
            name=plot_name,
            scale=scale,
            save=save,
            system=system,
            logger=logger,
            display=display,
        )

    elif plot == "feature":

        if hasattr(estimator, "coef_"):
            variables = abs(model.coef_[0])
        else:
            logger.warning("No coef_ found. Trying feature_importances_")
            variables = abs(model.feature_importances_)
        coef_df = pd.DataFrame({"Variable": X_train.columns, "Value": variables})
        sorted_df = (
            coef_df.sort_values(by="Value", ascending=False)
            .head(10)
            .sort_values(by="Value")
        )
        my_range = range(1, len(sorted_df.index) + 1)
        display.move_progress()
        plt.figure(figsize=(8, 5), dpi=100 * scale)
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

    logger.info(
        "plot_model() succesfully completed......................................"
    )


def evaluate_model(estimator):

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

    Returns
    -------
    User_Interface
        Displays the user interface for plotting.

    """

    from ipywidgets import widgets
    from ipywidgets.widgets import interact, fixed, interact_manual

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
        ],
        description="Plot Type:",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        icons=[""],
    )

    d = interact(
        plot_model,
        estimator=fixed(estimator),
        plot=a,
        save=fixed(False),
        verbose=fixed(True),
        system=fixed(True),
    )


def interpret_model(
    estimator,
    plot: str = "summary",
    feature: str = None,
    observation: int = None,
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

    plot : string, default = 'summary'
        Other available options are 'correlation' and 'reason'.

    feature: string, default = None
        This parameter is only needed when plot = 'correlation'. By default feature is 
        set to None which means the first column of the dataset will be used as a variable. 
        A feature parameter must be passed to change this.

    observation: integer, default = None
        This parameter only comes into effect when plot is set to 'reason'. If no observation
        number is provided, it will return an analysis of all observations with the option
        to select the feature on x and y axes through drop down interactivity. For analysis at
        the sample level, an observation parameter must be passed with the index value of the
        observation in test / hold-out set. 

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

    import sys

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
    import numpy as np
    import pandas as pd

    np.random.seed(seed)

    # storing estimator in model variable
    model = estimator

    # defining type of classifier
    shap_models_type1 = set(shap_models[shap_models["SHAP"] == "type1"].index)
    shap_models_type2 = set(shap_models[shap_models["SHAP"] == "type2"].index)

    logger.info(f"plot type: {plot}")

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
            shap_plot = shap.dependence_plot(
                dependence, shap_values[1], X_test, **kwargs
            )
        elif model_id in shap_models_type2:
            logger.info("model type detected: type 2")
            shap_plot = shap.dependence_plot(dependence, shap_values, X_test, **kwargs)

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

    return shap_plot


def calibrate_model(
    estimator,
    method: str = "sigmoid",
    fold: int = 10,
    round: int = 4,
    verbose: bool = True,
    display: Display = None,  # added in pycaret==2.2.0
):

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
    
    method : string, default = 'sigmoid'
        The method to use for calibration. Can be 'sigmoid' which corresponds to Platt's 
        method or 'isotonic' which is a non-parametric approach. It is not advised to use
        isotonic calibration with too few calibration samples

    fold: integer, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    verbose: Boolean, default = True
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

    # exception checking
    import sys

    # run_time
    import datetime, time

    runtime_start = time.time()

    # checking fold parameter
    if type(fold) is not int:
        raise TypeError("Fold parameter only accepts integer value.")

    # checking round parameter
    if type(round) is not int:
        raise TypeError("Round parameter only accepts integer value.")

    # checking verbose parameter
    if type(verbose) is not bool:
        raise TypeError("Verbose parameter can only take argument as True or False.")

    """
    
    ERROR HANDLING ENDS HERE
    
    """

    logger.info("Preloading libraries")

    # pre-load libraries
    import pandas as pd

    logger.info("Preparing display monitor")

    if not display:
        progress_args = {"max": fold + 3 + 4}
        master_display_columns = all_metrics["Display Name"].to_list()
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
            ["ETC", ". . . . . . . . . . . . . . . . . .", "Calculating ETC"],
        ]
        display = Display(
            verbose,
            html_param,
            progress_args,
            master_display_columns,
            monitor_rows,
            logger=logger,
        )

        display.display_progress()
        display.display_monitor()
        display.display_master_display()

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    logger.info("Importing libraries")
    # general dependencies
    import numpy as np
    from sklearn import metrics

    np.random.seed(seed)

    logger.info("Copying training dataset")
    # Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()

    # reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)

    display.move_progress()

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

    calibrated_model_definition = _all_models_internal.loc["Calibrated"]
    model = calibrated_model_definition["Class"](
        base_estimator=estimator,
        method=method,
        cv=fold,
        **calibrated_model_definition["Args"],
    )

    display.move_progress()

    logger.info("SubProcess create_model() called ==================================")
    model, model_fit_time = create_model(
        estimator=model,
        system=False,
        return_fit_time=True,
        display=display,
        fold=fold,
        round=round,
    )
    model_results = pull()
    logger.info("SubProcess create_model() end ==================================")

    model_results = model_results.round(round)

    # refitting the model on complete X_train, y_train
    display.update_monitor(1, "Compiling Final Model")
    display.display_monitor()

    model_fit_start = time.time()
    logger.info("Finalizing model")
    model.fit(data_X, data_y)
    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

    display.move_progress()

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # storing results in create_model_container
    logger.info("Uploading results into container")
    create_model_container.append(model_results)
    display_container.append(model_results)

    # storing results in master_model_container
    logger.info("Uploading model into container")
    master_model_container.append(model)

    # mlflow logging
    if logging_param:

        logger.info("Creating MLFlow logs")

        # Creating Logs message monitor
        display.update_monitor(1, "Creating Logs")
        display.update_monitor(2, "Almost Finished")
        display.display_monitor()

        # import mlflow
        import mlflow
        import mlflow.sklearn
        from pathlib import Path
        import os

        mlflow.set_experiment(exp_name_log)

        with mlflow.start_run(run_name=full_name) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            # Log model parameters
            params = model.get_params()

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(
                {
                    k: v
                    for k, v in model_results.drop("TT (Sec)", axis=1, errors="ignore")
                    .loc["Mean"]
                    .items()
                }
            )

            # set tag of compare_models
            mlflow.set_tag("Source", "calibrate_model")

            import secrets

            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time in seconds
            mlflow.log_metric("TT", model_fit_time)

            # Log the CV results as model_results.html artifact
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
            if log_plots_param:

                logger.info(
                    "SubProcess plot_model() called =================================="
                )

                try:
                    plot_model(
                        model, plot="auc", verbose=False, save=True, system=False
                    )
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

            # Log model and transformation pipeline
            from copy import deepcopy

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

            signature = infer_signature(
                data_before_preprocess.drop([target_param], axis=1)
            )
            input_example = (
                data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()
            )

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(["trained model", model])
            mlflow.sklearn.log_model(
                prep_pipe_temp,
                "model",
                conda_env=default_conda_env,
                signature=signature,
                input_example=input_example,
            )
            del prep_pipe_temp

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
    import sys
    import pandas as pd
    import numpy as np
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
            "optimize_threshold() cannot be used when target is multi-class. "
        )

    if _is_one_vs_rest(estimator):
        raise TypeError(
            "optimize_threshold() cannot be used when target is multi-class. "
        )

    # check predict_proba value
    if type(estimator) is not list:
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(
                "Estimator doesn't support predict_proba function and cannot be used in optimize_threshold().  "
            )

    # check cost function type
    allowed_types = [int, float]

    if type(true_positive) not in allowed_types:
        raise TypeError("true_positive parameter only accepts float or integer value. ")

    if type(true_negative) not in allowed_types:
        raise TypeError("true_negative parameter only accepts float or integer value. ")

    if type(false_positive) not in allowed_types:
        raise TypeError(
            "false_positive parameter only accepts float or integer value. "
        )

    if type(false_negative) not in allowed_types:
        raise TypeError(
            "false_negative parameter only accepts float or integer value. "
        )

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
    data: pandas.DataFrame = None,
    probability_threshold: float = None,
    round: int = 4,  # added in pycaret==2.2.0
    verbose: bool = True,
    display: Display = None,  # added in pycaret==2.2.0
):  # added in pycaret==2.0.0

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
        Shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
        All features used during training must be present in the new dataset.
    
    probability_threshold : float, default = None
        Threshold used to convert probability values into binary outcome. By default the
        probability threshold for all binary classifiers is 0.5 (50%). This can be changed
        using probability_threshold param.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    verbose: Boolean, default = True
        Holdout score grid is not printed when verbose is set to False.

    Returns
    -------
    Predictions
        Predictions (Label and Score) column attached to the original dataset
        and returned as pandas dataframe.

    score_grid
        A table containing the scoring metrics on hold-out / test set.
    
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing predict_model()")
    logger.info(f"predict_model({function_params_str})")

    logger.info("Checking exceptions")

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    """
    exception checking starts here
    """

    if probability_threshold is not None:
        if _is_one_vs_rest(estimator):
            raise TypeError(
                "probability_threshold parameter cannot be used when target is multi-class. "
            )

        # probability_threshold allowed types
        allowed_types = [int, float]
        if type(probability_threshold) not in allowed_types:
            raise TypeError(
                "probability_threshold parameter only accepts value between 0 to 1. "
            )

        if probability_threshold > 1:
            raise TypeError(
                "probability_threshold parameter only accepts value between 0 to 1. "
            )

        if probability_threshold < 0:
            raise TypeError(
                "probability_threshold parameter only accepts value between 0 to 1. "
            )

    """
    exception checking ends here
    """

    logger.info("Preloading libraries")

    # general dependencies
    import sys
    import numpy as np
    import pandas as pd
    import re
    from sklearn import metrics
    from copy import deepcopy

    np.random.seed(seed)

    if not display:
        display = Display(verbose, html_param, logger=logger,)

    # dataset
    if data is None:

        if "Pipeline" in str(type(estimator)):
            estimator = estimator[-1]

        Xtest = X_test.copy()
        ytest = y_test.copy()
        X_test_ = X_test.copy()
        y_test_ = y_test.copy()

        _, dtypes = next(step for step in prep_pipe.steps if step[0] == "dtypes")

        Xtest.reset_index(drop=True, inplace=True)
        ytest.reset_index(drop=True, inplace=True)
        X_test_.reset_index(drop=True, inplace=True)
        y_test_.reset_index(drop=True, inplace=True)

    else:

        if "Pipeline" in str(type(estimator)):
            _, dtypes = next(step for step in estimator.steps if step[0] == "dtypes")
        else:
            try:
                _, dtypes = next(
                    step for step in prep_pipe.steps if step[0] == "dtypes"
                )
                estimator_ = deepcopy(prep_pipe)
                estimator_.steps.append(["trained model", estimator])
                estimator = estimator_
                del estimator_

            except:
                raise ValueError("Pipeline not found")

        Xtest = data.copy()
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

    pred_ = estimator.predict(Xtest)

    try:
        pred_prob = estimator.predict_proba(Xtest)

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

    if data is None:
        metrics = _calculate_metrics(ytest, pred_, pred_prob)
        df_score = pd.DataFrame(metrics)
        df_score.insert(0, "Model", full_name)
        df_score = df_score.round(round)
        display.display(df_score.style.set_precision(round), clear=False)

    label = pd.DataFrame(pred_)
    label.columns = ["Label"]
    label["Label"] = label["Label"].astype(int)
    replace_lables_in_column(label["Label"])

    if data is None:
        replace_lables_in_column(ytest)
        X_test_ = pd.concat([Xtest, ytest, label], axis=1)
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
    try:
        display_container.append(df_score)
    except:
        pass

    return X_test_


def finalize_model(estimator, display=None):  # added in pycaret==2.2.0

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

    if not display:
        display = Display(False, html_param, logger=logger,)

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    # run_time
    import datetime, time

    runtime_start = time.time()

    logger.info("Importing libraries")
    # import depedencies
    from sklearn.base import clone
    from copy import deepcopy
    import numpy as np

    np.random.seed(seed)

    logger.info("Getting model name")

    _estimator_ = estimator

    if _is_one_vs_rest(estimator):
        if not hasattr(estimator, "voting"):
            estimator = estimator.estimator

    full_name = _get_model_name(estimator)

    estimator = _estimator_

    logger.info(f"Finalizing {full_name}")
    model_final = clone(estimator)
    display.clear_output()
    model_final = create_model(
        estimator=model_final,
        verbose=False,
        system=False,
        X_train_data=X,
        Y_train_data=y,
    )
    model_results = pull(pop=True)

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # mlflow logging
    if logging_param:

        logger.info("Creating MLFlow logs")

        # import mlflow
        import mlflow
        from pathlib import Path
        import mlflow.sklearn
        import os

        mlflow.set_experiment(exp_name_log)

        with mlflow.start_run(run_name=full_name) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            # Log model parameters
            try:
                params = model_final.get_params()

                for i in list(params):
                    v = params.get(i)
                    if len(str(v)) > 250:
                        params.pop(i)

                mlflow.log_params(params)

            except:
                pass

            # get metrics of non-finalized model and log it

            # Log metrics
            mlflow.log_metrics(
                {
                    k: v
                    for k, v in model_results.drop("TT (Sec)", axis=1, errors="ignore")
                    .loc["Mean"]
                    .items()
                }
            )

            # set tag of compare_models
            mlflow.set_tag("Source", "finalize_model")

            # create MRI (model registration id)
            mlflow.set_tag("Final", True)

            import secrets

            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time in seconds
            mlflow.log_metric("TT", runtime)

            # Log AUC and Confusion Matrix plot
            if log_plots_param:

                logger.info(
                    "SubProcess plot_model() called =================================="
                )

                try:
                    plot_model(
                        model_final, plot="auc", verbose=False, save=True, system=False
                    )
                    mlflow.log_artifact("AUC.png")
                    os.remove("AUC.png")
                except:
                    pass

                try:
                    plot_model(
                        model_final,
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
                        model_final,
                        plot="feature",
                        verbose=False,
                        save=True,
                        system=False,
                    )
                    mlflow.log_artifact("Feature Importance.png")
                    os.remove("Feature Importance.png")
                except:
                    pass

                logger.info(
                    "SubProcess plot_model() end =================================="
                )

            # Log model and transformation pipeline
            from copy import deepcopy

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

            signature = infer_signature(data_before_preprocess)

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(["trained model", model_final])
            mlflow.sklearn.log_model(
                prep_pipe_temp,
                "model",
                conda_env=default_conda_env,
                signature=signature,
            )
            del prep_pipe_temp

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

    return model_final


def deploy_model(model, model_name: str, authentication: dict, platform: str = "aws"):

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
    Before deploying a model to Google Cloud Platform (GCP), user has to create Project
    on the platform from consol. To do that, user must have google cloud account or
    create new one. After creating a service account, down the JSON authetication file
    and configure  GOOGLE_APPLICATION_CREDENTIALS= <path-to-json> from command line. If
    using google-colab then authetication can be done using `google.colab` auth method.
    Read below link for more details.

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
    
    model_name : string
        Name of model to be passed as a string.
    
    authentication : dict
        Dictionary of applicable authentication tokens.

        When platform = 'aws':
        {'bucket' : 'Name of Bucket on S3'}

        When platform = 'gcp':
        {'project': 'gcp_pycaret', 'bucket' : 'pycaret-test'}

        When platform = 'azure':
        {'container': 'pycaret-test'}
    
    platform: string, default = 'aws'
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
    
    model_name : string, default = none
        Name of pickle file to be passed as a string.
    
    model_only : bool, default = False
        When set to True, only trained model object is saved and all the 
        transformations are ignored.

    verbose: Boolean, default = True
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
    model_name, platform: str = None, authentication: dict = None, verbose: bool = True
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
    model_name : string, default = none
        Name of pickle file to be passed as a string.
      
    platform: string, default = None
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
    
    verbose: Boolean, default = True
        Success message is not printed when verbose is set to False.

    Returns
    -------
    Model Object

    """

    import pycaret.internal.persistence

    return pycaret.internal.persistence.load_model(
        model_name, platform, authentication, verbose
    )


def automl(optimize: str = "Accuracy", use_holdout: bool = False):

    """
    This function returns the best model out of all models created in 
    current active environment based on metric defined in optimize parameter. 

    Parameters
    ----------
    optimize : string, default = 'Accuracy'
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
        if not all_metrics[all_metrics["Name"] == optimize].iloc[0]["Multiclass"]:
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


def pull(pop=False) -> pandas.DataFrame:  # added in pycaret==2.2.0
    """
    Returns latest displayed table.

    Parameters
    ----------
    pop : Boolean, default = False
        If true, will pop (remove) the returned dataframe from the
        display container.

    Returns
    -------
    pandas.DataFrame
        Equivalent to get_config('display_container')[-1]

    """
    return display_container.pop(-1) if pop else display_container[-1]


def models(
    type: str = None, internal: bool = False, force_regenerate: bool = False
) -> pandas.DataFrame:

    """
    Returns table of models available in model library.

    Example
    -------
    >>> all_models = models()

    This will return pandas dataframe with all available 
    models and their metadata.

    Parameters
    ----------
    type : string, default = None
        - linear : filters and only return linear models
        - tree : filters and only return tree based models
        - ensemble : filters and only return ensemble models
    
    internal: Boolean, default = False
        If True, will return extra columns and rows used internally.

    force_regenerate: Boolean, default = False
        If True, will force the DataFrame to be regenerated,
        instead of using a cached version.

    Returns
    -------
    pandas.DataFrame

    """

    def filter_model_df_by_type(df):
        linear_models = ["lr", "ridge", "svm"]
        tree_models = ["dt"]
        ensemble_models = ["rf", "et", "gbc", "xgboost", "lightgbm", "catboost", "ada"]
        if type == "linear":
            df = df[df.index.isin(linear_models)]
        elif type == "tree":
            df = df[df.index.isin(tree_models)]
        elif type == "ensemble":
            df = df[df.index.isin(ensemble_models)]
        return df

    if not force_regenerate:
        try:
            if internal:
                return filter_model_df_by_type(_all_models_internal)
            else:
                return filter_model_df_by_type(all_models)
        except:
            pass

    import pandas as pd
    import numpy as np

    np.random.seed(seed)

    columns = ["ID", "Name", "Reference", "Turbo"]

    rows = [
        ("lr", "Logistic Regression", "sklearn.linear_model.LogisticRegression", True),
        (
            "knn",
            "K Neighbors Classifier",
            "sklearn.neighbors.KNeighborsClassifier",
            True,
        ),
        ("nb", "Naive Bayes", "sklearn.naive_bayes.GaussianNB", True),
        ("dt", "Decision Tree Classifier", "sklearn.tree.DecisionTreeClassifier", True),
        ("svm", "SVM - Linear Kernel", "sklearn.linear_model.SGDClassifier", True),
        ("rbfsvm", "SVM - Radial Kernel", "sklearn.svm.SVC", False),
        ("gpc", "Gaussian Process Classifier", "sklearn.gaussian_process.GPC", False),
        ("mlp", "MLP Classifier", "sklearn.neural_network.MLPClassifier", False),
        ("ridge", "Ridge Classifier", "sklearn.linear_model.RidgeClassifier", True),
        (
            "rf",
            "Random Forest Classifier",
            "sklearn.ensemble.RandomForestClassifier",
            True,
        ),
        (
            "qda",
            "Quadratic Discriminant Analysis",
            "sklearn.discriminant_analysis.QDA",
            True,
        ),
        ("ada", "Ada Boost Classifier", "sklearn.ensemble.AdaBoostClassifier", True),
        (
            "gbc",
            "Gradient Boosting Classifier",
            "sklearn.ensemble.GradientBoostingClassifier",
            True,
        ),
        (
            "lda",
            "Linear Discriminant Analysis",
            "sklearn.discriminant_analysis.LDA",
            True,
        ),
        ("et", "Extra Trees Classifier", "sklearn.ensemble.ExtraTreesClassifier", True),
        ("xgboost", "Extreme Gradient Boosting", "xgboost.readthedocs.io", True),
        (
            "lightgbm",
            "Light Gradient Boosting Machine",
            "github.com/microsoft/LightGBM",
            True,
        ),
        ("catboost", "CatBoost Classifier", "catboost.ai", True),
    ]

    df_internal = None

    if internal:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.svm import SVC
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import RidgeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.multiclass import OneVsRestClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier

        # special estimators
        from sklearn.ensemble import BaggingClassifier
        from sklearn.ensemble import StackingClassifier
        from sklearn.ensemble import VotingClassifier
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.calibration import CalibratedClassifierCV

        num_class = y.value_counts().count()

        rows += [
            (
                "Bagging",
                "Bagging Classifier",
                "sklearn.ensemble.BaggingClassifier",
                False,
            ),
            (
                "Stacking",
                "Stacking Classifier",
                "sklearn.ensemble.StackingClassifier",
                False,
            ),
            ("Voting", "Voting Classifier", "sklearn.ensemble.VotingClassifier", False),
            (
                "OneVsRest",
                "One Vs Rest Classifier",
                "sklearn.multiclass.OneVsRestClassifier",
                False,
            ),
            (
                "Calibrated",
                "Calibrated Classifier",
                "sklearn.calibration.CalibratedClassifierCV",
                False,
            ),
        ]

        internal_columns = [
            "Special",
            "Class",
            "Args",
            "Tune Grid",
            "Tune Args",
            "SHAP",
        ]
        internal_rows = [
            (
                False,
                LogisticRegression,
                {"random_state": seed},
                {
                    "C": np.arange(0, 10, 0.001),
                    "penalty": ["l1", "l2"],
                    "class_weight": ["balanced", None],
                },
                {"iid": False},
                False,
            ),
            (
                False,
                KNeighborsClassifier,
                {"n_jobs": n_jobs_param},
                {
                    "n_neighbors": range(1, 51),
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"],
                },
                {"iid": False},
                False,
            ),
            (
                False,
                GaussianNB,
                {},
                {
                    "var_smoothing": [
                        0.000000001,
                        0.000000002,
                        0.000000005,
                        0.000000008,
                        0.000000009,
                        0.0000001,
                        0.0000002,
                        0.0000003,
                        0.0000005,
                        0.0000007,
                        0.0000009,
                        0.00001,
                        0.001,
                        0.002,
                        0.003,
                        0.004,
                        0.005,
                        0.007,
                        0.009,
                        0.004,
                        0.005,
                        0.006,
                        0.007,
                        0.008,
                        0.009,
                        0.01,
                        0.1,
                        1,
                    ]
                },
                {},
                False,
            ),
            (
                False,
                DecisionTreeClassifier,
                {"random_state": seed},
                {
                    "max_depth": list(range(1, int(len(X_train.columns) + 1 * 0.85))),
                    "max_features": list(range(1, len(X_train.columns) + 1)),
                    "min_samples_leaf": [2, 3, 4, 5, 6],
                    "criterion": ["gini", "entropy"],
                },
                {"iid": False},
                "type1",
            ),
            (
                False,
                SGDClassifier,
                {
                    "max_iter": 1000,
                    "tol": 0.001,
                    "random_state": seed,
                    "n_jobs": n_jobs_param,
                },
                {
                    "penalty": ["l2", "l1", "elasticnet"],
                    "l1_ratio": np.arange(0, 1, 0.01),
                    "alpha": [
                        0.0001,
                        0.001,
                        0.01,
                        0.0002,
                        0.002,
                        0.02,
                        0.0005,
                        0.005,
                        0.05,
                    ],
                    "fit_intercept": [True, False],
                    "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
                    "eta0": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                },
                {},
                False,
            ),
            (
                False,
                SVC,
                {
                    "gamma": "auto",
                    "C": 1,
                    "probability": True,
                    "kernel": "rbf",
                    "random_state": seed,
                },
                {"C": np.arange(0, 50, 0.01), "class_weight": ["balanced", None],},
                {},
                False,
            ),
            (
                False,
                GaussianProcessClassifier,
                {"random_state": seed, "n_jobs": n_jobs_param},
                {
                    "max_iter_predict": [
                        100,
                        200,
                        300,
                        400,
                        500,
                        600,
                        700,
                        800,
                        900,
                        1000,
                    ]
                },
                {},
                False,
            ),
            (
                False,
                MLPClassifier,
                {"random_state": seed, "max_iter": 500},
                {
                    "learning_rate": ["constant", "invscaling", "adaptive"],
                    "solver": ["lbfgs", "sgd", "adam"],
                    "alpha": np.arange(0, 1, 0.0001),
                    "hidden_layer_sizes": [
                        (50, 50, 50),
                        (50, 100, 50),
                        (100,),
                        (100, 50, 100),
                        (100, 100, 100),
                    ],
                    "activation": ["tanh", "identity", "logistic", "relu"],
                },
                {"iid": False},
                False,
            ),
            (
                False,
                RidgeClassifier,
                {"random_state": seed},
                {
                    "alpha": np.arange(0, 1, 0.001),
                    "fit_intercept": [True, False],
                    "normalize": [True, False],
                },
                {},
                False,
            ),
            (
                False,
                RandomForestClassifier,
                {"n_estimators": 10, "random_state": seed, "n_jobs": n_jobs_param},
                {
                    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
                    "min_samples_split": [2, 5, 7, 9, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["auto", "sqrt", "log2"],
                    "bootstrap": [True, False],
                },
                {},
                "type1",
            ),
            (
                False,
                QuadraticDiscriminantAnalysis,
                {},
                {"reg_param": np.arange(0, 1, 0.01)},
                {},
                False,
            ),
            (
                False,
                AdaBoostClassifier,
                {"random_state": seed},
                {
                    "n_estimators": np.arange(10, 200, 5),
                    "learning_rate": np.arange(0, 1, 0.01),
                    "algorithm": ["SAMME", "SAMME.R"],
                },
                {},
                False,
            ),
            (
                False,
                GradientBoostingClassifier,
                {"random_state": seed},
                {
                    "n_estimators": np.arange(10, 200, 5),
                    "learning_rate": np.arange(0, 1, 0.01),
                    "subsample": np.arange(0.1, 1, 0.05),
                    "min_samples_split": [2, 4, 5, 7, 9, 10],
                    "min_samples_leaf": [1, 2, 3, 4, 5],
                    "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
                    "max_features": ["auto", "sqrt", "log2"],
                },
                {},
                "type2",
            ),
            (
                False,
                LinearDiscriminantAnalysis,
                {},
                {
                    "solver": ["lsqr", "eigen"],
                    "shrinkage": [
                        None,
                        0.0001,
                        0.001,
                        0.01,
                        0.0005,
                        0.005,
                        0.05,
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        1,
                    ],
                },
                {},
                False,
            ),
            (
                False,
                ExtraTreesClassifier,
                {"random_state": seed, "n_jobs": n_jobs_param},
                {
                    "n_estimators": np.arange(10, 200, 5),
                    "criterion": ["gini", "entropy"],
                    "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
                    "min_samples_split": [2, 5, 7, 9, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["auto", "sqrt", "log2"],
                    "bootstrap": [True, False],
                },
                {},
                "type1",
            ),
            (
                False,
                XGBClassifier,
                {"verbosity": 0, "random_state": seed, "n_jobs": n_jobs_param},
                {
                    "learning_rate": np.arange(0, 1, 0.01),
                    "n_estimators": np.arange(10, 500, 20),
                    "subsample": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                    "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
                    "colsample_bytree": [0.5, 0.7, 0.9, 1],
                    "min_child_weight": [1, 2, 3, 4],
                    "num_class": [num_class, num_class],
                }
                if num_class > 2
                else {
                    "learning_rate": np.arange(0, 1, 0.01),
                    "n_estimators": [
                        10,
                        30,
                        50,
                        100,
                        200,
                        300,
                        400,
                        500,
                        600,
                        700,
                        800,
                        900,
                        1000,
                    ],
                    "subsample": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                    "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
                    "colsample_bytree": [0.5, 0.7, 0.9, 1],
                    "min_child_weight": [1, 2, 3, 4],
                },
                {},
                "type2",
            ),
            (
                False,
                LGBMClassifier,
                {"random_state": seed, "n_jobs": n_jobs_param},
                {
                    "num_leaves": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],
                    "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
                    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    "n_estimators": [10, 30, 50, 70, 90, 100, 120, 150, 170, 200],
                    "min_split_gain": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    "reg_alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    "reg_lambda": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                },
                {},
                "type1",
            ),
            (
                False,
                CatBoostClassifier,
                {"random_state": seed, "silent": True, "thread_count": n_jobs_param},
                {
                    "depth": [3, 1, 2, 6, 4, 5, 7, 8, 9, 10],
                    "iterations": [250, 100, 500, 1000],
                    "learning_rate": [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
                    "l2_leaf_reg": [3, 1, 5, 10, 100],
                    "border_count": [32, 5, 10, 20, 50, 100, 200],
                },
                {},
                "type2",
            ),
            (
                True,
                BaggingClassifier,
                {"random_state": seed, "n_jobs": n_jobs_param,},
                {
                    "n_estimators": np.arange(10, 300, 10),
                    "bootstrap": [True, False],
                    "bootstrap_features": [True, False],
                },
                {},
                False,
            ),
            (True, StackingClassifier, {}, {}, {}, False),
            (True, VotingClassifier, {}, {}, {}, False),
            (True, OneVsRestClassifier, {"n_jobs": n_jobs_param}, {}, {}, False),
            (True, CalibratedClassifierCV, {}, {}, {}, False),
        ]

        df_internal = pd.DataFrame(internal_rows)
        df_internal.columns = internal_columns

        def is_boosting_supported(e):
            try:
                x = e()
                return bool(hasattr(x, "class_weights") or hasattr(x, "predict_proba"))
            except:
                return False

        def is_soft_voting_supported(e):
            try:
                x = e()
                return bool(hasattr(x, "predict_proba"))
            except:
                return False

        df_internal["Boosting Supported"] = [
            is_boosting_supported(x) for x in df_internal["Class"]
        ]

        df_internal["Soft Voting"] = [
            is_soft_voting_supported(x) for x in df_internal["Class"]
        ]

    df = pd.DataFrame(rows)
    df.columns = columns
    if df_internal is not None:
        df = pd.concat([df, df_internal], axis=1)

    df.set_index("ID", inplace=True)

    return filter_model_df_by_type(df)


def get_metrics(force_regenerate: bool = False) -> pandas.DataFrame:
    """
    Returns table of metrics available.

    Example
    -------
    >>> metrics = get_metrics()

    This will return pandas dataframe with all available 
    metrics and their metadata.

    Parameters
    ----------
    force_regenerate: Boolean, default = False
        If True, will force the DataFrame to be regenerated,
        instead of using a cached version.

    Returns
    -------
    pandas.DataFrame

    """

    if not force_regenerate:
        try:
            return all_metrics
        except:
            pass

    import pandas as pd
    import numpy as np
    from sklearn import metrics

    np.random.seed(seed)

    columns = [
        "ID",
        "Name",
        "Display Name",
        "Scorer",
        "Score Function",
        "Target",
        "Args",
        "Multiclass",
    ]
    rows = [
        (
            "acc",
            "Accuracy",
            "Accuracy",
            "accuracy",
            metrics.accuracy_score,
            "pred",
            {},
            True,
        ),
        ("auc", "AUC", "AUC", "roc_auc", metrics.roc_auc_score, "pred_prob", {}, False),
        (
            "recall",
            "Recall",
            "Recall",
            metrics.make_scorer(metrics.recall_score, average="macro")
            if _is_multiclass()
            else "recall",
            metrics.recall_score,
            "pred",
            {"average": "macro"} if _is_multiclass() else {},
            True,
        ),
        (
            "precision",
            "Precision",
            "Prec.",
            metrics.make_scorer(metrics.precision_score, average="weighted")
            if _is_multiclass()
            else "precision",
            metrics.precision_score,
            "pred",
            {"average": "weighted"} if _is_multiclass() else {},
            True,
        ),
        (
            "f1",
            "F1",
            "F1",
            metrics.make_scorer(metrics.f1_score, average="weighted")
            if _is_multiclass()
            else "f1",
            metrics.f1_score,
            "pred",
            {"average": "weighted"} if _is_multiclass() else {},
            True,
        ),
        (
            "kappa",
            "Kappa",
            "Kappa",
            metrics.make_scorer(metrics.cohen_kappa_score),
            metrics.cohen_kappa_score,
            "pred",
            {},
            True,
        ),
        (
            "mcc",
            "MCC",
            "MCC",
            "roc_auc"
            if _is_multiclass()
            else metrics.make_scorer(metrics.matthews_corrcoef),
            metrics.matthews_corrcoef,
            "pred",
            {},
            True,
        ),
        ("tt", "TT", "TT (Sec)", None, None, None, None, True),
    ]
    df = pd.DataFrame(rows)
    df.columns = columns

    df.set_index("ID", inplace=True)
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
    score_func,
    scorer=None,
    target: str = "pred",
    args: dict = {},
    multiclass: bool = True,
) -> pandas.Series:
    """
    Adds a custom metric to be used in all functions.

    Example
    -------
    >>> metrics = get_metrics()

    This will return pandas dataframe with all available 
    metrics and their metadata.

    Parameters
    ----------
    id: str
        Unique id for the metric.

    name: str
        Display name of the metric.

    score_func: callable
        Score function (or loss function) with signature score_func(y, y_pred, **kwargs).

    scorer: sklearn.metrics.Scorer, default = None
        The Scorer to be used in tuning and cross validation. If None, one will be created
        from score_func.

    target: str, default = 'pred'
        The target of the score function.
        - 'pred' for the prediction table
        - 'pred_proba' for pred_proba

    args: dict, default = {}
        Arguments to be passed to score function.

    multiclass: Boolean, default = True
        Whether the metric supports multiclass problems.

    Notes
    -----
    The row will be inserted into the second to last position. The last position is reserved
    for the Training Time (tt) metric.

    Returns
    -------
    pandas.Series
        The created row as Series.

    """

    if not "all_metrics" in globals():
        raise ValueError("setup() needs to be ran first.")

    import pandas as pd
    import numpy as np
    from sklearn import metrics

    np.random.seed(seed)

    new_metric = {
        "Name": name,
        "Display Name": name,
        "Scorer": scorer if scorer else metrics.make_scorer(score_func),
        "Score Function": score_func,
        "Target": target,
        "Args": args,
        "Multiclass": multiclass,
    }

    new_metric = pd.Series(new_metric, name=id.replace(" ", "_"))

    global all_metrics
    last_row = all_metrics.iloc[-1]
    all_metrics.drop(all_metrics.index[-1], inplace=True)
    all_metrics = all_metrics.append(new_metric)
    all_metrics = all_metrics.append(last_row)
    return all_metrics.iloc[-2]


def get_logs(experiment_name: str = None, save: bool = False) -> pandas.DataFrame:

    """
    Returns a table with experiment logs consisting
    run details, parameter, metrics and tags. 

    Example
    -------
    >>> logs = get_logs()

    This will return pandas dataframe.

    Parameters
    ----------
    experiment_name : string, default = None
        When set to None current active run is used.

    save : bool, default = False
        When set to True, csv file is saved in current directory.

    Returns
    -------
    pandas.DataFrame

    """

    import sys

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
    - folds_shuffle_param: shuffle parameter used in Kfolds
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
    - folds_shuffle_param: shuffle parameter used in Kfolds
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


def _is_one_vs_rest(e) -> bool:
    """
    Checks if the estimator is OneVsRestClassifier.
    """
    return type(e) == _all_models_internal.loc["OneVsRest"]["Class"]


def _fix_imbalance(
    Xtrain: pandas.DataFrame,
    ytrain: pandas.DataFrame,
    fix_imbalance_method_param: Any = None,
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:

    """
    Method to fix imbalance using fix_imbalance_method_param.
    """

    logger = get_logger()
    logger.info("Initializing SMOTE")

    if fix_imbalance_method_param is None:
        import six
        import sys

        sys.modules["sklearn.externals.six"] = six
        from imblearn.over_sampling import SMOTE

        resampler = SMOTE(random_state=seed)
    else:
        resampler = fix_imbalance_method_param

    Xtrain, ytrain = resampler.fit_sample(Xtrain, ytrain)
    logger.info("Resampling completed")
    return Xtrain, ytrain


def _choose_better(
    model,
    new_estimator_list: list,
    compare_dimension: str,
    fold: int,
    model_results=None,
    new_results_list: list = None,
    display: Display = None,
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
    display.update_monitor(2, "Almost Finished")
    display.display_monitor()

    scorer = []

    if model_results is None:
        logger.info(
            "SubProcess create_model() called =================================="
        )
        create_model(model, verbose=False, system=False, fold=fold)
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
            m = create_model(new_estimator, verbose=False, system=False, fold=fold)
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
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import random
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    import datetime, time

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

        X_, X__, y_, y__ = train_test_split(
            X,
            y,
            test_size=1 - i,
            stratify=y,
            random_state=seed,
            shuffle=data_split_shuffle,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_,
            y_,
            test_size=1 - train_size,
            stratify=y_,
            random_state=seed,
            shuffle=data_split_shuffle,
        )
        model.fit(X_train, y_train)
        pred_ = model.predict(X_test)
        try:
            pred_prob = model.predict_proba(X_test)[:, 1]
        except:
            logger.warning("model has no predict_proba attribute.")
            pred_prob = 0

        _calculate_metrics(y_test, pred_, pred_prob, score_dict)

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

        ttt = sum(split_perc_tt_total) / 60
        ttt = np.around(ttt, 2)

        if ttt < 1:
            ttt = str(np.around((ttt * 60), 2))
            ETC = f"{ttt} Seconds Remaining"

        else:
            ttt = str(ttt)
            ETC = f"{ttt} Minutes Remaining"

        display.update_monitor(2, ETC)
        display.display_monitor()

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

    if sample_size == "" or sample_size == "1":

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=1 - train_size,
            stratify=y,
            random_state=seed,
            shuffle=data_split_shuffle,
        )

    else:

        sample_n = float(sample_size)
        X_selected, X_discard, y_selected, y_discard = train_test_split(
            X,
            y,
            test_size=1 - sample_n,
            stratify=y,
            random_state=seed,
            shuffle=data_split_shuffle,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected,
            y_selected,
            test_size=1 - train_size,
            stratify=y_selected,
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


def _get_model_name(e) -> str:
    """
    Get model name.
    """
    import pycaret.internal.utils

    return pycaret.internal.utils.get_model_name(e, models(internal=True))


def _is_special_model(e) -> bool:
    """
    Is the model special (eg. VotingClassifier).
    """
    import pycaret.internal.utils

    return pycaret.internal.utils.is_special_model(e, models(internal=True))


def _calculate_metrics(ytest, pred_, pred_prob: float, score_dict: dict = None) -> dict:
    """
    Calculate all metrics in get_metrics().
    """
    from pycaret.internal.utils import calculate_metrics

    return calculate_metrics(get_metrics(), ytest, pred_, pred_prob, score_dict)
