# Module: Classification
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT
# Release: PyCaret 2.2
# Last modified : 19/10/2020

import pandas as pd
import numpy as np

import pycaret.internal.tabular
from pycaret.internal.Display import Display, is_in_colab, enable_colab
from typing import List, Tuple, Any, Union, Optional, Dict
import warnings
from IPython.utils import io
import traceback

from pycaret.internal.tabular import MLUsecase

warnings.filterwarnings("ignore")


def setup(
    data: pd.DataFrame,
    target: str,
    train_size: float = 0.7,
    test_data: Optional[pd.DataFrame] = None,
    preprocess: bool = True,
    imputation_type: str = "simple",
    iterative_imputation_iters: int = 5,
    categorical_features: Optional[List[str]] = None,
    categorical_imputation: str = "constant",
    categorical_iterative_imputer: Union[str, Any] = "lightgbm",
    ordinal_features: Optional[Dict[str, list]] = None,
    high_cardinality_features: Optional[List[str]] = None,
    high_cardinality_method: str = "frequency",
    numeric_features: Optional[List[str]] = None,
    numeric_imputation: str = "mean",
    numeric_iterative_imputer: Union[str, Any] = "lightgbm",
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
    remove_perfect_collinearity: bool = True,
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
    feature_selection_method: str = "classic", 
    feature_interaction: bool = False,
    feature_ratio: bool = False,
    interaction_threshold: float = 0.01,
    fix_imbalance: bool = False,
    fix_imbalance_method: Optional[Any] = None,
    data_split_shuffle: bool = True,
    data_split_stratify: Union[bool, List[str]] = False, 
    fold_strategy: Union[str, Any] = "stratifiedkfold", 
    fold: int = 10,
    fold_shuffle: bool = False,
    fold_groups: Optional[Union[str, pd.DataFrame]] = None,
    n_jobs: Optional[int] = -1,
    use_gpu: bool = False, 
    custom_pipeline: Union[
        Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]
    ] = None,
    html: bool = True,
    session_id: Optional[int] = None,
    log_experiment: bool = False,
    experiment_name: Optional[str] = None,
    log_plots: Union[bool, list] = False,
    log_profile: bool = False,
    log_data: bool = False,
    silent: bool = False,
    verbose: bool = True,
    profile: bool = False,
):

    """
    This function initializes the training environment and creates the transformation 
    pipeline. Setup function must called before executing any other function. It takes 
    two mandatory parameters: ``data`` and ``target``. All the other parameters are
    optional.

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')

    Parameters
    ----------
    data : pandas.DataFrame
        Shape (n_samples, n_features), where n_samples is the number of samples and 
        n_features is the number of features.

    target: str
        Name of the target column to be passed in as a string. The target variable can 
        be either binary or multiclass.

    train_size: float, default = 0.7
        Proportion of the dataset to be used for training and validation. Should be 
        between 0.0 and 1.0.

    test_data: pandas.DataFrame, default = None
        If not None, test_data is used as a hold-out set and ``train_size`` parameter is 
        ignored. test_data must be labelled and the shape of data and test_data must 
        match. 

    preprocess: bool, default = True
        When set to False, no transformations are applied except for train_test_split 
        and custom transformations passed in ``custom_pipeline`` param. Data must be 
        ready for modeling (no missing values, no dates, categorical data encoding), 
        when preprocess is set to False. 

    imputation_type: str, default = 'simple'
        The type of imputation to use. Can be either 'simple' or 'iterative'.

    iterative_imputation_iters: int, default = 5
        Number of iterations. Ignored when ``imputation_type`` is not 'iterative'.	

    categorical_features: list of str, default = None
        If the inferred data types are not correct or the silent param is set to True,
        categorical_features param can be used to overwrite or define the data types. 
        It takes a list of strings with column names that are categorical.

    categorical_imputation: str, default = 'constant'
        Missing values in categorical features are imputed with a constant 'not_available'
        value. The other available option is 'mode'.
    
    categorical_iterative_imputer: str, default = 'lightgbm'
        Estimator for iterative imputation of missing values in categorical features.
        Ignored when ``imputation_type`` is not 'iterative'. 

    ordinal_features: dict, default = None
        Encode categorical features as ordinal. For example, a categorical feature with 
        'low', 'medium', 'high' values where low < medium < high can be passed as  
        ordinal_features = { 'column_name' : ['low', 'medium', 'high'] }. 
    
    high_cardinality_features: list of str, default = None
        When categorical features contains many levels, it can be compressed into fewer
        levels using this parameter. It takes a list of strings with column names that 
        are categorical.
    
    high_cardinality_method: str, default = 'frequency'
        Categorical features with high cardinality are replaced with the frequency of
        values in each level occurring in the training dataset. Other available method
        is 'clustering' which trains the K-Means clustering algorithm on the statistical
        attribute of the training data and replaces the original value of feature with the 
        cluster label. The number of clusters is determined by optimizing Calinski-Harabasz 
        and Silhouette criterion. 
          
    numeric_features: list of str, default = None
        If the inferred data types are not correct or the silent param is set to True,
        numeric_features param can be used to overwrite or define the data types. 
        It takes a list of strings with column names that are numeric.

    numeric_imputation: str, default = 'mean'
        Missing values in numeric features are imputed with 'mean' value of the feature 
        in the training dataset. The other available option is 'median' or 'zero'.

    numeric_iterative_imputer: str, default = 'lightgbm'
        Estimator for iterative imputation of missing values in numeric features.
        Ignored when ``imputation_type`` is set to 'simple'. 

    date_features: list of str, default = None
        If the inferred data types are not correct or the silent param is set to True,
        date_features param can be used to overwrite or define the data types. It takes 
        a list of strings with column names that are DateTime.

    ignore_features: list of str, default = None
        ignore_features param can be used to ignore features during model training.
        It takes a list of strings with column names that are to be ignored.

    normalize: bool, default = False
        When set to True, it transforms the numeric features by scaling them to a given
        range. Type of scaling is defined by the ``normalize_method`` parameter.

    normalize_method: str, default = 'zscore'
        Defines the method for scaling. By default, normalize method is set to 'zscore'
        The standard zscore is calculated as z = (x - u) / s. Ignored when ``normalize`` 
        is not True. The other options are:
    
        - 'minmax': scales and translates each feature individually such that it is in 
          the range of 0 - 1.
        - 'maxabs': scales and translates each feature individually such that the 
          maximal absolute value of each feature will be 1.0. It does not 
          shift/center the data, and thus does not destroy any sparsity.
        - 'robust' : scales and translates each feature according to the Interquartile 
          range. When the dataset contains outliers, robust scaler often gives 
          better results.

    transformation: bool, default = False
        When set to True, it applies the power transform to make data more Gaussian-like.
        Type of transformation is defined by the ``transformation_method`` parameter.

    transformation_method: str, default = 'yeo-johnson'
        Defines the method for transformation. By default, the transformation method is set
        to 'yeo-johnson'. The other available option for transformation is 'quantile'. 
        Ignored when ``transformation`` is not True.
    
    handle_unknown_categorical: bool, default = True
        When set to True, unknown categorical levels in unseen data are replaced by the
        most or least frequent level as learned in the training dataset. 

    unknown_categorical_method: str, default = 'least_frequent'
        Method used to replace unknown categorical levels in unseen data. Method can be
        set to 'least_frequent' or 'most_frequent'.

    pca: bool, default = False
        When set to True, dimensionality reduction is applied to project the data into 
        a lower dimensional space using the method defined in ``pca_method`` parameter. 

    pca_method: str, default = 'linear'
        The 'linear' method performs uses Singular Value  Decomposition. Other options are:
        
        - kernel: dimensionality reduction through the use of RVF kernel.
        - incremental: replacement for 'linear' pca when the dataset is too large.

    pca_components: int or float, default = None
        Number of components to keep. if pca_components is a float, it is treated as a 
        target percentage for information retention. When pca_components is an integer
        it is treated as the number of features to be kept. pca_components must be less
        than the original number of features. Ignored when ``pca`` is not True.

    ignore_low_variance: bool, default = False
        When set to True, all categorical features with insignificant variances are 
        removed from the data. The variance is calculated using the ratio of unique 
        values to the number of samples, and the ratio of the most common value to the 
        frequency of the second most common value.
    
    combine_rare_levels: bool, default = False
        When set to True, frequency percentile for levels in categorical features below 
        a certain threshold is combined into a single level.
    
    rare_level_threshold: float, default = 0.1
        Percentile distribution below which rare categories are combined. Ignored when
        ``combine_rare_levels`` is not True.
    
    bin_numeric_features: list of str, default = None
        To convert numeric features into categorical, bin_numeric_features parameter can 
        be used. It takes a list of strings with column names to be discretized. It does
        so by using 'sturges' rule to determine the number of clusters and then apply
        ``KMeans`` algorithm. Original values of the feature are then replaced by the
        cluster label.

    remove_outliers: bool, default = False
        When set to True, outliers from the training data are removed using the Singular 
        Value Decomposition.

    outliers_threshold: float, default = 0.05
        The percentage outliers to be removed from the training dataset. Ignored when 
        ``remove_outliers`` is not True.

    remove_multicollinearity: bool, default = False
        When set to True, features with the inter-correlations higher than the defined 
        threshold are removed. When two features are highly correlated with each other, 
        the feature that is less correlated with the target variable is removed. 

    multicollinearity_threshold: float, default = 0.9
        Threshold for correlated features. Ignored when ``remove_multicollinearity``
        is not True.
    
    remove_perfect_collinearity: bool, default = True
        When set to True, perfect collinearity (features with correlation = 1) is removed
        from the dataset, when two features are 100% correlated, one of it is randomly 
        removed from the dataset.

    create_clusters: bool, default = False
        When set to True, an additional feature is created in training dataset where each 
        instance is assigned to a cluster. The number of clusters is determined by 
        optimizing Calinski-Harabasz and Silhouette criterion. 

    cluster_iter: int, default = 20
        Number of iterations for creating cluster. Each iteration represents cluster 
        size. Ignored when ``create_clusters`` is not True. 

    polynomial_features: bool, default = False
        When set to True, new features are derived using existing numeric features. 

    polynomial_degree: int, default = 2
        Degree of polynomial features. For example, if an input sample is two dimensional 
        and of the form [a, b], the polynomial features with degree = 2 are: 
        [1, a, b, a^2, ab, b^2]. Ignored when ``polynomial_features`` is not True.

    trigonometry_features: bool, default = False
        When set to True, new features are derived using existing numeric features.

    polynomial_threshold: float, default = 0.1
        When ``polynomial_features`` or ``trigonometry_features`` is True, new features
        are derived from the existing numeric features. This may sometimes result in too 
        large feature space. polynomial_threshold parameter can be used to deal with this  
        problem. It does so by using combination of Random Forest, AdaBoost and Linear 
        correlation. All derived features that falls within the percentile distribution 
        are kept and rest of the features are removed.

    group_features: list or list of list, default = None
        When the dataset contains features with related characteristics, group_features
        parameter can be used for feature extraction. It takes a list of strings with 
        column names that are related.
        
    group_names: list, default = None
        Group names to be used in naming new features. When the length of group_names 
        does not match with the length of ``group_features``, new features are named 
        sequentially group_1, group_2, etc. It is ignored when ``group_features`` is
        None.
    
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

    fold_strategy: str or sklearn CV generator object, default = 'stratifiedkfold'
        Choice of cross validation strategy. Possible values are:

        * 'kfold' for KFold CV,
        * 'stratifiedkfold' for Stratified KFold CV,
        * 'groupkfold' for Group KFold CV,
        * 'timeseries' for TimeSeriesSplit CV,
        * a custom CV generator object compatible with scikit-learn.

    fold: int, default = 10
        Number of folds to be used in cross validation. Must be at least 2.
        Ignored if fold_strategy is an object.

    fold_shuffle: bool, default = False
        If set to False, prevents shuffling of rows when using cross validation. 
        Only applicable for 'kfold' and 'stratifiedkfold' fold_strategy. Ignored 
        when fold_strategy is an object.
    
    fold_groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into 
        train/test set. If string is passed, will use the data column with that name 
        as the groups. Only used if a group based cross-validation generator is used.

    n_jobs: int, default = -1
        The number of jobs to run in parallel (for functions that supports parallel 
        processing) -1 means using all processors. To run all functions on single 
        processor set n_jobs to None.

    use_gpu: str or bool, default = False
        If set to 'force', will try to use GPU with all algorithms that support it,
        and raise exceptions if they are unavailable. If set to True, will use GPU 
        with algorithms that support it, and fall back to CPU if they are unavailable.
        If set to False, will only use CPU.

        GPU enabled algorithms:
        
        - CatBoost
        - XGBoost
        - LightGBM - requires https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
        - Logistic Regression, Ridge, Random Forest, KNN, SVM, SVC - requires cuML >= 0.15 
        to be installed. https://github.com/rapidsai/cuml

    custom_pipeline: transformer or list of transformers or tuple
        (str, transformer) or list of tuples (str, transformer), default = None
        If set, will append the passed transformers (including Pipelines) to the PyCaret
        preprocessing Pipeline applied after train-test split during model fitting.
        This Pipeline is applied on each CV fold separately and on the final fit.
        The transformers will be applied before PyCaret transformers (eg. SMOTE).

    html: bool, default = True
        If set to False, prevents runtime display of monitor. This must be set to False
        when using environment that does not support ``html``.

    session_id: int, default = None
        If None, a random seed is generated and returned in the Information grid. The 
        unique number is then distributed as a seed in all functions used during the 
        experiment. This can be used for later reproducibility of the entire experiment.

    log_experiment: bool, default = False
        When set to True, all metrics and parameters are logged on ``MLFlow`` server.

    experiment_name: str, default = None
        Name of the experiment for logging. Ignored when ``log_experiment`` is not True.

    log_plots: bool or list, default = False
        When set to True, AUC, Confusion Matrix and Feature Importance plots will be 
        logged. When set to a list of plot IDs (consult ``plot_model``), will log those 
        plots. If False, will log no plots. 

    log_profile: bool, default = False
        When set to True, data profile is logged on ``MLflow`` server as a html file. 

    log_data: bool, default = False
        When set to True, dataset is logged on ``MLflow`` as a csv file.

    silent: bool, default = False
        When set to True, confirmation of data types is not required. All preprocessing 
        will be performed assuming automatically inferred data types. Not recommended 
        for direct use except for established pipelines.
    
    verbose: bool, default = True
        When set to False, Information grid is not printed.

    profile: bool, default = False
        When set to true, an interactive EDA report is displayed. 

    Returns
    -------
    info_grid
        Information grid is printed.

    environment
        This function returns various outputs that are stored in variables
        as tuples. They are used by other functions in pycaret.
      
       
    """
    available_plots = {
        "parameter": "Hyperparameters",
        "auc": "AUC",
        "confusion_matrix": "Confusion Matrix",
        "threshold": "Threshold",
        "pr": "Precision Recall",
        "error": "Prediction Error",
        "class_report": "Class Report",
        "rfe": "Feature Selection",
        "learning": "Learning Curve",
        "manifold": "Manifold Learning",
        "calibration": "Calibration Curve",
        "vc": "Validation Curve",
        "dimension": "Dimensions",
        "feature": "Feature Importance",
        "feature_all": "Feature Importance (All)",
        "boundary": "Decision Boundary",
        "lift": "Lift Chart",
        "gain": "Gain Chart",
        "tree": "Decision Tree",
    }

    if log_plots == True:
        log_plots = ["auc", "confusion_matrix", "feature"]

    return pycaret.internal.tabular.setup(
        ml_usecase="classification",
        available_plots=available_plots,
        data=data,
        target=target,
        train_size=train_size,
        test_data=test_data,
        preprocess=preprocess,
        imputation_type=imputation_type,
        iterative_imputation_iters=iterative_imputation_iters,
        categorical_features=categorical_features,
        categorical_imputation=categorical_imputation,
        categorical_iterative_imputer=categorical_iterative_imputer,
        ordinal_features=ordinal_features,
        high_cardinality_features=high_cardinality_features,
        high_cardinality_method=high_cardinality_method,
        numeric_features=numeric_features,
        numeric_imputation=numeric_imputation,
        numeric_iterative_imputer=numeric_iterative_imputer,
        date_features=date_features,
        ignore_features=ignore_features,
        normalize=normalize,
        normalize_method=normalize_method,
        transformation=transformation,
        transformation_method=transformation_method,
        handle_unknown_categorical=handle_unknown_categorical,
        unknown_categorical_method=unknown_categorical_method,
        pca=pca,
        pca_method=pca_method,
        pca_components=pca_components,
        ignore_low_variance=ignore_low_variance,
        combine_rare_levels=combine_rare_levels,
        rare_level_threshold=rare_level_threshold,
        bin_numeric_features=bin_numeric_features,
        remove_outliers=remove_outliers,
        outliers_threshold=outliers_threshold,
        remove_multicollinearity=remove_multicollinearity,
        multicollinearity_threshold=multicollinearity_threshold,
        remove_perfect_collinearity=remove_perfect_collinearity,
        create_clusters=create_clusters,
        cluster_iter=cluster_iter,
        polynomial_features=polynomial_features,
        polynomial_degree=polynomial_degree,
        trigonometry_features=trigonometry_features,
        polynomial_threshold=polynomial_threshold,
        group_features=group_features,
        group_names=group_names,
        feature_selection=feature_selection,
        feature_selection_threshold=feature_selection_threshold,
        feature_selection_method=feature_selection_method,
        feature_interaction=feature_interaction,
        feature_ratio=feature_ratio,
        interaction_threshold=interaction_threshold,
        fix_imbalance=fix_imbalance,
        fix_imbalance_method=fix_imbalance_method,
        data_split_shuffle=data_split_shuffle,
        data_split_stratify=data_split_stratify,
        fold_strategy=fold_strategy,
        fold=fold,
        fold_shuffle=fold_shuffle,
        fold_groups=fold_groups,
        n_jobs=n_jobs,
        use_gpu=use_gpu,
        custom_pipeline=custom_pipeline,
        html=html,
        session_id=session_id,
        log_experiment=log_experiment,
        experiment_name=experiment_name,
        log_plots=log_plots,
        log_profile=log_profile,
        log_data=log_data,
        silent=silent,
        verbose=verbose,
        profile=profile,
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
) -> Union[Any, List[Any]]:

    """
    This function trains and evaluates performance of estimators available in the model
    library using cross validation. The output of this function is a score grid with 
    average cross validated score. Metrics evaluated during CV can be accessed using
    the ``get_metrics`` function. Custom metrics can be added or removed using ``add_metric``
    and ``remove_metric`` function.    

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
    >>> best_model = compare_models() 

    Parameters
    ----------
    include: list of strings or objects, default = None
        In order to run only certain models for the comparison, the model ID's can be 
        passed as a list of strings in include param. The list can also include estimator
        objects to be compared.

    exclude: list of strings, default = None
        In order to omit certain models from the comparison model ID's can be passed as 
        a list of strings in exclude param. 

    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use StratifiedKFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.

    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.

    cross_validation: bool, default = True
        When cross_validation set to False fold parameter is ignored and models are 
        trained on entire training dataset, returning metrics calculated using the train 
        (holdout) set.

    sort: str, default = 'Accuracy'
        The scoring measure specified is used for sorting the average score grid
        Other options are 'AUC', 'Recall', 'Precision', 'F1', 'Kappa' and 'MCC'.

    n_select: int, default = 1
        Number of top_n models to return. use negative argument for bottom selection.
        for example, n_select = -3 means bottom 3 models.

    budget_time: int or float, default = None
        If not None, will terminate execution of the function after budget_time 
        minutes have passed and return results up to that point.

    turbo: bool, default = True
        When turbo is set to True, it excludes estimators that have longer
        training time.

    errors: str, default = 'ignore'
        If 'ignore', will suppress model exceptions and continue.
        If 'raise', will allow exceptions to be raised.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model. The parameters 
        will be applied to all models, therefore it is recommended to set errors 
        parameter to 'ignore'.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into 
        train/test set. If string is passed, will use the data column with that name 
        as the groups. Only used if a group based cross-validation generator is used 
        (eg. GroupKFold). If None, will use the value set in fold_groups param in setup().

    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.
    
    Returns
    -------
    score_grid
        A table containing the scores of the model across the kfolds. 
        Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are also returned.

    list or model
        List or one fitted model objects that were compared.

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

    return pycaret.internal.tabular.compare_models(
        include=include,
        exclude=exclude,
        fold=fold,
        round=round,
        cross_validation=cross_validation,
        sort=sort,
        n_select=n_select,
        budget_time=budget_time,
        turbo=turbo,
        errors=errors,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
    )


def create_model(
    estimator: Union[str, Any],
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
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
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')

    This will create a trained Logistic Regression model.

    Parameters
    ----------
    estimator : str / object, default = None
        Enter ID of the estimators available in model library or pass an untrained 
        model object consistent with fit / predict API to train and evaluate model. 
        All estimators support binary or multiclass problem. List of estimators in 
        model library (ID - Name):

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

    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in 
        setup(). If integer, will use StratifiedKFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.

    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    cross_validation: bool, default = True
        When cross_validation set to False fold parameter is ignored and model is 
        trained on entire training dataset, returning metrics calculated using the 
        train (holdout) set.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into 
        train/test set. If string is passed, will use the data column with that name 
        as the groups. Only used if a group based cross-validation generator is used 
        (eg. GroupKFold). If None, will use the value set in fold_groups param in 
        setup().

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

    return pycaret.internal.tabular.create_model_supervised(
        estimator=estimator,
        fold=fold,
        round=round,
        cross_validation=cross_validation,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
        **kwargs,
    )


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
    early_stopping: Any = False,
    early_stopping_max_iters: int = 10,
    choose_better: bool = False,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    return_tuner: bool = False,
    verbose: bool = True,
    tuner_verbose: Union[int, bool] = True,
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
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
    >>> xgboost = create_model('xgboost')
    >>> tuned_xgboost = tune_model(xgboost) 

    This will tune the hyperparameters of Extreme Gradient Boosting Classifier.


    Parameters
    ----------
    estimator : object, default = None

    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use StratifiedKFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.

    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    n_iter: int, default = 10
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

    early_stopping: bool or str or object, default = False
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

    return_tuner: bool, default = False
        If True, will reutrn a tuple of (model, tuner_object). Otherwise,
        will return just the best model.

    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.

    tuner_verbose: bool or in, default = True
        If True or above 0, will print messages from the tuner. Higher values
        print more messages. Ignored if verbose param is False.

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

    tuner_object
        Only if return_tuner param is True. The object used for tuning.

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

    return pycaret.internal.tabular.tune_model_supervised(
        estimator=estimator,
        fold=fold,
        round=round,
        n_iter=n_iter,
        custom_grid=custom_grid,
        optimize=optimize,
        custom_scorer=custom_scorer,
        search_library=search_library,
        search_algorithm=search_algorithm,
        early_stopping=early_stopping,
        early_stopping_max_iters=early_stopping_max_iters,
        choose_better=choose_better,
        fit_kwargs=fit_kwargs,
        groups=groups,
        return_tuner=return_tuner,
        verbose=verbose,
        tuner_verbose=tuner_verbose,
        **kwargs,
    )


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
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
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
    
    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use StratifiedKFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.
    
    n_estimators: int, default = 10
        The number of base estimators in the ensemble.
        In case of perfect fit, the learning procedure is stopped early.

    round: int, default = 4
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

    return pycaret.internal.tabular.ensemble_model(
        estimator=estimator,
        method=method,
        fold=fold,
        n_estimators=n_estimators,
        round=round,
        choose_better=choose_better,
        optimize=optimize,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
    )


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
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> rf = create_model('rf')
    >>> knn = create_model('knn')
    >>> blend_three = blend_models(estimator_list = [lr,rf,knn])

    This will create a VotingClassifier of lr, rf and knn.

    Parameters
    ----------
    estimator_list : list of objects

    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use StratifiedKFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.

    round: int, default = 4
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

    """

    return pycaret.internal.tabular.blend_models(
        estimator_list=estimator_list,
        fold=fold,
        round=round,
        choose_better=choose_better,
        optimize=optimize,
        method=method,
        weights=weights,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
    )


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
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
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

    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use StratifiedKFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.

    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.

    method: str, default = 'auto'

        - if 'auto', it will try to invoke, for each estimator, 'predict_proba',
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
    - If target variable is multiclass (more than 2 classes), AUC will be returned 
      as zero (0.0).

    """

    return pycaret.internal.tabular.stack_models(
        estimator_list=estimator_list,
        meta_model=meta_model,
        fold=fold,
        round=round,
        method=method,
        restack=restack,
        choose_better=choose_better,
        optimize=optimize,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
    )


def plot_model(
    estimator,
    plot: str = "auc",
    scale: float = 1,  # added in pycaret==2.1.0
    save: bool = False,
    fold: Optional[Union[int, Any]] = None,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
) -> str:

    """
    This function takes a trained model object and returns a plot based on the
    test / hold-out set. The process may require the model to be re-trained in
    certain cases. See list of plots supported below. 

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> plot_model(lr)

    This will return an AUC plot of a trained Logistic Regression model.

    Parameters
    ----------
    estimator : object, default = none
        A trained model object should be passed as an estimator. 

    plot : str, default = 'auc'
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
        * 'feature_all' - Feature Importance (All)
        * 'parameter' - Model Hyperparameter
        * 'lift' - Lift Curve
        * 'gain' - Gain Chart

    scale: float, default = 1
        The resolution scale of the figure.

    save: bool, default = False
        When set to True, Plot is saved as a 'png' file in current working directory.

    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation used in certain plots. If None, will use the CV generator
        defined in setup(). If integer, will use StratifiedKFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    groups: str or array-like, with shape (n_samples,), default = None
        Optional Group labels for the samples used while splitting the dataset into train/test set.
        If string is passed, will use the data column with that name as the groups.
        Only used if a group based cross-validation generator is used (eg. GroupKFold).
        If None, will use the value set in fold_groups param in setup().

    verbose: bool, default = True
        Progress bar not shown when verbose set to False.

    Returns
    -------
    Visual_Plot
        Prints the visual plot. 
    str:
        If save param is True, will return the name of the saved file.

    Warnings
    --------
    -  'svm' and 'ridge' doesn't support the predict_proba method. As such, AUC and 
        calibration plots are not available for these estimators.
       
    -   When the 'max_features' parameter of a trained model object is not equal to 
        the number of samples in training set, the 'rfe' plot is not available.
              
    -   'calibration', 'threshold', 'manifold' and 'rfe' plots are not available for
         multiclass problems.

    """

    return pycaret.internal.tabular.plot_model(
        estimator=estimator,
        plot=plot,
        scale=scale,
        save=save,
        fold=fold,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
        system=True,
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
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> evaluate_model(lr)
    
    This will display the User Interface for all of the plots for a given
    estimator.

    Parameters
    ----------
    estimator : object, default = none
        A trained model object should be passed as an estimator. 

    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use StratifiedKFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.

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

    return pycaret.internal.tabular.evaluate_model(
        estimator=estimator, fold=fold, fit_kwargs=fit_kwargs, groups=groups,
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
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
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

    observation: int, default = None
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

    return pycaret.internal.tabular.interpret_model(
        estimator=estimator,
        plot=plot,
        feature=feature,
        observation=observation,
        **kwargs,
    )


def calibrate_model(
    estimator,
    method: str = "sigmoid",
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
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
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
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

    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use StratifiedKFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.

    round: int, default = 4
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

    return pycaret.internal.tabular.calibrate_model(
        estimator=estimator,
        method=method,
        fold=fold,
        round=round,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
    )


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
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
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

    return pycaret.internal.tabular.optimize_threshold(
        estimator=estimator,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def predict_model(
    estimator,
    data: Optional[pd.DataFrame] = None,
    probability_threshold: Optional[float] = None,
    encoded_labels: bool = False,  # added in pycaret==2.1.0
    round: int = 4,  # added in pycaret==2.2.0
    verbose: bool = True,
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
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
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

    encoded_labels: bool, default = False
        If True, will return labels encoded as an integer.

    round: int, default = 4
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

    return pycaret.internal.tabular.predict_model(
        estimator=estimator,
        data=data,
        probability_threshold=probability_threshold,
        encoded_labels=encoded_labels,
        round=round,
        verbose=verbose,
        ml_usecase=MLUsecase.CLASSIFICATION,
    )


def finalize_model(
    estimator,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    model_only: bool = True,
) -> Any:  # added in pycaret==2.2.0

    """
    This function fits the estimator onto the complete dataset passed during the
    setup() stage. The purpose of this function is to prepare for final model
    deployment after experimentation. 
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
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

    model_only : bool, default = True
        When set to True, only trained model object is saved and all the 
        transformations are ignored.

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

    return pycaret.internal.tabular.finalize_model(
        estimator=estimator, fit_kwargs=fit_kwargs, groups=groups, model_only=model_only,
    )


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
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
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

    Before deploying a model to Google Cloud Platform (GCP), project must be created 
    either using command line or GCP console. Once project is created, you must create 
    a service account and download the service account key as a JSON file, which is 
    then used to set environment variable. 

    https://cloud.google.com/docs/authentication/production

    - Google Cloud Project
    - Service Account Authetication

    For Azure users:

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

    return pycaret.internal.tabular.deploy_model(
        model=model,
        model_name=model_name,
        authentication=authentication,
        platform=platform,
    )


def save_model(model, model_name: str, model_only: bool = False, verbose: bool = True):

    """
    This function saves the transformation pipeline and trained model object 
    into the current active directory as a pickle file for later use. 
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
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
    (model, model_filename):
        Tuple of the model object and the filename it was saved under.

    """

    return pycaret.internal.tabular.save_model(
        model=model, model_name=model_name, model_only=model_only, verbose=verbose
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

    return pycaret.internal.tabular.load_model(
        model_name=model_name,
        platform=platform,
        authentication=authentication,
        verbose=verbose,
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

    return pycaret.internal.tabular.automl(optimize=optimize, use_holdout=use_holdout)


def pull(pop: bool = False) -> pd.DataFrame:  # added in pycaret==2.2.0
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
    return pycaret.internal.tabular.pull(pop=pop)


def models(
    type: Optional[str] = None, internal: bool = False, raise_errors: bool = True,
) -> pd.DataFrame:

    """
    Returns table of models available in model library.

    Example
    -------
    >>> _all_models = models()

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

    raise_errors: bool, default = True
        If False, will suppress all exceptions, ignoring models
        that couldn't be created.

    Returns
    -------
    pandas.DataFrame

    """
    return pycaret.internal.tabular.models(
        type=type, internal=internal, raise_errors=raise_errors
    )


def get_metrics(
    reset: bool = False, include_custom: bool = True, raise_errors: bool = True,
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

    return pycaret.internal.tabular.get_metrics(
        reset=reset, include_custom=include_custom, raise_errors=raise_errors,
    )


def add_metric(
    id: str,
    name: str,
    score_func: type,
    target: str = "pred",
    greater_is_better: bool = True,
    multiclass: bool = True,
    **kwargs,
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
        Score function (or loss function) with signature ``score_func(y, y_pred, **kwargs)``.

    target: str, default = 'pred'
        The target of the score function.

        - 'pred' for the prediction table
        - 'pred_proba' for pred_proba
        - 'threshold' for decision_function or predict_proba

    greater_is_better: bool, default = True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    multiclass: bool, default = True
        Whether the metric supports multiclass problems.

    **kwargs:
        Arguments to be passed to score function.

    Returns
    -------
    pandas.Series
        The created row as Series.

    """

    return pycaret.internal.tabular.add_metric(
        id=id,
        name=name,
        score_func=score_func,
        target=target,
        greater_is_better=greater_is_better,
        multiclass=multiclass,
        **kwargs,
    )


def remove_metric(name_or_id: str):
    """
    Removes a metric used in all functions.

    Parameters
    ----------
    name_or_id: str
        Display name or ID of the metric.

    """
    return pycaret.internal.tabular.remove_metric(name_or_id=name_or_id)


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

    return pycaret.internal.tabular.get_logs(experiment_name=experiment_name, save=save)


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

    return pycaret.internal.tabular.get_config(variable=variable)


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

    return pycaret.internal.tabular.set_config(variable=variable, value=value)


def save_config(file_name: str):

    """
    This function is used to save all enviroment variables to file,
    allowing to later resume modeling without rerunning setup().

    Example
    -------
    >>> save_config('myvars.pkl') 

    This will save all enviroment variables to 'myvars.pkl'.

    """

    return pycaret.internal.tabular.save_config(file_name=file_name)


def load_config(file_name: str):

    """
    This function is used to load enviroment variables from file created with save_config(),
    allowing to later resume modeling without rerunning setup().


    Example
    -------
    >>> load_config('myvars.pkl') 

    This will load all enviroment variables from 'myvars.pkl'.

    """

    return pycaret.internal.tabular.load_config(file_name=file_name)
