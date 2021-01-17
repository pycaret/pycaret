# Module: Regression
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT
# Release: PyCaret 2.2.0
# Last modified : 25/10/2020

import pandas as pd
import numpy as np

import pycaret.internal.tabular
from pycaret.internal.Display import Display, is_in_colab, enable_colab
from typing import List, Tuple, Any, Union, Optional, Dict
import warnings
from IPython.utils import io

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
    transform_target: bool = False,
    transform_target_method: str = "box-cox",
    data_split_shuffle: bool = True,
    data_split_stratify: Union[bool, List[str]] = False,
    fold_strategy: Union[str, Any] = "kfold",
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
    profile_kwargs: Dict[str, Any] = None,
):
    """
    This function initializes the training environment and creates the transformation 
    pipeline. Setup function must be called before executing any other function. It takes 
    two mandatory parameters: ``data`` and ``target``. All the other parameters are
    optional.

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')


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
    
        - minmax: scales and translates each feature individually such that it is in 
          the range of 0 - 1.
        - maxabs: scales and translates each feature individually such that the 
          maximal absolute value of each feature will be 1.0. It does not 
          shift/center the data, and thus does not destroy any sparsity.
        - robust: scales and translates each feature according to the Interquartile 
          range. When the dataset contains outliers, robust scaler often gives 
          better results.


    transformation: bool, default = False
        When set to True, it applies the power transform to make data more Gaussian-like.
        Type of transformation is defined by the ``transformation_method`` parameter.


    transformation_method: str, default = 'yeo-johnson'
        Defines the method for transformation. By default, the transformation method is 
        set to 'yeo-johnson'. The other available option for transformation is 'quantile'. 
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
        KMeans algorithm. Original values of the feature are then replaced by the
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
        When set to True, a subset of features are selected using a combination of 
        various permutation importance techniques including Random Forest, Adaboost 
        and Linear correlation with target variable. The size of the subset is 
        dependent on the ``feature_selection_threshold`` parameter. 


    feature_selection_threshold: float, default = 0.8
        Threshold value used for feature selection. When ``polynomial_features`` or 
        ``feature_interaction`` is True, it is recommended to keep the threshold low
        to avoid large feature spaces. Setting a very low value may be efficient but 
        could result in under-fitting.

    
    feature_selection_method: str, default = 'classic'
        Algorithm for feature selection. 'classic' method uses permutation feature
        importance techniques. Other possible value is 'boruta' which uses boruta
        algorithm for feature selection. 

    
    feature_interaction: bool, default = False 
        When set to True, new features are created by interacting (a * b) all the 
        numeric variables in the dataset. This feature is not scalable and may not
        work as expected on datasets with large feature space.

    
    feature_ratio: bool, default = False
        When set to True, new features are created by calculating the ratios (a / b) 
        between all numeric variables in the dataset. This feature is not scalable and 
        may not work as expected on datasets with large feature space.

    
    interaction_threshold: bool, default = 0.01
        Similar to polynomial_threshold, It is used to compress a sparse matrix of newly 
        created features through interaction. Features whose importance based on the 
        combination  of  Random Forest, AdaBoost and Linear correlation falls within the 
        percentile of the  defined threshold are kept in the dataset. Remaining features 
        are dropped before further processing.


    transform_target: bool, default = False
        When set to True, target variable is transformed using the method defined in
        ``transform_target_method`` param. Target transformation is applied separately
        from feature transformations. 


    transform_target_method: str, default = 'box-cox'
        'Box-cox' and 'yeo-johnson' methods are supported. Box-Cox requires input data to 
        be strictly positive, while Yeo-Johnson supports both positive or negative data.
        When transform_target_method is 'box-cox' and target variable contains negative
        values, method is internally forced to 'yeo-johnson' to avoid exceptions.
        

    data_split_shuffle: bool, default = True
        When set to False, prevents shuffling of rows during 'train_test_split'.


    data_split_stratify: bool or list, default = False
        Controls stratification during 'train_test_split'. When set to True, will 
        stratify by target column. To stratify on any other columns, pass a list of 
        column names. Ignored when ``data_split_shuffle`` is False.


    fold_strategy: str or sklearn CV generator object, default = 'kfold'
        Choice of cross validation strategy. Possible values are:

        * 'kfold'
        * 'stratifiedkfold'
        * 'groupkfold'
        * 'timeseries'
        * a custom CV generator object compatible with scikit-learn.


    fold: int, default = 10
        Number of folds to be used in cross validation. Must be at least 2. This is
        a global setting that can be over-written at function level by using ``fold``
        parameter. Ignored when ``fold_strategy`` is a custom object.


    fold_shuffle: bool, default = False
        Controls the shuffle parameter of CV. Only applicable when ``fold_strategy``
        is 'kfold' or 'stratifiedkfold'. Ignored when ``fold_strategy`` is a custom
        object.

    
    fold_groups: str or array-like, with shape (n_samples,), default = None
        Optional group labels when 'GroupKFold' is used for the cross validation.
        It takes an array with shape (n_samples, ) where n_samples is the number
        of rows in the training dataset. When string is passed, it is interpreted 
        as the column name in the dataset containing group labels.


    n_jobs: int, default = -1
        The number of jobs to run in parallel (for functions that supports parallel 
        processing) -1 means using all processors. To run all functions on single 
        processor set n_jobs to None.


    use_gpu: bool or str, default = False
        When set to True, it will use GPU for training with algorithms that support it, 
        and fall back to CPU if they are unavailable. When set to 'force', it will only
        use GPU-enabled algorithms and raise exceptions when they are unavailable. When 
        False, all algorithms are trained using CPU only.

        GPU enabled algorithms:
        
        - Extreme Gradient Boosting, requires no further installation

        - CatBoost Regressor, requires no further installation
          (GPU is only enabled when data > 50,000 rows)
        
        - Light Gradient Boosting Machine, requires GPU installation
          https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html

        - Linear Regression, Lasso Regression, Ridge Regression, K Neighbors Regressor,
          Random Forest, Support Vector Regression, Elastic Net requires cuML >= 0.15 
          https://github.com/rapidsai/cuml


    custom_pipeline: (str, transformer) or list of (str, transformer), default = None
        When passed, will append the custom transformers in the preprocessing pipeline
        and are applied on each CV fold separately and on the final fit. All the custom
        transformations are applied after 'train_test_split' and before pycaret's internal 
        transformations. 


    html: bool, default = True
        When set to False, prevents runtime display of monitor. This must be set to False
        when the environment does not support IPython. For example, command line terminal,
        Databricks Notebook, Spyder and other similar IDEs. 


    session_id: int, default = None
        Controls the randomness of experiment. It is equivalent to 'random_state' in
        scikit-learn. When None, a pseudo random number is generated. This can be used 
        for later reproducibility of the entire experiment.


    log_experiment: bool, default = False
        When set to True, all metrics and parameters are logged on the ``MLFlow`` server.


    experiment_name: str, default = None
        Name of the experiment for logging. Ignored when ``log_experiment`` is not True.


    log_plots: bool or list, default = False
        When set to True, certain plots are logged automatically in the ``MLFlow`` server. 
        To change the type of plots to be logged, pass a list containing plot IDs. Refer
        to documentation of ``plot_model``. Ignored when ``log_experiment`` is not True.


    log_profile: bool, default = False
        When set to True, data profile is logged on the ``MLflow`` server as a html file.
        Ignored when ``log_experiment`` is not True. 


    log_data: bool, default = False
        When set to True, dataset is logged on the ``MLflow`` server as a csv file.
        Ignored when ``log_experiment`` is not True.
        

    silent: bool, default = False
        Controls the confirmation input of data types when ``setup`` is executed. When
        executing in completely automated mode or on a remote kernel, this must be True.

    
    verbose: bool, default = True
        When set to False, Information grid is not printed.


    profile: bool, default = False
        When set to True, an interactive EDA report is displayed. 


    profile_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the ProfileReport method used
        to create the EDA report. Ignored if ``profile`` is False.


    Returns:
        Global variables that can be changed using the ``set_config`` function.
      
    """
    available_plots = {
        "parameter": "Hyperparameters",
        "residuals": "Residuals",
        "error": "Prediction Error",
        "cooks": "Cooks Distance",
        "rfe": "Feature Selection",
        "learning": "Learning Curve",
        "manifold": "Manifold Learning",
        "vc": "Validation Curve",
        "feature": "Feature Importance",
        "feature_all": "Feature Importance (All)",
        "tree": "Decision Tree",
    }

    if log_plots == True:
        log_plots = ["residuals", "error", "feature"]

    return pycaret.internal.tabular.setup(
        ml_usecase="regression",
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
        transform_target=transform_target,
        transform_target_method=transform_target_method,
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
        profile_kwargs=profile_kwargs,
    )


def compare_models(
    include: Optional[List[Union[str, Any]]] = None,
    exclude: Optional[List[str]] = None,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    sort: str = "R2",
    n_select: int = 1,
    budget_time: Optional[float] = None,
    turbo: bool = True,
    errors: str = "ignore",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
):

    """
    This function trains and evaluates performance of all estimators available in the 
    model library using cross validation. The output of this function is a score grid 
    with average cross validated scores. Metrics evaluated during CV can be accessed 
    using the ``get_metrics`` function. Custom metrics can be added or removed using 
    ``add_metric`` and ``remove_metric`` function.


    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> best_model = compare_models()


    include: list of str or scikit-learn compatible object, default = None
        To train and evaluate select models, list containing model ID or scikit-learn 
        compatible object can be passed in include param. To see a list of all models 
        available in the model library use the ``models`` function. 


    exclude: list of str, default = None
        To omit certain models from training and evaluation, pass a list containing 
        model id in the exclude parameter. To see a list of all models available
        in the model library use the ``models`` function. 


    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, the CV generator in the ``fold_strategy`` 
        parameter of the ``setup`` function is used. When an integer is passed, 
        it is interpreted as the 'n_splits' parameter of the CV generator in the 
        ``setup`` function.


    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.


    cross_validation: bool, default = True
        When set to False, metrics are evaluated on holdout set. ``fold`` param
        is ignored when cross_validation is set to False.


    sort: str, default = 'R2'
        The sort order of the score grid. It also accepts custom metrics that are
        added through the ``add_metric`` function.


    n_select: int, default = 1
        Number of top_n models to return. For example, to select top 3 models use
        n_select = 3.


    budget_time: int or float, default = None
        If not None, will terminate execution of the function after budget_time 
        minutes have passed and return results up to that point.


    turbo: bool, default = True
        When set to True, it excludes estimators with longer training times. To
        see which algorithms are excluded use the ``models`` function.


    errors: str, default = 'ignore'
        When set to 'ignore', will skip the model with exceptions and continue.
        If 'raise', will break the function when exceptions are raised.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    groups: str or array-like, with shape (n_samples,), default = None
        Optional group labels when 'GroupKFold' is used for the cross validation.
        It takes an array with shape (n_samples, ) where n_samples is the number
        of rows in the training dataset. When string is passed, it is interpreted 
        as the column name in the dataset containing group labels.


    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.
    
    
    Returns:
        Trained model or list of trained models, depending on the ``n_select`` param.


    Warnings
    --------
    - Changing turbo parameter to False may result in very high training times with 
      datasets exceeding 10,000 rows.

    - No models are logged in ``MLFlow`` when ``cross_validation`` parameter is False.

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
):

    """
    This function trains and evaluates the performance of a given estimator 
    using cross validation. The output of this function is a score grid with 
    CV scores by fold. Metrics evaluated during CV can be accessed using the 
    ``get_metrics`` function. Custom metrics can be added or removed using 
    ``add_metric`` and ``remove_metric`` function. All the available models
    can be accessed using the ``models`` function.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> lr = create_model('lr')
    

    estimator: str or scikit-learn compatible object
        ID of an estimator available in model library or pass an untrained 
        model object consistent with scikit-learn API. Estimators available  
        in the model library (ID - Name):

        * 'lr' - Linear Regression                   
        * 'lasso' - Lasso Regression                
        * 'ridge' - Ridge Regression                
        * 'en' - Elastic Net                   
        * 'lar' - Least Angle Regression                  
        * 'llar' - Lasso Least Angle Regression                   
        * 'omp' - Orthogonal Matching Pursuit                     
        * 'br' - Bayesian Ridge                   
        * 'ard' - Automatic Relevance Determination                  
        * 'par' - Passive Aggressive Regressor                    
        * 'ransac' - Random Sample Consensus       
        * 'tr' - TheilSen Regressor                   
        * 'huber' - Huber Regressor                               
        * 'kr' - Kernel Ridge                                     
        * 'svm' - Support Vector Regression                           
        * 'knn' - K Neighbors Regressor                           
        * 'dt' - Decision Tree Regressor                                   
        * 'rf' - Random Forest Regressor                                   
        * 'et' - Extra Trees Regressor                            
        * 'ada' - AdaBoost Regressor                              
        * 'gbr' - Gradient Boosting Regressor                               
        * 'mlp' - MLP Regressor
        * 'xgboost' - Extreme Gradient Boosting                   
        * 'lightgbm' - Light Gradient Boosting Machine                    
        * 'catboost' - CatBoost Regressor                         


    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, the CV generator in the ``fold_strategy`` 
        parameter of the ``setup`` function is used. When an integer is passed, 
        it is interpreted as the 'n_splits' parameter of the CV generator in the 
        ``setup`` function.
        

    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 


    cross_validation: bool, default = True
        When set to False, metrics are evaluated on holdout set. ``fold`` param
        is ignored when cross_validation is set to False.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    groups: str or array-like, with shape (n_samples,), default = None
        Optional group labels when GroupKFold is used for the cross validation.
        It takes an array with shape (n_samples, ) where n_samples is the number
        of rows in training dataset. When string is passed, it is interpreted as 
        the column name in the dataset containing group labels.


    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.


    **kwargs: 
        Additional keyword arguments to pass to the estimator.


    Returns:
        Trained Model


    Warnings
    --------
    - Models are not logged on the ``MLFlow`` server when ``cross_validation`` param
      is set to False.
      
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
    optimize: str = "R2",
    custom_scorer=None,
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
):

    """
    This function tunes the hyperparameters of a given estimator. The output of
    this function is a score grid with CV scores by fold of the best selected 
    model based on ``optimize`` parameter. Metrics evaluated during CV can be 
    accessed using the ``get_metrics`` function. Custom metrics can be added
    or removed using ``add_metric`` and ``remove_metric`` function. 


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> lr = create_model('lr')
    >>> tuned_lr = tune_model(lr) 


    estimator: scikit-learn compatible object
        Trained model object


    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, the CV generator in the ``fold_strategy`` 
        parameter of the ``setup`` function is used. When an integer is passed, 
        it is interpreted as the 'n_splits' parameter of the CV generator in the 
        ``setup`` function.
        

    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 


    n_iter: int, default = 10
        Number of iterations in the grid search. Increasing 'n_iter' may improve 
        model performance but also increases the training time.


    custom_grid: dictionary, default = None
        To define custom search space for hyperparameters, pass a dictionary with 
        parameter name and values to be iterated. Custom grids must be in a format 
        supported by the defined ``search_library``.


    optimize: str, default = 'R2'
        Metric name to be evaluated for hyperparameter tuning. It also accepts custom 
        metrics that are added through the ``add_metric`` function.


    custom_scorer: object, default = None
        custom scoring strategy can be passed to tune hyperparameters of the model. 
        It must be created using ``sklearn.make_scorer``. It is equivalent of adding
        custom metric using the ``add_metric`` function and passing the name of the
        custom metric in the ``optimize`` parameter. 
        Will be deprecated in future.


    search_library: str, default = 'scikit-learn'
        The search library used for tuning hyperparameters. Possible values:

        - 'scikit-learn' - default, requires no further installation
            https://github.com/scikit-learn/scikit-learn

        - 'scikit-optimize' - ``pip install scikit-optimize`` 
            https://scikit-optimize.github.io/stable/

        - 'tune-sklearn' - ``pip install tune-sklearn ray[tune]`` 
            https://github.com/ray-project/tune-sklearn

        - 'optuna' - ``pip install optuna`` 
            https://optuna.org/


    search_algorithm: str, default = None
        The search algorithm depends on the ``search_library`` parameter.
        Some search algorithms require additional libraries to be installed.
        If None, will use search library-specific default algorithm.

        - 'scikit-learn' possible values:
            - 'random' : random grid search (default)
            - 'grid' : grid search

        - 'scikit-optimize' possible values:
            - 'bayesian' : Bayesian search (default)

        - 'tune-sklearn' possible values:
            - 'random' : random grid search (default)
            - 'grid' : grid search
            - 'bayesian' : ``pip install scikit-optimize``
            - 'hyperopt' : ``pip install hyperopt``
            - 'optuna' : ``pip install optuna``
            - 'bohb' : ``pip install hpbandster ConfigSpace``

        - 'optuna' possible values:
            - 'random' : randomized search
            - 'tpe' : Tree-structured Parzen Estimator search (default)


    early_stopping: bool or str or object, default = False
        Use early stopping to stop fitting to a hyperparameter configuration 
        if it performs poorly. Ignored when ``search_library`` is scikit-learn, 
        or if the estimator does not have 'partial_fit' attribute. If False or 
        None, early stopping will not be used. Can be either an object accepted 
        by the search library or one of the following:

        - 'asha' for Asynchronous Successive Halving Algorithm
        - 'hyperband' for Hyperband
        - 'median' for Median Stopping Rule
        - If False or None, early stopping will not be used.


    early_stopping_max_iters: int, default = 10
        Maximum number of epochs to run for each sampled configuration.
        Ignored if ``early_stopping`` is False or None.


    choose_better: bool, default = False
        When set to True, the returned object is always better performing. The
        metric used for comparison is defined by the ``optimize`` parameter.  


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the tuner.


    groups: str or array-like, with shape (n_samples,), default = None
        Optional group labels when GroupKFold is used for the cross validation.
        It takes an array with shape (n_samples, ) where n_samples is the number
        of rows in training dataset. When string is passed, it is interpreted as 
        the column name in the dataset containing group labels.


    return_tuner: bool, default = False
        When set to True, will return a tuple of (model, tuner_object). 


    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.


    tuner_verbose: bool or in, default = True
        If True or above 0, will print messages from the tuner. Higher values
        print more messages. Ignored when ``verbose`` param is False.


    **kwargs: 
        Additional keyword arguments to pass to the optimizer.


    Returns:
        Trained Model and Optional Tuner Object when ``return_tuner`` is True. 


    Warnings
    --------
    - Using 'grid' as ``search_algorithm`` may result in very long computation.
      Only recommended with smaller search spaces that can be defined in the
      ``custom_grid`` parameter.

    - ``search_library`` 'tune-sklearn' does not support GPU models.

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
    optimize: str = "R2",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
) -> Any:

    """
    This function ensembles a given estimator. The output of this function is 
    a score grid with CV scores by fold. Metrics evaluated during CV can be 
    accessed using the ``get_metrics`` function. Custom metrics can be added
    or removed using ``add_metric`` and ``remove_metric`` function. 


    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> dt = create_model('dt')
    >>> bagged_dt = ensemble_model(dt, method = 'Bagging')


   estimator: scikit-learn compatible object
        Trained model object


    method: str, default = 'Bagging'
        Method for ensembling base estimator. It can be 'Bagging' or 'Boosting'. 


    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, the CV generator in the ``fold_strategy`` 
        parameter of the ``setup`` function is used. When an integer is passed, 
        it is interpreted as the 'n_splits' parameter of the CV generator in the 
        ``setup`` function.
        

    n_estimators: int, default = 10
        The number of base estimators in the ensemble. In case of perfect fit, the 
        learning procedure is stopped early.

        
    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 


    choose_better: bool, default = False
        When set to True, the returned object is always better performing. The
        metric used for comparison is defined by the ``optimize`` parameter. 


    optimize: str, default = 'R2'
        Metric to compare for model selection when ``choose_better`` is True.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    groups: str or array-like, with shape (n_samples,), default = None
        Optional group labels when GroupKFold is used for the cross validation.
        It takes an array with shape (n_samples, ) where n_samples is the number
        of rows in training dataset. When string is passed, it is interpreted as 
        the column name in the dataset containing group labels.


    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.


    Returns:
        Trained Model
      
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
    optimize: str = "R2",
    weights: Optional[List[float]] = None,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
):

    """
    This function trains a Voting Regressor for select models passed in the 
    ``estimator_list`` param. The output of this function is a score grid with 
    CV scores by fold. Metrics evaluated during CV can be accessed using the 
    ``get_metrics`` function. Custom metrics can be added or removed using 
    ``add_metric`` and ``remove_metric`` function.

    
    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> top3 = compare_models(n_select = 3)
    >>> blender = blend_models(top3)


    estimator_list: list of scikit-learn compatible objects
        List of trained model objects


    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, the CV generator in the ``fold_strategy`` 
        parameter of the ``setup`` function is used. When an integer is passed, 
        it is interpreted as the 'n_splits' parameter of the CV generator in the 
        ``setup`` function.


    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.


    choose_better: bool, default = False
        When set to True, the returned object is always better performing. The
        metric used for comparison is defined by the ``optimize`` parameter. 


    optimize: str, default = 'R2'
        Metric to compare for model selection when ``choose_better`` is True.


    weights: list, default = None
        Sequence of weights (float or int) to weight the occurrences of predicted class 
        labels (hard voting) or class probabilities before averaging (soft voting). Uses 
        uniform weights when None.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    groups: str or array-like, with shape (n_samples,), default = None
        Optional group labels when GroupKFold is used for the cross validation.
        It takes an array with shape (n_samples, ) where n_samples is the number
        of rows in training dataset. When string is passed, it is interpreted as 
        the column name in the dataset containing group labels.


    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.


    Returns:
        Trained Model
       
  
    """

    return pycaret.internal.tabular.blend_models(
        estimator_list=estimator_list,
        fold=fold,
        round=round,
        choose_better=choose_better,
        optimize=optimize,
        method="auto",
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
    restack: bool = True,
    choose_better: bool = False,
    optimize: str = "R2",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
):

    """
    This function trains a meta model over select estimators passed in 
    the ``estimator_list`` parameter. The output of this function is a 
    score grid with CV scores by fold. Metrics evaluated during CV can 
    be accessed using the ``get_metrics`` function. Custom metrics 
    can be added or removed using ``add_metric`` and ``remove_metric`` 
    function.


    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> top3 = compare_models(n_select = 3)
    >>> stacker = stack_models(top3)


    estimator_list: list of scikit-learn compatible objects
        List of trained model objects


    meta_model: scikit-learn compatible object, default = None
        When None, Linear Regression is trained as a meta model.


    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, the CV generator in the ``fold_strategy`` 
        parameter of the ``setup`` function is used. When an integer is passed, 
        it is interpreted as the 'n_splits' parameter of the CV generator in the 
        ``setup`` function.


    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.


    restack: bool, default = True
        When set to False, only the predictions of estimators will be used as 
        training data for the ``meta_model``.


    choose_better: bool, default = False
        When set to True, the returned object is always better performing. The
        metric used for comparison is defined by the ``optimize`` parameter. 


    optimize: str, default = 'R2'
        Metric to compare for model selection when ``choose_better`` is True.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    groups: str or array-like, with shape (n_samples,), default = None
        Optional group labels when GroupKFold is used for the cross validation.
        It takes an array with shape (n_samples, ) where n_samples is the number
        of rows in training dataset. When string is passed, it is interpreted as 
        the column name in the dataset containing group labels.


    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.


    Returns:
        Trained Model

    """

    return pycaret.internal.tabular.stack_models(
        estimator_list=estimator_list,
        meta_model=meta_model,
        fold=fold,
        round=round,
        method="auto",
        restack=restack,
        choose_better=choose_better,
        optimize=optimize,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
    )


def plot_model(
    estimator,
    plot: str = "residuals",
    scale: float = 1,
    save: bool = False,
    fold: Optional[Union[int, Any]] = None,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    use_train_data: bool = False,
    verbose: bool = True,
    display_format: Optional[str] = None
) -> str:

    """
    This function analyzes the performance of a trained model on holdout set. 
    It may require re-training the model in certain cases.


    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> lr = create_model('lr')
    >>> plot_model(lr, plot = 'residual')


    estimator: scikit-learn compatible object
        Trained model object
   

    plot: str, default = 'residual'
        List of available plots (ID - Name):

        * 'residuals' - Residuals Plot
        * 'error' - Prediction Error Plot
        * 'cooks' - Cooks Distance Plot
        * 'rfe' - Recursive Feat. Selection
        * 'learning' - Learning Curve
        * 'vc' - Validation Curve
        * 'manifold' - Manifold Learning
        * 'feature' - Feature Importance
        * 'feature_all' - Feature Importance (All)
        * 'parameter' - Model Hyperparameter
        * 'tree' - Decision Tree


    scale: float, default = 1
        The resolution scale of the figure.


    save: bool, default = False
        When set to True, plot is saved in the current working directory.


    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, the CV generator in the ``fold_strategy`` 
        parameter of the ``setup`` function is used. When an integer is passed, 
        it is interpreted as the 'n_splits' parameter of the CV generator in the 
        ``setup`` function.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    groups: str or array-like, with shape (n_samples,), default = None
        Optional group labels when GroupKFold is used for the cross validation.
        It takes an array with shape (n_samples, ) where n_samples is the number
        of rows in training dataset. When string is passed, it is interpreted as 
        the column name in the dataset containing group labels.


    use_train_data: bool, default = False
        When set to true, train data will be used for plots, instead
        of test data.


    verbose: bool, default = True
        When set to False, progress bar is not displayed.


    display_format: str, default = None
        To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.
        Currently, not all plots are supported.


    Returns:
        None

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
        use_train_data=use_train_data,
        system=True,
        display_format=display_format
    )


def evaluate_model(
    estimator,
    fold: Optional[Union[int, Any]] = None,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    use_train_data: bool = False,
):

    """
    This function displays a user interface for analyzing performance of a trained
    model. It calls the ``plot_model`` function internally. 
    
    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> lr = create_model('lr')
    >>> evaluate_model(lr)
    

    estimator: scikit-learn compatible object
        Trained model object


    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, the CV generator in the ``fold_strategy`` 
        parameter of the ``setup`` function is used. When an integer is passed, 
        it is interpreted as the 'n_splits' parameter of the CV generator in the 
        ``setup`` function.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    groups: str or array-like, with shape (n_samples,), default = None
        Optional group labels when GroupKFold is used for the cross validation.
        It takes an array with shape (n_samples, ) where n_samples is the number
        of rows in training dataset. When string is passed, it is interpreted as 
        the column name in the dataset containing group labels.


    use_train_data: bool, default = False
        When set to true, train data will be used for plots, instead
        of test data.


    Returns:
        None


    Warnings
    --------
    -   This function only works in IPython enabled Notebook.

    """

    return pycaret.internal.tabular.evaluate_model(
        estimator=estimator,
        fold=fold,
        fit_kwargs=fit_kwargs,
        groups=groups,
        use_train_data=use_train_data,
    )


def interpret_model(
    estimator,
    plot: str = "summary",
    feature: Optional[str] = None,
    observation: Optional[int] = None,
    use_train_data: bool = False,
    save: bool = False,
    **kwargs,
):

    """
    This function analyzes the predictions generated from a tree-based model. It is
    implemented based on the SHAP (SHapley Additive exPlanations). For more info on
    this, please see https://shap.readthedocs.io/en/latest/


    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp = setup(data = boston,  target = 'medv')
    >>> xgboost = create_model('xgboost')
    >>> interpret_model(xgboost)

 
    estimator: scikit-learn compatible object
        Trained model object


    plot: str, default = 'summary'
        Type of plot. Available options are: 'summary', 'correlation', and 'reason'.


    feature: str, default = None
        Feature to check correlation with. This parameter is only required when ``plot``
        type is 'correlation'. When set to None, it uses the first column in the train
        dataset.


    observation: int, default = None
        Observation index number in holdout set to explain. When ``plot`` is not
        'reason', this parameter is ignored. 


    use_train_data: bool, default = False
        When set to true, train data will be used for plots, instead
        of test data.


    save: bool, default = False
        When set to True, Plot is saved as a 'png' file in current working directory.


    **kwargs:
        Additional keyword arguments to pass to the plot.


    Returns:
        None

    """

    return pycaret.internal.tabular.interpret_model(
        estimator=estimator,
        plot=plot,
        feature=feature,
        observation=observation,
        use_train_data=use_train_data,
        save=save,
        **kwargs,
    )


def predict_model(
    estimator,
    data: Optional[pd.DataFrame] = None,
    round: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:

    """
    This function predicts ``Label`` using a trained model. When ``data`` is 
    None, it predicts label on the holdout set.
    

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> lr = create_model('lr')
    >>> pred_holdout = predict_model(lr)
    >>> pred_unseen = predict_model(lr, data = unseen_dataframe)


    estimator: scikit-learn compatible object
        Trained model object


    data : pandas.DataFrame
        Shape (n_samples, n_features). All features used during training 
        must be available in the unseen dataset.
        
    
    round: int, default = 4
        Number of decimal places to round predictions to.


    verbose: bool, default = True
        When set to False, holdout score grid is not printed.


    Returns:
        pandas.DataFrame


    Warnings
    --------
    - The behavior of the ``predict_model`` is changed in version 2.1 without backward 
      compatibility. As such, the pipelines trained using the version (<= 2.0), may not 
      work for inference with version >= 2.1. You can either retrain your models with a 
      newer version or downgrade the version for inference.
    
    
    """

    return pycaret.internal.tabular.predict_model(
        estimator=estimator,
        data=data,
        probability_threshold=None,
        encoded_labels=True,
        round=round,
        verbose=verbose,
        ml_usecase=MLUsecase.REGRESSION,
    )


def finalize_model(
    estimator,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    model_only: bool = True,
) -> Any:

    """
    This function trains a given estimator on the entire dataset including the 
    holdout set.

    
    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> lr = create_model('lr')
    >>> final_lr = finalize_model(lr)


    estimator: scikit-learn compatible object
        Trained model object


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    groups: str or array-like, with shape (n_samples,), default = None
        Optional group labels when GroupKFold is used for the cross validation.
        It takes an array with shape (n_samples, ) where n_samples is the number
        of rows in training dataset. When string is passed, it is interpreted as 
        the column name in the dataset containing group labels.


    model_only: bool, default = True
        When set to False, only model object is re-trained and all the 
        transformations in Pipeline are ignored.


    Returns:
        Trained Model
       
         
    """

    return pycaret.internal.tabular.finalize_model(
        estimator=estimator,
        fit_kwargs=fit_kwargs,
        groups=groups,
        model_only=model_only,
    )


def deploy_model(
    model, model_name: str, authentication: dict, platform: str = "aws",
):

    """
    This function deploys the transformation pipeline and trained model on cloud.
    
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> lr = create_model('lr')
    >>> deploy_model(model = lr, model_name = 'lr-for-deployment', platform = 'aws', authentication = {'bucket' : 'S3-bucket-name'})
        

    Amazon Web Service (AWS) users:
        To deploy a model on AWS S3 ('aws'), environment variables must be set in your
        local environment. To configure AWS environment variables, type ``aws configure`` 
        in the command line. Following information from the IAM portal of amazon console 
        account is required:

        - AWS Access Key ID
        - AWS Secret Key Access
        - Default Region Name (can be seen under Global settings on your AWS console)

        More info: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html


    Google Cloud Platform (GCP) users:
        To deploy a model on Google Cloud Platform ('gcp'), project must be created 
        using command line or GCP console. Once project is created, you must create 
        a service account and download the service account key as a JSON file to set 
        environment variables in your local environment. 

        More info: https://cloud.google.com/docs/authentication/production

    
    Microsoft Azure (Azure) users:
        To deploy a model on Microsoft Azure ('azure'), environment variables for connection
        string must be set in your local environment. Go to settings of storage account on
        Azure portal to access the connection string required. 

        More info: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json


    model: scikit-learn compatible object
        Trained model object
    

    model_name: str
        Name of model.
    

    authentication: dict
        Dictionary of applicable authentication tokens.

        When platform = 'aws':
        {'bucket' : 'S3-bucket-name'}

        When platform = 'gcp':
        {'project': 'gcp-project-name', 'bucket' : 'gcp-bucket-name'}

        When platform = 'azure':
        {'container': 'azure-container-name'}
    

    platform: str, default = 'aws'
        Name of the platform. Currently supported platforms: 'aws', 'gcp' and 'azure'.
    

    Returns:
        None
    
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
    into the current working directory as a pickle file for later use. 
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> lr = create_model('lr')
    >>> save_model(lr, 'saved_lr_model')
    

    model: scikit-learn compatible object
        Trained model object
    

    model_name: str
        Name of the model.
    

    model_only: bool, default = False
        When set to True, only trained model object is saved instead of the 
        entire pipeline.


    verbose: bool, default = True
        Success message is not printed when verbose is set to False.


    Returns:
        Tuple of the model object and the filename.

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
    This function loads a previously saved pipeline.
    
    Example
    -------
    >>> from pycaret.regression import load_model
    >>> saved_lr = load_model('saved_lr_model')
    

    model_name: str
        Name of the model.
      

    platform: str, default = None
        Name of the cloud platform. Currently supported platforms: 
        'aws', 'gcp' and 'azure'.
    

    authentication: dict, default = None
        dictionary of applicable authentication tokens.

        when platform = 'aws':
        {'bucket' : 'S3-bucket-name'}

        when platform = 'gcp':
        {'project': 'gcp-project-name', 'bucket' : 'gcp-bucket-name'}

        when platform = 'azure':
        {'container': 'azure-container-name'}
    

    verbose: bool, default = True
        Success message is not printed when verbose is set to False.


    Returns:
        Trained Model

    """

    return pycaret.internal.tabular.load_model(
        model_name=model_name,
        platform=platform,
        authentication=authentication,
        verbose=verbose,
    )


def automl(optimize: str = "R2", use_holdout: bool = False) -> Any:

    """
    This function returns the best model out of all trained models in
    current session based on the ``optimize`` parameter. Metrics
    evaluated can be accessed using the ``get_metrics`` function. 


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> top3 = compare_models(n_select = 3)
    >>> tuned_top3 = [tune_model(i) for i in top3]
    >>> blender = blend_models(tuned_top3)
    >>> stacker = stack_models(tuned_top3)
    >>> best_mae_model = automl(optimize = 'MAE')


    optimize: str, default = 'R2'
        Metric to use for model selection. It also accepts custom metrics
        added using the ``add_metric`` function. 


    use_holdout: bool, default = False
        When set to True, metrics are evaluated on holdout set instead of CV.
      

    Returns:
        Trained Model


    """

    return pycaret.internal.tabular.automl(optimize=optimize, use_holdout=use_holdout)


def pull(pop: bool = False) -> pd.DataFrame:
    """
    Returns last printed score grid. Use ``pull`` function after
    any training function to store the score grid in pandas.DataFrame.


    pop: bool, default = False
        If True, will pop (remove) the returned dataframe from the
        display container.


    Returns:
        pandas.DataFrame

    """
    return pycaret.internal.tabular.pull(pop=pop)


def models(
    type: Optional[str] = None, internal: bool = False, raise_errors: bool = True,
) -> pd.DataFrame:

    """
    Returns table of models available in the model library.

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')    
    >>> all_models = models()


    type: str, default = None
        - linear : filters and only return linear models
        - tree : filters and only return tree based models
        - ensemble : filters and only return ensemble models
    

    internal: bool, default = False
        When True, will return extra columns and rows used internally.


    raise_errors: bool, default = True
        When False, will suppress all exceptions, ignoring models
        that couldn't be created.


    Returns:
        pandas.DataFrame

    """
    return pycaret.internal.tabular.models(
        type=type, internal=internal, raise_errors=raise_errors
    )


def get_metrics(
    reset: bool = False, include_custom: bool = True, raise_errors: bool = True,
) -> pd.DataFrame:

    """
    Returns table of available metrics used for CV.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')    
    >>> all_metrics = get_metrics()


    reset: bool, default = False
        When True, will reset all changes made using the ``add_metric`` 
        and ``remove_metric`` function.


    include_custom: bool, default = True
        Whether to include user added (custom) metrics or not.


    raise_errors: bool, default = True
        If False, will suppress all exceptions, ignoring models that
        couldn't be created.


    Returns:
        pandas.DataFrame

    """

    return pycaret.internal.tabular.get_metrics(
        reset=reset, include_custom=include_custom, raise_errors=raise_errors,
    )


def add_metric(
    id: str, name: str, score_func: type, greater_is_better: bool = True, **kwargs,
) -> pd.Series:

    """
    Adds a custom metric to be used for CV.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv') 
    >>> from sklearn.metrics import explained_variance_score
    >>> add_metric('evs', 'EVS', explained_variance_score)


    id: str
        Unique id for the metric.


    name: str
        Display name of the metric.


    score_func: type
        Score function (or loss function) with signature ``score_func(y, y_pred, **kwargs)``.


    greater_is_better: bool, default = True
        Whether ``score_func`` is higher the better or not.


    **kwargs:
        Arguments to be passed to score function.


    Returns:
        pandas.Series

    """

    return pycaret.internal.tabular.add_metric(
        id=id,
        name=name,
        score_func=score_func,
        target="pred",
        greater_is_better=greater_is_better,
        **kwargs,
    )


def remove_metric(name_or_id: str):

    """
    Removes a metric from CV.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'mredv') 
    >>> remove_metric('MAPE')


    name_or_id: str
        Display name or ID of the metric.

    
    Returns:
        None

    """
    return pycaret.internal.tabular.remove_metric(name_or_id=name_or_id)


def get_logs(experiment_name: Optional[str] = None, save: bool = False) -> pd.DataFrame:

    """
    Returns a table of experiment logs. Only works when ``log_experiment``
    is True when initializing the ``setup`` function.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv', log_experiment = True) 
    >>> best = compare_models()
    >>> exp_logs = get_logs()


    experiment_name: str, default = None
        When None current active run is used.


    save: bool, default = False
        When set to True, csv file is saved in current working directory.


    Returns:
        pandas.DataFrame

    """

    return pycaret.internal.tabular.get_logs(experiment_name=experiment_name, save=save)


def get_config(variable: str):

    """
    This function retrieves the global variables created when initializing the 
    ``setup`` function. Following variables are accessible:

    - X: Transformed dataset (X)
    - y: Transformed dataset (y)  
    - X_train: Transformed train dataset (X)
    - X_test: Transformed test/holdout dataset (X)
    - y_train: Transformed train dataset (y)
    - y_test: Transformed test/holdout dataset (y)
    - seed: random state set through session_id
    - prep_pipe: Transformation pipeline
    - fold_shuffle_param: shuffle parameter used in Kfolds
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - create_model_container: results grid storage container
    - master_model_container: model storage container
    - display_container: results display container
    - exp_name_log: Name of experiment
    - logging_param: log_experiment param
    - log_plots_param: log_plots param
    - USI: Unique session ID parameter
    - fix_imbalance_param: fix_imbalance param
    - fix_imbalance_method_param: fix_imbalance_method param
    - data_before_preprocess: data before preprocessing
    - target_param: name of target variable
    - gpu_param: use_gpu param configured through setup
    - fold_generator: CV splitter configured in fold_strategy
    - fold_param: fold params defined in the setup
    - fold_groups_param: fold groups defined in the setup
    - stratify_param: stratify parameter defined in the setup
    - transform_target_param: transform_target_param in setup
    - transform_target_method_param: transform_target_method_param in setup


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv') 
    >>> X_train = get_config('X_train') 


    Returns:
        Global variable


    """

    return pycaret.internal.tabular.get_config(variable=variable)


def set_config(variable: str, value):

    """
    This function resets the global variables. Following variables are 
    accessible:

    - X: Transformed dataset (X)
    - y: Transformed dataset (y)  
    - X_train: Transformed train dataset (X)
    - X_test: Transformed test/holdout dataset (X)
    - y_train: Transformed train dataset (y)
    - y_test: Transformed test/holdout dataset (y)
    - seed: random state set through session_id
    - prep_pipe: Transformation pipeline
    - fold_shuffle_param: shuffle parameter used in Kfolds
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - create_model_container: results grid storage container
    - master_model_container: model storage container
    - display_container: results display container
    - exp_name_log: Name of experiment
    - logging_param: log_experiment param
    - log_plots_param: log_plots param
    - USI: Unique session ID parameter
    - fix_imbalance_param: fix_imbalance param
    - fix_imbalance_method_param: fix_imbalance_method param
    - data_before_preprocess: data before preprocessing
    - target_param: name of target variable
    - gpu_param: use_gpu param configured through setup
    - fold_generator: CV splitter configured in fold_strategy
    - fold_param: fold params defined in the setup
    - fold_groups_param: fold groups defined in the setup
    - stratify_param: stratify parameter defined in the setup
    - transform_target_param: transform_target_param in setup
    - transform_target_method_param: transform_target_method_param in setup


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv') 
    >>> set_config('seed', 123) 


    Returns:
        None

    """

    return pycaret.internal.tabular.set_config(variable=variable, value=value)


def save_config(file_name: str):

    """
    This function save all global variables to a pickle file, allowing to
    later resume without rerunning the ``setup``.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> from pycaret.regression import *
    >>> exp_name = setup(data = boston,  target = 'medv')
    >>> save_config('myvars.pkl') 


    Returns:
        None

    """

    return pycaret.internal.tabular.save_config(file_name=file_name)


def load_config(file_name: str):

    """
    This function loads global variables from a pickle file into Python
    environment.


    Example
    -------
    >>> from pycaret.regression import load_config
    >>> load_config('myvars.pkl') 


    Returns:
        Global variables

    """

    return pycaret.internal.tabular.load_config(file_name=file_name)
