# Module: Clustering
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT
# Release: PyCaret 2.2.0
# Last modified : 25/10/2020

import sys
import datetime, time
import warnings
from pycaret.internal.logging import get_logger
from pycaret.internal.Display import Display, is_in_colab, enable_colab
import pandas as pd
import numpy as np
import ipywidgets as ipw
from typing import List, Tuple, Any, Union, Optional, Dict
import pycaret.internal.tabular

from pycaret.internal.tabular import MLUsecase

warnings.filterwarnings("ignore")


def setup(
    data,
    preprocess: bool = True,
    imputation_type: str = "simple",
    iterative_imputation_iters: int = 5,
    categorical_features: Optional[List[str]] = None,
    categorical_imputation: str = "mode",
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
    remove_multicollinearity: bool = False,
    multicollinearity_threshold: float = 0.9,
    remove_perfect_collinearity: bool = False,
    group_features: Optional[List[str]] = None,
    group_names: Optional[List[str]] = None,
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
    pipeline. Setup function must be called before executing any other function. It
    takes one mandatory parameter: ``data``. All the other parameters are optional.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)


    data: pandas.DataFrame
        Shape (n_samples, n_features), where n_samples is the number of samples and 
        n_features is the number of features.


    preprocess: bool, default = True
        When set to False, no transformations are applied except for custom 
        transformations passed in ``custom_pipeline`` param. Data must be 
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


    group_features: list or list of list, default = None
        When the dataset contains features with related characteristics, group_features
        parameter can be used for feature extraction. It takes a list of strings with 
        column names that are related.

        
    group_names: list, default = None
        Group names to be used in naming new features. When the length of group_names 
        does not match with the length of ``group_features``, new features are named 
        sequentially group_1, group_2, etc. It is ignored when ``group_features`` is
        None.


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

        - Kmeans, DBSCAN, requires cuML >= 0.15 
          https://github.com/rapidsai/cuml


    custom_pipeline: (str, transformer) or list of (str, transformer), default = None
        When passed, will append the custom transformers in the preprocessing pipeline
        and are applied on each CV fold separately and on the final fit. All the custom
        transformations are applied before pycaret's internal transformations. 


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
        "cluster": "Cluster PCA Plot (2d)",
        "tsne": "Cluster TSnE (3d)",
        "elbow": "Elbow",
        "silhouette": "Silhouette",
        "distance": "Distance",
        "distribution": "Distribution",
    }

    if log_plots == True:
        log_plots = ["cluster", "distribution", "elbow"]

    return pycaret.internal.tabular.setup(
        ml_usecase="clustering",
        available_plots=available_plots,
        data=data,
        target=None,
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
        remove_outliers=False,
        remove_multicollinearity=remove_multicollinearity,
        multicollinearity_threshold=multicollinearity_threshold,
        remove_perfect_collinearity=remove_perfect_collinearity,
        create_clusters=False,
        polynomial_features=False,
        trigonometry_features=False,
        group_features=group_features,
        group_names=group_names,
        feature_selection=False,
        feature_interaction=False,
        feature_ratio=False,
        fix_imbalance=False,
        data_split_shuffle=False,
        data_split_stratify=False,
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


def create_model(
    model: Union[str, Any],
    num_clusters: int = 4,
    ground_truth: Optional[str] = None,
    round: int = 4,
    fit_kwargs: Optional[dict] = None,
    verbose: bool = True,
    **kwargs
):

    """
    This function trains and evaluates the performance of a given model.
    Metrics evaluated can be accessed using the ``get_metrics`` function. 
    Custom metrics can be added or removed using the ``add_metric`` and 
    ``remove_metric`` function. All the available models can be accessed 
    using the ``models`` function.
    
  
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')


    model: str or scikit-learn compatible object
        ID of an model available in the model library or pass an untrained 
        model object consistent with scikit-learn API. Models available  
        in the model library (ID - Name):

        * 'kmeans' - K-Means Clustering
        * 'ap' - Affinity Propagation
        * 'meanshift' - Mean shift Clustering
        * 'sc' - Spectral Clustering
        * 'hclust' - Agglomerative Clustering
        * 'dbscan' - Density-Based Spatial Clustering
        * 'optics' - OPTICS Clustering                               
        * 'birch' - Birch Clustering                                 
        * 'kmodes' - K-Modes Clustering                              


    num_clusters: int, default = 4
        The number of clusters to form.


    ground_truth: str, default = None
        ground_truth to be provided to evaluate metrics that require true labels.
        When None, such metrics are returned as 0.0. All metrics evaluated can 
        be accessed using ``get_metrics`` function.


    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    verbose: bool, default = True
        Status update is not printed when verbose is set to False.


    **kwargs: 
        Additional keyword arguments to pass to the estimator.


    Returns:
        Trained Model


    Warnings
    --------
    - ``num_clusters`` param not required for Affinity Propagation ('ap'), 
      Mean shift ('meanshift'), Density-Based Spatial Clustering ('dbscan')
      and OPTICS Clustering ('optics'). 
      
    - When fit doesn't converge in Affinity Propagation ('ap') model, all 
      datapoints are labelled as -1.
      
    - Noisy samples are given the label -1, when using Density-Based Spatial 
      ('dbscan') or OPTICS Clustering ('optics'). 
      
    - OPTICS ('optics') clustering may take longer training times on large 
      datasets.
    
       
    """

    return pycaret.internal.tabular.create_model_unsupervised(
        estimator=model,
        num_clusters=num_clusters,
        ground_truth=ground_truth,
        round=round,
        fit_kwargs=fit_kwargs,
        verbose=verbose,
        **kwargs,
    )


def assign_model(
    model, transformation: bool = False, verbose: bool = True
) -> pd.DataFrame:

    """
    This function assigns cluster labels to the dataset for a given model. 
    

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> kmeans_df = assign_model(kmeans)



    model: scikit-learn compatible object
        Trained model object
    

    transformation: bool, default = False
        Whether to apply cluster labels on the transformed dataset. 
    
    
    verbose: bool, default = True
        Status update is not printed when verbose is set to False.


    Returns:
        pandas.DataFrame
  
    """

    return pycaret.internal.tabular.assign_model(
        model, transformation=transformation, verbose=verbose
    )


def plot_model(
    model,
    plot: str = "cluster",
    feature: Optional[str] = None,
    label: bool = False,
    scale: float = 1,
    save: bool = False,
):

    """
    This function analyzes the performance of a trained model.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> plot_model(kmeans, plot = 'cluster')


    model: scikit-learn compatible object
        Trained Model Object

    
    plot: str, default = 'cluster'
        List of available plots (ID - Name):

        * 'cluster' - Cluster PCA Plot (2d)              
        * 'tsne' - Cluster TSnE (3d)
        * 'elbow' - Elbow Plot 
        * 'silhouette' - Silhouette Plot         
        * 'distance' - Distance Plot   
        * 'distribution' - Distribution Plot
    
    
    feature: str, default = None
        Feature to be evaluated when plot = 'distribution'. When ``plot`` type is 
        'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or 
        label when the ``label`` param is set to True. When the ``plot`` type is 
        'cluster' or 'tsne' and feature is None, first column of the dataset is
        used.
    

    label: bool, default = False
        Name of column to be used as data labels. Ignored when ``plot`` is not 
        'cluster' or 'tsne'. 


    scale: float, default = 1
        The resolution scale of the figure.


    save: bool, default = False
        When set to True, plot is saved in the current working directory.


    Returns:
        None

    """
    return pycaret.internal.tabular.plot_model(
        model, plot=plot, feature_name=feature, label=label, scale=scale, save=save
    )


def evaluate_model(
    model, feature: Optional[str] = None, fit_kwargs: Optional[dict] = None,
):

    """
    This function displays a user interface for analyzing performance of a trained
    model. It calls the ``plot_model`` function internally. 
    
    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> evaluate_model(kmeans)
    

    model: scikit-learn compatible object
        Trained model object


    feature: str, default = None
        Feature to be evaluated when plot = 'distribution'. When ``plot`` type is 
        'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or 
        label when the ``label`` param is set to True. When the ``plot`` type is 
        'cluster' or 'tsne' and feature is None, first column of the dataset is
        used.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    Returns:
        None


    Warnings
    --------
    -   This function only works in IPython enabled Notebook. 

    """

    return pycaret.internal.tabular.evaluate_model(
        estimator=model, feature_name=feature, fit_kwargs=fit_kwargs
    )


def tune_model(
    model,
    supervised_target: str,
    supervised_type: Optional[str] = None,
    supervised_estimator: Union[str, Any] = "lr",
    optimize: Optional[str] = None,
    custom_grid: Optional[List[int]] = None,
    fold: int = 10,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    round: int = 4,
    verbose: bool = True,
):

    """
    This function tunes the ``num_clusters`` parameter of a given model. 
    
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = juice)
    >>> tuned_kmeans = tune_model(model = 'kmeans', supervised_target = 'Purchase') 


    model: str
        ID of an model available in the model library. Models that can be
        tuned in this function (ID - Model):
        
        * 'kmeans' - K-Means Clustering
        * 'sc' - Spectral Clustering
        * 'hclust' - Agglomerative Clustering                        
        * 'birch' - Birch Clustering                                 
        * 'kmodes' - K-Modes Clustering    
    

    supervised_target: str
        Name of the target column containing labels.


    supervised_type: str, default = None
        Type of task. 'classification' or 'regression'. Automatically inferred
        when None.
        

    supervised_estimator: str, default = None
        Classification (ID - Name):
            * 'lr' - Logistic Regression (Default)           
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
        
        Regression (ID - Name):
            * 'lr' - Linear Regression (Default)                                
            * 'lasso' - Lasso Regression              
            * 'ridge' - Ridge Regression              
            * 'en' - Elastic Net                   
            * 'lar' - Least Angle Regression                
            * 'llar' - Lasso Least Angle Regression                     
            * 'omp' - Orthogonal Matching Pursuit                        
            * 'br' - Bayesian Ridge                                   
            * 'ard' - Automatic Relevance Determ.                     
            * 'par' - Passive Aggressive Regressor                      
            * 'ransac' - Random Sample Consensus              
            * 'tr' - TheilSen Regressor                               
            * 'huber' - Huber Regressor                                              
            * 'kr' - Kernel Ridge                                                       
            * 'svm' - Support Vector Machine                                   
            * 'knn' - K Neighbors Regressor                                    
            * 'dt' - Decision Tree                                                     
            * 'rf' - Random Forest                                                     
            * 'et' - Extra Trees Regressor                                     
            * 'ada' - AdaBoost Regressor                                               
            * 'gbr' - Gradient Boosting                                            
            * 'mlp' - Multi Level Perceptron                                  
            * 'xgboost' - Extreme Gradient Boosting                                   
            * 'lightgbm' - Light Gradient Boosting                           
            * 'catboost' - CatBoost Regressor                       
            

    optimize: str, default = None
        For Classification tasks:
            Accuracy, AUC, Recall, Precision, F1, Kappa (default = 'Accuracy')
        
        For Regression tasks:
            MAE, MSE, RMSE, R2, RMSLE, MAPE (default = 'R2')
            

    custom_grid: list, default = None
        By default, a pre-defined number of clusters is iterated over to 
        optimize the supervised objective. To overwrite default iteration,
        pass a list of num_clusters to iterate over in custom_grid param.
    

    fold: int, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2. 


    verbose: bool, default = True
        Status update is not printed when verbose is set to False.


    Returns:
        Trained Model with optimized ``num_clusters`` parameter.
    

    Warnings
    --------
    - Affinity Propagation, Mean shift, Density-Based Spatial Clustering
      and OPTICS Clustering cannot be used in this function since they donot 
      support the ``num_clusters`` param.

          
    """
    return pycaret.internal.tabular.tune_model_unsupervised(
        model=model,
        supervised_target=supervised_target,
        supervised_type=supervised_type,
        supervised_estimator=supervised_estimator,
        optimize=optimize,
        custom_grid=custom_grid,
        fold=fold,
        fit_kwargs=fit_kwargs,
        groups=groups,
        round=round,
        verbose=verbose,
    )


def predict_model(model, data: pd.DataFrame) -> pd.DataFrame:

    """
    This function generates cluster labels using a trained model.
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> kmeans_predictions = predict_model(model = kmeans, data = unseen_data)
        

    model: scikit-learn compatible object
        Trained Model Object.
    

    data : pandas.DataFrame
        Shape (n_samples, n_features) where n_samples is the number of samples and 
        n_features is the number of features.


    Returns:
        pandas.DataFrame


    Warnings
    --------
    - Models that do not support 'predict' method cannot be used in the ``predict_model``.

    - The behavior of the predict_model is changed in version 2.1 without backward compatibility.
      As such, the pipelines trained using the version (<= 2.0), may not work for inference 
      with version >= 2.1. You can either retrain your models with a newer version or downgrade
      the version for inference.
    

    """

    return pycaret.internal.tabular.predict_model_unsupervised(
        estimator=model, data=data, ml_usecase=MLUsecase.CLUSTERING,
    )


def deploy_model(
    model, model_name: str, authentication: dict, platform: str = "aws",
):

    """
    This function deploys the transformation pipeline and trained model on cloud.
    
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> deploy_model(model = kmeans, model_name = 'kmeans-for-deployment', platform = 'aws', authentication = {'bucket' : 'S3-bucket-name'})
        

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
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> save_model(lr, 'saved_kmeans_model')
    

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
    >>> from pycaret.clustering import load_model
    >>> saved_kmeans = load_model('saved_kmeans_model')
    

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


def models(internal: bool = False, raise_errors: bool = True) -> pd.DataFrame:

    """
    Returns table of models available in the model library.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> all_models = models()


    internal: bool, default = False
        If True, will return extra columns and rows used internally.


    raise_errors: bool, default = True
        If False, will suppress all exceptions, ignoring models
        that couldn't be created.


    Returns:
        pandas.DataFrame

    """

    return pycaret.internal.tabular.models(internal=internal, raise_errors=raise_errors)


def get_metrics(
    reset: bool = False, include_custom: bool = True, raise_errors: bool = True,
) -> pd.DataFrame:

    """
    Returns table of metrics available.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> all_metrics = get_metrics()


    reset: bool, default = False
        If True, will reset all changes made using add_metric() and get_metric().


    include_custom: bool, default = True
        Whether to include user added (custom) metrics or not.


    raise_errors: bool, default = True
        If False, will suppress all exceptions, ignoring models
        that couldn't be created.


    Returns:
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
    **kwargs
) -> pd.Series:
    """
    Adds a custom metric to be used in all functions.


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

    Returns:
        pandas.Series

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
    Removes a metric used for evaluation.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> remove_metric('cs')


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
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery,  log_experiment = True) 
    >>> kmeans = create_model('kmeans')
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
    - data_before_preprocess: data before preprocessing
    - seed: random state set through session_id
    - prep_pipe: Transformation pipeline configured through setup
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - create_model_container: results grid storage container
    - master_model_container: model storage container
    - display_container: results display container
    - exp_name_log: Name of experiment set through setup
    - logging_param: log_experiment param set through setup
    - log_plots_param: log_plots param set through setup
    - USI: Unique session ID parameter set through setup
    - gpu_param: use_gpu param configured through setup


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> X = get_config('X') 


    Returns:
        Global variable
    

    """

    return pycaret.internal.tabular.get_config(variable=variable)


def set_config(variable: str, value):

    """
    This function resets the global variables. Following variables are 
    accessible:

    - X: Transformed dataset (X)
    - data_before_preprocess: data before preprocessing
    - seed: random state set through session_id
    - prep_pipe: Transformation pipeline configured through setup
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - create_model_container: results grid storage container
    - master_model_container: model storage container
    - display_container: results display container
    - exp_name_log: Name of experiment set through setup
    - logging_param: log_experiment param set through setup
    - log_plots_param: log_plots param set through setup
    - USI: Unique session ID parameter set through setup
    - gpu_param: use_gpu param configured through setup


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
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
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
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
    >>> from pycaret.clustering import load_config
    >>> load_config('myvars.pkl') 
    

    Returns:
        Global variables

    """

    return pycaret.internal.tabular.load_config(file_name=file_name)


def get_clusters(
    data,
    model: Union[str, Any] = "kmeans",
    num_clusters: int = 4,
    ground_truth: Optional[str] = None,
    round: int = 4,
    fit_kwargs: Optional[dict] = None,
    preprocess: bool = True,
    imputation_type: str = "simple",
    iterative_imputation_iters: int = 5,
    categorical_features: Optional[List[str]] = None,
    categorical_imputation: str = "mode",
    categorical_iterative_imputer: Union[str, Any] = "lightgbm",
    ordinal_features: Optional[Dict[str, list]] = None,
    high_cardinality_features: Optional[List[str]] = None,
    high_cardinality_method: str = "frequency",
    numeric_features: Optional[List[str]] = None,
    numeric_imputation: str = "mean",  # method 'zero' added in pycaret==2.1
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
    remove_multicollinearity: bool = False,
    multicollinearity_threshold: float = 0.9,
    remove_perfect_collinearity: bool = False,
    group_features: Optional[List[str]] = None,
    group_names: Optional[List[str]] = None,
    n_jobs: Optional[int] = -1,
    session_id: Optional[int] = None,
    log_experiment: bool = False,
    experiment_name: Optional[str] = None,
    log_plots: Union[bool, list] = False,
    log_profile: bool = False,
    log_data: bool = False,
    profile: bool = False,
    **kwargs
) -> pd.DataFrame:

    """
    Callable from any external environment without requiring setup initialization.
    """
    setup(
        data=data,
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
        remove_multicollinearity=remove_multicollinearity,
        multicollinearity_threshold=multicollinearity_threshold,
        remove_perfect_collinearity=remove_perfect_collinearity,
        group_features=group_features,
        group_names=group_names,
        n_jobs=n_jobs,
        html=False,
        session_id=session_id,
        log_experiment=log_experiment,
        experiment_name=experiment_name,
        log_plots=log_plots,
        log_profile=log_profile,
        log_data=log_data,
        silent=True,
        verbose=False,
        profile=profile,
    )

    c = create_model(
        model=model,
        num_clusters=num_clusters,
        ground_truth=ground_truth,
        round=round,
        fit_kwargs=fit_kwargs,
        verbose=False,
        **kwargs,
    )
    dataset = assign_model(c, verbose=False)
    return dataset
