# Module: Anomaly Detection
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT
# Release: PyCaret 2.1.1
# Last modified : 29/08/2020

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
    pipeline. Setup function must be called before executing any other function. It
    takes one mandatory parameter: ``data``. All the other parameters are optional.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)


    data: pandas.DataFrame
        Shape (n_samples, n_features), where n_samples is the number of samples and 
        n_features is the number of features.


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


    use_gpu: str or bool, default = False
        When set to 'force', will try to use GPU with all algorithms that support it,
        and raise exceptions if they are unavailable. When set to True, will use GPU 
        with algorithms that support it, and fall back to CPU if they are unavailable.
        When False, all algorithms are trained using CPU only.

        GPU enabled algorithms:

        - None at this moment. 


    custom_pipeline: transformer or list of transformers or tuple
        (str, transformer) or list of tuples (str, transformer), default = None
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
        When set to true, an interactive EDA report is displayed. 
        

    Returns:
        Global variables that can be changed using the ``set_config`` function.

    """

    available_plots = {
        "tsne": "Anomaly TSnE (3d)",
        "umap": "UMAP Dimensionality",
    }

    if log_plots == True:
        log_plots = ["tsne"]

    return pycaret.internal.tabular.setup(
        ml_usecase="anomaly",
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
    )


def create_model(
    model: Union[str, Any],
    fraction: float = 0.05,
    verbose: bool = True,
    fit_kwargs: Optional[dict] = None,
    **kwargs
):

    """
    This function trains a given model from the model library. All available
    models can be accessed using the ``models`` function. 
    

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> knn = create_model('knn')


    model : string / object
        ID of an model available in the model library or pass an untrained 
        model object consistent with scikit-learn API. Estimators available  
        in the model library (ID - Name):

        * 'abod' - Angle-base Outlier Detection       
        * 'cluster' - Clustering-Based Local Outlier            
        * 'cof' - Connectivity-Based Outlier Factor                 
        * 'histogram' - Histogram-based Outlier Detection          
        * 'knn' - k-Nearest Neighbors Detector                       
        * 'lof' - Local Outlier Factor                            
        * 'svm' - One-class SVM detector                          
        * 'pca' - Principal Component Analysis                    
        * 'mcd' - Minimum Covariance Determinant                    
        * 'sod' - Subspace Outlier Detection                       
        * 'sos' - Stochastic Outlier Selection                       


    fraction: float, default = 0.05
        The percentage of outliers in the dataset.


    verbose: Boolean, default = True
        Status update is not printed when verbose is set to False.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    **kwargs: 
        Additional keyword arguments to pass to the estimator.


    Returns:
        Trained Model 

    """

    return pycaret.internal.tabular.create_model_unsupervised(
        estimator=model,
        fraction=fraction,
        fit_kwargs=fit_kwargs,
        verbose=verbose,
        **kwargs,
    )


def assign_model(
    model, transformation: bool = False, score: bool = True, verbose: bool = True
) -> pd.DataFrame:

    """
    This function assigns anomaly labels to the dataset (1 = outlier, 0 = inlier).


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> knn = create_model('knn')
    >>> knn_df = assign_model(knn)


    model: scikit-learn compatible object
        Trained model object


    transformation: bool, default = False
        Whether to apply cluster labels on transformed dataset. 
    
    
    score: Boolean, default = True
        Whether to show outlier score or not. 


    verbose: Boolean, default = True
        Status update is not printed when verbose is set to False.
        
    Returns:
        pandas.DataFrame
  
    """
    return pycaret.internal.tabular.assign_model(
        model, transformation=transformation, score=score, verbose=verbose
    )


def plot_model(
    model,
    plot: str = "tsne",
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
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> knn = create_model('knn')
    >>> plot_model(knn, plot = 'tsne')


    model: scikit-learn compatible object
        Trained Model Object


    plot : str, default = 'cluster'
        List of available plots (ID - Name):
        
        * 'tsne' - t-SNE (3d) Dimension Plot
        * 'umap' - UMAP Dimensionality Plot


    feature : str, default = None
        Feature to be evaluated when plot = 'distribution'. When ``plot`` type is 
        'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or 
        label when the ``label`` param is set to True. When the ``plot`` type is 
        'cluster' or 'tsne' and feature is None, first column of the dataset is
        used.
        

    label : bool, default = False
        Name of column to be used as data labels. 


    scale: float, default = 1
        The resolution scale of the figure.


    save: Boolean, default = False
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
    This function displays a user interface for analyzing model performance of a
    given estimator. It calls the ``plot_model`` function internally. 
    

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> knn = create_model('knn')
    >>> evaluate_model(knn)
    

    model: scikit-learn compatible object
        Trained model object


    feature : str, default = None
        Feature to be evaluated when plot = 'distribution'. When ``plot`` type is 
        'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or 
        label when the ``label`` param is set to True. When the ``plot`` type is 
        'cluster' or 'tsne' and feature is None, first column of the dataset is
        used.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    Returns:
        None

    """

    return pycaret.internal.tabular.evaluate_model(
        estimator=model, feature_name=feature, fit_kwargs=fit_kwargs
    )


def tune_model(
    model,
    supervised_target: str,
    supervised_type: Optional[str] = None,
    supervised_estimator: Union[str, Any] = "lr",
    method: str = "drop",
    optimize: Optional[str] = None,
    custom_grid: Optional[List[int]] = None,
    fold: int = 10,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    round: int = 4,
    verbose: bool = True,
):

    """
    This function tunes the fraction parameter using a predefined grid with
    the objective of optimizing a supervised learning metric as defined in 
    the optimize param. You can choose the supervised estimator from a large 
    library available in pycaret. By default, supervised estimator is Linear. 
    
    This function returns the tuned model object.
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston, normalize = True)
    >>> tuned_knn = tune_model(model = 'knn', supervised_target = 'medv') 
    
    This will return tuned k-Nearest Neighbors model.

    Parameters
    ----------
    model : str, default = None
        Enter ID of the models available in model library (ID - Model):

        * 'abod' - Angle-base Outlier Detection       
        * 'cluster' - Clustering-Based Local Outlier            
        * 'cof' - Connectivity-Based Outlier Factor                 
        * 'histogram' - Histogram-based Outlier Detection          
        * 'knn' - k-Nearest Neighbors Detector                       
        * 'lof' - Local Outlier Factor                            
        * 'svm' - One-class SVM detector                          
        * 'pca' - Principal Component Analysis                    
        * 'mcd' - Minimum Covariance Determinant                    
        * 'sod' - Subspace Outlier Detection                       
        * 'sos' - Stochastic Outlier Selection    
    
    supervised_target: string
        Name of the target column for supervised learning.
    
    method: str, default = 'drop'
        When method set to drop, it will drop the outlier rows from training dataset 
        of supervised estimator, when method set to 'surrogate', it will use the
        decision function and label as a feature without dropping the outliers from
        training dataset.
    
    estimator: str, default = None
        For Classification (ID - Name):

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

        For Regression (ID - Name):

        * 'lr' - Linear Regression                                  
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
        
        If set to None, Linear model is used by default for both classification
        and regression tasks.
    
    optimize: str, default = None
        For Classification tasks:
            Accuracy, AUC, Recall, Precision, F1, Kappa
        
        For Regression tasks:
            MAE, MSE, RMSE, R2, RMSLE, MAPE
        
        If set to None, default is 'Accuracy' for classification and 'R2' for 
        regression tasks.
    
    custom_grid: list, default = None
        By default, a pre-defined list of fraction values is iterated over to 
        optimize the supervised objective. To overwrite default iteration,
        pass a list of fraction value to iterate over in custom_grid param.
    
    fold: integer, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2. 

    verbose: Boolean, default = True
        Status update is not printed when verbose is set to False.

    Returns
    -------
    Visual_Plot
        Visual plot with fraction param on x-axis with metric to
        optimize on y-axis. Also, prints the best model metric.
    
    model
        trained model object with best fraction param. 
          
    """

    """
    exception handling starts here
    """
    return pycaret.internal.tabular.tune_model_unsupervised(
        model=model,
        supervised_target=supervised_target,
        supervised_type=supervised_type,
        supervised_estimator=supervised_estimator,
        method=method,
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
    This function is used to predict new data using a trained model. It requires a
    trained model object created using one of the function in pycaret that returns 
    a trained model object. New data must be passed to data param as pandas Dataframe. 
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> experiment_name = setup(data = anomaly)
    >>> knn = create_model('knn')
    >>> knn_predictions = predict_model(model = knn, data = anomaly)
        
    Parameters
    ----------
    model : object / string,  default = None
        When model is passed as string, load_model() is called internally to load the
        pickle file from active directory or cloud platform when platform param is passed.
    
    data : pandas.DataFrame
        Shape (n_samples, n_features) where n_samples is the number of samples and 
        n_features is the number of features. All features used during training must be 
        present in the new dataset.
     
    Returns
    -------
    info_grid
        Information grid is printed when data is None.             
    
    Warnings
    --------
    - The behavior of the predict_model is changed in version 2.1 without backward compatibility.
      As such, the pipelines trained using the version (<= 2.0), may not work for inference 
      with version >= 2.1. You can either retrain your models with a newer version or downgrade
      the version for inference.


    """

    return pycaret.internal.tabular.predict_model_unsupervised(
        estimator=model, data=data
    )


def get_outliers(
    data,
    model: Union[str, Any] = "knn",
    fraction: float = 0.05,
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
        fraction=fraction,
        fit_kwargs=fit_kwargs,
        verbose=False,
        **kwargs,
    )
    dataset = assign_model(c, verbose=False)
    return dataset


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


def models(
    internal: bool = False, raise_errors: bool = True,
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
        internal=internal, raise_errors=raise_errors
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
