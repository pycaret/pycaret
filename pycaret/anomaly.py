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
    use_gpu: bool = False,  # added in pycaret==2.1
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
    This function initializes the environment in pycaret. setup() must called before
    executing any other function in pycaret. It takes one mandatory parameter:
    data.

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> experiment_name = setup(data = jewellery, normalize = True)
    
    'jewellery' is a pandas.DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Shape (n_samples, n_features) where n_samples is the number of samples 
        and n_features is the number of features.
    
    categorical_features: str, default = None
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
    
    high_cardinality_features: str, default = None
        When the data containts features with high cardinality, they can be compressed
        into fewer levels by passing them as a list of column names with high cardinality.
        Features are compressed using frequency distribution. As such original features
        are replaced with the frequency distribution and converted into numeric variable. 
    
    numeric_features: str, default = None
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
    
        - 'minmax': scales and translates each feature individually such that it is in 
          the range of 0 - 1.
        - 'maxabs': scales and translates each feature individually such that the 
          maximal absolute value of each feature will be 1.0. It does not 
          shift/center the data, and thus does not destroy any sparsity.
        - 'robust' : scales and translates each feature according to the Interquartile 
          range. When the dataset contains outliers, robust scaler often gives 
          better results.

    transformation: bool, default = False
        When set to True, a power transformation is applied to make the data more normal /
        Gaussian-like. This is useful for modeling issues related to heteroscedasticity or 
        other situations where normality is desired. The optimal parameter for stabilizing 
        variance and minimizing skewness is estimated through maximum likelihood.
    
    transformation_method: str, default = 'yeo-johnson'
        Defines the method for transformation. By default, the transformation method is 
        set to 'yeo-johnson'. The other available option is 'quantile' transformation. 
        Both the transformation transforms the feature set to follow a Gaussian-like or 
        normal distribution. Note that the quantile transformer is non-linear and may 
        distort linear correlations between variables measured at the same scale.
    
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
        
        - kernel: dimensionality reduction through the use of RVF kernel.
        - incremental: replacement for 'linear' pca when the dataset to be decomposed is 
          too large to fit in memory

    pca_components: int/float, default = 0.99
        Number of components to keep. if pca_components is a float, it is treated as a 
        target percentage for information retention. When pca_components is an integer
        it is treated as the number of features to be kept. pca_components must be 
        strictly less than the original number of features in the dataset.
    
    ignore_low_variance: bool, default = False
        When set to True, all categorical features with statistically insignificant 
        variances are removed from the dataset. The variance is calculated using the 
        ratio of unique values to the number of samples, and the ratio of the most 
        common value to the frequency of the second most common value.
    
    combine_rare_levels: bool, default = False
        When set to True, all levels in categorical features below the threshold 
        defined in rare_level_threshold param are combined together as a single level. 
        There must be atleast two levels under the threshold for this to take effect. 
        rare_level_threshold represents the percentile distribution of level frequency. 
        Generally, this technique is applied to limit a sparse matrix caused by high 
        numbers of levels in categorical features. 
    
    rare_level_threshold: float, default = 0.1
        Percentile distribution below which rare categories are combined. Only comes 
        into effect when combine_rare_levels is set to True.
    
    bin_numeric_features: list, default = None
        When a list of numeric features is passed they are transformed into categorical
        features using KMeans, where values in each bin have the same nearest center of 
        a 1D k-means cluster. The number of clusters are determined based on the 'sturges' 
        method. It is only optimal for gaussian data and underestimates the number of bins 
        for large non-gaussian datasets.
    
    remove_multicollinearity: bool, default = False
        When set to True, the variables with inter-correlations higher than the threshold
        defined under the multicollinearity_threshold param are dropped. When two features
        are highly correlated with each other, the feature with higher average correlation 
        in the feature space is dropped. 
    
    multicollinearity_threshold: float, default = 0.9
        Threshold used for dropping the correlated features. Only comes into effect when 
        remove_multicollinearity is set to True.
    
    group_features: list or list of list, default = None
        When a dataset contains features that have related characteristics, the 
        group_features param can be used for statistical feature extraction. For example, 
        if a dataset has numeric features that are related with each other 
        (i.e 'Col1', 'Col2', 'Col3'), a list containing the column names can be passed 
        under group_features to extract statistical information such as the mean, median, 
        mode and standard deviation.
    
    group_names: list, default = None
        When group_features is passed, a name of the group can be passed into group_names 
        param as a list containing strings. The length of a group_names list must equal 
        to the length of group_features. When the length doesn't match or the name is 
        not passed, new features are sequentially named such as group_1, group_2 etc.
    
    n_jobs: int, default = -1
        The number of jobs to run in parallel (for functions that supports parallel 
        processing) -1 means using all processors. To run all functions on single 
        processor set n_jobs to None.

    use_gpu: str or bool, default = False
        If set to 'force', will try to use GPU with all algorithms that support it,
        and raise exceptions if they are unavailable.
        If set to True, will use GPU with algorithms that support it, and fall
        back to CPU if they are unavailable.
        If set to False, will only use CPU.

        GPU enabled algorithms:

        - KMeans, DBSCAN - requires cuML >= 0.15 to be installed.
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

    log_experiment: bool, default = True
        When set to True, all metrics and parameters are logged on MLFlow server.

    experiment_name: str, default = None
        Name of experiment for logging. When set to None, 'clu' is by default used as 
        alias for the experiment name.

    log_plots: bool, default = False
        When set to True, specific plots are logged in MLflow as a png file. 
        By default, it is set to False. 

    log_profile: bool, default = False
        When set to True, data profile is also logged on MLflow as a html file. 
        By default, it is set to False. 

    log_data: bool, default = False
        When set to True, train and test dataset are logged as csv. 

    silent: bool, default = False
        When set to True, confirmation of data types is not required. All preprocessing 
        will be performed assuming automatically inferred data types. Not recommended for 
        direct use except for established pipelines.

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
        This function returns various outputs that are stored in variable
        as tuple. They are used by other functions in pycaret.

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
    This function creates a model on the dataset passed as a data param during 
    the setup stage. setup() function must be called before using create_model().

    This function returns a trained model object. 
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> experiment_name = setup(data = anomaly, normalize = True)
    >>> knn = create_model('knn')

    This will return trained k-Nearest Neighbors model.

    Parameters
    ----------
    model : string / object
        Enter ID of the models available in model library or pass an untrained model 
        object consistent with fit / predict API to train and evaluate model. List of 
        models available in model library (ID - Model):

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
        The percentage / proportion of outliers in the dataset.

    verbose: Boolean, default = True
        Status update is not printed when verbose is set to False.

    system: Boolean, default = True
        Must remain True all times. Only to be changed by internal functions.

    **kwargs: 
        Additional keyword arguments to pass to the estimator.

    Returns
    -------
    model
        Trained model object.

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
    This function flags each of the data point in the dataset passed during setup
    stage as either outlier or inlier (1 = outlier, 0 = inlier) using trained model
    object passed as model param. create_model() function must be called before using 
    assign_model().
    
    This function returns dataframe with Outlier flag (1 = outlier, 0 = inlier) and 
    decision score, when score is set to True.

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> experiment_name = setup(data = anomaly, normalize = True)
    >>> knn = create_model('knn')
    >>> knn_df = assign_model(knn)

    This will return a dataframe with inferred outliers using trained model.

    Parameters
    ----------
    model : trained model object, default = None
    
    transformation: bool, default = False
        When set to True, assigned outliers are returned on transformed dataset instead 
        of original dataset passed during setup().
    
    score: Boolean, default = True
        The outlier scores of the training data. The higher, the more abnormal. 
        Outliers tend to have higher scores. This value is available once the model 
        is fitted. If set to False, it will only return the flag 
        (1 = outlier, 0 = inlier).

    verbose: Boolean, default = True
        Status update is not printed when verbose is set to False.

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with inferred outliers using a trained model.
  
    """
    return pycaret.internal.tabular.assign_model(
        model, transformation=transformation, score=score, verbose=verbose
    )


def plot_model(
    model,
    plot: str = "tsne",
    feature: Optional[str] = None,
    label: bool = False,
    scale: float = 1,  # added in pycaret==2.1
    save: bool = False,
):

    """
    This function takes a trained model object and returns a plot on the dataset 
    passed during setup stage. This function internally calls assign_model before 
    generating a plot.  

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> experiment_name = setup(data = jewellery, normalize = True)
    >>> kmeans = create_model('kmeans')
    >>> plot_model(kmeans)

    This will return a cluster scatter plot (by default). 

    Parameters
    ----------
    model : object, default = none
        A trained model object can be passed. Model must be created using create_model().

    plot : str, default = 'cluster'
        Enter abbreviation for type of plot. The current list of plots supported are 
        (Plot - Name):

        * 'tsne' - t-SNE (3d) Dimension Plot
        * 'umap' - UMAP Dimensionality Plot

    feature : str, default = None
        Name of feature column for x-axis of when plot = 'distribution'. When plot is
        'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or label
        when label is set to True. If no feature name is passed in 'cluster' or 'tsne'
        by default the first of column of dataset is chosen as hoverover tooltip.
    
    label : bool, default = False
        When set to True, data labels are shown in 'cluster' and 'tsne' plot.

    scale: float, default = 1
        The resolution scale of the figure.

    save: Boolean, default = False
        Plot is saved as png file in local directory when save parameter set to True.

    Returns
    -------
    Visual_Plot
        Prints the visual plot. 

    """
    return pycaret.internal.tabular.plot_model(
        model, plot=plot, feature_name=feature, label=label, scale=scale, save=save
    )


def evaluate_model(
    estimator, feature: Optional[str] = None, fit_kwargs: Optional[dict] = None,
):

    """
    This function displays a user interface for all of the available plots for 
    a given estimator. It internally uses the plot_model() function. 
    
    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> experiment_name = setup(data = jewellery, normalize = True)
    >>> kmeans = create_model('kmeans')
    >>> evaluate_model(kmeans)
    
    This will display the User Interface for all of the plots for a given
    estimator.

    Parameters
    ----------
    estimator : object, default = none
        A trained model object should be passed as an estimator. 

    feature : str, default = None
        Name of feature column for x-axis of when plot = 'distribution'. When plot is
        'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or label
        when label is set to True. If no feature name is passed in 'cluster' or 'tsne'
        by default the first of column of dataset is chosen as hoverover tooltip.

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    Returns
    -------
    User_Interface
        Displays the user interface for plotting.

    """

    return pycaret.internal.tabular.evaluate_model(
        estimator=estimator, feature_name=feature, fit_kwargs=fit_kwargs
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


def get_anomaly(
    data,
    model: Union[str, Any] = "kmeans",
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
