# Module: Clustering
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT
# Release: PyCaret 2.1.1
# Last modified : 29/08/2020

import sys
import datetime, time
import warnings
from pycaret.internal.logging import get_logger
from pycaret.internal.Display import Display
import pandas as pd
import numpy as np
import ipywidgets as ipw
from typing import List, Tuple, Any, Union, Optional, Dict
import pycaret.internal.tabular


def setup(
    data,
    preprocess: bool = True,
    imputation_type: str = "simple",
    iterative_imputation_iters: int = 10,
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
    display: Optional[Display] = None,
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
    
    categorical_features: string, default = None
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
    
    high_cardinality_features: string, default = None
        When the data containts features with high cardinality, they can be compressed
        into fewer levels by passing them as a list of column names with high cardinality.
        Features are compressed using frequency distribution. As such original features
        are replaced with the frequency distribution and converted into numeric variable. 
    
    numeric_features: string, default = None
        If the inferred data types are not correct, numeric_features can be used to
        overwrite the inferred type. If when running setup the type of 'column1' is 
        inferred as a categorical instead of numeric, then this parameter can be used 
        to overwrite by passing numeric_features = ['column1'].    

    numeric_imputation: string, default = 'mean'
        If missing values are found in numeric features, they will be imputed with the 
        mean value of the feature. The other available options are 'median' which imputes 
        the value using the median value in the training dataset and 'zero' which
        replaces missing values with zeroes.
    
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
        
        'maxabs'    : scales and translates each feature individually such that the 
                    maximal absolute value of each feature will be 1.0. It does not 
                    shift/center the data, and thus does not destroy any sparsity.
        
        'robust'    : scales and translates each feature according to the Interquartile 
                    range. When the dataset contains outliers, robust scaler often gives 
                    better results.
    
    transformation: bool, default = False
        When set to True, a power transformation is applied to make the data more normal /
        Gaussian-like. This is useful for modeling issues related to heteroscedasticity or 
        other situations where normality is desired. The optimal parameter for stabilizing 
        variance and minimizing skewness is estimated through maximum likelihood.
    
    transformation_method: string, default = 'yeo-johnson'
        Defines the method for transformation. By default, the transformation method is 
        set to 'yeo-johnson'. The other available option is 'quantile' transformation. 
        Both the transformation transforms the feature set to follow a Gaussian-like or 
        normal distribution. Note that the quantile transformer is non-linear and may 
        distort linear correlations between variables measured at the same scale.
    
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
        use_gpu=False,
        custom_pipeline=None,
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
    model,
    num_clusters: int = 4,
    ground_truth: Optional[str] = None,
    round: int = 4,
    fit_kwargs: Optional[dict] = None,
    verbose: bool = True,
    **kwargs
):

    """
    This function creates a model on the dataset passed as a data param during 
    the setup stage. setup() function must be called before using create_model().

    This function returns a trained model object. 

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> experiment_name = setup(data = jewellery, normalize = True)
    >>> kmeans = create_model('kmeans')

    This will return a trained K-Means clustering model.

    Parameters
    ----------
    model : str / object, default = None
        Enter ID of the models available in model library or pass an untrained model 
        object consistent with fit / predict API to train and evaluate model. List of 
        models available in model library (ID - Name):

        * 'kmeans' - K-Means Clustering
        * 'ap' - Affinity Propagation
        * 'meanshift' - Mean shift Clustering
        * 'sc' - Spectral Clustering
        * 'hclust' - Agglomerative Clustering
        * 'dbscan' - Density-Based Spatial Clustering
        * 'optics' - OPTICS Clustering                               
        * 'birch' - Birch Clustering                                 
        * 'kmodes' - K-Modes Clustering                              
    
    num_clusters: int, default = None
        Number of clusters to be generated with the dataset. If None, num_clusters 
        is set to 4. 

    ground_truth: string, default = None
        When ground_truth is provided, Homogeneity Score, Rand Index, and 
        Completeness Score is evaluated and printer along with other metrics.

    round: integer, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    verbose: Boolean, default = True
        Status update is not printed when verbose is set to False.

    **kwargs: 
        Additional keyword arguments to pass to the estimator.

    Returns
    -------
    score_grid
        A table containing the Silhouette, Calinski-Harabasz,  
        Davies-Bouldin, Homogeneity Score, Rand Index, and 
        Completeness Score. Last 3 are only evaluated when
        ground_truth param is provided.

    model
        trained model object

    Warnings
    --------
    - num_clusters not required for Affinity Propagation ('ap'), Mean shift 
      clustering ('meanshift'), Density-Based Spatial Clustering ('dbscan')
      and OPTICS Clustering ('optics'). num_clusters param for these models 
      are automatically determined.
      
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


def assign_model(model, transformation: bool = False, verbose: bool = True):

    """
    This function assigns each of the data point in the dataset passed during setup
    stage to one of the clusters using trained model object passed as model param.
    create_model() function must be called before using assign_model().
    
    This function returns a pandas.DataFrame.

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> experiment_name = setup(data = jewellery, normalize = True)
    >>> kmeans = create_model('kmeans')
    >>> kmeans_df = assign_model(kmeans)

    This will return a pandas.DataFrame with inferred clusters using trained model.

    Parameters
    ----------
    model: trained model object, default = None
    
    transformation: bool, default = False
        When set to True, assigned clusters are returned on transformed dataset instead 
        of original dataset passed during setup().
    
    verbose: Boolean, default = True
        Status update is not printed when verbose is set to False.

    Returns
    -------
    pandas.DataFrame
        Returns a DataFrame with assigned clusters using a trained model.
  
    """

    return pycaret.internal.tabular.assign_model(
        model, transformation=transformation, verbose=verbose
    )


def plot_model(
    model,
    plot="cluster",
    feature=None,
    label=False,
    scale=1,  # added in pycaret==2.1
    save=False,
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

    plot : string, default = 'cluster'
        Enter abbreviation for type of plot. The current list of plots supported are 
        (Plot - Name):

        * 'cluster' - Cluster PCA Plot (2d)              
        * 'tsne' - Cluster TSnE (3d)
        * 'elbow' - Elbow Plot 
        * 'silhouette' - Silhouette Plot         
        * 'distance' - Distance Plot   
        * 'distribution' - Distribution Plot
    
    feature : string, default = None
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


def tune_model(
    model=None,
    supervised_target=None,
    estimator=None,
    optimize=None,
    custom_grid=None,
    fold=10,
    verbose=True,
):

    """
    This function tunes the num_clusters model parameter using a predefined grid with
    the objective of optimizing a supervised learning metric as defined in the optimize
    param. You can choose the supervised estimator from a large library available in 
    pycaret. By default, supervised estimator is Linear. 
    
    This function returns the tuned model object.
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston, normalize = True)
    >>> tuned_kmeans = tune_model(model = 'kmeans', supervised_target = 'medv') 

    This will return tuned K-Means Clustering Model.

    Parameters
    ----------
    model : string, default = None
        Enter ID of the models available in model library (ID - Name):
        
        * 'kmeans' - K-Means Clustering
        * 'ap' - Affinity Propagation
        * 'meanshift' - Mean shift Clustering
        * 'sc' - Spectral Clustering
        * 'hclust' - Agglomerative Clustering
        * 'dbscan' - Density-Based Spatial Clustering
        * 'optics' - OPTICS Clustering                               
        * 'birch' - Birch Clustering                                 
        * 'kmodes' - K-Modes Clustering    
    
    supervised_target: string
        Name of the target column for supervised learning.
        
    estimator: string, default = None
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
        
        If set to None, Linear / Logistic model is used by default.
    
    optimize: string, default = None
        For Classification tasks:
            Accuracy, AUC, Recall, Precision, F1, Kappa
        
        For Regression tasks:
            MAE, MSE, RMSE, R2, RMSLE, MAPE
            
    If set to None, default is 'Accuracy' for classification and 'R2' for 
    regression tasks.
    
    custom_grid: list, default = None
        By default, a pre-defined number of clusters is iterated over to 
        optimize the supervised objective. To overwrite default iteration,
        pass a list of num_clusters to iterate over in custom_grid param.
    
    fold: integer, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2. 

    verbose: Boolean, default = True
        Status update is not printed when verbose is set to False.

    Returns
    -------
    Visual_Plot
        Visual plot with num_clusters param on x-axis with metric to
        optimize on y-axis. Also, prints the best model metric.
    
    model
        trained model object with best num_clusters param. 

    Warnings
    --------
    - Affinity Propagation, Mean shift clustering, Density-Based Spatial Clustering
      and OPTICS Clustering cannot be used in this function since they donot support
      num_clusters param.
           
          
    """

    """
    exception handling starts here
    """

    global data_, X

    import logging

    try:
        hasattr(logger, "name")
    except:
        logger = logging.getLogger("logs")
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()

        ch = logging.FileHandler("logs.log")
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing tune_model()")
    logger.info(
        """tune_model(model={}, supervised_target={}, estimator={}, optimize={}, custom_grid={}, fold={}, verbose={})""".format(
            str(model),
            str(supervised_target),
            str(estimator),
            str(optimize),
            str(custom_grid),
            str(fold),
            str(verbose),
        )
    )

    logger.info("Checking exceptions")

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    import sys

    # run_time
    import datetime, time

    runtime_start = time.time()

    # checking for model parameter
    if model is None:
        raise ValueError(
            "Model parameter Missing. Please see docstring for list of available models."
        )

    # checking for allowed models
    allowed_models = ["kmeans", "sc", "hclust", "birch", "kmodes"]

    if model not in allowed_models:
        raise ValueError(
            "Model Not Available for Tuning. Please see docstring for list of available models."
        )

    # check if supervised target is None:
    if supervised_target is None:
        raise ValueError(
            "supervised_target cannot be None. A column name must be given for estimator."
        )

    # check supervised target
    if supervised_target is not None:
        all_col = list(data_.columns)
        if supervised_target not in all_col:
            raise ValueError(
                "supervised_target not recognized. It can only be one of the following: "
                + str(all_col)
            )

    # checking estimator:
    if estimator is not None:

        available_estimators = [
            "lr",
            "knn",
            "nb",
            "dt",
            "svm",
            "rbfsvm",
            "gpc",
            "mlp",
            "ridge",
            "rf",
            "qda",
            "ada",
            "gbc",
            "lda",
            "et",
            "lasso",
            "ridge",
            "en",
            "lar",
            "llar",
            "omp",
            "br",
            "ard",
            "par",
            "ransac",
            "tr",
            "huber",
            "kr",
            "svm",
            "knn",
            "dt",
            "rf",
            "et",
            "ada",
            "gbr",
            "mlp",
            "xgboost",
            "lightgbm",
            "catboost",
        ]

        if estimator not in available_estimators:
            raise ValueError(
                "Estimator Not Available. Please see docstring for list of available estimators."
            )

    # checking optimize parameter
    if optimize is not None:

        available_optimizers = [
            "MAE",
            "MSE",
            "RMSE",
            "R2",
            "RMSLE",
            "MAPE",
            "Accuracy",
            "AUC",
            "Recall",
            "Precision",
            "F1",
            "Kappa",
        ]

        if optimize not in available_optimizers:
            raise ValueError(
                "optimize parameter Not Available. Please see docstring for list of available parameters."
            )

    # checking fold parameter
    if type(fold) is not int:
        raise TypeError("Fold parameter only accepts integer value.")

    """
    exception handling ends here
    """

    logger.info("Preloading libraries")

    # pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from ipywidgets import Output
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time

    logger.info("Preparing display monitor")

    # progress bar
    if custom_grid is None:
        max_steps = 25
    else:
        max_steps = 15 + len(custom_grid)

    progress = ipw.IntProgress(
        value=0, min=0, max=max_steps, step=1, description="Processing: "
    )

    if verbose:
        if html_param:
            display(progress)

    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")

    monitor = pd.DataFrame(
        [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
            ["Step", ". . . . . . . . . . . . . . . . . .", "Initializing"],
        ],
        columns=["", " ", "   "],
    ).set_index("")

    monitor_out = Output()

    if verbose:
        if html_param:
            display(monitor_out)
            with monitor_out:
                display(monitor, display_id="monitor")

    logger.info("Importing libraries")

    # General Dependencies
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn import metrics
    import numpy as np
    import plotly.express as px
    from copy import deepcopy

    logger.info("Copying environment variables")

    a = data_.copy()
    b = X.copy()
    c = deepcopy(prep_pipe)
    e = exp_name_log
    z = logging_param

    def retain_original(a, b, c, e, z):

        global data_, X, prep_pipe, exp_name_log, logging_param

        data_ = a.copy()
        X = b.copy()
        prep_pipe = deepcopy(c)
        exp_name_log = e
        logging_param = z

        return data_, X, prep_pipe, exp_name_log, logging_param

    # setting up cufflinks
    import cufflinks as cf

    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    progress.value += 1

    # define the problem
    if data_[supervised_target].value_counts().count() == 2:
        problem = "classification"
        logger.info("Objective : Classification")
    else:
        problem = "regression"
        logger.info("Objective : Regression")

    # define model name

    logger.info("Defining Model Name")

    if model == "kmeans":
        model_name = "K-Means Clustering"
    elif model == "ap":
        model_name = "Affinity Propagation"
    elif model == "meanshift":
        model_name = "Mean Shift Clustering"
    elif model == "sc":
        model_name = "Spectral Clustering"
    elif model == "hclust":
        model_name = "Agglomerative Clustering"
    elif model == "dbscan":
        model_name = "Density-Based Spatial Clustering"
    elif model == "optics":
        model_name = "OPTICS Clustering"
    elif model == "birch":
        model_name = "Birch Clustering"
    elif model == "kmodes":
        model_name = "K-Modes Clustering"

    logger.info("Defining Supervised Estimator")

    # defining estimator:
    if problem == "classification" and estimator is None:
        estimator = "lr"
    elif problem == "regression" and estimator is None:
        estimator = "lr"
    else:
        estimator = estimator

    logger.info("Defining Optimizer")
    # defining optimizer:
    if optimize is None and problem == "classification":
        optimize = "Accuracy"
    elif optimize is None and problem == "regression":
        optimize = "R2"
    else:
        optimize = optimize

    logger.info("Optimize: " + str(optimize))

    progress.value += 1

    # defining tuning grid
    logger.info("Defining Tuning Grid")

    if custom_grid is not None:

        logger.info("Custom Grid used")
        param_grid = custom_grid
        param_grid_with_zero = [0]

        for i in param_grid:
            param_grid_with_zero.append(i)

    else:

        logger.info("Pre-defined Grid used")
        param_grid = [4, 5, 6, 8, 10, 14, 18, 25, 30, 40]
        param_grid_with_zero = [0, 4, 5, 6, 8, 10, 14, 18, 25, 30, 40]

    master = []
    master_df = []

    monitor.iloc[1, 1:] = "Creating Clustering Model"
    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    """
    preprocess starts here
    """

    logger.info("Defining setup variables for preprocessing")

    # removing target variable from data by defining new setup
    _data_ = data_.copy()
    target_ = pd.DataFrame(_data_[supervised_target])
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    target_ = le.fit_transform(target_)

    cat_pass = prep_param.dtypes.categorical_features
    num_pass = prep_param.dtypes.numerical_features
    time_pass = prep_param.dtypes.time_features
    ignore_pass = prep_param.dtypes.features_todrop

    # PCA
    # ---#
    if "Empty" in str(prep_param.pca):
        pca_pass = False
        pca_method_pass = "linear"

    else:
        pca_pass = True

        if prep_param.pca.method == "pca_liner":
            pca_method_pass = "linear"
        elif prep_param.pca.method == "pca_kernal":
            pca_method_pass = "kernel"
        elif prep_param.pca.method == "incremental":
            pca_method_pass = "incremental"

    if pca_pass is True:
        pca_comp_pass = prep_param.pca.variance_retained
    else:
        pca_comp_pass = 0.99

    # IMPUTATION
    if "not_available" in prep_param.imputer.categorical_strategy:
        cat_impute_pass = "constant"
    elif "most frequent" in prep_param.imputer.categorical_strategy:
        cat_impute_pass = "mode"

    num_impute_pass = prep_param.imputer.numeric_strategy

    # NORMALIZE
    if "Empty" in str(prep_param.scaling):
        normalize_pass = False
    else:
        normalize_pass = True

    if normalize_pass is True:
        normalize_method_pass = prep_param.scaling.function_to_apply
    else:
        normalize_method_pass = "zscore"

    # FEATURE TRANSFORMATION
    if "Empty" in str(prep_param.P_transform):
        transformation_pass = False
    else:
        transformation_pass = True

    if transformation_pass is True:

        if "yj" in prep_param.P_transform.function_to_apply:
            transformation_method_pass = "yeo-johnson"
        elif "quantile" in prep_param.P_transform.function_to_apply:
            transformation_method_pass = "quantile"

    else:
        transformation_method_pass = "yeo-johnson"

    # BIN NUMERIC FEATURES
    if "Empty" in str(prep_param.binn):
        features_to_bin_pass = []
        apply_binning_pass = False

    else:
        features_to_bin_pass = prep_param.binn.features_to_discretize
        apply_binning_pass = True

    # COMBINE RARE LEVELS
    if "Empty" in str(prep_param.club_R_L):
        combine_rare_levels_pass = False
        combine_rare_threshold_pass = 0.1
    else:
        combine_rare_levels_pass = True
        combine_rare_threshold_pass = prep_param.club_R_L.threshold

    # ZERO NERO ZERO VARIANCE
    if "Empty" in str(prep_param.znz):
        ignore_low_variance_pass = False
    else:
        ignore_low_variance_pass = True

    # MULTI-COLLINEARITY
    if "Empty" in str(prep_param.fix_multi):
        remove_multicollinearity_pass = False
    else:
        remove_multicollinearity_pass = True

    if remove_multicollinearity_pass is True:
        multicollinearity_threshold_pass = prep_param.fix_multi.threshold
    else:
        multicollinearity_threshold_pass = 0.9

    # UNKNOWN CATEGORICAL LEVEL
    if "Empty" in str(prep_param.new_levels):
        handle_unknown_categorical_pass = False
    else:
        handle_unknown_categorical_pass = True

    if handle_unknown_categorical_pass is True:
        unknown_level_preprocess = prep_param.new_levels.replacement_strategy
        if unknown_level_preprocess == "least frequent":
            unknown_categorical_method_pass = "least_frequent"
        elif unknown_level_preprocess == "most frequent":
            unknown_categorical_method_pass = "most_frequent"
        else:
            unknown_categorical_method_pass = "least_frequent"
    else:
        unknown_categorical_method_pass = "least_frequent"

    # GROUP FEATURES
    if "Empty" in str(prep_param.group):
        apply_grouping_pass = False
    else:
        apply_grouping_pass = True

    if apply_grouping_pass is True:
        group_features_pass = prep_param.group.list_of_similar_features
    else:
        group_features_pass = None

    if apply_grouping_pass is True:
        group_names_pass = prep_param.group.group_name
    else:
        group_names_pass = None

    # ORDINAL FEATURES

    if "Empty" in str(prep_param.ordinal):
        ordinal_features_pass = None
    else:
        ordinal_features_pass = prep_param.ordinal.info_as_dict

    # HIGH CARDINALITY
    if "Empty" in str(prep_param.cardinality):
        high_cardinality_features_pass = None
    else:
        high_cardinality_features_pass = prep_param.cardinality.feature

    global setup_without_target

    logger.info("SubProcess setup() called")

    setup_without_target = setup(
        data=data_,
        categorical_features=cat_pass,
        categorical_imputation=cat_impute_pass,
        ordinal_features=ordinal_features_pass,
        high_cardinality_features=high_cardinality_features_pass,
        numeric_features=num_pass,
        numeric_imputation=num_impute_pass,
        date_features=time_pass,
        ignore_features=ignore_pass,
        normalize=normalize_pass,
        normalize_method=normalize_method_pass,
        transformation=transformation_pass,
        transformation_method=transformation_method_pass,
        handle_unknown_categorical=handle_unknown_categorical_pass,
        unknown_categorical_method=unknown_categorical_method_pass,
        pca=pca_pass,
        pca_components=pca_comp_pass,
        pca_method=pca_method_pass,
        ignore_low_variance=ignore_low_variance_pass,
        combine_rare_levels=combine_rare_levels_pass,
        rare_level_threshold=combine_rare_threshold_pass,
        bin_numeric_features=features_to_bin_pass,
        remove_multicollinearity=remove_multicollinearity_pass,
        multicollinearity_threshold=multicollinearity_threshold_pass,
        group_features=group_features_pass,
        group_names=group_names_pass,
        supervised=True,
        supervised_target=supervised_target,
        session_id=seed,
        log_experiment=False,  # added in pycaret==2.0.0
        profile=False,
        verbose=False,
    )

    data_without_target = setup_without_target[0]

    logger.info("SubProcess setup() end")

    """
    preprocess ends here
    """

    # adding dummy model in master
    master.append("No Model Required")
    master_df.append("No Model Required")

    model_fit_time_list = []

    for i in param_grid:
        logger.info("Fitting Model with num_clusters = " + str(i))
        progress.value += 1
        monitor.iloc[2, 1:] = "Fitting Model With " + str(i) + " Clusters"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        # create and assign the model to dataset d
        model_fit_start = time.time()
        logger.info(
            "SubProcess create_model() called=================================="
        )
        m = create_model(estimator=model, num_clusters=i, verbose=False, system=False)
        logger.info("SubProcess create_model() end==================================")
        model_fit_end = time.time()
        model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
        model_fit_time_list.append(model_fit_time)

        logger.info("Generating labels")
        logger.info(
            "SubProcess assign_model() called=================================="
        )
        d = assign_model(m, transformation=True, verbose=False)
        logger.info("SubProcess assign_model() ends==================================")
        d[str(supervised_target)] = target_

        master.append(m)
        master_df.append(d)

        # clustering model creation end's here

    # attaching target variable back
    data_[str(supervised_target)] = target_

    logger.info("Defining Supervised Estimator")

    if problem == "classification":

        logger.info("Problem : Classification")

        """
        
        defining estimator
        
        """

        monitor.iloc[1, 1:] = "Evaluating Clustering Model"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        if estimator == "lr":

            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(random_state=seed)
            full_name = "Logistic Regression"

        elif estimator == "knn":

            from sklearn.neighbors import KNeighborsClassifier

            model = KNeighborsClassifier()
            full_name = "K Nearest Neighbours"

        elif estimator == "nb":

            from sklearn.naive_bayes import GaussianNB

            model = GaussianNB()
            full_name = "Naive Bayes"

        elif estimator == "dt":

            from sklearn.tree import DecisionTreeClassifier

            model = DecisionTreeClassifier(random_state=seed)
            full_name = "Decision Tree"

        elif estimator == "svm":

            from sklearn.linear_model import SGDClassifier

            model = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed)
            full_name = "Support Vector Machine"

        elif estimator == "rbfsvm":

            from sklearn.svm import SVC

            model = SVC(
                gamma="auto", C=1, probability=True, kernel="rbf", random_state=seed
            )
            full_name = "RBF SVM"

        elif estimator == "gpc":

            from sklearn.gaussian_process import GaussianProcessClassifier

            model = GaussianProcessClassifier(random_state=seed)
            full_name = "Gaussian Process Classifier"

        elif estimator == "mlp":

            from sklearn.neural_network import MLPClassifier

            model = MLPClassifier(max_iter=500, random_state=seed)
            full_name = "Multi Level Perceptron"

        elif estimator == "ridge":

            from sklearn.linear_model import RidgeClassifier

            model = RidgeClassifier(random_state=seed)
            full_name = "Ridge Classifier"

        elif estimator == "rf":

            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=10, random_state=seed)
            full_name = "Random Forest Classifier"

        elif estimator == "qda":

            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

            model = QuadraticDiscriminantAnalysis()
            full_name = "Quadratic Discriminant Analysis"

        elif estimator == "ada":

            from sklearn.ensemble import AdaBoostClassifier

            model = AdaBoostClassifier(random_state=seed)
            full_name = "AdaBoost Classifier"

        elif estimator == "gbc":

            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(random_state=seed)
            full_name = "Gradient Boosting Classifier"

        elif estimator == "lda":

            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            model = LinearDiscriminantAnalysis()
            full_name = "Linear Discriminant Analysis"

        elif estimator == "et":

            from sklearn.ensemble import ExtraTreesClassifier

            model = ExtraTreesClassifier(random_state=seed)
            full_name = "Extra Trees Classifier"

        elif estimator == "xgboost":

            from xgboost import XGBClassifier

            model = XGBClassifier(random_state=seed, n_jobs=n_jobs_param, verbosity=0)
            full_name = "Extreme Gradient Boosting"

        elif estimator == "lightgbm":

            import lightgbm as lgb

            model = lgb.LGBMClassifier(random_state=seed)
            full_name = "Light Gradient Boosting Machine"

        elif estimator == "catboost":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                random_state=seed, silent=True
            )  # Silent is True to suppress CatBoost iteration results
            full_name = "CatBoost Classifier"

        logger.info(str(full_name) + " Imported Successfully")

        progress.value += 1

        """
        start model building here

        """

        logger.info("Creating Classifier without clusters")
        acc = []
        auc = []
        recall = []
        prec = []
        kappa = []
        f1 = []

        # build model without clustering
        monitor.iloc[2, 1:] = "Evaluating Classifier Without Clustering"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        d = master_df[1].copy()
        d.drop(["Cluster"], axis=1, inplace=True)

        # drop NA's caution
        d.dropna(axis=0, inplace=True)

        # get_dummies to caste categorical variables for supervised learning
        d = pd.get_dummies(d)

        # split the dataset
        X = d.drop(supervised_target, axis=1)
        y = d[supervised_target]

        # fit the model
        logger.info("Fitting Model")
        model.fit(X, y)

        # generate the prediction and evaluate metric
        logger.info("Evaluating Cross Val Predictions")
        pred = cross_val_predict(model, X, y, cv=fold, method="predict")

        acc_ = metrics.accuracy_score(y, pred)
        acc.append(acc_)

        recall_ = metrics.recall_score(y, pred)
        recall.append(recall_)

        precision_ = metrics.precision_score(y, pred)
        prec.append(precision_)

        kappa_ = metrics.cohen_kappa_score(y, pred)
        kappa.append(kappa_)

        f1_ = metrics.f1_score(y, pred)
        f1.append(f1_)

        if hasattr(model, "predict_proba"):
            pred_ = cross_val_predict(model, X, y, cv=fold, method="predict_proba")
            pred_prob = pred_[:, 1]
            auc_ = metrics.roc_auc_score(y, pred_prob)
            auc.append(auc_)

        else:
            auc.append(0)

        for i in range(1, len(master_df)):

            progress.value += 1
            param_grid_val = param_grid[i - 1]

            logger.info(
                "Creating Classifier with num_clusters = " + str(param_grid_val)
            )

            monitor.iloc[2, 1:] = (
                "Evaluating Classifier With " + str(param_grid_val) + " Clusters"
            )
            if verbose:
                if html_param:
                    update_display(monitor, display_id="monitor")

            # prepare the dataset for supervised problem
            d = master_df[i]

            # dropping NAs
            d.dropna(axis=0, inplace=True)

            # get_dummies to caste categorical variables for supervised learning
            d = pd.get_dummies(d)

            # split the dataset
            X = d.drop(supervised_target, axis=1)
            y = d[supervised_target]

            # fit the model
            logger.info("Fitting Model")
            model.fit(X, y)

            # generate the prediction and evaluate metric
            logger.info("Generating Cross Val Predictions")
            pred = cross_val_predict(model, X, y, cv=fold, method="predict")

            acc_ = metrics.accuracy_score(y, pred)
            acc.append(acc_)

            recall_ = metrics.recall_score(y, pred)
            recall.append(recall_)

            precision_ = metrics.precision_score(y, pred)
            prec.append(precision_)

            kappa_ = metrics.cohen_kappa_score(y, pred)
            kappa.append(kappa_)

            f1_ = metrics.f1_score(y, pred)
            f1.append(f1_)

            if hasattr(model, "predict_proba"):
                pred_ = cross_val_predict(model, X, y, cv=fold, method="predict_proba")
                pred_prob = pred_[:, 1]
                auc_ = metrics.roc_auc_score(y, pred_prob)
                auc.append(auc_)

            else:
                auc.append(0)

        monitor.iloc[1, 1:] = "Compiling Results"
        monitor.iloc[1, 1:] = "Almost Finished"

        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        logger.info("Creating metrics dataframe")
        df = pd.DataFrame(
            {
                "# of Clusters": param_grid_with_zero,
                "Accuracy": acc,
                "AUC": auc,
                "Recall": recall,
                "Precision": prec,
                "F1": f1,
                "Kappa": kappa,
            }
        )

        sorted_df = df.sort_values(by=optimize, ascending=False)
        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]
        best_model_tt = model_fit_time_list[ival]

        progress.value += 1
        logger.info("Rendering Visual")
        sd = pd.melt(
            df,
            id_vars=["# of Clusters"],
            value_vars=["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa"],
            var_name="Metric",
            value_name="Score",
        )

        fig = px.line(
            sd,
            x="# of Clusters",
            y="Score",
            color="Metric",
            line_shape="linear",
            range_y=[0, 1],
        )
        fig.update_layout(plot_bgcolor="rgb(245,245,245)")
        title = str(full_name) + " Metrics and Number of Clusters"
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
        logger.info("Visual Rendered Successfully")

        # monitor = ''

        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        if verbose:
            if html_param:
                monitor_out.clear_output()
                progress.close()

        best_k = np.array(sorted_df.head(1)["# of Clusters"])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0], 4)
        p = (
            "Best Model: "
            + model_name
            + " |"
            + " Number of Clusters : "
            + str(best_k)
            + " | "
            + str(optimize)
            + " : "
            + str(best_m)
        )
        print(p)

    elif problem == "regression":

        logger.info("Problem : Regression")

        """
        
        defining estimator
        
        """

        monitor.iloc[1, 1:] = "Evaluating Clustering Model"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        if estimator == "lr":

            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            full_name = "Linear Regression"

        elif estimator == "lasso":

            from sklearn.linear_model import Lasso

            model = Lasso(random_state=seed)
            full_name = "Lasso Regression"

        elif estimator == "ridge":

            from sklearn.linear_model import Ridge

            model = Ridge(random_state=seed)
            full_name = "Ridge Regression"

        elif estimator == "en":

            from sklearn.linear_model import ElasticNet

            model = ElasticNet(random_state=seed)
            full_name = "Elastic Net"

        elif estimator == "lar":

            from sklearn.linear_model import Lars

            model = Lars()
            full_name = "Least Angle Regression"

        elif estimator == "llar":

            from sklearn.linear_model import LassoLars

            model = LassoLars()
            full_name = "Lasso Least Angle Regression"

        elif estimator == "omp":

            from sklearn.linear_model import OrthogonalMatchingPursuit

            model = OrthogonalMatchingPursuit()
            full_name = "Orthogonal Matching Pursuit"

        elif estimator == "br":
            from sklearn.linear_model import BayesianRidge

            model = BayesianRidge()
            full_name = "Bayesian Ridge Regression"

        elif estimator == "ard":

            from sklearn.linear_model import ARDRegression

            model = ARDRegression()
            full_name = "Automatic Relevance Determination"

        elif estimator == "par":

            from sklearn.linear_model import PassiveAggressiveRegressor

            model = PassiveAggressiveRegressor(random_state=seed)
            full_name = "Passive Aggressive Regressor"

        elif estimator == "ransac":

            from sklearn.linear_model import RANSACRegressor

            model = RANSACRegressor(random_state=seed)
            full_name = "Random Sample Consensus"

        elif estimator == "tr":

            from sklearn.linear_model import TheilSenRegressor

            model = TheilSenRegressor(random_state=seed)
            full_name = "TheilSen Regressor"

        elif estimator == "huber":

            from sklearn.linear_model import HuberRegressor

            model = HuberRegressor()
            full_name = "Huber Regressor"

        elif estimator == "kr":

            from sklearn.kernel_ridge import KernelRidge

            model = KernelRidge()
            full_name = "Kernel Ridge"

        elif estimator == "svm":

            from sklearn.svm import SVR

            model = SVR()
            full_name = "Support Vector Regression"

        elif estimator == "knn":

            from sklearn.neighbors import KNeighborsRegressor

            model = KNeighborsRegressor()
            full_name = "Nearest Neighbors Regression"

        elif estimator == "dt":

            from sklearn.tree import DecisionTreeRegressor

            model = DecisionTreeRegressor(random_state=seed)
            full_name = "Decision Tree Regressor"

        elif estimator == "rf":

            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(random_state=seed)
            full_name = "Random Forest Regressor"

        elif estimator == "et":

            from sklearn.ensemble import ExtraTreesRegressor

            model = ExtraTreesRegressor(random_state=seed)
            full_name = "Extra Trees Regressor"

        elif estimator == "ada":

            from sklearn.ensemble import AdaBoostRegressor

            model = AdaBoostRegressor(random_state=seed)
            full_name = "AdaBoost Regressor"

        elif estimator == "gbr":

            from sklearn.ensemble import GradientBoostingRegressor

            model = GradientBoostingRegressor(random_state=seed)
            full_name = "Gradient Boosting Regressor"

        elif estimator == "mlp":

            from sklearn.neural_network import MLPRegressor

            model = MLPRegressor(random_state=seed)
            full_name = "MLP Regressor"

        elif estimator == "xgboost":

            from xgboost import XGBRegressor

            model = XGBRegressor(random_state=seed, n_jobs=n_jobs_param, verbosity=0)
            full_name = "Extreme Gradient Boosting Regressor"

        elif estimator == "lightgbm":

            import lightgbm as lgb

            model = lgb.LGBMRegressor(random_state=seed)
            full_name = "Light Gradient Boosting Machine"

        elif estimator == "catboost":

            from catboost import CatBoostRegressor

            model = CatBoostRegressor(random_state=seed, silent=True)
            full_name = "CatBoost Regressor"

        logger.info(str(full_name) + " Imported Successfully")

        progress.value += 1

        """
        start model building here

        """

        logger.info("Creating Regressor without clusters")

        score = []
        metric = []

        # build model without clustering
        monitor.iloc[2, 1:] = "Evaluating Regressor Without Clustering"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        d = master_df[1].copy()
        d.drop(["Cluster"], axis=1, inplace=True)

        # drop NA's caution
        d.dropna(axis=0, inplace=True)

        # get_dummies to caste categorical variables for supervised learning
        d = pd.get_dummies(d)

        # split the dataset
        X = d.drop(supervised_target, axis=1)
        y = d[supervised_target]

        # fit the model
        logger.info("Fitting Model")
        model.fit(X, y)

        # generate the prediction and evaluate metric
        logger.info("Generating Cross Val Predictions")
        pred = cross_val_predict(model, X, y, cv=fold, method="predict")

        if optimize == "R2":
            r2_ = metrics.r2_score(y, pred)
            score.append(r2_)

        elif optimize == "MAE":
            mae_ = metrics.mean_absolute_error(y, pred)
            score.append(mae_)

        elif optimize == "MSE":
            mse_ = metrics.mean_squared_error(y, pred)
            score.append(mse_)

        elif optimize == "RMSE":
            mse_ = metrics.mean_squared_error(y, pred)
            rmse_ = np.sqrt(mse_)
            score.append(rmse_)

        elif optimize == "RMSLE":
            rmsle = np.sqrt(
                np.mean(
                    np.power(
                        np.log(np.array(abs(pred)) + 1) - np.log(np.array(abs(y)) + 1),
                        2,
                    )
                )
            )
            score.append(rmsle)

        elif optimize == "MAPE":

            def calculate_mape(actual, prediction):
                mask = actual != 0
                return (np.fabs(actual - prediction) / actual)[mask].mean()

            mape = calculate_mape(y, pred)
            score.append(mape)

        metric.append(str(optimize))

        for i in range(1, len(master_df)):

            progress.value += 1
            param_grid_val = param_grid[i - 1]

            logger.info("Creating Regressor with num_clusters = " + str(param_grid_val))

            monitor.iloc[2, 1:] = (
                "Evaluating Regressor With " + str(param_grid_val) + " Clusters"
            )
            if verbose:
                if html_param:
                    update_display(monitor, display_id="monitor")

            # prepare the dataset for supervised problem
            d = master_df[i]

            # dropping NA's
            d.dropna(axis=0, inplace=True)

            # get_dummies to caste categorical variable for supervised learning
            d = pd.get_dummies(d)

            # split the dataset
            X = d.drop(supervised_target, axis=1)
            y = d[supervised_target]

            # fit the model
            logger.info("Fitting Model")
            model.fit(X, y)

            # generate the prediction and evaluate metric
            logger.info("Generating Cross Val Predictions")
            pred = cross_val_predict(model, X, y, cv=fold, method="predict")

            if optimize == "R2":
                r2_ = metrics.r2_score(y, pred)
                score.append(r2_)

            elif optimize == "MAE":
                mae_ = metrics.mean_absolute_error(y, pred)
                score.append(mae_)

            elif optimize == "MSE":
                mse_ = metrics.mean_squared_error(y, pred)
                score.append(mse_)

            elif optimize == "RMSE":
                mse_ = metrics.mean_squared_error(y, pred)
                rmse_ = np.sqrt(mse_)
                score.append(rmse_)

            elif optimize == "RMSLE":
                rmsle = np.sqrt(
                    np.mean(
                        np.power(
                            np.log(np.array(abs(pred)) + 1)
                            - np.log(np.array(abs(y)) + 1),
                            2,
                        )
                    )
                )
                score.append(rmsle)

            elif optimize == "MAPE":

                def calculate_mape(actual, prediction):
                    mask = actual != 0
                    return (np.fabs(actual - prediction) / actual)[mask].mean()

                mape = calculate_mape(y, pred)
                score.append(mape)

            metric.append(str(optimize))

        monitor.iloc[1, 1:] = "Compiling Results"
        monitor.iloc[1, 1:] = "Finalizing"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        logger.info("Creating metrics dataframe")
        df = pd.DataFrame(
            {"Clusters": param_grid_with_zero, "Score": score, "Metric": metric}
        )
        df.columns = ["# of Clusters", optimize, "Metric"]

        # sorting to return best model
        if optimize == "R2":
            sorted_df = df.sort_values(by=optimize, ascending=False)
        else:
            sorted_df = df.sort_values(by=optimize, ascending=True)

        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]
        best_model_tt = model_fit_time_list[ival]

        logger.info("Rendering Visual")

        fig = px.line(
            df,
            x="# of Clusters",
            y=optimize,
            line_shape="linear",
            title=str(full_name) + " Metrics and Number of Clusters",
            color="Metric",
        )

        fig.update_layout(plot_bgcolor="rgb(245,245,245)")
        progress.value += 1

        fig.show()

        logger.info("Visual Rendered Successfully")

        # monitor = ''

        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        if verbose:
            if html_param:
                monitor_out.clear_output()
                progress.close()

        best_k = np.array(sorted_df.head(1)["# of Clusters"])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0], 4)
        p = (
            "Best Model: "
            + model_name
            + " |"
            + " Number of Clusters: "
            + str(best_k)
            + " | "
            + str(optimize)
            + " : "
            + str(best_m)
        )
        print(p)

    logger.info("Resetting environment to original variables")
    org = retain_original(a, b, c, e, z)

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # mlflow logging
    if logging_param:

        logger.info("Creating MLFlow logs")

        # import mlflow
        import mlflow
        from pathlib import Path
        import os

        mlflow.set_experiment(exp_name_log)

        # Creating Logs message monitor
        monitor.iloc[1, 1:] = "Creating Logs"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        mlflow.set_experiment(exp_name_log)

        with mlflow.start_run(run_name=model_name) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            # Log model parameters

            try:
                params = best_model.get_params()

                for i in list(params):
                    v = params.get(i)
                    if len(str(v)) > 250:
                        params.pop(i)

                mlflow.log_params(params)

            except:
                pass

            # set tag of compare_models
            mlflow.set_tag("Source", "tune_model")

            import secrets

            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time in seconds
            mlflow.log_metric("TT", best_model_tt)  # change this

            # Log plot to html
            fig.write_html("Iterations.html")
            mlflow.log_artifact("Iterations.html")
            os.remove("Iterations.html")

            # Log model and transformation pipeline
            from copy import deepcopy

            # get default conda env
            from mlflow.sklearn import get_default_conda_env

            default_conda_env = get_default_conda_env()
            default_conda_env["name"] = str(exp_name_log) + "-env"
            default_conda_env.get("dependencies").pop(-3)
            dependencies = default_conda_env.get("dependencies")[-1]
            from pycaret.utils import __version__

            dep = "pycaret==" + str(__version__)
            dependencies["pip"] = [dep]

            # define model signature
            from mlflow.models.signature import infer_signature

            try:
                signature = infer_signature(data_)
            except:
                signature = None
            input_example = data_.iloc[0].to_dict()

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

    logger.info(str(best_model))
    logger.info(
        "tune_model() succesfully completed......................................"
    )

    return best_model


def predict_model(model, data):

    """
    This function is used to predict new data using a trained model. It requires a
    trained model object created using one of the function in pycaret that returns 
    a trained model object. New data must be passed to data param as a DataFrame.
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> experiment_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> kmeans_predictions = predict_model(model = kmeans, data = jewellery)
        
    Parameters
    ----------
    model : object,  default = None
        A trained model object / pipeline should be passed as an estimator. 
    
    data : pandas.DataFrame
        Shape (n_samples, n_features) where n_samples is the number of samples and 
        n_features is the number of features. All features used during training must 
        be present in the new dataset.
     
    Returns
    -------
    info_grid
        Information grid is printed when data is None.

    Warnings
    --------
    - Models that do not support 'predict' function cannot be used in predict_model(). 

    - The behavior of the predict_model is changed in version 2.1 without backward compatibility.
    As such, the pipelines trained using the version (<= 2.0), may not work for inference 
    with version >= 2.1. You can either retrain your models with a newer version or downgrade
    the version for inference.
    

    """

    return pycaret.internal.tabular.predict_model_unsupervised(
        estimator=model, data=data
    )


def models(internal: bool = False, raise_errors: bool = True,) -> pd.DataFrame:

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

    return pycaret.internal.tabular.models(internal=internal, raise_errors=raise_errors)


def get_clusters(
    data,
    model=None,
    num_clusters=4,
    ignore_features=None,
    normalize=True,
    transformation=False,
    pca=False,
    pca_components=0.99,
    ignore_low_variance=False,
    combine_rare_levels=False,
    rare_level_threshold=0.1,
    remove_multicollinearity=False,
    multicollinearity_threshold=0.9,
    n_jobs=None,
):

    """
    Callable from any external environment without requiring setup initialization.
    """

    if model is None:
        model = "kmeans"

    if ignore_features is None:
        ignore_features_pass = []
    else:
        ignore_features_pass = ignore_features

    global X, data_, seed, n_jobs_param, logging_param, logger

    data_ = data.copy()

    seed = 99

    n_jobs_param = n_jobs

    logging_param = False

    import logging

    logger = logging.getLogger("logs")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.FileHandler("logs.log")
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    from pycaret import preprocess

    X = preprocess.Preprocess_Path_Two(
        train_data=data,
        features_todrop=ignore_features_pass,
        display_types=False,
        scale_data=normalize,
        scaling_method="zscore",
        Power_transform_data=transformation,
        Power_transform_method="yj",
        apply_pca=pca,
        pca_variance_retained_or_number_of_components=pca_components,
        apply_zero_nearZero_variance=ignore_low_variance,
        club_rare_levels=combine_rare_levels,
        rara_level_threshold_percentage=rare_level_threshold,
        remove_multicollinearity=remove_multicollinearity,
        maximum_correlation_between_features=multicollinearity_threshold,
        random_state=seed,
    )

    try:
        c = create_model(
            estimator=model, num_clusters=num_clusters, verbose=False, system=False
        )
    except:
        c = create_model(estimator=model, verbose=False, system=False)
    dataset = assign_model(c, verbose=False)
    return dataset


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
    Success_Message
    
         
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
    **kwargs
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
