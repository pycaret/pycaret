# Module: Regression
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT
# Release: PyCaret 2.2
# Last modified : 29/08/2020

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
    feature_selection_method: str = "classic",  # boruta algorithm added in pycaret==2.1
    feature_interaction: bool = False,
    feature_ratio: bool = False,
    interaction_threshold: float = 0.01,
    transform_target: bool = False,
    transform_target_method: str = "box-cox",
    data_split_shuffle: bool = True,
    data_split_stratify: Union[bool, List[str]] = False,  # added in pycaret==2.2
    fold_strategy: Union[str, Any] = "kfold",  # added in pycaret==2.2
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
    log_plots: Union[bool, list] = False,
    log_profile: bool = False,
    log_data: bool = False,
    silent: bool = False,
    verbose: bool = True,
    profile: bool = False,
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
    >>> boston = get_data('boston')
    
    >>> experiment_name = setup(data = boston,  target = 'medv')

    'boston' is a pandas.DataFrame and 'medv' is the name of target column.

    Parameters
    ----------
    data : pandas.DataFrame
        Shape (n_samples, n_features) where n_samples is the number of samples and 
        n_features is the number of features.

    target: str
        Name of target column to be passed in as string. 
    
    train_size: float, default = 0.7
        Size of the training set. By default, 70% of the data will be used for training 
        and validation. The remaining data will be used for test / hold-out set.
    
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
        Features are compressed using method defined in high_cardinality_method param.
    
    high_cardinality_method: str, default = 'frequency'
        When method set to 'frequency' it will replace the original value of feature
        with the frequency distribution and convert the feature into numeric. Other
        available method is 'clustering' which performs the clustering on statistical
        attribute of data and replaces the original value of feature with cluster label.
        The number of clusters is determined using a combination of Calinski-Harabasz and 
        Silhouette criterion. 
    
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
    
    remove_perfect_collinearity: bool, default = True
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
    
    polynomial_degree: int, default = 2
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
        When a dataset contains features that have related characteristics, the 
        group_features param can be used for statistical feature extraction. For example, 
        if a dataset has numeric features that are related with each other 
        (i.e 'Col1', 'Col2', 'Col3'), a list containing the column names can be passed 
        under group_features to extract statistical information such as the mean, median, 
        mode and standard deviation.
    
    group_names: list, default = None
        When group_features is passed, a name of the group can be passed into the 
        group_names param as a list containing strings. The length of a group_names 
        list must equal to the length of group_features. When the length doesn't match 
        or the name is not passed, new features are sequentially named such as 
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
        Threshold used for feature selection (for newly created polynomial features).
        A higher value will result in a higher feature space. It is recommended to do 
        multiple trials with different values of feature_selection_threshold specially 
        in cases where polynomial_features and feature_interaction are used. Setting a 
        very low value may be efficient but could result in under-fitting.
    
    feature_selection_method: str, default = 'classic'
        Can be either 'classic' or 'boruta'. Selects the algorithm responsible for
        choosing a subset of features. For the 'classic' selection method, PyCaret will 
        use various permutation importance techniques. For the 'boruta' algorithm, PyCaret 
        will create an instance of boosted trees model, which will iterate with 
        permutation over all features and choose the best ones based on the distributions 
        of feature importance.
    
    feature_interaction: bool, default = False 
        When set to True, it will create new features by interacting (a * b) for all 
        numeric variables in the dataset including polynomial and trigonometric features 
        (if created). This feature is not scalable and may not work as expected on 
        datasets with large feature space.
    
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
    
    transform_target: bool, default = False
        When set to True, target variable is transformed using the method defined in
        transform_target_method param. Target transformation is applied separately from 
        feature transformations. 
    
    transform_target_method: str, default = 'box-cox'
        'Box-cox' and 'yeo-johnson' methods are supported. Box-Cox requires input data to 
        be strictly positive, while Yeo-Johnson supports both positive or negative data.
        When transform_target_method is 'box-cox' and target variable contains negative
        values, method is internally forced to 'yeo-johnson' to avoid exceptions.

    data_split_shuffle: bool, default = True
        If set to False, prevents shuffling of rows when splitting data.

    data_split_stratify: bool or list, default = False
        Whether to stratify when splitting data.
        If True, will stratify by the target column. If False, will not stratify.
        If list is passed, will stratify by the columns with the names in the list.
        Requires data_split_shuffle to be set to True.

    fold_strategy: str or scikit-learn compatible CV generator object, default = 'kfold'
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
        If set to 'force', will try to use GPU with all algorithms that support it,
        and raise exceptions if they are unavailable.
        If set to True, will use GPU with algorithms that support it, and fall
        back to CPU if they are unavailable.
        If set to False, will only use CPU.

        GPU enabled algorithms:
        
        - CatBoost
        - XGBoost
        - LightGBM - requires https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
        - Linear Regression, Lasso, Ridge, ElasticNet, SVR, KNN, Random Forest - requires cuML >= 0.15 to be installed.
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

    log_plots: bool or list, default = False
        When set to True, AUC, Confusion Matrix and Feature Importance plots will be logged.
        When set to a list of plot IDs (consult ``plot_model``), will log those plots.
        If False, will log no plots. 

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
    
    Returns
    -------
    info_grid
        Information grid is printed.

    environment
        This function returns various outputs that are stored in variable
        as tuple. They are used by other functions in pycaret.
      
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
    )


def compare_models(
    include: Optional[
        List[Union[str, Any]]
    ] = None,  # changed whitelist to include in pycaret==2.1
    exclude: Optional[List[str]] = None,  # changed blacklist to exclude in pycaret==2.1
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    sort: str = "R2",
    n_select: int = 1,
    budget_time: Optional[float] = None,  # added in pycaret==2.1.0
    turbo: bool = True,
    errors: str = "ignore",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
):

    """
    This function train all the models available in the model library and scores them 
    using Kfold Cross Validation. The output prints a score grid with MAE, MSE 
    RMSE, R2, RMSLE and MAPE (averaged accross folds), determined by fold parameter.
    
    This function returns the best model based on metric defined in sort parameter. 
    
    To select top N models, use n_select parameter that is set to 1 by default.
    Where n_select parameter > 1, it will return a list of trained model objects.

    When turbo is set to True ('kr', 'ard' and 'mlp') are excluded due to longer
    training times. By default turbo param is set to True.

    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston,  target = 'medv')
    >>> best_model = compare_models() 

    This will return the averaged score grid of all models except 'kr', 'ard' 
    and 'mlp'. When turbo param is set to False, all models including 'kr',
    'ard' and 'mlp' are used, but this may result in longer training times.
    
    >>> best_model = compare_models(exclude = ['knn','gbr'], turbo = False) 

    This will return a comparison of all models except K Nearest Neighbour and
    Gradient Boosting Regressor.
    
    >>> best_model = compare_models(exclude = ['knn','gbr'] , turbo = True) 

    This will return a comparison of all models except K Nearest Neighbour, 
    Gradient Boosting Regressor, Kernel Ridge Regressor, Automatic Relevance
    Determinant and Multi Level Perceptron.
        
    Parameters
    ----------
    exclude: list of strings, default = None
        In order to omit certain models from the comparison model ID's can be passed as 
        a list of strings in exclude param. 

    include: list of strings or objects, default = None
        In order to run only certain models for the comparison, the model ID's can be 
        passed as a list of strings in include param. The list can also include estimator
        objects to be compared.

    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.

    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.
  
    cross_validation: bool, default = True
        When cross_validation set to False fold parameter is ignored and models are trained
        on entire training dataset, returning metrics calculated using the train (holdout) set.

    sort: str, default = 'R2'
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
        Scoring metrics used are MAE, MSE, RMSE, R2, RMSLE and MAPE
        Mean and standard deviation of the scores across the folds is
        also returned.

    list or model
        List or one fitted model objects that were compared.

    Warnings
    --------
    - compare_models() though attractive, might be time consuming with large 
      datasets. By default turbo is set to True, which excludes models that
      have longer training times. Changing turbo parameter to False may result 
      in very high training times with datasets where number of samples exceed 
      10,000.

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
):

    """
    This function creates a model and scores it using Kfold Cross Validation. 
    The output prints a score grid that shows MAE, MSE, RMSE, RMSLE, R2 and 
    MAPE by fold (default = 10 Fold).

    This function returns a trained model object. 

    setup() function must be called before using create_model()

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston,  target = 'medv')
    
    >>> lr = create_model('lr')

    This will create a trained Linear Regression model.

    Parameters
    ----------
    estimator : str / object, default = None
        Enter ID of the estimators available in model library or pass an untrained 
        model object consistent with fit / predict API to train and evaluate model. 
        All estimators support binary or multiclass problem. List of estimators in 
        model library (ID - Name):

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
        * 'svm' - Support Vector Machine                           
        * 'knn' - K Neighbors Regressor                           
        * 'dt' - Decision Tree                                    
        * 'rf' - Random Forest                                    
        * 'et' - Extra Trees Regressor                            
        * 'ada' - AdaBoost Regressor                              
        * 'gbr' - Gradient Boosting Regressor                               
        * 'mlp' - Multi Level Perceptron                          
        * 'xgboost' - Extreme Gradient Boosting                   
        * 'lightgbm' - Light Gradient Boosting                    
        * 'catboost' - CatBoost Regressor                         

    fold: integer or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.

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
        Scoring metrics used are R2, AUC, Recall, Precision, F1, 
        Kappa and MCC. Mean and standard deviation of the scores across 
        the folds are highlighted in yellow.

    model
        trained model object
  
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
):

    """
    This function tunes the hyperparameters of a model and scores it using Kfold 
    Cross Validation. The output prints the score grid that shows MAE, MSE, RMSE, 
    R2, RMSLE and MAPE by fold (by default = 10 Folds).

    This function returns a trained model object.  

    tune_model() only accepts a string parameter for estimator.

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston,  target = 'medv')
    >>> xgboost = create_model('xgboost')
    >>> tuned_xgboost = tune_model(xgboost) 

    This will tune the hyperparameters of Extreme Gradient Boosting Regressor.

    Parameters
    ----------
    estimator : object, default = None

    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use KFold CV with that many folds.
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

    optimize: str, default = 'R2'
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
        Scoring metrics used are R2, AUC, Recall, Precision, F1, 
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
    optimize: str = "R2",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
) -> Any:
    """
    This function ensembles the trained base estimator using the method defined 
    in 'method' param (default = 'Bagging'). The output prints a score grid that 
    shows MAE, MSE, RMSE, R2, RMSLE and MAPE by fold (default CV = 10 Folds).

    This function returns a trained model object.  

    Model must be created using create_model() or tune_model().

    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston,  target = 'medv')
    >>> dt = create_model('dt')
    >>> ensembled_dt = ensemble_model(dt)

    This will return an ensembled Decision Tree model using 'Bagging'.

    Parameters
    ----------
    estimator : object, default = None

    method: str, default = 'Bagging'
        Bagging method will create an ensemble meta-estimator that fits base 
        regressor each on random subsets of the original dataset. The other
        available method is 'Boosting' that fits a regressor on the original 
        dataset and then fits additional copies of the regressor on the same 
        dataset but where the weights of instances are adjusted according to 
        the error of the current prediction. As such, subsequent regressors 
        focus more on difficult cases.
    
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

    optimize: str, default = 'R2'
        Only used when choose_better is set to True. optimize parameter is used
        to compare emsembled model with base estimator. Values accepted in 
        optimize parameter are 'R2', 'AUC', 'Recall', 'Precision', 'F1', 
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
        Scoring metrics used are MAE, MSE, RMSE, R2, RMSLE and MAPE.
        Mean and standard deviation of the scores across the folds are 
        also returned.

    model
        Trained ensembled model object.
    
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
    weights: Optional[List[float]] = None,  # added in pycaret==2.2.0
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
):

    """
    This function creates an ensemble meta-estimator that fits a base regressor on 
    the whole dataset. It then averages the predictions to form a final prediction. 
    By default, this function will use all estimators in the model library (excl. 
    the few estimators when turbo is True) or a specific trained estimator passed 
    as a list in estimator_list param. It scores it using Kfold Cross Validation. 
    The output prints the score grid that shows MAE, MSE, RMSE, R2, RMSLE and MAPE 
    by fold (default = 10 Fold). 

    This function returns a trained model object.  

    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston,  target = 'medv')
    >>> blend_all = blend_models() 

    This will result in VotingRegressor for all models in the library except 'ard',
    'kr' and 'mlp'.
    
    For specific models, you can use:

    >>> lr = create_model('lr')
    >>> rf = create_model('rf')
    >>> knn = create_model('knn')
    >>> blend_three = blend_models(estimator_list = [lr,rf,knn])

    This will create a VotingRegressor of lr, rf and knn.

    Parameters
    ----------
    estimator_list : str ('All') or list of objects, default = 'All'

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

    optimize: str, default = 'R2'
        Only used when choose_better is set to True. optimize parameter is used
        to compare emsembled model with base estimator. Values accepted in 
        optimize parameter are 'R2', 'AUC', 'Recall', 'Precision', 'F1', 
        'Kappa', 'MCC'.

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
        Scoring metrics used are MAE, MSE, RMSE, R2, RMSLE and MAPE. 
        Mean and standard deviation of the scores across the folds are 
        also returned.

    model
        Trained Voting Regressor model object. 
       
  
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
    method: str = "auto",
    restack: bool = True,
    choose_better: bool = False,
    optimize: str = "R2",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
):

    """
    This function trains a meta model and scores it using Kfold Cross Validation.
    The predictions from the base level models as passed in the estimator_list param 
    are used as input features for the meta model. The restacking parameter controls
    the ability to expose raw features to the meta model when set to True
    (default = False).

    The output prints a score grid that shows MAE, MSE, RMSE, R2, RMSLE and MAPE by 
    fold (default = 10 Folds).
    
    This function returns a trained model object. 

    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston,  target = 'medv')
    >>> dt = create_model('dt')
    >>> rf = create_model('rf')
    >>> ada = create_model('ada')
    >>> ridge = create_model('ridge')
    >>> knn = create_model('knn')
    >>> stacked_models = stack_models(estimator_list=[dt,rf,ada,ridge,knn])

    This will create a meta model that will use the predictions of all the 
    models provided in estimator_list param. By default, the meta model is 
    Linear Regression but can be changed with meta_model param.

    Parameters
    ----------
    estimator_list : list of objects

    meta_model : object, default = None
        If set to None, Linear Regression is used as a meta model.

    fold: int or scikit-learn compatible CV generator, default = None
        Controls cross-validation. If None, will use the CV generator defined in setup().
        If integer, will use StratifiedKFold CV with that many folds.
        When cross_validation is False, this parameter is ignored.

    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.

    restack: bool, default = True
        When restack is set to True, raw data will be exposed to meta model when
        making predictions, otherwise when False, only the predicted label or
        probabilities is passed to meta model when making final predictions.

    choose_better: bool, default = False
        When set to set to True, base estimator is returned when the metric doesn't 
        improve by ensemble_model. This gurantees the returned object would perform 
        atleast equivalent to base estimator created using create_model or model 
        returned by compare_models.

    optimize: str, default = 'R2'
        Only used when choose_better is set to True. optimize parameter is used
        to compare emsembled model with base estimator. Values accepted in 
        optimize parameter are 'R2', 'AUC', 'Recall', 'Precision', 'F1', 
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
        Scoring metrics used are MAE, MSE, RMSE, R2, RMSLE and MAPE.
        Mean and standard deviation of the scores across the folds are 
        also returned.

    model
        Trained model object.

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
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston,  target = 'medv')
    >>> lr = create_model('lr')
    >>> plot_model(lr)

    This will return an residuals plot of a trained Linear Regression model.

    Parameters
    ----------
    estimator : object, default = none
        A trained model object should be passed as an estimator. 
   
    plot : str, default = 'residual'
        Enter abbreviation of type of plot. The current list of plots supported are 
        (Plot - Name):

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
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston,  target = 'medv')
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
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston,  target = 'medv')
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
        This parameter is only needed when plot = 'correlation'. By default feature 
        is set to None which means the first column of the dataset will be used as a 
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

    """

    """
    Error Checking starts here
    
    """

    return pycaret.internal.tabular.interpret_model(
        estimator=estimator,
        plot=plot,
        feature=feature,
        observation=observation,
        **kwargs,
    )


def predict_model(
    estimator,
    data: Optional[pd.DataFrame] = None,
    round: int = 4,  # added in pycaret==2.2.0
    verbose: bool = True,
) -> pd.DataFrame:

    """
    This function is used to predict target value on the new dataset using a trained 
    estimator. New unseen data can be passed to data param as pandas.DataFrame.
    If data is not passed, the test / hold-out set separated at the time of 
    setup() is used to generate predictions. 
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston,  target = 'medv')
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
    
    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to. 

    verbose: bool, default = True
        Holdout score grid is not printed when verbose is set to False.

    Returns
    -------
    Predictions
        Predictions (Label) column attached to the original dataset
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
) -> Any:  # added in pycaret==2.2.0

    """
    This function fits the estimator onto the complete dataset passed during the
    setup() stage. The purpose of this function is to prepare for final model
    deployment after experimentation. 
    
    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> boston = get_data('boston')
    >>> experiment_name = setup(data = boston,  target = 'medv')
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


def automl(optimize: str = "R2", use_holdout: bool = False) -> Any:

    """
    This function returns the best model out of all models created in 
    current active environment based on metric defined in optimize parameter. 

    Parameters
    ----------
    optimize : str, default = 'R2'
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
    id: str, name: str, score_func: type, greater_is_better: bool = True, **kwargs,
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

    greater_is_better: bool, default = True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

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
        target="pred",
        greater_is_better=greater_is_better,
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
