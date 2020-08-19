# Module: Classification
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT
# Release: PyCaret 2.1
# Last modified : 19/08/2020

def setup(data,  
          target,   
          train_size = 0.7, 
          sampling = True, 
          sample_estimator = None,
          categorical_features = None,
          categorical_imputation = 'constant',
          ordinal_features = None,
          high_cardinality_features = None,
          high_cardinality_method = 'frequency',
          numeric_features = None,
          numeric_imputation = 'mean',
          date_features = None,
          ignore_features = None,
          normalize = False,
          normalize_method = 'zscore',
          transformation = False,
          transformation_method = 'yeo-johnson',
          handle_unknown_categorical = True,
          unknown_categorical_method = 'least_frequent',
          pca = False,
          pca_method = 'linear',
          pca_components = None,
          ignore_low_variance = False,
          combine_rare_levels = False,
          rare_level_threshold = 0.10,
          bin_numeric_features = None,
          remove_outliers = False,
          outliers_threshold = 0.05,
          remove_multicollinearity = False,
          multicollinearity_threshold = 0.9,
          remove_perfect_collinearity = False, #added in pycaret==2.0.0
          create_clusters = False,
          cluster_iter = 20,
          polynomial_features = False,                 
          polynomial_degree = 2,                       
          trigonometry_features = False,               
          polynomial_threshold = 0.1,                 
          group_features = None,                        
          group_names = None,                         
          feature_selection = False,                     
          feature_selection_threshold = 0.8,             
          feature_selection_method = 'classic',
          feature_interaction = False,                   
          feature_ratio = False,                         
          interaction_threshold = 0.01,
          fix_imbalance = False, #added in pycaret==2.0.0
          fix_imbalance_method = None, #added in pycaret==2.0.0
          data_split_shuffle = True, #added in pycaret==2.0.0
          folds_shuffle = False, #added in pycaret==2.0.0
          n_jobs = -1, #added in pycaret==2.0.0
          use_gpu = False, #added in pycaret==2.1
          html = True, #added in pycaret==2.0.0
          session_id = None,
          log_experiment = False, #added in pycaret==2.0.0
          experiment_name = None, #added in pycaret==2.0.0
          log_plots = False, #added in pycaret==2.0.0
          log_profile = False, #added in pycaret==2.0.0
          log_data = False, #added in pycaret==2.0.0
          silent=False,
          verbose=True, #added in pycaret==2.0.0
          profile = False):
    
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
        Features are compressed using method defined in high_cardinality_method param.
    
    high_cardinality_method: string, default = 'frequency'
        When method set to 'frequency' it will replace the original value of feature
        with the frequency distribution and convert the feature into numeric. Other
        available method is 'clustering' which performs the clustering on statistical
        attribute of data and replaces the original value of feature with cluster label.
        The number of clusters is determined using a combination of Calinski-Harabasz and 
        Silhouette criterion. 
          
    numeric_features: string, default = None
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
        by default is 'classic'but could be 'boruta' which lead pycaret to create boruta selection
        algorithm instance, more in:
        https://pdfs.semanticscholar.org/85a8/b1d9c52f9f795fda7e12376e751526953f38.pdf%3E


    feature_selection_threshold: float, default = 0.8
        Threshold used for feature selection (including newly created polynomial features).
        A higher value will result in a higher feature space. It is recommended to do multiple
        trials with different values of feature_selection_threshold specially in cases where 
        polynomial_features and feature_interaction are used. Setting a very low value may be 
        efficient but could result in under-fitting.

    feature_selection_method: str, default = classic
        User can use 'classic' or 'boruta' algorithm selection which is responsible for
        choosing a subset of features. For 'classic' selection method pycaret using a varius
        permutation importance techiques. If 'boruta' algorithm is selected pycaret will create 
        an instance of boosted trees model, which iterate with permutation over all
        features and choose the best one base on distributions of feature importance.
    
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
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 

    #exception checking   
    import sys
    
    from pycaret.utils import __version__
    ver = __version__()

    import logging

    # create logger
    global logger

    logger = logging.getLogger('logs')
    logger.setLevel(logging.DEBUG)
    
    # create console handler and set level to debug

    if logger.hasHandlers():
        logger.handlers.clear()
        
    ch = logging.FileHandler('logs.log')
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    logger.info("PyCaret Classification Module")
    logger.info('version ' + str(ver))
    logger.info("Initializing setup()")

    #generate USI for mlflow tracking
    import secrets
    global USI
    USI = secrets.token_hex(nbytes=2)
    logger.info('USI: ' + str(USI))

    logger.info("""setup(data={}, target={}, train_size={}, sampling={}, sample_estimator={}, categorical_features={}, categorical_imputation={}, ordinal_features={},
                    high_cardinality_features={}, high_cardinality_method={}, numeric_features={}, numeric_imputation={}, date_features={}, ignore_features={}, normalize={},
                    normalize_method={}, transformation={}, transformation_method={}, handle_unknown_categorical={}, unknown_categorical_method={}, pca={}, pca_method={},
                    pca_components={}, ignore_low_variance={}, combine_rare_levels={}, rare_level_threshold={}, bin_numeric_features={}, remove_outliers={}, outliers_threshold={},
                    remove_multicollinearity={}, multicollinearity_threshold={}, remove_perfect_collinearity={}, create_clusters={}, cluster_iter={},
                    polynomial_features={}, polynomial_degree={}, trigonometry_features={}, polynomial_threshold={}, group_features={},
                    group_names={}, feature_selection={}, feature_selection_threshold={}, feature_interaction={}, feature_ratio={}, interaction_threshold={},
                    fix_imbalance={}, fix_imbalance_method={}, data_split_shuffle={}, folds_shuffle={}, n_jobs={}, html={}, session_id={}, log_experiment={},
                    experiment_name={}, log_plots={}, log_profile={}, log_data={}, silent={}, verbose={}, profile={})""".format(\
            str(data.shape), str(target), str(train_size), str(sampling), str(sample_estimator), str(categorical_features), str(categorical_imputation), str(ordinal_features),\
            str(high_cardinality_features), str(high_cardinality_method), str(numeric_features), str(numeric_imputation), str(date_features), str(ignore_features),\
            str(normalize), str(normalize_method), str(transformation), str(transformation_method), str(handle_unknown_categorical), str(unknown_categorical_method), str(pca),\
            str(pca_method), str(pca_components), str(ignore_low_variance), str(combine_rare_levels), str(rare_level_threshold), str(bin_numeric_features), str(remove_outliers),\
            str(outliers_threshold), str(remove_multicollinearity), str(multicollinearity_threshold), str(remove_perfect_collinearity), str(create_clusters), str(cluster_iter),\
            str(polynomial_features), str(polynomial_degree), str(trigonometry_features), str(polynomial_threshold), str(group_features), str(group_names),\
            str(feature_selection), str(feature_selection_threshold), str(feature_interaction), str(feature_ratio), str(interaction_threshold), str(fix_imbalance),\
            str(fix_imbalance_method), str(data_split_shuffle), str(folds_shuffle), str(n_jobs), str(html), str(session_id), str(log_experiment), str(experiment_name),\
            str(log_plots), str(log_profile), str(log_data), str(silent), str(verbose), str(profile)))

    #logging environment and libraries
    logger.info("Checking environment")
    
    from platform import python_version, platform, python_build, machine

    logger.info("python_version: " + str(python_version()))
    logger.info("python_build: " + str(python_build()))
    logger.info("machine: " + str(machine()))
    logger.info("platform: " + str(platform()))

    try:
        import psutil
        logger.info("Memory: " + str(psutil.virtual_memory()))
        logger.info("Physical Core: " + str(psutil.cpu_count(logical=False)))
        logger.info("Logical Core: " + str(psutil.cpu_count(logical=True)))
    except:
        logger.warning("cannot find psutil installation. memory not traceable. Install psutil using pip to enable memory logging. ")

    logger.info("Checking libraries")

    try:
        from pandas import __version__
        logger.info("pd==" + str(__version__))
    except:
        logger.warning("pandas not found")

    try:
        from numpy import __version__
        logger.info("numpy==" + str(__version__))
    except:
        logger.warning("numpy not found")

    try:
        from sklearn import __version__
        logger.info("sklearn==" + str(__version__))
    except:
        logger.warning("sklearn not found")

    try:
        from xgboost import __version__
        logger.info("xgboost==" + str(__version__))
    except:
        logger.warning("xgboost not found")

    try:
        from lightgbm import __version__
        logger.info("lightgbm==" + str(__version__))
    except:
        logger.warning("lightgbm not found")

    try:
        from catboost import __version__
        logger.info("catboost==" + str(__version__))
    except:
        logger.warning("catboost not found")

    try:
        from mlflow.version import VERSION
        import warnings
        warnings.filterwarnings('ignore') 
        logger.info("mlflow==" + str(VERSION))
    except:
        logger.warning("mlflow not found")

    #run_time
    import datetime, time
    runtime_start = time.time()

    logger.info("Checking Exceptions")

    #checking data type
    if hasattr(data,'shape') is False:
        sys.exit('(Type Error): data passed must be of type pandas.DataFrame')

    #checking train size parameter
    if type(train_size) is not float:
        sys.exit('(Type Error): train_size parameter only accepts float value.')
    
    #checking sampling parameter
    if type(sampling) is not bool:
        sys.exit('(Type Error): sampling parameter only accepts True or False.')
        
    #checking sampling parameter
    if target not in data.columns:
        sys.exit('(Value Error): Target parameter doesnt exist in the data provided.')   

    #checking session_id
    if session_id is not None:
        if type(session_id) is not int:
            sys.exit('(Type Error): session_id parameter must be an integer.')   
    
    #checking sampling parameter
    if type(profile) is not bool:
        sys.exit('(Type Error): profile parameter only accepts True or False.')
        
    #checking normalize parameter
    if type(normalize) is not bool:
        sys.exit('(Type Error): normalize parameter only accepts True or False.')
        
    #checking transformation parameter
    if type(transformation) is not bool:
        sys.exit('(Type Error): transformation parameter only accepts True or False.')
        
    #checking categorical imputation
    allowed_categorical_imputation = ['constant', 'mode']
    if categorical_imputation not in allowed_categorical_imputation:
        sys.exit("(Value Error): categorical_imputation param only accepts 'constant' or 'mode' ")
     
    #ordinal_features
    if ordinal_features is not None:
        if type(ordinal_features) is not dict:
            sys.exit("(Type Error): ordinal_features must be of type dictionary with column name as key and ordered values as list. ")
    
    #ordinal features check
    if ordinal_features is not None:
        data_cols = data.columns
        data_cols = data_cols.drop(target)
        ord_keys = ordinal_features.keys()
                        
        for i in ord_keys:
            if i not in data_cols:
                sys.exit("(Value Error) Column name passed as a key in ordinal_features param doesnt exist. ")
                
        for k in ord_keys:
            if data[k].nunique() != len(ordinal_features.get(k)):
                sys.exit("(Value Error) Levels passed in ordinal_features param doesnt match with levels in data. ")

        for i in ord_keys:
            value_in_keys = ordinal_features.get(i)
            value_in_data = list(data[i].unique().astype(str))
            for j in value_in_keys:
                if j not in value_in_data:
                    text =  "Column name '" + str(i) + "' doesnt contain any level named '" + str(j) + "'."
                    sys.exit(text)
    
    #high_cardinality_features
    if high_cardinality_features is not None:
        if type(high_cardinality_features) is not list:
            sys.exit("(Type Error): high_cardinality_features param only accepts name of columns as a list. ")
        
    if high_cardinality_features is not None:
        data_cols = data.columns
        data_cols = data_cols.drop(target)
        for i in high_cardinality_features:
            if i not in data_cols:
                sys.exit("(Value Error): Column type forced is either target column or doesn't exist in the dataset.")
                
    #high_cardinality_methods
    high_cardinality_allowed_methods = ['frequency', 'clustering']     
    if high_cardinality_method not in high_cardinality_allowed_methods:
        sys.exit("(Value Error): high_cardinality_method param only accepts 'frequency' or 'clustering' ")
        
    #checking numeric imputation
    allowed_numeric_imputation = ['mean', 'median']
    if numeric_imputation not in allowed_numeric_imputation:
        sys.exit("(Value Error): numeric_imputation param only accepts 'mean' or 'median' ")
        
    #checking normalize method
    allowed_normalize_method = ['zscore', 'minmax', 'maxabs', 'robust']
    if normalize_method not in allowed_normalize_method:
        sys.exit("(Value Error): normalize_method param only accepts 'zscore', 'minxmax', 'maxabs' or 'robust'. ")        
    
    #checking transformation method
    allowed_transformation_method = ['yeo-johnson', 'quantile']
    if transformation_method not in allowed_transformation_method:
        sys.exit("(Value Error): transformation_method param only accepts 'yeo-johnson' or 'quantile'. ")        
    
    #handle unknown categorical
    if type(handle_unknown_categorical) is not bool:
        sys.exit('(Type Error): handle_unknown_categorical parameter only accepts True or False.')
        
    #unknown categorical method
    unknown_categorical_method_available = ['least_frequent', 'most_frequent']
    
    if unknown_categorical_method not in unknown_categorical_method_available:
        sys.exit("(Type Error): unknown_categorical_method only accepts 'least_frequent' or 'most_frequent'.")
    
    #check pca
    if type(pca) is not bool:
        sys.exit('(Type Error): PCA parameter only accepts True or False.')
        
    #pca method check
    allowed_pca_methods = ['linear', 'kernel', 'incremental']
    if pca_method not in allowed_pca_methods:
        sys.exit("(Value Error): pca method param only accepts 'linear', 'kernel', or 'incremental'. ")    
    
    #pca components check
    if pca is True:
        if pca_method != 'linear':
            if pca_components is not None:
                if(type(pca_components)) is not int:
                    sys.exit("(Type Error): pca_components parameter must be integer when pca_method is not 'linear'. ")

    #pca components check 2
    if pca is True:
        if pca_method != 'linear':
            if pca_components is not None:
                if pca_components > len(data.columns)-1:
                    sys.exit("(Type Error): pca_components parameter cannot be greater than original features space.")                
 
    #pca components check 3
    if pca is True:
        if pca_method == 'linear':
            if pca_components is not None:
                if type(pca_components) is not float:
                    if pca_components > len(data.columns)-1: 
                        sys.exit("(Type Error): pca_components parameter cannot be greater than original features space or float between 0 - 1.")      

    #check ignore_low_variance
    if type(ignore_low_variance) is not bool:
        sys.exit('(Type Error): ignore_low_variance parameter only accepts True or False.')
        
    #check ignore_low_variance
    if type(combine_rare_levels) is not bool:
        sys.exit('(Type Error): combine_rare_levels parameter only accepts True or False.')
        
    #check rare_level_threshold
    if type(rare_level_threshold) is not float:
        sys.exit('(Type Error): rare_level_threshold must be a float between 0 and 1. ')
    
    #bin numeric features
    if bin_numeric_features is not None:
        all_cols = list(data.columns)
        all_cols.remove(target)
        
        for i in bin_numeric_features:
            if i not in all_cols:
                sys.exit("(Value Error): Column type forced is either target column or doesn't exist in the dataset.")

    #remove_outliers
    if type(remove_outliers) is not bool:
        sys.exit('(Type Error): remove_outliers parameter only accepts True or False.')    
    
    #outliers_threshold
    if type(outliers_threshold) is not float:
        sys.exit('(Type Error): outliers_threshold must be a float between 0 and 1. ')   
        
    #remove_multicollinearity
    if type(remove_multicollinearity) is not bool:
        sys.exit('(Type Error): remove_multicollinearity parameter only accepts True or False.')
        
    #multicollinearity_threshold
    if type(multicollinearity_threshold) is not float:
        sys.exit('(Type Error): multicollinearity_threshold must be a float between 0 and 1. ')  
    
    #create_clusters
    if type(create_clusters) is not bool:
        sys.exit('(Type Error): create_clusters parameter only accepts True or False.')
        
    #cluster_iter
    if type(cluster_iter) is not int:
        sys.exit('(Type Error): cluster_iter must be a integer greater than 1. ')                 

    #polynomial_features
    if type(polynomial_features) is not bool:
        sys.exit('(Type Error): polynomial_features only accepts True or False. ')   
    
    #polynomial_degree
    if type(polynomial_degree) is not int:
        sys.exit('(Type Error): polynomial_degree must be an integer. ')
        
    #polynomial_features
    if type(trigonometry_features) is not bool:
        sys.exit('(Type Error): trigonometry_features only accepts True or False. ')    
        
    #polynomial threshold
    if type(polynomial_threshold) is not float:
        sys.exit('(Type Error): polynomial_threshold must be a float between 0 and 1. ')      
        
    #group features
    if group_features is not None:
        if type(group_features) is not list:
            sys.exit('(Type Error): group_features must be of type list. ')     
    
    if group_names is not None:
        if type(group_names) is not list:
            sys.exit('(Type Error): group_names must be of type list. ')         
    
    #cannot drop target
    if ignore_features is not None:
        if target in ignore_features:
            sys.exit("(Value Error): cannot drop target column. ")  
                
    #feature_selection
    if type(feature_selection) is not bool:
        sys.exit('(Type Error): feature_selection only accepts True or False. ')   
        
    #feature_selection_threshold
    if type(feature_selection_threshold) is not float:
        sys.exit('(Type Error): feature_selection_threshold must be a float between 0 and 1. ')  

    #feature_selection_method
    if feature_selection_method not in ['boruta', 'classic']:
        sys.exit("(Type Error): feature_selection_method must be string 'boruta', 'classic'")  

    #feature_interaction
    if type(feature_interaction) is not bool:
        sys.exit('(Type Error): feature_interaction only accepts True or False. ')  
        
    #feature_ratio
    if type(feature_ratio) is not bool:
        sys.exit('(Type Error): feature_ratio only accepts True or False. ')     
        
    #interaction_threshold
    if type(interaction_threshold) is not float:
        sys.exit('(Type Error): interaction_threshold must be a float between 0 and 1. ')  
        
    #forced type check
    all_cols = list(data.columns)
    all_cols.remove(target)
    
    #categorical
    if categorical_features is not None:
        for i in categorical_features:
            if i not in all_cols:
                sys.exit("(Value Error): Column type forced is either target column or doesn't exist in the dataset.")
        
    #numeric
    if numeric_features is not None:
        for i in numeric_features:
            if i not in all_cols:
                sys.exit("(Value Error): Column type forced is either target column or doesn't exist in the dataset.")    
    
    #date features
    if date_features is not None:
        for i in date_features:
            if i not in all_cols:
                sys.exit("(Value Error): Column type forced is either target column or doesn't exist in the dataset.")      
    
    #drop features
    if ignore_features is not None:
        for i in ignore_features:
            if i not in all_cols:
                sys.exit("(Value Error): Feature ignored is either target column or doesn't exist in the dataset.") 
    
    #log_experiment
    if type(log_experiment) is not bool:
        sys.exit("(Type Error): log_experiment parameter only accepts True or False. ")

    #log_profile
    if type(log_profile) is not bool:
        sys.exit("(Type Error): log_profile parameter only accepts True or False. ")

    #experiment_name
    if experiment_name is not None:
        if type(experiment_name) is not str:
            sys.exit("(Type Error): experiment_name parameter must be string if not None. ")
      
    #silent
    if type(silent) is not bool:
        sys.exit("(Type Error): silent parameter only accepts True or False. ")
    
    #remove_perfect_collinearity
    if type(remove_perfect_collinearity) is not bool:
        sys.exit('(Type Error): remove_perfect_collinearity parameter only accepts True or False.')

    #html
    if type(html) is not bool:
        sys.exit('(Type Error): html parameter only accepts True or False.')

    #use_gpu
    if type(use_gpu) is not bool:
        sys.exit('(Type Error): use_gpu parameter only accepts True or False.')

    #folds_shuffle
    if type(folds_shuffle) is not bool:
        sys.exit('(Type Error): folds_shuffle parameter only accepts True or False.')

    #data_split_shuffle
    if type(data_split_shuffle) is not bool:
        sys.exit('(Type Error): data_split_shuffle parameter only accepts True or False.')

    #log_plots
    if type(log_plots) is not bool:
        sys.exit('(Type Error): log_plots parameter only accepts True or False.')

    #log_data
    if type(log_data) is not bool:
        sys.exit('(Type Error): log_data parameter only accepts True or False.')

    #log_profile
    if type(log_profile) is not bool:
        sys.exit('(Type Error): log_profile parameter only accepts True or False.')

    #fix_imbalance
    if type(fix_imbalance) is not bool:
        sys.exit('(Type Error): fix_imbalance parameter only accepts True or False.')

    #fix_imbalance_method
    if fix_imbalance:
        if fix_imbalance_method is not None:
            if hasattr(fix_imbalance_method, 'fit_sample'):
                pass
            else:
                sys.exit('(Type Error): fix_imbalance_method must contain resampler with fit_sample method.')

    logger.info("Preloading libraries")

    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import os
    
    #pandas option
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
   
    #global html_param
    global html_param
    
    #create html_param
    html_param = html

    #silent parameter to also set sampling to False
    if silent:
        sampling = False

    logger.info("Preparing display monitor")

    #progress bar
    if sampling:
        max_steps = 10 + 3
    else:
        max_steps = 3
        
    progress = ipw.IntProgress(value=0, min=0, max=max_steps, step=1 , description='Processing: ')
    if verbose:
        if html_param:
            display(progress)
    
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    if verbose:
        if html_param:
            display(monitor, display_id = 'monitor')
    
    logger.info("Importing libraries")

    #general dependencies
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import random
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    
    #setting sklearn config to print all parameters including default
    import sklearn
    sklearn.set_config(print_changed_only=False)

    #define highlight function for function grid to display
    def highlight_max(s):
        is_max = s == True
        return ['background-color: lightgreen' if v else '' for v in is_max]
        
    #cufflinks
    import cufflinks as cf
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)
    
    #declaring global variables to be accessed by other functions
    logger.info("Declaring global variables")
    global X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, experiment__,\
        folds_shuffle_param, n_jobs_param, create_model_container, master_model_container,\
        display_container, exp_name_log, logging_param, log_plots_param,\
        fix_imbalance_param, fix_imbalance_method_param, data_before_preprocess,\
        target_param, gpu_param

    logger.info("Copying data for preprocessing")
    
    #copy original data for pandas profiler
    data_before_preprocess = data.copy()
    
    #generate seed to be used globally
    if session_id is None:
        seed = random.randint(150,9000)
    else:
        seed = session_id
        
    """
    preprocessing starts here
    """
    
    monitor.iloc[1,1:] = 'Preparing Data for Modeling'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
            
    #define parameters for preprocessor
    
    logger.info("Declaring preprocessing parameters")

    #categorical features
    if categorical_features is None:
        cat_features_pass = []
    else:
        cat_features_pass = categorical_features
    
    #numeric features
    if numeric_features is None:
        numeric_features_pass = []
    else:
        numeric_features_pass = numeric_features
     
    #drop features
    if ignore_features is None:
        ignore_features_pass = []
    else:
        ignore_features_pass = ignore_features
     
    #date features
    if date_features is None:
        date_features_pass = []
    else:
        date_features_pass = date_features
        
    #categorical imputation strategy
    if categorical_imputation == 'constant':
        categorical_imputation_pass = 'not_available'
    elif categorical_imputation == 'mode':
        categorical_imputation_pass = 'most frequent'
    
    #transformation method strategy
    if transformation_method == 'yeo-johnson':
        trans_method_pass = 'yj'
    elif transformation_method == 'quantile':
        trans_method_pass = 'quantile'
    
    #pass method
    if pca_method == 'linear':
        pca_method_pass = 'pca_liner'
            
    elif pca_method == 'kernel':
        pca_method_pass = 'pca_kernal'
            
    elif pca_method == 'incremental':
        pca_method_pass = 'incremental'
            
    elif pca_method == 'pls':
        pca_method_pass = 'pls'
        
    #pca components
    if pca is True:
        if pca_components is None:
            if pca_method == 'linear':
                pca_components_pass = 0.99
            else:
                pca_components_pass = int((len(data.columns)-1)*0.5)
                
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
    
    #trignometry
    if trigonometry_features is False:
        trigonometry_features_pass = []
    else:
        trigonometry_features_pass = ['sin', 'cos', 'tan']
    
    #group features
    #=============#
    
    #apply grouping
    if group_features is not None:
        apply_grouping_pass = True
    else:
        apply_grouping_pass = False
    
    #group features listing
    if apply_grouping_pass is True:
        
        if type(group_features[0]) is str:
            group_features_pass = []
            group_features_pass.append(group_features)
        else:
            group_features_pass = group_features
            
    else:
        
        group_features_pass = [[]]
    
    #group names
    if apply_grouping_pass is True:

        if (group_names is None) or (len(group_names) != len(group_features_pass)):
            group_names_pass = list(np.arange(len(group_features_pass)))
            group_names_pass = ['group_' + str(i) for i in group_names_pass]

        else:
            group_names_pass = group_names
            
    else:
        group_names_pass = []
    
    #feature interactions
    
    if feature_interaction or feature_ratio:
        apply_feature_interactions_pass = True
    else:
        apply_feature_interactions_pass = False
    
    interactions_to_apply_pass = []
    
    if feature_interaction:
        interactions_to_apply_pass.append('multiply')
    
    if feature_ratio:
        interactions_to_apply_pass.append('divide')
    
    #unknown categorical
    if unknown_categorical_method == 'least_frequent':
        unknown_categorical_method_pass = 'least frequent'
    elif unknown_categorical_method == 'most_frequent':
        unknown_categorical_method_pass = 'most frequent'
    
    #ordinal_features
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
        
    if high_cardinality_method == 'frequency':
        cardinal_method_pass = 'count'
    elif high_cardinality_method == 'clustering':
        cardinal_method_pass = 'cluster'
        
    if apply_cardinality_reduction_pass:
        cardinal_features_pass = high_cardinality_features
    else:
        cardinal_features_pass = []
    
    if silent:
        display_dtypes_pass = False
    else:
        display_dtypes_pass = True

    logger.info("Importing preprocessing module")

    #import library
    import pycaret.preprocess as preprocess
    
    logger.info("Creating preprocessing pipeline")

    data = preprocess.Preprocess_Path_One(train_data = data, 
                                          target_variable = target,
                                          categorical_features = cat_features_pass,
                                          apply_ordinal_encoding = apply_ordinal_encoding_pass,
                                          ordinal_columns_and_categories = ordinal_columns_and_categories_pass,
                                          apply_cardinality_reduction = apply_cardinality_reduction_pass, 
                                          cardinal_method = cardinal_method_pass, 
                                          cardinal_features = cardinal_features_pass, 
                                          numerical_features = numeric_features_pass,
                                          time_features = date_features_pass,
                                          features_todrop = ignore_features_pass,
                                          numeric_imputation_strategy = numeric_imputation,
                                          categorical_imputation_strategy = categorical_imputation_pass,
                                          scale_data = normalize,
                                          scaling_method = normalize_method,
                                          Power_transform_data = transformation,
                                          Power_transform_method = trans_method_pass,
                                          apply_untrained_levels_treatment= handle_unknown_categorical, 
                                          untrained_levels_treatment_method = unknown_categorical_method_pass,
                                          apply_pca = pca,
                                          pca_method = pca_method_pass,
                                          pca_variance_retained_or_number_of_components = pca_components_pass, 
                                          apply_zero_nearZero_variance = ignore_low_variance, 
                                          club_rare_levels = combine_rare_levels,
                                          rara_level_threshold_percentage = rare_level_threshold, 
                                          apply_binning = apply_binning_pass, 
                                          features_to_binn = features_to_bin_pass, 
                                          remove_outliers = remove_outliers, 
                                          outlier_contamination_percentage = outliers_threshold, 
                                          outlier_methods = ['pca'],
                                          remove_multicollinearity = remove_multicollinearity, 
                                          maximum_correlation_between_features = multicollinearity_threshold, 
                                          remove_perfect_collinearity = remove_perfect_collinearity,
                                          cluster_entire_data = create_clusters, 
                                          range_of_clusters_to_try = cluster_iter, 
                                          apply_polynomial_trigonometry_features = polynomial_features, 
                                          max_polynomial = polynomial_degree, 
                                          trigonometry_calculations = trigonometry_features_pass, 
                                          top_poly_trig_features_to_select_percentage = polynomial_threshold, 
                                          apply_grouping = apply_grouping_pass, 
                                          features_to_group_ListofList = group_features_pass, 
                                          group_name = group_names_pass, 
                                          apply_feature_selection = feature_selection, 
                                          feature_selection_top_features_percentage = feature_selection_threshold, 
                                          feature_selection_method = feature_selection_method,
                                          apply_feature_interactions = apply_feature_interactions_pass, 
                                          feature_interactions_to_apply = interactions_to_apply_pass, 
                                          feature_interactions_top_features_to_select_percentage=interaction_threshold, 
                                          display_types = display_dtypes_pass, #this is for inferred input box
                                          target_transformation = False, #not needed for classification
                                          random_state = seed)

    progress.value += 1
    logger.info("Preprocessing pipeline created successfully")

    if hasattr(preprocess.dtypes, 'replacement'):
            label_encoded = preprocess.dtypes.replacement
            label_encoded = str(label_encoded).replace("'", '')
            label_encoded = str(label_encoded).replace("{", '')
            label_encoded = str(label_encoded).replace("}", '')

    else:
        label_encoded = 'None'
    
    try:
        res_type = ['quit','Quit','exit','EXIT','q','Q','e','E','QUIT','Exit']
        res = preprocess.dtypes.response

        if res in res_type:
            sys.exit("(Process Exit): setup has been interupted with user command 'quit'. setup must rerun." )
            
    except:
        logger.error("(Process Exit): setup has been interupted with user command 'quit'. setup must rerun.") 
        
    #save prep pipe
    prep_pipe = preprocess.pipe
    
    logger.info("Creating grid variables")

    #generate values for grid show
    missing_values = data_before_preprocess.isna().sum().sum()
    if missing_values > 0:
        missing_flag = True
    else:
        missing_flag = False
    
    if normalize is True:
        normalize_grid = normalize_method
    else:
        normalize_grid = 'None'
        
    if transformation is True:
        transformation_grid = transformation_method
    else:
        transformation_grid = 'None'
    
    if pca is True:
        pca_method_grid = pca_method
    else:
        pca_method_grid = 'None'
   
    if pca is True:
        pca_components_grid = pca_components_pass
    else:
        pca_components_grid = 'None'
        
    if combine_rare_levels:
        rare_level_threshold_grid = rare_level_threshold
    else:
        rare_level_threshold_grid = 'None'
    
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
        if 'float' in str(i):
            float_type += 1
        elif 'object' in str(i):
            cat_type += 1
        elif 'int' in str(i):
            float_type += 1
    
    """
    preprocessing ends here
    """
    
    #reset pandas option
    pd.reset_option("display.max_rows") 
    pd.reset_option("display.max_columns")

    logger.info("Creating global containers")

    #create an empty list for pickling later.
    experiment__ = []

    #create folds_shuffle_param
    folds_shuffle_param = folds_shuffle

    #create n_jobs_param
    n_jobs_param = n_jobs

    #create create_model_container
    create_model_container = []

    #create master_model_container
    master_model_container = []

    #create display container
    display_container = []

    #create logging parameter
    logging_param = log_experiment

    #create exp_name_log param incase logging is False
    exp_name_log = 'no_logging'
    
    #create an empty log_plots_param
    if log_plots:
        log_plots_param = True
    else:
        log_plots_param = False

    #create a fix_imbalance_param and fix_imbalance_method_param
    fix_imbalance_param = fix_imbalance
    fix_imbalance_method_param = fix_imbalance_method
    
    if fix_imbalance_method_param is None:
        fix_imbalance_model_name = 'SMOTE'
    else:
        fix_imbalance_model_name = str(fix_imbalance_method_param).split("(")[0]

    # create target_param var
    target_param = target

    # create gpu_param var
    gpu_param = use_gpu

    #sample estimator
    if sample_estimator is None:
        model = LogisticRegression()
    else:
        model = sample_estimator
        
    model_name = str(model).split("(")[0]
    if 'CatBoostClassifier' in model_name:
        model_name = 'CatBoostClassifier'
        
    #creating variables to be used later in the function
    X = data.drop(target,axis=1)
    y = data[target]
    
    #determining target type
    if y.value_counts().count() > 2:
        target_type = 'Multiclass'
    else:
        target_type = 'Binary'
    
    progress.value += 1
    
    if sampling is True and data.shape[0] > 25000: #change this back to 25000
        
        logger.info("Sampling dataset")

        split_perc = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
        split_perc_text = ['10%','20%','30%','40%','50%','60%', '70%', '80%', '90%', '100%']
        split_perc_tt = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
        split_perc_tt_total = []
        split_percent = []

        metric_results = []
        metric_name = []
        
        counter = 0
        
        for i in split_perc:
            
            progress.value += 1
            
            t0 = time.time()
            
            '''
            MONITOR UPDATE STARTS
            '''
            
            perc_text = split_perc_text[counter]
            monitor.iloc[1,1:] = 'Fitting Model on ' + perc_text + ' sample'
            if verbose:
                if html_param:
                    update_display(monitor, display_id = 'monitor')

            '''
            MONITOR UPDATE ENDS
            '''
    
            X_, X__, y_, y__ = train_test_split(X, y, test_size=1-i, stratify=y, random_state=seed, shuffle=data_split_shuffle)
            X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=1-train_size, stratify=y_, random_state=seed, shuffle=data_split_shuffle)
            model.fit(X_train,y_train)
            pred_ = model.predict(X_test)
            try:
                pred_prob = model.predict_proba(X_test)[:,1]
            except:
                logger.warning("model has no predict_proba attribute.")
                pred_prob = 0
            
            #accuracy
            acc = metrics.accuracy_score(y_test,pred_)
            metric_results.append(acc)
            metric_name.append('Accuracy')
            split_percent.append(i)
            
            #auc
            if y.value_counts().count() > 2:
                pass
            else:
                try:
                    auc = metrics.roc_auc_score(y_test,pred_prob)
                    metric_results.append(auc)
                    metric_name.append('AUC')
                    split_percent.append(i)
                except:
                    pass
                
            #recall
            if y.value_counts().count() > 2:
                recall = metrics.recall_score(y_test,pred_, average='macro')
                metric_results.append(recall)
                metric_name.append('Recall')
                split_percent.append(i)
            else:    
                recall = metrics.recall_score(y_test,pred_)
                metric_results.append(recall)
                metric_name.append('Recall')
                split_percent.append(i)
                
            #precision
            if y.value_counts().count() > 2:
                precision = metrics.precision_score(y_test,pred_, average='weighted')
                metric_results.append(precision)
                metric_name.append('Precision')
                split_percent.append(i)
            else:    
                precision = metrics.precision_score(y_test,pred_)
                metric_results.append(precision)
                metric_name.append('Precision')
                split_percent.append(i)                

            #F1
            if y.value_counts().count() > 2:
                f1 = metrics.f1_score(y_test,pred_, average='weighted')
                metric_results.append(f1)
                metric_name.append('F1')
                split_percent.append(i)
            else:    
                f1 = metrics.precision_score(y_test,pred_)
                metric_results.append(f1)
                metric_name.append('F1')
                split_percent.append(i)
                
            #Kappa
            kappa = metrics.cohen_kappa_score(y_test,pred_)
            metric_results.append(kappa)
            metric_name.append('Kappa')
            split_percent.append(i)
            
            t1 = time.time()
                       
            '''
            Time calculation begins
            '''
          
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
                ETC = ttt + ' Seconds Remaining'

            else:
                ttt = str (ttt)
                ETC = ttt + ' Minutes Remaining'
                
            monitor.iloc[2,1:] = ETC
            if verbose:
                if html_param:
                    update_display(monitor, display_id = 'monitor')
            
            
            '''
            Time calculation Ends
            '''
            
            split_perc_tt_total = []
            counter += 1

        model_results = pd.DataFrame({'Sample' : split_percent, 'Metric' : metric_results, 'Metric Name': metric_name})
        fig = px.line(model_results, x='Sample', y='Metric', color='Metric Name', line_shape='linear', range_y = [0,1])
        fig.update_layout(plot_bgcolor='rgb(245,245,245)')
        title= str(model_name) + ' Metrics and Sample %'
        fig.update_layout(title={'text': title, 'y':0.95,'x':0.45,'xanchor': 'center','yanchor': 'top'})
        fig.show()
        
        monitor.iloc[1,1:] = 'Waiting for input'
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')
        
        
        print('Please Enter the sample % of data you would like to use for modeling. Example: Enter 0.3 for 30%.')
        print('Press Enter if you would like to use 100% of the data.')
                
        sample_size = input("Sample Size: ")
        
        if sample_size == '' or sample_size == '1':
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, stratify=y, random_state=seed, shuffle=data_split_shuffle)
        
        else:
            
            sample_n = float(sample_size)
            X_selected, X_discard, y_selected, y_discard = train_test_split(X, y, test_size=1-sample_n, stratify=y, 
                                                                random_state=seed, shuffle=data_split_shuffle)
            
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=1-train_size, stratify=y_selected, 
                                                                random_state=seed, shuffle=data_split_shuffle)

    else:
        
        monitor.iloc[1,1:] = 'Splitting Data'
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, stratify=y, random_state=seed, shuffle=data_split_shuffle)
        progress.value += 1

    '''
    Final display Starts
    '''
    clear_output()
    if profile:
        print('Setup Succesfully Completed! Loading Profile Now... Please Wait!')
    else:
        if verbose:
            print('Setup Succesfully Completed!')
        
    functions = pd.DataFrame ( [ ['session_id', seed ],
                                    ['Target Type', target_type],
                                    ['Label Encoded', label_encoded],
                                    ['Original Data', data_before_preprocess.shape ],
                                    ['Missing Values ', missing_flag],
                                    ['Numeric Features ', str(float_type) ],
                                    ['Categorical Features ', str(cat_type) ],
                                    ['Ordinal Features ', ordinal_features_grid],
                                    ['High Cardinality Features ', high_cardinality_features_grid],
                                    ['High Cardinality Method ', high_cardinality_method_grid],
                                    ['Sampled Data', '(' + str(X_train.shape[0] + X_test.shape[0]) + ', ' + str(data_before_preprocess.shape[1]) + ')' ], 
                                    ['Transformed Train Set', X_train.shape ], 
                                    ['Transformed Test Set',X_test.shape ],
                                    ['Numeric Imputer ', numeric_imputation],
                                    ['Categorical Imputer ', categorical_imputation],
                                    ['Normalize ', normalize ],
                                    ['Normalize Method ', normalize_grid ],
                                    ['Transformation ', transformation ],
                                    ['Transformation Method ', transformation_grid ],
                                    ['PCA ', pca],
                                    ['PCA Method ', pca_method_grid],
                                    ['PCA Components ', pca_components_grid],
                                    ['Ignore Low Variance ', ignore_low_variance],
                                    ['Combine Rare Levels ', combine_rare_levels],
                                    ['Rare Level Threshold ', rare_level_threshold_grid],
                                    ['Numeric Binning ', numeric_bin_grid],
                                    ['Remove Outliers ', remove_outliers],
                                    ['Outliers Threshold ', outliers_threshold_grid],
                                    ['Remove Multicollinearity ', remove_multicollinearity],
                                    ['Multicollinearity Threshold ', multicollinearity_threshold_grid],
                                    ['Clustering ', create_clusters],
                                    ['Clustering Iteration ', cluster_iter_grid],
                                    ['Polynomial Features ', polynomial_features],
                                    ['Polynomial Degree ', polynomial_degree_grid],
                                    ['Trignometry Features ', trigonometry_features],
                                    ['Polynomial Threshold ', polynomial_threshold_grid],
                                    ['Group Features ', group_features_grid],
                                    ['Feature Selection ', feature_selection],
                                    ['Features Selection Threshold ', feature_selection_threshold_grid],
                                    ['Feature Interaction ', feature_interaction], 
                                    ['Feature Ratio ', feature_ratio], 
                                    ['Interaction Threshold ', interaction_threshold_grid], 
                                    ['Fix Imbalance', fix_imbalance_param],
                                    ['Fix Imbalance Method', fix_imbalance_model_name] 
                                ], columns = ['Description', 'Value'] )
    
    functions_ = functions.style.apply(highlight_max)
    if verbose:
        if html_param:
            display(functions_)
        else:
            print(functions_.data)
    
    if profile:
        try:
            import pandas_profiling
            pf = pandas_profiling.ProfileReport(data_before_preprocess)
            clear_output()
            display(pf)
        except:
            print('Data Profiler Failed. No output to show, please continue with Modeling.')
            logger.error("Data Profiler Failed. No output to show, please continue with Modeling.")
        
    '''
    Final display Ends
    '''   
    
    #log into experiment
    experiment__.append(('Classification Setup Config', functions))
    experiment__.append(('X_training Set', X_train))
    experiment__.append(('y_training Set', y_train))
    experiment__.append(('X_test Set', X_test))
    experiment__.append(('y_test Set', y_test))
    experiment__.append(('Transformation Pipeline', prep_pipe))

    #end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    if logging_param:
        
        logger.info("Logging experiment in MLFlow")

        import mlflow
        from pathlib import Path

        if experiment_name is None:
            exp_name_ = 'clf-default-name'
        else:
            exp_name_ = experiment_name

        URI = secrets.token_hex(nbytes=4)    
        exp_name_log = exp_name_
        
        try:
            mlflow.create_experiment(exp_name_log)
        except:
            pass

        #mlflow logging
        mlflow.set_experiment(exp_name_log)

        run_name_ = 'Session Initialized ' + str(USI)

        with mlflow.start_run(run_name=run_name_) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id
            
            k = functions.copy()
            k.set_index('Description',drop=True,inplace=True)
            kdict = k.to_dict()
            params = kdict.get('Value')
            mlflow.log_params(params)

            #set tag of compare_models
            mlflow.set_tag("Source", "setup")
            
            import secrets
            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI) 
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log the transformation pipeline
            logger.info("SubProcess save_model() called ==================================")
            save_model(prep_pipe, 'Transformation Pipeline', verbose=False)
            logger.info("SubProcess save_model() end ==================================")
            mlflow.log_artifact('Transformation Pipeline' + '.pkl')
            os.remove('Transformation Pipeline.pkl')

            # Log pandas profile
            if log_profile:
                import pandas_profiling
                pf = pandas_profiling.ProfileReport(data_before_preprocess)
                pf.to_file("Data Profile.html")
                mlflow.log_artifact("Data Profile.html")
                os.remove("Data Profile.html")
                clear_output()
                display(functions_)

            # Log training and testing set
            if log_data:
                X_train.join(y_train).to_csv('Train.csv')
                X_test.join(y_test).to_csv('Test.csv')
                mlflow.log_artifact("Train.csv")
                mlflow.log_artifact("Test.csv")
                os.remove('Train.csv')
                os.remove('Test.csv')

    logger.info("create_model_container " + str(len(create_model_container)))
    logger.info("master_model_container " + str(len(master_model_container)))
    logger.info("display_container " + str(len(display_container)))

    logger.info(str(prep_pipe))
    logger.info("setup() succesfully completed......................................")

    return X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, experiment__,\
        folds_shuffle_param, n_jobs_param, html_param, create_model_container, master_model_container,\
        display_container, exp_name_log, logging_param, log_plots_param, USI,\
        fix_imbalance_param, fix_imbalance_method_param, logger, data_before_preprocess, target_param,\
        gpu_param

def compare_models(exclude = None,
                   include = None, #added in pycaret==2.0.0
                   fold = 10, 
                   round = 4, 
                   sort = 'Accuracy',
                   n_select = 1, #added in pycaret==2.0.0
                   budget_time = 0, #added in pycaret==2.1.0
                   turbo = True,
                   verbose = True): #added in pycaret==2.0.0
    
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
        

    Parameters
    ----------
    exclude: list of strings, default = None
        In order to omit certain models from the comparison model ID's can be passed as 
        a list of strings in exclude param. 

    include: list of strings, default = None
        In order to run only certain models for the comparison, the model ID's can be 
        passed as a list of strings in include param. 

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
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''

    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing compare_models()")
    logger.info("""compare_models(exclude={}, include={}, fold={}, round={}, sort={}, n_select={}, turbo={}, verbose={})""".\
        format(str(exclude), str(include), str(fold), str(round), str(sort), str(n_select), str(turbo), str(verbose)))

    logger.info("Checking exceptions")

    #exception checking   
    import sys
    
    #checking error for exclude (string)
    available_estimators = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 
                            'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']
    
    if exclude != None:
        for i in exclude:
            if i not in available_estimators:
                sys.exit('(Value Error): Estimator Not Available. Please see docstring for list of available estimators.')
        
    if include != None:   
        for i in include:
            if i not in available_estimators:
                sys.exit('(Value Error): Estimator Not Available. Please see docstring for list of available estimators.')

    #include and exclude together check
    if include is not None:
        if exclude is not None:
            sys.exit('(Type Error): Cannot use exclude parameter when include is used to compare models.')

    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    
    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')

    #checking n_select parameter
    if type(n_select) is not int:
        sys.exit('(Type Error): n_select parameter only accepts integer value.')
 
    #checking budget_time parameter
    if type(budget_time) is not int and type(budget_time) is not float:
        sys.exit('(Type Error): budget_time parameter only accepts integer or float values.')

    #checking sort parameter
    allowed_sort = ['Accuracy', 'Recall', 'Precision', 'F1', 'AUC', 'Kappa', 'MCC', 'TT (Sec)']
    if sort not in allowed_sort:
        sys.exit('(Value Error): Sort method not supported. See docstring for list of available parameters.')
    
    #checking optimize parameter for multiclass
    if y.value_counts().count() > 2:
        if sort == 'AUC':
            sys.exit('(Type Error): AUC metric not supported for multiclass problems. See docstring for list of other optimization parameters.')
            
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    logger.info("Preloading libraries")

    #pre-load libraries
    import pandas as pd
    import time, datetime
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    pd.set_option('display.max_columns', 500)

    logger.info("Preparing display monitor")

    #progress bar
    if exclude is None:
        len_of_exclude = 0
    else:
        len_of_exclude = len(exclude)
        
    if turbo:
        len_mod = 15 - len_of_exclude
    else:
        len_mod = 18 - len_of_exclude
    
    #n_select param
    if type(n_select) is list:
        n_select_num = len(n_select)
    else:
        n_select_num = abs(n_select)

    if n_select_num > len_mod:
        n_select_num = len_mod

    if include is not None:
        wl = len(include)
        bl = len_of_exclude
        len_mod = wl - bl

    if include is not None:
        opt = 10
    else:
        opt = 25
        
    progress = ipw.IntProgress(value=0, min=0, max=(fold*len_mod)+opt+n_select_num, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Model', 'Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)'])
    if verbose:
        if html_param:
            display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['Estimator' , '. . . . . . . . . . . . . . . . . .' , 'Compiling Library' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    if verbose:
        if html_param:
            display(monitor, display_id = 'monitor')
            display_ = display(master_display, display_id=True)
            display_id = display_.display_id

    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #general dependencies
    import numpy as np
    import random
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    import pandas.io.formats.style
    
    logger.info("Copying training dataset")
    #defining X_train and y_train as data_X and data_y
    data_X = X_train
    data_y=y_train
    
    progress.value += 1
    
    logger.info("Importing libraries")

    #import sklearn dependencies
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
    from catboost import CatBoostClassifier
    try:
        import lightgbm as lgb
    except:
        pass
        logger.info("LightGBM import failed")
    
   
    progress.value += 1
    
    #defining sort parameter (making Precision equivalent to Prec. )
    if sort == 'Precision':
        sort = 'Prec.'
    else:
        sort = sort
    
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Loading Estimator'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    logger.info("Importing untrained models")

    #creating model object 
    lr = LogisticRegression(random_state=seed) #dont add n_jobs_param here. It slows doesn Logistic Regression somehow.
    knn = KNeighborsClassifier(n_jobs=n_jobs_param)
    nb = GaussianNB()
    dt = DecisionTreeClassifier(random_state=seed)
    svm = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed, n_jobs=n_jobs_param)
    rbfsvm = SVC(gamma='auto', C=1, probability=True, kernel='rbf', random_state=seed)
    gpc = GaussianProcessClassifier(random_state=seed, n_jobs=n_jobs_param)
    mlp = MLPClassifier(max_iter=500, random_state=seed)
    ridge = RidgeClassifier(random_state=seed)
    rf = RandomForestClassifier(n_estimators=10, random_state=seed, n_jobs=n_jobs_param)
    qda = QuadraticDiscriminantAnalysis()
    ada = AdaBoostClassifier(random_state=seed)
    gbc = GradientBoostingClassifier(random_state=seed)
    lda = LinearDiscriminantAnalysis()
    et = ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs_param)
    xgboost = XGBClassifier(random_state=seed, verbosity=0, n_jobs=n_jobs_param)
    lightgbm = lgb.LGBMClassifier(random_state=seed, n_jobs=n_jobs_param)
    catboost = CatBoostClassifier(random_state=seed, silent = True, thread_count=n_jobs_param) 
    
    logger.info("Import successful")

    progress.value += 1
    
    model_dict = {'Logistic Regression' : 'lr',
                   'Linear Discriminant Analysis' : 'lda', 
                   'Ridge Classifier' : 'ridge', 
                   'Extreme Gradient Boosting' : 'xgboost',
                   'Ada Boost Classifier' : 'ada', 
                   'CatBoost Classifier' : 'catboost', 
                   'Light Gradient Boosting Machine' : 'lightgbm', 
                   'Gradient Boosting Classifier' : 'gbc', 
                   'Random Forest Classifier' : 'rf',
                   'Naive Bayes' : 'nb', 
                   'Extra Trees Classifier' : 'et',
                   'Decision Tree Classifier' : 'dt', 
                   'K Neighbors Classifier' : 'knn', 
                   'Quadratic Discriminant Analysis' : 'qda',
                   'SVM - Linear Kernel' : 'svm',
                   'Gaussian Process Classifier' : 'gpc',
                   'MLP Classifier' : 'mlp',
                   'SVM - Radial Kernel' : 'rbfsvm'}

    model_library = [lr, knn, nb, dt, svm, rbfsvm, gpc, mlp, ridge, rf, qda, ada, gbc, lda, et, xgboost, lightgbm, catboost]

    model_names = ['Logistic Regression',
                   'K Neighbors Classifier',
                   'Naive Bayes',
                   'Decision Tree Classifier',
                   'SVM - Linear Kernel',
                   'SVM - Radial Kernel',
                   'Gaussian Process Classifier',
                   'MLP Classifier',
                   'Ridge Classifier',
                   'Random Forest Classifier',
                   'Quadratic Discriminant Analysis',
                   'Ada Boost Classifier',
                   'Gradient Boosting Classifier',
                   'Linear Discriminant Analysis',
                   'Extra Trees Classifier',
                   'Extreme Gradient Boosting',
                   'Light Gradient Boosting Machine',
                   'CatBoost Classifier']          
    
    #checking for exclude models
    
    model_library_str = ['lr', 'knn', 'nb', 'dt', 'svm', 
                         'rbfsvm', 'gpc', 'mlp', 'ridge', 
                         'rf', 'qda', 'ada', 'gbc', 'lda', 
                         'et', 'xgboost', 'lightgbm', 'catboost']
    
    model_library_str_ = ['lr', 'knn', 'nb', 'dt', 'svm', 
                          'rbfsvm', 'gpc', 'mlp', 'ridge', 
                          'rf', 'qda', 'ada', 'gbc', 'lda', 
                          'et', 'xgboost', 'lightgbm', 'catboost']
    
    if exclude is not None:
        
        if turbo:
            internal_exclude = ['rbfsvm', 'gpc', 'mlp']
            compiled_exclude = exclude + internal_exclude
            exclude = list(set(compiled_exclude))
            
        else:
            exclude = exclude
        
        for i in exclude:
            model_library_str_.remove(i)
        
        si = []
        
        for i in model_library_str_:
            s = model_library_str.index(i)
            si.append(s)
        
        model_library_ = []
        model_names_= []
        for i in si:
            model_library_.append(model_library[i])
            model_names_.append(model_names[i])
            
        model_library = model_library_
        model_names = model_names_
        
        
    if exclude is None and turbo is True:
        
        model_library = [lr, knn, nb, dt, svm, ridge, rf, qda, ada, gbc, lda, et, xgboost, lightgbm, catboost]

        model_names = ['Logistic Regression',
                       'K Neighbors Classifier',
                       'Naive Bayes',
                       'Decision Tree Classifier',
                       'SVM - Linear Kernel',
                       'Ridge Classifier',
                       'Random Forest Classifier',
                       'Quadratic Discriminant Analysis',
                       'Ada Boost Classifier',
                       'Gradient Boosting Classifier',
                       'Linear Discriminant Analysis',
                       'Extra Trees Classifier',
                       'Extreme Gradient Boosting',
                       'Light Gradient Boosting Machine',
                       'CatBoost Classifier']
        
    #checking for include models
    if include is not None:

        model_library = []
        model_names = []

        for i in include:
            if i == 'lr':
                model_library.append(lr)
                model_names.append('Logistic Regression')
            elif i == 'knn':
                model_library.append(knn)
                model_names.append('K Neighbors Classifier')                
            elif i == 'nb':
                model_library.append(nb)
                model_names.append('Naive Bayes')   
            elif i == 'dt':
                model_library.append(dt)
                model_names.append('Decision Tree Classifier')   
            elif i == 'svm':
                model_library.append(svm)
                model_names.append('SVM - Linear Kernel')   
            elif i == 'rbfsvm':
                model_library.append(rbfsvm)
                model_names.append('SVM - Radial Kernel')
            elif i == 'gpc':
                model_library.append(gpc)
                model_names.append('Gaussian Process Classifier')   
            elif i == 'mlp':
                model_library.append(mlp)
                model_names.append('MLP Classifier')   
            elif i == 'ridge':
                model_library.append(ridge)
                model_names.append('Ridge Classifier')   
            elif i == 'rf':
                model_library.append(rf)
                model_names.append('Random Forest Classifier')   
            elif i == 'qda':
                model_library.append(qda)
                model_names.append('Quadratic Discriminant Analysis')   
            elif i == 'ada':
                model_library.append(ada)
                model_names.append('Ada Boost Classifier')   
            elif i == 'gbc':
                model_library.append(gbc)
                model_names.append('Gradient Boosting Classifier')   
            elif i == 'lda':
                model_library.append(lda)
                model_names.append('Linear Discriminant Analysis')   
            elif i == 'et':
                model_library.append(et)
                model_names.append('Extra Trees Classifier')   
            elif i == 'xgboost':
                model_library.append(xgboost)
                model_names.append('Extreme Gradient Boosting') 
            elif i == 'lightgbm':
                model_library.append(lightgbm)
                model_names.append('Light Gradient Boosting Machine') 
            elif i == 'catboost':
                model_library.append(catboost)
                model_names.append('CatBoost Classifier')   

    #multiclass check
    model_library_multiclass = []
    if y.value_counts().count() > 2:
        for i in model_library:
            model = OneVsRestClassifier(i)
            model_library_multiclass.append(model)
            
        model_library = model_library_multiclass
        
    progress.value += 1

    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    #cross validation setup starts here
    logger.info("Defining folds")
    kf = StratifiedKFold(fold, random_state=seed, shuffle=folds_shuffle_param)

    logger.info("Declaring metric variables")
    score_acc =np.empty((0,0))
    score_auc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    score_acc_running = np.empty((0,0)) ##running total
    score_mcc=np.empty((0,0))
    score_training_time=np.empty((0,0))
    avg_acc = np.empty((0,0))
    avg_auc = np.empty((0,0))
    avg_recall = np.empty((0,0))
    avg_precision = np.empty((0,0))
    avg_f1 = np.empty((0,0))
    avg_kappa = np.empty((0,0))
    avg_mcc=np.empty((0,0))
    avg_training_time=np.empty((0,0))
    
    #create URI (before loop)
    import secrets
    URI = secrets.token_hex(nbytes=4)

    name_counter = 0
      
    total_runtime_start = time.time()
    total_runtime = 0
    over_time_budget = False
    if budget_time and budget_time > 0:
        logger.info(f"Time budget is {budget_time} minutes")

    for model in model_library:

        logger.info("Initializing " + str(model_names[name_counter]))

        #run_time
        runtime_start = time.time()
        
        progress.value += 1
        
        '''
        MONITOR UPDATE STARTS
        '''
        monitor.iloc[2,1:] = model_names[name_counter]
        monitor.iloc[3,1:] = 'Calculating ETC'
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        fold_num = 1
        
        for train_i , test_i in kf.split(data_X,data_y):
            
            logger.info("Initializing Fold " + str(fold_num))
        
            progress.value += 1
            
            t0 = time.time()
            total_runtime += (t0 - total_runtime_start)/60
            logger.info(f"Total runtime is {total_runtime} minutes")
            over_time_budget = budget_time and budget_time > 0 and total_runtime > budget_time
            if over_time_budget:
                logger.info(f"Total runtime {total_runtime} is over time budget by {total_runtime - budget_time}, breaking loop")
                break
            total_runtime_start = t0

            '''
            MONITOR UPDATE STARTS
            '''
                
            monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
            if verbose:
                if html_param:
                    update_display(monitor, display_id = 'monitor')
            
            '''
            MONITOR UPDATE ENDS
            '''            
     
            Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
            ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
            
            if fix_imbalance_param:

                logger.info("Initializing SMOTE")
                
                if fix_imbalance_method_param is None:
                    import six
                    import sys
                    sys.modules['sklearn.externals.six'] = six
                    from imblearn.over_sampling import SMOTE
                    resampler = SMOTE(random_state = seed)
                else:
                    resampler = fix_imbalance_method_param

                Xtrain,ytrain = resampler.fit_sample(Xtrain, ytrain)
                logger.info("Resampling completed")

            if hasattr(model, 'predict_proba'):
                time_start=time.time()    
                logger.info("Fitting Model")
                model.fit(Xtrain,ytrain)
                logger.info("Evaluating Metrics")
                time_end=time.time()
                pred_prob = model.predict_proba(Xtest)
                pred_prob = pred_prob[:,1]
                pred_ = model.predict(Xtest)
                sca = metrics.accuracy_score(ytest,pred_)

                if y.value_counts().count() > 2:
                    sc = 0
                    recall = metrics.recall_score(ytest,pred_, average='macro')                
                    precision = metrics.precision_score(ytest,pred_, average = 'weighted')
                    f1 = metrics.f1_score(ytest,pred_, average='weighted')

                else:
                    try:
                        sc = metrics.roc_auc_score(ytest,pred_prob)
                    except:
                        sc = 0
                        logger.warning("model has no predict_proba attribute. AUC set to 0.00")
                    recall = metrics.recall_score(ytest,pred_)                
                    precision = metrics.precision_score(ytest,pred_)
                    f1 = metrics.f1_score(ytest,pred_)
            else:
                time_start=time.time()   
                logger.info("Fitting Model")
                model.fit(Xtrain,ytrain)
                logger.info("Evaluating Metrics")
                time_end=time.time()
                logger.warning("model has no predict_proba attribute. pred_prob set to 0.00")
                pred_prob = 0.00
                pred_ = model.predict(Xtest)
                sca = metrics.accuracy_score(ytest,pred_)

                if y.value_counts().count() > 2:
                    sc = 0
                    recall = metrics.recall_score(ytest,pred_, average='macro')                
                    precision = metrics.precision_score(ytest,pred_, average = 'weighted')
                    f1 = metrics.f1_score(ytest,pred_, average='weighted')

                else:
                    try:
                        sc = metrics.roc_auc_score(ytest,pred_prob)
                    except:
                        sc = 0
                        logger.warning("model has no predict_proba attribute. AUC set to 0.00")
                    recall = metrics.recall_score(ytest,pred_)                
                    precision = metrics.precision_score(ytest,pred_)
                    f1 = metrics.f1_score(ytest,pred_)
            
            logger.info("Compiling Metrics")
            mcc = metrics.matthews_corrcoef(ytest,pred_)
            kappa = metrics.cohen_kappa_score(ytest,pred_)
            training_time= time_end - time_start
            score_acc = np.append(score_acc,sca)
            score_auc = np.append(score_auc,sc)
            score_recall = np.append(score_recall,recall)
            score_precision = np.append(score_precision,precision)
            score_f1 =np.append(score_f1,f1)
            score_kappa =np.append(score_kappa,kappa) 
            score_mcc=np.append(score_mcc,mcc)
            score_training_time=np.append(score_training_time,training_time)
                
            '''
            TIME CALCULATION SUB-SECTION STARTS HERE
            '''
            t1 = time.time()
        
            tt = (t1 - t0) * (fold-fold_num) / 60
            tt = np.around(tt, 2)
        
            if tt < 1:
                tt = str(np.around((tt * 60), 2))
                ETC = tt + ' Seconds Remaining'
                
            else:
                tt = str (tt)
                ETC = tt + ' Minutes Remaining'
            
            fold_num += 1
            
            '''
            MONITOR UPDATE STARTS
            '''

            monitor.iloc[3,1:] = ETC
            if verbose:
                if html_param:
                    update_display(monitor, display_id = 'monitor')

            '''
            MONITOR UPDATE ENDS
            '''
        
        if over_time_budget:
            break

        logger.info("Calculating mean and std")
        avg_acc = np.append(avg_acc,np.mean(score_acc))
        avg_auc = np.append(avg_auc,np.mean(score_auc))
        avg_recall = np.append(avg_recall,np.mean(score_recall))
        avg_precision = np.append(avg_precision,np.mean(score_precision))
        avg_f1 = np.append(avg_f1,np.mean(score_f1))
        avg_kappa = np.append(avg_kappa,np.mean(score_kappa))
        avg_mcc=np.append(avg_mcc,np.mean(score_mcc))
        avg_training_time=np.append(avg_training_time,np.mean(score_training_time))
        
        logger.info("Creating metrics dataframe")
        compare_models_ = pd.DataFrame({'Model':model_names[name_counter], 'Accuracy':avg_acc, 'AUC':avg_auc, 
                           'Recall':avg_recall, 'Prec.':avg_precision, 
                           'F1':avg_f1, 'Kappa': avg_kappa, 'MCC':avg_mcc, 'TT (Sec)':avg_training_time})
        master_display = pd.concat([master_display, compare_models_],ignore_index=True)
        master_display = master_display.round(round)
        master_display = master_display.sort_values(by=sort,ascending=False)
        master_display.reset_index(drop=True, inplace=True)
        
        if verbose:
            if html_param:
                update_display(master_display, display_id = display_id)
        
        #end runtime
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

            run_name = model_names[name_counter]

            with mlflow.start_run(run_name=run_name) as run:  

                # Get active run to log as tag
                RunID = mlflow.active_run().info.run_id

                params = model.get_params()

                for i in list(params):
                    v = params.get(i)
                    if len(str(v)) > 250:
                        params.pop(i)
                        
                mlflow.log_params(params)

                #set tag of compare_models
                mlflow.set_tag("Source", "compare_models")
                mlflow.set_tag("URI", URI)
                mlflow.set_tag("USI", USI)
                mlflow.set_tag("Run Time", runtime)
                mlflow.set_tag("Run ID", RunID)

                #Log top model metrics
                mlflow.log_metric("Accuracy", avg_acc[0])
                mlflow.log_metric("AUC", avg_auc[0])
                mlflow.log_metric("Recall", avg_recall[0])
                mlflow.log_metric("Precision", avg_precision[0])
                mlflow.log_metric("F1", avg_f1[0])
                mlflow.log_metric("Kappa", avg_kappa[0])
                mlflow.log_metric("MCC", avg_mcc[0])
                mlflow.log_metric("TT", avg_training_time[0])

                # Log model and transformation pipeline
                from copy import deepcopy

                # get default conda env
                from mlflow.sklearn import get_default_conda_env
                default_conda_env = get_default_conda_env()
                default_conda_env['name'] = str(exp_name_log) + '-env'
                default_conda_env.get('dependencies').pop(-3)
                dependencies = default_conda_env.get('dependencies')[-1]
                from pycaret.utils import __version__
                dep = 'pycaret==' + str(__version__())
                dependencies['pip'] = [dep]
                
                # define model signature
                from mlflow.models.signature import infer_signature
                signature = infer_signature(data_before_preprocess.drop([target_param], axis=1))
                input_example = data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()

                # log model as sklearn flavor
                prep_pipe_temp = deepcopy(prep_pipe)
                prep_pipe_temp.steps.append(['trained model', model])
                mlflow.sklearn.log_model(prep_pipe_temp, "model", conda_env = default_conda_env, signature = signature, input_example = input_example)
                del(prep_pipe_temp)

        score_acc =np.empty((0,0))
        score_auc =np.empty((0,0))
        score_recall =np.empty((0,0))
        score_precision =np.empty((0,0))
        score_f1 =np.empty((0,0))
        score_kappa =np.empty((0,0))
        score_mcc =np.empty((0,0))
        score_training_time =np.empty((0,0))
        
        avg_acc = np.empty((0,0))
        avg_auc = np.empty((0,0))
        avg_recall = np.empty((0,0))
        avg_precision = np.empty((0,0))
        avg_f1 = np.empty((0,0))
        avg_kappa = np.empty((0,0))
        avg_mcc = np.empty((0,0))
        avg_training_time = np.empty((0,0))
        
        name_counter += 1
  
    progress.value += 1
    
    def highlight_max(s):
        to_highlight = s == s.max()
        return ['background-color: yellow' if v else '' for v in to_highlight]
    
    def highlight_cols(s):
        color = 'lightgrey'
        return 'background-color: %s' % color
    
    if y.value_counts().count() > 2:
        
        compare_models_ = master_display.style.apply(highlight_max,subset=['Accuracy','Recall',
                      'Prec.','F1','Kappa', 'MCC']).applymap(highlight_cols, subset = ['TT (Sec)'])
    else:
        
        compare_models_ = master_display.style.apply(highlight_max,subset=['Accuracy','AUC','Recall',
                      'Prec.','F1','Kappa', 'MCC']).applymap(highlight_cols, subset = ['TT (Sec)'])

    compare_models_ = compare_models_.set_precision(round)
    compare_models_ = compare_models_.set_properties(**{'text-align': 'left'})
    compare_models_ = compare_models_.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    
    progress.value += 1
    
    monitor.iloc[1,1:] = 'Compiling Final Model'
    monitor.iloc[3,1:] = 'Almost Finished'

    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')

    sorted_model_names = list(compare_models_.data['Model'])
    n_select = n_select if n_select <= len(sorted_model_names) else len(sorted_model_names)
    if n_select < 0:
        sorted_model_names = sorted_model_names[n_select:]
    else:
        sorted_model_names = sorted_model_names[:n_select]
    
    model_store_final = []

    model_fit_start = time.time()

    logger.info("Finalizing top_n models")

    logger.info("SubProcess create_model() called ==================================")
    for i in sorted_model_names:
        monitor.iloc[2,1:] = i
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')
        progress.value += 1
        k = model_dict.get(i)
        m = create_model(estimator=k, verbose = False, system=False, cross_validation=True)
        model_store_final.append(m)
    logger.info("SubProcess create_model() end ==================================")

    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

    if len(model_store_final) == 1:
        model_store_final = model_store_final[0]

    clear_output()

    if verbose:
        if html_param:
            display(compare_models_)
        else:
            print(compare_models_.data)

    pd.reset_option("display.max_columns")

    #store in display container
    display_container.append(compare_models_.data)

    logger.info("create_model_container: " + str(len(create_model_container)))
    logger.info("master_model_container: " + str(len(master_model_container)))
    logger.info("display_container: " + str(len(display_container)))

    logger.info(str(model_store_final))
    logger.info("compare_models() succesfully completed......................................")

    return model_store_final

def create_model(estimator = None, 
                 ensemble = False, 
                 method = None, 
                 fold = 10, 
                 round = 4,
                 cross_validation = True, #added in pycaret==2.0.0
                 verbose = True,
                 system = True, #added in pycaret==2.0.0
                 **kwargs): #added in pycaret==2.0.0

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

    cross_validation: bool, default = True
        When cross_validation set to False fold parameter is ignored and model is trained
        on entire training dataset. No metric evaluation is returned. 

    verbose: Boolean, default = True
        Score grid is not printed when verbose is set to False.

    system: Boolean, default = True
        Must remain True all times. Only to be changed by internal functions.

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


    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing create_model()")
    logger.info("""create_model(estimator={}, ensemble={}, method={}, fold={}, round={}, cross_validation={}, verbose={}, system={})""".\
        format(str(estimator), str(ensemble), str(method), str(fold), str(round), str(cross_validation), str(verbose), str(system)))

    logger.info("Checking exceptions")

    #exception checking   
    import sys

    #run_time
    import datetime, time
    runtime_start = time.time()
    
    #checking error for estimator (string)
    available_estimators = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 
                            'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']

    #only raise exception of estimator is of type string.
    if type(estimator) is str:
        if estimator not in available_estimators:
            sys.exit('(Value Error): Estimator Not Available. Please see docstring for list of available estimators.')

    #checking error for ensemble:
    if type(ensemble) is not bool:
        sys.exit('(Type Error): Ensemble parameter can only take argument as True or False.') 
    
    #checking error for method:
    
    #1 Check When method given and ensemble is not set to True.
    if ensemble is False and method is not None:
        sys.exit('(Type Error): Method parameter only accepts value when ensemble is set to True.')

    #2 Check when ensemble is set to True and method is not passed.
    if ensemble is True and method is None:
        sys.exit("(Type Error): Method parameter missing. Pass method = 'Bagging' or 'Boosting'.")
        
    #3 Check when ensemble is set to True and method is passed but not allowed.
    available_method = ['Bagging', 'Boosting']
    if ensemble is True and method not in available_method:
        sys.exit("(Value Error): Method parameter only accepts two values 'Bagging' or 'Boosting'.")
        
    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    
    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')
 
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.') 
        
    #checking system parameter
    if type(system) is not bool:
        sys.exit('(Type Error): System parameter can only take argument as True or False.') 

    #checking cross_validation parameter
    if type(cross_validation) is not bool:
        sys.exit('(Type Error): cross_validation parameter can only take argument as True or False.') 

    #checking boosting conflict with estimators
    boosting_not_supported = ['lda','qda','ridge','mlp','gpc','svm','knn', 'catboost']
    if method == 'Boosting' and estimator in boosting_not_supported:
        sys.exit("(Type Error): Estimator does not provide class_weights or predict_proba function and hence not supported for the Boosting method. Change the estimator or method to 'Bagging'.")
    
    
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''

    logger.info("Preloading libraries")

    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    logger.info("Preparing display monitor")

    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+4, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa', 'MCC'])
    if verbose:
        if html_param:
            display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    if verbose:
        if html_param:
            display(monitor, display_id = 'monitor')
    
    if verbose:
        if html_param:
            display_ = display(master_display, display_id=True)
            display_id = display_.display_id
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    logger.info("Copying training dataset")

    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()
    
    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
  
    logger.info("Importing libraries")

    #general dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    
    progress.value += 1
    
    logger.info("Defining folds")

    #cross validation setup starts here
    kf = StratifiedKFold(fold, random_state=seed, shuffle=folds_shuffle_param)

    logger.info("Declaring metric variables")

    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    score_mcc =np.empty((0,0))
    score_training_time =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    avgs_mcc =np.empty((0,0))
    avgs_training_time =np.empty((0,0))
    
  
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Selecting Estimator'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''

    logger.info("Importing untrained model")

    if estimator == 'lr':

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=seed, **kwargs)
        full_name = 'Logistic Regression'

    elif estimator == 'knn':
        
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_jobs=n_jobs_param, **kwargs)
        full_name = 'K Neighbors Classifier'

    elif estimator == 'nb':

        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(**kwargs)
        full_name = 'Naive Bayes'

    elif estimator == 'dt':

        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=seed, **kwargs)
        full_name = 'Decision Tree Classifier'

    elif estimator == 'svm':

        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'SVM - Linear Kernel'

    elif estimator == 'rbfsvm':

        from sklearn.svm import SVC
        model = SVC(gamma='auto', C=1, probability=True, kernel='rbf', random_state=seed, **kwargs)
        full_name = 'SVM - Radial Kernel'

    elif estimator == 'gpc':

        from sklearn.gaussian_process import GaussianProcessClassifier
        model = GaussianProcessClassifier(random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'Gaussian Process Classifier'

    elif estimator == 'mlp':

        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(max_iter=500, random_state=seed, **kwargs)
        full_name = 'MLP Classifier'    

    elif estimator == 'ridge':

        from sklearn.linear_model import RidgeClassifier
        model = RidgeClassifier(random_state=seed, **kwargs)
        full_name = 'Ridge Classifier'        

    elif estimator == 'rf':

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'Random Forest Classifier'    

    elif estimator == 'qda':

        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        model = QuadraticDiscriminantAnalysis(**kwargs)
        full_name = 'Quadratic Discriminant Analysis' 

    elif estimator == 'ada':

        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(random_state=seed, **kwargs)
        full_name = 'Ada Boost Classifier'        

    elif estimator == 'gbc':

        from sklearn.ensemble import GradientBoostingClassifier    
        model = GradientBoostingClassifier(random_state=seed, **kwargs)
        full_name = 'Gradient Boosting Classifier'    

    elif estimator == 'lda':

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis(**kwargs)
        full_name = 'Linear Discriminant Analysis'

    elif estimator == 'et':

        from sklearn.ensemble import ExtraTreesClassifier 
        model = ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'Extra Trees Classifier'

    elif estimator == 'xgboost':

        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=seed, verbosity=0, n_jobs=n_jobs_param, **kwargs)
        full_name = 'Extreme Gradient Boosting'
        
    elif estimator == 'lightgbm':
        
        import lightgbm as lgb
        model = lgb.LGBMClassifier(random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'Light Gradient Boosting Machine'
        
    elif estimator == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(random_state=seed, silent=True, thread_count=n_jobs_param, **kwargs) # Silent is True to suppress CatBoost iteration results 
        full_name = 'CatBoost Classifier'
        
    else:

        logger.info("Declaring custom model")

        model = estimator

        def get_model_name(e):
            return str(e).split("(")[0]

        model_dict_logging = {'ExtraTreesClassifier' : 'Extra Trees Classifier',
                            'GradientBoostingClassifier' : 'Gradient Boosting Classifier', 
                            'RandomForestClassifier' : 'Random Forest Classifier',
                            'LGBMClassifier' : 'Light Gradient Boosting Machine',
                            'XGBClassifier' : 'Extreme Gradient Boosting',
                            'AdaBoostClassifier' : 'Ada Boost Classifier', 
                            'DecisionTreeClassifier' : 'Decision Tree Classifier', 
                            'RidgeClassifier' : 'Ridge Classifier',
                            'LogisticRegression' : 'Logistic Regression',
                            'KNeighborsClassifier' : 'K Neighbors Classifier',
                            'GaussianNB' : 'Naive Bayes',
                            'SGDClassifier' : 'SVM - Linear Kernel',
                            'SVC' : 'SVM - Radial Kernel',
                            'GaussianProcessClassifier' : 'Gaussian Process Classifier',
                            'MLPClassifier' : 'MLP Classifier',
                            'QuadraticDiscriminantAnalysis' : 'Quadratic Discriminant Analysis',
                            'LinearDiscriminantAnalysis' : 'Linear Discriminant Analysis',
                            'CatBoostClassifier' : 'CatBoost Classifier',
                            'BaggingClassifier' : 'Bagging Classifier',
                            'VotingClassifier' : 'Voting Classifier'} 

        if y.value_counts().count() > 2:

            mn = get_model_name(estimator.estimator)

            if 'catboost' in mn:
                mn = 'CatBoostClassifier'

            if mn in model_dict_logging.keys():
                full_name = model_dict_logging.get(mn)
            else:
                full_name = mn
        
        else:

            mn = get_model_name(estimator)
            
            if 'catboost' in mn:
                mn = 'CatBoostClassifier'

            if mn in model_dict_logging.keys():
                full_name = model_dict_logging.get(mn)
            else:
                full_name = mn
    
    logger.info(str(full_name) + ' Imported succesfully')

    progress.value += 1
    
    #checking method when ensemble is set to True. 

    logger.info("Checking ensemble method")

    if method == 'Bagging':
        logger.info("Ensemble method set to Bagging")     
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier(model,bootstrap=True,n_estimators=10, random_state=seed, n_jobs=n_jobs_param)

    elif method == 'Boosting':
        logger.info("Ensemble method set to Boosting")     
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(model, n_estimators=10, random_state=seed)
    
    #multiclass checking
    if y.value_counts().count() > 2:
        logger.info("Target variable is Multiclass. OneVsRestClassifier activated")     
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(model, n_jobs=n_jobs_param)
    
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    if not cross_validation:
        monitor.iloc[1,1:] = 'Fitting ' + str(full_name)
    else:
        monitor.iloc[1,1:] = 'Initializing CV'
    
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    if not cross_validation:

        logger.info("Cross validation set to False")

        if fix_imbalance_param:
            logger.info("Initializing SMOTE")
            if fix_imbalance_method_param is None:
                import six
                import sys
                sys.modules['sklearn.externals.six'] = six
                from imblearn.over_sampling import SMOTE
                resampler = SMOTE(random_state=seed)
            else:
                resampler = fix_imbalance_method_param

            Xtrain,ytrain = resampler.fit_sample(data_X,data_y)
            logger.info("Resampling completed")

        logger.info("Fitting Model")
        model.fit(data_X,data_y)

        if verbose:
            clear_output()
        
        logger.info("create_model_container " + str(len(create_model_container)))
        logger.info("master_model_container " + str(len(master_model_container)))
        logger.info("display_container " + str(len(display_container)))
        
        logger.info(str(model))
        logger.info("create_models() succesfully completed......................................")
        
        return model
    
    fold_num = 1
    
    for train_i , test_i in kf.split(data_X,data_y):

        logger.info("Initializing Fold " + str(fold_num))
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
    
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
        time_start=time.time()

        if fix_imbalance_param:
            
            logger.info("Initializing SMOTE")

            if fix_imbalance_method_param is None:
                import six
                import sys
                sys.modules['sklearn.externals.six'] = six
                from imblearn.over_sampling import SMOTE
                resampler = SMOTE(random_state=seed)
            else:
                resampler = fix_imbalance_method_param

            Xtrain,ytrain = resampler.fit_sample(Xtrain, ytrain)
            logger.info("Resampling completed")

        if hasattr(model, 'predict_proba'):
            logger.info("Fitting Model")
            model.fit(Xtrain,ytrain)
            logger.info("Evaluating Metrics")
            pred_prob = model.predict_proba(Xtest)
            pred_prob = pred_prob[:,1]
            pred_ = model.predict(Xtest)
            sca = metrics.accuracy_score(ytest,pred_)
            
            if y.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(ytest,pred_, average='macro')                
                precision = metrics.precision_score(ytest,pred_, average = 'weighted')
                f1 = metrics.f1_score(ytest,pred_, average='weighted')
                
            else:
                try:
                    sc = metrics.roc_auc_score(ytest,pred_prob)
                except:
                    logger.warning("model has no predict_proba attribute. AUC set to 0.00")
                    sc = 0
                recall = metrics.recall_score(ytest,pred_)                
                precision = metrics.precision_score(ytest,pred_)
                f1 = metrics.f1_score(ytest,pred_)
        else:
            logger.info("Fitting Model")
            model.fit(Xtrain,ytrain)
            logger.info("Evaluating Metrics")
            logger.warning("model has no predict_proba attribute. pred_prob set to 0.00")
            pred_prob = 0.00
            pred_ = model.predict(Xtest)
            sca = metrics.accuracy_score(ytest,pred_)
            
            if y.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(ytest,pred_, average='macro')                
                precision = metrics.precision_score(ytest,pred_, average = 'weighted')
                f1 = metrics.f1_score(ytest,pred_, average='weighted')

            else:
                try:
                    sc = metrics.roc_auc_score(ytest,pred_prob)
                except:
                    sc = 0
                    logger.warning("model has no predict_proba attribute. AUC to 0.00")
                recall = metrics.recall_score(ytest,pred_)                
                precision = metrics.precision_score(ytest,pred_)
                f1 = metrics.f1_score(ytest,pred_)

        logger.info("Compiling Metrics")        
        time_end=time.time()
        kappa = metrics.cohen_kappa_score(ytest,pred_)
        mcc = metrics.matthews_corrcoef(ytest,pred_)
        training_time=time_end-time_start
        score_acc = np.append(score_acc,sca)
        score_auc = np.append(score_auc,sc)
        score_recall = np.append(score_recall,recall)
        score_precision = np.append(score_precision,precision)
        score_f1 =np.append(score_f1,f1)
        score_kappa =np.append(score_kappa,kappa)
        score_mcc=np.append(score_mcc,mcc)
        score_training_time = np.append(score_training_time,training_time)
   
        progress.value += 1
                
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa], 'MCC':[mcc]}).round(round)
        master_display = pd.concat([master_display, fold_results],ignore_index=True)
        fold_results = []
        
        '''
        TIME CALCULATION SUB-SECTION STARTS HERE
        '''
        t1 = time.time()
        
        tt = (t1 - t0) * (fold-fold_num) / 60
        tt = np.around(tt, 2)
        
        if tt < 1:
            tt = str(np.around((tt * 60), 2))
            ETC = tt + ' Seconds Remaining'
                
        else:
            tt = str (tt)
            ETC = tt + ' Minutes Remaining'
            
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[2,1:] = ETC
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
            
        fold_num += 1
        
        '''
        TIME CALCULATION ENDS HERE
        '''
        
        if verbose:
            if html_param:
                update_display(master_display, display_id = display_id)
            
        
        '''
        
        Update_display() ends here
        
        '''
    
    logger.info("Calculating mean and std")

    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    mean_mcc=np.mean(score_mcc)
    mean_training_time=np.sum(score_training_time) #changed it to sum from mean 
    
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    std_mcc=np.std(score_mcc)
    std_training_time=np.std(score_training_time)
    
    avgs_acc = np.append(avgs_acc, mean_acc)
    avgs_acc = np.append(avgs_acc, std_acc) 
    avgs_auc = np.append(avgs_auc, mean_auc)
    avgs_auc = np.append(avgs_auc, std_auc)
    avgs_recall = np.append(avgs_recall, mean_recall)
    avgs_recall = np.append(avgs_recall, std_recall)
    avgs_precision = np.append(avgs_precision, mean_precision)
    avgs_precision = np.append(avgs_precision, std_precision)
    avgs_f1 = np.append(avgs_f1, mean_f1)
    avgs_f1 = np.append(avgs_f1, std_f1)
    avgs_kappa = np.append(avgs_kappa, mean_kappa)
    avgs_kappa = np.append(avgs_kappa, std_kappa)
    avgs_mcc = np.append(avgs_mcc, mean_mcc)
    avgs_mcc = np.append(avgs_mcc, std_mcc)
    
    avgs_training_time = np.append(avgs_training_time, mean_training_time)
    avgs_training_time = np.append(avgs_training_time, std_training_time)
    
    progress.value += 1
    
    logger.info("Creating metrics dataframe")

    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa, 'MCC': score_mcc})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa, 'MCC': avgs_mcc},index=['Mean', 'SD'])

    
    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    
    # yellow the mean
    model_results=model_results.style.apply(lambda x: ['background: yellow' if (x.name == 'Mean') else '' for i in x], axis=1)
    model_results = model_results.set_precision(round)

    #refitting the model on complete X_train, y_train
    monitor.iloc[1,1:] = 'Finalizing Model'
    monitor.iloc[2,1:] = 'Almost Finished'    
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    model_fit_start = time.time()
    logger.info("Finalizing model")
    model.fit(data_X, data_y)
    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
    
    #end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)
    
    #mlflow logging
    if logging_param and system:
        
        logger.info("Creating MLFlow logs")
        
        #Creating Logs message monitor
        monitor.iloc[1,1:] = 'Creating Logs'
        monitor.iloc[2,1:] = 'Almost Finished'    
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        #import mlflow
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
            mlflow.log_metrics({"Accuracy": avgs_acc[0], "AUC": avgs_auc[0], "Recall": avgs_recall[0], "Precision" : avgs_precision[0],
                                "F1": avgs_f1[0], "Kappa": avgs_kappa[0], "MCC": avgs_mcc[0]})
            
            #set tag of compare_models
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
            model_results.data.to_html('Results.html', col_space=65, justify='left')
            mlflow.log_artifact('Results.html')
            os.remove('Results.html')

            # Generate hold-out predictions and save as html
            holdout = predict_model(model, verbose=False)
            holdout_score = pull()
            del(holdout)
            display_container.pop(-1)
            holdout_score.to_html('Holdout.html', col_space=65, justify='left')
            mlflow.log_artifact('Holdout.html')
            os.remove('Holdout.html')

            # Log AUC and Confusion Matrix plot
            
            if log_plots_param:
                
                logger.info("SubProcess plot_model() called ==================================")

                try:
                    plot_model(model, plot = 'auc', verbose=False, save=True, system=False)
                    mlflow.log_artifact('AUC.png')
                    os.remove("AUC.png")
                except:
                    pass

                try:
                    plot_model(model, plot = 'confusion_matrix', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Confusion Matrix.png')
                    os.remove("Confusion Matrix.png")
                except:
                    pass

                try:
                    plot_model(model, plot = 'feature', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Feature Importance.png')
                    os.remove("Feature Importance.png")
                except:
                    pass
                    
                logger.info("SubProcess plot_model() end ==================================")
            
            # Log model and transformation pipeline
            from copy import deepcopy

            # get default conda env
            from mlflow.sklearn import get_default_conda_env
            default_conda_env = get_default_conda_env()
            default_conda_env['name'] = str(exp_name_log) + '-env'
            default_conda_env.get('dependencies').pop(-3)
            dependencies = default_conda_env.get('dependencies')[-1]
            from pycaret.utils import __version__
            dep = 'pycaret==' + str(__version__())
            dependencies['pip'] = [dep]
            
            # define model signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(data_before_preprocess.drop([target_param], axis=1))
            input_example = data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(['trained model', model])
            mlflow.sklearn.log_model(prep_pipe_temp, "model", conda_env = default_conda_env, signature = signature, input_example = input_example)
            del(prep_pipe_temp)

    progress.value += 1

    logger.info("Uploading results into container")

    #storing results in create_model_container
    create_model_container.append(model_results.data)
    display_container.append(model_results.data)

    #storing results in master_model_container
    logger.info("Uploading model into container now")
    master_model_container.append(model)

    if verbose:
        clear_output()

        if html_param:
            display(model_results)
        else:
            print(model_results.data)

    logger.info("create_model_container: " + str(len(create_model_container)))
    logger.info("master_model_container: " + str(len(master_model_container)))
    logger.info("display_container: " + str(len(display_container)))
    
    logger.info(str(model))
    logger.info("create_model() succesfully completed......................................")
    return model

def tune_model(estimator = None, 
               fold = 10, 
               round = 4, 
               n_iter = 10,
               custom_grid = None, #added in pycaret==2.0.0 
               optimize = 'Accuracy',
               custom_scorer = None, #added in pycaret==2.1
               choose_better = False, #added in pycaret==2.0.0 
               verbose = True):
    
      
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

    optimize: string, default = 'accuracy'
        Measure used to select the best model through hyperparameter tuning.
        The default scoring measure is 'Accuracy'. Other measures include 'AUC',
        'Recall', 'Precision', 'F1'. 

    custom_scorer: object, default = None
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

    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing tune_model()")
    logger.info("""tune_model(estimator={}, fold={}, round={}, n_iter={}, custom_grid={}, optimize={}, choose_better={}, verbose={})""".\
        format(str(estimator), str(fold), str(round), str(n_iter), str(custom_grid), str(optimize), str(choose_better), str(verbose)))

    logger.info("Checking exceptions")

    #exception checking   
    import sys
    
    #run_time
    import datetime, time
    runtime_start = time.time()

    #checking estimator if string
    if type(estimator) is str:
        sys.exit('(Type Error): The behavior of tune_model in version 1.0.1 is changed. Please pass trained model object.')
    
    #restrict VotingClassifier
    if hasattr(estimator,'voting'):
         sys.exit('(Type Error): VotingClassifier not allowed under tune_model().')

    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    
    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')
 
    #checking n_iter parameter
    if type(n_iter) is not int:
        sys.exit('(Type Error): n_iter parameter only accepts integer value.')

    #checking optimize parameter
    allowed_optimize = ['Accuracy', 'Recall', 'Precision', 'F1', 'AUC', 'MCC']
    if optimize not in allowed_optimize:
        sys.exit('(Value Error): Optimization method not supported. See docstring for list of available parameters.')
    
    #checking optimize parameter for multiclass
    if y.value_counts().count() > 2:
        if optimize == 'AUC':
            sys.exit('(Type Error): AUC metric not supported for multiclass problems. See docstring for list of other optimization parameters.')
    
    if type(n_iter) is not int:
        sys.exit('(Type Error): n_iter parameter only accepts integer value.')
        
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.')     


    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    logger.info("Preloading libraries")

    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    logger.info("Preparing display monitor")

    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+6, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa', 'MCC'])
    if verbose:
        if html_param:
            display(progress)    
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    if verbose:
        if html_param:
            display(monitor, display_id = 'monitor')
            display_ = display(master_display, display_id=True)
            display_id = display_.display_id
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore')    

    logger.info("Copying training dataset")
    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()

    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
    
    logger.info("Creating estimator clone to inherit model parameters")
    #create estimator clone from sklearn.base
    from sklearn.base import clone
    estimator_clone = clone(estimator)
    
    progress.value += 1

    logger.info("Importing libraries")    
    #general dependencies
    import random
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import RandomizedSearchCV
    
    #setting numpy seed
    np.random.seed(seed)
    
    #setting optimize parameter   
    if optimize == 'Accuracy':
        optimize = 'accuracy'
        compare_dimension = 'Accuracy'
        
    elif optimize == 'AUC':
        optimize = 'roc_auc'
        compare_dimension = 'AUC'
        
    elif optimize == 'Recall':
        if y.value_counts().count() > 2:
            optimize = metrics.make_scorer(metrics.recall_score, average = 'macro')
        else:
            optimize = 'recall'
        compare_dimension = 'Recall'

    elif optimize == 'Precision':
        if y.value_counts().count() > 2:
            optimize = metrics.make_scorer(metrics.precision_score, average = 'weighted')
        else:
            optimize = 'precision'
        compare_dimension = 'Prec.'
   
    elif optimize == 'F1':
        if y.value_counts().count() > 2:
            optimize = metrics.make_scorer(metrics.f1_score, average = 'weighted')
        else:
            optimize = optimize = 'f1'
        compare_dimension = 'F1'

    elif optimize == 'MCC':
        optimize = 'roc_auc' # roc_auc instead because you cannot use MCC in gridsearchcv
        compare_dimension = 'MCC'
    
    # change optimize parameter if custom_score is not None
    if custom_scorer is not None:
        optimize = custom_scorer
        logger.info("custom_scorer set to user defined function")

    #convert trained estimator into string name for grids
    
    logger.info("Checking base model")
    def get_model_name(e):
        return str(e).split("(")[0]

    if len(estimator.classes_) > 2:
        mn = get_model_name(estimator.estimator)
    else:
        mn = get_model_name(estimator)

    if 'catboost' in mn:
        mn = 'CatBoostClassifier'
    
    model_dict = {'ExtraTreesClassifier' : 'et',
                'GradientBoostingClassifier' : 'gbc', 
                'RandomForestClassifier' : 'rf',
                'LGBMClassifier' : 'lightgbm',
                'XGBClassifier' : 'xgboost',
                'AdaBoostClassifier' : 'ada', 
                'DecisionTreeClassifier' : 'dt', 
                'RidgeClassifier' : 'ridge',
                'LogisticRegression' : 'lr',
                'KNeighborsClassifier' : 'knn',
                'GaussianNB' : 'nb',
                'SGDClassifier' : 'svm',
                'SVC' : 'rbfsvm',
                'GaussianProcessClassifier' : 'gpc',
                'MLPClassifier' : 'mlp',
                'QuadraticDiscriminantAnalysis' : 'qda',
                'LinearDiscriminantAnalysis' : 'lda',
                'CatBoostClassifier' : 'catboost',
                'BaggingClassifier' : 'Bagging'}

    model_dict_logging = {'ExtraTreesClassifier' : 'Extra Trees Classifier',
                        'GradientBoostingClassifier' : 'Gradient Boosting Classifier', 
                        'RandomForestClassifier' : 'Random Forest Classifier',
                        'LGBMClassifier' : 'Light Gradient Boosting Machine',
                        'XGBClassifier' : 'Extreme Gradient Boosting',
                        'AdaBoostClassifier' : 'Ada Boost Classifier', 
                        'DecisionTreeClassifier' : 'Decision Tree Classifier', 
                        'RidgeClassifier' : 'Ridge Classifier',
                        'LogisticRegression' : 'Logistic Regression',
                        'KNeighborsClassifier' : 'K Neighbors Classifier',
                        'GaussianNB' : 'Naive Bayes',
                        'SGDClassifier' : 'SVM - Linear Kernel',
                        'SVC' : 'SVM - Radial Kernel',
                        'GaussianProcessClassifier' : 'Gaussian Process Classifier',
                        'MLPClassifier' : 'MLP Classifier',
                        'QuadraticDiscriminantAnalysis' : 'Quadratic Discriminant Analysis',
                        'LinearDiscriminantAnalysis' : 'Linear Discriminant Analysis',
                        'CatBoostClassifier' : 'CatBoost Classifier',
                        'BaggingClassifier' : 'Bagging Classifier',
                        'VotingClassifier' : 'Voting Classifier'}

    _estimator_ = estimator

    estimator = model_dict.get(mn)

    logger.info('Base model : ' + str(model_dict_logging.get(mn)))

    progress.value += 1
    
    logger.info("Defining folds")
    kf = StratifiedKFold(fold, random_state=seed, shuffle=folds_shuffle_param)

    logger.info("Declaring metric variables")
    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    score_mcc=np.empty((0,0))
    score_training_time=np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    avgs_mcc=np.empty((0,0))
    avgs_training_time=np.empty((0,0))
    
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Searching Hyperparameters'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    logger.info("Defining Hyperparameters")
    logger.info("Initializing RandomizedSearchCV")

    #setting turbo parameters
    cv = 3

    if estimator == 'knn':
        
        from sklearn.neighbors import KNeighborsClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'n_neighbors': range(1,51),
                    'weights' : ['uniform', 'distance'],
                    'metric':["euclidean", "manhattan"]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone, param_distributions=param_grid, 
                                        scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       n_jobs=n_jobs_param, iid=False)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_
 
    elif estimator == 'lr':
        
        from sklearn.linear_model import LogisticRegression

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'C': np.arange(0, 10, 0.001),
                    "penalty": [ 'l1', 'l2'],
                    "class_weight": ["balanced", None]
                        }
        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv, 
                                        random_state=seed, iid=False, n_jobs=n_jobs_param)
        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'dt':
        
        from sklearn.tree import DecisionTreeClassifier
        
        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {"max_depth": np.random.randint(1, (len(X_train.columns)*.85),20),
                    "max_features": np.random.randint(1, len(X_train.columns),20),
                    "min_samples_leaf": [2,3,4,5,6],
                    "criterion": ["gini", "entropy"],
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone, param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       iid=False, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_
 
    elif estimator == 'mlp':
    
        from sklearn.neural_network import MLPClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'solver' : ['lbfgs', 'sgd', 'adam'],
                    'alpha': np.arange(0, 1, 0.0001),
                    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (100,50,100), (100,100,100)],
                    'activation': ["tanh", "identity", "logistic","relu"]
                    }

        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv, 
                                        random_state=seed, iid=False, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_
    
    elif estimator == 'gpc':
        
        from sklearn.gaussian_process import GaussianProcessClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {"max_iter_predict":[100,200,300,400,500,600,700,800,900,1000]}

        model_grid = RandomizedSearchCV(estimator=estimator_clone, param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_    

    elif estimator == 'rbfsvm':
        
        from sklearn.svm import SVC

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'C': np.arange(0, 50, 0.01),
                    "class_weight": ["balanced", None]}

        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_    
  
    elif estimator == 'nb':
        
        from sklearn.naive_bayes import GaussianNB

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'var_smoothing': [0.000000001, 0.000000002, 0.000000005, 0.000000008, 0.000000009,
                                            0.0000001, 0.0000002, 0.0000003, 0.0000005, 0.0000007, 0.0000009, 
                                            0.00001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.009,
                                            0.004, 0.005, 0.006, 0.007,0.008, 0.009, 0.01, 0.1, 1]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)
 
        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_        

    elif estimator == 'svm':
       
        from sklearn.linear_model import SGDClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'penalty': ['l2', 'l1','elasticnet'],
                        'l1_ratio': np.arange(0,1,0.01),
                        'alpha': [0.0001, 0.001, 0.01, 0.0002, 0.002, 0.02, 0.0005, 0.005, 0.05],
                        'fit_intercept': [True, False],
                        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                        'eta0': [0.001, 0.01,0.05,0.1,0.2,0.3,0.4,0.5]
                        }    

        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_     

    elif estimator == 'ridge':
        
        from sklearn.linear_model import RidgeClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'alpha': np.arange(0,1,0.001),
                        'fit_intercept': [True, False],
                        'normalize': [True, False]
                        }    

        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_     
   
    elif estimator == 'rf':
        
        from sklearn.ensemble import RandomForestClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'min_samples_split': [2, 5, 7, 9, 10],
                        'min_samples_leaf' : [1, 2, 4],
                        'max_features' : ['auto', 'sqrt', 'log2'],
                        'bootstrap': [True, False]
                        }    

        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_     
   
    elif estimator == 'ada':
        
        from sklearn.ensemble import AdaBoostClassifier        

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'n_estimators':  np.arange(10,200,5),
                        'learning_rate': np.arange(0,1,0.01),
                        'algorithm' : ["SAMME", "SAMME.R"]
                        }    

        if y.value_counts().count() > 2:
            base_estimator_input = _estimator_.estimator.base_estimator
        else:
            base_estimator_input = _estimator_.base_estimator

        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_   

    elif estimator == 'gbc':
        
        from sklearn.ensemble import GradientBoostingClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'n_estimators': np.arange(10,200,5),
                        'learning_rate': np.arange(0,1,0.01),
                        'subsample' : np.arange(0.1,1,0.05),
                        'min_samples_split' : [2,4,5,7,9,10],
                        'min_samples_leaf' : [1,2,3,4,5],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'max_features' : ['auto', 'sqrt', 'log2']
                        }    
            
        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_   

    elif estimator == 'qda':
        
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'reg_param': np.arange(0,1,0.01)}    

        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_      

    elif estimator == 'lda':
        
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'solver' : ['lsqr', 'eigen'],
                        'shrinkage': [None, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
                        }    

        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_        

    elif estimator == 'et':
        
        from sklearn.ensemble import ExtraTreesClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'n_estimators': np.arange(10,200,5),
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'min_samples_split': [2, 5, 7, 9, 10],
                        'min_samples_leaf' : [1, 2, 4],
                        'max_features' : ['auto', 'sqrt', 'log2'],
                        'bootstrap': [True, False]
                        }    

        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_ 
        
        
    elif estimator == 'xgboost':
        
        from xgboost import XGBClassifier
        
        num_class = y.value_counts().count()
        
        if custom_grid is not None:
            param_grid = custom_grid

        elif y.value_counts().count() > 2:
            
            param_grid = {'learning_rate': np.arange(0,1,0.01),
                          'n_estimators': np.arange(10,500,20),
                          'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
                          'colsample_bytree': [0.5, 0.7, 0.9, 1],
                          'min_child_weight': [1, 2, 3, 4],
                          'num_class' : [num_class, num_class]
                         }
        else:
            param_grid = {'learning_rate': np.arange(0,1,0.01),
                          'n_estimators':[10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
                          'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
                          'colsample_bytree': [0.5, 0.7, 0.9, 1],
                          'min_child_weight': [1, 2, 3, 4],
                         }

        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)
        
        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_ 
        
        
    elif estimator == 'lightgbm':
        
        import lightgbm as lgb
        
        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'num_leaves': [10,20,30,40,50,60,70,80,90,100,150,200],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                        'n_estimators': [10, 30, 50, 70, 90, 100, 120, 150, 170, 200], 
                        'min_split_gain' : [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                        'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'reg_lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        }
    
        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_ 
        
        
    elif estimator == 'catboost':
        
        from catboost import CatBoostClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'depth':[3,1,2,6,4,5,7,8,9,10],
                        'iterations':[250,100,500,1000], 
                        'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
                        'l2_leaf_reg':[3,1,5,10,100], 
                        'border_count':[32,5,10,20,50,100,200], 
                        }
        
        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_ 
        
    elif estimator == 'Bagging':
        
        from sklearn.ensemble import BaggingClassifier

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'n_estimators': np.arange(10,300,10),
                        'bootstrap': [True, False],
                        'bootstrap_features': [True, False],
                        }
            
        model_grid = RandomizedSearchCV(estimator=estimator_clone, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_ 
        
    progress.value += 1
    progress.value += 1
    progress.value += 1

    logger.info("Random search completed")
        
    #multiclass checking
    if y.value_counts().count() > 2:
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(model)
        best_model = model
        
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    fold_num = 1
    
    for train_i , test_i in kf.split(data_X,data_y):
        
        logger.info("Initializing Fold " + str(fold_num))

        t0 = time.time()
        
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
        time_start=time.time()

        if fix_imbalance_param:
            
            logger.info("Initializing SMOTE")

            if fix_imbalance_method_param is None:
                import six
                import sys
                sys.modules['sklearn.externals.six'] = six
                from imblearn.over_sampling import SMOTE
                resampler = SMOTE(random_state = seed)
            else:
                resampler = fix_imbalance_method_param

            Xtrain,ytrain = resampler.fit_sample(Xtrain, ytrain)
            logger.info("Resampling completed")

        if hasattr(model, 'predict_proba'):
            logger.info("Fitting Model")
            model.fit(Xtrain,ytrain)
            logger.info("Evaluating Metrics")
            pred_prob = model.predict_proba(Xtest)
            pred_prob = pred_prob[:,1]
            pred_ = model.predict(Xtest)
            sca = metrics.accuracy_score(ytest,pred_)
            
            if y.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(ytest,pred_, average='macro')                
                precision = metrics.precision_score(ytest,pred_, average = 'weighted')
                f1 = metrics.f1_score(ytest,pred_, average='weighted')
                
            else:
                try:
                    sc = metrics.roc_auc_score(ytest,pred_prob)
                except:
                    sc = 0
                    logger.warning("model has no predict_proba attribute. AUC set to 0.00")
                recall = metrics.recall_score(ytest,pred_)                
                precision = metrics.precision_score(ytest,pred_)
                f1 = metrics.f1_score(ytest,pred_)
                
        else:
            logger.info("Fitting Model")
            model.fit(Xtrain,ytrain)
            logger.info("Evaluating Metrics")
            pred_prob = 0.00
            logger.warning("model has no predict_proba attribute. pred_prob set to 0.00")
            pred_ = model.predict(Xtest)
            sca = metrics.accuracy_score(ytest,pred_)
            
            if y.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(ytest,pred_, average='macro')                
                precision = metrics.precision_score(ytest,pred_, average = 'weighted')
                f1 = metrics.f1_score(ytest,pred_, average='weighted')

            else:
                try:
                    sc = metrics.roc_auc_score(ytest,pred_prob)
                except:
                    sc = 0
                    logger.warning("model has no predict_proba attribute. AUC set to 0.00")
                recall = metrics.recall_score(ytest,pred_)                
                precision = metrics.precision_score(ytest,pred_)
                f1 = metrics.f1_score(ytest,pred_)

        logger.info("Compiling Metrics")
        time_end=time.time()
        kappa = metrics.cohen_kappa_score(ytest,pred_)
        mcc = metrics.matthews_corrcoef(ytest,pred_)
        training_time=time_end-time_start
        score_acc = np.append(score_acc,sca)
        score_auc = np.append(score_auc,sc)
        score_recall = np.append(score_recall,recall)
        score_precision = np.append(score_precision,precision)
        score_f1 =np.append(score_f1,f1)
        score_kappa =np.append(score_kappa,kappa)
        score_mcc=np.append(score_mcc,mcc)
        score_training_time=np.append(score_training_time,training_time)
        
        progress.value += 1
            
            
        '''
        
        This section is created to update_display() as code loops through the fold defined.
        
        '''

        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa], 'MCC':[mcc]}).round(round)
        master_display = pd.concat([master_display, fold_results],ignore_index=True)
        fold_results = []
        
        '''
        
        TIME CALCULATION SUB-SECTION STARTS HERE
        
        '''
        
        t1 = time.time()
        
        tt = (t1 - t0) * (fold-fold_num) / 60
        tt = np.around(tt, 2)
        
        if tt < 1:
            tt = str(np.around((tt * 60), 2))
            ETC = tt + ' Seconds Remaining'
                
        else:
            tt = str (tt)
            ETC = tt + ' Minutes Remaining'
            
        if verbose:
            if html_param:
                update_display(ETC, display_id = 'ETC')

        fold_num += 1
        
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[2,1:] = ETC
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
       
        '''
        
        TIME CALCULATION ENDS HERE
        
        '''
        
        if verbose:
            if html_param:
                update_display(master_display, display_id = display_id)
        
        '''
        
        Update_display() ends here
        
        '''
        
    progress.value += 1
    
    logger.info("Calculating mean and std")
    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    mean_mcc=np.mean(score_mcc)
    mean_training_time=np.sum(score_training_time)
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    std_mcc=np.std(score_mcc)
    std_training_time=np.std(score_training_time)
    
    avgs_acc = np.append(avgs_acc, mean_acc)
    avgs_acc = np.append(avgs_acc, std_acc) 
    avgs_auc = np.append(avgs_auc, mean_auc)
    avgs_auc = np.append(avgs_auc, std_auc)
    avgs_recall = np.append(avgs_recall, mean_recall)
    avgs_recall = np.append(avgs_recall, std_recall)
    avgs_precision = np.append(avgs_precision, mean_precision)
    avgs_precision = np.append(avgs_precision, std_precision)
    avgs_f1 = np.append(avgs_f1, mean_f1)
    avgs_f1 = np.append(avgs_f1, std_f1)
    avgs_kappa = np.append(avgs_kappa, mean_kappa)
    avgs_kappa = np.append(avgs_kappa, std_kappa)
    
    avgs_mcc = np.append(avgs_mcc, mean_mcc)
    avgs_mcc = np.append(avgs_mcc, std_mcc)
    avgs_training_time = np.append(avgs_training_time, mean_training_time)
    avgs_training_time = np.append(avgs_training_time, std_training_time)
    
    progress.value += 1
    
    logger.info("Creating metrics dataframe")
    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa, 'MCC':score_mcc})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa, 'MCC':avgs_mcc},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    
    # yellow the mean
    model_results=model_results.style.apply(lambda x: ['background: yellow' if (x.name == 'Mean') else '' for i in x], axis=1)
    model_results = model_results.set_precision(round)

    progress.value += 1
    
    #refitting the model on complete X_train, y_train
    monitor.iloc[1,1:] = 'Finalizing Model'
    monitor.iloc[2,1:] = 'Almost Finished'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    model_fit_start = time.time()
    logger.info("Finalizing model")
    best_model.fit(data_X, data_y)
    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
    
    progress.value += 1
    
    #storing results in create_model_container
    logger.info("Uploading results into container")
    create_model_container.append(model_results.data)
    display_container.append(model_results.data)

    #storing results in master_model_container
    logger.info("Uploading model into container")
    master_model_container.append(best_model)

    '''
    When choose_better sets to True. optimize metric in scoregrid is
    compared with base model created using create_model so that tune_model
    functions return the model with better score only. This will ensure 
    model performance is atleast equivalent to what is seen is compare_models 
    '''
    if choose_better:
        logger.info("choose_better activated")
        if verbose:
            if html_param:
                monitor.iloc[1,1:] = 'Compiling Final Results'
                monitor.iloc[2,1:] = 'Almost Finished'
                update_display(monitor, display_id = 'monitor')

        #creating base model for comparison
        logger.info("SubProcess create_model() called ==================================")
        if estimator in ['Bagging', 'ada']:
            base_model = create_model(estimator=_estimator_, verbose = False, system=False)
        else:
            base_model = create_model(estimator=estimator, verbose = False, system=False)
        logger.info("SubProcess create_model() called ==================================")
        base_model_results = create_model_container[-1][compare_dimension][-2:][0]
        tuned_model_results = create_model_container[-2][compare_dimension][-2:][0]

        if tuned_model_results > base_model_results:
            best_model = best_model
        else:
            best_model = base_model

        #re-instate display_constainer state 
        display_container.pop(-1)
        logger.info("choose_better completed")

    #end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)
    
    #mlflow logging
    if logging_param:

        logger.info("Creating MLFlow logs")

        #Creating Logs message monitor
        monitor.iloc[1,1:] = 'Creating Logs'
        monitor.iloc[2,1:] = 'Almost Finished'    
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        import mlflow
        from pathlib import Path
        import os
        
        mlflow.set_experiment(exp_name_log)
        full_name = model_dict_logging.get(mn)

        with mlflow.start_run(run_name=full_name) as run:    

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            params = best_model.get_params()

            # Log model parameters
            params = model.get_params()

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            mlflow.log_params(params)

            mlflow.log_metrics({"Accuracy": avgs_acc[0], "AUC": avgs_auc[0], "Recall": avgs_recall[0], "Precision" : avgs_precision[0],
                                "F1": avgs_f1[0], "Kappa": avgs_kappa[0], "MCC": avgs_mcc[0]})

            #set tag of compare_models
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
            model_results.data.to_html('Results.html', col_space=65, justify='left')
            mlflow.log_artifact('Results.html')
            os.remove('Results.html')

            # Generate hold-out predictions and save as html
            holdout = predict_model(best_model, verbose=False)
            holdout_score = pull()
            del(holdout)
            display_container.pop(-1)
            holdout_score.to_html('Holdout.html', col_space=65, justify='left')
            mlflow.log_artifact('Holdout.html')
            os.remove('Holdout.html')

            # Log AUC and Confusion Matrix plot
            if log_plots_param:

                logger.info("SubProcess plot_model() called ==================================")

                try:
                    plot_model(model, plot = 'auc', verbose=False, save=True, system=False)
                    mlflow.log_artifact('AUC.png')
                    os.remove("AUC.png")
                except:
                    pass

                try:
                    plot_model(model, plot = 'confusion_matrix', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Confusion Matrix.png')
                    os.remove("Confusion Matrix.png")
                except:
                    pass

                try:
                    plot_model(model, plot = 'feature', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Feature Importance.png')
                    os.remove("Feature Importance.png")
                except:
                    pass

                logger.info("SubProcess plot_model() end ==================================")

            # Log hyperparameter tuning grid
            d1 = model_grid.cv_results_.get('params')
            dd = pd.DataFrame.from_dict(d1)
            dd['Score'] = model_grid.cv_results_.get('mean_test_score')
            dd.to_html('Iterations.html', col_space=75, justify='left')
            mlflow.log_artifact('Iterations.html')
            os.remove('Iterations.html')

            # Log model and transformation pipeline
            from copy import deepcopy

            # get default conda env
            from mlflow.sklearn import get_default_conda_env
            default_conda_env = get_default_conda_env()
            default_conda_env['name'] = str(exp_name_log) + '-env'
            default_conda_env.get('dependencies').pop(-3)
            dependencies = default_conda_env.get('dependencies')[-1]
            from pycaret.utils import __version__
            dep = 'pycaret==' + str(__version__())
            dependencies['pip'] = [dep]
            
            # define model signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(data_before_preprocess.drop([target_param], axis=1))
            input_example = data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(['trained model', model])
            mlflow.sklearn.log_model(prep_pipe_temp, "model", conda_env = default_conda_env, signature = signature, input_example = input_example)
            del(prep_pipe_temp)
        
    if verbose:
        clear_output()
        if html_param:
            display(model_results)
        else:
            print(model_results.data)
    
    logger.info("create_model_container: " + str(len(create_model_container)))
    logger.info("master_model_container: " + str(len(master_model_container)))
    logger.info("display_container: " + str(len(display_container)))

    logger.info(str(best_model))
    logger.info("tune_model() succesfully completed......................................")
    
    return best_model

def ensemble_model(estimator,
                   method = 'Bagging', 
                   fold = 10,
                   n_estimators = 10,
                   round = 4,  
                   choose_better = False, #added in pycaret==2.0.0
                   optimize = 'Accuracy', #added in pycaret==2.0.0
                   verbose = True):
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
    
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing ensemble_model()")
    logger.info("""ensemble_model(estimator={}, method={}, fold={}, n_estimators={}, round={}, choose_better={}, optimize={}, verbose={})""".\
        format(str(estimator), str(method), str(fold), str(n_estimators), str(round), str(choose_better), str(optimize), str(verbose)))

    logger.info("Checking exceptions")

    #exception checking   
    import sys

    #run_time
    import datetime, time
    runtime_start = time.time()
        
    #Check for allowed method
    available_method = ['Bagging', 'Boosting']
    if method not in available_method:
        sys.exit("(Value Error): Method parameter only accepts two values 'Bagging' or 'Boosting'.")
    
    
    #check boosting conflict
    if method == 'Boosting':
        
        from sklearn.ensemble import AdaBoostClassifier
        
        try:
            if hasattr(estimator,'n_classes_'):
                if estimator.n_classes_ > 2:
                    check_model = estimator.estimator
                    check_model = AdaBoostClassifier(check_model, n_estimators=10, random_state=seed)
                    from sklearn.multiclass import OneVsRestClassifier
                    check_model = OneVsRestClassifier(check_model)
                    check_model.fit(X_train, y_train)
            else:
                check_model = AdaBoostClassifier(estimator, n_estimators=10, random_state=seed)
                check_model.fit(X_train, y_train)
        except:
            sys.exit("(Type Error): Estimator does not provide class_weights or predict_proba function and hence not supported for the Boosting method. Change the estimator or method to 'Bagging'.") 
        
    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    
    #checking n_estimators parameter
    if type(n_estimators) is not int:
        sys.exit('(Type Error): n_estimators parameter only accepts integer value.')
    
    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')
 
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.') 
    
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''    
    
    logger.info("Preloading libraries")

    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    logger.info("Preparing display monitor")

    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+4, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa', 'MCC'])
    if verbose:
        if html_param:
            display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    if verbose:
        if html_param:
            display(monitor, display_id = 'monitor')
    
    if verbose:
        if html_param:
            display_ = display(master_display, display_id=True)
            display_id = display_.display_id

    logger.info("Importing libraries")

    #dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold   
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore')    
    
    logger.info("Copying training dataset")

    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()
    
    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
    
    progress.value += 1
    
    #defining estimator as model
    model = estimator
    
    if optimize == 'Accuracy':
        compare_dimension = 'Accuracy' 
    elif optimize == 'AUC':
        compare_dimension = 'AUC' 
    elif optimize == 'Recall':
        compare_dimension = 'Recall'
    elif optimize == 'Precision':
        compare_dimension = 'Prec.'
    elif optimize == 'F1':
        compare_dimension = 'F1' 
    elif optimize == 'Kappa':
        compare_dimension = 'Kappa'
    elif optimize == 'MCC':
        compare_dimension = 'MCC' 
    
    logger.info("Checking base model")

    def get_model_name(e):
        return str(e).split("(")[0]

    if y.value_counts().count() > 2:
        mn = get_model_name(estimator.estimator)
    else:
        mn = get_model_name(estimator)

    if 'catboost' in str(estimator):
        mn = 'CatBoostClassifier'
    
    model_dict = {'ExtraTreesClassifier' : 'et',
                'GradientBoostingClassifier' : 'gbc', 
                'RandomForestClassifier' : 'rf',
                'LGBMClassifier' : 'lightgbm',
                'XGBClassifier' : 'xgboost',
                'AdaBoostClassifier' : 'ada', 
                'DecisionTreeClassifier' : 'dt', 
                'RidgeClassifier' : 'ridge',
                'LogisticRegression' : 'lr',
                'KNeighborsClassifier' : 'knn',
                'GaussianNB' : 'nb',
                'SGDClassifier' : 'svm',
                'SVC' : 'rbfsvm',
                'GaussianProcessClassifier' : 'gpc',
                'MLPClassifier' : 'mlp',
                'QuadraticDiscriminantAnalysis' : 'qda',
                'LinearDiscriminantAnalysis' : 'lda',
                'CatBoostClassifier' : 'catboost',
                'BaggingClassifier' : 'Bagging'}

    estimator__ = model_dict.get(mn)

    model_dict_logging = {'ExtraTreesClassifier' : 'Extra Trees Classifier',
                        'GradientBoostingClassifier' : 'Gradient Boosting Classifier', 
                        'RandomForestClassifier' : 'Random Forest Classifier',
                        'LGBMClassifier' : 'Light Gradient Boosting Machine',
                        'XGBClassifier' : 'Extreme Gradient Boosting',
                        'AdaBoostClassifier' : 'Ada Boost Classifier', 
                        'DecisionTreeClassifier' : 'Decision Tree Classifier', 
                        'RidgeClassifier' : 'Ridge Classifier',
                        'LogisticRegression' : 'Logistic Regression',
                        'KNeighborsClassifier' : 'K Neighbors Classifier',
                        'GaussianNB' : 'Naive Bayes',
                        'SGDClassifier' : 'SVM - Linear Kernel',
                        'SVC' : 'SVM - Radial Kernel',
                        'GaussianProcessClassifier' : 'Gaussian Process Classifier',
                        'MLPClassifier' : 'MLP Classifier',
                        'QuadraticDiscriminantAnalysis' : 'Quadratic Discriminant Analysis',
                        'LinearDiscriminantAnalysis' : 'Linear Discriminant Analysis',
                        'CatBoostClassifier' : 'CatBoost Classifier',
                        'BaggingClassifier' : 'Bagging Classifier'}

    logger.info('Base model : ' + str(model_dict_logging.get(mn)))

    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Selecting Estimator'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    if hasattr(estimator,'n_classes_'):
        if estimator.n_classes_ > 2:
            model = estimator.estimator

    logger.info("Importing untrained ensembler")

    if method == 'Bagging':
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier(model,bootstrap=True,n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs_param)
        logger.info("BaggingClassifier() succesfully imported")

    else:
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(model, n_estimators=n_estimators, random_state=seed)
        logger.info("AdaBoostClassifier() succesfully imported")

    if y.value_counts().count() > 2:
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(model)
        logger.info("OneVsRestClassifier() succesfully imported")
        
    progress.value += 1
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    logger.info("Defining folds")
    kf = StratifiedKFold(fold, random_state=seed, shuffle=folds_shuffle_param)
    
    logger.info("Declaring metric variables")
    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    score_mcc =np.empty((0,0))
    score_training_time =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    avgs_mcc =np.empty((0,0))
    avgs_training_time =np.empty((0,0))
    
    
    fold_num = 1 
    
    for train_i , test_i in kf.split(data_X,data_y):

        logger.info("Initializing Fold " + str(fold_num))
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
        time_start=time.time()

        if fix_imbalance_param:
            logger.info("Initializing SMOTE")
            
            if fix_imbalance_method_param is None:
                import six
                import sys
                sys.modules['sklearn.externals.six'] = six
                from imblearn.over_sampling import SMOTE
                resampler = SMOTE(random_state=seed)
            else:
                resampler = fix_imbalance_method_param

            Xtrain,ytrain = resampler.fit_sample(Xtrain, ytrain)
            logger.info("Resampling completed")

        if hasattr(model, 'predict_proba'):
            logger.info("Fitting Model")
            model.fit(Xtrain,ytrain)
            logger.info("Evaluating Metrics")
            pred_prob = model.predict_proba(Xtest)
            pred_prob = pred_prob[:,1]
            pred_ = model.predict(Xtest)
            sca = metrics.accuracy_score(ytest,pred_)
            
            if y.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(ytest,pred_, average='macro')                
                precision = metrics.precision_score(ytest,pred_, average = 'weighted')
                f1 = metrics.f1_score(ytest,pred_, average='weighted')
                
            else:
                try:
                    sc = metrics.roc_auc_score(ytest,pred_prob)
                except:
                    sc = 0
                    logger.warning("model has no predict_proba attribute. AUC set to 0.00")
                recall = metrics.recall_score(ytest,pred_)                
                precision = metrics.precision_score(ytest,pred_)
                f1 = metrics.f1_score(ytest,pred_)
        else:
            logger.info("Fitting Model")
            model.fit(Xtrain,ytrain)
            logger.info("Evaluating Metrics")
            pred_prob = 0.00
            logger.warning("model has no predict_proba attribute. pred_prob set to 0.00")
            pred_ = model.predict(Xtest)
            sca = metrics.accuracy_score(ytest,pred_)
            
            if y.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(ytest,pred_, average='macro')                
                precision = metrics.precision_score(ytest,pred_, average = 'weighted')
                f1 = metrics.f1_score(ytest,pred_, average='weighted')

            else:
                try:
                    sc = metrics.roc_auc_score(ytest,pred_prob)
                except:
                    sc = 0
                    logger.warning("model has no predict_proba attribute. AUC set to 0.00")
                recall = metrics.recall_score(ytest,pred_)                
                precision = metrics.precision_score(ytest,pred_)
                f1 = metrics.f1_score(ytest,pred_)

        logger.info("Compiling Metrics")        
        time_end=time.time()
        kappa = metrics.cohen_kappa_score(ytest,pred_)
        mcc = metrics.matthews_corrcoef(ytest,pred_)
        training_time=time_end-time_start
        score_acc = np.append(score_acc,sca)
        score_auc = np.append(score_auc,sc)
        score_recall = np.append(score_recall,recall)
        score_precision = np.append(score_precision,precision)
        score_f1 =np.append(score_f1,f1)
        score_kappa =np.append(score_kappa,kappa) 
        score_mcc =np.append(score_mcc,mcc)
        score_training_time =np.append(score_training_time,training_time)
        progress.value += 1
        
                
        '''
        This section is created to update_display() as code loops through the fold defined.
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa], 'MCC':[mcc]}).round(round)
        master_display = pd.concat([master_display, fold_results],ignore_index=True)
        fold_results = []
        
        '''
        
        TIME CALCULATION SUB-SECTION STARTS HERE
        
        '''
        t1 = time.time()
        
        tt = (t1 - t0) * (fold-fold_num) / 60
        tt = np.around(tt, 2)
        
        if tt < 1:
            tt = str(np.around((tt * 60), 2))
            ETC = tt + ' Seconds Remaining'
                
        else:
            tt = str (tt)
            ETC = tt + ' Minutes Remaining'
            
        if verbose:
            if html_param:
                update_display(ETC, display_id = 'ETC')
            
        fold_num += 1
        
        
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[2,1:] = ETC
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        '''
        
        TIME CALCULATION ENDS HERE
        
        '''

        if verbose:
            if html_param:
                update_display(master_display, display_id = display_id)
        
        '''
        
        Update_display() ends here
        
        '''
        
    logger.info("Calculating mean and std")
    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    mean_mcc=np.mean(score_mcc)
    mean_training_time=np.sum(score_training_time)
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    std_mcc=np.std(score_mcc)
    std_training_time=np.std(score_training_time)

    avgs_acc = np.append(avgs_acc, mean_acc)
    avgs_acc = np.append(avgs_acc, std_acc) 
    avgs_auc = np.append(avgs_auc, mean_auc)
    avgs_auc = np.append(avgs_auc, std_auc)
    avgs_recall = np.append(avgs_recall, mean_recall)
    avgs_recall = np.append(avgs_recall, std_recall)
    avgs_precision = np.append(avgs_precision, mean_precision)
    avgs_precision = np.append(avgs_precision, std_precision)
    avgs_f1 = np.append(avgs_f1, mean_f1)
    avgs_f1 = np.append(avgs_f1, std_f1)
    avgs_kappa = np.append(avgs_kappa, mean_kappa)
    avgs_kappa = np.append(avgs_kappa, std_kappa)
    
    avgs_mcc = np.append(avgs_mcc, mean_mcc)
    avgs_mcc = np.append(avgs_mcc, std_mcc)
    
    avgs_training_time = np.append(avgs_training_time, mean_training_time)
    avgs_training_time = np.append(avgs_training_time, std_training_time)

    logger.info("Creating metrics dataframe")
    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa, 'MCC':score_mcc})
    model_results_unpivot = pd.melt(model_results,value_vars=['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa','MCC'])
    model_results_unpivot.columns = ['Metric', 'Measure']
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa,'MCC':avgs_mcc},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)  
    
    # yellow the mean
    model_results=model_results.style.apply(lambda x: ['background: yellow' if (x.name == 'Mean') else '' for i in x], axis=1)
    model_results = model_results.set_precision(round)

    progress.value += 1
    
    #refitting the model on complete X_train, y_train
    monitor.iloc[1,1:] = 'Finalizing Model'
    monitor.iloc[2,1:] = 'Almost Finished'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    model_fit_start = time.time()
    logger.info("Finalizing model")
    model.fit(data_X, data_y)
    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
    
    #storing results in create_model_container
    logger.info("Uploading results into container")
    create_model_container.append(model_results.data)
    display_container.append(model_results.data)

    #storing results in master_model_container
    logger.info("Uploading model into container")
    master_model_container.append(model)

    progress.value += 1
    
    '''
    When choose_better sets to True. optimize metric in scoregrid is
    compared with base model created using create_model so that ensemble_model
    functions return the model with better score only. This will ensure 
    model performance is atleast equivalent to what is seen is compare_models 
    '''
    if choose_better:

        logger.info("choose_better activated")

        if verbose:
            if html_param:
                monitor.iloc[1,1:] = 'Compiling Final Results'
                monitor.iloc[2,1:] = 'Almost Finished'
                update_display(monitor, display_id = 'monitor')

        #creating base model for comparison
        logger.info("SubProcess create_model() called ==================================")
        base_model = create_model(estimator=estimator, verbose = False, system=False)
        logger.info("SubProcess create_model() end ==================================")
        base_model_results = create_model_container[-1][compare_dimension][-2:][0]
        ensembled_model_results = create_model_container[-2][compare_dimension][-2:][0]

        if ensembled_model_results > base_model_results:
            model = model
        else:
            model = base_model

        #re-instate display_constainer state 
        display_container.pop(-1)
        logger.info("choose_better completed")

    #end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)
    
    if logging_param:

        logger.info("Creating MLFlow logs")

        #Creating Logs message monitor
        monitor.iloc[1,1:] = 'Creating Logs'
        monitor.iloc[2,1:] = 'Almost Finished'    
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')


        import mlflow
        from pathlib import Path
        import os

        mlflow.set_experiment(exp_name_log)
        full_name = model_dict_logging.get(mn)

        with mlflow.start_run(run_name=full_name) as run:        

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            params = model.get_params()

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            mlflow.log_params(params)
            mlflow.log_metrics({"Accuracy": avgs_acc[0], "AUC": avgs_auc[0], "Recall": avgs_recall[0], "Precision" : avgs_precision[0],
                                "F1": avgs_f1[0], "Kappa": avgs_kappa[0], "MCC": avgs_mcc[0]})
            
            #set tag of compare_models
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
            holdout_score = pull()
            del(holdout)
            display_container.pop(-1)
            holdout_score.to_html('Holdout.html', col_space=65, justify='left')
            mlflow.log_artifact('Holdout.html')
            os.remove('Holdout.html')

            # Log AUC and Confusion Matrix plot
            if log_plots_param:
                
                logger.info("SubProcess plot_model() called ==================================")
                
                try:
                    plot_model(model, plot = 'auc', verbose=False, save=True, system=False)
                    mlflow.log_artifact('AUC.png')
                    os.remove("AUC.png")
                except:
                    pass

                try:
                    plot_model(model, plot = 'confusion_matrix', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Confusion Matrix.png')
                    os.remove("Confusion Matrix.png")
                except:
                    pass

                try:
                    plot_model(model, plot = 'feature', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Feature Importance.png')
                    os.remove("Feature Importance.png")
                except:
                    pass

                logger.info("SubProcess plot_model() end ==================================")

            # Log the CV results as model_results.html artifact
            model_results.data.to_html('Results.html', col_space=65, justify='left')
            mlflow.log_artifact('Results.html')
            os.remove('Results.html')

            # Log model and transformation pipeline
            from copy import deepcopy

            # get default conda env
            from mlflow.sklearn import get_default_conda_env
            default_conda_env = get_default_conda_env()
            default_conda_env['name'] = str(exp_name_log) + '-env'
            default_conda_env.get('dependencies').pop(-3)
            dependencies = default_conda_env.get('dependencies')[-1]
            from pycaret.utils import __version__
            dep = 'pycaret==' + str(__version__())
            dependencies['pip'] = [dep]
            
            # define model signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(data_before_preprocess.drop([target_param], axis=1))
            input_example = data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(['trained model', model])
            mlflow.sklearn.log_model(prep_pipe_temp, "model", conda_env = default_conda_env, signature = signature, input_example = input_example)
            del(prep_pipe_temp)

    if verbose:
        clear_output()
        if html_param:
            display(model_results)
        else:
            print(model_results.data)
    else:
        clear_output()

    logger.info("create_model_container: " + str(len(create_model_container)))
    logger.info("master_model_container: " + str(len(master_model_container)))
    logger.info("display_container: " + str(len(display_container)))

    logger.info(str(model))
    logger.info("ensemble_model() succesfully completed......................................")

    return model

def blend_models(estimator_list = 'All', 
                 fold = 10, 
                 round = 4,
                 choose_better = False, #added in pycaret==2.0.0 
                 optimize = 'Accuracy', #added in pycaret==2.0.0 
                 method = 'hard',
                 turbo = True,
                 verbose = True):
    
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
    
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing blend_models()")
    logger.info("""blend_models(estimator_list={}, fold={}, round={}, choose_better={}, optimize={}, method={}, turbo={}, verbose={})""".\
        format(str(estimator_list), str(fold), str(round), str(choose_better), str(optimize), str(method), str(turbo), str(verbose)))

    logger.info("Checking exceptions")

    #exception checking   
    import sys

    #run_time
    import datetime, time
    runtime_start = time.time()
    
    #checking error for estimator_list (string)
    
    if estimator_list != 'All':
        if type(estimator_list) is not list:
            sys.exit("(Value Error): estimator_list parameter only accepts 'All' as string or list of trained models.")

    if estimator_list != 'All':
        for i in estimator_list:
            if 'sklearn' not in str(type(i)) and 'CatBoostClassifier' not in str(type(i)):
                sys.exit("(Value Error): estimator_list parameter only accepts 'All' as string or trained model object.")

    #checking method param with estimator list
    if estimator_list != 'All':
        if method == 'soft':
            
            check = 0
            
            for i in estimator_list:
                if hasattr(i, 'predict_proba'):
                    pass
                else:
                    check += 1
            
            if check >= 1:
                sys.exit('(Type Error): Estimator list contains estimator that doesnt support probabilities and method is forced to soft. Either change the method or drop the estimator.')
    
    #checking catboost:
    if estimator_list != 'All':
        for i in estimator_list:
            if 'CatBoostClassifier' in str(i):
                sys.exit('(Type Error): CatBoost Classifier not supported in this function.')
    
    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    
    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')
 
    #checking method parameter
    available_method = ['soft', 'hard']
    if method not in available_method:
        sys.exit("(Value Error): Method parameter only accepts 'soft' or 'hard' as a parameter. See Docstring for details.")
    
    #checking verbose parameter
    if type(turbo) is not bool:
        sys.exit('(Type Error): Turbo parameter can only take argument as True or False.') 
        
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.') 
        
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    logger.info("Preloading libraries")
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    logger.info("Preparing display monitor")
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+4, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa', 'MCC'])
    if verbose:
        if html_param:
            display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    if verbose:
        if html_param:
            display(monitor, display_id = 'monitor')
    
    if verbose:
        if html_param:
            display_ = display(master_display, display_id=True)
            display_id = display_.display_id
        
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    logger.info("Importing libraries")
    #general dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold  
    from sklearn.ensemble import VotingClassifier
    import re
    
    logger.info("Copying training dataset")
    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()
    
    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
    
    if optimize == 'Accuracy':
        compare_dimension = 'Accuracy' 
    elif optimize == 'AUC':
        compare_dimension = 'AUC' 
    elif optimize == 'Recall':
        compare_dimension = 'Recall'
    elif optimize == 'Precision':
        compare_dimension = 'Prec.'
    elif optimize == 'F1':
        compare_dimension = 'F1' 
    elif optimize == 'Kappa':
        compare_dimension = 'Kappa'
    elif optimize == 'MCC':
        compare_dimension = 'MCC' 

    #estimator_list_flag
    if estimator_list == 'All':
        all_flag = True
    else:
        all_flag = False
        
    progress.value += 1
    
    logger.info("Declaring metric variables")
    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    score_mcc =np.empty((0,0))
    score_training_time =np.empty((0,0))
    
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    avgs_mcc =np.empty((0,0))
    avgs_training_time =np.empty((0,0))
    
    avg_acc = np.empty((0,0))
    avg_auc = np.empty((0,0))
    avg_recall = np.empty((0,0))
    avg_precision = np.empty((0,0))
    avg_f1 = np.empty((0,0))
    avg_kappa = np.empty((0,0))
    avg_mcc = np.empty((0,0))
    avg_training_time = np.empty((0,0))
    
    logger.info("Defining folds")
    kf = StratifiedKFold(fold, random_state=seed, shuffle=folds_shuffle_param)
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Compiling Estimators'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    if estimator_list == 'All':

        logger.info("Importing untrained models")
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
        from sklearn.ensemble import BaggingClassifier 
        from xgboost import XGBClassifier
        import lightgbm as lgb
        
        lr = LogisticRegression(random_state=seed) #don't add n_jobs parameter as it slows down the LR
        knn = KNeighborsClassifier(n_jobs=n_jobs_param)
        nb = GaussianNB()
        dt = DecisionTreeClassifier(random_state=seed)
        svm = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed, n_jobs=n_jobs_param)
        rbfsvm = SVC(gamma='auto', C=1, probability=True, kernel='rbf', random_state=seed)
        gpc = GaussianProcessClassifier(random_state=seed, n_jobs=n_jobs_param)
        mlp = MLPClassifier(max_iter=500, random_state=seed)
        ridge = RidgeClassifier(random_state=seed)
        rf = RandomForestClassifier(n_estimators=10, random_state=seed, n_jobs=n_jobs_param)
        qda = QuadraticDiscriminantAnalysis()
        ada = AdaBoostClassifier(random_state=seed)
        gbc = GradientBoostingClassifier(random_state=seed)
        lda = LinearDiscriminantAnalysis()
        et = ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs_param)
        xgboost = XGBClassifier(random_state=seed, verbosity=0, n_jobs=n_jobs_param)
        lightgbm = lgb.LGBMClassifier(random_state=seed, n_jobs=n_jobs_param)

        logger.info("Import successful")

        progress.value += 1
        
        logger.info("Defining estimator list")
        if turbo:
            if method == 'hard':
                estimator_list = [lr,knn,nb,dt,svm,ridge,rf,qda,ada,gbc,lda,et,xgboost,lightgbm]
                voting = 'hard'
            elif method == 'soft':
                estimator_list = [lr,knn,nb,dt,rf,qda,ada,gbc,lda,et,xgboost,lightgbm]
                voting = 'soft'
        else:
            if method == 'hard':
                estimator_list = [lr,knn,nb,dt,svm,rbfsvm,gpc,mlp,ridge,rf,qda,ada,gbc,lda,et,xgboost,lightgbm]
                voting = 'hard'
            elif method == 'soft':
                estimator_list = [lr,knn,nb,dt,rbfsvm,gpc,mlp,rf,qda,ada,gbc,lda,et,xgboost,lightgbm]
                voting = 'soft'
                
    else:

        estimator_list = estimator_list
        voting = method  
        
    logger.info("Defining model names in estimator_list")
    model_names = []

    for names in estimator_list:

        model_names = np.append(model_names, str(names).split("(")[0])

    def putSpace(input):
        words = re.findall('[A-Z][a-z]*', input)
        words = ' '.join(words)
        return words  

    model_names_modified = []
    
    for i in model_names:
        
        model_names_modified.append(putSpace(i))
        model_names = model_names_modified
    
    model_names_final = []
  
    for j in model_names_modified:

        if j == 'Gaussian N B':
            model_names_final.append('Naive Bayes')

        elif j == 'M L P Classifier':
            model_names_final.append('MLP Classifier')

        elif j == 'S G D Classifier':
            model_names_final.append('SVM - Linear Kernel')

        elif j == 'S V C':
            model_names_final.append('SVM - Radial Kernel')
        
        elif j == 'X G B Classifier':
            model_names_final.append('Extreme Gradient Boosting')
        
        elif j == 'L G B M Classifier':
            model_names_final.append('Light Gradient Boosting Machine')
            
        else: 
            model_names_final.append(j)
            
    model_names = model_names_final
    
    #adding n in model_names to avoid duplicate exception when custom list is passed for eg. BaggingClassifier
    
    model_names_n = []
    counter = 0
    
    for i in model_names:
        mn = str(i) + '_' + str(counter)
        model_names_n.append(mn)
        counter += 1
        
    model_names = model_names_n

    estimator_list = estimator_list

    estimator_list_ = zip(model_names, estimator_list)
    estimator_list_ = set(estimator_list_)
    estimator_list_ = list(estimator_list_)
    
    try:
        model = VotingClassifier(estimators=estimator_list_, voting=voting, n_jobs=n_jobs_param)
        model.fit(data_X,data_y)
        logger.info("n_jobs multiple passed")
    except:
        logger.info("n_jobs multiple failed")
        model = VotingClassifier(estimators=estimator_list_, voting=voting)
    
    progress.value += 1
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    fold_num = 1
    
    for train_i , test_i in kf.split(data_X,data_y):
        
        logger.info("Initializing Fold " + str(fold_num))

        progress.value += 1
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
    
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]    
        time_start=time.time()

        if fix_imbalance_param:
            logger.info("Initializing SMOTE")
            if fix_imbalance_method_param is None:
                import six
                import sys
                sys.modules['sklearn.externals.six'] = six
                from imblearn.over_sampling import SMOTE
                resampler = SMOTE(random_state = seed)
            else:
                resampler = fix_imbalance_method_param

            Xtrain,ytrain = resampler.fit_sample(Xtrain, ytrain)
            logger.info("Resampling completed")

        if voting == 'hard':
            logger.info("Fitting Model")
            model.fit(Xtrain,ytrain)
            logger.info("Evaluating Metrics")
            pred_prob = 0.0
            logger.warning("model has no predict_proba attribute. pred_prob set to 0.00")
            pred_ = model.predict(Xtest)
            sca = metrics.accuracy_score(ytest,pred_)
            sc = 0.0
            if y.value_counts().count() > 2:
                recall = metrics.recall_score(ytest,pred_, average='macro')
                precision = metrics.precision_score(ytest,pred_, average='weighted')
                f1 = metrics.f1_score(ytest,pred_, average='weighted')    
            else:
                recall = metrics.recall_score(ytest,pred_)
                precision = metrics.precision_score(ytest,pred_)
                f1 = metrics.f1_score(ytest,pred_) 
                
        else:
            logger.info("Fitting Model")
            model.fit(Xtrain,ytrain)
            logger.info("Evaluating Metrics")
            pred_ = model.predict(Xtest)
            sca = metrics.accuracy_score(ytest,pred_)
            
            if y.value_counts().count() > 2:
                pred_prob = 0
                sc = 0
                recall = metrics.recall_score(ytest,pred_, average='macro')
                precision = metrics.precision_score(ytest,pred_, average='weighted')
                f1 = metrics.f1_score(ytest,pred_, average='weighted')
            else:
                try:
                    pred_prob = model.predict_proba(Xtest)
                    pred_prob = pred_prob[:,1]
                    sc = metrics.roc_auc_score(ytest,pred_prob)
                except:
                    sc = 0
                recall = metrics.recall_score(ytest,pred_)
                precision = metrics.precision_score(ytest,pred_)
                f1 = metrics.f1_score(ytest,pred_)
            
        logger.info("Compiling Metrics")
        time_end=time.time()
        kappa = metrics.cohen_kappa_score(ytest,pred_)
        mcc = metrics.matthews_corrcoef(ytest,pred_)
        training_time=time_end-time_start
        score_acc = np.append(score_acc,sca)
        score_auc = np.append(score_auc,sc)
        score_recall = np.append(score_recall,recall)
        score_precision = np.append(score_precision,precision)
        score_f1 =np.append(score_f1,f1)
        score_kappa =np.append(score_kappa,kappa)
        score_mcc =np.append(score_mcc,mcc)
        score_training_time =np.append(score_training_time,training_time)
    
    
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa], 'MCC':[mcc]}).round(round)
        master_display = pd.concat([master_display, fold_results],ignore_index=True)
        fold_results = []
        
        '''
        TIME CALCULATION SUB-SECTION STARTS HERE
        '''
        t1 = time.time()
        
        tt = (t1 - t0) * (fold-fold_num) / 60
        tt = np.around(tt, 2)
        
        if tt < 1:
            tt = str(np.around((tt * 60), 2))
            ETC = tt + ' Seconds Remaining'
                
        else:
            tt = str (tt)
            ETC = tt + ' Minutes Remaining'
            
        fold_num += 1
        
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[2,1:] = ETC
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        '''
        TIME CALCULATION ENDS HERE
        '''
        
        if verbose:
            if html_param:
                update_display(master_display, display_id = display_id)
            
        
        '''
        
        Update_display() ends here
        
        '''
    logger.info("Calculating mean and std")
    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    mean_mcc=np.mean(score_mcc)
    mean_training_time=np.sum(score_training_time)
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    std_mcc=np.std(score_mcc)
    std_training_time=np.std(score_training_time)
    
    avgs_acc = np.append(avgs_acc, mean_acc)
    avgs_acc = np.append(avgs_acc, std_acc) 
    avgs_auc = np.append(avgs_auc, mean_auc)
    avgs_auc = np.append(avgs_auc, std_auc)
    avgs_recall = np.append(avgs_recall, mean_recall)
    avgs_recall = np.append(avgs_recall, std_recall)
    avgs_precision = np.append(avgs_precision, mean_precision)
    avgs_precision = np.append(avgs_precision, std_precision)
    avgs_f1 = np.append(avgs_f1, mean_f1)
    avgs_f1 = np.append(avgs_f1, std_f1)
    avgs_kappa = np.append(avgs_kappa, mean_kappa)
    avgs_kappa = np.append(avgs_kappa, std_kappa)
    
    avgs_mcc = np.append(avgs_mcc, mean_mcc)
    avgs_mcc = np.append(avgs_mcc, std_mcc)
    avgs_training_time = np.append(avgs_training_time, mean_training_time)
    avgs_training_time = np.append(avgs_training_time, std_training_time)
    
    progress.value += 1
    
    logger.info("Creating metrics dataframe")
    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa, 'MCC' : score_mcc})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa, 'MCC' : avgs_mcc},index=['Mean', 'SD'])
    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    
    # yellow the mean
    model_results=model_results.style.apply(lambda x: ['background: yellow' if (x.name == 'Mean') else '' for i in x], axis=1)
    model_results = model_results.set_precision(round)

    progress.value += 1
    
    #refitting the model on complete X_train, y_train
    monitor.iloc[1,1:] = 'Finalizing Model'
    monitor.iloc[2,1:] = 'Almost Finished'
    
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    model_fit_start = time.time()
    logger.info("Finalizing model")
    model.fit(data_X, data_y)
    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
    
    progress.value += 1
    
    #storing results in create_model_container
    logger.info("Uploading results into container")
    create_model_container.append(model_results.data)
    display_container.append(model_results.data)

    #storing results in master_model_container
    logger.info("Uploading model into container")
    master_model_container.append(model)

    '''
    When choose_better sets to True. optimize metric in scoregrid is
    compared with base model created using create_model so that stack_models
    functions return the model with better score only. This will ensure 
    model performance is atleast equivalent to what is seen in compare_models 
    '''
    
    scorer = []

    blend_model_results = create_model_container[-1][compare_dimension][-2:][0]
    
    scorer.append(blend_model_results)

    if choose_better and all_flag is False:
        logger.info("choose_better activated")
        if verbose:
            if html_param:
                monitor.iloc[1,1:] = 'Compiling Final Results'
                monitor.iloc[2,1:] = 'Almost Finished'
                update_display(monitor, display_id = 'monitor')

        base_models_ = []
        logger.info("SubProcess create_model() called ==================================")
        for i in estimator_list:
            m = create_model(i,verbose=False, system=False)
            s = create_model_container[-1][compare_dimension][-2:][0]
            scorer.append(s)
            base_models_.append(m)

            #re-instate display_constainer state 
            display_container.pop(-1)
            
        logger.info("SubProcess create_model() called ==================================")
        logger.info("choose_better completed")

    index_scorer = scorer.index(max(scorer))

    if index_scorer == 0:
        model = model
    else:
        model = base_models_[index_scorer-1]

    #end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    if logging_param:
        
        logger.info("Creating MLFlow logs")

        #Creating Logs message monitor
        monitor.iloc[1,1:] = 'Creating Logs'
        monitor.iloc[2,1:] = 'Almost Finished'    
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        import mlflow
        from pathlib import Path
        import os

        with mlflow.start_run(run_name='Voting Classifier') as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            mlflow.log_metrics({"Accuracy": avgs_acc[0], "AUC": avgs_auc[0], "Recall": avgs_recall[0], "Precision" : avgs_precision[0],
                                "F1": avgs_f1[0], "Kappa": avgs_kappa[0], "MCC": avgs_mcc[0]})
            
            # Generate hold-out predictions and save as html
            holdout = predict_model(model, verbose=False)
            holdout_score = pull()
            del(holdout)
            display_container.pop(-1)
            holdout_score.to_html('Holdout.html', col_space=65, justify='left')
            mlflow.log_artifact('Holdout.html')
            os.remove('Holdout.html')

            #set tag of compare_models
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

                logger.info("SubProcess plot_model() called ==================================")

                try:
                    plot_model(model, plot = 'confusion_matrix', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Confusion Matrix.png')
                    os.remove("Confusion Matrix.png")
                except:
                    pass

                logger.info("SubProcess plot_model() end ==================================")

            # Log the CV results as model_results.html artifact
            model_results.data.to_html('Results.html', col_space=65, justify='left')
            mlflow.log_artifact('Results.html')
            os.remove('Results.html')

            # Log model and transformation pipeline
            from copy import deepcopy

            # get default conda env
            from mlflow.sklearn import get_default_conda_env
            default_conda_env = get_default_conda_env()
            default_conda_env['name'] = str(exp_name_log) + '-env'
            default_conda_env.get('dependencies').pop(-3)
            dependencies = default_conda_env.get('dependencies')[-1]
            from pycaret.utils import __version__
            dep = 'pycaret==' + str(__version__())
            dependencies['pip'] = [dep]
            
            # define model signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(data_before_preprocess.drop([target_param], axis=1))
            input_example = data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(['trained model', model])
            mlflow.sklearn.log_model(prep_pipe_temp, "model", conda_env = default_conda_env, signature = signature, input_example = input_example)
            del(prep_pipe_temp)

    if verbose:
        clear_output()
        if html_param:
            display(model_results)
        else:
            print(model_results.data)
    
    logger.info("create_model_container: " + str(len(create_model_container)))
    logger.info("master_model_container: " + str(len(master_model_container)))
    logger.info("display_container: " + str(len(display_container)))

    logger.info(str(model))
    logger.info("blend_models() succesfully completed......................................")

    return model

def stack_models(estimator_list, 
                 meta_model = None, 
                 fold = 10,
                 round = 4, 
                 method = 'auto', 
                 restack = True, 
                 choose_better = False, #added in pycaret==2.0.0
                 optimize = 'Accuracy', #added in pycaret==2.0.0
                 verbose = True):
    
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
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing stack_models()")
    logger.info("""stack_models(estimator_list={}, meta_model={}, fold={}, round={}, method={}, restack={}, choose_better={}, optimize={}, verbose={})""".\
        format(str(estimator_list), str(meta_model), str(fold), str(round), str(method), str(restack), str(choose_better), str(optimize), str(verbose)))

    logger.info("Checking exceptions")

    #exception checking   
    import sys
    
    #run_time
    import datetime, time
    runtime_start = time.time()

    #checking error for estimator_list
    for i in estimator_list:
        if 'sklearn' not in str(type(i)) and 'CatBoostClassifier' not in str(type(i)):
            sys.exit("(Value Error): estimator_list parameter only trained model object")
            
    #checking meta model
    if meta_model is not None:
        if 'sklearn' not in str(type(meta_model)) and 'CatBoostClassifier' not in str(type(meta_model)):
            sys.exit("(Value Error): estimator_list parameter only accepts trained model object")
            
    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    
    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')
 
    #checking method parameter
    available_method = ['auto', 'predict_proba', 'decision_function', 'predict']
    if method not in available_method:
        sys.exit("(Value Error): Method parameter not acceptable. It only accepts 'auto', 'predict_proba', 'decision_function', 'predict'.")
    
    #checking restack parameter
    if type(restack) is not bool:
        sys.exit('(Type Error): Restack parameter can only take argument as True or False.')    
        
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.') 
        
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    logger.info("Preloading libraries")
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    from copy import deepcopy
    from sklearn.base import clone
    from sklearn.ensemble import StackingClassifier
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    logger.info("Defining meta model")
    #Defining meta model.
    if meta_model == None:
        from sklearn.linear_model import LogisticRegression
        meta_model = LogisticRegression()
    else:
        meta_model = clone(meta_model)
        
    clear_output()

    if optimize == 'Accuracy':
        compare_dimension = 'Accuracy' 
    elif optimize == 'AUC':
        compare_dimension = 'AUC' 
    elif optimize == 'Recall':
        compare_dimension = 'Recall'
    elif optimize == 'Precision':
        compare_dimension = 'Prec.'
    elif optimize == 'F1':
        compare_dimension = 'F1' 
    elif optimize == 'Kappa':
        compare_dimension = 'Kappa'
    elif optimize == 'MCC':
        compare_dimension = 'MCC' 

    logger.info("Preparing display monitor")
    #progress bar
    max_progress = fold + 4
    progress = ipw.IntProgress(value=0, min=0, max=max_progress, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa', 'MCC'])
    if verbose:
        if html_param:
            display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    if verbose:
        if html_param:
            display(monitor, display_id = 'monitor')
    
    if verbose:
        if html_param:
            display_ = display(master_display, display_id=True)
            display_id = display_.display_id
    
    logger.info("Importing libraries")
    #dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_predict
    
    logger.info("Defining folds")
    #cross validation setup starts here
    kf = StratifiedKFold(fold, random_state=seed, shuffle=folds_shuffle_param)

    logger.info("Declaring metric variables")
    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    score_mcc =np.empty((0,0))
    score_training_time =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    avgs_mcc =np.empty((0,0))
    avgs_training_time =np.empty((0,0))

    progress.value += 1

    logger.info("Copying training dataset")
    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()
    
    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
    
    logger.info("Getting model names")
    #defining model_library model names
    model_names = np.zeros(0)
    for item in estimator_list:
        model_names = np.append(model_names, str(item).split("(")[0])
     
    model_names_fixed = []
    
    for i in model_names:
        if 'CatBoostClassifier' in i:
            a = 'CatBoostClassifier'
            model_names_fixed.append(a)
        else:
            model_names_fixed.append(i)
            
    model_names = model_names_fixed
    
    model_names_fixed = []

    counter = 0
    for i in model_names:
        s = str(i) + '_' + str(counter)
        model_names_fixed.append(s)
        counter += 1

    logger.info("Compiling estimator_list parameter")

    counter = 0
    
    estimator_list_tuples = []
    
    for i in estimator_list:
        estimator_list_tuples.append(tuple([model_names_fixed[counter], estimator_list[counter]]))
        counter += 1

    logger.info("Creating StackingClassifier()")

    model = StackingClassifier(estimators = estimator_list_tuples, final_estimator = meta_model, cv = fold,\
            stack_method = method, n_jobs = n_jobs_param, passthrough = restack)


    model_fit_start = time.time()

    fold_num = 1
    
    for train_i , test_i in kf.split(data_X,data_y):

        logger.info("Initializing Fold " + str(fold_num))
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
    
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
        time_start=time.time()

        if fix_imbalance_param:
            
            logger.info("Initializing SMOTE")

            if fix_imbalance_method_param is None:
                import six
                import sys
                sys.modules['sklearn.externals.six'] = six
                from imblearn.over_sampling import SMOTE
                resampler = SMOTE(random_state=seed)
            else:
                resampler = fix_imbalance_method_param

            Xtrain,ytrain = resampler.fit_sample(Xtrain, ytrain)
            logger.info("Resampling completed")

        logger.info("Fitting Model")
        model.fit(Xtrain,ytrain)
        logger.info("Evaluating Metrics")

        try:
            pred_prob = model.predict_proba(Xtest)
            pred_prob = pred_prob[:,1]
        except:
            pass
        pred_ = model.predict(Xtest)
        sca = metrics.accuracy_score(ytest,pred_)
        
        if y.value_counts().count() > 2:
            sc = 0
            recall = metrics.recall_score(ytest,pred_, average='macro')                
            precision = metrics.precision_score(ytest,pred_, average = 'weighted')
            f1 = metrics.f1_score(ytest,pred_, average='weighted')
            
        else:
            try:
                sc = metrics.roc_auc_score(ytest,pred_prob)
            except:
                sc = 0
            recall = metrics.recall_score(ytest,pred_)                
            precision = metrics.precision_score(ytest,pred_)
            f1 = metrics.f1_score(ytest,pred_)
        
        logger.info("Compiling Metrics")        
        time_end=time.time()
        kappa = metrics.cohen_kappa_score(ytest,pred_)
        mcc = metrics.matthews_corrcoef(ytest,pred_)
        training_time=time_end-time_start
        score_acc = np.append(score_acc,sca)
        score_auc = np.append(score_auc,sc)
        score_recall = np.append(score_recall,recall)
        score_precision = np.append(score_precision,precision)
        score_f1 =np.append(score_f1,f1)
        score_kappa =np.append(score_kappa,kappa)
        score_mcc=np.append(score_mcc,mcc)
        score_training_time = np.append(score_training_time,training_time)
   
        progress.value += 1
                
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa], 'MCC':[mcc]}).round(round)
        master_display = pd.concat([master_display, fold_results],ignore_index=True)
        fold_results = []
        
        '''
        TIME CALCULATION SUB-SECTION STARTS HERE
        '''
        t1 = time.time()
        
        tt = (t1 - t0) * (fold-fold_num) / 60
        tt = np.around(tt, 2)
        
        if tt < 1:
            tt = str(np.around((tt * 60), 2))
            ETC = tt + ' Seconds Remaining'
                
        else:
            tt = str (tt)
            ETC = tt + ' Minutes Remaining'
            
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[2,1:] = ETC
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
            
        fold_num += 1
        
        '''
        TIME CALCULATION ENDS HERE
        '''
        
        if verbose:
            if html_param:
                update_display(master_display, display_id = display_id)
            
        
        '''
        
        Update_display() ends here
        
        '''
    
    logger.info("Calculating mean and std")

    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    mean_mcc=np.mean(score_mcc)
    mean_training_time=np.sum(score_training_time) #changed it to sum from mean 
    
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    std_mcc=np.std(score_mcc)
    std_training_time=np.std(score_training_time)
    
    avgs_acc = np.append(avgs_acc, mean_acc)
    avgs_acc = np.append(avgs_acc, std_acc) 
    avgs_auc = np.append(avgs_auc, mean_auc)
    avgs_auc = np.append(avgs_auc, std_auc)
    avgs_recall = np.append(avgs_recall, mean_recall)
    avgs_recall = np.append(avgs_recall, std_recall)
    avgs_precision = np.append(avgs_precision, mean_precision)
    avgs_precision = np.append(avgs_precision, std_precision)
    avgs_f1 = np.append(avgs_f1, mean_f1)
    avgs_f1 = np.append(avgs_f1, std_f1)
    avgs_kappa = np.append(avgs_kappa, mean_kappa)
    avgs_kappa = np.append(avgs_kappa, std_kappa)
    avgs_mcc = np.append(avgs_mcc, mean_mcc)
    avgs_mcc = np.append(avgs_mcc, std_mcc)
    
    avgs_training_time = np.append(avgs_training_time, mean_training_time)
    avgs_training_time = np.append(avgs_training_time, std_training_time)
    
    progress.value += 1
    
    logger.info("Creating metrics dataframe")

    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa, 'MCC': score_mcc})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa, 'MCC': avgs_mcc},index=['Mean', 'SD'])

    
    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    
    # yellow the mean
    model_results=model_results.style.apply(lambda x: ['background: yellow' if (x.name == 'Mean') else '' for i in x], axis=1)
    model_results = model_results.set_precision(round)

    #refitting the model on complete X_train, y_train
    monitor.iloc[1,1:] = 'Finalizing Model'
    monitor.iloc[2,1:] = 'Almost Finished'    
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    model_fit_start = time.time()
    logger.info("Finalizing model")
    model.fit(data_X, data_y)
    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
    
    progress.value += 1

    #end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    #storing results in create_model_container
    logger.info("Uploading results into container")
    create_model_container.append(model_results.data)
    display_container.append(model_results.data)

    #storing results in master_model_container
    logger.info("Uploading model into container")
    master_model_container.append(model)    

    progress.value += 1

    '''
    When choose_better sets to True. optimize metric in scoregrid is
    compared with base model created using create_model so that stack_models
    functions return the model with better score only. This will ensure 
    model performance is atleast equivalent to what is seen in compare_models 
    '''

    scorer = []

    stack_model_results = create_model_container[-1][compare_dimension][-2:][0]
    
    scorer.append(stack_model_results)

    if choose_better:
        logger.info("choose_better activated")

        if verbose:
            if html_param:
                monitor.iloc[1,1:] = 'Compiling Final Results'
                monitor.iloc[2,1:] = 'Almost Finished'
                update_display(monitor, display_id = 'monitor')

        base_models_ = []
        logger.info("SubProcess create_model() called ==================================")
        for i in estimator_list:
            m = create_model(i,verbose=False, system=False)
            s = create_model_container[-1][compare_dimension][-2:][0]
            scorer.append(s)
            base_models_.append(m)

            #re-instate display_constainer state 
            display_container.pop(-1)

        meta_model_clone = clone(meta_model)
        mm = create_model(meta_model_clone, verbose=False, system=False)
        base_models_.append(mm)
        s = create_model_container[-1][compare_dimension][-2:][0]
        scorer.append(s)

        #re-instate display_constainer state 
        display_container.pop(-1)
        logger.info("SubProcess create_model() end ==================================")
        logger.info("choose_better completed")

    #returning better model
    index_scorer = scorer.index(max(scorer))
    
    if index_scorer == 0:
        model = model
    else:
        model = base_models_[index_scorer-1]

    #end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    if logging_param:
        
        logger.info("Creating MLFlow logs")

        import mlflow
        from pathlib import Path
        import os

        #Creating Logs message monitor
        monitor.iloc[1,1:] = 'Creating Logs'
        monitor.iloc[2,1:] = 'Almost Finished'    
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        with mlflow.start_run(run_name='Stacking Classifier') as run:   

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
            
            mlflow.log_metrics({"Accuracy": avgs_acc[0], "AUC": avgs_auc[0], "Recall": avgs_recall[0], "Precision" : avgs_precision[0],
                                "F1": avgs_f1[0], "Kappa": avgs_kappa[0], "MCC": avgs_mcc[0]})
            
            #set tag of stack_models
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
            model_results.data.to_html('Results.html', col_space=65, justify='left')
            mlflow.log_artifact('Results.html')
            os.remove('Results.html')

            # Generate hold-out predictions and save as html
            holdout = predict_model(model, verbose=False)
            holdout_score = pull()
            del(holdout)
            display_container.pop(-1)
            holdout_score.to_html('Holdout.html', col_space=65, justify='left')
            mlflow.log_artifact('Holdout.html')
            os.remove('Holdout.html')

            # Log AUC and Confusion Matrix plot
            if log_plots_param:
                
                logger.info("SubProcess plot_model() called ==================================")

                try:
                    plot_model(model, plot = 'auc', verbose=False, save=True, system=False)
                    mlflow.log_artifact('AUC.png')
                    os.remove("AUC.png")
                except:
                    pass

                try:
                    plot_model(model, plot = 'confusion_matrix', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Confusion Matrix.png')
                    os.remove("Confusion Matrix.png")
                except:
                    pass

                try:
                    plot_model(model, plot = 'feature', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Feature Importance.png')
                    os.remove("Feature Importance.png")
                except:
                    pass
                    
                logger.info("SubProcess plot_model() end ==================================")

            # Log model and transformation pipeline
            from copy import deepcopy

            # get default conda env
            from mlflow.sklearn import get_default_conda_env
            default_conda_env = get_default_conda_env()
            default_conda_env['name'] = str(exp_name_log) + '-env'
            default_conda_env.get('dependencies').pop(-3)
            dependencies = default_conda_env.get('dependencies')[-1]
            from pycaret.utils import __version__
            dep = 'pycaret==' + str(__version__())
            dependencies['pip'] = [dep]
            
            # define model signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(data_before_preprocess.drop([target_param], axis=1))
            input_example = data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(['trained model', model])
            mlflow.sklearn.log_model(prep_pipe_temp, "model", conda_env = default_conda_env, signature = signature, input_example = input_example)
            del(prep_pipe_temp)

    if verbose:
        clear_output()
        if html_param:
            display(model_results)
        else:
            print(model_results.data)

    logger.info("create_model_container: " + str(len(create_model_container)))
    logger.info("master_model_container: " + str(len(master_model_container)))
    logger.info("display_container: " + str(len(display_container)))

    logger.info(str(model))
    logger.info("stack_models() succesfully completed......................................")

    return model

def plot_model(estimator, 
               plot = 'auc',
               scale = 1, #added in pycaret 2.1.0
               save = False, #added in pycaret 2.0.0
               verbose = True, #added in pycaret 2.0.0
               system = True): #added in pycaret 2.0.0
    
    
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
    
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #exception checking   
    import sys
    
    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing plot_model()")
    logger.info("""plot_model(estimator={}, plot={}, save={}, verbose={}, system={})""".\
        format(str(estimator), str(plot), str(save), str(verbose), str(system)))

    logger.info("Checking exceptions")

    #checking plots (string)
    available_plots = ['auc', 'threshold', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'rfe', 'learning',
                       'manifold', 'calibration', 'vc', 'dimension', 'feature', 'parameter']
    
    if plot not in available_plots:
        sys.exit('(Value Error): Plot Not Available. Please see docstring for list of available Plots.')
    
    #multiclass plot exceptions:
    multiclass_not_available = ['calibration', 'threshold', 'manifold', 'rfe']
    if y.value_counts().count() > 2:
        if plot in multiclass_not_available:
            sys.exit('(Value Error): Plot Not Available for multiclass problems. Please see docstring for list of available Plots.')
        
    #exception for CatBoost
    if 'CatBoostClassifier' in str(type(estimator)):
        sys.exit('(Estimator Error): CatBoost estimator is not compatible with plot_model function, try using Catboost with interpret_model instead.')
        
    #checking for auc plot
    if not hasattr(estimator, 'predict_proba') and plot == 'auc':
        sys.exit('(Type Error): AUC plot not available for estimators with no predict_proba attribute.')
    
    #checking for auc plot
    if not hasattr(estimator, 'predict_proba') and plot == 'auc':
        sys.exit('(Type Error): AUC plot not available for estimators with no predict_proba attribute.')
    
    #checking for calibration plot
    if not hasattr(estimator, 'predict_proba') and plot == 'calibration':
        sys.exit('(Type Error): Calibration plot not available for estimators with no predict_proba attribute.')
     
    #checking for rfe
    if hasattr(estimator,'max_features') and plot == 'rfe' and estimator.max_features_ != X_train.shape[1]:
        sys.exit('(Type Error): RFE plot not available when max_features parameter is not set to None.')
        
    #checking for feature plot
    if not ( hasattr(estimator, 'coef_') or hasattr(estimator,'feature_importances_') ) and plot == 'feature':
        sys.exit('(Type Error): Feature Importance plot not available for estimators that doesnt support coef_ or feature_importances_ attribute.')
    
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    logger.info("Preloading libraries")
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    logger.info("Preparing display monitor")
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=5, step=1 , description='Processing: ')
    if verbose:
        if html_param:
            display(progress)
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    logger.info("Importing libraries")
    #general dependencies
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    progress.value += 1
    
    #defining estimator as model locally
    model = estimator
    
    progress.value += 1
    
    #plots used for logging (controlled through plots_log_param) 
    #AUC, #Confusion Matrix and #Feature Importance

    logger.info("plot type: " + str(plot)) 

    if plot == 'auc':

        from yellowbrick.classifier import ROCAUC
        progress.value += 1
        visualizer = ROCAUC(model)
        visualizer.fig.set_dpi(visualizer.fig.dpi * scale)
        logger.info("Fitting Model")
        visualizer.fit(X_train, y_train)
        progress.value += 1
        logger.info("Scoring test/hold-out set")
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'AUC.png' in current active directory")
            if system:
                visualizer.show(outpath="AUC.png")
            else:
                visualizer.show(outpath="AUC.png", clear_figure=True)
        else:
            visualizer.show()

        logger.info("Visual Rendered Successfully")
        
    elif plot == 'threshold':
        
        from yellowbrick.classifier import DiscriminationThreshold
        progress.value += 1
        visualizer = DiscriminationThreshold(model, random_state=seed)
        visualizer.fig.set_dpi(visualizer.fig.dpi * scale)
        logger.info("Fitting Model")
        visualizer.fit(X_train, y_train)
        progress.value += 1
        logger.info("Scoring test/hold-out set")
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Threshold Curve.png' in current active directory")
            if system:
                visualizer.show(outpath="Threshold Curve.png")
            else:
                visualizer.show(outpath="Threshold Curve.png", clear_figure=True)
        else:
            visualizer.show()

        logger.info("Visual Rendered Successfully")

    elif plot == 'pr':
        
        from yellowbrick.classifier import PrecisionRecallCurve
        progress.value += 1
        visualizer = PrecisionRecallCurve(model, random_state=seed)
        visualizer.fig.set_dpi(visualizer.fig.dpi * scale)
        logger.info("Fitting Model")
        visualizer.fit(X_train, y_train)
        progress.value += 1
        logger.info("Scoring test/hold-out set")
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Precision Recall.png' in current active directory")
            if system:
                visualizer.show(outpath="Precision Recall.png")
            else:
                visualizer.show(outpath="Precision Recall.png", clear_figure=True)
        else:
            visualizer.show()

        logger.info("Visual Rendered Successfully")

    elif plot == 'confusion_matrix':
        
        from yellowbrick.classifier import ConfusionMatrix
        progress.value += 1
        visualizer = ConfusionMatrix(model, random_state=seed, fontsize = 15, cmap="Greens")
        visualizer.fig.set_dpi(visualizer.fig.dpi * scale)
        logger.info("Fitting Model")
        visualizer.fit(X_train, y_train)
        progress.value += 1
        logger.info("Scoring test/hold-out set")
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Confusion Matrix.png' in current active directory")
            if system:
                visualizer.show(outpath="Confusion Matrix.png")
            else:
                visualizer.show(outpath="Confusion Matrix.png", clear_figure=True)
        else:
            visualizer.show()
            
        logger.info("Visual Rendered Successfully")

    elif plot == 'error':
        
        from yellowbrick.classifier import ClassPredictionError
        progress.value += 1
        visualizer = ClassPredictionError(model, random_state=seed)
        visualizer.fig.set_dpi(visualizer.fig.dpi * scale)
        logger.info("Fitting Model")
        visualizer.fit(X_train, y_train)
        progress.value += 1
        logger.info("Scoring test/hold-out set")
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Class Prediction Error.png' in current active directory")
            if system:
                visualizer.show(outpath="Class Prediction Error.png")
            else:
                visualizer.show(outpath="Class Prediction Error.png", clear_figure=True)
        else:
            visualizer.show()

        logger.info("Visual Rendered Successfully")

    elif plot == 'class_report':
        
        from yellowbrick.classifier import ClassificationReport
        progress.value += 1
        visualizer = ClassificationReport(model, random_state=seed, support=True)
        visualizer.fig.set_dpi(visualizer.fig.dpi * scale)
        logger.info("Fitting Model")
        visualizer.fit(X_train, y_train)
        progress.value += 1
        logger.info("Scoring test/hold-out set")
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Classification Report.png' in current active directory")
            if system:
                visualizer.show(outpath="Classification Report.png")
            else:
                visualizer.show(outpath="Classification Report.png", clear_figure=True)
        else:
            visualizer.show()

        logger.info("Visual Rendered Successfully")
        
    elif plot == 'boundary':
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from yellowbrick.contrib.classifier import DecisionViz        
        from copy import deepcopy
        model2 = deepcopy(estimator)
        
        progress.value += 1
        
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()
        X_train_transformed = X_train_transformed.select_dtypes(include='float64')
        X_test_transformed = X_test_transformed.select_dtypes(include='float64')
        logger.info("Fitting StandardScaler()")
        X_train_transformed = StandardScaler().fit_transform(X_train_transformed)
        X_test_transformed = StandardScaler().fit_transform(X_test_transformed)
        pca = PCA(n_components=2, random_state = seed)
        logger.info("Fitting PCA()")
        X_train_transformed = pca.fit_transform(X_train_transformed)
        X_test_transformed = pca.fit_transform(X_test_transformed)
        
        progress.value += 1
        
        y_train_transformed = y_train.copy()
        y_test_transformed = y_test.copy()
        y_train_transformed = np.array(y_train_transformed)
        y_test_transformed = np.array(y_test_transformed)
        
        viz_ = DecisionViz(model2)
        viz_.fig.set_dpi(viz_.fig.dpi * scale)
        logger.info("Fitting Model")
        viz_.fit(X_train_transformed, y_train_transformed, features=['Feature One', 'Feature Two'], classes=['A', 'B'])
        viz_.draw(X_test_transformed, y_test_transformed)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Decision Boundary.png' in current active directory")
            if system:
                viz_.show(outpath="Decision Boundary.png")
            else:
                viz_.show(outpath="Decision Boundary.png", clear_figure=True)
        else:
            viz_.show()

        logger.info("Visual Rendered Successfully")

    elif plot == 'rfe':
        
        from yellowbrick.model_selection import RFECV 
        progress.value += 1
        visualizer = RFECV(model, cv=10)
        visualizer.fig.set_dpi(visualizer.fig.dpi * scale)
        progress.value += 1
        logger.info("Fitting Model")
        visualizer.fit(X_train, y_train)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Recursive Feature Selection.png' in current active directory")
            if system:
                visualizer.show(outpath="Recursive Feature Selection.png")
            else:
                visualizer.show(outpath="Recursive Feature Selection.png", clear_figure=True)
        else:
            visualizer.show()

        logger.info("Visual Rendered Successfully")
           
    elif plot == 'learning':
        
        from yellowbrick.model_selection import LearningCurve
        progress.value += 1
        sizes = np.linspace(0.3, 1.0, 10)  
        visualizer = LearningCurve(model, cv=10, train_sizes=sizes, n_jobs=n_jobs_param, random_state=seed)
        visualizer.fig.set_dpi(visualizer.fig.dpi * scale)
        progress.value += 1
        logger.info("Fitting Model")
        visualizer.fit(X_train, y_train)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Learning Curve.png' in current active directory")
            if system:
                visualizer.show(outpath="Learning Curve.png")
            else:
                visualizer.show(outpath="Learning Curve.png", clear_figure=True)
        else:
            visualizer.show()

        logger.info("Visual Rendered Successfully")

    elif plot == 'manifold':
        
        from yellowbrick.features import Manifold
        
        progress.value += 1
        X_train_transformed = X_train.select_dtypes(include='float64') 
        visualizer = Manifold(manifold='tsne', random_state = seed)
        visualizer.fig.set_dpi(visualizer.fig.dpi * scale)
        progress.value += 1
        logger.info("Fitting Model")
        visualizer.fit_transform(X_train_transformed, y_train)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Manifold Plot.png' in current active directory")
            if system:
                visualizer.show(outpath="Manifold Plot.png")
            else:
                visualizer.show(outpath="Manifold Plot.png", clear_figure=True)
        else:
            visualizer.show()

        logger.info("Visual Rendered Successfully")

    elif plot == 'calibration':      
                
        from sklearn.calibration import calibration_curve
        
        model_name = str(model).split("(")[0]
        
        plt.figure(figsize=(7, 6), dpi=100*scale)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        progress.value += 1
        logger.info("Scoring test/hold-out set")
        prob_pos = model.predict_proba(X_test)[:, 1]
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
        progress.value += 1
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s" % (model_name, ))
    
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([0, 1])
        ax1.set_xlim([0, 1])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')
        ax1.set_facecolor('white')
        ax1.grid(b=True, color='grey', linewidth=0.5, linestyle = '-')
        plt.tight_layout()
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Calibration Plot.png' in current active directory")
            if system:
                plt.savefig("Calibration Plot.png")
            else:
                plt.show()
        else:
            plt.show() 
        
        logger.info("Visual Rendered Successfully")

    elif plot == 'vc':
        
        model_name = str(model).split("(")[0]
        
        logger.info("Determining param_name")

        #SGD Classifier
        if model_name == 'SGDClassifier':
            param_name='l1_ratio'
            param_range = np.arange(0,1, 0.01)
            
        elif model_name == 'LinearDiscriminantAnalysis':
            sys.exit('(Value Error): Shrinkage Parameter not supported in Validation Curve Plot.')
        
        #tree based models
        elif hasattr(model, 'max_depth'):
            param_name='max_depth'
            param_range = np.arange(1,11)
        
        #knn
        elif hasattr(model, 'n_neighbors'):
            param_name='n_neighbors'
            param_range = np.arange(1,11)            
            
        #MLP / Ridge
        elif hasattr(model, 'alpha'):
            param_name='alpha'
            param_range = np.arange(0,1,0.1)     
            
        #Logistic Regression
        elif hasattr(model, 'C'):
            param_name='C'
            param_range = np.arange(1,11)
            
        #Bagging / Boosting 
        elif hasattr(model, 'n_estimators'):
            param_name='n_estimators'
            param_range = np.arange(1,100,10)   
            
        #Bagging / Boosting / gbc / ada / 
        elif hasattr(model, 'n_estimators'):
            param_name='n_estimators'
            param_range = np.arange(1,100,10)   
            
        #Naive Bayes
        elif hasattr(model, 'var_smoothing'):
            param_name='var_smoothing'
            param_range = np.arange(0.1, 1, 0.01)
            
        #QDA
        elif hasattr(model, 'reg_param'):
            param_name='reg_param'
            param_range = np.arange(0,1,0.1)
            
        #GPC
        elif hasattr(model, 'max_iter_predict'):
            param_name='max_iter_predict'
            param_range = np.arange(100,1000,100)        
        
        else:
            clear_output()
            sys.exit('(Type Error): Plot not supported for this estimator. Try different estimator.')
        
        logger.info("param_name: " + str(param_name))
            
        progress.value += 1
            
        from yellowbrick.model_selection import ValidationCurve
        viz = ValidationCurve(model, param_name=param_name, param_range=param_range,cv=10, 
                              random_state=seed)
        viz.fig.set_dpi(viz.fig.dpi * scale)
        logger.info("Fitting Model")
        viz.fit(X_train, y_train)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Validation Curve.png' in current active directory")
            if system:
                viz.show(outpath="Validation Curve.png")
            else:
                viz.show(outpath="Validation Curve.png", clear_figure=True)
        else:
            viz.show()
        
        logger.info("Visual Rendered Successfully")
        
    elif plot == 'dimension':
    
        from yellowbrick.features import RadViz
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        progress.value += 1
        X_train_transformed = X_train.select_dtypes(include='float64') 
        logger.info("Fitting StandardScaler()")
        X_train_transformed = StandardScaler().fit_transform(X_train_transformed)
        y_train_transformed = np.array(y_train)
        
        features=min(round(len(X_train.columns) * 0.3,0),5)
        features = int(features)
        
        pca = PCA(n_components=features, random_state=seed)
        logger.info("Fitting PCA()")
        X_train_transformed = pca.fit_transform(X_train_transformed)
        progress.value += 1
        classes = y_train.unique().tolist()
        visualizer = RadViz(classes=classes, alpha=0.25)
        visualizer.fig.set_dpi(visualizer.fig.dpi * scale)
        logger.info("Fitting Model")
        visualizer.fit(X_train_transformed, y_train_transformed)     
        visualizer.transform(X_train_transformed)
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Dimension Plot.png' in current active directory")
            if system:
                visualizer.show(outpath="Dimension Plot.png")
            else:
                visualizer.show(outpath="Dimension Plot.png", clear_figure=True)
        else:
            visualizer.show()

        logger.info("Visual Rendered Successfully")
        
    elif plot == 'feature':
        
        if hasattr(estimator,'coef_'):
            variables = abs(model.coef_[0])
        else:
            logger.warning("No coef_ found. Trying feature_importances_")
            variables = abs(model.feature_importances_)
        col_names = np.array(X_train.columns)
        coef_df = pd.DataFrame({'Variable': X_train.columns, 'Value': variables})
        sorted_df = coef_df.sort_values(by='Value')
        sorted_df = sorted_df.sort_values(by='Value', ascending=False)
        sorted_df = sorted_df.head(10)
        sorted_df = sorted_df.sort_values(by='Value')
        my_range=range(1,len(sorted_df.index)+1)
        progress.value += 1
        plt.figure(figsize=(8,5), dpi=100*scale)
        plt.hlines(y=my_range, xmin=0, xmax=sorted_df['Value'], color='skyblue')
        plt.plot(sorted_df['Value'], my_range, "o")
        progress.value += 1
        plt.yticks(my_range, sorted_df['Variable'])
        plt.title("Feature Importance Plot")
        plt.xlabel('Variable Importance')
        plt.ylabel('Features')
        progress.value += 1
        clear_output()
        if save:
            logger.info("Saving 'Feature Importance.png' in current active directory")
            if system:
                plt.savefig("Feature Importance.png")
            else:
                plt.savefig("Feature Importance.png")
                plt.close()
        else:
            plt.show() 
        
        logger.info("Visual Rendered Successfully")
    
    elif plot == 'parameter':
        
        clear_output()
        param_df = pd.DataFrame.from_dict(estimator.get_params(estimator), orient='index', columns=['Parameters'])
        display(param_df)
        logger.info("Visual Rendered Successfully")

    logger.info("plot_model() succesfully completed......................................")

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
                            options=[('Hyperparameters', 'parameter'),
                                     ('AUC', 'auc'), 
                                     ('Confusion Matrix', 'confusion_matrix'), 
                                     ('Threshold', 'threshold'),
                                     ('Precision Recall', 'pr'),
                                     ('Error', 'error'),
                                     ('Class Report', 'class_report'),
                                     ('Feature Selection', 'rfe'),
                                     ('Learning Curve', 'learning'),
                                     ('Manifold Learning', 'manifold'),
                                     ('Calibration Curve', 'calibration'),
                                     ('Validation Curve', 'vc'),
                                     ('Dimensions', 'dimension'),
                                     ('Feature Importance', 'feature'),
                                     ('Decision Boundary', 'boundary')
                                    ],

                            description='Plot Type:',

                            disabled=False,

                            button_style='', # 'success', 'info', 'warning', 'danger' or ''

                            icons=['']
    )
    
  
    d = interact(plot_model, estimator = fixed(estimator), plot = a, save = fixed(False), verbose = fixed(True), system = fixed(True))

def interpret_model(estimator,
                   plot = 'summary',
                   feature = None, 
                   observation = None,
                   **kwargs): #added in pycaret==2.1
    
    
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
    
    
    
    '''
    Error Checking starts here
    
    '''
    
    import sys
    
    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing interpret_model()")
    logger.info("""interpret_model(estimator={}, plot={}, feature={}, observation={})""".\
        format(str(estimator), str(plot), str(feature), str(observation)))

    logger.info("Checking exceptions")

    #checking if shap available
    try:
        import shap
    except:
        logger.error("shap library not found. pip install shap to use interpret_model function.")
        sys.exit("shap library not found. pip install shap to use interpret_model function.")   

    #allowed models
    allowed_models = ['RandomForestClassifier',
                      'DecisionTreeClassifier',
                      'ExtraTreesClassifier',
                      'GradientBoostingClassifier',
                      'XGBClassifier',
                      'LGBMClassifier',
                      'CatBoostClassifier']
    
    model_name = str(estimator).split("(")[0]
    
    #Statement to find CatBoost and change name :
    if model_name.find("catboost.core.CatBoostClassifier") != -1:
        model_name = 'CatBoostClassifier'
    
    if model_name not in allowed_models:
        sys.exit('(Type Error): This function only supports tree based models for binary classification.')
        
    #plot type
    allowed_types = ['summary', 'correlation', 'reason']
    if plot not in allowed_types:
        sys.exit("(Value Error): type parameter only accepts 'summary', 'correlation' or 'reason'.")   
           
    
    '''
    Error Checking Ends here
    
    '''
        
    logger.info("Importing libraries")
    #general dependencies
    import numpy as np
    import pandas as pd
    import shap
    
    #storing estimator in model variable
    model = estimator

    #defining type of classifier
    type1 = ['RandomForestClassifier','DecisionTreeClassifier','ExtraTreesClassifier', 'LGBMClassifier']
    type2 = ['GradientBoostingClassifier', 'XGBClassifier', 'CatBoostClassifier']
    
    if plot == 'summary':

        logger.info("plot type: summary")
        
        if model_name in type1:
            
            logger.info("model type detected: type 1")
            logger.info("Creating TreeExplainer")
            explainer = shap.TreeExplainer(model)
            logger.info("Compiling shap values")
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, **kwargs)
            logger.info("Visual Rendered Successfully")
            
        elif model_name in type2:
            
            logger.info("model type detected: type 2")
            logger.info("Creating TreeExplainer")
            explainer = shap.TreeExplainer(model)
            logger.info("Compiling shap values")
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, **kwargs)
            logger.info("Visual Rendered Successfully")
                              
    elif plot == 'correlation':
        
        logger.info("plot type: correlation")

        if feature == None:
            
            logger.warning("No feature passed. Default value of feature used for correlation plot: " + str(X_test.columns[0]))
            dependence = X_test.columns[0]
            
        else:
            
            logger.warning("feature value passed. Feature used for correlation plot: " + str(X_test.columns[0]))
            dependence = feature
        
        if model_name in type1:
            logger.info("model type detected: type 1")
            logger.info("Creating TreeExplainer")    
            explainer = shap.TreeExplainer(model)
            logger.info("Compiling shap values")
            shap_values = explainer.shap_values(X_test)
            shap.dependence_plot(dependence, shap_values[1], X_test, **kwargs)
            logger.info("Visual Rendered Successfully")
        
        elif model_name in type2:
            logger.info("model type detected: type 2")
            logger.info("Creating TreeExplainer")  
            explainer = shap.TreeExplainer(model)
            logger.info("Compiling shap values")
            shap_values = explainer.shap_values(X_test) 
            shap.dependence_plot(dependence, shap_values, X_test, **kwargs)
            logger.info("Visual Rendered Successfully")
        
    elif plot == 'reason':
        
        logger.info("plot type: reason")

        if model_name in type1:
            logger.info("model type detected: type 1")

            if observation is None:
                logger.warning("Observation set to None. Model agnostic plot will be rendered.")
                logger.info("Creating TreeExplainer") 
                explainer = shap.TreeExplainer(model)
                logger.info("Compiling shap values")
                shap_values = explainer.shap_values(X_test)
                shap.initjs()
                logger.info("Visual Rendered Successfully")
                logger.info("interpret_model() succesfully completed......................................")
                return shap.force_plot(explainer.expected_value[1], shap_values[1], X_test, **kwargs)
            
            else: 
                
                if model_name == 'LGBMClassifier':
                    logger.info("model type detected: LGBMClassifier")

                    row_to_show = observation
                    data_for_prediction = X_test.iloc[row_to_show]
                    logger.info("Creating TreeExplainer") 
                    explainer = shap.TreeExplainer(model)
                    logger.info("Compiling shap values")  
                    shap_values = explainer.shap_values(X_test)
                    shap.initjs()
                    logger.info("Visual Rendered Successfully")
                    logger.info("interpret_model() succesfully completed......................................")
                    return shap.force_plot(explainer.expected_value[1], shap_values[0][row_to_show], data_for_prediction, **kwargs)    
                
                else:
                    logger.info("model type detected: Unknown")
                    row_to_show = observation
                    data_for_prediction = X_test.iloc[row_to_show]
                    logger.info("Creating TreeExplainer")  
                    explainer = shap.TreeExplainer(model)
                    logger.info("Compiling shap values")  
                    shap_values = explainer.shap_values(data_for_prediction)
                    shap.initjs()
                    logger.info("Visual Rendered Successfully")
                    logger.info("interpret_model() succesfully completed......................................")
                    return shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, **kwargs)        

            
        elif model_name in type2:
            logger.info("model type detected: type 2")

            if observation is None:
                logger.warning("Observation set to None. Model agnostic plot will be rendered.")
                logger.info("Creating TreeExplainer")  
                explainer = shap.TreeExplainer(model)
                logger.info("Compiling shap values")  
                shap_values = explainer.shap_values(X_test)
                shap.initjs()
                logger.info("Visual Rendered Successfully")
                logger.info("interpret_model() succesfully completed......................................")
                return shap.force_plot(explainer.expected_value, shap_values, X_test, **kwargs)  
                
            else:
                
                row_to_show = observation
                data_for_prediction = X_test.iloc[row_to_show]
                logger.info("Creating TreeExplainer") 
                explainer = shap.TreeExplainer(model)
                logger.info("Compiling shap values")  
                shap_values = explainer.shap_values(X_test)
                shap.initjs()
                logger.info("Visual Rendered Successfully")
                logger.info("interpret_model() succesfully completed......................................")
                return shap.force_plot(explainer.expected_value, shap_values[row_to_show,:], X_test.iloc[row_to_show,:], **kwargs)

    logger.info("interpret_model() succesfully completed......................................")

def calibrate_model(estimator,
                    method = 'sigmoid',
                    fold=10,
                    round=4,
                    verbose=True):
    
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


    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing calibrate_model()")
    logger.info("""calibrate_model(estimator={}, method={}, fold={}, round={}, verbose={})""".\
        format(str(estimator), str(method), str(fold), str(round), str(verbose)))

    logger.info("Checking exceptions")

    #exception checking   
    import sys

    #run_time
    import datetime, time
    runtime_start = time.time()

    #Statement to find CatBoost and change name
    model_name = str(estimator).split("(")[0]
    if model_name.find("catboost.core.CatBoostClassifier") != -1:
        model_name = 'CatBoostClassifier'

    #catboost not allowed
    not_allowed = ['CatBoostClassifier']
    if model_name in not_allowed:
        sys.exit('(Type Error): calibrate_model doesnt support CatBoost Classifier. Try different estimator.')
    
    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    
    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')
 
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.') 
        
    
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    logger.info("Preloading libraries")

    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display

    logger.info("Preparing display monitor")    
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+4, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa','MCC'])
    if verbose:
        if html_param:
            display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    if verbose:
        if html_param:
            display(monitor, display_id = 'monitor')
    
    if verbose:
        if html_param:
            display_ = display(master_display, display_id=True)
            display_id = display_.display_id
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    logger.info("Copying training dataset")
    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()
    
    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
    
    logger.info("Importing libraries")
    #general dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    
    progress.value += 1
    
    logger.info("Getting model name")

    def get_model_name(e):
        return str(e).split("(")[0]

    if len(estimator.classes_) > 2:

        if hasattr(estimator, 'voting'):
            mn = get_model_name(estimator)
        else:
            mn = get_model_name(estimator.estimator)

    else:
        if hasattr(estimator, 'voting'):
            mn = 'VotingClassifier'
        else:
            mn = get_model_name(estimator)

    if 'catboost' in mn:
        mn = 'CatBoostClassifier' 

    model_dict_logging = {'ExtraTreesClassifier' : 'Extra Trees Classifier',
                        'GradientBoostingClassifier' : 'Gradient Boosting Classifier', 
                        'RandomForestClassifier' : 'Random Forest Classifier',
                        'LGBMClassifier' : 'Light Gradient Boosting Machine',
                        'XGBClassifier' : 'Extreme Gradient Boosting',
                        'AdaBoostClassifier' : 'Ada Boost Classifier', 
                        'DecisionTreeClassifier' : 'Decision Tree Classifier', 
                        'RidgeClassifier' : 'Ridge Classifier',
                        'LogisticRegression' : 'Logistic Regression',
                        'KNeighborsClassifier' : 'K Neighbors Classifier',
                        'GaussianNB' : 'Naive Bayes',
                        'SGDClassifier' : 'SVM - Linear Kernel',
                        'SVC' : 'SVM - Radial Kernel',
                        'GaussianProcessClassifier' : 'Gaussian Process Classifier',
                        'MLPClassifier' : 'MLP Classifier',
                        'QuadraticDiscriminantAnalysis' : 'Quadratic Discriminant Analysis',
                        'LinearDiscriminantAnalysis' : 'Linear Discriminant Analysis',
                        'CatBoostClassifier' : 'CatBoost Classifier',
                        'BaggingClassifier' : 'Bagging Classifier',
                        'VotingClassifier' : 'Voting Classifier'}

    base_estimator_full_name = model_dict_logging.get(mn)

    logger.info("Base model : " + str(base_estimator_full_name))

    #cross validation setup starts here
    logger.info("Defining folds")
    kf = StratifiedKFold(fold, random_state=seed, shuffle=folds_shuffle_param)

    logger.info("Declaring metric variables")
    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    score_mcc =np.empty((0,0))
    score_training_time =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    avgs_mcc =np.empty((0,0))
    avgs_training_time =np.empty((0,0))
  
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Selecting Estimator'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    #calibrating estimator
    
    logger.info("Importing untrained CalibratedClassifierCV")
    model = CalibratedClassifierCV(base_estimator=estimator, method=method, cv=fold)
    full_name = str(model).split("(")[0]
    
    progress.value += 1
    
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    
    fold_num = 1
    
    for train_i , test_i in kf.split(data_X,data_y):

        logger.info("Initializing Fold " + str(fold_num))
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
    
        
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
        time_start=time.time()

        if fix_imbalance_param:
            
            logger.info("Initializing SMOTE")

            if fix_imbalance_method_param is None:
                import six
                import sys
                sys.modules['sklearn.externals.six'] = six
                from imblearn.over_sampling import SMOTE
                resampler = SMOTE(random_state = seed)
            else:
                resampler = fix_imbalance_method_param

            Xtrain,ytrain = resampler.fit_sample(Xtrain, ytrain)
            logger.info("Resampling completed")

        if hasattr(model, 'predict_proba'):
        
            logger.info("Fitting Model")
            model.fit(Xtrain,ytrain)
            logger.info("Evaluating Metrics")
            pred_prob = model.predict_proba(Xtest)
            pred_prob = pred_prob[:,1]
            pred_ = model.predict(Xtest)
            sca = metrics.accuracy_score(ytest,pred_)
            
            if y.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(ytest,pred_, average='macro')                
                precision = metrics.precision_score(ytest,pred_, average = 'weighted')
                f1 = metrics.f1_score(ytest,pred_, average='weighted')
                
            else:
                try:
                    sc = metrics.roc_auc_score(ytest,pred_prob)
                except:
                    sc = 0
                recall = metrics.recall_score(ytest,pred_)                
                precision = metrics.precision_score(ytest,pred_)
                f1 = metrics.f1_score(ytest,pred_)
                
        else:
            logger.info("Fitting Model")
            model.fit(Xtrain,ytrain)
            logger.info("Evaluating Metrics")
            pred_prob = 0.00
            pred_ = model.predict(Xtest)
            sca = metrics.accuracy_score(ytest,pred_)
            
            if y.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(ytest,pred_, average='macro')                
                precision = metrics.precision_score(ytest,pred_, average = 'weighted')
                f1 = metrics.f1_score(ytest,pred_, average='weighted')

            else:
                try:
                    sc = metrics.roc_auc_score(ytest,pred_prob)
                except:
                    sc = 0
                recall = metrics.recall_score(ytest,pred_)                
                precision = metrics.precision_score(ytest,pred_)
                f1 = metrics.f1_score(ytest,pred_)

        logger.info("Compiling Metrics") 
        time_end=time.time()
        kappa = metrics.cohen_kappa_score(ytest,pred_)
        mcc = metrics.matthews_corrcoef(ytest,pred_)
        training_time=time_end-time_start
        score_acc = np.append(score_acc,sca)
        score_auc = np.append(score_auc,sc)
        score_recall = np.append(score_recall,recall)
        score_precision = np.append(score_precision,precision)
        score_f1 =np.append(score_f1,f1)
        score_kappa =np.append(score_kappa,kappa) 
        score_mcc =np.append(score_mcc,mcc)
        score_training_time =np.append(score_training_time,training_time)
       
        progress.value += 1
        
        
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa],'MCC':[mcc]}).round(round)
        master_display = pd.concat([master_display, fold_results],ignore_index=True)
        fold_results = []
        
        '''
        TIME CALCULATION SUB-SECTION STARTS HERE
        '''
        t1 = time.time()
        
        tt = (t1 - t0) * (fold-fold_num) / 60
        tt = np.around(tt, 2)
        
        if tt < 1:
            tt = str(np.around((tt * 60), 2))
            ETC = tt + ' Seconds Remaining'
                
        else:
            tt = str (tt)
            ETC = tt + ' Minutes Remaining'
            
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[2,1:] = ETC
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
            
        fold_num += 1
        
        '''
        TIME CALCULATION ENDS HERE
        '''
        
        if verbose:
            if html_param:
                update_display(master_display, display_id = display_id)
            
        
        '''
        
        Update_display() ends here
        
        '''

    logger.info("Calculating mean and std")        
    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    mean_mcc=np.mean(score_mcc)
    mean_training_time=np.sum(score_training_time)
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    std_mcc=np.std(score_mcc)
    std_training_time=np.std(score_training_time)
    
    avgs_acc = np.append(avgs_acc, mean_acc)
    avgs_acc = np.append(avgs_acc, std_acc) 
    avgs_auc = np.append(avgs_auc, mean_auc)
    avgs_auc = np.append(avgs_auc, std_auc)
    avgs_recall = np.append(avgs_recall, mean_recall)
    avgs_recall = np.append(avgs_recall, std_recall)
    avgs_precision = np.append(avgs_precision, mean_precision)
    avgs_precision = np.append(avgs_precision, std_precision)
    avgs_f1 = np.append(avgs_f1, mean_f1)
    avgs_f1 = np.append(avgs_f1, std_f1)
    avgs_kappa = np.append(avgs_kappa, mean_kappa)
    avgs_kappa = np.append(avgs_kappa, std_kappa)
    avgs_mcc = np.append(avgs_mcc, mean_mcc)
    avgs_mcc = np.append(avgs_mcc, std_mcc)
    avgs_training_time = np.append(avgs_training_time, mean_training_time)
    avgs_training_time = np.append(avgs_training_time, std_training_time)
    
    progress.value += 1

    logger.info("Creating metrics dataframe")   
    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa,'MCC' : score_mcc})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa,'MCC' : avgs_mcc},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    
    # yellow the mean
    model_results=model_results.style.apply(lambda x: ['background: yellow' if (x.name == 'Mean') else '' for i in x], axis=1)
    model_results=model_results.set_precision(round)
    
    #refitting the model on complete X_train, y_train
    monitor.iloc[1,1:] = 'Compiling Final Model'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    model_fit_start = time.time()
    logger.info("Finalizing model")
    model.fit(data_X, data_y)
    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
    
    progress.value += 1
    
    #end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    #storing results in create_model_container
    logger.info("Uploading results into container")
    create_model_container.append(model_results.data)
    display_container.append(model_results.data)

    #storing results in master_model_container
    logger.info("Uploading model into container")
    master_model_container.append(model)

    #mlflow logging
    if logging_param:
        
        logger.info("Creating MLFlow logs")

        #Creating Logs message monitor
        monitor.iloc[1,1:] = 'Creating Logs'
        monitor.iloc[2,1:] = 'Almost Finished'    
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        #import mlflow
        import mlflow
        import mlflow.sklearn
        from pathlib import Path
        import os

        mlflow.set_experiment(exp_name_log)

        with mlflow.start_run(run_name=base_estimator_full_name) as run:

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
            mlflow.log_metrics({"Accuracy": avgs_acc[0], "AUC": avgs_auc[0], "Recall": avgs_recall[0], "Precision" : avgs_precision[0],
                                "F1": avgs_f1[0], "Kappa": avgs_kappa[0], "MCC": avgs_mcc[0]})
            

            #set tag of compare_models
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
            model_results.data.to_html('Results.html', col_space=65, justify='left')
            mlflow.log_artifact('Results.html')
            os.remove('Results.html')

            # Generate hold-out predictions and save as html
            holdout = predict_model(model, verbose=False)
            holdout_score = pull()
            del(holdout)
            display_container.pop(-1)
            holdout_score.to_html('Holdout.html', col_space=65, justify='left')
            mlflow.log_artifact('Holdout.html')
            os.remove('Holdout.html')

            # Log AUC and Confusion Matrix plot
            if log_plots_param:

                logger.info("SubProcess plot_model() called ==================================")

                try:
                    plot_model(model, plot = 'auc', verbose=False, save=True, system=False)
                    mlflow.log_artifact('AUC.png')
                    os.remove("AUC.png")
                except:
                    pass

                try:
                    plot_model(model, plot = 'confusion_matrix', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Confusion Matrix.png')
                    os.remove("Confusion Matrix.png")
                except:
                    pass

                try:
                    plot_model(model, plot = 'feature', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Feature Importance.png')
                    os.remove("Feature Importance.png")
                except:
                    pass
                
                logger.info("SubProcess plot_model() end ==================================")

            # Log model and transformation pipeline
            from copy import deepcopy

            # get default conda env
            from mlflow.sklearn import get_default_conda_env
            default_conda_env = get_default_conda_env()
            default_conda_env['name'] = str(exp_name_log) + '-env'
            default_conda_env.get('dependencies').pop(-3)
            dependencies = default_conda_env.get('dependencies')[-1]
            from pycaret.utils import __version__
            dep = 'pycaret==' + str(__version__())
            dependencies['pip'] = [dep]
            
            # define model signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(data_before_preprocess.drop([target_param], axis=1))
            input_example = data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(['trained model', model])
            mlflow.sklearn.log_model(prep_pipe_temp, "model", conda_env = default_conda_env, signature = signature, input_example = input_example)
            del(prep_pipe_temp)

    if verbose:
        clear_output()
        if html_param:
            display(model_results)
        else:
            print(model_results.data)
    
    logger.info("create_model_container: " + str(len(create_model_container)))
    logger.info("master_model_container: " + str(len(master_model_container)))
    logger.info("display_container: " + str(len(display_container)))

    logger.info(str(model))
    logger.info("calibrate_model() succesfully completed......................................")

    return model

def optimize_threshold(estimator, 
                       true_positive = 0, 
                       true_negative = 0, 
                       false_positive = 0, 
                       false_negative = 0):
    
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
    
    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing optimize_threshold()")
    logger.info("""optimize_threshold(estimator={}, true_positive={}, true_negative={}, false_positive={}, false_negative={})""".\
        format(str(estimator), str(true_positive), str(true_negative), str(false_positive), str(false_negative)))

    logger.info("Importing libraries")
    
    #import libraries
    import sys
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from IPython.display import clear_output
    
    #cufflinks
    import cufflinks as cf
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)
    
    
    '''
    ERROR HANDLING STARTS HERE
    '''
    
    logger.info("Checking exceptions")

    #exception 1 for multi-class
    if y.value_counts().count() > 2:
        sys.exit("(Type Error) optimize_threshold() cannot be used when target is multi-class. ")
    
    model_name = str(estimator).split("(")[0]
    if 'OneVsRestClassifier' in model_name:
        sys.exit("(Type Error) optimize_threshold() cannot be used when target is multi-class. ")
    
    #check predict_proba value
    if type(estimator) is not list:
        if not hasattr(estimator, 'predict_proba'):
            sys.exit("(Type Error) Estimator doesn't support predict_proba function and cannot be used in optimize_threshold().  ")        
        
    #check cost function type
    allowed_types = [int, float]
    
    if type(true_positive) not in allowed_types:
        sys.exit("(Type Error) true_positive parameter only accepts float or integer value. ")
        
    if type(true_negative) not in allowed_types:
        sys.exit("(Type Error) true_negative parameter only accepts float or integer value. ")
        
    if type(false_positive) not in allowed_types:
        sys.exit("(Type Error) false_positive parameter only accepts float or integer value. ")
        
    if type(false_negative) not in allowed_types:
        sys.exit("(Type Error) false_negative parameter only accepts float or integer value. ")
    
    

    '''
    ERROR HANDLING ENDS HERE
    '''        

        
    #define model as estimator
    model = estimator
    
    model_name = str(model).split("(")[0]
    if 'CatBoostClassifier' in model_name:
        model_name = 'CatBoostClassifier'
        
    #generate predictions and store actual on y_test in numpy array
    actual = np.array(y_test)
    
    if type(model) is list:
        logger.info("Model Type : Stacking")
        predicted = predict_model(model)
        model_name = 'Stacking'
        clear_output()
        try:
            predicted = np.array(predicted['Score'])
        except:
            logger.info("Meta model doesn't support predict_proba function.")
            sys.exit("(Type Error) Meta model doesn't support predict_proba function. Cannot be used in optimize_threshold(). ")        
        
    else:
        predicted = model.predict_proba(X_test)
        predicted = predicted[:,1]

    """
    internal function to calculate loss starts here
    """
    
    logger.info("Defining loss function")

    def calculate_loss(actual,predicted,
                       tp_cost=true_positive,tn_cost=true_negative,
                       fp_cost=false_positive,fn_cost=false_negative):
        
        #true positives
        tp = predicted + actual
        tp = np.where(tp==2, 1, 0)
        tp = tp.sum()
        
        #true negative
        tn = predicted + actual
        tn = np.where(tn==0, 1, 0)
        tn = tn.sum()
        
        #false positive
        fp = (predicted > actual).astype(int)
        fp = np.where(fp==1, 1, 0)
        fp = fp.sum()
        
        #false negative
        fn = (predicted < actual).astype(int)
        fn = np.where(fn==1, 1, 0)
        fn = fn.sum()
        
        total_cost = (tp_cost*tp) + (tn_cost*tn) + (fp_cost*fp) + (fn_cost*fn)
        
        return total_cost
    
    
    """
    internal function to calculate loss ends here
    """
    
    grid = np.arange(0,1,0.01)
    
    #loop starts here
    
    cost = []
    #global optimize_results
    
    logger.info("Iteration starts at 0")

    for i in grid:
        
        pred_prob = (predicted >= i).astype(int)
        cost.append(calculate_loss(actual,pred_prob))
        
    optimize_results = pd.DataFrame({'Probability Threshold' : grid, 'Cost Function' : cost })
    fig = px.line(optimize_results, x='Probability Threshold', y='Cost Function', line_shape='linear')
    fig.update_layout(plot_bgcolor='rgb(245,245,245)')
    title= str(model_name) + ' Probability Threshold Optimization'
    
    #calculate vertical line
    y0 = optimize_results['Cost Function'].min()
    y1 = optimize_results['Cost Function'].max()
    x0 = optimize_results.sort_values(by='Cost Function', ascending=False).iloc[0][0]
    x1 = x0
    
    t = x0.round(2)
    
    fig.add_shape(dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,line=dict(color="red",width=2)))
    fig.update_layout(title={'text': title, 'y':0.95,'x':0.45,'xanchor': 'center','yanchor': 'top'})
    logger.info("Figure ready for render")
    fig.show()
    print('Optimized Probability Threshold: ' + str(t) + ' | ' + 'Optimized Cost Function: ' + str(y1))
    logger.info("optimize_threshold() succesfully completed......................................")

def predict_model(estimator, 
                  data=None,
                  probability_threshold=None,
                  verbose=True): #added in pycaret==2.0.0
    
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
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #general dependencies
    import sys
    import numpy as np
    import pandas as pd
    import re
    from sklearn import metrics
    from copy import deepcopy
    from IPython.display import clear_output, display, update_display
    
    """
    exception checking starts here
    """

    model_name = str(estimator).split("(")[0]
    if probability_threshold is not None:
        if 'OneVsRestClassifier' in model_name:
            sys.exit("(Type Error) probability_threshold parameter cannot be used when target is multi-class. ")
            
    #probability_threshold allowed types    
    if probability_threshold is not None:
        allowed_types = [int,float]
        if type(probability_threshold) not in allowed_types:
            sys.exit("(Type Error) probability_threshold parameter only accepts value between 0 to 1. ")
    
    #probability_threshold allowed types
    if probability_threshold is not None:
        if probability_threshold > 1:
            sys.exit("(Type Error) probability_threshold parameter only accepts value between 0 to 1. ")
    
    #probability_threshold allowed types    
    if probability_threshold is not None:
        if probability_threshold < 0:
            sys.exit("(Type Error) probability_threshold parameter only accepts value between 0 to 1. ")

    """
    exception checking ends here
    """

    #dataset
    if data is None:
        
        if 'Pipeline' in str(type(estimator)):
            estimator = estimator[-1]

        Xtest = X_test.copy()
        ytest = y_test.copy()
        X_test_ = X_test.copy()
        y_test_ = y_test.copy()

        _, dtypes = next(step for step in prep_pipe.steps if step[0] == "dtypes")
        
        index = None
        Xtest.reset_index(drop=True, inplace=True)
        ytest.reset_index(drop=True, inplace=True)
        X_test_.reset_index(drop=True, inplace=True)
        y_test_.reset_index(drop=True, inplace=True)

    else:

        if 'Pipeline' in str(type(estimator)):
            _, dtypes = next(step for step in estimator.steps if step[0] == "dtypes")
        else:
            try:
                _, dtypes = next(step for step in prep_pipe.steps if step[0] == "dtypes")
                estimator_ = deepcopy(prep_pipe)
                estimator_.steps.append(['trained model',estimator])
                estimator = estimator_
                del(estimator_)

            except:
                sys.exit("Pipeline not found")
            
        Xtest = data.copy()
        X_test_ = data.copy()
        
    # function to replace encoded labels with their original values
    # will not run if categorical_labels is false
    def replace_lables_in_column(label_column):
        if dtypes and hasattr(dtypes, "replacement"):
            replacement_mapper = {int(v): k for k, v in dtypes.replacement.items()}
            label_column.replace(replacement_mapper, inplace=True)

    #model name
    full_name = str(estimator).split("(")[0]
    def putSpace(input):
        words = re.findall('[A-Z][a-z]*', input)
        words = ' '.join(words)
        return words  
    full_name = putSpace(full_name)

    if full_name == 'Gaussian N B':
        full_name = 'Naive Bayes'

    elif full_name == 'M L P Classifier':
        full_name = 'MLP Classifier'

    elif full_name == 'S G D Classifier':
        full_name = 'SVM - Linear Kernel'

    elif full_name == 'S V C':
        full_name = 'SVM - Radial Kernel'

    elif full_name == 'X G B Classifier':
        full_name = 'Extreme Gradient Boosting'

    elif full_name == 'L G B M Classifier':
        full_name = 'Light Gradient Boosting Machine'

    elif 'Cat Boost Classifier' in full_name:
        full_name = 'CatBoost Classifier'

    #prediction starts here
    
    pred_ = estimator.predict(Xtest)
    
    try:
        pred_prob = estimator.predict_proba(Xtest)
        
        if len(pred_prob[0]) > 2:
            p_counter = 0
            d = []
            for i in range(0,len(pred_prob)):
                d.append(pred_prob[i][pred_[p_counter]])
                p_counter += 1
                
            pred_prob = d
            
        else:
            pred_prob = pred_prob[:,1]

    except:
        pass
    
    if probability_threshold is not None:
        try:
            pred_ = (pred_prob >= probability_threshold).astype(int)
        except:
            pass
    
    if data is None:

        sca = metrics.accuracy_score(ytest,pred_)

        try:
            sc = metrics.roc_auc_score(ytest,pred_prob)
        except:
            sc = 0
        
        if y.value_counts().count() > 2:
            recall = metrics.recall_score(ytest,pred_, average='macro')
            precision = metrics.precision_score(ytest,pred_, average = 'weighted')
            f1 = metrics.f1_score(ytest,pred_, average='weighted')
        else:
            recall = metrics.recall_score(ytest,pred_)
            precision = metrics.precision_score(ytest,pred_)
            f1 = metrics.f1_score(ytest,pred_)                
            
        kappa = metrics.cohen_kappa_score(ytest,pred_)
        mcc = metrics.matthews_corrcoef(ytest,pred_)

        df_score = pd.DataFrame( {'Model' : [full_name], 'Accuracy' : [sca], 'AUC' : [sc], 'Recall' : [recall], 'Prec.' : [precision],
                            'F1' : [f1], 'Kappa' : [kappa], 'MCC':[mcc]})
        df_score = df_score.round(4)
        
        if verbose:
            display(df_score)
        
    label = pd.DataFrame(pred_)
    label.columns = ['Label']
    label['Label']=label['Label'].astype(int)
    replace_lables_in_column(label['Label'])
    
    if data is None:
        replace_lables_in_column(ytest)
        X_test_ = pd.concat([Xtest,ytest,label], axis=1)
    else:
        X_test_.insert(len(X_test_.columns), "Label", label["Label"].to_list())

    if hasattr(estimator,'predict_proba'):
        try:
            score = pd.DataFrame(pred_prob)
            score.columns = ['Score']
            score = score.round(4)
            X_test_ = pd.concat([X_test_,score], axis=1)
        except:
            pass
    
    #store predictions on hold-out in display_container
    try:
        display_container.append(df_score)
    except:
        pass

    return X_test_

def finalize_model(estimator):
    
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
    
    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing finalize_model()")
    logger.info("""finalize_model(estimator={})""".\
        format(str(estimator)))

    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #run_time
    import datetime, time
    runtime_start = time.time()

    logger.info("Importing libraries")
    #import depedencies
    from IPython.display import clear_output, update_display
    from sklearn.base import clone
    from copy import deepcopy
    import numpy as np
    
    logger.info("Getting model name")
    
    #determine runname for logging
    def get_model_name(e):
        return str(e).split("(")[0]
    
    model_dict_logging = {'ExtraTreesClassifier' : 'Extra Trees Classifier',
                            'GradientBoostingClassifier' : 'Gradient Boosting Classifier', 
                            'RandomForestClassifier' : 'Random Forest Classifier',
                            'LGBMClassifier' : 'Light Gradient Boosting Machine',
                            'XGBClassifier' : 'Extreme Gradient Boosting',
                            'AdaBoostClassifier' : 'Ada Boost Classifier', 
                            'DecisionTreeClassifier' : 'Decision Tree Classifier', 
                            'RidgeClassifier' : 'Ridge Classifier',
                            'LogisticRegression' : 'Logistic Regression',
                            'KNeighborsClassifier' : 'K Neighbors Classifier',
                            'GaussianNB' : 'Naive Bayes',
                            'SGDClassifier' : 'SVM - Linear Kernel',
                            'SVC' : 'SVM - Radial Kernel',
                            'GaussianProcessClassifier' : 'Gaussian Process Classifier',
                            'MLPClassifier' : 'MLP Classifier',
                            'QuadraticDiscriminantAnalysis' : 'Quadratic Discriminant Analysis',
                            'LinearDiscriminantAnalysis' : 'Linear Discriminant Analysis',
                            'CatBoostClassifier' : 'CatBoost Classifier',
                            'BaggingClassifier' : 'Bagging Classifier',
                            'VotingClassifier' : 'Voting Classifier',
                            'StackingClassifier' : 'Stacking Classifier'}
                            

    if len(estimator.classes_) > 2:

        if hasattr(estimator, 'voting'):
            mn = get_model_name(estimator)
        else:
            mn = get_model_name(estimator.estimator)

    else:

        if hasattr(estimator, 'voting'):
            mn = 'VotingClassifier'
        else:
            mn = get_model_name(estimator)

        if 'BaggingClassifier' in mn:
            mn = get_model_name(estimator.base_estimator_)

        if 'CalibratedClassifierCV' in mn:
            mn = get_model_name(estimator.base_estimator)

    if 'catboost' in mn:
        mn = 'CatBoostClassifier'

    full_name = model_dict_logging.get(mn)
    
    logger.info("Finalizing " + str(full_name))
    model_final = clone(estimator)
    clear_output()
    model_final.fit(X,y)
    model = create_model(estimator=estimator, verbose=False, system=False)
    results = pull()

    #end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    #mlflow logging
    if logging_param:

        logger.info("Creating MLFlow logs")

        #import mlflow
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
            mlflow.log_metrics({"Accuracy": results.iloc[-2]['Accuracy'], "AUC": results.iloc[-2]['AUC'], "Recall": results.iloc[-2]['Recall'],\
                                "Precision" : results.iloc[-2]['Prec.'], "F1": results.iloc[-2]['F1'], "Kappa": results.iloc[-2]['Kappa'],\
                                "MCC": results.iloc[-2]['MCC']})

            #set tag of compare_models
            mlflow.set_tag("Source", "finalize_model")
            
            #create MRI (model registration id)
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

                logger.info("SubProcess plot_model() called ==================================")

                try:
                    plot_model(model_final, plot = 'auc', verbose=False, save=True, system=False)
                    mlflow.log_artifact('AUC.png')
                    os.remove("AUC.png")
                except:
                    pass

                try:
                    plot_model(model_final, plot = 'confusion_matrix', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Confusion Matrix.png')
                    os.remove("Confusion Matrix.png")
                except:
                    pass

                try:
                    plot_model(model_final, plot = 'feature', verbose=False, save=True, system=False)
                    mlflow.log_artifact('Feature Importance.png')
                    os.remove("Feature Importance.png")
                except:
                    pass
                
                logger.info("SubProcess plot_model() end ==================================")

            # Log model and transformation pipeline
            from copy import deepcopy

            # get default conda env
            from mlflow.sklearn import get_default_conda_env
            default_conda_env = get_default_conda_env()
            default_conda_env['name'] = str(exp_name_log) + '-env'
            default_conda_env.get('dependencies').pop(-3)
            dependencies = default_conda_env.get('dependencies')[-1]
            from pycaret.utils import __version__
            dep = 'pycaret==' + str(__version__())
            dependencies['pip'] = [dep]
            
            # define model signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(data_before_preprocess)

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(['trained model', model_final])
            mlflow.sklearn.log_model(prep_pipe_temp, "model", conda_env = default_conda_env, signature = signature)
            del(prep_pipe_temp)

    logger.info("create_model_container: " + str(len(create_model_container)))
    logger.info("master_model_container: " + str(len(master_model_container)))
    logger.info("display_container: " + str(len(display_container)))

    logger.info(str(model_final))
    logger.info("finalize_model() succesfully completed......................................")

    return model_final

def deploy_model(model, 
                 model_name, 
                 platform,
                 authentication):
    
    """
    This function deploys the transformation pipeline and trained model object for
    production use. The platform of deployment can be defined under the platform
    param along with the applicable authentication tokens which are passed as a
    dictionary to the authentication param.
        
    Platform: AWS
    -------------
    Before deploying a model to an AWS S3 ('aws'), environment variables must be 
    configured using the command line interface. To configure AWS env. variables, 
    type aws configure in your python command line. The following information is
    required which can be generated using the Identity and Access Management (IAM) 
    portal of your AWS console account:

    - AWS Access Key ID
    - AWS Secret Key Access
    - Default Region Name (can be seen under Global settings on your AWS console)
    - Default output format (must be left blank)

    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> deploy_model(model = lr, model_name = 'deploy_lr', platform = 'aws', authentication = {'bucket' : 'bucket-name'})

    Platform: GCP
    --------------
    Before deploying a model to Google Cloud Platform (GCP), project must be created either
    using command line or GCP console. Once project is created, you must create a service 
    account and download the service account key as a JSON file, which is then used to 
    set environment variable. 

    Learn more : https://cloud.google.com/docs/authentication/production

    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'c:/path-to-json-file.json' 
    >>> deploy_model(model = lr, model_name = 'deploy_lr', platform = 'gcp', authentication = {'project' : 'project-name', 'bucket' : 'bucket-name'})

    Platform: Azure
    ---------------
    Before deploying a model to Microsoft Azure, environment variables for connection 
    string must be set. Connection string can be obtained from 'Access Keys' of your 
    storage account in Azure.

    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> os.environ['AZURE_STORAGE_CONNECTION_STRING'] = 'connection-string-here' 
    >>> deploy_model(model = lr, model_name = 'deploy_lr', platform = 'azure', authentication = {'container' : 'container-name'})

    Parameters
    ----------
    model : object
        A trained model object should be passed as an estimator. 
    
    model_name : string
        Name of model to be passed as a string.
    
    platform: string
        Name of platform for deployment. 
        Currently accepts: 'aws', 'gcp', 'azure'

    authentication : dict
        Dictionary of applicable authentication tokens.

        When platform = 'aws':
        {'bucket' : 'name of bucket'}

        When platform = 'gcp':
        {'project': 'name of project', 'bucket' : 'name of bucket'}

        When platform = 'azure':
        {'container': 'name of container'}

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
    
    import sys
    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing deploy_model()")
    logger.info("""deploy_model(model={}, model_name={}, authentication={}, platform={})""".\
        format(str(model), str(model_name), str(authentication), str(platform)))

    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #general dependencies
    import ipywidgets as ipw
    import pandas as pd
    from IPython.display import clear_output, update_display
    import os

    if platform == 'aws':
        
        logger.info("Platform : AWS S3")

        #checking if awscli available
        try:
            import awscli
        except:
            logger.error("awscli library not found. pip install awscli to use deploy_model function.")
            sys.exit("awscli library not found. pip install awscli to use deploy_model function.")  
        
        import boto3
        
        logger.info("Saving model in active working directory")
        logger.info("SubProcess save_model() called ==================================")
        save_model(model, model_name = model_name, verbose=False)
        logger.info("SubProcess save_model() end ==================================")
        
        #initiaze s3
        logger.info("Initializing S3 client")
        s3 = boto3.client('s3')
        filename = str(model_name)+'.pkl'
        key = str(model_name)+'.pkl'
        bucket_name = authentication.get('bucket')
        s3.upload_file(filename,bucket_name,key)
        clear_output()
        
        os.remove(filename)
        
        print("Model Succesfully Deployed on AWS S3")
        logger.info("Model Succesfully Deployed on AWS S3")
        logger.info(str(model))

    elif platform == 'gcp':

        logger.info("Platform : GCP")

        try:
            import google.cloud
        except:
            logger.error("google-cloud-storage library not found. pip install google-cloud-storage to use deploy_model function with GCP.")
            sys.exit("google-cloud-storage library not found. pip install google-cloud-storage to use deploy_model function with GCP.")

        logger.info("Saving model in active working directory")
        logger.info("SubProcess save_model() called ==================================")
        save_model(model, model_name=model_name, verbose=False)
        logger.info("SubProcess save_model() end ==================================")

        # initialize deployment
        filename = str(model_name) + '.pkl'
        key = str(model_name) + '.pkl'
        bucket_name = authentication.get('bucket')
        project_name = authentication.get('project')
        try:
            _create_bucket_gcp(project_name, bucket_name)
            _upload_blob_gcp(project_name, bucket_name, filename, key)
        except:
            _upload_blob_gcp(project_name, bucket_name, filename, key)
        
        os.remove(filename)
        
        print("Model Succesfully Deployed on GCP")
        logger.info("Model Succesfully Deployed on GCP")
        logger.info(str(model))

    elif platform == 'azure':

        try:
            import azure.storage.blob
        except:
            logger.error("azure-storage-blob library not found. pip install azure-storage-blob to use deploy_model function with Azure.")
            sys.exit("azure-storage-blob library not found. pip install azure-storage-blob to use deploy_model function with Azure.")

        logger.info("Platform : Azure Blob Storage")

        logger.info("Saving model in active working directory")
        logger.info("SubProcess save_model() called ==================================")
        save_model(model, model_name=model_name, verbose=False)
        logger.info("SubProcess save_model() end ==================================")

        # initialize deployment
        filename = str(model_name) + '.pkl'
        key = str(model_name) + '.pkl'
        container_name = authentication.get('container')
        try:
            container_client = _create_container_azure(container_name)
            _upload_blob_azure(container_name, filename, key)
            del(container_client)
        except:
            _upload_blob_azure(container_name, filename, key)

        os.remove(filename)

        print("Model Succesfully Deployed on Azure Storage Blob")
        logger.info("Model Succesfully Deployed on Azure Storage Blob")
        logger.info(str(model))

    else:
        logger.error('Platform {} is not supported by pycaret or illegal option'.format(platform))
        sys.exit('Platform {} is not supported by pycaret or illegal option'.format(platform))
        
    logger.info("deploy_model() succesfully completed......................................")

def save_model(model, model_name, model_only=False, verbose=True):
    
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
    
    import logging
    from copy import deepcopy

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing save_model()")
    logger.info("""save_model(model={}, model_name={}, model_only={}, verbose={})""".\
        format(str(model), str(model_name), str(model_only), str(verbose)))
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    logger.info("Adding model into prep_pipe")

    if model_only:
        model_ = deepcopy(model)
        logger.warning("Only Model saved. Transformations in prep_pipe are ignored.")
    else:
        model_ = deepcopy(prep_pipe)
        model_.steps.append(['trained model',model]) 
    
    import joblib
    model_name = model_name + '.pkl'
    joblib.dump(model_, model_name)
    if verbose:
        print('Transformation Pipeline and Model Succesfully Saved')
    
    logger.info(str(model_name) + ' saved in current working directory')
    logger.info(str(model_))
    logger.info("save_model() succesfully completed......................................")

def load_model(model_name, 
               platform = None, 
               authentication = None,
               verbose=True):
    
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
        Name of platform, if loading model from cloud. 
        Currently available options are: 'aws', 'gcp', 'azure'.
    
    authentication : dict
        dictionary of applicable authentication tokens.

        When platform = 'aws':
        {'bucket' : 'name of bucket'}

        When platform = 'gcp':
        {'project': 'name of project', 'bucket' : 'name of bucket'}

        When platform = 'azure':
        {'container': 'name of container'}
    
    verbose: Boolean, default = True
        Success message is not printed when verbose is set to False.

    Returns
    -------
    Model Object

    """
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #exception checking
    import sys
    
    if platform is not None:
        if authentication is None:
            sys.exit("(Value Error): Authentication is missing.")

    if platform is None:

        import joblib
        model_name = model_name + '.pkl'
        if verbose:
            print('Transformation Pipeline and Model Successfully Loaded')
        return joblib.load(model_name)
    
    # cloud providers
    elif platform == 'aws':

        import boto3
        bucketname = authentication.get('bucket')
        filename = str(model_name) + '.pkl'
        s3 = boto3.resource('s3')
        s3.Bucket(bucketname).download_file(filename, filename)
        filename = str(model_name)
        model = load_model(filename, verbose=False)
        model = load_model(filename, verbose=False)

        if verbose:
            print('Transformation Pipeline and Model Successfully Loaded')

        return model

    elif platform == 'gcp':

        bucket_name = authentication.get('bucket')
        project_name = authentication.get('project')
        filename = str(model_name) + '.pkl'

        model_downloaded = _download_blob_gcp(project_name,
                                              bucket_name, filename, filename)

        model = load_model(model_name, verbose=False)

        if verbose:
            print('Transformation Pipeline and Model Successfully Loaded')
        return model

    elif platform == 'azure':

        container_name = authentication.get('container')
        filename = str(model_name) + '.pkl'

        model_downloaded = _download_blob_azure(container_name, filename, filename)

        model = load_model(model_name, verbose=False)

        if verbose:
            print('Transformation Pipeline and Model Successfully Loaded')
        return model
    else:
        print('Platform { } is not supported by pycaret or illegal option'.format(platform))

def automl(optimize='Accuracy', use_holdout=False):
    
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

    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing automl()")
    logger.info("""automl(optimize={}, use_holdout={})""".\
        format(str(optimize), str(use_holdout)))

    if optimize == 'Accuracy':
        compare_dimension = 'Accuracy' 
    elif optimize == 'AUC':
        compare_dimension = 'AUC' 
    elif optimize == 'Recall':
        compare_dimension = 'Recall'
    elif optimize == 'Precision':
        compare_dimension = 'Prec.'
    elif optimize == 'F1':
        compare_dimension = 'F1' 
    elif optimize == 'Kappa':
        compare_dimension = 'Kappa'
    elif optimize == 'MCC':
        compare_dimension = 'MCC' 
        
    scorer = []

    if use_holdout:
        logger.info("Model Selection Basis : Holdout set")
        for i in master_model_container:
            pred_holdout = predict_model(i, verbose=False)
            p = pull()
            display_container.pop(-1)
            p = p[compare_dimension][0]
            scorer.append(p)

    else:
        logger.info("Model Selection Basis : CV Results on Training set")
        for i in create_model_container:
            r = i[compare_dimension][-2:][0]
            scorer.append(r)

    #returning better model
    index_scorer = scorer.index(max(scorer))
    
    automl_result = master_model_container[index_scorer]

    logger.info("SubProcess finalize_model() called ==================================")
    automl_finalized = finalize_model(automl_result)
    logger.info("SubProcess finalize_model() end ==================================")

    logger.info(str(automl_finalized))
    logger.info("automl() succesfully completed......................................")

    return automl_finalized

def pull():
    """
    Returns latest displayed table.

    Returns
    -------
    pandas.DataFrame
        Equivalent to get_config('display_container')[-1]

    """
    return display_container[-1]

def models(type=None):

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
      
    Returns
    -------
    pandas.DataFrame

    """
    
    import pandas as pd

    model_id = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']
    
    model_name = ['Logistic Regression',
                    'K Neighbors Classifier',
                    'Naive Bayes',
                    'Decision Tree Classifier',
                    'SVM - Linear Kernel',
                    'SVM - Radial Kernel',
                    'Gaussian Process Classifier',
                    'MLP Classifier',
                    'Ridge Classifier',
                    'Random Forest Classifier',
                    'Quadratic Discriminant Analysis',
                    'Ada Boost Classifier',
                    'Gradient Boosting Classifier',
                    'Linear Discriminant Analysis',
                    'Extra Trees Classifier',
                    'Extreme Gradient Boosting',
                    'Light Gradient Boosting Machine',
                    'CatBoost Classifier']    

    model_ref = ['sklearn.linear_model.LogisticRegression',
                'sklearn.neighbors.KNeighborsClassifier',
                'sklearn.naive_bayes.GaussianNB',
                'sklearn.tree.DecisionTreeClassifier',
                'sklearn.linear_model.SGDClassifier',
                'sklearn.svm.SVC',
                'sklearn.gaussian_process.GPC',
                'sklearn.neural_network.MLPClassifier',
                'sklearn.linear_model.RidgeClassifier',
                'sklearn.ensemble.RandomForestClassifier',
                'sklearn.discriminant_analysis.QDA',
                'sklearn.ensemble.AdaBoostClassifier',
                'sklearn.ensemble.GradientBoostingClassifier',
                'sklearn.discriminant_analysis.LDA', 
                'sklearn.ensemble.ExtraTreesClassifier',
                'xgboost.readthedocs.io',
                'github.com/microsoft/LightGBM',
                'catboost.ai']

    model_turbo = [True, True, True, True, True, False, False, False, True,
                   True, True, True, True, True, True, True, True, True]

    df = pd.DataFrame({'ID' : model_id, 
                       'Name' : model_name,
                       'Reference' : model_ref,
                        'Turbo' : model_turbo})

    df.set_index('ID', inplace=True)

    linear_models = ['lr', 'ridge', 'svm']
    tree_models = ['dt'] 
    ensemble_models = ['rf', 'et', 'gbc', 'xgboost', 'lightgbm', 'catboost', 'ada']

    if type == 'linear':
        df = df[df.index.isin(linear_models)]
    if type == 'tree':
        df = df[df.index.isin(tree_models)]
    if type == 'ensemble':
        df = df[df.index.isin(ensemble_models)]

    return df

def get_logs(experiment_name = None, save = False):

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
        sys.exit('No active run found. Check logging parameter in setup or to get logs for inactive run pass experiment_name.')
    
    exp_id = client.get_experiment_by_name(exp_name_log_).experiment_id    
    runs = mlflow.search_runs(exp_id)

    if save:
        file_name = str(exp_name_log_) + '_logs.csv'
        runs.to_csv(file_name, index=False)

    return runs

def get_config(variable):

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

    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing get_config()")
    logger.info("""get_config(variable={})""".\
        format(str(variable)))

    if variable == 'X':
        global_var = X
    
    if variable == 'y':
        global_var = y

    if variable == 'X_train':
        global_var = X_train

    if variable == 'X_test':
        global_var = X_test

    if variable == 'y_train':
        global_var = y_train

    if variable == 'y_test':
        global_var = y_test

    if variable == 'seed':
        global_var = seed

    if variable == 'prep_pipe':
        global_var = prep_pipe

    if variable == 'folds_shuffle_param':
        global_var = folds_shuffle_param
        
    if variable == 'n_jobs_param':
        global_var = n_jobs_param

    if variable == 'html_param':
        global_var = html_param

    if variable == 'create_model_container':
        global_var = create_model_container

    if variable == 'master_model_container':
        global_var = master_model_container

    if variable == 'display_container':
        global_var = display_container

    if variable == 'exp_name_log':
        global_var = exp_name_log

    if variable == 'logging_param':
        global_var = logging_param

    if variable == 'log_plots_param':
        global_var = log_plots_param

    if variable == 'USI':
        global_var = USI

    if variable == 'fix_imbalance_param':
        global_var = fix_imbalance_param

    if variable == 'fix_imbalance_method_param':
        global_var = fix_imbalance_method_param

    if variable == 'data_before_preprocess':
        global_var = data_before_preprocess

    if variable == 'target_param':
        global_var = target_param

    if variable == 'gpu_param':
        global_var = gpu_param

    logger.info("Global variable: " + str(variable) + ' returned')
    logger.info("get_config() succesfully completed......................................")

    return global_var

def set_config(variable,value):

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
    - target_param: name of target variable
    - gpu_param: use_gpu param configured through setup

    Example
    -------
    >>> set_config('seed', 123) 

    This will set the global seed to '123'.

    """

    import logging

    try:
        hasattr(logger, 'name')
    except:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)
        
        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()
        
        ch = logging.FileHandler('logs.log')
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing set_config()")
    logger.info("""set_config(variable={}, value={})""".\
        format(str(variable), str(value)))

    if variable == 'X':
        global X
        X = value

    if variable == 'y':
        global y
        y = value

    if variable == 'X_train':
        global X_train
        X_train = value

    if variable == 'X_test':
        global X_test
        X_test = value

    if variable == 'y_train':
        global y_train
        y_train = value

    if variable == 'y_test':
        global y_test
        y_test = value

    if variable == 'seed':
        global seed
        seed = value

    if variable == 'prep_pipe':
        global prep_pipe
        prep_pipe = value

    if variable == 'folds_shuffle_param':
        global folds_shuffle_param
        folds_shuffle_param = value

    if variable == 'n_jobs_param':
        global n_jobs_param
        n_jobs_param = value

    if variable == 'html_param':
        global html_param
        html_param = value

    if variable == 'create_model_container':
        global create_model_container
        create_model_container = value

    if variable == 'master_model_container':
        global master_model_container
        master_model_container = value

    if variable == 'display_container':
        global display_container
        display_container = value

    if variable == 'exp_name_log':
        global exp_name_log
        exp_name_log = value

    if variable == 'logging_param':
        global logging_param
        logging_param = value

    if variable == 'log_plots_param':
        global log_plots_param
        log_plots_param = value

    if variable == 'USI':
        global USI
        USI = value

    if variable == 'fix_imbalance_param':
        global fix_imbalance_param
        fix_imbalance_param = value

    if variable == 'fix_imbalance_method_param':
        global fix_imbalance_method_param
        fix_imbalance_method_param = value

    if variable == 'data_before_preprocess':
        global data_before_preprocess
        data_before_preprocess = value

    if variable == 'target_param':
        global target_param
        target_param = value

    if variable == 'gpu_param':
        global gpu_param
        gpu_param = value

    logger.info("Global variable:  " + str(variable) + ' updated')
    logger.info("set_config() succesfully completed......................................")

def get_system_logs():

    """
    Read and print 'logs.log' file from current active directory
    """

    file = open('logs.log', 'r')
    lines = file.read().splitlines()
    file.close()

    for line in lines:
        if not line:
            continue

        columns = [col.strip() for col in line.split(':') if col]
        print(columns)

def _create_bucket_gcp(project_name, bucket_name):
    """
    Creates a bucket on Google Cloud Platform if it does not exists already

    Example
    -------
    >>> _create_bucket_gcp(project_name='GCP-Essentials', bucket_name='test-pycaret-gcp')

    Parameters
    ----------
    project_name : string
        A Project name on GCP Platform (Must have been created from console).

    bucket_name : string
        Name of the storage bucket to be created if does not exists already.

    Returns
    -------
    None
    """

    # bucket_name = "your-new-bucket-name"
    from google.cloud import storage
    storage_client = storage.Client(project_name)

    buckets = storage_client.list_buckets()

    if bucket_name not in buckets:
        bucket = storage_client.create_bucket(bucket_name)
        logger.info("Bucket {} created".format(bucket.name))
    else:
        raise FileExistsError('{} already exists'.format(bucket_name))

def _upload_blob_gcp(project_name, bucket_name, source_file_name, destination_blob_name):

    """
    Upload blob to GCP storage bucket

    Example
    -------
    >>> _upload_blob_gcp(project_name='GCP-Essentials', bucket_name='test-pycaret-gcp', \
                        source_file_name='model-101.pkl', destination_blob_name='model-101.pkl')

    Parameters
    ----------
    project_name : string
        A Project name on GCP Platform (Must have been created from console).

    bucket_name : string
        Name of the storage bucket to be created if does not exists already.

    source_file_name : string
        A blob/file name to copy to GCP

    destination_blob_name : string
        Name of the destination file to be stored on GCP

    Returns
    -------
    None
    """

    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    from google.cloud import storage
    storage_client = storage.Client(project_name)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    logger.info(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def _download_blob_gcp(project_name, bucket_name, source_blob_name, destination_file_name):
    """
    Download a blob from GCP storage bucket

    Example
    -------
    >>> _download_blob_gcp(project_name='GCP-Essentials', bucket_name='test-pycaret-gcp', \
                          source_blob_name='model-101.pkl', destination_file_name='model-101.pkl')

    Parameters
    ----------
    project_name : string
        A Project name on GCP Platform (Must have been created from console).

    bucket_name : string
        Name of the storage bucket to be created if does not exists already.

    source_blob_name : string
        A blob/file name to download from GCP bucket

    destination_file_name : string
        Name of the destination file to be stored locally

    Returns
    -------
    Model Object
    """

    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    from google.cloud import storage
    storage_client = storage.Client(project_name)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    if destination_file_name is not None:
        blob.download_to_filename(destination_file_name)

        logger.info(
            "Blob {} downloaded to {}.".format(
                source_blob_name, destination_file_name
            )
        )

    return blob

def _create_container_azure(container_name):
    """
    Creates a storage container on Azure Platform. gets the connection string from the environment variables.

    Example
    -------
    >>>  container_client = _create_container_azure(container_name='test-pycaret-azure')

    Parameters
    ----------
    container_name : string
        Name of the storage container to be created if does not exists already.

    Returns
    -------
    cotainer_client
    """

    # Create the container
    import os, uuid
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.create_container(container_name)
    return container_client

def _upload_blob_azure(container_name, source_file_name, destination_blob_name):
    """
    Upload blob to Azure storage  container

    Example
    -------
    >>>  _upload_blob_azure(container_name='test-pycaret-azure', source_file_name='model-101.pkl', \
                           destination_blob_name='model-101.pkl')

    Parameters
    ----------
    container_name : string
        Name of the storage bucket to be created if does not exists already.

    source_file_name : string
        A blob/file name to copy to Azure

    destination_blob_name : string
        Name of the destination file to be stored on Azure

    Returns
    -------
    None
    """

    import os, uuid
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=destination_blob_name)

    # Upload the created file
    with open(source_file_name, "rb") as data:
      blob_client.upload_blob(data, overwrite=True)

def _download_blob_azure(container_name, source_blob_name, destination_file_name):
    """
    Download blob from Azure storage  container

    Example
    -------
    >>>  _download_blob_azure(container_name='test-pycaret-azure', source_blob_name='model-101.pkl', \
                             destination_file_name='model-101.pkl')

    Parameters
    ----------
    container_name : string
        Name of the storage bucket to be created if does not exists already.

    source_blob_name : string
        A blob/file name to download from Azure storage container

    destination_file_name : string
        Name of the destination file to be stored locally

    Returns
    -------
    None
    """

    import os, uuid
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=source_blob_name)

    if destination_file_name is not None:
        with open(destination_file_name, "wb") as download_file:
          download_file.write(blob_client.download_blob().readall())
