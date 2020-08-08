# Module: Anomaly Detection
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT
# Release: PyCaret 2.0x
# Last modified : 30/07/2020

def setup(data, 
          categorical_features = None,
          categorical_imputation = 'constant',
          ordinal_features = None, 
          high_cardinality_features = None,
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
          remove_multicollinearity = False,
          multicollinearity_threshold = 0.9,
          group_features = None,
          group_names = None,
          supervised = False,
          supervised_target = None,
          n_jobs = -1, #added in pycaret==2.0.0
          html = True, #added in pycaret==2.0.0
          session_id = None,
          log_experiment = False, #added in pycaret==2.0.0
          experiment_name = None, #added in pycaret==2.0.0
          log_plots = False, #added in pycaret==2.0.0
          log_profile = False, #added in pycaret==2.0.0
          log_data = False, #added in pycaret==2.0.0
          silent=False, #added in pycaret==2.0.0
          verbose=True,
          profile = False):
    
    """
    This function initializes the environment in pycaret. setup() must called before
    executing any other function in pycaret. It takes one mandatory parameter:
    dataframe {array-like, sparse matrix}. 

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> experiment_name = setup(data = anomaly, normalize = True)
    
    'anomaly' is a pandas Dataframe.

    Parameters
    ----------
    data : {array-like, sparse matrix}
        Shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features in dataframe.
    
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
    
    remove_multicollinearity: bool, default = False
        When set to True, the variables with inter-correlations higher than the threshold
        defined under the multicollinearity_threshold param are dropped. When two features
        are highly correlated with each other, the feature with higher average correlation 
        in the feature space is dropped. 
    
    multicollinearity_threshold: float, default = 0.9
        Threshold used for dropping the correlated features. Only comes into effect when 
        remove_multicollinearity is set to True.
    
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
    
    supervised: bool, default = False
        When set to True, supervised_target column is ignored for transformation. This
        param is only for internal use. 
    
    supervised_target: string, default = None
        Name of supervised_target column that will be ignored for transformation. Only
        applicable when tune_model() function is used. This param is only for internal use.

    n_jobs: int, default = -1
        The number of jobs to run in parallel (for functions that supports parallel 
        processing) -1 means using all processors. To run all functions on single processor 
        set n_jobs to None.

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
        Name of experiment for logging. When set to None, 'clf' is by default used as 
        alias for the experiment name.    

    log_plots: bool, default = False
        When set to True, specific plots are logged in MLflow as a png file. By default,
        it is set to False. 

    log_profile: bool, default = False
        When set to True, data profile is also logged on MLflow as a html file. By default,
        it is set to False. 

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
        This function returns various outputs that are stored in variable
        as tuple. They are used by other functions in pycaret.
      
          
    """
    
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

    logger.info("PyCaret Anomaly Detection Module")
    logger.info('version ' + str(ver))
    logger.info("Initializing setup()")

    #generate USI for mlflow tracking
    import secrets
    global USI
    USI = secrets.token_hex(nbytes=2)
    logger.info('USI: ' + str(USI))

    logger.info("""setup(data={}, categorical_features={}, categorical_imputation={}, ordinal_features={}, high_cardinality_features={}, 
                    numeric_features={}, numeric_imputation={}, date_features={}, ignore_features={}, normalize={},
                    normalize_method={}, transformation={}, transformation_method={}, handle_unknown_categorical={}, unknown_categorical_method={}, pca={}, pca_method={},
                    pca_components={}, ignore_low_variance={}, combine_rare_levels={}, rare_level_threshold={}, bin_numeric_features={},
                    remove_multicollinearity={}, multicollinearity_threshold={}, group_features={},
                    group_names={}, supervised={}, supervised_target={}, n_jobs={}, html={}, session_id={}, log_experiment={},
                    experiment_name={}, log_plots={}, log_profile={}, log_data={}, silent={}, verbose={}, profile={})""".format(\
            str(data.shape), str(categorical_features), str(categorical_imputation), str(ordinal_features),\
            str(high_cardinality_features), str(numeric_features), str(numeric_imputation), str(date_features), str(ignore_features),\
            str(normalize), str(normalize_method), str(transformation), str(transformation_method), str(handle_unknown_categorical), str(unknown_categorical_method), str(pca),\
            str(pca_method), str(pca_components), str(ignore_low_variance), str(combine_rare_levels), str(rare_level_threshold), str(bin_numeric_features),\
            str(remove_multicollinearity), str(multicollinearity_threshold), str(group_features),str(group_names),str(supervised), str(supervised_target), str(n_jobs), str(html),\
            str(session_id),str(log_experiment), str(experiment_name), str(log_plots),str(log_profile), str(log_data), str(silent), str(verbose), str(profile)))

    #logging environment and libraries
    logger.info("Checking environment")
    
    from platform import python_version, platform, python_build, machine

    try:
        logger.info("python_version: " + str(python_version()))
    except:
        logger.warning("cannot find platform.python_version")

    try:
        logger.info("python_build: " + str(python_build()))
    except:
        logger.warning("cannot find platform.python_build")

    try:
        logger.info("machine: " + str(machine()))
    except:
        logger.warning("cannot find platform.machine")

    try:
        logger.info("platform: " + str(platform()))
    except:
        logger.warning("cannot find platform.platform")

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
        from pyod import __version__
        logger.info("pyod==" + str(__version__))
    except:
        logger.warning("pyod not found")

    try:
        from mlflow.version import VERSION
        import warnings
        warnings.filterwarnings('ignore') 
        logger.info("mlflow==" + str(VERSION))
    except:
        logger.warning("mlflow not found")

    logger.info("Checking Exceptions")

    #run_time
    import datetime, time
    runtime_start = time.time()

    """
    error handling starts here
    """
    
    #checking data type
    if hasattr(data,'shape') is False:
        sys.exit('(Type Error): data passed must be of type pandas.DataFrame')  

    #checking session_id
    if session_id is not None:
        if type(session_id) is not int:
            sys.exit('(Type Error): session_id parameter must be an integer.')  
            
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
        #data_cols = data_cols.drop(target)
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
        #data_cols = data_cols.drop(target)
        for i in high_cardinality_features:
            if i not in data_cols:
                sys.exit("(Value Error): Column type forced is either target column or doesn't exist in the dataset.")
                
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
        sys.exit("(Value Error): transformation_method param only accepts 'yeo-johnson' or 'quantile' ")        
    
    #handle unknown categorical
    if type(handle_unknown_categorical) is not bool:
        sys.exit('(Type Error): handle_unknown_categorical parameter only accepts True or False.')
        
    #unknown categorical method
    unknown_categorical_method_available = ['least_frequent', 'most_frequent']
    
    #forced type check
    all_cols = list(data.columns)
    
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
    
    #check pca
    if type(pca) is not bool:
        sys.exit('(Type Error): PCA parameter only accepts True or False.')
        
    #pca method check
    allowed_pca_methods = ['linear', 'kernel', 'incremental']
    if pca_method not in allowed_pca_methods:
        sys.exit("(Value Error): pca method param only accepts 'linear', 'kernel', or 'incremental'. ")    
    
    #pca components check
    if pca is True:
        if pca_method is not 'linear':
            if pca_components is not None:
                if(type(pca_components)) is not int:
                    sys.exit("(Type Error): pca_components parameter must be integer when pca_method is not 'linear'. ")

    #pca components check 2
    if pca is True:
        if pca_method is not 'linear':
            if pca_components is not None:
                if pca_components > len(data.columns):
                    sys.exit("(Type Error): pca_components parameter cannot be greater than original features space.")                
 
    #pca components check 3
    if pca is True:
        if pca_method is 'linear':
            if pca_components is not None:
                if type(pca_components) is not float:
                    if pca_components > len(data.columns): 
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
        
        for i in bin_numeric_features:
            if i not in all_cols:
                sys.exit("(Value Error): Column type forced is either target column or doesn't exist in the dataset.")
    
    #remove_multicollinearity
    if type(remove_multicollinearity) is not bool:
        sys.exit('(Type Error): remove_multicollinearity parameter only accepts True or False.')
        
    #multicollinearity_threshold
    if type(multicollinearity_threshold) is not float:
        sys.exit('(Type Error): multicollinearity_threshold must be a float between 0 and 1. ')  
    
    #group features
    if group_features is not None:
        if type(group_features) is not list:
            sys.exit('(Type Error): group_features must be of type list. ')     
    
    if group_names is not None:
        if type(group_names) is not list:
            sys.exit('(Type Error): group_names must be of type list. ')        

    #silent
    if type(silent) is not bool:
        sys.exit("(Type Error): silent parameter only accepts True or False. ")
        
    #html
    if type(html) is not bool:
        sys.exit('(Type Error): html parameter only accepts True or False.')

    #log_experiment
    if type(log_experiment) is not bool:
        sys.exit('(Type Error): log_experiment parameter only accepts True or False.')

    #log_plots
    if type(log_plots) is not bool:
        sys.exit('(Type Error): log_plots parameter only accepts True or False.')

    #log_data
    if type(log_data) is not bool:
        sys.exit('(Type Error): log_data parameter only accepts True or False.')

    #log_profile
    if type(log_profile) is not bool:
        sys.exit('(Type Error): log_profile parameter only accepts True or False.')
        
    
    """
    error handling ends here
    """
    
    logger.info("Preloading libraries")
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    import secrets
    
    #pandas option
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    
    #global html_param
    global html_param

    #create html_param
    html_param = html

    logger.info("Preparing display monitor")

    #progress bar
    max_steps = 4
        
    progress = ipw.IntProgress(value=0, min=0, max=max_steps, step=1 , description='Processing: ')
    
        
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ] ],
                             #['Step' , '. . . . . . . . . . . . . . . . . .',  'Step 0 of ' + str(total_steps)] ],
                              columns=['', ' ', '   ']).set_index('')
    
    if verbose:
        if html_param:
            display(progress)
            display(monitor, display_id = 'monitor')
    
    logger.info("Importing libraries")
    #general dependencies
    import numpy as np
    import pandas as pd
    import random
    
    #setting sklearn config to print all parameters including default
    import sklearn
    sklearn.set_config(print_changed_only=False)

    #define highlight function for function grid to display
    def highlight_max(s):
        is_max = s == True
        return ['background-color: lightgreen' if v else '' for v in is_max]
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    logger.info("Declaring global variables")
    #defining global variables
    global data_, X, seed, prep_pipe, prep_param, experiment__,\
        n_jobs_param, exp_name_log, logging_param, log_plots_param
    
    logger.info("Copying data for preprocessing")
    #copy original data for pandas profiler
    data_before_preprocess = data.copy()
    
    #copying data
    data_ = data.copy()
    
    #data without target
    if supervised:
        data_without_target = data.copy()
        data_without_target.drop(supervised_target, axis=1, inplace=True)
    
    if supervised:
        data_for_preprocess = data_without_target.copy()
    else:
        data_for_preprocess = data_.copy()
        
    #generate seed to be used globally
    if session_id is None:
        seed = random.randint(150,9000)
    else:
        seed = session_id    

    """
    preprocessing starts here
    """
    
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)

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
                pca_components_pass = int((len(data.columns))*0.5)
                
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
       
    #high cardinality
    if apply_ordinal_encoding_pass is True:
        ordinal_columns_and_categories_pass = ordinal_features
    else:
        ordinal_columns_and_categories_pass = {}
        
    if high_cardinality_features is not None:
        apply_cardinality_reduction_pass = True
    else:
        apply_cardinality_reduction_pass = False
        
    cardinal_method_pass = 'count'
        
    if apply_cardinality_reduction_pass:
        cardinal_features_pass = high_cardinality_features
    else:
        cardinal_features_pass = []
    
    #display dtypes
    if supervised is False:
        display_types_pass = True
    else:
        display_types_pass = False
    
    if silent:
        display_types_pass = False

    logger.info("Importing preprocessing module")

    #import library
    from pycaret import preprocess
    
    logger.info("Creating preprocessing pipeline")

    X = preprocess.Preprocess_Path_Two(train_data = data_for_preprocess, 
                                       categorical_features = cat_features_pass,
                                       apply_ordinal_encoding = apply_ordinal_encoding_pass, #new
                                       ordinal_columns_and_categories = ordinal_columns_and_categories_pass,
                                       apply_cardinality_reduction = apply_cardinality_reduction_pass, #latest
                                       cardinal_method = cardinal_method_pass, #latest
                                       cardinal_features = cardinal_features_pass, #latest
                                       numerical_features = numeric_features_pass,
                                       time_features = date_features_pass,
                                       features_todrop = ignore_features_pass,
                                       display_types = display_types_pass,
                                       numeric_imputation_strategy = numeric_imputation,
                                       categorical_imputation_strategy = categorical_imputation_pass,
                                       scale_data = normalize,
                                       scaling_method = normalize_method,
                                       Power_transform_data = transformation,
                                       Power_transform_method = trans_method_pass,
                                       apply_untrained_levels_treatment= handle_unknown_categorical, #new
                                       untrained_levels_treatment_method = unknown_categorical_method_pass, #new
                                       apply_pca = pca,
                                       pca_method = pca_method_pass, #new
                                       pca_variance_retained_or_number_of_components = pca_components_pass, #new
                                       apply_zero_nearZero_variance = ignore_low_variance, #new
                                       club_rare_levels = combine_rare_levels, #new
                                       rara_level_threshold_percentage = rare_level_threshold, #new
                                       apply_binning = apply_binning_pass, #new
                                       features_to_binn = features_to_bin_pass, #new
                                       remove_multicollinearity = remove_multicollinearity, #new
                                       maximum_correlation_between_features = multicollinearity_threshold, #new
                                       apply_grouping = apply_grouping_pass, #new
                                       features_to_group_ListofList = group_features_pass, #new
                                       group_name = group_names_pass, #new
                                       random_state = seed)
        
    progress.value += 1
    logger.info("Preprocessing pipeline created successfully")

    try:
        res_type = ['quit','Quit','exit','EXIT','q','Q','e','E','QUIT','Exit']
        res = preprocess.dtypes.response
        if res in res_type:
            sys.exit("(Process Exit): setup has been interupted with user command 'quit'. setup must rerun." )
    except:
        pass
    

    #save prep pipe
    prep_pipe = preprocess.pipe
    prep_param = preprocess
    
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

    if ordinal_features is not None:
        ordinal_features_grid = True
    else:
        ordinal_features_grid = False
    
    if remove_multicollinearity is False:
        multicollinearity_threshold_grid = None
    else:
        multicollinearity_threshold_grid = multicollinearity_threshold
     
    if group_features is not None:
        group_features_grid = True
    else:
        group_features_grid = False
     
    if high_cardinality_features is not None:
        high_cardinality_features_grid = True
    else:
        high_cardinality_features_grid = False

    learned_types = preprocess.dtypes.learent_dtypes
    #learned_types.drop(target, inplace=True)

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
    if supervised is False:
        experiment__ = []
    else:
        try:
            experiment__.append('dummy')
            experiment__.remove('dummy')
        except:
            experiment__ = []
    
    #create n_jobs_param
    n_jobs_param = n_jobs

    #create logging parameter
    logging_param = log_experiment

    #create exp_name_log param incase logging is False
    exp_name_log = 'no_logging'

    #create an empty log_plots_param
    if log_plots:
        log_plots_param = True
    else:
        log_plots_param = False

    progress.value += 1
    
    #monitor update
    monitor.iloc[1,1:] = 'Compiling Results'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
        
    '''
    Final display Starts
    '''
    
    shape = data.shape
    shape_transformed = X.shape
    
    if profile:
        if verbose:
            print('Setup Succesfully Completed! Loading Profile Now... Please Wait!')
    else:
        if verbose:
            print('Setup Succesfully Completed!')

    functions = pd.DataFrame ( [ ['session_id ', seed ],
                                 ['Original Data ', shape ],
                                 ['Missing Values ', missing_flag],
                                 ['Numeric Features ', str(float_type-1) ],
                                 ['Categorical Features ', str(cat_type) ],
                                 ['Ordinal Features ', ordinal_features_grid],
                                 ['High Cardinality Features ', high_cardinality_features_grid],
                                 ['Transformed Data ', shape_transformed ],
                                 ['Numeric Imputer ', numeric_imputation],
                                 ['Categorical Imputer ', categorical_imputation],
                                 ['Normalize ', normalize ],
                                 ['Normalize Method ', normalize_grid ],
                                 ['Transformation ', transformation ],
                                 ['Transformation Method ', transformation_grid ],
                                 ['PCA ', pca],
                                 ['PCA Method ', pca_method_grid],
                                 ['PCA components ', pca_components_grid],
                                 ['Ignore Low Variance ', ignore_low_variance],
                                 ['Combine Rare Levels ', combine_rare_levels],
                                 ['Rare Level Threshold ', rare_level_threshold_grid],
                                 ['Numeric Binning ', numeric_bin_grid],
                                 ['Remove Multicollinearity ', remove_multicollinearity],
                                 ['Multicollinearity Threshold ', multicollinearity_threshold_grid],
                                 ['Group Features ', group_features_grid],
                               ], columns = ['Description', 'Value'] )

    functions_ = functions.style.apply(highlight_max)
    
    progress.value += 1
    
    if verbose:
        if html_param:
            clear_output()
            print('Setup Succesfully Completed!')
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

    '''
    Final display Ends
    '''   
    
    #log into experiment
    if verbose:
        experiment__.append(('Anomaly Setup Config', functions))
        experiment__.append(('Orignal Dataset', data_))
        experiment__.append(('Transformed Dataset', X))
        experiment__.append(('Transformation Pipeline', prep_pipe))
    
    
    #end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    if logging_param:
        
        logger.info("Logging experiment in MLFlow")

        import mlflow
        from pathlib import Path
        import os

        if experiment_name is None:
            exp_name_ = 'ano-default-name'
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
            size_bytes = Path('Transformation Pipeline.pkl').stat().st_size
            size_kb = np.round(size_bytes/1000, 2)
            mlflow.set_tag("Size KB", size_kb)
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
                data_before_preprocess.to_csv('data.csv')
                mlflow.log_artifact('data.csv')
                os.remove('data.csv')

            # Log input.txt that contains name of columns required in dataset 
            # to use this pipeline based on USI/URI.

            input_cols = list(data_before_preprocess.columns)

            with open("input.txt", "w") as output:
                output.write(str(input_cols))
            
            mlflow.log_artifact("input.txt")
            os.remove('input.txt')

    logger.info(str(prep_pipe))
    logger.info("setup() succesfully completed......................................")

    return X, data_, seed, prep_pipe, prep_param, experiment__,\
        n_jobs_param, html_param, exp_name_log, logging_param, log_plots_param, USI

def create_model(model = None, 
                 fraction = 0.05,
                 verbose = True,
                 system=True, #added in pycaret==2.0.0
                 **kwargs): #added in pycaret==2.0.0
    
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
    model : string / object, default = None
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

    try:
        logger.info("Initializing create_model()")
        logger.info("""create_model(model={}, fraction={}, verbose={}, system={})""".\
        format(str(model), str(fraction), str(verbose), str(system)))

        logger.info("Checking exceptions")
    
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
        logger.info("Checking exceptions")

    #exception checking   
    import sys        

    #run_time
    import datetime, time
    runtime_start = time.time()

    #ignore warings
    import warnings
    warnings.filterwarnings('ignore') 
    
    """
    error handling starts here
    """
    
    #checking for model parameter
    if model is None:
        sys.exit('(Value Error): Model parameter Missing. Please see docstring for list of available models.')
        
    #checking for allowed models
    allowed_models = ['abod', 'iforest', 'cluster', 'cof', 'histogram', 'knn', 'lof', 'svm', 'pca', 'mcd', 'sod', 'sos']
    
    if model not in allowed_models:
        sys.exit('(Value Error): Model Not Available. Please see docstring for list of available models.')
        
    #checking fraction type:
    if fraction <= 0 or fraction >= 1:
        sys.exit('(Type Error): Fraction parameter can only take value as float between 0 to 1.')
        
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.') 
        
    """
    error handling ends here
    """
    
    logger.info("Preloading libraries")

    #pre-load libraries
    import pandas as pd
    import numpy as np
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    
    """
    monitor starts
    """
    
    logger.info("Preparing display monitor")

    #progress bar and monitor control    
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    progress = ipw.IntProgress(value=0, min=0, max=4, step=1 , description='Processing: ')
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                              ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Initializing'] ],
                              columns=['', ' ', '  ']).set_index('')
    if verbose:
        if html_param:
            display(progress)
            display(monitor, display_id = 'monitor')
        
    progress.value += 1
    
    """
    monitor ends
    """
    
    #monitor update
    monitor.iloc[1,1:] = 'Importing the Model'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    progress.value += 1
    
    #create model
    logger.info("Importing untrained model")

    if model == 'abod':
        from pyod.models.abod import ABOD
        model = ABOD(contamination=fraction, **kwargs)
        full_name = 'Angle-base Outlier Detection'
        
    elif model == 'cluster':
        from pyod.models.cblof import CBLOF
        try:
            model = CBLOF(contamination=fraction, n_clusters=8, random_state=seed, **kwargs)
            model.fit(X)
        except:
            try:
                model = CBLOF(contamination=fraction, n_clusters=12, random_state=seed, **kwargs)
                model.fit(X)
            except:
                sys.exit("(Type Error) Could not form valid cluster separation")
                
        full_name = 'Clustering-Based Local Outlier'
        
    elif model == 'cof':
        from pyod.models.cof import COF
        model = COF(contamination=fraction, **kwargs)        
        full_name = 'Connectivity-Based Outlier Factor'
        
    elif model == 'iforest':
        from pyod.models.iforest import IForest
        model = IForest(contamination=fraction, behaviour = 'new', random_state=seed, **kwargs)    
        full_name = 'Isolation Forest'
        
    elif model == 'histogram':
        from pyod.models.hbos import HBOS
        model = HBOS(contamination=fraction, **kwargs) 
        full_name = 'Histogram-based Outlier Detection'
        
    elif model == 'knn':
        from pyod.models.knn import KNN
        model = KNN(contamination=fraction, **kwargs)  
        full_name = 'k-Nearest Neighbors Detector'
        
    elif model == 'lof':
        from pyod.models.lof import LOF
        model = LOF(contamination=fraction, **kwargs)
        full_name = 'Local Outlier Factor'
        
    elif model == 'svm':
        from pyod.models.ocsvm import OCSVM
        model = OCSVM(contamination=fraction, **kwargs)
        full_name = 'One-class SVM detector'
        
    elif model == 'pca':
        from pyod.models.pca import PCA
        model = PCA(contamination=fraction, random_state=seed, **kwargs)  
        full_name = 'Principal Component Analysis'
        
    elif model == 'mcd':
        from pyod.models.mcd import MCD
        model = MCD(contamination=fraction, random_state=seed, **kwargs)
        full_name = 'Minimum Covariance Determinant'
        
    elif model == 'sod':
        from pyod.models.sod import SOD
        model = SOD(contamination=fraction, **kwargs)         
        full_name = 'Subspace Outlier Detection'
        
    elif model == 'sos':
        from pyod.models.sos import SOS
        model = SOS(contamination=fraction, **kwargs)   
        full_name = 'Stochastic Outlier Selection'
    
    else:    
        def get_model_name(e):
            return str(e).split("(")[0]

        model == model
        full_name = get_model_name(model)
        
    logger.info(str(full_name) + ' Imported succesfully')

    #monitor update
    monitor.iloc[1,1:] = 'Fitting the Model'
    progress.value += 1
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
        
    #fitting the model
    model_fit_start = time.time()
    logger.info("Fitting Model")
    model.fit(X)
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
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        #import mlflow
        import mlflow
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

            # Log AUC and Confusion Matrix plot
            if log_plots_param:

                logger.info("SubProcess plot_model() called ==================================")

                try:
                    plot_model(model, plot = 'tsne', save=True, system=False)
                    mlflow.log_artifact('TSNE.html')
                    os.remove("TSNE.html")
                except:
                    pass
                    
                logger.info("SubProcess plot_model() end ==================================")

            # Log model and transformation pipeline
            logger.info("SubProcess save_model() called ==================================")
            save_model(model, 'Trained Model', verbose=False)
            logger.info("SubProcess save_model() end ==================================")
            mlflow.log_artifact('Trained Model' + '.pkl')
            size_bytes = Path('Trained Model.pkl').stat().st_size
            size_kb = np.round(size_bytes/1000, 2)
            mlflow.set_tag("Size KB", size_kb)
            os.remove('Trained Model.pkl')

    progress.value += 1
    
    if verbose:
        clear_output()

    logger.info(str(model))
    logger.info("create_models() succesfully completed......................................")

    return model

def assign_model(model,
                 transformation=False,
                 score=True,
                 verbose=True):
    
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
        is fitted. If set to False, it will only return the flag (1 = outlier, 0 = inlier).

    verbose: Boolean, default = True
        Status update is not printed when verbose is set to False.

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with inferred outliers using a trained model.
  
    """
    
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
        
    logger.info("Initializing assign_model()")
    logger.info("""assign_model(model={}, transformation={}, score={}, verbose={})""".\
        format(str(model), str(transformation), str(score), str(verbose)))

    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    
    """
    error handling starts here
    """
    
    #determine model type and store in string
    mod_type = str(type(model))
    
    #checking for allowed models
    allowed_type = ['pyod']
    if 'pyod' not in mod_type:
        sys.exit('(Value Error): Model Not Recognized. Please see docstring for list of available models.')
    
    #checking transformation parameter
    if type(transformation) is not bool:
        sys.exit('(Type Error): Transformation parameter can only take argument as True or False.')   
        
    #checking verbose parameter
    if type(score) is not bool:
        sys.exit('(Type Error): Score parameter can only take argument as True or False.')    
        
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.')     
    
    
    """
    error handling ends here
    """
    
    logger.info("Preloading libraries")
    #pre-load libraries
    import numpy as np
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    
    logger.info("Copying data")
    #copy data_
    if transformation:
        data__ = X.copy()
    else:
        data__ = data_.copy()
    
    logger.info("Preparing display monitor")
    #progress bar and monitor control 
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    progress = ipw.IntProgress(value=0, min=0, max=3, step=1 , description='Processing: ')
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                              ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Initializing'] ],
                              columns=['', ' ', '  ']).set_index('')
    if verbose:
        if html_param:
            display(progress)
            display(monitor, display_id = 'monitor')
        
    progress.value += 1
    
    monitor.iloc[1,1:] = 'Inferring Outliers from Model'
    
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    progress.value += 1
    
    #calculation labels and attaching to dataframe
    pred_labels = model.labels_
    data__['Label'] = pred_labels
    
    progress.value += 1
    
    #calculating score and attaching to dataframe
    if score:
        pred_score = model.decision_scores_
        data__['Score'] = pred_score
    
    progress.value += 1

    logger.info("Determining Trained Model")

    mod_type = str(model).split("(")[0]
    
    if 'ABOD' in mod_type:
        name_ = 'Angle-base Outlier Detection' 
        
    elif 'IForest' in mod_type:
        name_ = 'Isolation Forest'
        
    elif 'CBLOF' in mod_type:
        name_ = 'Clustering-Based Local Outlier'        
        
    elif 'COF' in mod_type:
        name_ = 'Connectivity-Based Outlier Factor'
        
    elif 'HBOS' in mod_type:
        name_ = 'Histogram-based Outlier Detection'
        
    elif 'KNN' in mod_type:
        name_ = 'k-Nearest Neighbors Detector'
        
    elif 'LOF' in mod_type:
        name_ = 'Local Outlier Factor'
        
    elif 'OCSVM' in mod_type:
        name_ = 'One-class SVM detector'
        
    elif 'PCA' in mod_type:
        name_ = 'Principal Component Analysis'
    
    elif 'MCD' in mod_type:
        name_ = 'Minimum Covariance Determinant'
        
    elif 'SOD' in mod_type:
        name_ = 'Subspace Outlier Detection'
        
    elif 'SOS' in mod_type:
        name_ = 'Stochastic Outlier Selection'
        
    else:
        name_ = 'Unknown Anomaly Detector'
        
    name_ = 'Assigned ' + str(name_)

    logger.info("Trained Model : " + str(name_))
    
    if verbose:
        clear_output()

    logger.info(str(data__.shape))
    logger.info("assign_model() succesfully completed......................................")

    return data__

def tune_model(model=None,
               supervised_target=None,
               method='drop',
               estimator=None,
               optimize=None,
               custom_grid = None, #added in pycaret 2.0.0
               fold=10,
               verbose=True): #added in pycaret 2.0.0
    
    
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
    model : string, default = None
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
    
    method: string, default = 'drop'
        When method set to drop, it will drop the outlier rows from training dataset 
        of supervised estimator, when method set to 'surrogate', it will use the
        decision function and label as a feature without dropping the outliers from
        training dataset.
    
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
        
        If set to None, Linear model is used by default for both classification
        and regression tasks.
    
    optimize: string, default = None
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
    
    global data_, X
    
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
    logger.info("""tune_model(model={}, supervised_target={}, method={}, estimator={}, optimize={}, custom_grid={}, fold={}, verbose={})""".\
        format(str(model), str(supervised_target), str(method), str(estimator), str(optimize), str(custom_grid), str(fold), str(verbose)))

    logger.info("Checking exceptions")

    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    import sys
    
    #run_time
    import datetime, time
    runtime_start = time.time()

    #checking for model parameter
    if model is None:
        sys.exit('(Value Error): Model parameter Missing. Please see docstring for list of available models.')
        
    #checking for allowed models
    allowed_models = ['abod', 'iforest', 'cluster', 'cof', 'histogram', 'knn', 'lof', 'svm', 'pca', 'mcd', 'sod', 'sos']
    
    if model not in allowed_models:
        sys.exit('(Value Error): Model Not Available for Tuning. Please see docstring for list of available models.')
    
    #check method
    allowed_methods = ['drop', 'surrogate']
    if method not in allowed_methods:
        sys.exit('(Value Error): Method not recognized. See docstring for list of available methods.')
        
    #check if supervised target is None:
    if supervised_target is None:
        sys.exit('(Value Error): supervised_target cannot be None. A column name must be given for estimator.')
    
    #check supervised target
    if supervised_target is not None:
        all_col = list(data_.columns)
        if supervised_target not in all_col:
            sys.exit('(Value Error): supervised_target not recognized. It can only be one of the following: ' + str(all_col))
    
    #checking estimator:
    if estimator is not None:
        
        available_estimators = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 
                            'gbc', 'lda', 'et', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 
                            'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 
                            'mlp', 'xgboost', 'lightgbm', 'catboost']
                
        if estimator not in available_estimators:
            sys.exit('(Value Error): Estimator Not Available. Please see docstring for list of available estimators.')
    
    
    #checking optimize parameter
    if optimize is not None:
        
        available_optimizers = ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE', 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa']
        
        if optimize not in available_optimizers:
            sys.exit('(Value Error): optimize parameter Not Available. Please see docstring for list of available parameters.')
    
    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    
    
    """
    exception handling ends here
    """
    
    logger.info("Preloading libraries")

    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from ipywidgets import Output
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    
    logger.info("Preparing display monitor")

    #progress bar
    if custom_grid is None:
        max_steps = 25
    else:
        max_steps = 15 + len(custom_grid)
        
    progress = ipw.IntProgress(value=0, min=0, max=max_steps, step=1 , description='Processing: ')
    
    if verbose:
        if html_param:
            display(progress)
    
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies'],
                             ['Step' , '. . . . . . . . . . . . . . . . . .',  'Initializing' ] ],
                              columns=['', ' ', '   ']).set_index('')
    
    monitor_out = Output()
    
    if verbose:
        if html_param:
            display(monitor_out)
            with monitor_out:
                display(monitor, display_id = 'monitor')

    logger.info("Importing libraries")

    #General Dependencies
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn import metrics
    import numpy as np
    import plotly.express as px
    from copy import deepcopy
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    logger.info("Copying environment variables")

    a = data_.copy()
    b = X.copy()
    c = deepcopy(prep_pipe)
    e = exp_name_log
    z = logging_param
    
    def retain_original(a,b,c,e,z):
        
        global data_, X, prep_pipe, exp_name_log, logging_param
        
        data_ = a.copy()
        X = b.copy()
        prep_pipe = deepcopy(c)
        exp_name_log = e
        logging_param = z

        return data_, X, prep_pipe, exp_name_log, logging_param
            
    #setting up cufflinks
    import cufflinks as cf
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)
    
    progress.value += 1 
    
    #define the problem
    if data_[supervised_target].value_counts().count() == 2: 
        problem = 'classification'
        logger.info("Objective : Classification")
    else:
        problem = 'regression'    
        logger.info("Objective : Regression")
    
    #define model name
    
    logger.info("Defining Model Name")    

    if model == 'abod':
        model_name = 'Angle-base Outlier Detection'  
    elif model == 'iforest':
        model_name = 'Isolation Forest'        
    elif model == 'cluster':
        model_name = 'Clustering-Based Local Outlier'   
    elif model == 'cof':
        model_name = 'Connectivity-Based Outlier Factor'   
    elif model == 'histogram':
        model_name = 'Histogram-based Outlier Detection'   
    elif model == 'knn':
        model_name = 'k-Nearest Neighbors Detector'   
    elif model == 'lof':
        model_name = 'Local Outlier Factor'   
    elif model == 'svm':
        model_name = 'One-class SVM detector'   
    elif model == 'pca':
        model_name = 'Principal Component Analysis'   
    elif model == 'mcd':
        model_name = 'Minimum Covariance Determinant'   
    elif model == 'sod':
        model_name = 'Subspace Outlier Detection'   
    elif model == 'sos':
        model_name = 'Stochastic Outlier Selection'   
    
    logger.info("Defining Supervised Estimator")

    #defining estimator:
    if problem == 'classification' and estimator is None:
        estimator = 'lr'
    elif problem == 'regression' and estimator is None:
        estimator = 'lr'        
    else:
        estimator = estimator
    
    logger.info("Defining Optimizer")

    #defining optimizer:
    if optimize is None and problem == 'classification':
        optimize = 'Accuracy'
    elif optimize is None and problem == 'regression':
        optimize = 'R2'
    else:
        optimize=optimize
    
    logger.info("Optimize: " + str(optimize))

    progress.value += 1 
            
    #defining tuning grid
    logger.info("Defining Tuning Grid")

    if custom_grid is not None:
        
        logger.info("Custom Grid used")
        param_grid = custom_grid
        param_grid_with_zero = [0]

        for i in param_grid:
            param_grid_with_zero.append(i)

    else:
        
        logger.info("Pre-defined Grid used")
        param_grid_with_zero = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10] 
        param_grid = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10] 
    
    master = []; master_df = []
    
    monitor.iloc[1,1:] = 'Creating Outlier Detection Model'
    if verbose:
        if html_param:
            update_display(monitor, display_id = 'monitor')
    
    """
    preprocess starts here
    """
    
    logger.info("Defining setup variables for preprocessing")

    #removing target variable from data by defining new setup
    _data_ = data_.copy()
    target_ = pd.DataFrame(_data_[supervised_target])
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    target_ = le.fit_transform(target_)
    
    cat_pass = prep_param.dtypes.categorical_features
    num_pass = prep_param.dtypes.numerical_features
    time_pass = prep_param.dtypes.time_features
    ignore_pass = prep_param.dtypes.features_todrop
    
    #PCA
    if 'Empty' in str(prep_param.pca): 
        pca_pass = False
        pca_method_pass = 'linear'
    
    else:
        pca_pass = True
        
        if prep_param.pca.method == 'pca_liner':
            pca_method_pass = 'linear'
        elif prep_param.pca.method == 'pca_kernal':
            pca_method_pass = 'kernel'
        elif prep_param.pca.method == 'incremental':
            pca_method_pass = 'incremental'
        
    if pca_pass is True:
        pca_comp_pass = prep_param.pca.variance_retained
    else:
        pca_comp_pass = 0.99
    
    #IMPUTATION
    if 'not_available' in prep_param.imputer.categorical_strategy:
        cat_impute_pass = 'constant'
    elif 'most frequent' in prep_param.imputer.categorical_strategy:
        cat_impute_pass = 'mode'
    
    num_impute_pass = prep_param.imputer.numeric_strategy
    
    #NORMALIZE
    if 'Empty' in str(prep_param.scaling):
        normalize_pass = False
    else:
        normalize_pass = True
        
    if normalize_pass is True:
        normalize_method_pass = prep_param.scaling.function_to_apply
    else:
        normalize_method_pass = 'zscore'
    
    #FEATURE TRANSFORMATION
    if 'Empty' in str(prep_param.P_transform):
        transformation_pass = False
    else:
        transformation_pass = True
        
    if transformation_pass is True:
        
        if 'yj' in prep_param.P_transform.function_to_apply:
            transformation_method_pass = 'yeo-johnson'
        elif 'quantile' in prep_param.P_transform.function_to_apply:
            transformation_method_pass = 'quantile'
            
    else:
        transformation_method_pass = 'yeo-johnson'
    
    #BIN NUMERIC FEATURES
    if 'Empty' in str(prep_param.binn):
        features_to_bin_pass = []
        apply_binning_pass = False
        
    else:
        features_to_bin_pass = prep_param.binn.features_to_discretize
        apply_binning_pass = True
    
    #COMBINE RARE LEVELS
    if 'Empty' in str(prep_param.club_R_L):
        combine_rare_levels_pass = False
        combine_rare_threshold_pass = 0.1
    else:
        combine_rare_levels_pass = True
        combine_rare_threshold_pass = prep_param.club_R_L.threshold
        
    #ZERO NERO ZERO VARIANCE
    if 'Empty' in str(prep_param.znz):
        ignore_low_variance_pass = False
    else:
        ignore_low_variance_pass = True
    
    #MULTI-COLLINEARITY
    if 'Empty' in str(prep_param.fix_multi):
        remove_multicollinearity_pass = False
    else:
        remove_multicollinearity_pass = True
        
    if remove_multicollinearity_pass is True:
        multicollinearity_threshold_pass = prep_param.fix_multi.threshold
    else:
        multicollinearity_threshold_pass = 0.9
    
    #UNKNOWN CATEGORICAL LEVEL
    if 'Empty' in str(prep_param.new_levels):
        handle_unknown_categorical_pass = False
    else:
        handle_unknown_categorical_pass = True
        
    if handle_unknown_categorical_pass is True:
        unknown_level_preprocess = prep_param.new_levels.replacement_strategy
        if unknown_level_preprocess == 'least frequent':
            unknown_categorical_method_pass = 'least_frequent'
        elif unknown_level_preprocess == 'most frequent':
            unknown_categorical_method_pass = 'most_frequent'
        else:
            unknown_categorical_method_pass = 'least_frequent'
    else:
        unknown_categorical_method_pass = 'least_frequent'
    
    #GROUP FEATURES
    if 'Empty' in str(prep_param.group):
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
    
    #ORDINAL FEATURES    
    if 'Empty' in str(prep_param.ordinal):
        ordinal_features_pass = None
    else:
        ordinal_features_pass = prep_param.ordinal.info_as_dict
    
    #HIGH CARDINALITY    
    if 'Empty' in str(prep_param.cardinality):
        high_cardinality_features_pass = None
    else:
        high_cardinality_features_pass = prep_param.cardinality.feature
        
    global setup_without_target
    
    logger.info("SubProcess setup() called")

    setup_without_target = setup(data = data_,
                                 categorical_features = cat_pass,
                                 categorical_imputation = cat_impute_pass,
                                 ordinal_features = ordinal_features_pass, #new
                                 high_cardinality_features = high_cardinality_features_pass, #latest
                                 numeric_features = num_pass,
                                 numeric_imputation = num_impute_pass,
                                 date_features = time_pass,
                                 ignore_features = ignore_pass,
                                 normalize = normalize_pass,
                                 normalize_method = normalize_method_pass,
                                 transformation = transformation_pass,
                                 transformation_method = transformation_method_pass,
                                 handle_unknown_categorical = handle_unknown_categorical_pass, 
                                 unknown_categorical_method = unknown_categorical_method_pass, 
                                 pca = pca_pass,
                                 pca_components = pca_comp_pass, 
                                 pca_method = pca_method_pass, 
                                 ignore_low_variance = ignore_low_variance_pass, 
                                 combine_rare_levels = combine_rare_levels_pass, 
                                 rare_level_threshold = combine_rare_threshold_pass,
                                 bin_numeric_features = features_to_bin_pass, 
                                 remove_multicollinearity = remove_multicollinearity_pass, 
                                 multicollinearity_threshold = multicollinearity_threshold_pass, 
                                 group_features = group_features_pass,
                                 group_names = group_names_pass,
                                 supervised = True,
                                 supervised_target = supervised_target,
                                 session_id = seed,
                                 log_experiment = False, #added in pycaret==2.0.0
                                 profile=False,
                                 verbose=False)
    
    data_without_target = setup_without_target[0]
    
    logger.info("SubProcess setup() end")

    """
    preprocess ends here
    """
    
    #adding dummy model in master
    master.append('No Model Required')
    master_df.append('No Model Required')
    
    model_fit_time_list = []

    for i in param_grid:
        logger.info("Fitting Model with Fraction = " +str(i))
        progress.value += 1                      
        monitor.iloc[2,1:] = 'Fitting Model With ' + str(i) + ' Fraction'
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')
                             
        #create and assign the model to dataset d
        model_fit_start = time.time()
        logger.info("SubProcess create_model() called==================================")
        m = create_model(model=model, fraction=i, verbose=False, system=False)
        logger.info("SubProcess create_model() end==================================")
        model_fit_end = time.time()
        model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
        model_fit_time_list.append(model_fit_time)

        logger.info("Generating labels")
        logger.info("SubProcess assign_model() called==================================")
        d = assign_model(m, transformation=True, score=True, verbose=False)
        logger.info("SubProcess assign_model() ends==================================")
        d[str(supervised_target)] = target_

        master.append(m)
        master_df.append(d)

        
    #attaching target variable back
    data_[str(supervised_target)] = target_

    logger.info("Defining Supervised Estimator")
    
    if problem == 'classification':
        
        logger.info("Problem : Classification")

        """
        
        defining estimator
        
        """
        
        monitor.iloc[1,1:] = 'Evaluating Anomaly Model'
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')
                             
        if estimator == 'lr':

            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=seed)
            full_name = 'Logistic Regression'

        elif estimator == 'knn':

            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier()
            full_name = 'K Nearest Neighbours'

        elif estimator == 'nb':

            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            full_name = 'Naive Bayes'

        elif estimator == 'dt':

            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(random_state=seed)
            full_name = 'Decision Tree'

        elif estimator == 'svm':

            from sklearn.linear_model import SGDClassifier
            model = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed)
            full_name = 'Support Vector Machine'

        elif estimator == 'rbfsvm':

            from sklearn.svm import SVC
            model = SVC(gamma='auto', C=1, probability=True, kernel='rbf', random_state=seed)
            full_name = 'RBF SVM'

        elif estimator == 'gpc':

            from sklearn.gaussian_process import GaussianProcessClassifier
            model = GaussianProcessClassifier(random_state=seed)
            full_name = 'Gaussian Process Classifier'

        elif estimator == 'mlp':

            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(max_iter=500, random_state=seed)
            full_name = 'Multi Level Perceptron'    

        elif estimator == 'ridge':

            from sklearn.linear_model import RidgeClassifier
            model = RidgeClassifier(random_state=seed)
            full_name = 'Ridge Classifier'        

        elif estimator == 'rf':

            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=seed)
            full_name = 'Random Forest Classifier'    

        elif estimator == 'qda':

            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            model = QuadraticDiscriminantAnalysis()
            full_name = 'Quadratic Discriminant Analysis' 

        elif estimator == 'ada':

            from sklearn.ensemble import AdaBoostClassifier
            model = AdaBoostClassifier(random_state=seed)
            full_name = 'AdaBoost Classifier'        

        elif estimator == 'gbc':

            from sklearn.ensemble import GradientBoostingClassifier    
            model = GradientBoostingClassifier(random_state=seed)
            full_name = 'Gradient Boosting Classifier'    

        elif estimator == 'lda':

            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis()
            full_name = 'Linear Discriminant Analysis'

        elif estimator == 'et':

            from sklearn.ensemble import ExtraTreesClassifier 
            model = ExtraTreesClassifier(random_state=seed)
            full_name = 'Extra Trees Classifier'
            
        elif estimator == 'xgboost':
            
            from xgboost import XGBClassifier
            model = XGBClassifier(random_state=seed, n_jobs=-1, verbosity=0)
            full_name = 'Extreme Gradient Boosting'
            
        elif estimator == 'lightgbm':
            
            import lightgbm as lgb
            model = lgb.LGBMClassifier(random_state=seed)
            full_name = 'Light Gradient Boosting Machine'
            
        elif estimator == 'catboost':
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(random_state=seed, silent=True) # Silent is True to suppress CatBoost iteration results 
            full_name = 'CatBoost Classifier'
        
        logger.info(str(full_name) + " Imported Successfully")

        progress.value += 1 
        
        """
        start model building here

        """
                             
        logger.info("Creating Classifier without Anomaly")         
        acc = [];  auc = []; recall = []; prec = []; kappa = []; f1 = []
        
        #build model without anomaly
        monitor.iloc[2,1:] = 'Evaluating Classifier Without Anomaly Detector'
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')   

        d = master_df[1].copy()
        d.drop(['Label', 'Score'], axis=1, inplace=True)

        #drop NA's caution
        d.dropna(axis=0, inplace=True)
        
        #get_dummies to caste categorical variables for supervised learning 
        d = pd.get_dummies(d)

        #split the dataset
        X = d.drop(supervised_target, axis=1)
        y = d[supervised_target]

        #fit the model
        logger.info("Fitting Model") 
        model.fit(X,y)
        
        #generate the prediction and evaluate metric
        logger.info("Evaluating Cross Val Predictions") 
        pred = cross_val_predict(model,X,y,cv=fold, method = 'predict')

        acc_ = metrics.accuracy_score(y,pred)
        acc.append(acc_)

        recall_ = metrics.recall_score(y,pred)
        recall.append(recall_)

        precision_ = metrics.precision_score(y,pred)
        prec.append(precision_)

        kappa_ = metrics.cohen_kappa_score(y,pred)
        kappa.append(kappa_)

        f1_ = metrics.f1_score(y,pred)
        f1.append(f1_)
        
        if hasattr(model,'predict_proba'):
            pred_ = cross_val_predict(model,X,y,cv=fold, method = 'predict_proba')
            pred_prob = pred_[:,1]
            auc_ = metrics.roc_auc_score(y,pred_prob)
            auc.append(auc_)

        else:
            auc.append(0)

        for i in range(1,len(master_df)):
            progress.value += 1 
            param_grid_val = param_grid[i-1]
            
            logger.info("Creating Classifier with Fraction = " + str(param_grid_val))

            monitor.iloc[2,1:] = 'Evaluating Classifier With ' + str(param_grid_val) + ' Fraction'
            if verbose:
                if html_param:
                    update_display(monitor, display_id = 'monitor')                
                             
            #prepare the dataset for supervised problem
            d = master_df[i]
            
            #cleaning the dataframe for supervised learning
            d.dropna(axis=0, inplace=True)
            Score_ = pd.DataFrame(d['Score'])
            Score = scaler.fit_transform(Score_)
            d['Score'] = Score

            if method == 'drop':
                d = d[d['Label'] == 0]
                d.drop(['Label'], axis=1, inplace=True)
                
            #get_dummies to caste categorical variables for supervised learning 
            d = pd.get_dummies(d)

            #split the dataset
            X = d.drop(supervised_target, axis=1)
            y = d[supervised_target]

            #fit the model
            logger.info("Fitting Model") 
            model.fit(X,y)

            #generate the prediction and evaluate metric
            logger.info("Generating Cross Val Predictions") 
            pred = cross_val_predict(model,X,y,cv=fold, method = 'predict')

            acc_ = metrics.accuracy_score(y,pred)
            acc.append(acc_)

            recall_ = metrics.recall_score(y,pred)
            recall.append(recall_)

            precision_ = metrics.precision_score(y,pred)
            prec.append(precision_)

            kappa_ = metrics.cohen_kappa_score(y,pred)
            kappa.append(kappa_)

            f1_ = metrics.f1_score(y,pred)
            f1.append(f1_)

            if hasattr(model,'predict_proba'):
                pred_ = cross_val_predict(model,X,y,cv=fold, method = 'predict_proba')
                pred_prob = pred_[:,1]
                auc_ = metrics.roc_auc_score(y,pred_prob)
                auc.append(auc_)

            else:
                auc.append(0)

                             
        monitor.iloc[1,1:] = 'Compiling Results'
        monitor.iloc[1,1:] = 'Finalizing'
        
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        logger.info("Creating metrics dataframe")                        
        df = pd.DataFrame({'Fraction %': param_grid_with_zero, 'Accuracy' : acc, 'AUC' : auc, 'Recall' : recall, 
                   'Precision' : prec, 'F1' : f1, 'Kappa' : kappa})
        
        sorted_df = df.sort_values(by=optimize, ascending=False)
        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]
        best_model_tt = model_fit_time_list[ival]

        progress.value += 1 
        logger.info("Rendering Visual")
        sd = pd.melt(df, id_vars=['Fraction %'], value_vars=['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa'], 
                     var_name='Metric', value_name='Score')

        fig = px.line(sd, x='Fraction %', y='Score', color='Metric', line_shape='linear', range_y = [0,1])
        fig.update_layout(plot_bgcolor='rgb(245,245,245)')
        title= str(full_name) + ' Metrics and Fraction %'
        fig.update_layout(title={'text': title, 'y':0.95,'x':0.45,'xanchor': 'center','yanchor': 'top'})
        
        fig.show()
        logger.info("Visual Rendered Successfully")
        
        #monitor = ''
        
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')
        
        if verbose:
            if html_param:
                monitor_out.clear_output()
                progress.close()
        
        best_k = np.array(sorted_df.head(1)['Fraction %'])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0],4)
        p = 'Best Model: ' + model_name + ' |' + ' Fraction %: ' + str(best_k) + ' | ' + str(optimize) + ' : ' + str(best_m)
        print(p)

    elif problem == 'regression':
        
        logger.info("Problem : Regression")

        """
        
        defining estimator
        
        """
        
        monitor.iloc[1,1:] = 'Evaluating Anomaly Model'
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')
                                    
        if estimator == 'lr':
        
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            full_name = 'Linear Regression'
        
        elif estimator == 'lasso':

            from sklearn.linear_model import Lasso
            model = Lasso(random_state=seed)
            full_name = 'Lasso Regression'

        elif estimator == 'ridge':

            from sklearn.linear_model import Ridge
            model = Ridge(random_state=seed)
            full_name = 'Ridge Regression'

        elif estimator == 'en':

            from sklearn.linear_model import ElasticNet
            model = ElasticNet(random_state=seed)
            full_name = 'Elastic Net'

        elif estimator == 'lar':

            from sklearn.linear_model import Lars
            model = Lars()
            full_name = 'Least Angle Regression'

        elif estimator == 'llar':

            from sklearn.linear_model import LassoLars
            model = LassoLars()
            full_name = 'Lasso Least Angle Regression'

        elif estimator == 'omp':

            from sklearn.linear_model import OrthogonalMatchingPursuit
            model = OrthogonalMatchingPursuit()
            full_name = 'Orthogonal Matching Pursuit'

        elif estimator == 'br':
            from sklearn.linear_model import BayesianRidge
            model = BayesianRidge()
            full_name = 'Bayesian Ridge Regression' 

        elif estimator == 'ard':

            from sklearn.linear_model import ARDRegression
            model = ARDRegression()
            full_name = 'Automatic Relevance Determination'        

        elif estimator == 'par':

            from sklearn.linear_model import PassiveAggressiveRegressor
            model = PassiveAggressiveRegressor(random_state=seed)
            full_name = 'Passive Aggressive Regressor'    

        elif estimator == 'ransac':

            from sklearn.linear_model import RANSACRegressor
            model = RANSACRegressor(random_state=seed)
            full_name = 'Random Sample Consensus'   

        elif estimator == 'tr':

            from sklearn.linear_model import TheilSenRegressor
            model = TheilSenRegressor(random_state=seed)
            full_name = 'TheilSen Regressor'     

        elif estimator == 'huber':

            from sklearn.linear_model import HuberRegressor
            model = HuberRegressor()
            full_name = 'Huber Regressor'   

        elif estimator == 'kr':

            from sklearn.kernel_ridge import KernelRidge
            model = KernelRidge()
            full_name = 'Kernel Ridge'

        elif estimator == 'svm':

            from sklearn.svm import SVR
            model = SVR()
            full_name = 'Support Vector Regression'  

        elif estimator == 'knn':

            from sklearn.neighbors import KNeighborsRegressor
            model = KNeighborsRegressor()
            full_name = 'Nearest Neighbors Regression' 

        elif estimator == 'dt':

            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor(random_state=seed)
            full_name = 'Decision Tree Regressor'

        elif estimator == 'rf':

            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=seed)
            full_name = 'Random Forest Regressor'

        elif estimator == 'et':

            from sklearn.ensemble import ExtraTreesRegressor
            model = ExtraTreesRegressor(random_state=seed)
            full_name = 'Extra Trees Regressor'    

        elif estimator == 'ada':

            from sklearn.ensemble import AdaBoostRegressor
            model = AdaBoostRegressor(random_state=seed)
            full_name = 'AdaBoost Regressor'   

        elif estimator == 'gbr':

            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(random_state=seed)
            full_name = 'Gradient Boosting Regressor'       

        elif estimator == 'mlp':

            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(random_state=seed)
            full_name = 'MLP Regressor'
            
        elif estimator == 'xgboost':
            
            from xgboost import XGBRegressor
            model = XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0)
            full_name = 'Extreme Gradient Boosting Regressor'
            
        elif estimator == 'lightgbm':
            
            import lightgbm as lgb
            model = lgb.LGBMRegressor(random_state=seed)
            full_name = 'Light Gradient Boosting Machine'
            
        elif estimator == 'catboost':
            
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(random_state=seed, silent = True)
            full_name = 'CatBoost Regressor'
            
        logger.info(str(full_name) + " Imported Successfully")

        progress.value += 1 
        
        """
        start model building here

        """
        
        logger.info("Creating Regressor without anomaly")

        score = []
        metric = []
        
        #build model without anomaly
        monitor.iloc[2,1:] = 'Evaluating Regressor Without Anomaly Detector'
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')   

        d = master_df[1].copy()
        d.drop(['Label', 'Score'], axis=1, inplace=True)

        #drop NA's caution
        d.dropna(axis=0, inplace=True)
        
        #get_dummies to caste categorical variables for supervised learning 
        d = pd.get_dummies(d)
        
        #split the dataset
        X = d.drop(supervised_target, axis=1)
        y = d[supervised_target]
            
        #fit the model
        logger.info("Fitting Model") 
        model.fit(X,y)

        #generate the prediction and evaluate metric
        logger.info("Generating Cross Val Predictions")
        pred = cross_val_predict(model,X,y,cv=fold, method = 'predict')

        if optimize == 'R2':
            r2_ = metrics.r2_score(y,pred)
            score.append(r2_)

        elif optimize == 'MAE':          
            mae_ = metrics.mean_absolute_error(y,pred)
            score.append(mae_)

        elif optimize == 'MSE':
            mse_ = metrics.mean_squared_error(y,pred)
            score.append(mse_)

        elif optimize == 'RMSE':
            mse_ = metrics.mean_squared_error(y,pred)        
            rmse_ = np.sqrt(mse_)
            score.append(rmse_)

        elif optimize == 'RMSLE':
            rmsle = np.sqrt(np.mean(np.power(np.log(np.array(abs(pred))+1) - np.log(np.array(abs(y))+1), 2)))
            score.append(rmsle)
            
        elif optimize == 'MAPE':
            
            def calculate_mape(actual, prediction):
                mask = actual != 0
                return (np.fabs(actual - prediction)/actual)[mask].mean()
            
            mape = calculate_mape(y,pred)
            score.append(mape)      
            
        metric.append(str(optimize))
        
        for i in range(1,len(master_df)):
            progress.value += 1 
            param_grid_val = param_grid[i-1]
            
            logger.info("Creating Regressor with Fraction = " + str(param_grid_val)) 

            monitor.iloc[2,1:] = 'Evaluating Regressor With ' + str(param_grid_val) + ' Fraction'
            if verbose:
                if html_param:
                    update_display(monitor, display_id = 'monitor')    
                             
            #prepare the dataset for supervised problem
            d = master_df[i]
                    
            #cleaning the dataframe for supervised learning
            d.dropna(axis=0, inplace=True)
            Score_ = pd.DataFrame(d['Score'])
            Score = scaler.fit_transform(Score_)
            d['Score'] = Score
            
            if method == 'drop':
                d = d[d['Label'] == 0]
                d.drop(['Label'], axis=1, inplace=True)
            
            #get_dummies to caste categorical variable for supervised learning
            d = pd.get_dummies(d)
                        
            #split the dataset
            X = d.drop(supervised_target, axis=1)
            y = d[supervised_target]

            #fit the model
            logger.info("Fitting Model") 
            model.fit(X,y)

            #generate the prediction and evaluate metric
            logger.info("Generating Cross Val Predictions") 
            pred = cross_val_predict(model,X,y,cv=fold, method = 'predict')

            if optimize == 'R2':
                r2_ = metrics.r2_score(y,pred)
                score.append(r2_)
                
            elif optimize == 'MAE':          
                mae_ = metrics.mean_absolute_error(y,pred)
                score.append(mae_)

            elif optimize == 'MSE':
                mse_ = metrics.mean_squared_error(y,pred)
                score.append(mse_)
                
            elif optimize == 'RMSE':
                mse_ = metrics.mean_squared_error(y,pred)        
                rmse_ = np.sqrt(mse_)
                score.append(rmse_)
            
            elif optimize == 'RMSLE':
                rmsle = np.sqrt(np.mean(np.power(np.log(np.array(abs(pred))+1) - np.log(np.array(abs(y))+1), 2)))
                score.append(rmsle)

            elif optimize == 'MAPE':

                def calculate_mape(actual, prediction):
                    mask = actual != 0
                    return (np.fabs(actual - prediction)/actual)[mask].mean()
                
                mape = calculate_mape(y,pred)
                score.append(mape)
                
            metric.append(str(optimize))
        
        monitor.iloc[1,1:] = 'Compiling Results'
        monitor.iloc[1,1:] = 'Finalizing'
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')                    

        logger.info("Creating metrics dataframe")  
        df = pd.DataFrame({'Fraction': param_grid_with_zero, 'Score' : score, 'Metric': metric})
        df.columns = ['Fraction %', optimize, 'Metric']
        
        #sorting to return best model
        if optimize == 'R2':
            sorted_df = df.sort_values(by=optimize, ascending=False)
        else: 
            sorted_df = df.sort_values(by=optimize, ascending=True)
            
        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]
        best_model_tt = model_fit_time_list[ival]

        logger.info("Rendering Visual")

        fig = px.line(df, x='Fraction %', y=optimize, line_shape='linear', 
                      title= str(full_name) + ' Metrics and Fraction %', color='Metric')

        fig.update_layout(plot_bgcolor='rgb(245,245,245)')
        progress.value += 1 

        fig.show()
        
        logger.info("Visual Rendered Successfully")

        #monitor = ''
        
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')
        monitor_out.clear_output()
        progress.close()
        
        best_k = np.array(sorted_df.head(1)['Fraction %'])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0],4)
        p = 'Best Model: ' + model_name + ' |' + ' Fraction %: ' + str(best_k) + ' | ' + str(optimize) + ' : ' + str(best_m)
        print(p)
        
    logger.info("Resetting environment to original variables")
    org = retain_original(a,b,c,e,z)

    #end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    #mlflow logging
    if logging_param:
        
        logger.info("Creating MLFlow logs")

        #import mlflow
        import mlflow
        from pathlib import Path
        import os

        mlflow.set_experiment(exp_name_log)

        #Creating Logs message monitor
        monitor.iloc[1,1:] = 'Creating Logs'
        if verbose:
            if html_param:
                update_display(monitor, display_id = 'monitor')

        mlflow.set_experiment(exp_name_log)

        with mlflow.start_run(run_name=model_name) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            # Log model parameters
            params = best_model.get_params()

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            mlflow.log_params(params)
            
            #set tag of compare_models
            mlflow.set_tag("Source", "tune_model")
            
            import secrets
            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)   
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time in seconds
            mlflow.log_metric("TT", best_model_tt) #change this

            # Log plot to html
            fig.write_html("Iterations.html")
            mlflow.log_artifact('Iterations.html')
            os.remove('Iterations.html')

            # Log model and transformation pipeline
            logger.info("SubProcess save_model() called ==================================")
            save_model(best_model, 'Trained Model', verbose=False)
            logger.info("SubProcess save_model() end ==================================")
            mlflow.log_artifact('Trained Model' + '.pkl')
            size_bytes = Path('Trained Model.pkl').stat().st_size
            size_kb = np.round(size_bytes/1000, 2)
            mlflow.set_tag("Size KB", size_kb)
            os.remove('Trained Model.pkl')
    
    logger.info(str(best_model))
    logger.info("tune_model() succesfully completed......................................")

    return best_model

def plot_model(model,
               plot = 'tsne',
               feature = None,
               save = False, #added in pycaret 2.0.0
               system = True): #added in pycaret 2.0.0
               
    
    """
    This function takes a trained model object and returns a plot on the dataset 
    passed during setup stage. This function internally calls assign_model before 
    generating a plot.  

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> experiment_name = setup(data = anomaly, normalize = True)
    >>> knn = create_model('knn')
    >>> plot_model(knn)

    Parameters
    ----------
    model : object
        A trained model object can be passed. Model must be created using create_model().

    plot : string, default = 'tsne'
        Enter abbreviation of type of plot. The current list of plots supported are (Plot - Name):

        * 'tsne' - t-SNE (3d) Dimension Plot
        * 'umap' - UMAP Dimensionality Plot

    feature : string, default = None
        Feature column is used as a hoverover tooltip. By default, first of column of the
        dataset is chosen as hoverover tooltip, when no feature is passed.
    
    save: Boolean, default = False
        Plot is saved as png file in local directory when save parameter set to True.

    system: Boolean, default = True
        Must remain True all times. Only to be changed by internal functions.

    Returns
    -------
    Visual_Plot
        Prints the visual plot. 

    """  
    
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
    logger.info("""plot_model(model={}, plot={}, feature={}, save={}, system={})""".\
        format(str(model), str(plot), str(feature), str(save), str(system)))

    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
        
    """
    exception handling starts here
    """
    
    logger.info("Checking exceptions")

    #plot checking
    allowed_plots = ['tsne', 'umap']  
    if plot not in allowed_plots:
        sys.exit('(Value Error): Plot Not Available. Please see docstring for list of available plots.')
     

    
    """
    error handling ends here
    """
    
    logger.info("Importing libraries")
    #import dependencies
    import pandas as pd
    import numpy
    
    #import cufflinks
    import cufflinks as cf
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    logger.info("plot type: " + str(plot))

    if plot == 'tsne':
        
        logger.info("SubProcess assign_model() called ==================================")
        b = assign_model(model, verbose=False, transformation=True, score=False)
        logger.info("SubProcess assign_model() end ==================================")
        Label = pd.DataFrame(b['Label'])
        b.dropna(axis=0, inplace=True) #droping rows with NA's
        b.drop(['Label'], axis=1, inplace=True)
        
        logger.info("Getting dummies to cast categorical variables")
        b = pd.get_dummies(b) #casting categorical variables

        from sklearn.manifold import TSNE
        logger.info("Fitting TSNE()")
        X_embedded = TSNE(n_components=3).fit_transform(b)

        X = pd.DataFrame(X_embedded)
        X['Label'] = Label
        
        if feature is not None: 
            X['Feature'] = data_[feature]
        else:
            X['Feature'] = data_[data_.columns[0]]

        import plotly.express as px
        df = X

        logger.info("Rendering Visual")

        fig = px.scatter_3d(df, x=0, y=1, z=2, hover_data=['Feature'], color='Label', title='3d TSNE Plot for Outliers', 
                                opacity=0.7, width=900, height=800)
            
            
        if system:
            fig.show()

        logger.info("Visual Rendered Successfully")

        if save:
            fig.write_html("TSNE.html")
            logger.info("Saving 'TSNE.html' in current active directory") 

    elif plot == 'umap':
        
        logger.info("SubProcess assign_model() called ==================================")
        b = assign_model(model, verbose=False, transformation=True, score=False)
        logger.info("SubProcess assign_model() end ==================================")

        Label = pd.DataFrame(b['Label'])
        b.dropna(axis=0, inplace=True) #droping rows with NA's
        b.drop(['Label'], axis=1, inplace=True)
        
        logger.info("Getting dummies to cast categorical variables")
        b = pd.get_dummies(b) #casting categorical variables
        
        import umap
        reducer = umap.UMAP()
        logger.info("Fitting UMAP()")
        embedding = reducer.fit_transform(b)
        X = pd.DataFrame(embedding)

        import plotly.express as px
        df = X
        df['Label'] = Label
        
        if feature is not None: 
            df['Feature'] = data_[feature]
        else:
            df['Feature'] = data_[data_.columns[0]]
        
        logger.info("Rendering Visual")

        fig = px.scatter(df, x=0, y=1,
                      color='Label', title='uMAP Plot for Outliers', hover_data=['Feature'], opacity=0.7, 
                         width=900, height=800)
        if system:
            fig.show() 

        logger.info("Visual Rendered Successfully")
        
        if save:
            fig.write_html("UMAP.html")
            logger.info("Saving 'UMAP.html' in current active directory") 
    
    logger.info("plot_model() succesfully completed......................................")

def save_model(model, model_name, verbose=True):
    
    """
    This function saves the transformation pipeline and trained model object 
    into the current active directory as a pickle file for later use. 
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> experiment_name = setup(data = anomaly, normalize = True)
    >>> knn = create_model('knn')  
    >>> save_model(knn, 'knn_model_23122019')
    
    This will save the transformation pipeline and model as a binary pickle
    file in the current directory. 

    Parameters
    ----------
    model : object, default = none
        A trained model object should be passed.
    
    model_name : string, default = none
        Name of pickle file to be passed as a string.

    verbose : bool, default = True
        When set to False, success message is not printed.

    Returns
    -------
    Success_Message
       
         
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

    logger.info("Initializing save_model()")
    logger.info("""save_model(model={}, model_name={}, verbose={})""".\
        format(str(model), str(model_name), str(verbose)))

    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    logger.info("Appending prep pipeline")
    model_ = []
    model_.append(prep_pipe)
    model_.append(model)
    
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
    >>> saved_knn = load_model('knn_model_23122019')
    
    This will load the previously saved model in saved_lr variable. The file 
    must be in the current directory.

    Parameters
    ----------
    model_name : string, default = none
        Name of pickle file to be passed as a string.

    platform: string, default = None
        Name of platform, if loading model from cloud. Current available options are:
        'aws'.
    
    authentication : dict
        Dictionary of applicable authentication tokens. 
    
        When platform = 'aws': 
        {'bucket' : 'Name of Bucket on S3'}

    verbose: Boolean, default = True
        Success message is not printed when verbose is set to False. 

    Returns
    -------    
    Success_Message  
       
         
    """

    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #exception checking
    import sys
    
    if platform is not None:
        if authentication is None:
            sys.exit("(Value Error): Authentication is missing.")
        
    #cloud provider
    if platform == 'aws':
        
        import boto3
        bucketname = authentication.get('bucket')
        filename = str(model_name) + '.pkl'
        s3 = boto3.resource('s3')
        s3.Bucket(bucketname).download_file(filename, filename)
        filename = str(model_name)
        model = load_model(filename, verbose=False)
        
        if verbose:
            print('Transformation Pipeline and Model Sucessfully Loaded')

        return model
        
    import joblib
    model_name = model_name + '.pkl'
    if verbose:
        print('Transformation Pipeline and Model Sucessfully Loaded')

    return joblib.load(model_name)

def predict_model(model, 
                  data,
                  platform=None,
                  authentication=None):
    
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
    
    data : {array-like, sparse matrix}
        Shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
        All features used during training must be present in the new dataset.
    
    platform: string, default = None
        Name of platform, if loading model from cloud. Current available options are:
        'aws'.
    
    authentication : dict
        Dictionary of applicable authentication tokens. 
        
        When platform = 'aws': 
        {'bucket' : 'Name of Bucket on S3'}
     
    Returns
    -------
    info_grid
        Information grid is printed when data is None.             
    
    """
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #testing
    #no active tests
    
    #general dependencies
    from IPython.display import clear_output, update_display
    import numpy as np
    import pandas as pd
    import re
    from sklearn import metrics
    from copy import deepcopy
    import sys
    
    #copy data and model
    data__ = data.copy()
    model_ = deepcopy(model)
    clear_output()
    
    if type(model) is str:
        if platform == 'aws':
            model_ = load_model(str(model), platform='aws', 
                                   authentication={'bucket': authentication.get('bucket')},
                                   verbose=False)
            
        else:
            model_ = load_model(str(model), verbose=False)
            
        
    #separate prep_data pipeline
    if type(model_) is list:
        prep_pipe_transformer = model_[0]
        model = model_[1]
        
    else:
        try: 
            prep_pipe_transformer = prep_pipe
        except:
            sys.exit('Transformation Pipeline Missing')
    
    #exception checking for predict param
    if hasattr(model, 'predict'):
        pass
    else:
        sys.exit("(Type Error): Model doesn't support predict parameter.")
    
    
    #predictions start here
    _data_ = prep_pipe_transformer.transform(data__)
    pred = model.predict(_data_)
    pred_score = model.decision_function(_data_)
        
    data__['Label'] = pred
    data__['Score'] = pred_score
    
    return data__

def deploy_model(model, 
                 model_name, 
                 authentication,
                 platform = 'aws'):
    
    """
    (In Preview)

    This function deploys the transformation pipeline and trained model object for
    production use. The platform of deployment can be defined under the platform
    param along with the applicable authentication tokens which are passed as a
    dictionary to the authentication param.
    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> experiment_name = setup(data = anomaly, normalize=True)
    >>> knn = create_model('knn')
    >>> deploy_model(model = knn, model_name = 'deploy_knn', platform = 'aws', authentication = {'bucket' : 'pycaret-test'})
    
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
    
    platform: string, default = 'aws'
        Name of platform for deployment. Current available options are: 'aws'.

    Returns
    -------    
    Success_Message
      
       
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

    #checking if awscli available
    try:
        import awscli
    except:
        logger.error("awscli library not found. pip install awscli to use deploy_model function.")
        sys.exit("awscli library not found. pip install awscli to use deploy_model function.")  
        
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
        
        import boto3
        logger.info("Saving model in current working directory")
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
        logger.info(str(model))
        logger.info("deploy_model() succesfully completed......................................")

def get_outliers(data, 
                 model = None, 
                 fraction=0.05, 
                 ignore_features = None, 
                 normalize = True, 
                 transformation = False,
                 pca = False,
                 pca_components = 0.99,
                 ignore_low_variance=False,
                 combine_rare_levels=False,
                 rare_level_threshold=0.1,
                 remove_multicollinearity=False,
                 multicollinearity_threshold=0.9,
                 n_jobs = None):
    
    """
    Magic function to get outliers in Power Query / Power BI.    
    
    """
    
    if model is None:
        model = 'knn'
        
    if ignore_features is None:
        ignore_features_pass = []
    else:
        ignore_features_pass = ignore_features
    
    global X, data_, seed, n_jobs_param, logging_param, logger
    
    n_jobs_param = n_jobs

    logging_param = False

    import logging

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
    
    data_ = data.copy()
    
    seed = 99
    
    from pycaret import preprocess
    
    X = preprocess.Preprocess_Path_Two(train_data = data, 
                                       features_todrop = ignore_features_pass,
                                       display_types = False,
                                       scale_data = normalize,
                                       scaling_method = 'zscore',
                                       Power_transform_data = transformation,
                                       Power_transform_method = 'yj',
                                       apply_pca = pca,
                                       pca_variance_retained_or_number_of_components=pca_components,
                                       apply_zero_nearZero_variance = ignore_low_variance,
                                       club_rare_levels=combine_rare_levels,
                                       rara_level_threshold_percentage=rare_level_threshold,
                                       remove_multicollinearity=remove_multicollinearity,
                                       maximum_correlation_between_features=multicollinearity_threshold,
                                       random_state = seed)
    
    
    
    c = create_model(model=model, fraction=fraction, verbose=False, system=False)
    
    dataset = assign_model(c, verbose=False)
    
    return dataset

def models():

    """
    Returns table of models available in model library.

    Example
    -------
    >>> all_models = models()

    This will return pandas dataframe with all available 
    models and their metadata.     
    
    Returns
    -------
    pandas.DataFrame

    """

    import pandas as pd

    model_id = ['abod', 'iforest', 'cluster', 'cof', 'histogram', 'knn', 'lof', 'svm', 'pca', 'mcd', 'sod', 'sos']

    model_name = ['Angle-base Outlier Detection',
                  'Isolation Forest',
                  'Clustering-Based Local Outlier',
                  'Connectivity-Based Outlier Factor',
                  'Histogram-based Outlier Detection',
                  'k-Nearest Neighbors Detector',
                  'Local Outlier Factor',
                  'One-class SVM detector',
                  'Principal Component Analysis',
                  'Minimum Covariance Determinant',
                  'Subspace Outlier Detection',
                  'Stochastic Outlier Selection']

    model_ref = ['pyod.models.abod.ABOD',
                  'pyod.models.iforest',
                  'pyod.models.cblof',
                  'pyod.models.cof',
                  'pyod.models.hbos',
                  'pyod.models.knn',
                  'pyod.models.lof',
                  'pyod.models.ocsvm',
                  'pyod.models.pca',
                  'pyod.models.mcd',
                  'pyod.models.sod',
                  'pyod.models.sos']

    df = pd.DataFrame({'ID' : model_id, 
                        'Name' : model_name,
                        'Reference' : model_ref})

    df.set_index('ID', inplace=True)

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

    - X: Transformed dataset
    - data_: Original dataset  
    - seed: random state set through session_id
    - prep_pipe: Transformation pipeline configured through setup
    - prep_param: prep_param configured through setup
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - exp_name_log: Name of experiment set through setup
    - logging_param: log_experiment param set through setup
    - log_plots_param: log_plots param set through setup
    - USI: Unique session ID parameter set through setup

    Example
    -------
    >>> X = get_config('X') 

    This will return transformed dataset.

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
    
    if variable == 'data_':
        global_var = data_

    if variable == 'seed':
        global_var = seed

    if variable == 'prep_pipe':
        global_var = prep_pipe

    if variable == 'prep_param':
        global_var = prep_param
        
    if variable == 'n_jobs_param':
        global_var = n_jobs_param

    if variable == 'html_param':
        global_var = html_param

    if variable == 'exp_name_log':
        global_var = exp_name_log

    if variable == 'logging_param':
        global_var = logging_param

    if variable == 'log_plots_param':
        global_var = log_plots_param

    if variable == 'USI':
        global_var = USI

    logger.info("Global variable: " + str(variable) + ' returned')
    logger.info("get_config() succesfully completed......................................")

    return global_var

def set_config(variable,value):

    """
    This function is used to reset global environment variables.
    Following variables can be accessed:

    - X: Transformed dataset
    - data_: Original dataset  
    - seed: random state set through session_id
    - prep_pipe: Transformation pipeline configured through setup
    - prep_param: prep_param configured through setup
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - exp_name_log: Name of experiment set through setup
    - logging_param: log_experiment param set through setup
    - log_plots_param: log_plots param set through setup
    - USI: Unique session ID parameter set through setup

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

    if variable == 'data_':
        global data_
        data_ = value

    if variable == 'seed':
        global seed
        seed = value

    if variable == 'prep_pipe':
        global prep_pipe
        prep_pipe = value

    if variable == 'prep_param':
        global prep_param
        prep_param = value

    if variable == 'n_jobs_param':
        global n_jobs_param
        n_jobs_param = value

    if variable == 'html_param':
        global html_param
        html_param = value

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