# Module: Clustering
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT



def setup(data, 
          categorical_features = None,
          categorical_imputation = 'constant',
          ordinal_features = None, #new
          high_cardinality_features = None, #latest
          numeric_features = None,
          numeric_imputation = 'mean',
          date_features = None,
          ignore_features = None,
          normalize = False,
          normalize_method = 'zscore',
          transformation = False,
          transformation_method = 'yeo-johnson',
          handle_unknown_categorical = True, #new             
          unknown_categorical_method = 'least_frequent', #new 
          pca = False,
          pca_method = 'linear',
          pca_components = None,
          ignore_low_variance = False, 
          combine_rare_levels = False, 
          rare_level_threshold = 0.10, 
          bin_numeric_features = None, 
          remove_multicollinearity = False, #new
          multicollinearity_threshold = 0.9, #new
          group_features = None, #new
          group_names = None, #new  
          supervised = False,
          supervised_target = None,
          session_id = None,
          profile = False,
          verbose=True):
    
    """
        
    Description:
    ------------
    This function initializes the environment in pycaret. setup() must called before
    executing any other function in pycaret. It takes one mandatory parameter:
    dataframe {array-like, sparse matrix}. 

        Example
        -------
        from pycaret.datasets import get_data
        jewellery = get_data('jewellery')

        experiment_name = setup(data = jewellery, normalize = True)
        
        'jewellery' is a pandas Dataframe.

    Parameters
    ----------
    data : {array-like, sparse matrix}, shape (n_samples, n_features) where n_samples 
    is the number of samples and n_features is the number of features in dataframe.
    
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
    applciable when tune_model() function is used. This param is only for internal use.

    session_id: int, default = None
    If None, a random seed is generated and returned in the Information grid. The 
    unique number is then distributed as a seed in all functions used during the 
    experiment. This can be used for later reproducibility of the entire experiment.
    
    profile: bool, default = False
    If set to true, a data profile for Exploratory Data Analysis will be displayed 
    in an interactive HTML report. 

    verbose: Boolean, default = True
    Information grid is not printed when verbose is set to False.
    
    Returns:
    --------

    info grid:    Information grid is printed.
    -----------      

    environment:  This function returns various outputs that are stored in variable
    -----------   as tuple. They are used by other functions in pycaret.

    Warnings:
    ---------
    None
      
          
    """
    
    #exception checking   
    import sys
    
    
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

    
    """
    error handling ends here
    """
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    
    #pandas option
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    
    #progress bar
    max_steps = 4
        
    progress = ipw.IntProgress(value=0, min=0, max=max_steps, step=1 , description='Processing: ')
    
        
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ] ],
                             #['Step' , '. . . . . . . . . . . . . . . . . .',  'Step 0 of ' + str(total_steps)] ],
                              columns=['', ' ', '   ']).set_index('')
    
    if verbose:
        display(progress)
        display(monitor, display_id = 'monitor')
    
    #general dependencies
    import numpy as np
    import pandas as pd
    import random
    
    #define highlight function for function grid to display
    def highlight_max(s):
        is_max = s == True
        return ['background-color: lightgreen' if v else '' for v in is_max]
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #defining global variables
    global data_, X, seed, prep_pipe, prep_param, experiment__
    
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
    update_display(monitor, display_id = 'monitor')
            
    #define parameters for preprocessor
    
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
    
    #import library
    from pycaret import preprocess
    
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
    
    #create an empty list for pickling later.
    if supervised is False:
        experiment__ = []
    else:
        try:
            experiment__.append('dummy')
            experiment__.remove('dummy')
        except:
            experiment__ = []
    
    progress.value += 1
    
    #monitor update
    monitor.iloc[1,1:] = 'Compiling Results'
    if verbose:
        update_display(monitor, display_id = 'monitor')
        
    '''
    Final display Starts
    '''
    
    shape = data.shape
    shape_transformed = X.shape
    
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
        if profile:
            clear_output()
            print('')
            print('Setup Succesfully Completed! Loading Profile Now... Please Wait!')
            display(functions_)
        else:
            clear_output()
            print('')
            print('Setup Succesfully Completed!')
            display(functions_)            
        
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
        experiment__.append(('Clustering Setup Config', functions))
        experiment__.append(('Orignal Dataset', data_))
        experiment__.append(('Transformed Dataset', X))
        experiment__.append(('Transformation Pipeline', prep_pipe))
    
    
    return X, data_, seed, prep_pipe, prep_param, experiment__




def create_model(model = None, 
                 num_clusters = None,
                 verbose=True):
    
    
    
    """  
     
    Description:
    ------------
    This function creates a model on the dataset passed as a data param during 
    the setup stage. setup() function must be called before using create_model().

    This function returns a trained model object. 

        Example
        -------
        from pycaret.datasets import get_data
        jewellery = get_data('jewellery')
        experiment_name = setup(data = jewellery, normalize = True)
        
        kmeans = create_model('kmeans')

        This will return a trained K-Means clustering model.

    Parameters
    ----------
    model : string, default = None

    Enter abbreviated string of the model class. List of available models supported:

    Model                              Abbreviated String   Original Implementation 
    ---------                          ------------------   -----------------------
    K-Means clustering                 'kmeans'             sklearn.cluster.KMeans.html
    Affinity Propagation               'ap'                 AffinityPropagation.html
    Mean shift clustering              'meanshift'          sklearn.cluster.MeanShift.html
    Spectral Clustering                'sc'                 SpectralClustering.html
    Agglomerative Clustering           'hclust'             AgglomerativeClustering.html
    Density-Based Spatial Clustering   'dbscan'             sklearn.cluster.DBSCAN.html
    OPTICS Clustering                  'optics'             sklearn.cluster.OPTICS.html
    Birch Clustering                   'birch'              sklearn.cluster.Birch.html
    K-Modes clustering                 'kmodes'             git/nicodv/kmodes
    
    num_clusters: int, default = None
    Number of clusters to be generated with the dataset. If None, num_clusters is set to 4. 

    verbose: Boolean, default = True
    Status update is not printed when verbose is set to False.

    Returns:
    --------

    model:    trained model object
    ------

    Warnings:
    ---------
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
    
    #testing
    #no test available
    
    #exception checking   
    import sys        
        
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
    allowed_models = ['kmeans', 'ap', 'meanshift', 'sc', 'hclust', 'dbscan', 'optics', 'birch', 'kmodes']
    
    #check num_clusters parameter:
    if num_clusters is not None:
        no_num_required = ['ap', 'meanshift', 'dbscan', 'optics']
        if model in no_num_required: 
            sys.exit('(Value Error): num_clusters parameter not required for specified model. Remove num_clusters to run this model.')
    
    if model not in allowed_models:
        sys.exit('(Value Error): Model Not Available. Please see docstring for list of available models.')
        
    #checking num_clusters type:
    if num_clusters is not None:
        if type(num_clusters) is not int:
            sys.exit('(Type Error): num_clusters parameter can only take value integer value greater than 1.')
        
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.') 
        
    """
    error handling ends here
    """
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    
    #determine num_clusters
    if num_clusters is None:
        num_clusters = 4
    else:
        num_clusters = num_clusters
        
    """
    monitor starts
    """
    
    #progress bar and monitor control    
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    progress = ipw.IntProgress(value=0, min=0, max=3, step=1 , description='Processing: ')
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                              ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Initializing'] ],
                              columns=['', ' ', '  ']).set_index('')
    if verbose:
        display(progress)
        display(monitor, display_id = 'monitor')
        
    progress.value += 1
    
    """
    monitor ends
    """
    
    if model == 'kmeans':
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters = num_clusters, random_state=seed)
        full_name = 'K-Means Clustering'

    elif model == 'ap':
        from sklearn.cluster import AffinityPropagation
        model = AffinityPropagation(damping=0.5)
        full_name = 'Affinity Propagation'

    elif model == 'meanshift':
        from sklearn.cluster import MeanShift
        model = MeanShift()
        full_name = 'Mean Shift Clustering'

    elif model == 'sc':
        from sklearn.cluster import SpectralClustering
        model = SpectralClustering(n_clusters=num_clusters, random_state=seed, n_jobs=-1)
        full_name = 'Spectral Clustering'

    elif model == 'hclust':
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=num_clusters)
        full_name = 'Agglomerative Clustering'

    elif model == 'dbscan':
        from sklearn.cluster import DBSCAN
        model = DBSCAN(eps=0.5, n_jobs=-1)
        full_name = 'Density-Based Spatial Clustering'

    elif model == 'optics':
        from sklearn.cluster import OPTICS
        model = OPTICS(n_jobs=-1)
        full_name = 'OPTICS Clustering'

    elif model == 'birch':
        from sklearn.cluster import Birch
        model = Birch(n_clusters=num_clusters)
        full_name = 'Birch Clustering'
        
    elif model == 'kmodes':
        from kmodes.kmodes import KModes
        model = KModes(n_clusters=num_clusters, n_jobs=1, random_state=seed)
        full_name = 'K-Modes Clustering'
        
    #elif model == 'skmeans':
    #    from spherecluster import SphericalKMeans
    #    model = SphericalKMeans(n_clusters=num_clusters, n_jobs=1, random_state=seed)
    #    full_name = 'Spherical K-Means Clustering'
        
    #monitor update
    monitor.iloc[1,1:] = 'Fitting ' + str(full_name) + ' Model'
    progress.value += 1
    if verbose:
        update_display(monitor, display_id = 'monitor')
        
    #fitting the model
    model.fit(X)
    
    #storing in experiment__
    full_name_ = str(full_name) + ' Model'
    if verbose:
        tup = (full_name_,model)
        experiment__.append(tup)  
    
    progress.value += 1
    
    if verbose:
        clear_output()

    return model



def assign_model(model, 
                 transformation=False,
                 verbose=True):
    
    """  
     
    Description:
    ------------
    This function assigns each of the data point in the dataset passed during setup
    stage to one of the clusters using trained model object passed as model param.
    create_model() function must be called before using assign_model().
    
    This function returns a pandas Dataframe.

        Example
        -------
        from pycaret.datasets import get_data
        jewellery = get_data('jewellery')
        experiment_name = setup(data = jewellery, normalize = True)
        kmeans = create_model('kmeans')
        
        kmeans_df = assign_model(kmeans)

        This will return a dataframe with inferred clusters using trained model.

    Parameters
    ----------
    model: trained model object, default = None
    
    transformation: bool, default = False
    When set to True, assigned clusters are returned on transformed dataset instead 
    of original dataset passed during setup().
    
    verbose: Boolean, default = True
    Status update is not printed when verbose is set to False.

    Returns:
    --------

    dataframe:   Returns a dataframe with assigned clusters using a trained model.
    ---------

    Warnings:
    ---------
    None
  
    """
    
    #exception checking   
    import sys
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    """
    error handling starts here
    """
    
    #determine model type and store in string
    mod_type = str(type(model))
    
    #checking for allowed models
    if 'sklearn' not in mod_type and 'KModes' not in mod_type and 'SphericalKMeans' not in mod_type:
        sys.exit('(Value Error): Model Not Recognized. Please see docstring for list of available models.') 
        
    #checking transformation parameter
    if type(transformation) is not bool:
        sys.exit('(Type Error): Transformation parameter can only take argument as True or False.')    
        
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.')     
    
    
    """
    error handling ends here
    """
    
    #pre-load libraries
    import numpy as np
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    
    #copy data_
    if transformation:
        data__ = X.copy()
    else:
        data__ = data_.copy()
    
    #progress bar and monitor control 
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    progress = ipw.IntProgress(value=0, min=0, max=3, step=1 , description='Processing: ')
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                              ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Initializing'] ],
                              columns=['', ' ', '  ']).set_index('')
    if verbose:
        display(progress)
        display(monitor, display_id = 'monitor')
        
    progress.value += 1
    
    monitor.iloc[1,1:] = 'Inferring Clusters from Model'
    
    if verbose:
        update_display(monitor, display_id = 'monitor')
    
    progress.value += 1
    
    #calculation labels and attaching to dataframe
    
    labels = []
    
    for i in model.labels_:
        a = 'Cluster ' + str(i)
        labels.append(a)
        
    data__['Cluster'] = labels

    
    progress.value += 1
    
    mod_type = str(model).split("(")[0]
    
    if 'KMeans' in mod_type:
        name_ = 'K-Means Clustering' 
        
    elif 'AffinityPropagation' in mod_type:
        name_ = 'Affinity Propagation'
        
    elif 'MeanShift' in mod_type:
        name_ = 'Mean Shift Clustering'        
        
    elif 'SpectralClustering' in mod_type:
        name_ = 'Spectral Clustering'
        
    elif 'AgglomerativeClustering' in mod_type:
        name_ = 'Agglomerative Clustering'
        
    elif 'DBSCAN' in mod_type:
        name_ = 'Density-Based Spatial Clustering'
        
    elif 'OPTICS' in mod_type:
        name_ = 'OPTICS Clustering'
        
    elif 'Birch' in mod_type:
        name_ = 'Birch Clustering'
        
    elif 'KModes' in mod_type:
        name_ = 'K-Modes Clustering'
    
    else:
        name_ = 'Unknown Clustering'
        
    name_ = 'Assigned ' + str(name_)
    #storing in experiment__
    if verbose:
        tup = (name_,data__)
        experiment__.append(tup)  
    
    if verbose:
        clear_output()
        
    return data__



def tune_model(model=None,
               supervised_target=None,
               estimator=None,
               optimize=None,
               fold=10):
    
    
    """
        
    Description:
    ------------
    This function tunes the num_clusters model parameter using a predefined grid with
    the objective of optimizing a supervised learning metric as defined in the optimize
    param. You can choose the supervised estimator from a large library available in pycaret.
    By default, supervised estimator is Linear. 
    
    This function returns the tuned model object.
    
        Example
        -------
        from pycaret.datasets import get_data
        boston = get_data('boston')
        experiment_name = setup(data = boston, normalize = True)
        
        tuned_kmeans = tune_model(model = 'kmeans', supervised_target = 'medv') 

        This will return tuned K Means Clustering Model.

    Parameters
    ----------
    model : string, default = None

    Enter abbreviated name of the model. List of available models supported: 
    
    Model                              Abbreviated String   Original Implementation 
    ---------                          ------------------   -----------------------
    K-Means clustering                 'kmeans'             sklearn.cluster.KMeans.html
    Spectral Clustering                'sc'                 SpectralClustering.html
    Agglomerative Clustering           'hclust'             AgglomerativeClustering.html
    Birch Clustering                   'birch'              sklearn.cluster.Birch.html
    K-Modes clustering                 'kmodes'             git/nicodv/kmodes
    
    supervised_target: string
    Name of the target column for supervised learning.
    
    estimator: string, default = None

    Estimator                     Abbreviated String     Task 
    ---------                     ------------------     ---------------
    Logistic Regression           'lr'                   Classification
    K Nearest Neighbour           'knn'                  Classification
    Naives Bayes                  'nb'                   Classification
    Decision Tree                 'dt'                   Classification
    SVM (Linear)                  'svm'                  Classification
    SVM (RBF)                     'rbfsvm'               Classification
    Gaussian Process              'gpc'                  Classification
    Multi Level Perceptron        'mlp'                  Classification
    Ridge Classifier              'ridge'                Classification
    Random Forest                 'rf'                   Classification
    Quadratic Disc. Analysis      'qda'                  Classification
    AdaBoost                      'ada'                  Classification
    Gradient Boosting             'gbc'                  Classification
    Linear Disc. Analysis         'lda'                  Classification
    Extra Trees Classifier        'et'                   Classification
    Extreme Gradient Boosting     'xgboost'              Classification
    Light Gradient Boosting       'lightgbm'             Classification
    CatBoost Classifier           'catboost'             Classification
    Linear Regression             'lr'                   Regression
    Lasso Regression              'lasso'                Regression
    Ridge Regression              'ridge'                Regression
    Elastic Net                   'en'                   Regression
    Least Angle Regression        'lar'                  Regression
    Lasso Least Angle Regression  'llar'                 Regression
    Orthogonal Matching Pursuit   'omp'                  Regression
    Bayesian Ridge                'br'                   Regression
    Automatic Relevance Determ.   'ard'                  Regression
    Passive Aggressive Regressor  'par'                  Regression
    Random Sample Consensus       'ransac'               Regression
    TheilSen Regressor            'tr'                   Regression
    Huber Regressor               'huber'                Regression
    Kernel Ridge                  'kr'                   Regression
    Support Vector Machine        'svm'                  Regression
    K Neighbors Regressor         'knn'                  Regression
    Decision Tree                 'dt'                   Regression
    Random Forest                 'rf'                   Regression
    Extra Trees Regressor         'et'                   Regression
    AdaBoost Regressor            'ada'                  Regression
    Gradient Boosting             'gbr'                  Regression
    Multi Level Perceptron        'mlp'                  Regression
    Extreme Gradient Boosting     'xgboost'              Regression
    Light Gradient Boosting       'lightgbm'             Regression
    CatBoost Classifier           'catboost'             Regression
    
    If set to None, Linear model is used by default for both classification
    and regression tasks.
    
    optimize: string, default = None
    
    For Classification tasks:
    Accuracy, AUC, Recall, Precision, F1, Kappa
    
    For Regression tasks:
    MAE, MSE, RMSE, R2, RMSLE, MAPE
    
    If set to None, default is 'Accuracy' for classification and 'R2' for 
    regression tasks.
    
    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    
    Returns:
    --------

    visual plot:  Visual plot with num_clusters param on x-axis with metric to
    -----------   optimize on y-axis. Also, prints the best model metric.
    
    model:        trained model object with best num_clusters param. 
    -----------

    Warnings:
    ---------
    - Affinity Propagation, Mean shift clustering, Density-Based Spatial Clustering
      and OPTICS Clustering cannot be used in this function since they donot support
      num_clusters param.
           
          
    """
    
    
    
    """
    exception handling starts here
    """
    
    global data_, X
    
    #testing
    global target_, master_df
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    import sys
    
    #checking for model parameter
    if model is None:
        sys.exit('(Value Error): Model parameter Missing. Please see docstring for list of available models.')
        
    #checking for allowed models
    allowed_models = ['kmeans', 'sc', 'hclust', 'birch', 'kmodes']
    
    if model not in allowed_models:
        sys.exit('(Value Error): Model Not Available for Tuning. Please see docstring for list of available models.')
    
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
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from ipywidgets import Output
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    
    #progress bar
    max_steps = 25

    progress = ipw.IntProgress(value=0, min=0, max=max_steps, step=1 , description='Processing: ')
    display(progress)

    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies'],
                             ['Step' , '. . . . . . . . . . . . . . . . . .',  'Initializing' ] ],
                              columns=['', ' ', '   ']).set_index('')
    
    monitor_out = Output()
    display(monitor_out)
    with monitor_out:
        display(monitor, display_id = 'monitor')

    
    #General Dependencies
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn import metrics
    import numpy as np
    import plotly.express as px
    from copy import deepcopy
    
    a = data_.copy()
    b = X.copy()
    c = deepcopy(prep_pipe)
    
    def retain_original(a,b,c):
        
        global data_, X, prep_pipe
        
        data_ = a.copy()
        X = b.copy()
        prep_pipe = deepcopy(c)

        return data_, X, prep_pipe
            
    #setting up cufflinks
    import cufflinks as cf
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)
    
    progress.value += 1 
    
    #define the problem
    if data_[supervised_target].value_counts().count() == 2: 
        problem = 'classification'
    else:
        problem = 'regression'    
    
    #define model name
    
    if model == 'kmeans':
        model_name = 'K-Means Clustering'
    elif model == 'ap':
        model_name = 'Affinity Propagation'
    elif model == 'meanshift':
        model_name = 'Mean Shift Clustering'
    elif model == 'sc':
        model_name = 'Spectral Clustering'
    elif model == 'hclust':
        model_name = 'Agglomerative Clustering'
    elif model == 'dbscan':
        model_name = 'Density-Based Spatial Clustering'
    elif model == 'optics':
        model_name = 'OPTICS Clustering'
    elif model == 'birch':
        model_name = 'Birch Clustering'
    elif model == 'kmodes':
        model_name = 'K-Modes Clustering'
    
    #defining estimator:
    if problem == 'classification' and estimator is None:
        estimator = 'lr'
    elif problem == 'regression' and estimator is None:
        estimator = 'lr'        
    else:
        estimator = estimator
    
    #defining optimizer:
    if optimize is None and problem == 'classification':
        optimize = 'Accuracy'
    elif optimize is None and problem == 'regression':
        optimize = 'R2'
    else:
        optimize=optimize
    
    progress.value += 1 
            
    #defining tuning grid
    param_grid_with_zero = [0, 4, 5, 6, 8, 10, 14, 18, 25, 30, 40] 
    param_grid = [4, 5, 6, 8, 10, 14, 18, 25, 30, 40] 
    
    master = []; master_df = []
    
    monitor.iloc[1,1:] = 'Creating Clustering Model'
    update_display(monitor, display_id = 'monitor')
    
    """
    preprocess starts here
    """
    
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
    #---# 
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
    #----------# 
    if 'not_available' in prep_param.imputer.categorical_strategy:
        cat_impute_pass = 'constant'
    elif 'most frequent' in prep_param.imputer.categorical_strategy:
        cat_impute_pass = 'mode'
    
    num_impute_pass = prep_param.imputer.numeric_strategy
    
    #NORMALIZE
    #---------#  
    if 'Empty' in str(prep_param.scaling):
        normalize_pass = False
    else:
        normalize_pass = True
        
    if normalize_pass is True:
        normalize_method_pass = prep_param.scaling.function_to_apply
    else:
        normalize_method_pass = 'zscore'
    
    #FEATURE TRANSFORMATION
    #---------------------#  
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
    #--------------------#  
    if 'Empty' in str(prep_param.binn):
        features_to_bin_pass = []
        apply_binning_pass = False
        
    else:
        features_to_bin_pass = prep_param.binn.features_to_discretize
        apply_binning_pass = True
    
    #COMBINE RARE LEVELS
    #-------------------#  
    if 'Empty' in str(prep_param.club_R_L):
        combine_rare_levels_pass = False
        combine_rare_threshold_pass = 0.1
    else:
        combine_rare_levels_pass = True
        combine_rare_threshold_pass = prep_param.club_R_L.threshold
        
    #ZERO NERO ZERO VARIANCE
    #----------------------#  
    if 'Empty' in str(prep_param.znz):
        ignore_low_variance_pass = False
    else:
        ignore_low_variance_pass = True
    
    #MULTI-COLLINEARITY
    #------------------#
    if 'Empty' in str(prep_param.fix_multi):
        remove_multicollinearity_pass = False
    else:
        remove_multicollinearity_pass = True
        
    if remove_multicollinearity_pass is True:
        multicollinearity_threshold_pass = prep_param.fix_multi.threshold
    else:
        multicollinearity_threshold_pass = 0.9
    
    #UNKNOWN CATEGORICAL LEVEL
    #------------------------#
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
    #--------------#
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
    #----------------#
    
    if 'Empty' in str(prep_param.ordinal):
        ordinal_features_pass = None
    else:
        ordinal_features_pass = prep_param.ordinal.info_as_dict
    
    #HIGH CARDINALITY
    #---------------#
    
    if 'Empty' in str(prep_param.cardinality):
        high_cardinality_features_pass = None
    else:
        high_cardinality_features_pass = prep_param.cardinality.feature
    
    global setup_without_target
    
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
                                 handle_unknown_categorical = handle_unknown_categorical_pass, #new
                                 unknown_categorical_method = unknown_categorical_method_pass, #new
                                 pca = pca_pass,
                                 pca_components = pca_comp_pass, #new
                                 pca_method = pca_method_pass, #new
                                 ignore_low_variance = ignore_low_variance_pass, #new
                                 combine_rare_levels = combine_rare_levels_pass, #new
                                 rare_level_threshold = combine_rare_threshold_pass, #new
                                 bin_numeric_features = features_to_bin_pass, #new
                                 remove_multicollinearity = remove_multicollinearity_pass, #new
                                 multicollinearity_threshold = multicollinearity_threshold_pass, #new
                                 group_features = group_features_pass, #new
                                 group_names = group_names_pass, #new
                                 supervised = True,
                                 supervised_target = supervised_target,
                                 session_id = seed,
                                 profile=False,
                                 verbose=False)
    
    data_without_target = setup_without_target[0]
    
    """
    preprocess ends here
    """
    
    #adding dummy model in master
    master.append('No Model Required')
    master_df.append('No Model Required')
    
    for i in param_grid:
        progress.value += 1                      
        monitor.iloc[2,1:] = 'Fitting Model With ' + str(i) + ' Clusters'
        update_display(monitor, display_id = 'monitor')
                             
        #create and assign the model to dataset d
        m = create_model(model=model, num_clusters=i, verbose=False)
        d = assign_model(m, transformation=True, verbose=False)
        d[str(supervised_target)] = target_

        master.append(m)
        master_df.append(d)

        #clustering model creation end's here
        
    #attaching target variable back
    data_[str(supervised_target)] = target_

    
    if problem == 'classification':
        
        """
        
        defining estimator
        
        """
        
        monitor.iloc[1,1:] = 'Evaluating Clustering Model'
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
        
        
        progress.value += 1 
        
        """
        start model building here

        """
                             
        acc = [];  auc = []; recall = []; prec = []; kappa = []; f1 = []
        
        #build model without clustering
        monitor.iloc[2,1:] = 'Evaluating Classifier Without Clustering'
        update_display(monitor, display_id = 'monitor')   

        d = master_df[1].copy()
        d.drop(['Cluster'], axis=1, inplace=True)

        #drop NA's caution
        d.dropna(axis=0, inplace=True)
        
        #get_dummies to caste categorical variables for supervised learning 
        d = pd.get_dummies(d)

        #split the dataset
        X = d.drop(supervised_target, axis=1)
        y = d[supervised_target]

        #fit the model
        model.fit(X,y)
        
        #generate the prediction and evaluate metric
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
            
            monitor.iloc[2,1:] = 'Evaluating Classifier With ' + str(param_grid_val) + ' Clusters'
            update_display(monitor, display_id = 'monitor')                
                             
            #prepare the dataset for supervised problem
            d = master_df[i]
            
            #dropping NAs
            d.dropna(axis=0, inplace=True)

            #get_dummies to caste categorical variables for supervised learning 
            d = pd.get_dummies(d)

            #split the dataset
            X = d.drop(supervised_target, axis=1)
            y = d[supervised_target]

            #fit the model
            model.fit(X,y)

            #generate the prediction and evaluate metric
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
        update_display(monitor, display_id = 'monitor')
                             
        df = pd.DataFrame({'# of Clusters': param_grid_with_zero, 'Accuracy' : acc, 'AUC' : auc, 'Recall' : recall, 
                   'Precision' : prec, 'F1' : f1, 'Kappa' : kappa})
        
        sorted_df = df.sort_values(by=optimize, ascending=False)
        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]
        progress.value += 1 
        sd = pd.melt(df, id_vars=['# of Clusters'], value_vars=['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa'], 
                     var_name='Metric', value_name='Score')

        fig = px.line(sd, x='# of Clusters', y='Score', color='Metric', line_shape='linear', range_y = [0,1])
        fig.update_layout(plot_bgcolor='rgb(245,245,245)')
        title= str(full_name) + ' Metrics and Number of Clusters'
        fig.update_layout(title={'text': title, 'y':0.95,'x':0.45,'xanchor': 'center','yanchor': 'top'})

        fig.show()
        
        monitor = ''
        update_display(monitor, display_id = 'monitor')
        
        monitor_out.clear_output()
        progress.close()

        best_k = np.array(sorted_df.head(1)['# of Clusters'])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0],4)
        p = 'Best Model: ' + model_name + ' |' + ' Number of Clusters : ' + str(best_k) + ' | ' + str(optimize) + ' : ' + str(best_m)
        print(p)

    elif problem == 'regression':
        
        """
        
        defining estimator
        
        """
        
        monitor.iloc[1,1:] = 'Evaluating Clustering Model'
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
            
        progress.value += 1 
        
        """
        start model building here

        """
        
        score = []
        metric = []
        
        #build model without clustering
        monitor.iloc[2,1:] = 'Evaluating Regressor Without Clustering'
        update_display(monitor, display_id = 'monitor')   

        d = master_df[1].copy()
        d.drop(['Cluster'], axis=1, inplace=True)

        #drop NA's caution
        d.dropna(axis=0, inplace=True)
        
        #get_dummies to caste categorical variables for supervised learning 
        d = pd.get_dummies(d)
        
        #split the dataset
        X = d.drop(supervised_target, axis=1)
        y = d[supervised_target]
            
        #fit the model
        model.fit(X,y)

        #generate the prediction and evaluate metric
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
            
            monitor.iloc[2,1:] = 'Evaluating Regressor With ' + str(param_grid_val) + ' Clusters'
            update_display(monitor, display_id = 'monitor')    
                             
            #prepare the dataset for supervised problem
            d = master_df[i]
                    
            #dropping NA's
            d.dropna(axis=0, inplace=True)
            
            #get_dummies to caste categorical variable for supervised learning
            d = pd.get_dummies(d)
                        
            #split the dataset
            X = d.drop(supervised_target, axis=1)
            y = d[supervised_target]

            #fit the model
            model.fit(X,y)

            #generate the prediction and evaluate metric
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
        update_display(monitor, display_id = 'monitor')                    
         
        df = pd.DataFrame({'Clusters': param_grid_with_zero, 'Score' : score, 'Metric': metric})
        df.columns = ['# of Clusters', optimize, 'Metric']
        
        #sorting to return best model
        if optimize == 'R2':
            sorted_df = df.sort_values(by=optimize, ascending=False)
        else: 
            sorted_df = df.sort_values(by=optimize, ascending=True)
            
        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]

        fig = px.line(df, x='# of Clusters', y=optimize, line_shape='linear', 
                      title= str(full_name) + ' Metrics and Number of Clusters', color='Metric')

        fig.update_layout(plot_bgcolor='rgb(245,245,245)')
        progress.value += 1 
        
        fig.show()
        
        monitor = ''
        update_display(monitor, display_id = 'monitor')
        
        monitor_out.clear_output()
        progress.close()

        best_k = np.array(sorted_df.head(1)['# of Clusters'])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0],4)
        p = 'Best Model: ' + model_name + ' |' + ' Number of Clusters: ' + str(best_k) + ' | ' + str(optimize) + ' : ' + str(best_m)
        print(p)
        
    #storing into experiment
    tup = ('Best Model',best_model)
    experiment__.append(tup)    
    
    org = retain_original(a,b,c)
    
    return best_model

    



def plot_model(model, plot='cluster', feature = None, label = False):
    
    
    """
          
    Description:
    ------------
    This function takes a trained model object and returns a plot on the dataset 
    passed during setup stage. This function internally calls assign_model before 
    generating a plot.  

        Example:
        --------
        from pycaret.datasets import get_data
        jewellery = get_data('jewellery')
        experiment_name = setup(data = jewellery, normalize = True)
        kmeans = create_model('kmeans')
        
        plot_model(kmeans)

        This will return a cluster scatter plot (by default). 

    Parameters
    ----------
    model : object, default = none
    A trained model object can be passed. Model must be created using create_model().

    plot : string, default = 'cluster'
    Enter abbreviation for type of plot. The current list of plots supported are:

    Name                           Abbreviated String     
    ---------                      ------------------     
    Cluster PCA Plot (2d)          'cluster'              
    Cluster TSnE (3d)              'tsne'
    Elbow Plot                     'elbow'
    Silhouette Plot                'silhouette'
    Distance Plot                  'distance'
    Distribution Plot              'distribution'
    
    feature : string, default = None
    Name of feature column for x-axis of when plot = 'distribution'. When plot is
    'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or label
    when label is set to True. If no feature name is passed in 'cluster' or 'tsne'
    by default the first of column of dataset is chosen as hoverover tooltip.
    
    label : bool, default = False
    When set to True, data labels are shown in 'cluster' and 'tsne' plot.
    
    Returns:
    --------

    Visual Plot:  Prints the visual plot. 
    ------------

    Warnings:
    ---------
    None
              

    """  
    
    #exception checking   
    import sys
    
    """
    exception handling starts here
    """

    #plot checking
    allowed_plots = ['cluster', 'tsne', 'elbow', 'silhouette', 'distance', 'distribution']  
    if plot not in allowed_plots:
        sys.exit('(Value Error): Plot Not Available. Please see docstring for list of available plots.')
        
    if type(label) is not bool:
        sys.exit('(Type Error): Label param only accepts True or False. ')
        
    if feature is not None:
        if type(feature) is not str:
            sys.exit('(Type Error): feature parameter must be string containing column name of dataset. ') 
    
    
    
    
    #specific disallowed plots
    
    """
    error handling ends here
    """
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #general dependencies
    import pandas as pd
    import numpy as np
    import plotly.express as px
        
    #import cufflinks
    import cufflinks as cf
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)
    
    
    if plot == 'cluster':
        
        b = assign_model(model, verbose=False, transformation=True)       
        
        """
        sorting
        """
        clus_num = []
        for i in b.Cluster:
            a = int(i.split()[1])
            clus_num.append(a)

        b['cnum'] = clus_num
        b.sort_values(by='cnum', inplace=True)
        b.reset_index(inplace=True, drop=True)
        
        clus_label = []
        for i in b.cnum:
            a = 'Cluster ' + str(i)
            clus_label.append(a)
        
        b.drop(['Cluster', 'cnum'], inplace=True, axis=1)
        b['Cluster'] = clus_label
        
        """
        sorting ends
        """
            
        cluster = b['Cluster']
        b.drop(['Cluster'], axis=1, inplace=True)
        b = pd.get_dummies(b) #casting categorical variable
        c = b.copy()
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=seed)
        pca_ = pca.fit_transform(b)
        pca_ = pd.DataFrame(pca_)
        pca_ = pca_.rename(columns={0: "PCA1", 1: "PCA2"})
        pca_['Cluster'] = cluster
        
        if feature is not None: 
            pca_['Feature'] = data_[feature]
        else:
            pca_['Feature'] = data_[data_.columns[0]]
            
        if label:
                pca_['Label'] = pca_['Feature']

        if label:
            fig = px.scatter(pca_, x="PCA1", y="PCA2", text='Label', color='Cluster', opacity=0.5)
        else:
            fig = px.scatter(pca_, x="PCA1", y="PCA2", hover_data=['Feature'], color='Cluster', opacity=0.5)

        fig.update_traces(textposition='top center')
        fig.update_layout(plot_bgcolor='rgb(240,240,240)')

        fig.update_layout(
            height=600,
            title_text='2D Cluster PCA Plot'
        )

        fig.show()
        
        
    elif plot == 'tsne':
        
        b = assign_model(model, verbose=False, transformation=True)
        
        """
        sorting
        """
        clus_num = []
        for i in b.Cluster:
            a = int(i.split()[1])
            clus_num.append(a)

        b['cnum'] = clus_num
        b.sort_values(by='cnum', inplace=True)
        b.reset_index(inplace=True, drop=True)
        
        clus_label = []
        for i in b.cnum:
            a = 'Cluster ' + str(i)
            clus_label.append(a)
        
        b.drop(['Cluster', 'cnum'], inplace=True, axis=1)
        b['Cluster'] = clus_label
        
        """
        sorting ends
        """
    
        cluster = b['Cluster']
        b.drop(['Cluster'], axis=1, inplace=True)
        
        from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components=3, random_state=seed).fit_transform(b)
        X_embedded = pd.DataFrame(X_embedded)
        X_embedded['Cluster'] = cluster
        
        if feature is not None: 
            X_embedded['Feature'] = data_[feature]
        else:
            X_embedded['Feature'] = data_[data_.columns[0]]
            
        if label:
                X_embedded['Label'] = X_embedded['Feature']
                
        import plotly.express as px
        df = X_embedded
        
        if label:
            
            fig = px.scatter_3d(df, x=0, y=1, z=2, color='Cluster', title='3d TSNE Plot for Clusters', 
                    text = 'Label', opacity=0.7, width=900, height=800)
            
        else:
            fig = px.scatter_3d(df, x=0, y=1, z=2, color='Cluster', title='3d TSNE Plot for Clusters', 
                                hover_data = ['Feature'], opacity=0.7, width=900, height=800)
        
        fig.show()
        
        
    elif plot == 'distribution':
        
        import plotly.express as px
        
        d = assign_model(model, verbose = False)
        
        """
        sorting
        """
        clus_num = []
        for i in d.Cluster:
            a = int(i.split()[1])
            clus_num.append(a)

        d['cnum'] = clus_num
        d.sort_values(by='cnum', inplace=True)
        d.reset_index(inplace=True, drop=True)
        
        clus_label = []
        for i in d.cnum:
            a = 'Cluster ' + str(i)
            clus_label.append(a)
        
        d.drop(['Cluster', 'cnum'], inplace=True, axis=1)
        d['Cluster'] = clus_label
        
        """
        sorting ends
        """
        
        
        if feature is None:
            x_col = 'Cluster'
        else:
            x_col = feature
        
        fig = px.histogram(d, x=x_col, color="Cluster",
                   marginal="box", opacity = 0.7,
                   hover_data=d.columns)
        fig.show()


    elif plot == 'elbow':
        
        from copy import deepcopy
        model_ = deepcopy(model)
        
        try: 
            from yellowbrick.cluster import KElbowVisualizer
            visualizer = KElbowVisualizer(model_,timings=False)
            visualizer.fit(X)
            visualizer.poof()
            
        except: 
            sys.exit('(Type Error): Plot Type not supported for this model.')
        
    elif plot == 'silhouette':
        
        try:
            from yellowbrick.cluster import SilhouetteVisualizer
            visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
            visualizer.fit(X)
            visualizer.poof()
        
        except: 
            sys.exit('(Type Error): Plot Type not supported for this model.')
            
    elif plot == 'distance':  
        
        try:    
            from yellowbrick.cluster import InterclusterDistance
            visualizer = InterclusterDistance(model)
            visualizer.fit(X)
            visualizer.poof()
            
        except:
            sys.exit('(Type Error): Plot Type not supported for this model.')




def save_model(model, model_name, verbose=True):
    
    """
          
    Description:
    ------------
    This function saves the transformation pipeline and trained model object 
    into the current active directory as a pickle file for later use. 
    
        Example:
        --------
        from pycaret.datasets import get_data
        jewellery = get_data('jewellery')
        experiment_name = setup(data = jewellery, normalize = True)
        kmeans = create_model('kmeans')
        
        save_model(kmeans, 'kmeans_model_23122019')
        
        This will save the transformation pipeline and model as a binary pickle
        file in the current directory. 

    Parameters
    ----------
    model : object, default = none
    A trained model object should be passed.
    
    model_name : string, default = none
    Name of pickle file to be passed as a string.

    Returns:
    --------    
    Success Message

    Warnings:
    ---------
    None    
       
         
    """
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    model_ = []
    model_.append(prep_pipe)
    model_.append(model)
    
    import joblib
    model_name = model_name + '.pkl'
    joblib.dump(model_, model_name)
    if verbose:
        print('Transformation Pipeline and Model Succesfully Saved')



def load_model(model_name, 
               platform = None,
               authentication = None,
               verbose = True):
    
    """
          
    Description:
    ------------
    This function loads a previously saved transformation pipeline and model 
    from the current active directory into the current python environment. 
    Load object must be a pickle file.
    
        Example:
        --------
        saved_kmeans = load_model('kmeans_model_23122019')
        
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
    dictionary of applicable authentication tokens. 
    
     When platform = 'aws': 
     {'bucket' : 'Name of Bucket on S3'}
     
    verbose: Boolean, default = True
    Success message is not printed when verbose is set to False. 
    
    Returns:
    --------    
    Success Message

    Warnings:
    ---------
    None    
       
         
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



def save_experiment(experiment_name=None):
    
        
    """
          
    Description:
    ------------
    This function saves the entire experiment into the current active directory. 
    All outputs using pycaret are internally saved into a binary list which is
    pickilized when save_experiment() is used. 
    
        Example:
        --------
        save_experiment()
        
        This will save the entire experiment into the current active directory. By 
        default, the name of the experiment will use the session_id generated during 
        setup(). To use a custom name, a string must be passed to the experiment_name 
        param. For example:
        
        save_experiment('experiment_23122019')

    Parameters
    ----------
    experiment_name : string, default = none
    Name of pickle file to be passed as a string.

    Returns:
    --------    
    Success Message
    
    Warnings:
    ---------
    None    
       
         
    """
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #general dependencies
    import joblib
    global experiment__
    
    #defining experiment name
    if experiment_name is None:
        experiment_name = 'experiment_' + str(seed)
        
    else:
        experiment_name = experiment_name  
        
    experiment_name = experiment_name + '.pkl'
    joblib.dump(experiment__, experiment_name)
    
    print('Experiment Succesfully Saved')



def load_experiment(experiment_name):
    
    """
          
    Description:
    ------------
    This function loads a previously saved experiment from the current active 
    directory into current python environment. Load object must be a pickle file.
    
        Example:
        --------
        saved_experiment = load_experiment('experiment_23122019')
        
        This will load the entire experiment pipeline into the object saved_experiment.
        The experiment file must be in current directory.
        
    Parameters
    ---------- 
    experiment_name : string, default = none
    Name of pickle file to be passed as a string.

    Returns:
    --------    
    Information Grid containing details of saved objects in experiment pipeline.
    
    Warnings:
    ---------
    None    
       
         
    """
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #general dependencies
    import joblib
    import pandas as pd
    
    experiment_name = experiment_name + '.pkl'
    temp = joblib.load(experiment_name)
    
    name = []
    exp = []

    for i in temp:
        name.append(i[0])
        exp.append(i[-1])

    ind = pd.DataFrame(name, columns=['Object'])
    display(ind)

    return exp



def predict_model(model, 
                  data,
                  platform=None,
                  authentication=None):
    
    """
       
    Description:
    ------------
    This function is used to predict new data using a trained model. It requires a
    trained model object created using one of the function in pycaret that returns 
    a trained model object. New data must be passed to data param as pandas Dataframe. 
    
        Example:
        --------
        from pycaret.datasets import get_data
        jewellery = get_data('jewellery')
        experiment_name = setup(data = jewellery)
        kmeans = create_model('kmeans')
        
        kmeans_predictions = predict_model(model = kmeans, data = jewellery)
        
    Parameters
    ----------
    model : object / string,  default = None
    When model is passed as string, load_model() is called internally to load the
    pickle file from active directory or cloud platform when platform param is passed.
    
    data : {array-like, sparse matrix}, shape (n_samples, n_features) where n_samples 
    is the number of samples and n_features is the number of features. All features 
    used during training must be present in the new dataset.
    
    platform: string, default = None
    Name of platform, if loading model from cloud. Current available options are:
    'aws'.
    
    authentication : dict
    dictionary of applicable authentication tokens. 
    
     When platform = 'aws': 
     {'bucket' : 'Name of Bucket on S3'}
     
    Returns:
    --------

    info grid:  Information grid is printed when data is None.
    ----------      

    Warnings:
    ---------
    - Models that donot support 'predict' function cannot be used in predict_model(). 
  
             
    
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
    
    pred_ = []
    
    for i in pred:
        a = 'Cluster ' + str(i)
        pred_.append(a)
        
    data__['Cluster'] = pred_
    
    return data__



def deploy_model(model, 
                 model_name, 
                 authentication,
                 platform = 'aws'):
    
    """
       
    Description:
    ------------
    (In Preview)

    This function deploys the transformation pipeline and trained model object for
    production use. The platform of deployment can be defined under the platform
    param along with the applicable authentication tokens which are passed as a
    dictionary to the authentication param.
    
        Example:
        --------
        from pycaret.datasets import get_data
        jewellery = get_data('jewellery')
        experiment_name = setup(data = jewellery,  normalize = True)
        kmeans = create_model('kmeans')
        
        deploy_model(model = kmeans, model_name = 'deploy_kmeans', platform = 'aws', 
                     authentication = {'bucket' : 'pycaret-test'})
        
        This will deploy the model on an AWS S3 account under bucket 'pycaret-test'
        
        For AWS users:
        --------------
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
    dictionary of applicable authentication tokens. 
      
     When platform = 'aws': 
     {'bucket' : 'Name of Bucket on S3'}
    
    platform: string, default = 'aws'
    Name of platform for deployment. Current available options are: 'aws'.

    Returns:
    --------    
    Success Message
    
    Warnings:
    ---------
    None    
    
    """
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #general dependencies
    import ipywidgets as ipw
    import pandas as pd
    from IPython.display import clear_output, update_display
        
    try:
        model = finalize_model(model)
    except:
        pass
    
    if platform == 'aws':
        
        import boto3
        
        save_model(model, model_name = model_name, verbose=False)
        
        #initiaze s3
        s3 = boto3.client('s3')
        filename = str(model_name)+'.pkl'
        key = str(model_name)+'.pkl'
        bucket_name = authentication.get('bucket')
        s3.upload_file(filename,bucket_name,key)
        clear_output()
        print("Model Succesfully Deployed on AWS S3")



def get_clusters(data, 
                 model = None, 
                 num_clusters = 4, 
                 ignore_features = None, 
                 normalize = True, 
                 transformation = False,
                 pca = False,
                 pca_components = 0.99,
                 ignore_low_variance=False,
                 combine_rare_levels=False,
                 rare_level_threshold=0.1,
                 remove_multicollinearity=False,
                 multicollinearity_threshold=0.9):
    
    """
    Magic function to get clusters in Power Query / Power BI.    
    
    """
    
    if model is None:
        model = 'kmeans'
        
    if ignore_features is None:
        ignore_features_pass = []
    else:
        ignore_features_pass = ignore_features
    
    global X, data_, seed
    
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
                                       pca_variance_retained_or_number_of_components = pca_components,
                                       apply_zero_nearZero_variance = ignore_low_variance,
                                       club_rare_levels=combine_rare_levels,
                                       rara_level_threshold_percentage=rare_level_threshold,
                                       remove_multicollinearity=remove_multicollinearity,
                                       maximum_correlation_between_features=multicollinearity_threshold,
                                       random_state = seed)
      
    try:
        c = create_model(model=model, num_clusters=num_clusters, verbose=False)
    except:
        c = create_model(model=model, verbose=False)
    dataset = assign_model(c, verbose=False)
    return dataset