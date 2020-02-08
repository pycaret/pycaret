# Module: Classification
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT


def setup(data,  
          target,   
          train_size = 0.7, 
          sampling = True, 
          sample_estimator = None,
          categorical_features = None,
          categorical_imputation = 'constant',
          ordinal_features = None,
          high_cardinality_features = None, #latest
          high_cardinality_method = 'frequency', #latest
          numeric_features = None,
          numeric_imputation = 'mean',
          date_features = None,
          ignore_features = None,
          normalize = False,
          normalize_method = 'zscore',
          transformation = False,
          transformation_method = 'yeo-johnson',
          handle_unknown_categorical = True, #new             #create docstring and exception
          unknown_categorical_method = 'least_frequent', #new  #create docstring and exception
          pca = False, #new
          pca_method = 'linear', #new
          pca_components = None, #new
          ignore_low_variance = False, #new
          combine_rare_levels = False, #new
          rare_level_threshold = 0.10, #new
          bin_numeric_features = None, #new
          remove_outliers = False, #new
          outliers_threshold = 0.05, #new
          remove_multicollinearity = False, #new
          multicollinearity_threshold = 0.9, #new
          create_clusters = False, #new
          cluster_iter = 20, #new
          polynomial_features = False, #new                  #create checking exceptions and docstring
          polynomial_degree = 2, #new                        #create checking exceptions and docstring
          trigonometry_features = False, #new                #create checking exceptions and docstring
          polynomial_threshold = 0.1, #new                   #create checking exceptions and docstring
          group_features = None, #new                        #create checking exceptions and docstring
          group_names = None, #new                           #create checking exceptions and docstring
          feature_selection = False, #new                    #create checking exceptions and docstring
          feature_selection_threshold = 0.8, #new            #create checking exceptions and docstring
          feature_interaction = False, #new                  #create checking exceptions and docstring
          feature_ratio = False, #new                        #create checking exceptions and docstring
          interaction_threshold = 0.01,    #new              #create checking exceptions and docstring
          session_id = None,
          silent=False,
          profile = False):
    
    """
        
    Description:
    ------------    
    This function initializes the environment in pycaret and creates the transformation
    pipeline to prepare the data for modeling and deployment. setup() must called before
    executing any other function in pycaret. It takes two mandatory parameters:
    dataframe {array-like, sparse matrix} and name of the target column. 
    
    All other parameters are optional.

        Example
        -------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        
        experiment_name = setup(data = juice,  target = 'Purchase')

        'juice' is a pandas DataFrame and 'Purchase' is the name of target column.
        
    Parameters
    ----------
    data : {array-like, sparse matrix}, shape (n_samples, n_features) where n_samples 
    is the number of samples and n_features is the number of features.

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
    feature_selection_threshold param with a lower value.

    feature_selection_threshold: float, default = 0.8
    Threshold used for feature selection (including newly created polynomial features).
    A higher value will result in a higher feature space. It is recommended to do multiple
    trials with different values of feature_selection_threshold specially in cases where 
    polynomial_features and feature_interaction are used. Setting a very low value may be 
    efficient but could result in under-fitting.
    
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
    
    session_id: int, default = None
    If None, a random seed is generated and returned in the Information grid. The 
    unique number is then distributed as a seed in all functions used during the 
    experiment. This can be used for later reproducibility of the entire experiment.
    
    silent: bool, default = False
    When set to True, confirmation of data types is not required. All preprocessing will 
    be performed assuming automatically inferred data types. Not recommended for direct use 
    except for established pipelines.
    
    profile: bool, default = False
    If set to true, a data profile for Exploratory Data Analysis will be displayed 
    in an interactive HTML report. 
    
    Returns:
    --------

    info grid:    Information grid is printed.
    -----------      

    environment:  This function returns various outputs that are stored in variables
    -----------   as tuples. They are used by other functions in pycaret.

    Warnings:
    ---------
    None
      
       
    """
    
    #testing
    #no active test
    
    #exception checking   
    import sys
    
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
        if pca_method is not 'linear':
            if pca_components is not None:
                if(type(pca_components)) is not int:
                    sys.exit("(Type Error): pca_components parameter must be integer when pca_method is not 'linear'. ")

    #pca components check 2
    if pca is True:
        if pca_method is not 'linear':
            if pca_components is not None:
                if pca_components > len(data.columns)-1:
                    sys.exit("(Type Error): pca_components parameter cannot be greater than original features space.")                
 
    #pca components check 3
    if pca is True:
        if pca_method is 'linear':
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
    
    #silent
    if type(silent) is not bool:
        sys.exit("(Type Error): silent parameter only accepts True or False. ")
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    
    #pandas option
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
   
    #progress bar
    if sampling:
        max_steps = 10 + 3
    else:
        max_steps = 3
        
    progress = ipw.IntProgress(value=0, min=0, max=max_steps, step=1 , description='Processing: ')
    display(progress)
    
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id = 'monitor')
    
    #general dependencies
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import random
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    
    #define highlight function for function grid to display
    def highlight_max(s):
        is_max = s == True
        return ['background-color: lightgreen' if v else '' for v in is_max]
        
    #cufflinks
    import cufflinks as cf
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #copy original data for pandas profiler
    data_before_preprocess = data.copy()
    
    #declaring global variables to be accessed by other functions
    global X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, experiment__
    
    #generate seed to be used globally
    if session_id is None:
        seed = random.randint(150,9000)
    else:
        seed = session_id
        
    """
    preprocessing starts here
    """
    
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
        
    #import library
    from pycaret import preprocess
    
    data = preprocess.Preprocess_Path_One(train_data = data, 
                                          target_variable = target,
                                          categorical_features = cat_features_pass,
                                          apply_ordinal_encoding = apply_ordinal_encoding_pass, #new
                                          ordinal_columns_and_categories = ordinal_columns_and_categories_pass, #new
                                          apply_cardinality_reduction = apply_cardinality_reduction_pass, #latest
                                          cardinal_method = cardinal_method_pass, #latest
                                          cardinal_features = cardinal_features_pass, #latest
                                          numerical_features = numeric_features_pass,
                                          time_features = date_features_pass,
                                          features_todrop = ignore_features_pass,
                                          numeric_imputation_strategy = numeric_imputation,
                                          categorical_imputation_strategy = categorical_imputation_pass,
                                          scale_data = normalize,
                                          scaling_method = normalize_method,
                                          Power_transform_data = transformation,
                                          Power_transform_method = trans_method_pass,
                                          apply_untrained_levels_treatment= handle_unknown_categorical, #new
                                          untrained_levels_treatment_method = unknown_categorical_method_pass, #new
                                          apply_pca = pca, #new
                                          pca_method = pca_method_pass, #new
                                          pca_variance_retained_or_number_of_components = pca_components_pass, #new
                                          apply_zero_nearZero_variance = ignore_low_variance, #new
                                          club_rare_levels = combine_rare_levels, #new
                                          rara_level_threshold_percentage = rare_level_threshold, #new
                                          apply_binning = apply_binning_pass, #new
                                          features_to_binn = features_to_bin_pass, #new
                                          remove_outliers = remove_outliers, #new
                                          outlier_contamination_percentage = outliers_threshold, #new
                                          outlier_methods = ['pca'], #pca hardcoded
                                          remove_multicollinearity = remove_multicollinearity, #new
                                          maximum_correlation_between_features = multicollinearity_threshold, #new
                                          cluster_entire_data = create_clusters, #new
                                          range_of_clusters_to_try = cluster_iter, #new
                                          apply_polynomial_trigonometry_features = polynomial_features, #new
                                          max_polynomial = polynomial_degree, #new
                                          trigonometry_calculations = trigonometry_features_pass, #new
                                          top_poly_trig_features_to_select_percentage = polynomial_threshold, #new
                                          apply_grouping = apply_grouping_pass, #new
                                          features_to_group_ListofList = group_features_pass, #new
                                          group_name = group_names_pass, #new
                                          apply_feature_selection = feature_selection, #new
                                          feature_selection_top_features_percentage = feature_selection_threshold, #new
                                          apply_feature_interactions = apply_feature_interactions_pass, #new
                                          feature_interactions_to_apply = interactions_to_apply_pass, #new
                                          feature_interactions_top_features_to_select_percentage=interaction_threshold, #new
                                          display_types = display_dtypes_pass, #this is for inferred input box
                                          target_transformation = False, #not needed for classification
                                          random_state = seed)

    progress.value += 1
    

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
        pass
        
    #save prep pipe
    prep_pipe = preprocess.pipe
    
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
      
    #create an empty list for pickling later.
    experiment__ = []
        
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
            update_display(monitor, display_id = 'monitor')

            '''
            MONITOR UPDATE ENDS
            '''
    
            X_, X__, y_, y__ = train_test_split(X, y, test_size=1-i, stratify=y, random_state=seed)
            X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, stratify=y_, random_state=seed)
            model.fit(X_train,y_train)
            pred_ = model.predict(X_test)
            try:
                pred_prob = model.predict_proba(X_test)[:,1]
            except:
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
                
            #recall
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
        update_display(monitor, display_id = 'monitor')
        
        
        print('Please Enter the sample % of data you would like to use for modeling. Example: Enter 0.3 for 30%.')
        print('Press Enter if you would like to use 100% of the data.')
        
        print(' ')
        
        sample_size = input("Sample Size: ")
        
        if sample_size == '' or sample_size == '1':
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, stratify=y, random_state=seed)
            
            '''
            Final display Starts
            '''
            clear_output()
            print(' ')
            if profile:
                print('Setup Succesfully Completed! Loading Profile Now... Please Wait!')
            else:
                print('Setup Succesfully Completed!')
            
            functions = pd.DataFrame ( [ ['session_id', seed ],
                                         ['Target Type', target_type],
                                         ['Label Encoded', label_encoded],
                                         ['Original Data', data_before_preprocess.shape ],
                                         ['Missing Values ', missing_flag],
                                         ['Numeric Features ', str(float_type) ],
                                         ['Categorical Features ', str(cat_type) ],
                                         ['Ordinal Features ', ordinal_features_grid], #new
                                         ['High Cardinality Features ', high_cardinality_features_grid], #latest
                                         ['High Cardinality Method ', high_cardinality_method_grid], #latest
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
                                         ['Polynomial Features ', polynomial_features], #new
                                         ['Polynomial Degree ', polynomial_degree_grid], #new
                                         ['Trignometry Features ', trigonometry_features], #new
                                         ['Polynomial Threshold ', polynomial_threshold_grid], #new
                                         ['Group Features ', group_features_grid], #new
                                         ['Feature Selection ', feature_selection], #new
                                         ['Features Selection Threshold ', feature_selection_threshold_grid], #new
                                         ['Feature Interaction ', feature_interaction], #new
                                         ['Feature Ratio ', feature_ratio], #new
                                         ['Interaction Threshold ', interaction_threshold_grid], #new
                                       ], columns = ['Description', 'Value'] )

            #functions_ = functions.style.hide_index()
            functions_ = functions.style.apply(highlight_max)
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
            experiment__.append(('Classification Setup Config', functions))
            experiment__.append(('X_training Set', X_train))
            experiment__.append(('y_training Set', y_train))
            experiment__.append(('X_test Set', X_test))
            experiment__.append(('y_test Set', y_test)) 
            experiment__.append(('Transformation Pipeline', prep_pipe))
            
            return X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, experiment__
        
        else:
            
            sample_n = float(sample_size)
            X_selected, X_discard, y_selected, y_discard = train_test_split(X, y, test_size=1-sample_n, stratify=y, 
                                                                random_state=seed)
            
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=1-train_size, stratify=y_selected, 
                                                                random_state=seed)
            clear_output()
            
            
            '''
            Final display Starts
            '''

                
            clear_output()
            print(' ')
            if profile:
                print('Setup Succesfully Completed! Loading Profile Now... Please Wait!')
            else:
                print('Setup Succesfully Completed!')
                
            functions = pd.DataFrame ( [ ['session_id', seed ],
                                         ['Target Type', target_type],
                                         ['Label Encoded', label_encoded],
                                         ['Original Data', data_before_preprocess.shape ],
                                         ['Missing Values ', missing_flag],
                                         ['Numeric Features ', str(float_type) ],
                                         ['Categorical Features ', str(cat_type) ],
                                         ['Ordinal Features ', ordinal_features_grid], #new
                                         ['High Cardinality Features ', high_cardinality_features_grid],
                                         ['High Cardinality Method ', high_cardinality_method_grid], #latest
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
                                         ['Polynomial Features ', polynomial_features], #new
                                         ['Polynomial Degree ', polynomial_degree_grid], #new
                                         ['Trignometry Features ', trigonometry_features], #new
                                         ['Polynomial Threshold ', polynomial_threshold_grid], #new
                                         ['Group Features ', group_features_grid], #new
                                         ['Feature Selection ', feature_selection], #new
                                         ['Features Selection Threshold ', feature_selection_threshold_grid], #new
                                         ['Feature Interaction ', feature_interaction], #new
                                         ['Feature Ratio ', feature_ratio], #new
                                         ['Interaction Threshold ', interaction_threshold_grid], #new
                                       ], columns = ['Description', 'Value'] )
            
            #functions_ = functions.style.hide_index()
            functions_ = functions.style.apply(highlight_max)
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
            experiment__.append(('Classification Setup Config', functions))
            experiment__.append(('X_training Set', X_train))
            experiment__.append(('y_training Set', y_train))
            experiment__.append(('X_test Set', X_test))
            experiment__.append(('y_test Set', y_test)) 
            experiment__.append(('Transformation Pipeline', prep_pipe))
            
            return X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, experiment__

    else:
        
        monitor.iloc[1,1:] = 'Splitting Data'
        update_display(monitor, display_id = 'monitor')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, stratify=y, random_state=seed)
        progress.value += 1
        
        clear_output()
        
        '''
        Final display Starts
        '''
        clear_output()
        print(' ')
        if profile:
            print('Setup Succesfully Completed! Loading Profile Now... Please Wait!')
        else:
            print('Setup Succesfully Completed!')
            
        functions = pd.DataFrame ( [ ['session_id', seed ],
                                     ['Target Type', target_type],
                                     ['Label Encoded', label_encoded],
                                     ['Original Data', data_before_preprocess.shape ],
                                     ['Missing Values ', missing_flag],
                                     ['Numeric Features ', str(float_type) ],
                                     ['Categorical Features ', str(cat_type) ],
                                     ['Ordinal Features ', ordinal_features_grid], #new
                                     ['High Cardinality Features ', high_cardinality_features_grid],
                                     ['High Cardinality Method ', high_cardinality_method_grid], #latest
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
                                     ['Polynomial Features ', polynomial_features], #new
                                     ['Polynomial Degree ', polynomial_degree_grid], #new
                                     ['Trignometry Features ', trigonometry_features], #new
                                     ['Polynomial Threshold ', polynomial_threshold_grid], #new
                                     ['Group Features ', group_features_grid], #new
                                     ['Feature Selection ', feature_selection], #new
                                     ['Features Selection Threshold ', feature_selection_threshold_grid], #new
                                     ['Feature Interaction ', feature_interaction], #new
                                     ['Feature Ratio ', feature_ratio], #new
                                     ['Interaction Threshold ', interaction_threshold_grid], #new
                                   ], columns = ['Description', 'Value'] )
        
        #functions = functions.style.hide_index()
        functions_ = functions.style.apply(highlight_max)
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
        experiment__.append(('Classification Setup Config', functions))
        experiment__.append(('X_training Set', X_train))
        experiment__.append(('y_training Set', y_train))
        experiment__.append(('X_test Set', X_test))
        experiment__.append(('y_test Set', y_test))
        experiment__.append(('Transformation Pipeline', prep_pipe))
        
        return X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, experiment__




def create_model(estimator = None, 
                 ensemble = False, 
                 method = None, 
                 fold = 10, 
                 round = 4,  
                 verbose = True):
    
     
    """  
     
    Description:
    ------------
    This function creates a model and scores it using Stratified Cross Validation. 
    The output prints a score grid that shows Accuracy, AUC, Recall, Precision, 
    F1 and Kappa by fold (default = 10 Fold). 

    This function returns a trained model object. 

    setup() function must be called before using create_model()

        Example
        -------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        
        lr = create_model('lr')

        This will create a trained Logistic Regression model.

    Parameters
    ----------
    estimator : string, default = None

    Enter abbreviated string of the estimator class. All estimators support binary or 
    multiclass problem. List of estimators supported:

    Estimator                   Abbreviated String     Original Implementation 
    ---------                   ------------------     -----------------------
    Logistic Regression         'lr'                   linear_model.LogisticRegression
    K Nearest Neighbour         'knn'                  neighbors.KNeighborsClassifier
    Naives Bayes                'nb'                   naive_bayes.GaussianNB
    Decision Tree               'dt'                   tree.DecisionTreeClassifier
    SVM (Linear)                'svm'                  linear_model.SGDClassifier
    SVM (RBF)                   'rbfsvm'               svm.SVC
    Gaussian Process            'gpc'                  gaussian_process.GPC
    Multi Level Perceptron      'mlp'                  neural_network.MLPClassifier
    Ridge Classifier            'ridge'                linear_model.RidgeClassifier
    Random Forest               'rf'                   ensemble.RandomForestClassifier
    Quadratic Disc. Analysis    'qda'                  discriminant_analysis.QDA
    AdaBoost                    'ada'                  ensemble.AdaBoostClassifier
    Gradient Boosting           'gbc'                  ensemble.GradientBoostingClassifier
    Linear Disc. Analysis       'lda'                  discriminant_analysis.LDA
    Extra Trees Classifier      'et'                   ensemble.ExtraTreesClassifier
    Extreme Gradient Boosting   'xgboost'              xgboost.readthedocs.io
    Light Gradient Boosting     'lightgbm'             github.com/microsoft/LightGBM
    CatBoost Classifier         'catboost'             https://catboost.ai

    ensemble: Boolean, default = False
    True would result in an ensemble of estimator using the method parameter defined. 

    method: String, 'Bagging' or 'Boosting', default = None.
    method must be defined when ensemble is set to True. Default method is set to None. 

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places the metrics in the score grid will be rounded to. 

    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are Accuracy, AUC, Recall, Precision, F1 
                  and Kappa. Mean and standard deviation of the scores across the 
                  folds are also returned.

    model:        trained model object
    -----------

    Warnings:
    ---------
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
    
    #exception checking   
    import sys
    
    #checking error for estimator (string)
    available_estimators = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 
                            'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']
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
        
    #checking boosting conflict with estimators
    boosting_not_supported = ['lda','qda','ridge','mlp','gpc','svm','knn', 'catboost']
    if method is 'Boosting' and estimator in boosting_not_supported:
        sys.exit("(Type Error): Estimator does not provide class_weights or predict_proba function and hence not supported for the Boosting method. Change the estimator or method to 'Bagging'.")
    
    
    
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
        
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+4, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa'])
    display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id = 'monitor')
    
    if verbose:
        display_ = display(master_display, display_id=True)
        display_id = display_.display_id
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()
    
    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
  
    #general dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    
    progress.value += 1
    
    #cross validation setup starts here
    kf = StratifiedKFold(fold, random_state=seed)

    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    
  
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Selecting Estimator'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
        
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
        
    else:
        model = estimator
        full_name = str(model).split("(")[0]
    
    progress.value += 1
    
    #checking method when ensemble is set to True. 

    if method == 'Bagging':
        
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier(model,bootstrap=True,n_estimators=10, random_state=seed)

    elif method == 'Boosting':
        
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(model, n_estimators=10, random_state=seed)
    
    
    #multiclass checking
    if y.value_counts().count() > 2:
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(model)
    
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    
    fold_num = 1
    
    for train_i , test_i in kf.split(data_X,data_y):
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
    
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
    
        if hasattr(model, 'predict_proba'):
        
            model.fit(Xtrain,ytrain)
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
                
            kappa = metrics.cohen_kappa_score(ytest,pred_)
            score_acc = np.append(score_acc,sca)
            score_auc = np.append(score_auc,sc)
            score_recall = np.append(score_recall,recall)
            score_precision = np.append(score_precision,precision)
            score_f1 =np.append(score_f1,f1)
            score_kappa =np.append(score_kappa,kappa)

        else:
            
            model.fit(Xtrain,ytrain)
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

            kappa = metrics.cohen_kappa_score(ytest,pred_)
            score_acc = np.append(score_acc,sca)
            score_auc = np.append(score_auc,sc)
            score_recall = np.append(score_recall,recall)
            score_precision = np.append(score_precision,precision)
            score_f1 =np.append(score_f1,f1)
            score_kappa =np.append(score_kappa,kappa) 
       
        progress.value += 1
        
        
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa]}).round(round)
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
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
            
        fold_num += 1
        
        '''
        TIME CALCULATION ENDS HERE
        '''
        
        if verbose:
            update_display(master_display, display_id = display_id)
            
        
        '''
        
        Update_display() ends here
        
        '''
            
    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    
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
    
    progress.value += 1
    
    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    
    #refitting the model on complete X_train, y_train
    monitor.iloc[1,1:] = 'Compiling Final Model'
    update_display(monitor, display_id = 'monitor')
    
    model.fit(data_X, data_y)
    
    progress.value += 1
    
    #storing into experiment
    tup = (full_name,model)
    experiment__.append(tup)
    nam = str(full_name) + ' Score Grid'
    tup = (nam, model_results)
    experiment__.append(tup)
    
    if verbose:
        clear_output()
        display(model_results)
        return model
    else:
        clear_output()
        return model


def ensemble_model(estimator,
                   method = 'Bagging', 
                   fold = 10,
                   n_estimators = 10,
                   round = 4,  
                   verbose = True):
    """
       
    
    Description:
    ------------
    This function ensembles the trained base estimator using the method defined in 
    'method' param (default = 'Bagging'). The output prints a score grid that shows 
    Accuracy, AUC, Recall, Precision, F1 and Kappa by fold (default = 10 Fold). 

    This function returns a trained model object.  

    Model must be created using create_model() or tune_model().

        Example
        -------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        dt = create_model('dt')
        
        ensembled_dt = ensemble_model(dt)

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

    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are Accuracy, AUC, Recall, Precision, F1 
                  and Kappa. Mean and standard deviation of the scores across the 
                  folds are also returned.

    model:        trained ensembled model object
    -----------

    Warnings:
    ---------  
    - If target variable is multiclass (more than 2 classes), AUC will be returned 
      as zero (0.0).
        
        
    
    """
    
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #exception checking   
    import sys
        
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
    
    #pre-load libraries
    import pandas as pd
    import datetime, time
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+4, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa'])
    display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id = 'monitor')
    
    if verbose:
        display_ = display(master_display, display_id=True)
        display_id = display_.display_id
        
    #dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold   
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore')    
    
    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()
    
    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
    
    progress.value += 1
    
    #defining estimator as model
    model = estimator
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Selecting Estimator'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    if hasattr(estimator,'n_classes_'):
        if estimator.n_classes_ > 2:
            model = estimator.estimator
            
    if method == 'Bagging':
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier(model,bootstrap=True,n_estimators=n_estimators, random_state=seed)
        
    else:
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(model, n_estimators=n_estimators, random_state=seed)
    
    if y.value_counts().count() > 2:
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(model)
        
    progress.value += 1
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    kf = StratifiedKFold(fold, random_state=seed)
    
    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    
    fold_num = 1 
    
    for train_i , test_i in kf.split(data_X,data_y):
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
    
        if hasattr(model, 'predict_proba'):
        
            model.fit(Xtrain,ytrain)
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
                
            kappa = metrics.cohen_kappa_score(ytest,pred_)
            score_acc = np.append(score_acc,sca)
            score_auc = np.append(score_auc,sc)
            score_recall = np.append(score_recall,recall)
            score_precision = np.append(score_precision,precision)
            score_f1 =np.append(score_f1,f1)
            score_kappa =np.append(score_kappa,kappa)

        else:
            
            model.fit(Xtrain,ytrain)
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

            kappa = metrics.cohen_kappa_score(ytest,pred_)
            score_acc = np.append(score_acc,sca)
            score_auc = np.append(score_auc,sc)
            score_recall = np.append(score_recall,recall)
            score_precision = np.append(score_precision,precision)
            score_f1 =np.append(score_f1,f1)
            score_kappa =np.append(score_kappa,kappa) 
        
        progress.value += 1
        
                
        '''
        
        This section is created to update_display() as code loops through the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa]}).round(round)
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
            
        update_display(ETC, display_id = 'ETC')
            
        fold_num += 1
        
        
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[2,1:] = ETC
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        '''
        
        TIME CALCULATION ENDS HERE
        
        '''

        if verbose:
            update_display(master_display, display_id = display_id)
        
        '''
        
        Update_display() ends here
        
        '''
        
    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)

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

    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa})
    model_results_unpivot = pd.melt(model_results,value_vars=['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa'])
    model_results_unpivot.columns = ['Metric', 'Measure']
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)  
    
    progress.value += 1
    
    #refitting the model on complete X_train, y_train
    monitor.iloc[1,1:] = 'Compiling Final Model'
    update_display(monitor, display_id = 'monitor')
    
    model.fit(data_X, data_y)
    
    progress.value += 1
    
    #storing into experiment
    model_name = str(model).split("(")[0]
    tup = (model_name,model)
    experiment__.append(tup)
    
    nam = str(model_name) + ' Score Grid'
    tup = (nam, model_results)
    experiment__.append(tup)
    
    if verbose:
        clear_output()
        display(model_results)
        return model
    else:
        clear_output()
        return model



def plot_model(estimator, 
               plot = 'auc'): 
    
    
    """
          
    Description:
    ------------
    This function takes a trained model object and returns a plot based on the
    test / hold-out set. The process may require the model to be re-trained in
    certain cases. See list of plots supported below. 
    
    Model must be created using create_model() or tune_model().

        Example:
        --------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        lr = create_model('lr')
        
        plot_model(lr)

        This will return an AUC plot of a trained Logistic Regression model.

    Parameters
    ----------
    estimator : object, default = none
    A trained model object should be passed as an estimator. 

    plot : string, default = auc
    Enter abbreviation of type of plot. The current list of plots supported are:

    Name                        Abbreviated String     Original Implementation 
    ---------                   ------------------     -----------------------
    Area Under the Curve         'auc'                 .. / rocauc.html
    Discrimination Threshold     'threshold'           .. / threshold.html
    Precision Recall Curve       'pr'                  .. / prcurve.html
    Confusion Matrix             'confusion_matrix'    .. / confusion_matrix.html
    Class Prediction Error       'error'               .. / class_prediction_error.html
    Classification Report        'class_report'        .. / classification_report.html
    Decision Boundary            'boundary'            .. / boundaries.html
    Recursive Feat. Selection    'rfe'                 .. / rfecv.html
    Learning Curve               'learning'            .. / learning_curve.html
    Manifold Learning            'manifold'            .. / manifold.html
    Calibration Curve            'calibration'         .. / calibration_curve.html
    Validation Curve             'vc'                  .. / validation_curve.html
    Dimension Learning           'dimension'           .. / radviz.html
    Feature Importance           'feature'                   N/A 
    Model Hyperparameter         'parameter'                 N/A 

    ** https://www.scikit-yb.org/en/latest/api/classifier/<reference>

    Returns:
    --------

    Visual Plot:  Prints the visual plot. 
    ------------

    Warnings:
    ---------
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
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=5, step=1 , description='Processing: ')
    display(progress)
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #general dependencies
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    progress.value += 1
    
    #defining estimator as model locally
    model = estimator
    
    progress.value += 1
    
        
    if plot == 'auc':
        
        from yellowbrick.classifier import ROCAUC
        progress.value += 1
        visualizer = ROCAUC(model)
        visualizer.fit(X_train, y_train)
        progress.value += 1
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        visualizer.poof()
        
    elif plot == 'threshold':
        
        from yellowbrick.classifier import DiscriminationThreshold
        progress.value += 1
        visualizer = DiscriminationThreshold(model, random_state=seed)
        visualizer.fit(X_train, y_train)
        progress.value += 1
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        visualizer.poof()
    
    elif plot == 'pr':
        
        from yellowbrick.classifier import PrecisionRecallCurve
        progress.value += 1
        visualizer = PrecisionRecallCurve(model, random_state=seed)
        visualizer.fit(X_train, y_train)
        progress.value += 1
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        visualizer.poof()

    elif plot == 'confusion_matrix':
        
        from yellowbrick.classifier import ConfusionMatrix
        progress.value += 1
        visualizer = ConfusionMatrix(model, random_state=seed, fontsize = 15, cmap="Greens")
        visualizer.fit(X_train, y_train)
        progress.value += 1
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        visualizer.poof()
    
    elif plot == 'error':
        
        from yellowbrick.classifier import ClassPredictionError
        progress.value += 1
        visualizer = ClassPredictionError(model, random_state=seed)
        visualizer.fit(X_train, y_train)
        progress.value += 1
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        visualizer.poof()

    elif plot == 'class_report':
        
        from yellowbrick.classifier import ClassificationReport
        progress.value += 1
        visualizer = ClassificationReport(model, random_state=seed, support=True)
        visualizer.fit(X_train, y_train)
        progress.value += 1
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        visualizer.poof()
        
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
        X_train_transformed = StandardScaler().fit_transform(X_train_transformed)
        X_test_transformed = StandardScaler().fit_transform(X_test_transformed)
        pca = PCA(n_components=2, random_state = seed)
        X_train_transformed = pca.fit_transform(X_train_transformed)
        X_test_transformed = pca.fit_transform(X_test_transformed)
        
        progress.value += 1
        
        y_train_transformed = y_train.copy()
        y_test_transformed = y_test.copy()
        y_train_transformed = np.array(y_train_transformed)
        y_test_transformed = np.array(y_test_transformed)
        
        viz_ = DecisionViz(model2)
        viz_.fit(X_train_transformed, y_train_transformed, features=['Feature One', 'Feature Two'], classes=['A', 'B'])
        viz_.draw(X_test_transformed, y_test_transformed)
        progress.value += 1
        clear_output()
        viz_.poof()
        
    elif plot == 'rfe':
        
        from yellowbrick.model_selection import RFECV 
        progress.value += 1
        visualizer = RFECV(model, cv=10)
        progress.value += 1
        visualizer.fit(X_train, y_train)
        progress.value += 1
        clear_output()
        visualizer.poof()
           
    elif plot == 'learning':
        
        from yellowbrick.model_selection import LearningCurve
        progress.value += 1
        sizes = np.linspace(0.3, 1.0, 10)  
        visualizer = LearningCurve(model, cv=10, train_sizes=sizes, n_jobs=1, random_state=seed)
        progress.value += 1
        visualizer.fit(X_train, y_train)
        progress.value += 1
        clear_output()
        visualizer.poof()
        
    elif plot == 'manifold':
        
        from yellowbrick.features import Manifold
        
        progress.value += 1
        X_train_transformed = X_train.select_dtypes(include='float64') 
        visualizer = Manifold(manifold='tsne', random_state = seed)
        progress.value += 1
        visualizer.fit_transform(X_train_transformed, y_train)
        progress.value += 1
        clear_output()
        visualizer.poof()       
        
    elif plot == 'calibration':      
                
        from sklearn.calibration import calibration_curve
        
        model_name = str(model).split("(")[0]
        
        plt.figure(figsize=(7, 6))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        progress.value += 1
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
        plt.show() 
        
    elif plot == 'vc':
        
        model_name = str(model).split("(")[0]
        
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
        #max_iter_predict
            
        progress.value += 1
            
        from yellowbrick.model_selection import ValidationCurve
        viz = ValidationCurve(model, param_name=param_name, param_range=param_range,cv=10, 
                              random_state=seed)
        viz.fit(X_train, y_train)
        progress.value += 1
        clear_output()
        viz.poof()
        
    elif plot == 'dimension':
    
        from yellowbrick.features import RadViz
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        progress.value += 1
        X_train_transformed = X_train.select_dtypes(include='float64') 
        X_train_transformed = StandardScaler().fit_transform(X_train_transformed)
        y_train_transformed = np.array(y_train)
        
        features=min(round(len(X_train.columns) * 0.3,0),5)
        features = int(features)
        
        pca = PCA(n_components=features, random_state=seed)
        X_train_transformed = pca.fit_transform(X_train_transformed)
        progress.value += 1
        #classes = ["1", "0"]
        classes = y_train.unique().tolist()
        visualizer = RadViz(classes=classes, alpha=0.25)
        visualizer.fit(X_train_transformed, y_train_transformed)     
        visualizer.transform(X_train_transformed)
        progress.value += 1
        clear_output()
        visualizer.poof()
        
    elif plot == 'feature':
        
        if hasattr(estimator,'coef_'):
            variables = abs(model.coef_[0])
        else:
            variables = abs(model.feature_importances_)
        col_names = np.array(X_train.columns)
        coef_df = pd.DataFrame({'Variable': X_train.columns, 'Value': variables})
        sorted_df = coef_df.sort_values(by='Value')
        sorted_df = sorted_df.sort_values(by='Value', ascending=False)
        sorted_df = sorted_df.head(10)
        sorted_df = sorted_df.sort_values(by='Value')
        my_range=range(1,len(sorted_df.index)+1)
        progress.value += 1
        plt.figure(figsize=(8,5))
        plt.hlines(y=my_range, xmin=0, xmax=sorted_df['Value'], color='skyblue')
        plt.plot(sorted_df['Value'], my_range, "o")
        progress.value += 1
        plt.yticks(my_range, sorted_df['Variable'])
        plt.title("Feature Importance Plot")
        plt.xlabel('Variable Importance')
        plt.ylabel('Features')
        progress.value += 1
        clear_output()
    
    elif plot == 'parameter':
        
        clear_output()
        param_df = pd.DataFrame.from_dict(estimator.get_params(estimator), orient='index', columns=['Parameters'])
        display(param_df)


def compare_models(blacklist = None,
                   fold = 10, 
                   round = 4, 
                   sort = 'Accuracy',
                   turbo = True):
    
    """
      
    Description:
    ------------
    This function uses all models in the model library and scores them using Stratified 
    Cross Validation. The output prints a score grid that shows Accuracy, AUC, Recall,
    Precision, F1 and Kappa by fold (default CV = 10 Folds) of all the available models 
    in the model library.
    
    When turbo is set to True ('rbfsvm', 'gpc' and 'mlp') are excluded due to longer
    training times. By default turbo param is set to True.

    List of models that support binary or multiclass problems in Model Library:

    Estimator                   Abbreviated String     sklearn Implementation 
    ---------                   ------------------     -----------------------
    Logistic Regression         'lr'                   linear_model.LogisticRegression
    K Nearest Neighbour         'knn'                  neighbors.KNeighborsClassifier
    Naives Bayes                'nb'                   naive_bayes.GaussianNB
    Decision Tree               'dt'                   tree.DecisionTreeClassifier
    SVM (Linear)                'svm'                  linear_model.SGDClassifier
    SVM (RBF)                   'rbfsvm'               svm.SVC
    Gaussian Process            'gpc'                  gaussian_process.GPC
    Multi Level Perceptron      'mlp'                  neural_network.MLPClassifier
    Ridge Classifier            'ridge'                linear_model.RidgeClassifier
    Random Forest               'rf'                   ensemble.RandomForestClassifier
    Quadratic Disc. Analysis    'qda'                  discriminant_analysis.QDA 
    AdaBoost                    'ada'                  ensemble.AdaBoostClassifier
    Gradient Boosting           'gbc'                  ensemble.GradientBoostingClassifier
    Linear Disc. Analysis       'lda'                  discriminant_analysis.LDA 
    Extra Trees Classifier      'et'                   ensemble.ExtraTreesClassifier
    Extreme Gradient Boosting   'xgboost'              xgboost.readthedocs.io
    Light Gradient Boosting     'lightgbm'             github.com/microsoft/LightGBM
    CatBoost Classifier         'catboost'             https://catboost.ai

        Example:
        --------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        
        compare_models() 

        This will return the averaged score grid of all the models except 'rbfsvm', 'gpc' 
        and 'mlp'. When turbo param is set to False, all models including 'rbfsvm', 'gpc' 
        and 'mlp' are used but this may result in longer training times.
        
        compare_models( blacklist = [ 'knn', 'gbc' ] , turbo = False) 

        This will return a comparison of all models except K Nearest Neighbour and
        Gradient Boosting Classifier.
        
        compare_models( blacklist = [ 'knn', 'gbc' ] , turbo = True) 

        This will return comparison of all models except K Nearest Neighbour, 
        Gradient Boosting Classifier, SVM (RBF), Gaussian Process Classifier and
        Multi Level Perceptron.
        

    Parameters
    ----------
    blacklist: string, default = None
    In order to omit certain models from the comparison, the abbreviation string 
    (see above list) can be passed as list in blacklist param. This is normally
    done to be more efficient with time. 

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places the metrics in the score grid will be rounded to.
  
    sort: string, default = 'Accuracy'
    The scoring measure specified is used for sorting the average score grid
    Other options are 'AUC', 'Recall', 'Precision', 'F1' and 'Kappa'.

    turbo: Boolean, default = True
    When turbo is set to True, it blacklists estimators that have longer
    training times.
    
    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are Accuracy, AUC, Recall, Precision, F1 
                  and Kappa. Mean and standard deviation of the scores across the 
                  folds are also returned.

    Warnings:
    ---------
    - compare_models() though attractive, might be time consuming with large 
      datasets. By default turbo is set to True, which blacklists models that
      have longer training times. Changing turbo parameter to False may result 
      in very high training times with datasets where number of samples exceed 
      10,000.
      
    - If target variable is multiclass (more than 2 classes), AUC will be 
      returned as zero (0.0)
      
    - This function doesn't return model object.
      
             
    
    """
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #exception checking   
    import sys
    
    #checking error for blacklist (string)
    available_estimators = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 
                            'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']
    
    if blacklist != None:
        for i in blacklist:
            if i not in available_estimators:
                sys.exit('(Value Error): Estimator Not Available. Please see docstring for list of available estimators.')
        
    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    
    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')
 
    #checking sort parameter
    allowed_sort = ['Accuracy', 'Recall', 'Precision', 'F1', 'AUC', 'Kappa']
    if sort not in allowed_sort:
        sys.exit('(Value Error): Sort method not supported. See docstring for list of available parameters.')
    
    #checking optimize parameter for multiclass
    if y.value_counts().count() > 2:
        if sort == 'AUC':
            sys.exit('(Type Error): AUC metric not supported for multiclass problems. See docstring for list of other optimization parameters.')
            
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    #pre-load libraries
    import pandas as pd
    import time, datetime
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    #progress bar
    if blacklist is None:
        len_of_blacklist = 0
    else:
        len_of_blacklist = len(blacklist)
        
    if turbo:
        len_mod = 15 - len_of_blacklist
    else:
        len_mod = 18 - len_of_blacklist
        
    progress = ipw.IntProgress(value=0, min=0, max=(fold*len_mod)+20, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Model', 'Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa'])
    display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['Estimator' , '. . . . . . . . . . . . . . . . . .' , 'Compiling Library' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
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
    
    #defining X_train and y_train as data_X and data_y
    data_X = X_train
    data_y=y_train
    
    progress.value += 1
    
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
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    try:
        import lightgbm as lgb
    except:
        pass
    
   
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
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    #creating model object 
    lr = LogisticRegression(random_state=seed)
    knn = KNeighborsClassifier()
    nb = GaussianNB()
    dt = DecisionTreeClassifier(random_state=seed)
    svm = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed)
    rbfsvm = SVC(gamma='auto', C=1, probability=True, kernel='rbf', random_state=seed)
    gpc = GaussianProcessClassifier(random_state=seed)
    mlp = MLPClassifier(max_iter=500, random_state=seed)
    ridge = RidgeClassifier(random_state=seed)
    rf = RandomForestClassifier(n_estimators=10, random_state=seed)
    qda = QuadraticDiscriminantAnalysis()
    ada = AdaBoostClassifier(random_state=seed)
    gbc = GradientBoostingClassifier(random_state=seed)
    lda = LinearDiscriminantAnalysis()
    et = ExtraTreesClassifier(random_state=seed)
    xgboost = XGBClassifier(random_state=seed, n_jobs=-1, verbosity=0)
    lightgbm = lgb.LGBMClassifier(random_state=seed)
    catboost = CatBoostClassifier(random_state=seed, silent = True) 
    
    progress.value += 1
    
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
    
    
    #checking for blacklist models
    
    model_library_str = ['lr', 'knn', 'nb', 'dt', 'svm', 
                         'rbfsvm', 'gpc', 'mlp', 'ridge', 
                         'rf', 'qda', 'ada', 'gbc', 'lda', 
                         'et', 'xgboost', 'lightgbm', 'catboost']
    
    model_library_str_ = ['lr', 'knn', 'nb', 'dt', 'svm', 
                          'rbfsvm', 'gpc', 'mlp', 'ridge', 
                          'rf', 'qda', 'ada', 'gbc', 'lda', 
                          'et', 'xgboost', 'lightgbm', 'catboost']
    
    if blacklist is not None:
        
        if turbo:
            internal_blacklist = ['rbfsvm', 'gpc', 'mlp']
            compiled_blacklist = blacklist + internal_blacklist
            blacklist = list(set(compiled_blacklist))
            
        else:
            blacklist = blacklist
        
        for i in blacklist:
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
        
        
    if blacklist is None and turbo is True:
        
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
        
            
    progress.value += 1

    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    #cross validation setup starts here
    kf = StratifiedKFold(fold, random_state=seed)

    score_acc =np.empty((0,0))
    score_auc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    score_acc_running = np.empty((0,0)) ##running total
    avg_acc = np.empty((0,0))
    avg_auc = np.empty((0,0))
    avg_recall = np.empty((0,0))
    avg_precision = np.empty((0,0))
    avg_f1 = np.empty((0,0))
    avg_kappa = np.empty((0,0))
    
    name_counter = 0
      
    for model in model_library:
        
        progress.value += 1
        
        '''
        MONITOR UPDATE STARTS
        '''
        monitor.iloc[2,1:] = model_names[name_counter]
        monitor.iloc[3,1:] = 'Calculating ETC'
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        fold_num = 1
        
        for train_i , test_i in kf.split(data_X,data_y):
        
            progress.value += 1
            
            t0 = time.time()
            
            '''
            MONITOR UPDATE STARTS
            '''
                
            monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
            update_display(monitor, display_id = 'monitor')
            
            '''
            MONITOR UPDATE ENDS
            '''            
     
            Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
            ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
        
            if hasattr(model, 'predict_proba'):

                model.fit(Xtrain,ytrain)
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

                kappa = metrics.cohen_kappa_score(ytest,pred_)
                score_acc = np.append(score_acc,sca)
                score_auc = np.append(score_auc,sc)
                score_recall = np.append(score_recall,recall)
                score_precision = np.append(score_precision,precision)
                score_f1 =np.append(score_f1,f1)
                score_kappa =np.append(score_kappa,kappa)

            else:

                model.fit(Xtrain,ytrain)
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

                kappa = metrics.cohen_kappa_score(ytest,pred_)
                score_acc = np.append(score_acc,sca)
                score_auc = np.append(score_auc,sc)
                score_recall = np.append(score_recall,recall)
                score_precision = np.append(score_precision,precision)
                score_f1 =np.append(score_f1,f1)
                score_kappa =np.append(score_kappa,kappa) 
                
                
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
            update_display(monitor, display_id = 'monitor')

            '''
            MONITOR UPDATE ENDS
            '''
        
        avg_acc = np.append(avg_acc,np.mean(score_acc))
        avg_auc = np.append(avg_auc,np.mean(score_auc))
        avg_recall = np.append(avg_recall,np.mean(score_recall))
        avg_precision = np.append(avg_precision,np.mean(score_precision))
        avg_f1 = np.append(avg_f1,np.mean(score_f1))
        avg_kappa = np.append(avg_kappa,np.mean(score_kappa))
        
        compare_models_ = pd.DataFrame({'Model':model_names[name_counter], 'Accuracy':avg_acc, 'AUC':avg_auc, 
                           'Recall':avg_recall, 'Prec.':avg_precision, 
                           'F1':avg_f1, 'Kappa': avg_kappa})
        master_display = pd.concat([master_display, compare_models_],ignore_index=True)
        master_display = master_display.round(round)
        master_display = master_display.sort_values(by=sort,ascending=False)
        master_display.reset_index(drop=True, inplace=True)
        
        update_display(master_display, display_id = display_id)
        
        score_acc =np.empty((0,0))
        score_auc =np.empty((0,0))
        score_recall =np.empty((0,0))
        score_precision =np.empty((0,0))
        score_f1 =np.empty((0,0))
        score_kappa =np.empty((0,0))
        
        avg_acc = np.empty((0,0))
        avg_auc = np.empty((0,0))
        avg_recall = np.empty((0,0))
        avg_precision = np.empty((0,0))
        avg_f1 = np.empty((0,0))
        avg_kappa = np.empty((0,0))
        
        name_counter += 1
  
    progress.value += 1
    
    #storing into experiment
    model_name = 'Compare Models Score Grid'
    tup = (model_name,master_display)
    experiment__.append(tup)
    
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]
    
    
    if y.value_counts().count() > 2:
        
        compare_models_ = master_display.style.apply(highlight_max,subset=['Accuracy','Recall',
                      'Prec.','F1','Kappa'])
    
    else:
        
        compare_models_ = master_display.style.apply(highlight_max,subset=['Accuracy','AUC','Recall',
                      'Prec.','F1','Kappa'])
    compare_models_ = compare_models_.set_properties(**{'text-align': 'left'})
    compare_models_ = compare_models_.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    
    progress.value += 1
    
    clear_output()

    return compare_models_




def tune_model(estimator = None, 
               fold = 10, 
               round = 4, 
               n_iter = 10, 
               optimize = 'Accuracy',
               ensemble = False, 
               method = None,
               verbose = True):
    
      
    """
        
    Description:
    ------------
    This function tunes the hyperparameters of a model and scores it using Stratified 
    Cross Validation. The output prints a score grid that shows Accuracy, AUC, Recall
    Precision, F1 and Kappa by fold (by default = 10 Folds).

    This function returns a trained model object.  

    tune_model() only accepts a string parameter for estimator.

        Example
        -------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        
        tuned_xgboost = tune_model('xgboost') 

        This will tune the hyperparameters of Extreme Gradient Boosting Classifier.


    Parameters
    ----------
    estimator : string, default = None

    Enter abbreviated name of the estimator class. List of estimators supported:

    Estimator                   Abbreviated String     Original Implementation 
    ---------                   ------------------     -----------------------
    Logistic Regression         'lr'                   linear_model.LogisticRegression
    K Nearest Neighbour         'knn'                  neighbors.KNeighborsClassifier
    Naives Bayes                'nb'                   naive_bayes.GaussianNB
    Decision Tree               'dt'                   tree.DecisionTreeClassifier
    SVM (Linear)                'svm'                  linear_model.SGDClassifier
    SVM (RBF)                   'rbfsvm'               svm.SVC
    Gaussian Process            'gpc'                  gaussian_process.GPC
    Multi Level Perceptron      'mlp'                  neural_network.MLPClassifier
    Ridge Classifier            'ridge'                linear_model.RidgeClassifier
    Random Forest               'rf'                   ensemble.RandomForestClassifier
    Quadratic Disc. Analysis    'qda'                  discriminant_analysis.QDA 
    AdaBoost                    'ada'                  ensemble.AdaBoostClassifier
    Gradient Boosting           'gbc'                  ensemble.GradientBoostingClassifier
    Linear Disc. Analysis       'lda'                  discriminant_analysis.LDA 
    Extra Trees Classifier      'et'                   ensemble.ExtraTreesClassifier
    Extreme Gradient Boosting   'xgboost'              xgboost.readthedocs.io
    Light Gradient Boosting     'lightgbm'             github.com/microsoft/LightGBM
    CatBoost Classifier         'catboost'             https://catboost.ai

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places the metrics in the score grid will be rounded to. 

    n_iter: integer, default = 10
    Number of iterations within the Random Grid Search. For every iteration, 
    the model randomly selects one value from the pre-defined grid of hyperparameters.

    optimize: string, default = 'accuracy'
    Measure used to select the best model through hyperparameter tuning.
    The default scoring measure is 'Accuracy'. Other measures include 'AUC',
    'Recall', 'Precision', 'F1'. 

    ensemble: Boolean, default = None
    True enables ensembling of the model through method defined in 'method' param.

    method: String, 'Bagging' or 'Boosting', default = None
    method comes into effect only when ensemble = True. Default is set to None. 

    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are Accuracy, AUC, Recall, Precision, F1 
                  and Kappa. Mean and standard deviation of the scores across the 
                  folds are also returned.

    model:        trained and tuned model object. 
    -----------

    Warnings:
    ---------
    - estimator parameter takes an abbreviated string. Passing a trained model object
      returns an error. The tune_model() function internally calls create_model() 
      before tuning the hyperparameters.
   
    - If target variable is multiclass (more than 2 classes), optimize param 'AUC' is 
      not acceptable.
      
    - If target variable is multiclass (more than 2 classes), AUC will be returned as
      zero (0.0)
        
          
    
  """
 


    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #exception checking   
    import sys
    
    #checking error for estimator (string)
    available_estimators = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 
                            'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']
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
 
    #checking n_iter parameter
    if type(n_iter) is not int:
        sys.exit('(Type Error): n_iter parameter only accepts integer value.')

    #checking optimize parameter
    allowed_optimize = ['Accuracy', 'Recall', 'Precision', 'F1', 'AUC']
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
        
    #checking boosting conflict with estimators
    boosting_not_supported = ['lda','qda','ridge','mlp','gpc','svm','knn', 'catboost']
    if method is 'Boosting' and estimator in boosting_not_supported:
        sys.exit("(Type Error): Estimator does not provide class_weights or predict_proba function and hence not supported for the Boosting method. Change the estimator or method to 'Bagging'.")
    
    
    
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    
    #pre-load libraries
    import pandas as pd
    import time, datetime
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+6, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa'])
    display(progress)    
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id = 'monitor')
    
    if verbose:
        display_ = display(master_display, display_id=True)
        display_id = display_.display_id
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore')    

    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()
    
    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
    
    progress.value += 1
            
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
        
    elif optimize == 'AUC':
        optimize = 'roc_auc'
        
    elif optimize == 'Recall':
        if y.value_counts().count() > 2:
            optimize = metrics.make_scorer(metrics.recall_score, average = 'macro')
        else:
            optimize = 'recall'

    elif optimize == 'Precision':
        if y.value_counts().count() > 2:
            optimize = metrics.make_scorer(metrics.precision_score, average = 'weighted')
        else:
            optimize = 'precision'
   
    elif optimize == 'F1':
        if y.value_counts().count() > 2:
            optimize = metrics.make_scorer(metrics.f1_score, average = 'weighted')
        else:
            optimize = optimize = 'f1'
        
    progress.value += 1
    
    kf = StratifiedKFold(fold, random_state=seed)

    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Tuning Hyperparameters'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    #setting turbo parameters
    cv = 3
        
    if estimator == 'knn':
        
        from sklearn.neighbors import KNeighborsClassifier
        
        param_grid = {'n_neighbors': range(1,51),
                 'weights' : ['uniform', 'distance'],
                 'metric':["euclidean", "manhattan"]
                     }        
        model_grid = RandomizedSearchCV(estimator=KNeighborsClassifier(), param_distributions=param_grid, 
                                        scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       n_jobs=-1, iid=False)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_
 
    elif estimator == 'lr':
        
        from sklearn.linear_model import LogisticRegression

        param_grid = {'C': np.arange(0, 10, 0.001), #[1,5,10,25,50,100],
                  "penalty": [ 'l1', 'l2'],
                  "class_weight": ["balanced", None]
                     }
        model_grid = RandomizedSearchCV(estimator=LogisticRegression(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv, 
                                        random_state=seed, iid=False,n_jobs=-1)
        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'dt':
        
        from sklearn.tree import DecisionTreeClassifier
        
        param_grid = {"max_depth": np.random.randint(1, (len(X_train.columns)*.85),20),
                  "max_features": np.random.randint(3, len(X_train.columns),20),
                  "min_samples_leaf": [2,3,4,5,6],
                  "criterion": ["gini", "entropy"],
                     }

        model_grid = RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=seed), param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       iid=False, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_
 
    elif estimator == 'mlp':
    
        from sklearn.neural_network import MLPClassifier
        
        param_grid = {'learning_rate': ['constant', 'invscaling', 'adaptive'],
                 'solver' : ['lbfgs', 'sgd', 'adam'],
                 'alpha': np.arange(0, 1, 0.0001), #[0.0001, 0.05],
                 'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (100,50,100), (100,100,100)], #np.random.randint(5,15,5),
                 'activation': ["tanh", "identity", "logistic","relu"]
                 }

        model_grid = RandomizedSearchCV(estimator=MLPClassifier(max_iter=1000, random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv, 
                                        random_state=seed, iid=False, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_
    
    elif estimator == 'gpc':
        
        from sklearn.gaussian_process import GaussianProcessClassifier
        
        param_grid = {"max_iter_predict":[100,200,300,400,500,600,700,800,900,1000]}

        model_grid = RandomizedSearchCV(estimator=GaussianProcessClassifier(random_state=seed), param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_    

    elif estimator == 'rbfsvm':
        
        from sklearn.svm import SVC
        
        param_grid = {'C': np.arange(0, 50, 0.01), #[.5,1,10,50,100],
                "class_weight": ["balanced", None]}

        model_grid = RandomizedSearchCV(estimator=SVC(gamma='auto', C=1, probability=True, kernel='rbf', random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_    
  
    elif estimator == 'nb':
        
        from sklearn.naive_bayes import GaussianNB

        param_grid = {'var_smoothing': [0.000000001, 0.000000002, 0.000000005, 0.000000008, 0.000000009,
                                        0.0000001, 0.0000002, 0.0000003, 0.0000005, 0.0000007, 0.0000009, 
                                        0.00001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.009,
                                        0.004, 0.005, 0.006, 0.007,0.008, 0.009, 0.01, 0.1, 1]
                     }

        model_grid = RandomizedSearchCV(estimator=GaussianNB(), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)
 
        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_        

    elif estimator == 'svm':
       
        from sklearn.linear_model import SGDClassifier
        
        param_grid = {'penalty': ['l2', 'l1','elasticnet'],
                      'l1_ratio': np.arange(0,1,0.01), #[0,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                      'alpha': [0.0001, 0.001, 0.01, 0.0002, 0.002, 0.02, 0.0005, 0.005, 0.05],
                      'fit_intercept': [True, False],
                      'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                      'eta0': [0.001, 0.01,0.05,0.1,0.2,0.3,0.4,0.5]
                     }    

        model_grid = RandomizedSearchCV(estimator=SGDClassifier(loss='hinge', random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_     

    elif estimator == 'ridge':
        
        from sklearn.linear_model import RidgeClassifier
        
        param_grid = {'alpha': np.arange(0,1,0.001), #[0.0001,0.001,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                      'fit_intercept': [True, False],
                      'normalize': [True, False]
                     }    

        model_grid = RandomizedSearchCV(estimator=RidgeClassifier(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_     
   
    elif estimator == 'rf':
        
        from sklearn.ensemble import RandomForestClassifier
        
        param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                      'min_samples_split': [2, 5, 7, 9, 10],
                      'min_samples_leaf' : [1, 2, 4],
                      'max_features' : ['auto', 'sqrt', 'log2'],
                      'bootstrap': [True, False]
                     }    

        model_grid = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_     
   
    elif estimator == 'ada':
        
        from sklearn.ensemble import AdaBoostClassifier        

        param_grid = {'n_estimators':  np.arange(10,200,5), #[10, 40, 70, 80, 90, 100, 120, 140, 150],
                      'learning_rate': np.arange(0,1,0.01), #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                      'algorithm' : ["SAMME", "SAMME.R"]
                     }    

        model_grid = RandomizedSearchCV(estimator=AdaBoostClassifier(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_   

    elif estimator == 'gbc':
        
        from sklearn.ensemble import GradientBoostingClassifier

        param_grid = {#'loss': ['deviance', 'exponential'],
                      'n_estimators': np.arange(10,200,5), #[10, 40, 70, 80, 90, 100, 120, 140, 150],
                      'learning_rate': np.arange(0,1,0.01), #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                      'subsample' : np.arange(0.1,1,0.05), #[0.1,0.3,0.5,0.7,0.9,1],
                      'min_samples_split' : [2,4,5,7,9,10],
                      'min_samples_leaf' : [1,2,3,4,5],
                      'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                      'max_features' : ['auto', 'sqrt', 'log2']
                     }    

            
        model_grid = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_   

    elif estimator == 'qda':
        
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        param_grid = {'reg_param': np.arange(0,1,0.01), #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
                     }    

        model_grid = RandomizedSearchCV(estimator=QuadraticDiscriminantAnalysis(), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_      

    elif estimator == 'lda':
        
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        param_grid = {'solver' : ['lsqr', 'eigen'],
                      'shrinkage': [None, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
                     }    

        model_grid = RandomizedSearchCV(estimator=LinearDiscriminantAnalysis(), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_        

    elif estimator == 'et':
        
        from sklearn.ensemble import ExtraTreesClassifier

        param_grid = {'n_estimators': np.arange(10,200,5), #[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                      'min_samples_split': [2, 5, 7, 9, 10],
                      'min_samples_leaf' : [1, 2, 4],
                      'max_features' : ['auto', 'sqrt', 'log2'],
                      'bootstrap': [True, False]
                     }    

        model_grid = RandomizedSearchCV(estimator=ExtraTreesClassifier(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_ 
        
        
    elif estimator == 'xgboost':
        
        from xgboost import XGBClassifier
        
        num_class = y.value_counts().count()
        
        if y.value_counts().count() > 2:
            
            param_grid = {'learning_rate': np.arange(0,1,0.01), #[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                          'n_estimators': np.arange(10,500,20), #[10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
                          'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
                          'colsample_bytree': [0.5, 0.7, 0.9, 1],
                          'min_child_weight': [1, 2, 3, 4],
                          'num_class' : [num_class, num_class]
                         }
        else:
            param_grid = {'learning_rate': np.arange(0,1,0.01), #[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                          'n_estimators':[10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
                          'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
                          'colsample_bytree': [0.5, 0.7, 0.9, 1],
                          'min_child_weight': [1, 2, 3, 4],
                          #'num_class' : [num_class, num_class]
                         }

        model_grid = RandomizedSearchCV(estimator=XGBClassifier(random_state=seed, n_jobs=-1, verbosity=0), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)
        
        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_ 
        
        
    elif estimator == 'lightgbm':
        
        import lightgbm as lgb
        
        param_grid = {#'boosting_type' : ['gbdt', 'dart', 'goss', 'rf'],
                      'num_leaves': [10,20,30,40,50,60,70,80,90,100,150,200],
                      'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                      'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                      'n_estimators': [10, 30, 50, 70, 90, 100, 120, 150, 170, 200], 
                      'min_split_gain' : [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                      'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                      'reg_lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                      }
    
        model_grid = RandomizedSearchCV(estimator=lgb.LGBMClassifier(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_ 
        
        
    elif estimator == 'catboost':
        
        from catboost import CatBoostClassifier
        
        param_grid = {'depth':[3,1,2,6,4,5,7,8,9,10],
                      'iterations':[250,100,500,1000], 
                      'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
                      'l2_leaf_reg':[3,1,5,10,100], 
                      'border_count':[32,5,10,20,50,100,200], 
                      #'ctr_border_count':[50,5,10,20,100,200]
                      }
        
        model_grid = RandomizedSearchCV(estimator=CatBoostClassifier(random_state=seed, silent = True), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_ 
        
        
    progress.value += 1
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Tuning Hyperparameters of Ensemble'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    if estimator == 'dt' and ensemble == True and method == 'Bagging':
        
        from sklearn.ensemble import BaggingClassifier
    
        #when using normal BaggingClassifier() DT estimator raise's an exception for max_features parameter. Hence a separate 
        #call has been made for estimator='dt' and method = 'Bagging' where max_features has been removed from param_grid_dt.
    
        param_grid = {'n_estimators': np.arange(10,300,10), #[10,15,20,25,30],
                     'bootstrap': [True, False],
                     'bootstrap_features': [True, False],
                     }

        param_grid_dt = {"max_depth": np.random.randint(3, (len(X_train.columns)*.85),20),
                      "min_samples_leaf": [2,3,4],
                      "criterion": ["gini", "entropy"]}


        model_grid = RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=seed), param_distributions=param_grid_dt,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       iid=False, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

        best_model = BaggingClassifier(best_model, random_state=seed)

        model_grid = RandomizedSearchCV(estimator=best_model, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, iid=False, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_    
  
        progress.value += 1
    
    elif ensemble and method == 'Bagging':
        
        from sklearn.ensemble import BaggingClassifier
    
        param_grid = {'n_estimators': np.arange(10,300,10), #[10,15,20,25,30],
                     'bootstrap': [True, False],
                     'bootstrap_features': [True, False],
                     }

        best_model = BaggingClassifier(best_model, random_state=seed)

        model_grid = RandomizedSearchCV(estimator=best_model, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, iid=False, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_    
     
    elif ensemble and method =='Boosting':
        
        from sklearn.ensemble import AdaBoostClassifier
        
        param_grid = {'n_estimators': np.arange(10,200,10), #[25,35,50,60,70,75],
                     'learning_rate': np.arange(0,1,0.01), #[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2],
                     }        

        best_model = AdaBoostClassifier(best_model, random_state=seed)

        model_grid = RandomizedSearchCV(estimator=best_model, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, iid=False, n_jobs=-1)

    progress.value += 1

        
    #multiclass checking
    if y.value_counts().count() > 2:
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(model)
        
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    fold_num = 1
    
    for train_i , test_i in kf.split(data_X,data_y):
        
        t0 = time.time()
        
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
    
        if hasattr(model, 'predict_proba'):
        
            model.fit(Xtrain,ytrain)
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
                
            kappa = metrics.cohen_kappa_score(ytest,pred_)
            score_acc = np.append(score_acc,sca)
            score_auc = np.append(score_auc,sc)
            score_recall = np.append(score_recall,recall)
            score_precision = np.append(score_precision,precision)
            score_f1 =np.append(score_f1,f1)
            score_kappa =np.append(score_kappa,kappa)

        else:
            
            model.fit(Xtrain,ytrain)
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

            kappa = metrics.cohen_kappa_score(ytest,pred_)
            score_acc = np.append(score_acc,sca)
            score_auc = np.append(score_auc,sc)
            score_recall = np.append(score_recall,recall)
            score_precision = np.append(score_precision,precision)
            score_f1 =np.append(score_f1,f1)
            score_kappa =np.append(score_kappa,kappa)             
        
        progress.value += 1
            
            
        '''
        
        This section is created to update_display() as code loops through the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa]}).round(round)
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
            
        update_display(ETC, display_id = 'ETC')
            
        fold_num += 1
        
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[2,1:] = ETC
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
       
        '''
        
        TIME CALCULATION ENDS HERE
        
        '''
        
        if verbose:
            update_display(master_display, display_id = display_id)
        
        '''
        
        Update_display() ends here
        
        '''
        
    progress.value += 1
    
    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)

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

    progress.value += 1
    
    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)

    progress.value += 1
    
    #refitting the model on complete X_train, y_train
    monitor.iloc[1,1:] = 'Compiling Final Model'
    update_display(monitor, display_id = 'monitor')
    
    best_model.fit(data_X, data_y)
    
    progress.value += 1
    
    #storing into experiment
    model_name = 'Tuned ' + str(model).split("(")[0]
    tup = (model_name,best_model)
    experiment__.append(tup)
    nam = str(model_name) + ' Score Grid'
    tup = (nam, model_results)
    experiment__.append(tup)
    
    
    if verbose:
        clear_output()
        display(model_results)
        return best_model
    else:
        clear_output()
        return best_model



def blend_models(estimator_list = 'All', 
                 fold = 10, 
                 round = 4, 
                 method = 'hard',
                 turbo = True,
                 verbose = True):
    
    """
        
    Description:
    ------------
    This function creates a Soft Voting / Majority Rule classifier for all the 
    estimators in the model library (excluding the few when turbo is True) or 
    for specific trained estimators passed as a list in estimator_list param.
    It scores it using Stratified Cross Validation. The output prints a score
    grid that shows Accuracy,  AUC, Recall, Precision, F1 and Kappa by fold 
    (default CV = 10 Folds). 

    This function returns a trained model object.  

        Example:
        --------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        
        blend_all = blend_models() 

        This will create a VotingClassifier for all models in the model library 
        except for 'rbfsvm', 'gpc' and 'mlp'.

        For specific models, you can use:

        lr = create_model('lr')
        rf = create_model('rf')
        knn = create_model('knn')

        blend_three = blend_models(estimator_list = [lr,rf,knn])
    
        This will create a VotingClassifier of lr, rf and knn.

    Parameters
    ----------
    estimator_list : string ('All') or list of object, default = 'All'

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places the metrics in the score grid will be rounded to.

    method: string, default = 'hard'
    'hard' uses predicted class labels for majority rule voting.'soft', predicts 
    the class label based on the argmax of the sums of the predicted probabilities, 
    which is recommended for an ensemble of well-calibrated classifiers. 

    turbo: Boolean, default = True
    When turbo is set to True, it blacklists estimator that uses Radial Kernel.

    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are Accuracy, AUC, Recall, Precision, F1 
                  and Kappa. Mean and standard deviation of the scores across the 
                  folds are also returned.

    model:        trained Voting Classifier model object. 
    -----------

    Warnings:
    ---------
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
    
    #exception checking   
    import sys
    
    #checking error for estimator_list (string)
    
    if estimator_list != 'All':
        for i in estimator_list:
            if 'sklearn' not in str(type(i)) and 'CatBoostClassifier' not in str(type(i)):
                sys.exit("(Value Error): estimator_list parameter only accepts 'All' as string or trained model object")
   
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
    
    #testing
    #no active testing
    
    #pre-load libraries
    import pandas as pd
    import time, datetime
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+4, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa'])
    display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id = 'monitor')
    
    if verbose:
        display_ = display(master_display, display_id=True)
        display_id = display_.display_id
        
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #general dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold  
    from sklearn.ensemble import VotingClassifier
    import re
    
    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()
    
    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
    
    progress.value += 1
    
    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    avg_acc = np.empty((0,0))
    avg_auc = np.empty((0,0))
    avg_recall = np.empty((0,0))
    avg_precision = np.empty((0,0))
    avg_f1 = np.empty((0,0))
    avg_kappa = np.empty((0,0))

    kf = StratifiedKFold(fold, random_state=seed)
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Compiling Estimators'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    if estimator_list == 'All':

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
        
        #from catboost import CatBoostClassifier
        
        #creating CatBoost estimator
        lr = LogisticRegression(random_state=seed)
        knn = KNeighborsClassifier()
        nb = GaussianNB()
        dt = DecisionTreeClassifier(random_state=seed)
        svm = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed)
        rbfsvm = SVC(gamma='auto', C=1, probability=True, kernel='rbf', random_state=seed)
        gpc = GaussianProcessClassifier(random_state=seed)
        mlp = MLPClassifier(max_iter=500, random_state=seed)
        ridge = RidgeClassifier(random_state=seed)
        rf = RandomForestClassifier(n_estimators=10, random_state=seed)
        qda = QuadraticDiscriminantAnalysis()
        ada = AdaBoostClassifier(random_state=seed)
        gbc = GradientBoostingClassifier(random_state=seed)
        lda = LinearDiscriminantAnalysis()
        et = ExtraTreesClassifier(random_state=seed)
        xgboost = XGBClassifier(random_state=seed, n_jobs=-1, verbosity=0)
        lightgbm = lgb.LGBMClassifier(random_state=seed)
        #catboost = CatBoostClassifier(random_state=seed, silent = True)
        
        progress.value += 1
        
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
        model = VotingClassifier(estimators=estimator_list_, voting=voting, n_jobs=-1)
        model.fit(Xtrain,ytrain)
    except:
        model = VotingClassifier(estimators=estimator_list_, voting=voting)
    
    progress.value += 1
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    fold_num = 1
    
    for train_i , test_i in kf.split(data_X,data_y):
        
        progress.value += 1
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
    
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]    
    
        if voting == 'hard':
        
            model.fit(Xtrain,ytrain)
            pred_prob = 0.0
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
                
            kappa = metrics.cohen_kappa_score(ytest,pred_)
            score_acc = np.append(score_acc,sca)
            score_auc = np.append(score_auc,sc)
            score_recall = np.append(score_recall,recall)
            score_precision = np.append(score_precision,precision)
            score_f1 =np.append(score_f1,f1)
            score_kappa =np.append(score_kappa,kappa)
        
        else:
        
            model.fit(Xtrain,ytrain)
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
                
            kappa = metrics.cohen_kappa_score(ytest,pred_)
            
            score_acc = np.append(score_acc,sca)
            score_auc = np.append(score_auc,sc)
            score_recall = np.append(score_recall,recall)
            score_precision = np.append(score_precision,precision)
            score_f1 =np.append(score_f1,f1)
            score_kappa =np.append(score_kappa,kappa)
    
    
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa]}).round(round)
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
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        '''
        TIME CALCULATION ENDS HERE
        '''
        
        if verbose:
            update_display(master_display, display_id = display_id)
            
        
        '''
        
        Update_display() ends here
        
        '''
    
    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)

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
    
    progress.value += 1
    
    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    
    progress.value += 1
    
    #refitting the model on complete X_train, y_train
    monitor.iloc[1,1:] = 'Compiling Final Model'
    update_display(monitor, display_id = 'monitor')
    
    model.fit(data_X, data_y)
    
    progress.value += 1
    
    #storing into experiment
    model_name = 'Voting Classifier'
    tup = (model_name,model)
    experiment__.append(tup)
    nam = str(model_name) + ' Score Grid'
    tup = (nam, model_results)
    experiment__.append(tup)
    
    if verbose:
        clear_output()
        display(model_results)
        return model
    
    else:
        clear_output()
        return model



def stack_models(estimator_list, 
                 meta_model = None, 
                 fold = 10,
                 round = 4, 
                 method = 'soft', 
                 restack = True, 
                 plot = False,
                 finalize = False,
                 verbose = True):
    
    """
            
    Description:
    ------------
    This function creates a meta model and scores it using Stratified Cross Validation.
    The predictions from the base level models as passed in the estimator_list param 
    are used as input features for the meta model. The restacking parameter controls
    the ability to expose raw features to the meta model when set to True
    (default = False).

    The output prints the score grid that shows Accuracy, AUC, Recall, Precision, 
    F1 and Kappa by fold (default = 10 Folds). 
    
    This function returns a container which is the list of all models in stacking. 

        Example:
        --------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        dt = create_model('dt')
        rf = create_model('rf')
        ada = create_model('ada')
        ridge = create_model('ridge')
        knn = create_model('knn')

        stacked_models = stack_models(estimator_list=[dt,rf,ada,ridge,knn])

        This will create a meta model that will use the predictions of all the 
        models provided in estimator_list param. By default, the meta model is 
        Logistic Regression but can be changed with meta_model param.

    Parameters
    ----------
    estimator_list : list of objects

    meta_model : object, default = None
    if set to None, Logistic Regression is used as a meta model.

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places the metrics in the score grid will be rounded to.

    method: string, default = 'soft'
    'soft', uses predicted probabilities as an input to the meta model.
    'hard', uses predicted class labels as an input to the meta model. 

    restack: Boolean, default = True
    When restack is set to True, raw data will be exposed to meta model when
    making predictions, otherwise when False, only the predicted label or
    probabilities is passed to meta model when making final predictions.

    plot: Boolean, default = False
    When plot is set to True, it will return the correlation plot of prediction
    from all base models provided in estimator_list.
    
    finalize: Boolean, default = False
    When finalize is set to True, it will fit the stacker on entire dataset
    including the hold-out sample created during the setup() stage. It is not 
    recommended to set this to True here, If you would like to fit the stacker 
    on the entire dataset including the hold-out, use finalize_model().
    
    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are Accuracy, AUC, Recall, Precision, F1 
                  and Kappa. Mean and standard deviation of the scores across the 
                  folds are also returned.

    container:    list of all the models where last element is meta model.
    ----------

    Warnings:
    ---------
    -  When the method is forced to be 'soft' and estimator_list param includes 
       estimators that donot support the predict_proba method such as 'svm' or 
       'ridge',  predicted values for those specific estimators only are used 
       instead of probability  when building the meta_model. The same rule applies
       when the stacker is used under predict_model() function.
        
    -  If target variable is multiclass (more than 2 classes), AUC will be returned 
       as zero (0.0).
       
    -  method 'soft' not supported for when target is multiclass.
         
            
    """
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #testing
    #no active test
    
    #exception checking   
    import sys
    
    #checking error for estimator_list
    for i in estimator_list:
        if 'sklearn' not in str(type(i)) and 'CatBoostClassifier' not in str(type(i)):
            sys.exit("(Value Error): estimator_list parameter only trained model object")
            
    #checking meta model
    if meta_model is not None:
        if 'sklearn' not in str(type(meta_model)) and 'CatBoostClassifier' not in str(type(meta_model)):
            sys.exit("(Value Error): estimator_list parameter only accepts trained model object")
    
    #stacking with multiclass
    if y.value_counts().count() > 2:
        if method == 'soft':
            sys.exit("(Type Error): method 'soft' not supported for multiclass problems.")
            
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
    
    #checking restack parameter
    if type(restack) is not bool:
        sys.exit('(Type Error): Restack parameter can only take argument as True or False.')    
    
    #checking plot parameter
    if type(restack) is not bool:
        sys.exit('(Type Error): Plot parameter can only take argument as True or False.')  
        
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.') 
        
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    #testing
    #no active test
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import time, datetime
    from copy import deepcopy
    from sklearn.base import clone
    
    #copy estimator_list
    estimator_list = deepcopy(estimator_list)
    
    #Defining meta model.
    if meta_model == None:
        from sklearn.linear_model import LogisticRegression
        meta_model = LogisticRegression()
    else:
        meta_model = deepcopy(meta_model)
        
    clear_output()
        
    #progress bar
    max_progress = len(estimator_list) + fold + 4
    progress = ipw.IntProgress(value=0, min=0, max=max_progress, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa'])
    display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id = 'monitor')
    
    if verbose:
        display_ = display(master_display, display_id=True)
        display_id = display_.display_id
        
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_predict
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    progress.value += 1
    
    #Capturing the method of stacking required by user. method='soft' means 'predict_proba' else 'predict'
    if method == 'soft':
        predict_method = 'predict_proba'
    elif method == 'hard':
        predict_method = 'predict'
    
    
    #defining data_X and data_y
    if finalize:
        data_X = X.copy()
        data_y = y.copy()
    else:       
        data_X = X_train.copy()
        data_y = y_train.copy()
        
    #reset index
    data_X.reset_index(drop=True,inplace=True)
    data_y.reset_index(drop=True,inplace=True)
    
    #models_ for appending
    models_ = []
    
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
    
    base_array = np.zeros((0,0))
    base_prediction = pd.DataFrame(data_y) #changed to data_y
    base_prediction = base_prediction.reset_index(drop=True)
    
    counter = 0
    
    for model in estimator_list:
        
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[1,1:] = 'Evaluating ' + model_names[counter]
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        #fitting and appending
        model.fit(data_X, data_y)
        models_.append(model)
        
        progress.value += 1
        
        try:
            base_array = cross_val_predict(model,data_X,data_y,cv=fold, method=predict_method)
        except:
            base_array = cross_val_predict(model,data_X,data_y,cv=fold, method='predict')
        if method == 'soft':
            try:
                base_array = base_array[:,1]
            except:
                base_array = base_array
        elif method == 'hard':
            base_array = base_array
        base_array_df = pd.DataFrame(base_array)
        base_prediction = pd.concat([base_prediction,base_array_df],axis=1)
        base_array = np.empty((0,0))
        
        counter += 1
        
    #fill nas for base_prediction
    base_prediction.fillna(value=0, inplace=True)
    
    #defining column names now
    target_col_name = np.array(base_prediction.columns[0])
    model_names = np.append(target_col_name, model_names_fixed) #added fixed here
    base_prediction.columns = model_names #defining colum names now
    
    #defining data_X and data_y dataframe to be used in next stage.
    
    #drop column from base_prediction
    base_prediction.drop(base_prediction.columns[0],axis=1,inplace=True)
    
    if restack:
        data_X = pd.concat([data_X, base_prediction], axis=1)
        
    else:
        data_X = base_prediction
    
    #Correlation matrix of base_prediction
    #base_prediction_cor = base_prediction.drop(base_prediction.columns[0],axis=1)
    base_prediction_cor = base_prediction.corr()
    
    #Meta Modeling Starts Here
    model = meta_model #this defines model to be used below as model = meta_model (as captured above)
    
    #appending in models
    model.fit(data_X, data_y)
    models_.append(model)
    
    kf = StratifiedKFold(fold, random_state=seed) #capturing fold requested by user

    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    
    progress.value += 1
    
    fold_num = 1
    
    for train_i , test_i in kf.split(data_X,data_y):
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Meta Model Fold ' + str(fold_num) + ' of ' + str(fold)
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        progress.value += 1
        
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]

        model.fit(Xtrain,ytrain)
        
        try:
            pred_prob = model.predict_proba(Xtest)
            pred_prob = pred_prob[:,1]
        except:
            pass
        pred_ = model.predict(Xtest)
        sca = metrics.accuracy_score(ytest,pred_)
        try: 
            sc = metrics.roc_auc_score(ytest,pred_prob)
        except:
            sc = 0
            
        if y.value_counts().count() > 2:
            recall = metrics.recall_score(ytest,pred_,average='macro')
            precision = metrics.precision_score(ytest,pred_,average='weighted')
            f1 = metrics.f1_score(ytest,pred_,average='weighted')
            
        else:
            recall = metrics.recall_score(ytest,pred_)
            precision = metrics.precision_score(ytest,pred_)
            f1 = metrics.f1_score(ytest,pred_)
            
        kappa = metrics.cohen_kappa_score(ytest,pred_)
        score_acc = np.append(score_acc,sca)
        score_auc = np.append(score_auc,sc)
        score_recall = np.append(score_recall,recall)
        score_precision = np.append(score_precision,precision)
        score_f1 =np.append(score_f1,f1)
        score_kappa =np.append(score_kappa,kappa)
        
        
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa]}).round(round)
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
        
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        #update_display(ETC, display_id = 'ETC')
            
        fold_num += 1
        
        
        '''
        
        TIME CALCULATION ENDS HERE
        
        '''
        
        if verbose:
            update_display(master_display, display_id = display_id)
            
        
        '''
        
        Update_display() ends here
        
        '''
     
    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    
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
      
    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa},index=['Mean', 'SD'])
  
    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)  
    
    progress.value += 1
    
    #appending method into models_
    models_.append(method)
    models_.append(restack)
    
    #storing into experiment
    model_name = 'Stacking Classifier (Single Layer)'
    tup = (model_name,models_)
    experiment__.append(tup)
    nam = str(model_name) + ' Score Grid'
    tup = (nam, model_results)
    experiment__.append(tup)
    
    if plot:
        clear_output()
        plt.subplots(figsize=(15,7))
        ax = sns.heatmap(base_prediction_cor, vmin=0.2, vmax=1, center=0,cmap='magma', square=True, annot=True, 
                         linewidths=1)
        ax.set_ylim(sorted(ax.get_xlim(), reverse=True))

    if verbose:
        clear_output()
        display(model_results)
        return models_
    else:
        clear_output()
        return models_



def create_stacknet(estimator_list,
                    meta_model = None,
                    fold = 10,
                    round = 4,
                    method = 'soft',
                    restack = True,
                    finalize = False,
                    verbose = True):
    
    """
         
    Description:
    ------------
    This function creates a sequential stack net using cross validated predictions 
    at each layer. The final score grid contains predictions from the meta model 
    using Stratified Cross Validation. Base level models can be passed as 
    estimator_list param, the layers can be organized as a sub list within the 
    estimator_list object.  Restacking param controls the ability to expose raw 
    features to meta model.

        Example:
        --------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        dt = create_model('dt')
        rf = create_model('rf')
        ada = create_model('ada')
        ridge = create_model('ridge')
        knn = create_model('knn')

        stacknet = create_stacknet(estimator_list =[[dt,rf],[ada,ridge,knn]])

        This will result in the stacking of models in multiple layers. The first layer 
        contains dt and rf, the predictions of which are used by models in the second 
        layer to generate predictions which are then used by the meta model to generate
        final predictions. By default, the meta model is Logistic Regression but can be 
        changed with meta_model param.

    Parameters
    ----------
    estimator_list : nested list of objects

    meta_model : object, default = None
    if set to None, Logistic Regression is used as a meta model.

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places the metrics in the score grid will be rounded to.
  
    method: string, default = 'soft'
    'soft', uses predicted probabilities as an input to the meta model.
    'hard', uses predicted class labels as an input to the meta model. 
    
    restack: Boolean, default = True
    When restack is set to True, raw data and prediction of all layers will be 
    exposed to the meta model when making predictions. When set to False, only 
    the predicted label or probabilities of last layer is passed to meta model 
    when making final predictions.
    
    finalize: Boolean, default = False
    When finalize is set to True, it will fit the stacker on entire dataset
    including the hold-out sample created during the setup() stage. It is not 
    recommended to set this to True here, if you would like to fit the stacker 
    on the entire dataset including the hold-out, use finalize_model().
    
    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are Accuracy, AUC, Recall, Precision, F1 
                  and Kappa. Mean and standard deviation of the scores across the 
                  folds are also returned.

    container:    list of all models where the last element is the meta model.
    ----------

    Warnings:
    ---------
    -  When the method is forced to be 'soft' and estimator_list param includes 
       estimators that donot support the predict_proba method such as 'svm' or 
       'ridge',  predicted values for those specific estimators only are used 
       instead of probability  when building the meta_model. The same rule applies
       when the stacker is used under predict_model() function.
    
    -  If target variable is multiclass (more than 2 classes), AUC will be returned 
       as zero (0.0)
       
    -  method 'soft' not supported for when target is multiclass.
    
      
    """

    
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #testing
    #global inter_level_names
    
    #exception checking   
    import sys
    
    #checking estimator_list
    if type(estimator_list[0]) is not list:
        sys.exit("(Type Error): estimator_list parameter must be list of list. ")
        
    #blocking stack_models usecase
    if len(estimator_list) == 1:
        sys.exit("(Type Error): Single Layer stacking must be performed using stack_models(). ")
        
    #checking error for estimator_list
    for i in estimator_list:
        for j in i:
            if 'sklearn' not in str(type(j)) and 'CatBoostClassifier' not in str(type(j)):
                sys.exit("(Value Error): estimator_list parameter only trained model object")
    
    #checking meta model
    if meta_model is not None:
        if 'sklearn' not in str(type(meta_model)) and 'CatBoostClassifier' not in str(type(meta_model)):
            sys.exit("(Value Error): estimator_list parameter only trained model object")
    
    #stacknet with multiclass
    if y.value_counts().count() > 2:
        if method == 'soft':
            sys.exit("(Type Error): method 'soft' not supported for multiclass problems.")
        
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
    
    #checking restack parameter
    if type(restack) is not bool:
        sys.exit('(Type Error): Restack parameter can only take argument as True or False.')    
    
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.') 
        
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import time, datetime
    from copy import deepcopy
    from sklearn.base import clone
    
    #copy estimator_list
    estimator_list = deepcopy(estimator_list)
    
    #copy meta_model
    if meta_model is None:
        from sklearn.linear_model import LogisticRegression
        meta_model = LogisticRegression()
    else:
        meta_model = deepcopy(meta_model)
        
    clear_output()
    
    #progress bar
    max_progress = len(estimator_list) + fold + 4
    progress = ipw.IntProgress(value=0, min=0, max=max_progress, step=1 , description='Processing: ')
    display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id = 'monitor')
    
    if verbose:
        master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa'])
        display_ = display(master_display, display_id=True)
        display_id = display_.display_id
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #models_ list
    models_ = []
    
    #general dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_predict
    
    progress.value += 1
    
    base_level = estimator_list[0]
    base_level_names = []
    
    #defining base_level_names
    for item in base_level:
        base_level_names = np.append(base_level_names, str(item).split("(")[0])
        
    base_level_fixed = []
    
    for i in base_level_names:
        if 'CatBoostClassifier' in i:
            a = 'CatBoostClassifier'
            base_level_fixed.append(a)
    else:
        base_level_fixed.append(i)
        
    base_level_fixed_2 = []
    
    counter = 0
    for i in base_level_names:
        s = str(i) + '_' + 'BaseLevel_' + str(counter)
        base_level_fixed_2.append(s)
        counter += 1
    
    base_level_fixed = base_level_fixed_2
    
    inter_level = estimator_list[1:]
    inter_level_names = []
   
    #defining inter_level names
    for item in inter_level:
        level_list=[]
        for m in item:
            if 'CatBoostClassifier' in str(m).split("(")[0]:
                level_list.append('CatBoostClassifier')
            else:
                level_list.append(str(m).split("(")[0])
        inter_level_names.append(level_list)
    
    #defining data_X and data_y
    if finalize:
        data_X = X.copy()
        data_y = y.copy()
    else:       
        data_X = X_train.copy()
        data_y = y_train.copy()
    
    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
    
    
    #Capturing the method of stacking required by user. method='soft' means 'predict_proba' else 'predict'
    if method == 'soft':
        predict_method = 'predict_proba'
    elif method == 'hard':
        predict_method = 'predict'
        
    base_array = np.zeros((0,0))
    base_array_df = pd.DataFrame()
    base_prediction = pd.DataFrame(data_y) #change to data_y
    base_prediction = base_prediction.reset_index(drop=True)
    
    base_counter = 0
    
    base_models_ = []
    
    for model in base_level:
        
        base_models_.append(model.fit(data_X,data_y)) #changed to data_X and data_y
        
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[1,1:] = 'Evaluating ' + base_level_names[base_counter]
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        progress.value += 1
        
        if method == 'soft':
            try:
                base_array = cross_val_predict(model,data_X,data_y,cv=fold, method=predict_method)
                base_array = base_array[:,1]
            except:
                base_array = cross_val_predict(model,data_X,data_y,cv=fold, method='predict')
        else:
            base_array = cross_val_predict(model,data_X,data_y,cv=fold, method='predict')
            
        base_array = pd.DataFrame(base_array)
        base_array_df = pd.concat([base_array_df, base_array], axis=1)
        base_array = np.empty((0,0))  
        
        base_counter += 1
    
    base_array_df.fillna(value=0, inplace=True) #fill na's with zero
    base_array_df.columns = base_level_fixed
    
    if restack:
        base_array_df = pd.concat([data_X,base_array_df], axis=1)
        
    early_break = base_array_df.copy()
    
    models_.append(base_models_)
    
    inter_counter = 0
    
    for level in inter_level:
        inter_inner = []
        model_counter = 0
        inter_array_df = pd.DataFrame()
        
        for model in level:
            
            '''
            MONITOR UPDATE STARTS
            '''

            monitor.iloc[1,1:] = 'Evaluating ' + inter_level_names[inter_counter][model_counter]
            update_display(monitor, display_id = 'monitor')

            '''
            MONITOR UPDATE ENDS
            '''
            
            model = clone(model)
            inter_inner.append(model.fit(X = base_array_df, y = data_y)) #changed to data_y
            
            if method == 'soft':
                try:
                    base_array = cross_val_predict(model, X = base_array_df, y = data_y, cv=fold, method=predict_method)
                    base_array = base_array[:,1]
                except:
                    base_array = cross_val_predict(model, X = base_array_df, y = data_y, cv=fold, method='predict')
                    
            
            else:
                base_array = cross_val_predict(model, X = base_array_df, y = data_y, cv=fold, method='predict')
                
            base_array = pd.DataFrame(base_array)
            
            """
            defining columns
            """
            
            col = str(model).split("(")[0]
            if 'CatBoostClassifier' in col:
                col = 'CatBoostClassifier'
            col = col + '_InterLevel_' + str(inter_counter) + '_' + str(model_counter)
            base_array.columns = [col]
            
            """
            defining columns end here
            """
            
            inter_array_df = pd.concat([inter_array_df, base_array], axis=1)
            base_array = np.empty((0,0))
            
            model_counter += 1
            
        base_array_df = pd.concat([base_array_df,inter_array_df], axis=1)
        base_array_df.fillna(value=0, inplace=True) #fill na's with zero
        
        models_.append(inter_inner)
    
        if restack == False:
            i = base_array_df.shape[1] - len(level)
            base_array_df = base_array_df.iloc[:,i:]
        
        inter_counter += 1
        progress.value += 1
        
    model = meta_model
    
    #redefine data_X and data_y
    data_X = base_array_df.copy()
    
    meta_model_ = model.fit(data_X,data_y)
    
    kf = StratifiedKFold(fold, random_state=seed) #capturing fold requested by user

    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    
    fold_num = 1
    
    for train_i , test_i in kf.split(data_X,data_y):
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Meta Model Fold ' + str(fold_num) + ' of ' + str(fold)
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
        
        model.fit(Xtrain,ytrain)
        try:
            pred_prob = model.predict_proba(Xtest)
            pred_prob = pred_prob[:,1]
        except:
            pass
        pred_ = model.predict(Xtest)
        sca = metrics.accuracy_score(ytest,pred_)
        try:
            sc = metrics.roc_auc_score(ytest,pred_prob)
        except:
            sc = 0
            
        if y.value_counts().count() > 2:
            recall = metrics.recall_score(ytest,pred_,average='macro')
            precision = metrics.precision_score(ytest,pred_,average='weighted')
            f1 = metrics.f1_score(ytest,pred_,average='weighted')
            
        else:
            recall = metrics.recall_score(ytest,pred_)
            precision = metrics.precision_score(ytest,pred_)
            f1 = metrics.f1_score(ytest,pred_) 
            
        kappa = metrics.cohen_kappa_score(ytest,pred_)
        score_acc = np.append(score_acc,sca)
        score_auc = np.append(score_auc,sc)
        score_recall = np.append(score_recall,recall)
        score_precision = np.append(score_precision,precision)
        score_f1 =np.append(score_f1,f1)
        score_kappa =np.append(score_kappa,kappa)

        progress.value += 1
        
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa]}).round(round)
        
        if verbose:
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
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        '''
        TIME CALCULATION ENDS HERE
        '''
        
        if verbose:
            update_display(master_display, display_id = display_id)
            
        
        '''
        
        Update_display() ends here
        
        '''
        
    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    
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
    
    progress.value += 1
    
    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa},index=['Mean', 'SD'])
  
    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)      
    
    progress.value += 1
        
    
    #appending meta_model into models_
    models_.append(meta_model_)
        
    #appending method into models_
    models_.append([str(method)])
    
    #appending restack param
    models_.append(restack)
    
    #storing into experiment
    model_name = 'Stacking Classifier (Multi Layer)'
    tup = (model_name,models_)
    experiment__.append(tup)
    nam = str(model_name) + ' Score Grid'
    tup = (nam, model_results)
    experiment__.append(tup)
    
    if verbose:
        clear_output()
        display(model_results)
        return models_
    
    else:
        clear_output()
        return models_  




def interpret_model(estimator,
                   plot = 'summary',
                   feature = None, 
                   observation = None):
    
    
    """
          
    Description:
    ------------
    This function takes a trained model object and returns an interpretation plot 
    based on the test / hold-out set. It only supports tree based algorithms. 

    This function is implemented based on the SHAP (SHapley Additive exPlanations),
    which is a unified approach to explain the output of any machine learning model. 
    SHAP connects game theory with local explanations.

    For more information : https://shap.readthedocs.io/en/latest/

        Example
        -------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        dt = create_model('dt')
        
        interpret_model(dt)

        This will return a summary interpretation plot of Decision Tree model.

    Parameters
    ----------
    estimator : object, default = none
    A trained tree based model object should be passed as an estimator. 

    plot : string, default = 'summary'
    other available options are 'correlation' and 'reason'.

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

    Returns:
    --------

    Visual Plot:  Returns the visual plot.
    -----------   Returns the interactive JS plot when plot = 'reason'.

    Warnings:
    --------- 
    - interpret_model doesn't support multiclass problems.
      
         
         
    """
    
    
    
    '''
    Error Checking starts here
    
    '''
    
    import sys
    
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
        
        if model_name in type1:
        
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test)
            
        elif model_name in type2:
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test)
                              
    elif plot == 'correlation':
        
        if feature == None:
            
            dependence = X_test.columns[0]
            
        else:
            
            dependence = feature
        
        if model_name in type1:
                
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.dependence_plot(dependence, shap_values[1], X_test)
        
        elif model_name in type2:
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test) 
            shap.dependence_plot(dependence, shap_values, X_test)
        
    elif plot == 'reason':
        
        if model_name in type1:
            
            if observation is None:
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                shap.initjs()
                return shap.force_plot(explainer.expected_value[1], shap_values[1], X_test)
            
            else: 
                
                if model_name == 'LGBMClassifier':
                    
                    row_to_show = observation
                    data_for_prediction = X_test.iloc[row_to_show]
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)
                    shap.initjs()
                    return shap.force_plot(explainer.expected_value[1], shap_values[0][row_to_show], data_for_prediction)    
                
                else:
                    
                    row_to_show = observation
                    data_for_prediction = X_test.iloc[row_to_show]
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(data_for_prediction)
                    shap.initjs()
                    return shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)        

            
        elif model_name in type2:

            if observation is None:
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                shap.initjs()
                return shap.force_plot(explainer.expected_value, shap_values, X_test)  
                
            else:
                
                row_to_show = observation
                data_for_prediction = X_test.iloc[row_to_show]
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                shap.initjs()
                return shap.force_plot(explainer.expected_value, shap_values[row_to_show,:], X_test.iloc[row_to_show,:])



def calibrate_model(estimator,
                    method = 'sigmoid',
                    fold=10,
                    round=4,
                    verbose=True):
    
    """  
     
    Description:
    ------------
    This function takes the input of trained estimator and performs probability 
    calibration with sigmoid or isotonic regression. The output prints a score 
    grid that shows Accuracy, AUC, Recall, Precision, F1 and Kappa by fold 
    (default = 10 Fold). The ouput of the original estimator and the calibrated 
    estimator (created using this function) might not differ much. In order 
    to see the calibration differences, use 'calibration' plot in plot_model to 
    see the difference before and after.

    This function returns a trained model object. 

        Example
        -------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        dt_boosted = create_model('dt', ensemble = True, method = 'Boosting')
        
        calibrated_dt = calibrate_model(dt_boosted)

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

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are Accuracy, AUC, Recall, Precision, F1 
                  and Kappa. Mean and standard deviation of the scores across the 
                  folds are also returned.

    model:        trained and calibrated model object.
    -----------

    Warnings:
    ---------
    - Avoid isotonic calibration with too few calibration samples (<1000) since it 
      tends to overfit.
      
    - calibration plot not available for multiclass problems.
      
    
  
    """


    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #exception checking   
    import sys
    
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
    
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
        
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+4, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Accuracy','AUC','Recall', 'Prec.', 'F1', 'Kappa'])
    display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id = 'monitor')
    
    if verbose:
        display_ = display(master_display, display_id=True)
        display_id = display_.display_id
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train.copy()
    data_y = y_train.copy()
    
    #reset index
    data_X.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)
  
    #general dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    
    progress.value += 1
    
    #cross validation setup starts here
    kf = StratifiedKFold(fold, random_state=seed)

    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    
  
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Selecting Estimator'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    #calibrating estimator
            
    model = CalibratedClassifierCV(base_estimator=estimator, method=method, cv=fold)
    full_name = str(model).split("(")[0]
    
    progress.value += 1
    
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    
    fold_num = 1
    
    for train_i , test_i in kf.split(data_X,data_y):
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1,1:] = 'Fitting Fold ' + str(fold_num) + ' of ' + str(fold)
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
    
        
        Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
        ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
    
        if hasattr(model, 'predict_proba'):
        
            model.fit(Xtrain,ytrain)
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
                
            kappa = metrics.cohen_kappa_score(ytest,pred_)
            score_acc = np.append(score_acc,sca)
            score_auc = np.append(score_auc,sc)
            score_recall = np.append(score_recall,recall)
            score_precision = np.append(score_precision,precision)
            score_f1 =np.append(score_f1,f1)
            score_kappa =np.append(score_kappa,kappa)

        else:
            
            model.fit(Xtrain,ytrain)
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

            kappa = metrics.cohen_kappa_score(ytest,pred_)
            score_acc = np.append(score_acc,sca)
            score_auc = np.append(score_auc,sc)
            score_recall = np.append(score_recall,recall)
            score_precision = np.append(score_precision,precision)
            score_f1 =np.append(score_f1,f1)
            score_kappa =np.append(score_kappa,kappa) 
       
        progress.value += 1
        
        
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall], 
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa]}).round(round)
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
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
            
        fold_num += 1
        
        '''
        TIME CALCULATION ENDS HERE
        '''
        
        if verbose:
            update_display(master_display, display_id = display_id)
            
        
        '''
        
        Update_display() ends here
        
        '''
            
    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    
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
    
    progress.value += 1
    
    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision , 
                     'F1' : score_f1, 'Kappa' : score_kappa})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision , 
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    
    #refitting the model on complete X_train, y_train
    monitor.iloc[1,1:] = 'Compiling Final Model'
    update_display(monitor, display_id = 'monitor')
    
    model.fit(data_X, data_y)
    
    progress.value += 1
    
    #storing into experiment
    tup = (full_name,model)
    experiment__.append(tup)
    nam = str(full_name) + ' Score Grid'
    tup = (nam, model_results)
    experiment__.append(tup)
    
    
    if verbose:
        clear_output()
        display(model_results)
        return model
    else:
        clear_output()
        return model



def evaluate_model(estimator):
    
    """
          
    Description:
    ------------
    This function displays a user interface for all of the available plots for 
    a given estimator. It internally uses the plot_model() function. 
    
        Example:
        --------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        lr = create_model('lr')
        
        evaluate_model(lr)
        
        This will display the User Interface for all of the plots for a given
        estimator.

    Parameters
    ----------
    estimator : object, default = none
    A trained model object should be passed as an estimator. 

    Returns:
    --------

    User Interface:  Displays the user interface for plotting.
    --------------

    Warnings:
    ---------
    None    
       
         
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
    
  
    d = interact(plot_model, estimator = fixed(estimator), plot = a)



def finalize_model(estimator):
    
    """
          
    Description:
    ------------
    This function fits the estimator onto the complete dataset passed during the
    setup() stage. The purpose of this function is to prepare for final model
    deployment after experimentation. 
    
        Example:
        --------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        lr = create_model('lr')
        
        final_lr = finalize_model(lr)
        
        This will return the final model object fitted to complete dataset. 

    Parameters
    ----------
    estimator : object, default = none
    A trained model object should be passed as an estimator. 

    Returns:
    --------

    Model:  Trained model object fitted on complete dataset.
    ------   

    Warnings:
    ---------
    - If the model returned by finalize_model(), is used on predict_model() without 
      passing a new unseen dataset, then the information grid printed is misleading 
      as the model is trained on the complete dataset including test / hold-out sample. 
      Once finalize_model() is used, the model is considered ready for deployment and
      should be used on new unseens dataset only.
       
         
    """
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #import depedencies
    from copy import deepcopy
    
    if type(estimator) is list:
        
        if type(estimator[0]) is not list:
            
            """
            Single Layer Stacker
            """
            
            stacker_final = deepcopy(estimator)
            stack_restack = stacker_final.pop()
            stack_method_final = stacker_final.pop()
            stack_meta_final = stacker_final.pop()
            
            model_final = stack_models(estimator_list = stacker_final, 
                                       meta_model = stack_meta_final, 
                                       method = stack_method_final,
                                       restack = stack_restack,
                                       finalize=True, 
                                       verbose=False)
            
        else:
            
            """
            multiple layer stacknet
            """
            
            stacker_final = deepcopy(estimator)
            stack_restack = stacker_final.pop()
            stack_method_final = stacker_final.pop()[0]
            stack_meta_final = stacker_final.pop()
            
            model_final = create_stacknet(estimator_list = stacker_final,
                                          meta_model = stack_meta_final,
                                          method = stack_method_final,
                                          restack = stack_restack,
                                          finalize = True,
                                          verbose = False)

    else:
        
        model_final = deepcopy(estimator)
        model_final.fit(X,y)
    
    #storing into experiment
    model_name = str(estimator).split("(")[0]
    model_name = 'Final ' + model_name
    tup = (model_name,model_final)
    experiment__.append(tup)
    
    return model_final


def save_model(model, model_name, verbose=True):
    
    """
          
    Description:
    ------------
    This function saves the transformation pipeline and trained model object 
    into the current active directory as a pickle file for later use. 
    
        Example:
        --------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        lr = create_model('lr')
        
        save_model(lr, 'lr_model_23122019')
        
        This will save the transformation pipeline and model as a binary pickle
        file in the current directory. 

    Parameters
    ----------
    model : object, default = none
    A trained model object should be passed as an estimator. 
    
    model_name : string, default = none
    Name of pickle file to be passed as a string.
    
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
               verbose=True):
    
    """
          
    Description:
    ------------
    This function loads a previously saved transformation pipeline and model 
    from the current active directory into the current python environment. 
    Load object must be a pickle file.
    
        Example:
        --------
        saved_lr = load_model('lr_model_23122019')
        
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



def predict_model(estimator, 
                  data=None,
                  platform=None,
                  authentication=None):
    
    """
       
    Description:
    ------------
    This function is used to predict new data using a trained estimator. It accepts
    an estimator created using one of the function in pycaret that returns a trained 
    model object or a list of trained model objects created using stack_models() or 
    create_stacknet(). New unseen data can be passed to data param as pandas Dataframe. 
    If data is not passed, the test / hold-out set separated at the time of setup() is
    used to generate predictions. 
    
        Example:
        --------
        from pycaret.datasets import get_data
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        lr = create_model('lr')
        
        lr_predictions_holdout = predict_model(lr)
        
    Parameters
    ----------
    estimator : object or list of objects / string,  default = None
    When estimator is passed as string, load_model() is called internally to load the
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
    - if the estimator passed is created using finalize_model() then the metrics 
      printed in the information grid maybe misleading as the model is trained on
      the complete dataset including the test / hold-out set. Once finalize_model() 
      is used, the model is considered ready for deployment and should be used on new 
      unseen datasets only.
         
           
    
    """
    
    #testing
    #global base_pred_df, base_pred_df_no_restack, df, df_restack, stacker_method, combined_df, inter_pred_df
    
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
    from IPython.display import clear_output, update_display
    
    estimator = deepcopy(estimator)
    clear_output()
    
    if type(estimator) is str:
        if platform == 'aws':
            estimator_ = load_model(str(estimator), platform='aws', 
                                   authentication={'bucket': authentication.get('bucket')},
                                   verbose=False)
            
        else:
            estimator_ = load_model(str(estimator), verbose=False)
            
    else:
        
        estimator_ = estimator

    if type(estimator_) is list:

        if 'sklearn.pipeline.Pipeline' in str(type(estimator_[0])):

            prep_pipe_transformer = estimator_.pop(0)
            model = estimator_[0]
            estimator = estimator_[0]
                
        else:
            
            try:

                prep_pipe_transformer = prep_pipe
                model = estimator
                estimator = estimator
                
            except:
                
                sys.exit("(Type Error): Transformation Pipe Missing. ")
            
    else:
        
        try:

            prep_pipe_transformer = prep_pipe
            model = estimator
            estimator = estimator
            
        except:
            
            sys.exit("(Type Error): Transformation Pipe Missing. ")
        
    #dataset
    if data is None:
        
        Xtest = X_test.copy()
        ytest = y_test.copy()
        X_test_ = X_test.copy()
        y_test_ = y_test.copy()
        
        Xtest.reset_index(drop=True, inplace=True)
        ytest.reset_index(drop=True, inplace=True)
        X_test_.reset_index(drop=True, inplace=True)
        y_test_.reset_index(drop=True, inplace=True)
        
        model = estimator
        estimator_ = estimator
        
    else:
        
        Xtest = prep_pipe_transformer.transform(data)                     
        X_test_ = data.copy() #original concater

        Xtest.reset_index(drop=True, inplace=True)
        X_test_.reset_index(drop=True, inplace=True)
    
        estimator_ = estimator
        
    #try:
    #    model = finalize_model(estimator)
    #except:
    #    model = estimator

    if type(estimator) is list:
        
        if type(estimator[0]) is list:
        
            """
            Multiple Layer Stacking
            """
            
            #utility
            stacker = model
            restack = stacker.pop()
            stacker_method = stacker.pop()
            #stacker_method = stacker_method[0]
            stacker_meta = stacker.pop()
            stacker_base = stacker.pop(0)

            #base model names
            base_model_names = []

            #defining base_level_names
            for i in stacker_base:
                b = str(i).split("(")[0]
                base_model_names.append(b)

            base_level_fixed = []

            for i in base_model_names:
                if 'CatBoostClassifier' in i:
                    a = 'CatBoostClassifier'
                    base_level_fixed.append(a)
                else:
                    base_level_fixed.append(i)

            base_level_fixed_2 = []

            counter = 0
            for i in base_level_fixed:
                s = str(i) + '_' + 'BaseLevel_' + str(counter)
                base_level_fixed_2.append(s)
                counter += 1

            base_level_fixed = base_level_fixed_2

            """
            base level predictions
            """
            base_pred = []
            for i in stacker_base:
                if 'soft' in stacker_method:
                    try:
                        a = i.predict_proba(Xtest) #change
                        a = a[:,1]
                    except:
                        a = i.predict(Xtest) #change
                else:
                    a = i.predict(Xtest) #change
                base_pred.append(a)

            base_pred_df = pd.DataFrame()
            for i in base_pred:
                a = pd.DataFrame(i)
                base_pred_df = pd.concat([base_pred_df, a], axis=1)

            base_pred_df.columns = base_level_fixed
            
            base_pred_df_no_restack = base_pred_df.copy()
            base_pred_df = pd.concat([Xtest,base_pred_df], axis=1)


            """
            inter level predictions
            """

            inter_pred = []
            combined_df = pd.DataFrame(base_pred_df)

            inter_counter = 0

            for level in stacker:
                
                inter_pred_df = pd.DataFrame()

                model_counter = 0 

                for model in level:
                    
                    try:
                        if inter_counter == 0:
                            if 'soft' in stacker_method: #changed
                                try:
                                    p = model.predict_proba(base_pred_df)
                                    p = p[:,1]
                                except:
                                    try:
                                        p = model.predict_proba(base_pred_df_no_restack)
                                        p = p[:,1]                                    
                                    except:
                                        try:
                                            p = model.predict(base_pred_df)
                                        except:
                                            p = model.predict(base_pred_df_no_restack)
                            else:
                                try:
                                    p = model.predict(base_pred_df)
                                except:
                                    p = model.predict(base_pred_df_no_restack)
                        else:
                            if 'soft' in stacker_method:
                                try:
                                    p = model.predict_proba(last_level_df)
                                    p = p[:,1]
                                except:
                                    p = model.predict(last_level_df)
                            else:
                                p = model.predict(last_level_df)
                    except:
                        if 'soft' in stacker_method:
                            try:
                                p = model.predict_proba(combined_df)
                                p = p[:,1]
                            except:
                                p = model.predict(combined_df)        
                    
                    p = pd.DataFrame(p)
                    
                    col = str(model).split("(")[0]
                    if 'CatBoostClassifier' in col:
                        col = 'CatBoostClassifier'
                    col = col + '_InterLevel_' + str(inter_counter) + '_' + str(model_counter)
                    p.columns = [col]

                    inter_pred_df = pd.concat([inter_pred_df, p], axis=1)

                    model_counter += 1

                last_level_df = inter_pred_df.copy()

                inter_counter += 1

                combined_df = pd.concat([combined_df,inter_pred_df], axis=1)

            """
            meta final predictions
            """

            #final meta predictions
            
            try:
                pred_ = stacker_meta.predict(combined_df)
            except:
                pred_ = stacker_meta.predict(inter_pred_df)

            try:
                pred_prob = stacker_meta.predict_proba(combined_df)
                pred_prob = pred_prob[:,1]
            except:
                try:
                    pred_prob = stacker_meta.predict_proba(inter_pred_df)
                    pred_prob = pred_prob[:,1]
                except:
                    pass

            #print('Success')

            if data is None:
                sca = metrics.accuracy_score(ytest,pred_)

                try:
                    sc = metrics.roc_auc_score(ytest,pred_prob,average='weighted')
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

                df_score = pd.DataFrame( {'Model' : 'Stacking Classifier', 'Accuracy' : [sca], 'AUC' : [sc], 'Recall' : [recall], 'Prec.' : [precision],
                                    'F1' : [f1], 'Kappa' : [kappa]})
                df_score = df_score.round(4)
                display(df_score)
        
            label = pd.DataFrame(pred_)
            label.columns = ['Label']
            label['Label']=label['Label'].astype(int)

            if data is None:
                X_test_ = pd.concat([Xtest,ytest,label], axis=1)
            else:
                X_test_ = pd.concat([X_test_,label], axis=1) #change here

            if hasattr(stacker_meta,'predict_proba'):
                try:
                    score = pd.DataFrame(pred_prob)
                    score.columns = ['Score']
                    score = score.round(4)
                    X_test_ = pd.concat([X_test_,score], axis=1)
                except:
                    pass

        else:
            
            """
            Single Layer Stacking
            """
            
            #copy
            stacker = model
            
            #restack
            restack = stacker.pop()
            
            #method
            method = stacker.pop()

            #separate metamodel
            meta_model = stacker.pop()

            model_names = []
            for i in stacker:
                model_names = np.append(model_names, str(i).split("(")[0])

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

            model_names = model_names_fixed

            base_pred = []

            for i in stacker:
                if method == 'hard':
                    #print('done')
                    p = i.predict(Xtest) #change

                else:
                    
                    try:
                        p = i.predict_proba(Xtest) #change
                        p = p[:,1]
                    except:
                        p = i.predict(Xtest) #change

                base_pred.append(p)

            df = pd.DataFrame()
            for i in base_pred:
                i = pd.DataFrame(i)
                df = pd.concat([df,i], axis=1)

            df.columns = model_names
            
            df_restack = pd.concat([Xtest,df], axis=1) #change

            #ytest = ytest #change

            #meta predictions starts here
            
            df.fillna(value=0,inplace=True)
            df_restack.fillna(value=0,inplace=True)
            
            #restacking check
            try:
                pred_ = meta_model.predict(df)
            except:
                pred_ = meta_model.predict(df_restack) 
                
            try:
                pred_prob = meta_model.predict_proba(df)
                pred_prob = pred_prob[:,1]

            except:
                try:
                    pred_prob = meta_model.predict_proba(df_restack)
                    pred_prob = pred_prob[:,1]
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

                df_score = pd.DataFrame( {'Model' : 'Stacking Classifier', 'Accuracy' : [sca], 'AUC' : [sc], 'Recall' : [recall], 'Prec.' : [precision],
                                    'F1' : [f1], 'Kappa' : [kappa]})
                df_score = df_score.round(4)
                display(df_score)

            label = pd.DataFrame(pred_)
            label.columns = ['Label']
            label['Label']=label['Label'].astype(int)

            if data is None:
                X_test_ = pd.concat([Xtest,ytest,label], axis=1) #changed
            else:
                X_test_ = pd.concat([X_test_,label], axis=1) #change here
      
            if hasattr(meta_model,'predict_proba'):
                try:
                    score = pd.DataFrame(pred_prob)
                    score.columns = ['Score']
                    score = score.round(4)
                    X_test_ = pd.concat([X_test_,score], axis=1)
                except:
                    pass

    else:
        
        #model name
        full_name = str(model).split("(")[0]
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
        try:
            pred_prob = model.predict_proba(Xtest)
            pred_prob = pred_prob[:,1]
        except:
            pass

        pred_ = model.predict(Xtest)
        
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
            

            df_score = pd.DataFrame( {'Model' : [full_name], 'Accuracy' : [sca], 'AUC' : [sc], 'Recall' : [recall], 'Prec.' : [precision],
                                'F1' : [f1], 'Kappa' : [kappa]})
            df_score = df_score.round(4)
            display(df_score)
            
        label = pd.DataFrame(pred_)
        label.columns = ['Label']
        label['Label']=label['Label'].astype(int)
        
        if data is None:
            X_test_ = pd.concat([Xtest,ytest,label], axis=1)
        else:
            X_test_ = pd.concat([X_test_,label], axis=1)
        
        if hasattr(model,'predict_proba'):
            try:
                score = pd.DataFrame(pred_prob)
                score.columns = ['Score']
                score = score.round(4)
                X_test_ = pd.concat([X_test_,score], axis=1)
            except:
                pass
            

    return X_test_



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
        juice = get_data('juice')
        experiment_name = setup(data = juice,  target = 'Purchase')
        lr = create_model('lr')
        
        deploy_model(model = lr, model_name = 'deploy_lr', platform = 'aws', 
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
    - This function uses file storage services to deploy the model on cloud platform. 
      As such, this is efficient for batch-use. Where the production objective is to 
      obtain prediction at an instance level, this may not be the efficient choice as 
      it transmits the binary pickle file between your local python environment and
      the platform. 
    
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






