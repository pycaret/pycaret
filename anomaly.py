def setup(data, 
          categorical_features = None,
          categorical_imputation = 'constant',
          numeric_features = None,
          numeric_imputation = 'mean',
          date_features = None,
          ignore_features = None,
          normalize = False,
          normalize_method = 'zscore',
          transformation = False,
          transformation_method = 'yeo-johnson',
          pca = False,
          pca_components = 0.99,
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
        anomaly = get_data('anomaly')

        experiment_name = setup(data = anomaly, normalize = True)
        
        'anomaly' is a pandas Dataframe.

    Parameters
    ----------
    data : {array-like, sparse matrix}, shape (n_samples, n_features) where n_samples 
    is the number of samples and n_features is the number of features in dataframe.
    
    categorical_features: string, default = None
    If the inferred data types are not correct, categorical_features can be used to
    overwrite the inferred type. For example upon running setup if type of column1
    is inferred as numeric instead of categorical, this parameter can be used to 
    overwrite by passing categorical_features = 'column1'
    
    categorical_imputation: string, default = 'constant'
    If missing values are found in categorical features, it will be imputed with a
    constant 'not_available' value. Other option available is 'mode' in which case
    imputation is done by most frequent value.
    
    numeric_features: string, default = None
    If the inferred data types are not correct, numeric_features can be used to
    overwrite the inferred type. For example upon running setup if type of column1
    is inferred as categorical instead of numeric, this parameter can be used to 
    overwrite by passing numeric_features = 'column1'    

    numeric_imputation: string, default = 'mean'
    If missing values are found in numeric features, it will be imputed with mean
    value of feature. Other option available is 'median' in which case imputation
    will be done by median value.
    
    date_features: string, default = None
    If data has DateTime column and is not automatically detected when running
    setup, this parameter can be used to define date_feature by passing
    data_features = 'date_column_name'. It can work with multiple date columns.
    Date columns is not used in modeling, instead feature extraction is performed
    and date column is dropped from the dataset. Incase the date column as time
    stamp, it will also extract features related to time / hours.
    
    ignore_features: string, default = None
    If any feature has to be ignored for modeling, it can be passed in the param
    ignore_features. ID and DateTime column when inferred, is automatically set
    ignore for modeling. 
    
    normalize: bool, default = False
    When set to True, transform feature space using normalize_method param defined.
    Normally, linear algorithms perform better with normalized data. However, the
    results may vary and it is advised to run multiple experiments to evaluate the
    benefit of normalization.
    
    normalize_method: string, default = 'zscore'
    Defines the method to be used for normalization. By default, normalize method
    is set to 'zscore'. The other available option is 'minmax'.
    
    transformation: bool, default = False
    When set to True, apply a power transformation to make data more Gaussian-like
    This is useful for modeling issues related to heteroscedasticity or other 
    situations where normality is desired. The optimal parameter for stabilizing 
    variance and minimizing skewness is estimated through maximum likelihood.
    
    transformation_method: string, default = 'yeo-johnson'
    Defines the method for transformation. By default, transformation method is set
    to 'yeo-johnson'. The other available option is 'quantile' transformation. Both 
    the transformation transforms the feature set to follow Gaussian-like or normal
    distribution. Note that quantile transformer is non-linear and may distort linear 
    correlations between variables measured at the same scale.
    
    pca: bool, default = False
    When set to True, it will perform Linear dimensionality reduction using Singular 
    Value Decomposition of the data to project it to a lower dimensional space. It 
    is recommended when dataset has mix of categorical and numeric features.
    
    pca_components: int/float, default = 0.99
    Number of components to keep. if pca_components is a float, it is treated as 
    goal percentage for information retention. When pca_components param is integer
    it is treated as number of features to be kept. pca_components must be strictly
    less than the original features in dataset.
    
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
        
    #checking numeric imputation
    allowed_numeric_imputation = ['mean', 'median']
    if numeric_imputation not in allowed_numeric_imputation:
        sys.exit("(Value Error): numeric_imputation param only accepts 'mean' or 'median' ")
        
    #checking normalize method
    allowed_normalize_method = ['zscore', 'minmax']
    if normalize_method not in allowed_normalize_method:
        sys.exit("(Value Error): normalize_method param only accepts 'zscore' or 'minxmax' ")        
    
    #checking transformation method
    allowed_transformation_method = ['yeo-johnson', 'quantile']
    if transformation_method not in allowed_transformation_method:
        sys.exit("(Value Error): transformation_method param only accepts 'yeo-johnson' or 'quantile' ")        
        
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
    
    #checking pca parameter
    if type(pca) is not bool:
        sys.exit('(Type Error): pca parameter only accepts True or False.')
        
    
    """
    error handling ends here
    """
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    
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
    
    #display dtypes
    if supervised is False:
        display_types_pass = True
    else:
        display_types_pass = False
    
    #import library
    from pycaret import preprocess
    
    X = preprocess.Preprocess_Path_Two(train_data = data_for_preprocess, 
                                       categorical_features = cat_features_pass,
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
                                       apply_pca = pca,
                                       pca_variance_retained=pca_components,
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
        missing_flag = 'True'
    else:
        missing_flag = 'False'
    
    if normalize is True:
        normalize_grid = normalize_method
    else:
        normalize_grid = 'None'
        
    if transformation is True:
        transformation_grid = transformation_method
    else:
        transformation_grid = 'None'
    
    pca_grid = pca
    
    if pca_grid is False:
        pca_comp_grid = None
    else:
        pca_comp_grid = pca_components
    
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
        pass
    
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
                                 ['Transformed Data ', shape_transformed ],
                                 ['Categorical Features ', cat_type ],
                                 ['Numeric Features ', float_type ],
                                 ['Normalize ', normalize ],
                                 ['Normalize Method ', normalize_grid ],
                                 ['Transformation ', transformation ],
                                 ['Transformation Method ', transformation_grid ],
                                 ['Missing Values ', missing_flag],
                                 ['PCA ', pca_grid],
                                 ['PCA components ', pca_comp_grid],
                                 ['Numeric Imputer ', numeric_imputation],
                                 ['Categorical Imputer ', categorical_imputation],
                               ], columns = ['Description', 'Value'] )

    functions_ = functions.style.hide_index()
    
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
        experiment__.append(('Anomaly Info', functions))
        experiment__.append(('Orignal Dataset', data_))
        experiment__.append(('Transformed Dataset', X))
        experiment__.append(('Transformation Pipeline', prep_pipe))
    
    
    return X, data_, seed, prep_pipe, prep_param, experiment__





def create_model(model = None, 
                 fraction = 0.05,
                 verbose = True):
    
    
    
    """  
     
    Description:
    ------------
    This function creates a model on the dataset passed as a data param during 
    the setup stage. setup() function must be called before using create_model().

    This function returns a trained model object. 
    
        Example
        -------
        from pycaret.datasets import get_data
        anomaly = get_data('anomaly')
        experiment_name = setup(data = anomaly, normalize = True)
        
        knn = create_model('knn')

        This will return trained k-Nearest Neighbors model.

    Parameters
    ----------
    model : string, default = None

    Enter abbreviated string of the model class. List of available models supported:

    Model                              Abbreviated String   Original Implementation 
    ---------                          ------------------   -----------------------
    Angle-base Outlier Detection       'abod'               pyod.models.abod.ABOD
    Isolation Forest                   'iforest'            module-pyod.models.iforest
    Clustering-Based Local Outlier     'cluster'            pyod.models.cblof
    Connectivity-Based Outlier Factor  'cof'                module-pyod.models.cof
    Histogram-based Outlier Detection  'histogram'          module-pyod.models.hbos
    k-Nearest Neighbors Detector       'knn'                module-pyod.models.knn
    Local Outlier Factor               'lof'                module-pyod.models.lof
    One-class SVM detector             'svm'                module-pyod.models.ocsvm
    Principal Component Analysis       'pca'                module-pyod.models.pca
    Minimum Covariance Determinant     'mcd'                module-pyod.models.mcd
    Subspace Outlier Detection         'sod'                module-pyod.models.sod
    Stochastic Outlier Selection       'sos'                module-pyod.models.sos

    fraction: float, default = 0.05
    The percentage / proportion of outliers in the dataset.

    verbose: Boolean, default = True
    Status update is not printed when verbose is set to False.

    Returns:
    --------

    model:    trained model object
    ------

    Warnings:
    ---------
    None
      
       
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
    allowed_models = ['abod', 'iforest', 'cluster', 'cof', 'histogram', 'knn', 'lof', 'svm', 'pca', 'mcd', 'sod', 'sos']
    
    if model not in allowed_models:
        sys.exit('(Value Error): Model Not Available. Please see docstring for list of available models.')
        
    #checking fraction type:
    if type(fraction) is not float:
        sys.exit('(Type Error): Fraction parameter can only take value as float between 0 to 1.')
        
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
    
    """
    monitor starts
    """
    
    #progress bar and monitor control    
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    progress = ipw.IntProgress(value=0, min=0, max=4, step=1 , description='Processing: ')
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
    
    #monitor update
    monitor.iloc[1,1:] = 'Importing the Model'
    if verbose:
        update_display(monitor, display_id = 'monitor')
    
    progress.value += 1
    
    #create model
    if model == 'abod':
        from pyod.models.abod import ABOD
        model = ABOD(contamination=fraction)
        full_name = 'Angle-base Outlier Detection'
        
    elif model == 'cluster':
        from pyod.models.cblof import CBLOF
        try:
            model = CBLOF(contamination=fraction, n_clusters=8, random_state=seed)
            model.fit(X)
        except:
            try:
                model = CBLOF(contamination=fraction, n_clusters=12, random_state=seed)
                model.fit(X)
            except:
                sys.exit("(Type Error) Could not form valid cluster separation")
                
        full_name = 'Clustering-Based Local Outlier'
        
    elif model == 'cof':
        from pyod.models.cof import COF
        model = COF(contamination=fraction)        
        full_name = 'Connectivity-Based Outlier Factor'
        
    elif model == 'iforest':
        from pyod.models.iforest import IForest
        model = IForest(contamination=fraction, behaviour = 'new', random_state=seed)    
        full_name = 'Isolation Forest'
        
    elif model == 'histogram':
        from pyod.models.hbos import HBOS
        model = HBOS(contamination=fraction) 
        full_name = 'Histogram-based Outlier Detection'
        
    elif model == 'knn':
        from pyod.models.knn import KNN
        model = KNN(contamination=fraction)  
        full_name = 'k-Nearest Neighbors Detector'
        
    elif model == 'lof':
        from pyod.models.lof import LOF
        model = LOF(contamination=fraction)
        full_name = 'Local Outlier Factor'
        
    elif model == 'svm':
        from pyod.models.ocsvm import OCSVM
        model = OCSVM(contamination=fraction)
        full_name = 'One-class SVM detector'
        
    elif model == 'pca':
        from pyod.models.pca import PCA
        model = PCA(contamination=fraction, random_state=seed)  
        full_name = 'Principal Component Analysis'
        
    elif model == 'mcd':
        from pyod.models.mcd import MCD
        model = MCD(contamination=fraction, random_state=seed)
        full_name = 'Minimum Covariance Determinant'
        
    elif model == 'sod':
        from pyod.models.sod import SOD
        model = SOD(contamination=fraction)         
        full_name = 'Subspace Outlier Detection'
        
    elif model == 'sos':
        from pyod.models.sos import SOS
        model = SOS(contamination=fraction)   
        full_name = 'Stochastic Outlier Selection'
    
    #monitor update
    monitor.iloc[1,1:] = 'Fitting the Model'
    progress.value += 1
    if verbose:
        update_display(monitor, display_id = 'monitor')
        
    #fitting the model
    model.fit(X)
    
    #storing in experiment__
    if verbose:
        tup = (full_name,model)
        experiment__.append(tup)  
    
    progress.value += 1
    
    if verbose:
        clear_output()

    return model




def assign_model(model,
                 transformation=False,
                 score=True,
                 verbose=True):
    
    """  
     
    Description:
    ------------
    This function assigns each of the data point in the dataset passed during setup
    stage to one of the clusters using trained model object passed as model param.
    create_model() function must be called before using assign_model().
    
    This function returns dataframe with Outlier flag (1 = outlier, 0 = inlier) and 
    decision score, when score is set to True.

        Example
        -------
        from pycaret.datasets import get_data
        anomaly = get_data('anomaly')
        experiment_name = setup(data = anomaly, normalize = True)
        knn = create_model('knn')
        
        knn_df = assign_model(knn)

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

    Returns:
    --------

    dataframe:   Returns a dataframe with inferred outliers using a trained model.
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
    
    monitor.iloc[1,1:] = 'Inferring Outliers from Model'
    
    if verbose:
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
    
    #storing in experiment__
    if verbose:
        tup = (name_,data__)
        experiment__.append(tup)  
    
    if verbose:
        clear_output()
        
    return data__



def tune_model(model=None,
               supervised_target=None,
               method='drop',
               estimator=None,
               optimize=None,
               fold=10):
    
    
    """
        
    Description:
    ------------
   This function tunes the fraction parameter using a predefined grid with
    the objective of optimizing a supervised learning metric as defined in 
    the optimize param. You can choose the supervised estimator from a large 
    library available in pycaret. By default, supervised estimator is Linear. 
    
    This function returns the tuned model object.
    
        Example
        -------
        from pycaret.datasets import get_data
        boston = get_data('boston')
        experiment_name = setup(data = boston, normalize = True)
        
        tuned_knn = tune_model(model = 'knn', supervised_target = 'medv', optimize = 'R2') 
        
        This will return tuned k-Nearest Neighbors model.

    Parameters
    ----------
    model : string, default = None

    Enter abbreviated name of the model. List of available models supported: 
    
    Model                              Abbreviated String   Original Implementation 
    ---------                          ------------------   -----------------------
    Angle-base Outlier Detection       'abod'               pyod.models.abod.ABOD
    Isolation Forest                   'iforest'            module-pyod.models.iforest
    Clustering-Based Local Outlier     'cluster'            pyod.models.cblof
    Connectivity-Based Outlier Factor  'cof'                module-pyod.models.cof
    Histogram-based Outlier Detection  'histogram'          module-pyod.models.hbos
    k-Nearest Neighbors Detector       'knn'                module-pyod.models.knn
    Local Outlier Factor               'lof'                module-pyod.models.lof
    One-class SVM detector             'svm'                module-pyod.models.ocsvm
    Principal Component Analysis       'pca'                module-pyod.models.pca
    Minimum Covariance Determinant     'mcd'                module-pyod.models.mcd
    Subspace Outlier Detection         'sod'                module-pyod.models.sod
    Stochastic Outlier Selection       'sos'                module-pyod.models.sos
    
    supervised_target: string
    Name of target column for supervised learning. It cannot be None.
    
    method: string, default = 'drop'
    When method set to drop, it will drop the outlier rows from training dataset 
    of supervised estimator, when method set to 'surrogate', it will use the
    decision function and label as a feature without dropping the outliers from
    training dataset.
    
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
    CatBoost Regressor            'catboost'             Regression
    
    If set to None, default is Linear model for both classification
    and regression tasks.
    
    optimize: string, default = None
    
    For Classification tasks:
    Accuracy, AUC, Recall, Precision, F1, Kappa
    
    For Regression tasks:
    MAE, MSE, RMSE, R2, ME
    
    If set to None, default is 'Accuracy' for classification and 'R2' for 
    regression tasks.
    
    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    Returns:
    --------

    visual plot:  Visual plot with fraction param on x-axis with metric to
    -----------   optimize on y-axis. Also, prints the best model metric.
    
    model:        trained model object with best fraction param. 
    -----------

    Warnings:
    ---------
    None
           
          
    """
    
    
    
    """
    exception handling starts here
    """
    
    global data_, X
    
    #testing
    #no active testing
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    import sys
    
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
        
        available_optimizers = ['MAE', 'MSE', 'RMSE', 'R2', 'ME', 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa']
        
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
    
    display(monitor, display_id = 'monitor')
        
    
    #General Dependencies
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn import metrics
    import numpy as np
    import plotly.express as px
    from copy import deepcopy
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
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
    
    #define model name
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
    param_grid_with_zero = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10] 
    param_grid = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10] 
    
    master = []; master_df = []
    
    monitor.iloc[1,1:] = 'Creating Outlier Detection Model'
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
    
    if 'Empty' in str(prep_param.pca): 
        pca_pass = False
    else:
        pca_pass = True
        
    if pca_pass is True:
        pca_comp_pass = prep_param.pca.variance_retained
    else:
        pca_comp_pass = 0.99
    
    if 'not_available' in prep_param.imputer.categorical_strategy:
        cat_impute_pass = 'constant'
    elif 'most frequent' in prep_param.imputer.categorical_strategy:
        cat_impute_pass = 'mode'
    
    num_impute_pass = prep_param.imputer.numeric_strategy
    
    if 'Empty' in str(prep_param.scaling):
        normalize_pass = False
    else:
        normalize_pass = True
        
    if normalize_pass is True:
        normalize_method_pass = prep_param.scaling.function_to_apply
    else:
        normalize_method_pass = 'zscore'
    
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
    
    global setup_without_target
    
    setup_without_target = setup(data = data_,
                                 categorical_features = cat_pass,
                                 categorical_imputation = cat_impute_pass,
                                 numeric_features = num_pass,
                                 numeric_imputation = num_impute_pass,
                                 date_features = time_pass,
                                 ignore_features = ignore_pass,
                                 normalize = normalize_pass,
                                 normalize_method = normalize_method_pass,
                                 transformation = transformation_pass,
                                 transformation_method = transformation_method_pass,
                                 pca = pca_pass,
                                 pca_components = pca_comp_pass,
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
        monitor.iloc[2,1:] = 'Fitting Model With ' + str(i) + ' Fraction'
        update_display(monitor, display_id = 'monitor')
                             
        #create and assign the model to dataset d
        m = create_model(model=model, fraction=i, verbose=False)
        d = assign_model(m, transformation=True, score=True, verbose=False)
        d[str(supervised_target)] = target_

        master.append(m)
        master_df.append(d)

        
    #attaching target variable back
    data_[str(supervised_target)] = target_

    
    if problem == 'classification':
        
        """
        
        defining estimator
        
        """
        
        monitor.iloc[1,1:] = 'Evaluating Anomaly Model'
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
        
        #build model without anomaly
        monitor.iloc[2,1:] = 'Evaluating Classifier Without Anomaly Detector'
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
            
            monitor.iloc[2,1:] = 'Evaluating Classifier With ' + str(param_grid_val) + ' Fraction'
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
                             
        df = pd.DataFrame({'Fraction %': param_grid_with_zero, 'Accuracy' : acc, 'AUC' : auc, 'Recall' : recall, 
                   'Precision' : prec, 'F1' : f1, 'Kappa' : kappa})
        
        sorted_df = df.sort_values(by=optimize, ascending=False)
        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]
        progress.value += 1 
        sd = pd.melt(df, id_vars=['Fraction %'], value_vars=['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa'], 
                     var_name='Metric', value_name='Score')

        fig = px.line(sd, x='Fraction %', y='Score', color='Metric', line_shape='linear', range_y = [0,1])
        fig.update_layout(plot_bgcolor='rgb(245,245,245)')
        title= str(full_name) + ' Metrics and Fraction %'
        fig.update_layout(title={'text': title, 'y':0.95,'x':0.45,'xanchor': 'center','yanchor': 'top'})
        
        clear_output()

        fig.show()
        
        best_k = np.array(sorted_df.head(1)['Fraction %'])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0],4)
        p = 'Best Model: ' + model_name + ' |' + ' Fraction %: ' + str(best_k) + ' | ' + str(optimize) + ' : ' + str(best_m)
        print(p)

    elif problem == 'regression':
        
        """
        
        defining estimator
        
        """
        
        monitor.iloc[1,1:] = 'Evaluating Anomaly Model'
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
        
        #build model without anomaly
        monitor.iloc[2,1:] = 'Evaluating Regressor Without Anomaly Detector'
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

        elif optimize == 'ME':
            max_error_ = metrics.max_error(y,pred)
            score.append(max_error_)

        metric.append(str(optimize))
        
        for i in range(1,len(master_df)):
            progress.value += 1 
            param_grid_val = param_grid[i-1]
            
            monitor.iloc[2,1:] = 'Evaluating Regressor With ' + str(param_grid_val) + ' Fraction'
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
            
            elif optimize == 'ME':
                max_error_ = metrics.max_error(y,pred)
                score.append(max_error_)
                
            metric.append(str(optimize))
        
        monitor.iloc[1,1:] = 'Compiling Results'
        monitor.iloc[1,1:] = 'Finalizing'
        update_display(monitor, display_id = 'monitor')                    
         
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

        fig = px.line(df, x='Fraction %', y=optimize, line_shape='linear', 
                      title= str(full_name) + ' Metrics and Fraction %', color='Metric')

        fig.update_layout(plot_bgcolor='rgb(245,245,245)')
        progress.value += 1 
        clear_output()
        
        fig.show()
        best_k = np.array(sorted_df.head(1)['Fraction %'])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0],4)
        p = 'Best Model: ' + model_name + ' |' + ' Fraction %: ' + str(best_k) + ' | ' + str(optimize) + ' : ' + str(best_m)
        print(p)
        
    #storing into experiment
    tup = ('Best Model',best_model)
    experiment__.append(tup)    
    
    org = retain_original(a,b,c)
    
    return best_model



def plot_model(model,
               plot = 'tsne'):
    
    
    """
          
    Description:
    ------------
    This function takes a trained model object and returns a plot on the dataset 
    passed during setup stage. This function internally calls assign_model before 
    generating a plot.  

        Example:
        --------
        from pycaret.datasets import get_data
        anomaly = get_data('anomaly')
        experiment_name = setup(data = anomaly, normalize = True)
        knn = create_model('knn')
        
        plot_model(knn)

    Parameters
    ----------

    model : object
    A trained model object can be passed. Model must be created using create_model().

    plot : string, default = 'frequency'
    Enter abbreviation of type of plot. The current list of plots supported are:

    Name                           Abbreviated String     
    ---------                      ------------------     
    t-SNE (3d) Dimension Plot      'tsne'
    UMAP Dimensionality Plot       'umap'

    
    Returns:
    --------

    Visual Plot:  Prints the visual plot. 
    ------------

    Warnings:
    ---------
    
    -  None
              

    """  
    
    #exception checking   
    import sys
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
        
    """
    exception handling starts here
    """
    

    #plot checking
    allowed_plots = ['tsne', 'umap']  
    if plot not in allowed_plots:
        sys.exit('(Value Error): Plot Not Available. Please see docstring for list of available plots.')
     

    
    """
    error handling ends here
    """
    
    #import dependencies
    import pandas as pd
    import numpy
    
    #import cufflinks
    import cufflinks as cf
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    if plot == 'tsne':
        
        b = assign_model(model, verbose=False, transformation=True, score=False)
        Label = pd.DataFrame(b['Label'])
        b.dropna(axis=0, inplace=True) #droping rows with NA's
        b.drop(['Label'], axis=1, inplace=True)
        
        b = pd.get_dummies(b) #casting categorical variables

        from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components=3).fit_transform(b)

        X = pd.DataFrame(X_embedded)
        X['Label'] = Label

        import plotly.express as px
        df = X
        fig = px.scatter_3d(df, x=0, y=1, z=2,
                      color='Label', title='3d TSNE Plot for Outliers', opacity=0.7, width=900, height=800)
        fig.show()
        
    elif plot == 'umap':

        b = assign_model(model, verbose=False, transformation=True, score=False)
        Label = pd.DataFrame(b['Label'])
        b.dropna(axis=0, inplace=True) #droping rows with NA's
        b.drop(['Label'], axis=1, inplace=True)
        b = pd.get_dummies(b) #casting categorical variables
        
        import umap
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(b)
        X = pd.DataFrame(embedding)

        import plotly.express as px
        df = X
        df['Label'] = Label
        fig = px.scatter(df, x=0, y=1,
                      color='Label', title='uMAP Plot for Outliers', opacity=0.7, width=900, height=800)
        fig.show() 



def save_model(model, model_name, verbose=True):
    
    """
          
    Description:
    ------------
    This function saves the transformation pipeline and trained model object 
    into the current active directory as a pickle file for later use. 
    
        Example:
        --------
        from pycaret.datasets import get_data
        anomaly = get_data('anomaly')
        experiment_name = setup(data = anomaly, normalize = True)
        knn = create_model('knn')
        
        save_model(knn, 'knn_model_23122019')
        
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
    
    model_ = []
    model_.append(prep_pipe)
    model_.append(model)
    
    import joblib
    model_name = model_name + '.pkl'
    joblib.dump(model_, model_name)
    if verbose:
        print('Transformation Pipeline and Model Succesfully Saved')




def load_model(model_name, verbose=True):
    
    """
          
    Description:
    ------------
    This function loads a previously saved transformation pipeline and model 
    from the current active directory into the current python environment. 
    Load object must be a pickle file.
    
        Example:
        --------
        saved_knn = load_model('knn_model_23122019')
        
        This will load the previously saved model in saved_lr variable. The file 
        must be in the current directory.

    Parameters
    ----------
    model_name : string, default = none
    Name of pickle file to be passed as a string.

    Returns:
    --------    
    Success Message

    Warnings:
    ---------
    None    
       
         
    """
        
        
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
                  data):
    
    """
       
    Description:
    ------------
    This function is used to predict new data using a trained model. It requires a
    trained model object created using one of the function in pycaret that returns 
    a trained model object. New data must be passed to data param as pandas Dataframe. 
    
        Example:
        --------
        from pycaret.datasets import get_data
        anomaly = get_data('anomaly')
        experiment_name = setup(data = anomaly)
        knn = create_model('knn')
        
        knn_predictions = predict_model(model = knn, data = anomaly)
        
    Parameters
    ----------
    model : object, default = None
    
    data : {array-like, sparse matrix}, shape (n_samples, n_features) where n_samples 
    is the number of samples and n_features is the number of features. All features 
    used during training must be present in the new dataset.
    
    Returns:
    --------

    info grid:  Information grid is printed when data is None.
    ----------      

    Warnings:
    ---------
    - Models that donot support 'predict' function cannot be used in predict_model(). 
  
             
    
    """
    
    #testing
    #no active tests
    
    #general dependencies
    import numpy as np
    import pandas as pd
    import re
    from sklearn import metrics
    from copy import deepcopy
    import sys
    
    #copy data and model
    data__ = data.copy()
    model_ = deepcopy(model)
    
    #check if estimator is string, then load model
    if type(model) is str:
        model_ = load_model(model, verbose=False)
        
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
        
    data__['Label'] = pred
    
    return data__



def get_outliers(data, 
                 model = None, 
                 fraction=0.05, 
                 ignore_features = None, 
                 normalize = True, 
                 transformation = False,
                 pca = False,
                 pca_components = 0.99):
    
    """
    Magic function to get outliers in Power Query / Power BI.    
    
    """
    
    if model is None:
        model = 'knn'
        
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
                                       pca_variance_retained=pca_components,
                                       random_state = seed)
    
    
    
    c = create_model(model=model, fraction=fraction, verbose=False)
    
    dataset = assign_model(c, verbose=False)
    
    return dataset
