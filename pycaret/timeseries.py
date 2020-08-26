from typing import Optional


def setup(data, 
          target, 
          train_size: float=0.7,
          numeric_features: Optional[list]=None,
          numeric_imputation='mean',
          date_features: Optional[list]=None,
          ignore_features: Optional[list]=None,
          normalize: bool=False,
          normalize_method: str='zscore',
          transformation: bool=False,
          transformation_method: str='yeo-johnson',
          remove_outliers: bool=False, #new
          outliers_threshold: float=0.05, #new
          transform_target: bool=False, #new
          transform_target_method: str='box-cox', #new
          session_id: Optional[int]=None,
          silent: bool=False,
          profile: bool=False):
    
    """
        
    Description:
    ------------    
    This function initializes the environment in pycaret and creates the transformation
    pipeline to prepare the data for modeling and deployment. setup() must called before
    executing any other function in pycaret. It takes two mandatory parameters:
    dataframe {array-like, sparse matrix} and name of the target column. 
    

    Parameters
    ----------
    * data : {array-like, sparse matrix}, shape (n_samples, n_features) where n_samples 
    is the number of samples and n_features is the number of features.

    * target: string
    Name of target column to be passed in as string. 
    
    * train_size: float, default = 0.7
    Size of the training set. By default, 70% of the data will be used for training 
    and validation. The remaining data will be used for test / hold-out set.
    
    * numeric_features: list, default = None
    If the inferred data types are not correct, numeric_features can be used to
    overwrite the inferred type. If when running setup the type of 'column1' is 
    inferred as a categorical instead of numeric, then this parameter can be used 
    to overwrite by passing numeric_features = ['column1'].    
    
    * numeric_imputation: string, default = 'mean'
    If missing values are found in numeric features, they will be imputed with the 
    mean value of the feature. The other available option is 'median' which imputes 
    the value using the median value in the training dataset. 
    
    * date_features: list, default = None
    If the data has a DateTime column that is not automatically detected when running
    setup, this parameter can be used by passing date_features = 'date_column_name'. 
    It can work with multiple date columns. Date columns are not used in modeling. 
    Instead, feature extraction is performed and date columns are dropped from the 
    dataset. If the date column includes a time stamp, features related to time will 
    also be extracted.
    
    * ignore_features: list, default = None
    If any feature should be ignored for modeling, it can be passed to the param
    ignore_features. The ID and DateTime columns when inferred, are automatically 
    set to ignore for modeling. 
    
    * normalize: bool, default = False
    When set to True, the feature space is transformed using the normalized_method
    param. Generally, linear algorithms perform better with normalized data however, 
    the results may vary and it is advised to run multiple experiments to evaluate
    the benefit of normalization.
    
    * normalize_method: string, default = 'zscore'
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
    
    * transformation: bool, default = False
    When set to True, a power transformation is applied to make the data more normal /
    Gaussian-like. This is useful for modeling issues related to heteroscedasticity or 
    other situations where normality is desired. The optimal parameter for stabilizing 
    variance and minimizing skewness is estimated through maximum likelihood.
    
    * transformation_method: string, default = 'yeo-johnson'
    Defines the method for transformation. By default, the transformation method is set
    to 'yeo-johnson'. The other available option is 'quantile' transformation. Both 
    the transformation transforms the feature set to follow a Gaussian-like or normal
    distribution. Note that the quantile transformer is non-linear and may distort linear 
    correlations between variables measured at the same scale.
    
    * remove_outliers: bool, default = False
    When set to True, outliers from the training data are removed using PCA linear
    dimensionality reduction using the Singular Value Decomposition technique.
    
    * outliers_threshold: float, default = 0.05
    The percentage / proportion of outliers in the dataset can be defined using
    the outliers_threshold param. By default, 0.05 is used which means 0.025 of the 
    values on each side of the distribution's tail are dropped from training data.
    
    * transform_target: bool, default = False
    When set to True, target variable is transformed using the method defined in
    transform_target_method param. Target transformation is applied separately from 
    feature transformations. 
    
    * transform_target_method: string, default = 'box-cox'
    'Box-cox' and 'yeo-johnson' methods are supported. Box-Cox requires input data to 
    be strictly positive, while Yeo-Johnson supports both positive or negative data.
    When transform_target_method is 'box-cox' and target variable contains negative
    values, method is internally forced to 'yeo-johnson' to avoid exceptions.
    
    * session_id: int, default = None
    If None, a random seed is generated and returned in the Information grid. The 
    unique number is then distributed as a seed in all functions used during the 
    experiment. This can be used for later reproducibility of the entire experiment.
    
    * silent: bool, default = False
    When set to True, confirmation of data types is not required. All preprocessing will 
    be performed assuming automatically inferred data types. Not recommended for direct use 
    except for established pipelines.
    
    * profile: bool, default = False
    If set to true, a data profile for Exploratory Data Analysis will be displayed 
    in an interactive HTML report. 
    
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


    #----------------------------------  Exception checking    --------------------------  
    import sys
    
    #checking train size parameter
    if not isinstance(train_size, float):
        sys.exit('(Type Error): train_size parameter only accepts float value.') 
        
    #checking target parameter
    if target not in data.columns:
        sys.exit('(Value Error): Target parameter doesnt exist in the data provided.')   

    #checking session_id
    if session_id is not None:
        if not isinstance(session_id, int):
            sys.exit('(Type Error): session_id parameter must be an integer.')   
    
    #checking profile parameter
    if not isinstance(profile,bool):
        sys.exit('(Type Error): profile parameter only accepts True or False.')
      
    #checking normalize parameter
    if not isinstance(normalize,bool):
        sys.exit('(Type Error): normalize parameter only accepts True or False.')
        
    #checking transformation parameter
    if not isinstance(transformation,bool):
        sys.exit('(Type Error): transformation parameter only accepts True or False.')
        
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
    
    #check transform_target
    if not isinstance(transform_target,bool):
        sys.exit('(Type Error): transform_target parameter only accepts True or False.')
        
    #transform_target_method
    allowed_transform_target_method = ['box-cox', 'yeo-johnson']
    if transform_target_method not in allowed_transform_target_method:
        sys.exit("(Value Error): transform_target_method param only accepts 'box-cox' or 'yeo-johnson'. ") 
    
    #remove_outliers
    if not isinstance(remove_outliers,bool):
        sys.exit('(Type Error): remove_outliers parameter only accepts True or False.')    
    
    #outliers_threshold
    if not isinstance(outliers_threshold,float):
        sys.exit('(Type Error): outliers_threshold must be a float between 0 and 1. ')   
    
    #cannot drop target
    if ignore_features is not None:
        if target in ignore_features:
            sys.exit("(Value Error): cannot drop target column. ")  
        
    #forced type check
    all_cols = list(data.columns)
    all_cols.remove(target)
    
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
    if not isinstance(silent,bool):
        sys.exit("(Type Error): silent parameter only accepts True or False. ")
        

    #----------------------------------------------------   Initialize components   --------------------------------------
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time

    #pandas option
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 500)
        
    progress = ipw.IntProgress(value=0, min=0, max=3, step=1 , description='Processing: ')
    display(progress)
    
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id='monitor')
    
    #general dependencies
    import numpy as np
    from sklearn.linear_model import LinearRegression
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
    global X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, target_inverse_transformer, experiment__, preprocess
    
    #generate seed to be used globally
    if session_id is None:
        seed = random.randint(150, 9000)
    else:
        seed = session_id

    '''
    PREPROCESS BEGINS HERE
    '''
     
    monitor.iloc[1, 1:] = 'Preparing Data for Modeling'
    update_display(monitor, display_id='monitor')
            
    #define parameters for preprocessor
    
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
    
    #transformation method strategy
    if transformation_method == 'yeo-johnson':
        trans_method_pass = 'yj'
    elif transformation_method == 'quantile':
        trans_method_pass = 'quantile'
        
    if silent:
        display_dtypes_pass = False
    else:
        display_dtypes_pass = True
    
    #transform target method 

    ### TO DO: Verify that data is positive to apply box-cox

    if transform_target_method == 'box-cox':
        transform_target_method_pass = 'bc'
    elif transform_target_method == 'yeo-johnson':
        transform_target_method_pass = 'yj'
        
    #import library
    from pycaret import preprocess
    
    data = preprocess.Preprocess_Path_One(train_data=data, 
                                          target_variable=target,
                                          numerical_features=numeric_features_pass,
                                          time_features=date_features_pass,
                                          features_todrop=ignore_features_pass,
                                          numeric_imputation_strategy=numeric_imputation,
                                          scale_data=normalize,
                                          scaling_method=normalize_method,
                                          Power_transform_data=transformation,
                                          Power_transform_method=trans_method_pass,
                                          remove_outliers=remove_outliers, #new
                                          outlier_contamination_percentage=outliers_threshold, #new
                                          outlier_methods=['pca'], #pca hardcoded
                                          display_types=display_dtypes_pass, #new #to be parameterized in setup later.
                                          target_transformation=transform_target, #new
                                          target_transformation_method=transform_target_method_pass, #new
                                          random_state=seed)

    # Update progress bar 
    progress.value += 1

    if hasattr(preprocess.dtypes, 'replacement'):
        label_encoded = preprocess.dtypes.replacement
        label_encoded = str(label_encoded).replace("'", '')
        label_encoded = str(label_encoded).replace("{", '')
        label_encoded = str(label_encoded).replace("}", '')

    else:
        label_encoded = 'None'

    try:
        res_type = ['quit', 'Quit', 'exit', 'EXIT', 'q', 'Q', 'e', 'E', 'QUIT', 'Exit']
        res = preprocess.dtypes.response
        if res in res_type:
            sys.exit("(Process Exit): setup has been interupted with user command 'quit'. setup must rerun.")
    except:
        pass
    
    #save prep pipe
    prep_pipe = preprocess.pipe
    
    #save target inverse transformer
    try:
        target_inverse_transformer = preprocess.pt_target.p_transform_target
    except:
        target_inverse_transformer = None
    
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
    
    if remove_outliers is False:
        outliers_threshold_grid = None
    else:
        outliers_threshold_grid = outliers_threshold

        
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
    
    #target transformation method
    if transform_target is False:
        transform_target_method_grid = None
    else:
        transform_target_method_grid = preprocess.pt_target.function_to_apply
     
    '''
    MONITOR UPDATES STARTS HERE
    '''

    #reset pandas option
    pd.reset_option("display.max_rows") #switch back on 
    pd.reset_option("display.max_columns")
    
    #create an empty list for pickling later.
    experiment__ = []
        
    #creating variables to be used later in the function
    X = data.drop(target, axis=1)
    y = data[target]
    
    # Update progress bar 
    progress.value += 1

    '''
    SAMPLING STARTS HERE
    '''
    
    ### If there is no sampling 

    monitor.iloc[1, 1:] = 'Splitting Data'
    update_display(monitor, display_id='monitor')


    ### TO DO: Verify if necessary to split data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=seed)

    # Update progress bar 
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
    functions = pd.DataFrame ([ 
        ['session_id', seed],
        ['Transform Target ', transform_target],
        ['Transform Target Method', transform_target_method_grid],
        ['Original Data', data_before_preprocess.shape ],
        ['Missing Values ', missing_flag],
        ['Numeric Features ', str(float_type)],
        ['Transformed Train Set', X_train.shape], 
        ['Transformed Test Set',X_test.shape],
        ['Numeric Imputer ', numeric_imputation],
        ['Normalize ', normalize],
        ['Normalize Method ', normalize_grid],
        ['Transformation ', transformation],
        ['Transformation Method ', transformation_grid],
        ['Remove Outliers ', remove_outliers],
        ['Outliers Threshold ', outliers_threshold_grid],
        ], columns=['Description', 'Value'] 
        )
    
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
    experiment__.append(('Forecast Setup Config', functions))
    experiment__.append(('X_training Set', X_train))
    experiment__.append(('y_training Set', y_train))
    experiment__.append(('X_test Set', X_test))
    experiment__.append(('y_test Set', y_test))
    experiment__.append(('Transformation Pipeline', prep_pipe))
    try:
        experiment__.append(('Target Inverse Transformer', target_inverse_transformer))
    except:
        pass
    
    return X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, target_inverse_transformer, experiment__
    


def create_model(estimator: str='Auto_arima', 
                 splits: int=5,
                 round: int=4,
                 verbose: bool=True):
     
    """  
     
    Description:
    ------------
    This function creates an autoregressive model. 
    The output prints a score grid that shows MAE, MSE, 
    RMSE, MAPE, SSE, AIC and BIC.
    This function returns a trained model object. 
    setup() function must be called before using create_model()

    Example
        -------
        from pycaret.datasets import get_data
        from pycaret.forecast import *
        
        data = get_data('air_passengers')
        s = setup(data, target='#Passengers')
        
        model = create_model('auto_arima')
        This will create an arima model.
    

    Parameters
    ----------
    estimator : string, default = None
    Enter abbreviated string of the estimator class. List of estimators supported:
    Estimator                     Abbreviated String     Original Implementation 
    ---------                     ------------------     -----------------------
    Simple Exponential Smoothing  'sem'                  tsa.api.SimpleExpSmoothing
    Holt                          'holt'                 tsa.api.Holt
    Auto_Arima                    'auto_arima'           pmdarima.auto_arima
    
    splits: integer, default = 5
    Number of splits to be used in TimeSeriesSplit. Must be at least 2. 
    round: integer, default = 4
    Number of decimal places the metrics in the score grid will be rounded to. 
    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False. Model_results is 
    returned when verbose is set to True

    Returns:
    --------
    model_results:   A table containing the scores of the model.
                     Scoring metrics used are MAE, MSE, RMSE, MAPE, AIC and BIC.
    
    model:        trained model object
    -----------
    
    Warnings:
    ---------
    None
    """


    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #exception checking   
    import sys
    
    #checking error for estimator (string)
    available_estimators = ['sem', 'holt', 'auto_arima']
    
    if estimator not in available_estimators:
        sys.exit('(Value Error): Estimator Not Available. Please see docstring for list of available estimators.')


    #checking fold parameter
    if type(splits) is not int:
        sys.exit('(Type Error): Splits parameter only accepts integer value.')
    
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
    progress = ipw.IntProgress(value=0, min=0, max=splits+4, step=1, description='Processing: ')
    master_display = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'MAPE', 'AIC', 'BIC'])
    display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame([ 
        ['Initiated', '. . . . . . . . . . . . . . . . . .', timestampStr], 
        ['Status', '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies'],
        ['ETC', '. . . . . . . . . . . . . . . . . .', 'Calculating ETC'] 
        ],
        columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id='monitor')
    
    if verbose:
        display_ = display(master_display, display_id=True)
        display_id = display_.display_id
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 

    #Storing y in data_y parameter
    data_y = y.copy()

    #reset index
    data_y.reset_index(drop=True, inplace=True)


    #general dependencies 
    import numpy as np 
    from sklearn import metrics 
    from sklearn.model_selection import TimeSeriesSplit
    from statsmodels.tsa.api import SimpleExpSmoothing
    from statsmodels.tsa.api import Holt
    from pmdarima import auto_arima

    # Update progress bar 
    progress.value += 1

    #cross validation setup starts here
    tscv = TimeSeriesSplit(n_splits=splits)

    score_mae = score_mse = np.empty((0, 0))
    score_rmse = score_mape = np.empty((0, 0))
    score_aic = score_bic = np.empty((0, 0))
    avgs_mae = avgs_mse = np.empty((0, 0))
    avgs_rmse = avgs_mape = np.empty((0, 0))
    avgs_aic = avgs_bic = np.empty((0, 0))


    def calculate_mape(actual, prediction):
        mask = actual != 0
        return (np.fabs(actual - prediction)/actual)[mask].mean()

  
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1, 1:] = 'Selecting Estimator'
    update_display(monitor, display_id='monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''

    def initizalize_model(estimator, ts):
        '''
            Description:
            ------------
            This function calls a time series model given 
            by the value of the estimator. 

            Parameters:
            ------------
            estimator: Time Series model name 
            ts: univariate time series data

            Returns:
            ------------
            model: initialized class with the time series data
        '''

        if estimator == 'sem':
            model = SimpleExpSmoothing(endog=ts)

        elif estimator == 'holt':
            model = Holt(endog=ts) 

        elif estimator == 'auto_arima':
            model = auto_arima(y=ts, stepwise=False)

        return model 


    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1, 1:] = 'Initializing CV'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''


    split_num = 1
    
    for train_i , test_i in tscv.split(data_y):
        
        t0 = time.time()
        
        '''
        MONITOR UPDATE STARTS
        '''
    
        monitor.iloc[1, 1:] = 'Fitting Split ' + str(split_num) + ' of ' + str(splits)
        update_display(monitor, display_id='monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        # Split ts data in train-test
        ytrain, ytest = data_y.iloc[train_i], data_y.iloc[test_i]
        
        # Initialize model and forecast data        
        if estimator == 'auto_arima':
            mdl = initizalize_model(estimator, ytrain)
            pred_ = mdl.predict(len(ytest))
        else:
            mdl = initizalize_model(estimator, ytrain)
            mdl = mdl.fit()
            pred_ = mdl.forecast(len(ytest))
        
        try:
            # Apply inverse transform of a prior call from setup function 
            pred_ = target_inverse_transformer.inverse_transform(np.array(pred_).reshape(-1, 1))
            ytest = target_inverse_transformer.inverse_transform(np.array(ytest).reshape(-1, 1))
            pred_ = np.nan_to_num(pred_)
            ytest = np.nan_to_num(ytest)
            
        except:
            pass
        

        # Evaluate model metrics 
        mae = metrics.mean_absolute_error(ytest, pred_)
        mse = metrics.mean_squared_error(ytest, pred_)
        rmse = np.sqrt(mse)
        mape = calculate_mape(ytest, pred_)
        aic = mdl.aic() if estimator == 'auto_arima' else mdl.aic
        bic = mdl.bic() if estimator == 'auto_arima' else mdl.bic
        score_mae = np.append(score_mae, mae)
        score_mse = np.append(score_mse, mse)
        score_rmse = np.append(score_rmse, rmse)
        score_mape = np.append(score_mape, mape)
        score_aic = np.append(score_aic, aic)
        score_bic =np.append(score_bic, bic)
       

        # Update progress bar 
        progress.value += 1
        
        
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        split_results = pd.DataFrame({'MAE':[mae], 'MSE': [mse], 'RMSE': [rmse], 'MAPE': [mape],
                                      'AIC' : [aic], 'BIC': [bic]}).round(round)
        master_display = pd.concat([master_display, split_results], ignore_index=True)
        split_results = []
        
        '''
        TIME CALCULATION SUB-SECTION STARTS HERE
        '''
        t1 = time.time()
        
        tt = (t1 - t0) * (splits-split_num) / 60
        tt = np.around(tt, 2)
        
        if tt < 1:
            tt = str(np.around((tt * 60), 2))
            ETC = tt + ' Seconds Remaining'
                
        else:
            tt = str(tt)
            ETC = tt + ' Minutes Remaining'
            
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[2, 1:] = ETC
        update_display(monitor, display_id='monitor')

        '''
        MONITOR UPDATE ENDS
        '''
            
        split_num += 1
        
        '''
        TIME CALCULATION ENDS HERE
        '''
        
        if verbose:
            update_display(master_display, display_id=display_id)
            
        
        '''
        
        Update_display() ends here
        
        '''    

    # Calculate average of the metrics across the different splits
    mean_mae = np.mean(score_mae)
    mean_mse = np.mean(score_mse)
    mean_rmse = np.mean(score_rmse)
    mean_mape = np.mean(score_mape)
    mean_aic = np.mean(score_aic)
    mean_bic = np.mean(score_bic)
    std_mae = np.std(score_mae)
    std_mse = np.std(score_mse)
    std_rmse = np.std(score_rmse)
    std_mape = np.std(score_mape)
    std_aic = np.std(score_aic)
    std_bic = np.std(score_bic)

    avgs_mae = np.append(avgs_mae, mean_mae)
    avgs_mae = np.append(avgs_mae, std_mae) 
    avgs_mse = np.append(avgs_mse, mean_mse)
    avgs_mse = np.append(avgs_mse, std_mse)
    avgs_rmse = np.append(avgs_rmse, mean_rmse)
    avgs_rmse = np.append(avgs_rmse, std_rmse)
    avgs_mape = np.append(avgs_mape, mean_mape)
    avgs_mape = np.append(avgs_mape, std_mape)
    avgs_aic = np.append(avgs_aic, mean_aic)
    avgs_aic = np.append(avgs_aic, std_aic)
    avgs_bic = np.append(avgs_bic, mean_bic)
    avgs_bic = np.append(avgs_bic, std_bic)


    # Update progress bar 
    progress.value += 1
    
    model_results = pd.DataFrame({'MAE': score_mae, 'MSE': score_mse, 'RMSE' : score_rmse, 'MAPE' : score_mape,
                                  'AIC': score_aic, 'BIC': score_bic})
    model_avgs = pd.DataFrame({'MAE': avgs_mae, 'MSE': avgs_mse, 'RMSE' : avgs_rmse, 'MAPE' : avgs_mape,
                               'AIC' : avgs_aic, 'BIC' : avgs_bic}, index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)


    # Refit model on complete data 
    monitor.iloc[1, 1:] = 'Compiling Final Model'
    update_display(monitor, display_id='monitor')
    
    model = initizalize_model(estimator, data_y)
    model = model if estimator == 'auto_arima' else model.fit()

    # Update progress bar 
    progress.value += 1

    #storing into experiment
    tup = (estimator, model)
    experiment__.append(tup)
    nam = str(estimator) + ' Score Grid'
    tup = (nam, model_results)
    experiment__.append(tup)
    
    if verbose:
        clear_output()
        display(model_results)
        return (model, model_results)
    else:
        clear_output()
        return model




def auto_select(splits: int=5,
                round: int=4,
                metric: str='rmse',
                verbose: bool=True):

    """  
     
    Description:
    ------------
    This function chooses the best model based on the lower value 
    of the input metric.

    Example
        -------
        from pycaret.datasets import get_data
        from pycaret.forecast import *
        
        data = get_data('air_passengers')
        s = setup(data, target='#Passengers')
        
        model = auto_select(splits=10, metric='mae')
        This will output the best model based on the MAE
    

    Parameters
    ----------
    * splits: integer, default = 5
    Number of splits to be used in TimeSeriesSplit. Must be at least 2. 
    * round: integer, default = 4
    Number of decimal places the metrics in the score grid will be rounded to. 
    * metric: string, default = 'rmse'
    Select the model based which has the lower value of this metric. 
    * verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False. Model_results is 
    returned when verbose is set to True

    
    Returns:
    --------
    model: Name of fitted estimator 
    -------- 
    model_results: A table containing the scores of the model.
                   Scoring metrics used are MAE, MSE, RMSE, MAPE, AIC and BIC.

    
    Warnings:
    ---------
    None
      
    
  
    """

    '''

    ERROR HANDLING STARTS HERE
    
    '''
    
    #exception checking   
    import sys
    
    #checking fold parameter
    if not isinstance(splits, int):
        sys.exit('(Type Error): Splits parameter only accepts integer value.')
    
    #checking round parameter
    if not isinstance(round, int):
        sys.exit('(Type Error): Round parameter only accepts integer value.')

    #checking metric parameter
    if not isinstance(metric, str):
        sys.exit('(Type Error): Metric parameter only accepts string value.')
 
    #checking verbose parameter
    if not isinstance(verbose, bool):
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
    progress = ipw.IntProgress(value=0, min=0, max=splits+4, step=1, description='Processing: ')
    display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame([
        ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr], 
        ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies'],
        ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] 
        ],
        columns=['', ' ', '   ']).set_index('')

    display(monitor, display_id='monitor')
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 

    # Update progress bar
    progress.value += 1
  
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1, 1:] = 'Selecting Model'
    update_display(monitor, display_id='monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''


    '''
    MODEL SELECTION STARTS HERE
    '''

    available_estimators = ['sem', 'holt', 'auto_arima']
    metric_results = {}
    results = {}

    for estimator in available_estimators:
        model, model_results = create_model(
                                estimator=estimator,
                                splits=splits,
                                round=round,
                                verbose=True
                                )

        metric_results[estimator] = model_results.loc['Mean', metric.upper()]
        results[estimator] = (model, model_results)

    # Select the model with the lowest value of metric
    best_model = min(metric_results, key=metric_results.get)
    results = results[best_model]

    # Save name of the best model and results of it 
    full_name = best_model
    model, model_results = results[0], results[1]

    '''
        MODEL SELECTION ENDS HERE
    '''


    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1, 1:] = 'Calling Best Model'
    update_display(monitor, display_id='monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''

    #storing into experiment
    tup = (full_name, model)
    experiment__.append(tup)
    nam = str(full_name) + ' Score Grid'
    tup = (nam, model_results)
    experiment__.append(tup)
    
    if verbose:
        clear_output()
        display(model_results)
        return (model, model_results)
    else:
        clear_output()
        return model


def forecast(estimator, 
             steps: int=5,
             plot: bool=False,
             style: Optional[str]=None):

    """  
    Description:
    ------------
    Forecast data given an estimator. 

    Example
        -------
        from pycaret.datasets import get_data
        from pycaret.forecast import *
        
        data = get_data('air_passengers')
        s = setup(data, target='#Passengers')
        
        model = auto_select(splits=10, metric='mae')
        forecast = forecast(auto_select)
        This will output the best model based on the MAE
    
    
    Parameters
    ----------
    * steps: integer, default = 5
    Number of steps ahead to be forecasted.  
    * plot: Boolean, default = False 
    Wheter or not to plot a graph with the original data and the 
    point forecast.   
    * style: Style sheet of pyplot graph. See the style sheets reference of 
    matplotlib for avaible options. 

    Returns:
    --------
    forecast:   Point estimate values (forecast values).
    -----------   
    
    
    Warnings:
    ---------
    None
      
    """

    '''
    ERROR HANDLING STARTS HERE
    '''
    
    #exception checking   
    import sys
    
    #checking steps parameter
    if not isinstance(steps, int):
        sys.exit('(Type Error): Steps parameter only accepts integer value.')
    
    #checking verbose parameter
    if not isinstance(plot, bool):
        sys.exit('(Type Error): Plot parameter can only take argument as True or False.') 

    #checking style parameter
    if not isinstance(style, str):
        sys.exit('(Type Error): Style parameter only accepts string value.') 

    '''
    ERROR HANDLING ENDS HERE
    '''

    #pre-load libraries
    import pandas as pd
    import numpy as np 
    import ipywidgets as ipw
    import matplotlib.pyplot as plt 
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=3, step=1, description='Processing: ')
    display(progress)
    
    #display monitor
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame([
        ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr], 
        ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies'],
        ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] 
        ],
        columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id='monitor')
    

    # Update progress bar
    progress.value += 1
  
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1, 1:] = 'Calling forecast method of estimator'
    update_display(monitor, display_id='monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''


    '''
    FORECAST STARTS HERE
    '''

    # Call predict/forecast method of estimator 
    if 'forecast' in dir(estimator):
        forecast = estimator.forecast(steps)
    else:
        forecast = estimator.predict(steps)

    '''
    FORECAST ENDS HERE
    '''
    
    # Update progress bar
    progress.value += 1

    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1, 1:] = 'Plotting results'
    update_display(monitor, display_id='monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''

    '''
    PLOT STARTS HERE
    '''

    # Plot results 
    if plot: 

        # Plot up to 100 observations of the data 
        n_observations = 100 if len(y) > 100 else len(y)

        plt.style.use(style)
        plt.figure(figsize=(12, 6))
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.plot(np.arange(0, n_observations), y.tail(n_observations), linewidth=4, color='red')
        plt.plot(np.arange(n_observations, n_observations+steps), forecast, linewidth=4, color='blue')
        plt.grid(False)
        plt.title(f'Original data and {steps} steps ahead Forecast')

    
    # Update progress bar 
    progress.value += 1

    '''
    PLOT ENDS HERE
    '''
    clear_output()

    return forecast 