def setup(data, 
          session_id = None, 
          normalize = False, 
          verbose=True):
    
    """
        
    Description:
    ------------
    This function initialize the environment in pycaret. setup() must called before
    executing any other function in pycaret. It takes one mandatory parameters i.e.
    dataframe {array-like, sparse matrix}. 

        Example
        -------
        experiment_name = setup(data)

        data is a pandas DataFrame.

    Parameters
    ----------

    data : {array-like, sparse matrix}, shape (n_samples, n_features) where n_samples 
    is the number of samples and n_features is the number of features or object of type
    list with n length.

    session_id: int, default = None
    If None, random seed is generated and returned in Information grid. The unique number 
    is then distributed as a seed in all other functions used during experiment. This can
    be used later for reproducibility of entire experiment.

    normalize: bool, default = False
    scaling of feature set using MinMaxScaler. by default normalize is set to False. 
    
    Returns:
    --------

    info grid:    Information grid is printed.
    -----------      

    environment:  This function returns various outputs that are stored in variable
    -----------   as tuple. They are being used by other functions in pycaret.

    Warnings:
    ---------
    
    - None
    
    
    
    """
    
    #exception checking   
    import sys
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    
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
            
            
    """
    error handling ends here
    """
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time

    '''
    generate monitor starts 
    '''
    
    #progress bar
    max_steps = 2
        
    progress = ipw.IntProgress(value=0, min=0, max=max_steps, step=1 , description='Processing: ')
    
        
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ] ],
                             #['Step' , '. . . . . . . . . . . . . . . . . .',  'Step 0 of ' + str(total_steps)] ],
                              columns=['', ' ', '   ']).set_index('')
    
    if verbose:
        display(progress)
        display(monitor, display_id = 'monitor')
    
    '''
    generate monitor end
    '''
    
    #general dependencies
    import numpy as np
    import pandas as pd
    import random
    
    #defining global variables
    global X, data_, experiment__, seed
    
    #copying data
    data_ = data.copy()
    
    #create an empty list for pickling later.
    try:
        experiment__.append('dummy')
        experiment__.pop()
    
    except:
        experiment__ = []
    
    #generate seed to be used globally
    if session_id is None:
        seed = random.randint(150,9000)
    else:
        seed = session_id
    
    progress.value += 1

    #scaling
    if normalize:
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = pd.get_dummies(data_)
        scaler = scaler.fit(X)
        
        #append to experiment__
        experiment__.append(('Scaler',scaler))
        
        X = scaler.transform(X)
        X = pd.DataFrame(X)
        
    else:
        X = data_.copy()
        X = pd.get_dummies(data_)
        
    '''
    Final display Starts
    '''
    
    shape = data.shape
    
    #normalize param
    if normalize:
        
        scaling = 'True'
    else:
        scaling = 'False'
        
    functions = pd.DataFrame ( [ ['session_id', seed ],
                                 ['Scaling', scaling],
                                 ['Shape', shape ], 
                               ], columns = ['Description', 'Value'] )

    functions_ = functions.style.hide_index()
    
    if verbose:
        clear_output()
        display(functions_)

    '''
    Final display Ends
    '''   

    #log into experiment
    if verbose:
        experiment__.append(('Anomaly Info', functions))
        experiment__.append(('Dataset', data_))
        experiment__.append(('Scaled Dataset', X))

    return X, data_, seed, experiment__



def create_model(model = None, 
                 fraction = 0.05,
                 verbose = True):
    
    
    
    """  
     
    Description:
    ------------
    This function creates a model using training data passed during setup stage. 
    Hence dataset doesn't need to be specified during create_model. This Function 
    returns trained model object can then be used for inference the training data 
    or new unseen data. 

    setup() function must be called before using create_model()

        Example
        -------
        abod = create_model('abod')

        This will return trained Angle-base Outlier Detection model.

    Parameters
    ----------

    model : string, default = None

    Enter abbreviated string of the model class. List of model supported:

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
    The percentage / proportion of outliers in the data set.

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
        model = CBLOF(contamination=fraction, random_state=seed)
        full_name = 'Clustering-Based Local Outlier'
        
    elif model == 'cof':
        from pyod.models.cof import COF
        model = COF(contamination=fraction)        
        full_name = 'Connectivity-Based Outlier Factor'
        
    elif model == 'iforest':
        from pyod.models.iforest import IForest
        model = IForest(contamination=fraction, random_state=seed)    
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
                 score=True,
                 verbose=True):
    
    """  
     
    Description:
    ------------
    This function is used for inference of outliers on training data passed in setup
    function using trained model created using create_model function. The function 
    returns dataframe with Outlier flag (1 = outlier, 0 = inlier) and decision score.

    create_model() function must be called before using assign_model()

        Example
        -------
        abod = create_model('abod')
        
        abod_df = assign_model(abod)

        This will return dataframe with outlier inferred using trained model passed
        as model param. 

    Parameters
    ----------

    model : trained model object, default = None
    
    score: Boolean, default = True
    The outlier scores of the training data. The higher, the more abnormal. 
    Outliers tend to have higher scores. This value is available once the model 
    is fitted. If set to False, it will only return the flag (1 = outlier, 0 = inlier).

    verbose: Boolean, default = True
    Status update is not printed when verbose is set to False.

    Returns:
    --------

    dataframe:   Returns dataframe with inferred outliers using trained model.
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
    
    name_ = mod_type + 'Outlier Assignment'
    #storing in experiment__
    if verbose:
        tup = (name_,data__)
        experiment__.append(tup)  
    
    if verbose:
        clear_output()
        
    return data__



def tune_model(model=None,
               supervised_target=None,
               method = 'drop',
               estimator=None,
               optimize=None,
               fold=10):
    
    
    """
        
    Description:
    ------------
    This function tunes the model fraction parameter of outlier detection models
    using a pre-defined  diverse grid with objective to optimize supervised learning 
    metric as defined in optimize param. This function cannot be used unsupervised.
    This  function allows to select estimator from a large library available in pycaret.
    By default supervised estimator is Linear. 
    
    This function returns the fraction param that are considered best using optimize 
    param.
    
    setup() function must be called prior to using this function.
    
    
        Example
        -------
        tuned_abod = tune_model('abod', supervised_target = 'medv', optimize='R2') 

        This will return trained Angle-based Outlier Detector with fraction param 
        that is optimized to improve 'R2' as defined in optimize param. By 
        default optimize param is 'Accuracy' for classification tasks and 'R2' for
        regression tasks. Task is determined automatically based on supervised_target
        param.


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
    
    If set to None, by default Linear model is used for both classification
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
    
    - None
           
        
          
    """
    
    
    
    """
    exception handling starts here
    """
    
    #testing
    global master, master_df
    
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
        sys.exit('(Value Error): Model Not Available. Please see docstring for list of available models.')
    
    #check method
    allowed_methods = ['drop', 'surrogate']
    if method not in allowed_methods:
        sys.exit('(Value Error): Method not recognized. See docstring for list of available methods.')
     
    #check if supervised target is None:
    if supervised_target is None:
        sys.exit('(Value Error): supervised_target cannot be None. A column name must be given for estimator.')
        
    #check supervised target:
    if supervised_target is not None:
        all_col = list(data_.columns)
        #target = str(supervised_target)
        #all_col.remove(target)
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
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
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
    
    #adding dummy model in master
    master.append('No Model Required')
    master_df.append('No Model Required')
    
    #removing target variable from data by defining new setup
    target_ = pd.DataFrame(data_[supervised_target])
    data_without_target = data_.copy()
    data_without_target.drop([supervised_target], axis=1, inplace=True)
    setup_without_target = setup(data_without_target, verbose=False, session_id=seed)
    
    for i in param_grid:
        progress.value += 1                      
        monitor.iloc[2,1:] = 'Fitting Model With ' + str(i) + ' Fraction'
        update_display(monitor, display_id = 'monitor')
                             
        #create and assign the model to dataset d
        m = create_model(model=model, fraction=i, verbose=False)
        d = assign_model(m, score=True, verbose=False)
        d[str(supervised_target)] = target_

        master.append(m)
        master_df.append(d)

        #anomaly model creation end's here
        
    #attaching target variable back
    data_[str(supervised_target)] = target_

    
    if problem == 'classification':
        
        """
        
        defining estimator
        
        """
        
        monitor.iloc[1,1:] = 'Evaluating Outlier Model'
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

        
        #build model without anomaly detection
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
        
        monitor.iloc[1,1:] = 'Evaluating Outlier Model'
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
        monitor.iloc[2,1:] = 'Evaluating Regressor Without Anomaly'
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
        
    return best_model
    


def plot_model(model,
               plot = 'tsne'):
    
    
    """
          
    Description:
    ------------
    This function takes a trained model object and returns the plot on of inferred
    outliers on training dataset. This function internally calls assign_model before
    generating a plot. 

        Example:
        --------
        
        abod = create_model('abod')
        plot_model(lda, plot='tsne')


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
        
        b = assign_model(model, score=False)
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

        b = assign_model(model, score=False)
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


def save_model(model, model_name):
    
    """
          
    Description:
    ------------
    This function saves the trained model object in current active directory
    as a pickle file for later use. 
    
        Example:
        --------
        
        abod = create_model('abod')
        save_model(abod, 'abod_model_23122019')
        
        This will save the model as binary pickle file in current directory. 

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
    
    import joblib
    model_name = model_name + '.pkl'
    joblib.dump(model, model_name)
    print('Model Succesfully Saved')


def load_model(model_name):
    
    """
          
    Description:
    ------------
    This function loads the prior saved model from current active directory into
    current python notebook. Load object must be a pickle file.
    
        Example:
        --------
        
        saved_abod = load_model('abod_model_23122019')
        
        This will call the trained model in saved_abod variable using model_name param.
        The file must be in current directory.

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
    print('Model Sucessfully Loaded')
    return joblib.load(model_name)

def save_experiment(experiment_name=None):
    
        
    """
          
    Description:
    ------------
    This function saves the entire experiment in current active directory. All 
    the outputs using pycaret are internally saved into a binary list which is
    pickilized when save_experiment() is used. 
    
        Example:
        --------
        
        save_experiment()
        
        This will save the entire experiment in current active directory. By 
        default name of experiment will use session_id generated during setup().
        To use custom name, experiment_name param has to be passed as string.
        
        For example:
        
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
    This function loads the prior saved experiment from current active directory 
    into current python notebook. Load object must be a pickle file.
    
        Example:
        --------
        
        saved_experiment = load_experiment('experiment_23122019')
        
        This will load the entire experiment pipeline into object saved_experiment
        using experiment_name param. The experiment file must be in current directory.
        
        
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


def get_outliers(data, model=None, fraction=0.05):
    
    """
    Magic function to get outliers in Power Query / Power BI.
    """
    
    if model is None:
        model = 'knn'
        
    s = setup(data, normalize=True, verbose=False)
    c = create_model(model=model, fraction=fraction, verbose=False)
    dataset = assign_model(c, verbose=False)
    return dataset