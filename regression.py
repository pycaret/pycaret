def setup(data, 
          target, 
          train_size=0.7,
          sampling=True,
          sample_estimator = None,
          session_id = None):
    
    """
        
    Description:
    ------------
    This function initialize the environment in pycaret. setup() must called before
    executing any other function in pycaret. It takes two mandatory parameters i.e.
    dataframe {array-like, sparse matrix} and name of the target column. 
    
    All other parameters are optional.

        Example
        -------
        experiment_name = setup(data, 'target')

        data is a pandas DataFrame and 'target' is the name of the column in dataframe.

    Parameters
    ----------

    data : {array-like, sparse matrix}, shape (n_samples, n_features) where n_samples 
    is the number of samples and n_features is the number of features.

    target: string
    Name of target column to be passed in as string.

    train_size: float, default = 0.7
    Size of training set. By default 70% of the data will be used for training and 
    validation.

    sampling: bool, default = True
    When sample size exceed 25,000 samples, pycaret creates base estimator at various
    sample level of the original dataset. This will return the performance plot of
    R2 at various sample level, that will help you decide sample size for modeling.
    You are then required to enter the desired sample size that will be considered
    for training and validation in the pycaret environment. 1 - sample size 
    will be discarded and not be used any further.
    
    sample_estimator: object, default = None
    If None, Linear Regression is used by default.
    
    session_id: int, default = None
    If None, random seed is generated and returned in Information grid. The unique 
    number is then distributed as a seed in all other functions used during experiment.
    This can be used later for reproducibility of entire experiment.


    Returns:
    --------

    info grid:    Information grid is printed.
    -----------      

    environment:  This function returns various outputs that are stored in variable
    -----------   as tuple. They are being used by other functions in pycaret.

    Warnings:
    ---------
    None
    
    """
    
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
        
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
   
    #progress bar
    if sampling:
        max = 10 + 2
    else:
        max = 2
        
    progress = ipw.IntProgress(value=0, min=0, max=max, step=1 , description='Processing: ')
    display(progress)
    
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['ETC' , '. . . . . . . . . . . . . . . . . .',  'Calculating ETC'] ],
                              columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id = 'monitor')
    
    #general dependencies
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import random
    import seaborn as sns
    import matplotlib.pyplot as plt

    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #declaring global variables to be accessed by other functions
    global X, y, X_train, X_test, y_train, y_test, seed, experiment__
    
    #generate seed to be used globally
    if session_id is None:
        seed = random.randint(150,9000)
    else:
        seed = session_id
    
    #create an empty list for pickling later.
    experiment__ = []
    
    #sample estimator
    if sample_estimator is None:
        model = LinearRegression()
    else:
        model = sample_estimator
        
    model_name = str(model).split("(")[0]
        
    #creating variables to be used later in the function
    X = data.drop(target,axis=1)
    y = data[target]
    
    progress.value += 1
    
    if sampling is True and data.shape[0] > 25000:
    
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
    
            X_, X__, y_, y__ = train_test_split(X, y, test_size=1-i)
            X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=seed)
            model.fit(X_train,y_train)
            pred_ = model.predict(X_test)
            
            r2 = metrics.r2_score(y_test,pred_)
            metric_results.append(r2)
            metric_name.append('R2')
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

        model_results = pd.DataFrame({'Sample Size' : split_percent, 'Metric' : metric_results, 'Metric Name': metric_name})
        
        #fig, ax = plt.subplots(figsize=(8, 5))
        plt.figure(figsize=(8, 5))
        plt.grid(True, which='both')
        plt.xlim(0,1)
        plt.ylim(-1,1)
        plt.tick_params(axis='both', which='major', bottom=False)
        #plt.majorticks_on()
        sns.lineplot(x="Sample Size", y="Metric", hue="Metric Name", data=model_results, color='blue', lw=2).set_title('Metric of ' + model_name + ' at Different Sample Size', fontsize=15).set_style("normal")
        #sns.set_style("whitegrid")
        print(' ')
        plt.show()
        
        
        monitor.iloc[1,1:] = 'Waiting for input'
        update_display(monitor, display_id = 'monitor')
        
        
        print('Please Enter the sample % of data you would like to use for modeling. Example: Enter 0.3 for 30%.')
        print('Press Enter if you would like to use 100% of the data.')
        
        print(' ')
        
        sample_size = input("Sample Size: ")
        
        if sample_size == '' or sample_size == '1':
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=seed)
            
            '''
            Final display Starts
            '''
            clear_output()
            print(' ')
            print('Setup Succesfully Completed!')
            functions = pd.DataFrame ( [ ['session_id', seed ],
                                         ['Original Data',X.shape ], 
                                         ['Sampled Data',X.shape ], 
                                         ['Sample %',X.shape[0] / X.shape[0]], 
                                         ['Training Set', X_train.shape ], 
                                         ['Testing Set',X_test.shape ], 
                                       ], columns = ['Description', 'Value'] )

            functions_ = functions.style.hide_index()
            display(functions_)
            
            '''
            Final display Ends
            '''   
            
            #log into experiment
            experiment__.append(('Info', functions))
            experiment__.append(('X_training Set', X_train))
            experiment__.append(('y_training Set', y_train))
            experiment__.append(('X_test Set', X_test))
            experiment__.append(('y_test Set', y_test)) 
        
            return X, y, X_train, X_test, y_train, y_test, seed, experiment__
        
        else:
            
            sample_n = float(sample_size)
            X_selected, X_discard, y_selected, y_discard = train_test_split(X, y, test_size=1-sample_n,  
                                                                random_state=seed)
            
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=1-train_size, 
                                                                random_state=seed)
            clear_output()
            
            
            '''
            Final display Starts
            '''
            
            clear_output()
            print(' ')
            print('Setup Succesfully Completed!')
            functions = pd.DataFrame ( [ ['session_id', seed ],
                                         ['Original Data',X.shape ], 
                                         ['Sampled Data',X_selected.shape ], 
                                         ['Sample %',X_selected.shape[0] / X.shape[0]], 
                                         ['Training Set', X_train.shape ], 
                                         ['Testing Set',X_test.shape ], 
                                       ], columns = ['Description', 'Value'] )
            
            functions_ = functions.style.hide_index()
            display(functions_)
            
            '''
            Final display Ends
            ''' 
            
            #log into experiment
            experiment__.append(('Info', functions))
            experiment__.append(('X_training Set', X_train))
            experiment__.append(('y_training Set', y_train))
            experiment__.append(('X_test Set', X_test))
            experiment__.append(('y_test Set', y_test)) 
            
            return X, y, X_train, X_test, y_train, y_test, seed, experiment__

    else:
        
        monitor.iloc[1,1:] = 'Splitting Data'
        update_display(monitor, display_id = 'monitor')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=seed)
        progress.value += 1
        
        clear_output()
        
        '''
        Final display Starts
        '''
        clear_output()
        print(' ')
        print('Setup Succesfully Completed!')
        functions = pd.DataFrame ( [ ['session_id', seed ],
                                     ['Original Data',X.shape ], 
                                     ['Sampled Data',X.shape ], 
                                     ['Sample %',X.shape[0] / X.shape[0]], 
                                     ['Training Set', X_train.shape ], 
                                     ['Testing Set',X_test.shape ], 
                                   ], columns = ['Description', 'Value'] )
        
        functions_ = functions.style.hide_index()
        display(functions_)
            
        '''
        Final display Ends
        '''   
        
        #log into experiment
        experiment__.append(('Info', functions))
        experiment__.append(('X_training Set', X_train))
        experiment__.append(('y_training Set', y_train))
        experiment__.append(('X_test Set', X_test))
        experiment__.append(('y_test Set', y_test))      
        
        return X, y, X_train, X_test, y_train, y_test, seed, experiment__


def create_model(estimator = None, 
                 ensemble = False, 
                 method = None, 
                 fold = 10, 
                 round = 4,  
                 verbose = True):
    
     
    """  
     
    Description:
    ------------
    This function creates a model and scores it using Kfold Cross Validation. 
    (default = 10 Fold). The output prints the score grid that shows MAE, MSE, 
    RMSE, R2 and Max Error (ME). 

    Function also returns a trained model object that can be used for further 
    processing in pycaret or can be used to call any method available in sklearn. 

    setup() function must be called before using create_model()

        Example
        -------
        lr = create_model('lr')

        This will return trained Linear Regression.

    Parameters
    ----------

    estimator : string, default = None

    Enter abbreviated string of the estimator class. List of estimators supported:

    Estimator                     Abbreviated String     Original Implementation 
    ---------                     ------------------     -----------------------
    Linear Regression             'lr'                   linear_model.LinearRegression
    Lasso Regression              'lasso'                linear_model.Lasso
    Ridge Regression              'ridge'                linear_model.Ridge
    Elastic Net                   'en'                   linear_model.ElasticNet
    Least Angle Regression        'lar'                  linear_model.Lars
    Lasso Least Angle Regression  'llar'                 linear_model.LassoLars
    Orthogonal Matching Pursuit   'omp'                  linear_model.OMP
    Bayesian Ridge                'br'                   linear_model.BayesianRidge
    Automatic Relevance Determ.   'ard'                  linear_model.ARDRegression
    Passive Aggressive Regressor  'par'                  linear_model.PAR
    Random Sample Consensus       'ransac'               linear_model.RANSACRegressor
    TheilSen Regressor            'tr'                   linear_model.TheilSenRegressor
    Huber Regressor               'huber'                linear_model.HuberRegressor 
    Kernel Ridge                  'kr'                   kernel_ridge.KernelRidge
    Support Vector Machine        'svm'                  svm.SVR
    K Neighbors Regressor         'knn'                  neighbors.KNeighborsRegressor 
    Decision Tree                 'dt'                   tree.DecisionTreeRegressor
    Random Forest                 'rf'                   ensemble.RandomForestRegressor
    Extra Trees Regressor         'et'                   ensemble.ExtraTreesRegressor
    AdaBoost Regressor            'ada'                  ensemble.AdaBoostRegressor
    Gradient Boosting             'gbr'                  ensemble.GradientBoostingRegressor 
    Multi Level Perceptron        'mlp'                  neural_network.MLPRegressor
    Extreme Gradient Boosting     'xgboost'              xgboost.readthedocs.io
    Light Gradient Boosting       'lightgbm'             github.com/microsoft/LightGBM

    ensemble: Boolean, default = False
    True would result in ensemble of estimator using the method parameter defined (see below). 

    method: String, 'Bagging' or 'Boosting', default = None.
    method must be defined when ensemble is set to True. Default method is set to None. 

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places metrics in score grid will be rounded to. 

    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are MAE, MSE, RMSE, R2 and ME. Mean and 
                  standard deviation of the scores across the folds is also returned.

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
    available_estimators = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 
                            'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 
                            'mlp', 'xgboost', 'lightgbm']
    
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
    
    
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
        
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+3, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['MAE','MSE','RMSE', 'R2', 'ME'])
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
    data_X = X_train
    data_y = y_train
  
    #general dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import KFold
    
    progress.value += 1
    
    #cross validation setup starts here
    kf = KFold(fold, random_state=seed)
    
    score_mae =np.empty((0,0))
    score_mse =np.empty((0,0))
    score_rmse =np.empty((0,0))
    score_r2 =np.empty((0,0))
    score_max_error =np.empty((0,0))
    avgs_mae =np.empty((0,0))
    avgs_mse =np.empty((0,0))
    avgs_rmse =np.empty((0,0))
    avgs_r2 =np.empty((0,0))
    avgs_max_error =np.empty((0,0))    
  
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Selecting Estimator'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
        
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
        
    else:
        model = estimator
        full_name = str(model).split("(")[0]
    
    progress.value += 1
    
    #checking method when ensemble is set to True. 

    if method == 'Bagging':
        
        from sklearn.ensemble import BaggingRegressor
        model = BaggingRegressor(model,bootstrap=True,n_estimators=10, random_state=seed)

    elif method == 'Boosting':
        
        from sklearn.ensemble import AdaBoostRegressor
        model = AdaBoostRegressor(model, n_estimators=10, random_state=seed)
    
    
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
        model.fit(Xtrain,ytrain)
        pred_ = model.predict(Xtest)
        mae = metrics.mean_absolute_error(ytest,pred_)
        mse = metrics.mean_squared_error(ytest,pred_)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(ytest,pred_)
        max_error_ = metrics.max_error(ytest,pred_)
        score_mae = np.append(score_mae,mae)
        score_mse = np.append(score_mse,mse)
        score_rmse = np.append(score_rmse,rmse)
        score_r2 =np.append(score_r2,r2)
        score_max_error = np.append(score_max_error,max_error_)
       
        progress.value += 1
        
        
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'MAE':[mae], 'MSE': [mse], 'RMSE': [rmse], 
                                     'R2': [r2], 'ME': [max_error_] }).round(round)
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

    mean_mae=np.mean(score_mae)
    mean_mse=np.mean(score_mse)
    mean_rmse=np.mean(score_rmse)
    mean_r2=np.mean(score_r2)
    mean_max_error=np.mean(score_max_error)
    std_mae=np.std(score_mae)
    std_mse=np.std(score_mse)
    std_rmse=np.std(score_rmse)
    std_r2=np.std(score_r2)
    std_max_error=np.std(score_max_error)

    avgs_mae = np.append(avgs_mae, mean_mae)
    avgs_mae = np.append(avgs_mae, std_mae) 
    avgs_mse = np.append(avgs_mse, mean_mse)
    avgs_mse = np.append(avgs_mse, std_mse)
    avgs_rmse = np.append(avgs_rmse, mean_rmse)
    avgs_rmse = np.append(avgs_rmse, std_rmse)
    avgs_r2 = np.append(avgs_r2, mean_r2)
    avgs_r2 = np.append(avgs_r2, std_r2)
    avgs_max_error = np.append(avgs_max_error, mean_max_error)
    avgs_max_error = np.append(avgs_max_error, std_max_error)
    
    progress.value += 1
    
    model_results = pd.DataFrame({'MAE': score_mae, 'MSE': score_mse, 'RMSE' : score_rmse, 'R2' : score_r2, 
                     'ME' : score_max_error})
    model_avgs = pd.DataFrame({'MAE': avgs_mae, 'MSE': avgs_mse, 'RMSE' : avgs_rmse, 'R2' : avgs_r2 , 
                     'ME' : avgs_max_error},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    
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
    This function ensemble the trained base estimator using method defined in 'method' 
    param (by default method = 'Bagging'). The output prints the score grid that shows 
    MAE, MSE, RMSE, R2 and Max Error (ME) by fold (default CV = 10 Folds).

    Function also returns a trained model object that can be used for further 
    processing in pycaret or can be used to call any method available in sklearn. 

    Model must be created using create_model() or tune_model() in pycaret or using any
    other package that returns sklearn object.

        Example:
        --------

        ensembled_lr = ensemble_model(lr)

        This will return ensembled Linear Regression.
        variable 'lr' is created used lr = create_model('lr')
        Using ensemble = True and method = 'Bagging' in create_model() is equivalent 
        to using ensemble_model(lr) where lr is created using create_model().


    Parameters
    ----------

    estimator : object, default = None

    method: String, default = 'Bagging' 
    Bagging implementation is based on sklearn.ensemble.BaggingRegressor
    Boosting implementation is based on sklearn.ensemble.AdaBoostRegressor

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2.

    round: integer, default = 4
    Number of decimal places metrics in score grid will be rounded to.

    n_estimators: integer, default = 10
    The number of base estimators in the ensemble.
    In case of perfect fit, the learning procedure is stopped early.

    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.


    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are MAE, MSE, RMSE, R2 and ME. Mean and
                  standard deviation of the scores across the folds is also
                  returned.

    model:        trained ensembled model object
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
        
    #Check for allowed method
    available_method = ['Bagging', 'Boosting']
    if method not in available_method:
        sys.exit("(Value Error): Method parameter only accepts two values 'Bagging' or 'Boosting'.")
    
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
    progress = ipw.IntProgress(value=0, min=0, max=fold+3, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['MAE','MSE','RMSE', 'R2', 'ME'])
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
    from sklearn.model_selection import KFold   
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore')    
    
    #Storing X_train and y_train in data_X and data_y parameter  
    data_X = X_train
    data_y = y_train
      
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
    
    if method == 'Bagging':
        
        from sklearn.ensemble import BaggingRegressor
        model = BaggingRegressor(model,bootstrap=True,n_estimators=n_estimators, random_state=seed)
         
    else:
        
        from sklearn.ensemble import AdaBoostRegressor
        model = AdaBoostRegressor(model, n_estimators=n_estimators, random_state=seed)
    
    progress.value += 1
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    kf = KFold(fold, random_state=seed)
    
    score_mae =np.empty((0,0))
    score_mse =np.empty((0,0))
    score_rmse =np.empty((0,0))
    score_r2 =np.empty((0,0))
    score_max_error =np.empty((0,0))
    avgs_mae =np.empty((0,0))
    avgs_mse =np.empty((0,0))
    avgs_rmse =np.empty((0,0))
    avgs_r2 =np.empty((0,0))
    avgs_max_error =np.empty((0,0))
    
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
        model.fit(Xtrain,ytrain)
        pred_ = model.predict(Xtest)
        mae = metrics.mean_absolute_error(ytest,pred_)
        mse = metrics.mean_squared_error(ytest,pred_)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(ytest,pred_)
        max_error_ = metrics.max_error(ytest,pred_)
        score_mae = np.append(score_mae,mae)
        score_mse = np.append(score_mse,mse)
        score_rmse = np.append(score_rmse,rmse)
        score_r2 =np.append(score_r2,r2)
        score_max_error = np.append(score_max_error,max_error_)
        
        progress.value += 1
        
                
        '''
        
        This section is created to update_display() as code loops through the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'MAE':[mae], 'MSE': [mse], 'RMSE': [rmse], 
                                     'R2': [r2], 'ME': [max_error_]}).round(round)
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
        
    mean_mae=np.mean(score_mae)
    mean_mse=np.mean(score_mse)
    mean_rmse=np.mean(score_rmse)
    mean_r2=np.mean(score_r2)
    mean_max_error=np.mean(score_max_error)
    std_mae=np.std(score_mae)
    std_mse=np.std(score_mse)
    std_rmse=np.std(score_rmse)
    std_r2=np.std(score_r2)
    std_max_error=np.std(score_max_error)

    avgs_mae = np.append(avgs_mae, mean_mae)
    avgs_mae = np.append(avgs_mae, std_mae) 
    avgs_mse = np.append(avgs_mse, mean_mse)
    avgs_mse = np.append(avgs_mse, std_mse)
    avgs_rmse = np.append(avgs_rmse, mean_rmse)
    avgs_rmse = np.append(avgs_rmse, std_rmse)
    avgs_r2 = np.append(avgs_r2, mean_r2)
    avgs_r2 = np.append(avgs_r2, std_r2)
    avgs_max_error = np.append(avgs_max_error, mean_max_error)
    avgs_max_error = np.append(avgs_max_error, std_max_error)

    model_results = pd.DataFrame({'MAE': score_mae, 'MSE': score_mse, 'RMSE' : score_rmse, 'R2' : score_r2 , 
                     'ME' : score_max_error})
    model_avgs = pd.DataFrame({'MAE': avgs_mae, 'MSE': avgs_mse, 'RMSE' : avgs_rmse, 'R2' : avgs_r2 , 
                     'ME' : avgs_max_error},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)  
    
    progress.value += 1
    
    model = model
    
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

def compare_models(blacklist = None,
                   fold = 10, 
                   round = 4, 
                   sort = 'R2',
                   turbo = True):
    
    """
   
    Description:
    ------------
    This function creates all models in model library and scores it using Kfold Cross 
    Validation. The output prints the score grid that shows MAE, MSE, RMSE, R2 and ME
    by fold (default CV = 10 Folds) of all the available model in model library.
    
    When turbo is set to True ('kr', 'ard' and 'mlp') are excluded due to longer
    training times. By default turbo param is set to True.

    List of models in Model Library

    Estimator                     Abbreviated String     Original Implementation 
    ---------                     ------------------     -----------------------
    Linear Regression             'lr'                   linear_model.LinearRegression
    Lasso Regression              'lasso'                linear_model.Lasso
    Ridge Regression              'ridge'                linear_model.Ridge
    Elastic Net                   'en'                   linear_model.ElasticNet
    Least Angle Regression        'lar'                  linear_model.Lars
    Lasso Least Angle Regression  'llar'                 linear_model.LassoLars
    Orthogonal Matching Pursuit   'omp'                  linear_model.OMP
    Bayesian Ridge                'br'                   linear_model.BayesianRidge
    Automatic Relevance Determ.   'ard'                  linear_model.ARDRegression
    Passive Aggressive Regressor  'par'                  linear_model.PAR
    Random Sample Consensus       'ransac'               linear_model.RANSACRegressor
    TheilSen Regressor            'tr'                   linear_model.TheilSenRegressor
    Huber Regressor               'huber'                linear_model.HuberRegressor 
    Kernel Ridge                  'kr'                   kernel_ridge.KernelRidge
    Support Vector Machine        'svm'                  svm.SVR
    K Neighbors Regressor         'knn'                  neighbors.KNeighborsRegressor 
    Decision Tree                 'dt'                   tree.DecisionTreeRegressor
    Random Forest                 'rf'                   ensemble.RandomForestRegressor
    Extra Trees Regressor         'et'                   ensemble.ExtraTreesRegressor
    AdaBoost Regressor            'ada'                  ensemble.AdaBoostRegressor
    Gradient Boosting             'gbr'                  ensemble.GradientBoostingRegressor 
    Multi Level Perceptron        'mlp'                  neural_network.MLPRegressor
    Extreme Gradient Boosting     'xgboost'              xgboost.readthedocs.io
    Light Gradient Boosting       'lightgbm'             github.com/microsoft/LightGBM

        Example:
        --------

        compare_models() 

        This will return the averaged score grid of all the models except 'kr', 'ard' 
        and 'mlp'. When turbo param is set to False, all models are included including
        'kr', 'ard' and 'mlp', However this may result in longer training times.
        
        compare_models( blacklist = [ 'knn', 'gbr' ] , turbo = False) 

        This will return comparison of all models except K Nearest Neighbour and
        Gradient Boosting Regressor.
        
        compare_models( blacklist = [ 'knn', 'gbr' ] , turbo = True) 

        This will return comparison of all models except K Nearest Neighbour, 
        Gradient Boosting Regressor, Kernel Ridge Regressor, Automatic Relevance
        Determinant and Multi Level Perceptron.
        

    Parameters
    ----------

    blacklist: string, default = None
    In order to omit certain models from the comparison, the abbreviation string 
    of such models (see above list) can be passed as list of strings. This is 
    normally done to be more efficient with time. 

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places metrics in score grid will be rounded to.
  
    sort: string, default = 'MAE'
    The scoring measure specified is used for sorting the average score grid
    Other options are 'MAE', 'MSE', 'RMSE', 'R2' and 'ME'.

    turbo: Boolean, default = True
    When turbo is set to True, it blacklists estimator that uses Radial Kernel.
    
    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are MAE, MSE, RMSE, R2 and Max Error (ME) 
                  Mean and standard deviation of the scores across the folds is
                  also returned.

    Warnings:
    ---------
    
    - compare_models() though attractive, might be time consuming with large 
      datasets. By default turbo is set to True, that will blacklists model that
      takes longer training time. Changing turbo parameter to False may result in 
      very high training times with datasets where number of sample size exceed 
      10,000.
         
           
    
    """
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #exception checking   
    import sys
    
    #checking error for blacklist (string)
    available_estimators = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 
                            'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 
                            'mlp', 'xgboost', 'lightgbm']

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
    allowed_sort = ['MAE', 'MSE', 'RMSE', 'R2', 'ME']
    if sort not in allowed_sort:
        sys.exit('(Value Error): Sort method not supported. See docstring for list of available parameters.')
    
    
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
        len_mod = 21 - len_of_blacklist
    else:
        len_mod = 24 - len_of_blacklist
        
    progress = ipw.IntProgress(value=0, min=0, max=(fold*len_mod)+20, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['Model', 'MAE','MSE','RMSE', 'R2', 'ME'])
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
    from sklearn.model_selection import KFold
    import pandas.io.formats.style
    
    #defining X_train and y_train as data_X and data_y
    data_X = X_train
    data_y=y_train
    
    progress.value += 1
    
    #import sklearn dependencies
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import Lars
    from sklearn.linear_model import LassoLars
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.linear_model import BayesianRidge
    from sklearn.linear_model import ARDRegression
    from sklearn.linear_model import PassiveAggressiveRegressor
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import TheilSenRegressor
    from sklearn.linear_model import HuberRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    import lightgbm as lgb
   
    progress.value += 1

    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Loading Estimator'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    #creating model object
    lr = LinearRegression()
    lasso = Lasso(random_state=seed)
    ridge = Ridge(random_state=seed)
    en = ElasticNet(random_state=seed)
    lar = Lars()
    llar = LassoLars()
    omp = OrthogonalMatchingPursuit()
    br = BayesianRidge()
    ard = ARDRegression()
    par = PassiveAggressiveRegressor(random_state=seed)
    ransac = RANSACRegressor(random_state=seed)
    tr = TheilSenRegressor(random_state=seed)
    huber = HuberRegressor()
    kr = KernelRidge()
    svm = SVR()
    knn = KNeighborsRegressor()
    dt = DecisionTreeRegressor(random_state=seed)
    rf = RandomForestRegressor(random_state=seed)
    et = ExtraTreesRegressor(random_state=seed)
    ada = AdaBoostRegressor(random_state=seed)
    gbr = GradientBoostingRegressor(random_state=seed)
    mlp = MLPRegressor(random_state=seed)
    xgboost = XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0)
    lightgbm = lgb.LGBMRegressor(random_state=seed)
    
    progress.value += 1
    
    model_library = [lr, lasso, ridge, en, lar, llar, omp, br, ard, par, ransac, tr, huber, kr, 
                     svm, knn, dt, rf, et, ada, gbr, mlp, xgboost, lightgbm]
    
    model_names = ['Linear Regression',
                   'Lasso Regression',
                   'Ridge Regression',
                   'Elastic Net',
                   'Least Angle Regression',
                   'Lasso Least Angle Regression',
                   'Orthogonal Matching Pursuit',
                   'Bayesian Ridge',
                   'Automatic Relevance Determination',
                   'Passive Aggressive Regressor',
                   'Random Sample Consensus',
                   'TheilSen Regressor',
                   'Huber Regressor',
                   'Kernel Ridge',
                   'Support Vector Machine',
                   'K Neighbors Regressor',
                   'Decision Tree',
                   'Random Forest',
                   'Extra Trees Regressor',
                   'AdaBoost Regressor',
                   'Gradient Boosting Regressor',
                   'Multi Level Perceptron',
                   'Extreme Gradient Boosting',
                   'Light Gradient Boosting Machine']
    
    
    #checking for blacklist models
    
    model_library_str = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard',
                         'par', 'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 
                         'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm']
    
    model_library_str_ = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard',
                         'par', 'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 
                         'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm']
    
    if blacklist is not None:
        
        if turbo:
            internal_blacklist = ['kr', 'ard', 'mlp']
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
        
        model_library = [lr, lasso, ridge, en, lar, llar, omp, br, par, ransac, tr, huber, 
                         svm, knn, dt, rf, et, ada, gbr, xgboost, lightgbm]
    
        model_names = ['Linear Regression',
                       'Lasso Regression',
                       'Ridge Regression',
                       'Elastic Net',
                       'Least Angle Regression',
                       'Lasso Least Angle Regression',
                       'Orthogonal Matching Pursuit',
                       'Bayesian Ridge',
                       'Passive Aggressive Regressor',
                       'Random Sample Consensus',
                       'TheilSen Regressor',
                       'Huber Regressor',
                       'Support Vector Machine',
                       'K Neighbors Regressor',
                       'Decision Tree',
                       'Random Forest',
                       'Extra Trees Regressor',
                       'AdaBoost Regressor',
                       'Gradient Boosting Regressor',
                       'Extreme Gradient Boosting',
                       'Light Gradient Boosting Machine']
    
        
            
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
    kf = KFold(fold, random_state=seed)

    score_mae =np.empty((0,0))
    score_mse =np.empty((0,0))
    score_rmse =np.empty((0,0))
    score_r2 =np.empty((0,0))
    score_max_error =np.empty((0,0))
    avgs_mae =np.empty((0,0))
    avgs_mse =np.empty((0,0))
    avgs_rmse =np.empty((0,0))
    avgs_r2 =np.empty((0,0))
    avgs_max_error =np.empty((0,0))  
    
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
            model.fit(Xtrain,ytrain)
            pred_ = model.predict(Xtest)
            mae = metrics.mean_absolute_error(ytest,pred_)
            mse = metrics.mean_squared_error(ytest,pred_)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(ytest,pred_)
            max_error_ = metrics.max_error(ytest,pred_)
            score_mae = np.append(score_mae,mae)
            score_mse = np.append(score_mse,mse)
            score_rmse = np.append(score_rmse,rmse)
            score_r2 =np.append(score_r2,r2)
            score_max_error = np.append(score_max_error,max_error_)            
                
                
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
        
        avgs_mae = np.append(avgs_mae,np.mean(score_mae))
        avgs_mse = np.append(avgs_mse,np.mean(score_mse))
        avgs_rmse = np.append(avgs_rmse,np.mean(score_rmse))
        avgs_r2 = np.append(avgs_r2,np.mean(score_r2))
        avgs_max_error = np.append(avgs_max_error,np.mean(score_max_error))
        
        compare_models_ = pd.DataFrame({'Model':model_names[name_counter], 'MAE':avgs_mae, 'MSE':avgs_mse, 
                           'RMSE':avgs_rmse, 'R2':avgs_r2, 
                           'ME':avgs_max_error})
        master_display = pd.concat([master_display, compare_models_],ignore_index=True)
        master_display = master_display.round(round)
        
        if sort == 'R2':
            master_display = master_display.sort_values(by=sort,ascending=False)
        else:
            master_display = master_display.sort_values(by=sort,ascending=True)
        
        master_display.reset_index(drop=True, inplace=True)
        
        update_display(master_display, display_id = display_id)
        
        score_mae =np.empty((0,0))
        score_mse =np.empty((0,0))
        score_rmse =np.empty((0,0))
        score_r2 =np.empty((0,0))
        score_max_error =np.empty((0,0))
        
        avgs_mae = np.empty((0,0))
        avgs_mse = np.empty((0,0))
        avgs_rmse = np.empty((0,0))
        avgs_r2 = np.empty((0,0))
        avgs_max_error = np.empty((0,0))
        
        name_counter += 1
  
    progress.value += 1
    
    #storing into experiment
    model_name = 'Compare Models Score Grid'
    tup = (model_name,master_display)
    experiment__.append(tup)
    
    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: yellow' if v else '' for v in is_min]

    compare_models_ = master_display.style.apply(highlight_min,subset=['MAE','MSE','RMSE','ME'])
    compare_models_ = compare_models_.set_properties(**{'text-align': 'left'})
    compare_models_ = compare_models_.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    
    progress.value += 1
    
    clear_output()

    return compare_models_

def blend_models(estimator_list = 'All', 
                 fold = 10, 
                 round = 4, 
                 turbo = True,
                 verbose = True):
    
    """
        
    Description:
    ------------
    This function creates an ensemble meta-estimator that fits base regressor on the
    whole dataset. It then, averages the predictions to form a final prediction.
    By default this function will use all the estimators in model library (excluding 
    few estimators when turbo is True) or specific trained estimator passed as a list
    in estimator_list param. It scores it using Kfold Cross Validation. The output prints 
    the score grid that shows MAE, MSE, RMSE, R2 and ME by fold (default CV = 10 Folds). 

    Function also returns a trained model object that can be used for further 
    processing in pycaret or can be used to call any method available in sklearn. 

        Example:
        --------

        blend_models() 

        This will result in VotingRegressor for all models in library except 'ard',
        'kr' and 'mlp'.
        
        blend_models(turbo=False) 

        This will result in VotingRegressor for all models in library. Training time may
        increase significantly.

        For specific models, you can use:

        lr = create_model( 'lr' )
        rf = create_model( 'rf' )

        blend_models( [ lr, rf ] )
    
        This will result in VotingRegressor of lr and rf.

    Parameters
    ----------

    estimator_list : string ('All') or list of object, default = 'All'

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places metrics in score grid will be rounded to.

    turbo: Boolean, default = True
    When turbo is set to True, it blacklists estimator that uses Radial Kernel.

    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are MAE, MSE, RMSE, R2 and ME. Mean and
                  standard deviation of the scores across the folds is also returned.

    model:        trained model object which is Voting Regressor. 
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
    
    #checking error for estimator_list (string)
    
    if estimator_list != 'All':
        for i in estimator_list:
            if 'sklearn' not in str(type(i)):
                sys.exit("(Value Error): estimator_list parameter only accepts 'All' as string or trained model object")
   
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
    import time, datetime
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+3, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['MAE','MSE','RMSE', 'R2', 'ME'])
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
    from sklearn.model_selection import KFold  
    from sklearn.ensemble import VotingRegressor
    import re
    
    #Storing X_train and y_train in data_X and data_y parameter
    data_X = X_train
    data_y = y_train
    
    progress.value += 1
    
    score_mae =np.empty((0,0))
    score_mse =np.empty((0,0))
    score_rmse =np.empty((0,0))
    score_r2 =np.empty((0,0))
    score_max_error =np.empty((0,0))
    avgs_mae =np.empty((0,0))
    avgs_mse =np.empty((0,0))
    avgs_rmse =np.empty((0,0))
    avgs_r2 =np.empty((0,0))
    avgs_max_error =np.empty((0,0))

    kf = KFold(fold, random_state=seed)
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Compiling Estimators'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    if estimator_list == 'All':

        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Ridge
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import ElasticNet
        from sklearn.linear_model import Lars
        from sklearn.linear_model import LassoLars
        from sklearn.linear_model import OrthogonalMatchingPursuit
        from sklearn.linear_model import BayesianRidge
        from sklearn.linear_model import ARDRegression
        from sklearn.linear_model import PassiveAggressiveRegressor
        from sklearn.linear_model import RANSACRegressor
        from sklearn.linear_model import TheilSenRegressor
        from sklearn.linear_model import HuberRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from xgboost import XGBRegressor
        import lightgbm as lgb

        lr = LinearRegression()
        lasso = Lasso(random_state=seed)
        ridge = Ridge(random_state=seed)
        en = ElasticNet(random_state=seed)
        lar = Lars()
        llar = LassoLars()
        omp = OrthogonalMatchingPursuit()
        br = BayesianRidge()
        ard = ARDRegression()
        par = PassiveAggressiveRegressor(random_state=seed)
        ransac = RANSACRegressor(random_state=seed)
        tr = TheilSenRegressor(random_state=seed)
        huber = HuberRegressor()
        kr = KernelRidge()
        svm = SVR()
        knn = KNeighborsRegressor()
        dt = DecisionTreeRegressor(random_state=seed)
        rf = RandomForestRegressor(random_state=seed)
        et = ExtraTreesRegressor(random_state=seed)
        ada = AdaBoostRegressor(random_state=seed)
        gbr = GradientBoostingRegressor(random_state=seed)
        mlp = MLPRegressor(random_state=seed)
        xgboost = XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0)
        lightgbm = lgb.LGBMRegressor(random_state=seed)

        progress.value += 1
        
        if turbo:
            
            estimator_list = [lr, lasso, ridge, en, lar, llar, omp, br, par, ransac, tr, huber, 
                             svm, knn, dt, rf, et, ada, gbr, xgboost, lightgbm]

        else:
            
            estimator_list = [lr, lasso, ridge, en, lar, llar, omp, br, ard, par, ransac, tr, huber, kr, 
                             svm, knn, dt, rf, et, ada, gbr, mlp, xgboost, lightgbm]
            

    else:

        estimator_list = estimator_list
        
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

    global model_names_final
    
    model_names_final = []
  
    for j in model_names_modified:

        if j == 'A R D Regression':
            model_names_final.append('Automatic Relevance Determination')

        elif j == 'M L P Regressor':
            model_names_final.append('MLP Regressor')

        elif j == 'R A N S A C Regressor':
            model_names_final.append('RANSAC Regressor')

        elif j == 'S V R':
            model_names_final.append('Support Vector Regressor')
            
        elif j == 'Lars':
            model_names_final.append('Least Angle Regression')

        else: 
            model_names_final.append(j)
            
        model_names = model_names_final
        estimator_list = estimator_list

        estimator_list_ = zip(model_names, estimator_list)
        estimator_list_ = set(estimator_list_)
        estimator_list_ = list(estimator_list_)
        
        try:
            model = VotingRegressor(estimators=estimator_list_, n_jobs=-1)
            model.fit(Xtrain,ytrain)
        except:
            model = VotingRegressor(estimators=estimator_list_)
    
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
        model.fit(Xtrain,ytrain)
        pred_ = model.predict(Xtest)
        mae = metrics.mean_absolute_error(ytest,pred_)
        mse = metrics.mean_squared_error(ytest,pred_)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(ytest,pred_)
        max_error_ = metrics.max_error(ytest,pred_)
        score_mae = np.append(score_mae,mae)
        score_mse = np.append(score_mse,mse)
        score_rmse = np.append(score_rmse,rmse)
        score_r2 =np.append(score_r2,r2)
        score_max_error = np.append(score_max_error,max_error_)
    
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'MAE':[mae], 'MSE': [mse], 'RMSE': [rmse], 
                                     'R2': [r2], 'ME': [max_error_]}).round(round)
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
    
    mean_mae=np.mean(score_mae)
    mean_mse=np.mean(score_mse)
    mean_rmse=np.mean(score_rmse)
    mean_r2=np.mean(score_r2)
    mean_max_error=np.mean(score_max_error)
    std_mae=np.std(score_mae)
    std_mse=np.std(score_mse)
    std_rmse=np.std(score_rmse)
    std_r2=np.std(score_r2)
    std_max_error=np.std(score_max_error)
    
    avgs_mae = np.append(avgs_mae, mean_mae)
    avgs_mae = np.append(avgs_mae, std_mae) 
    avgs_mse = np.append(avgs_mse, mean_mse)
    avgs_mse = np.append(avgs_mse, std_mse)
    avgs_rmse = np.append(avgs_rmse, mean_rmse)
    avgs_rmse = np.append(avgs_rmse, std_rmse)
    avgs_r2 = np.append(avgs_r2, mean_r2)
    avgs_r2 = np.append(avgs_r2, std_r2)
    avgs_max_error = np.append(avgs_max_error, mean_max_error)
    avgs_max_error = np.append(avgs_max_error, std_max_error)
    
    
    progress.value += 1
    
    model_results = pd.DataFrame({'MAE': score_mae, 'MSE': score_mse, 'RMSE' : score_rmse, 'R2' : score_r2, 
                     'ME' : score_max_error})
    model_avgs = pd.DataFrame({'MAE': avgs_mae, 'MSE': avgs_mse, 'RMSE' : avgs_rmse, 'R2' : avgs_r2, 
                     'ME' : avgs_max_error},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    
    progress.value += 1
    
    #storing into experiment
    model_name = 'Voting Regressor'
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

def tune_model(estimator = None, 
               fold = 10, 
               round = 4, 
               n_iter = 10, 
               optimize = 'r2',
               ensemble = False, 
               method = None,
               verbose = True):
    
      
    """
        
    Description:
    ------------
    This function tunes hyperparameter of a model and scores it using Kfold Cross 
    Validation. The output prints the score grid that shows MAE, MSE, RMSE, R2 and 
    ME by fold (by default = 10 Folds).

    Function also return a trained model object that can be used for further 
    processing in pycaret or can be used to call any method available in sklearn. 

    tune_model() accepts string parameter for estimator.

        Example
        -------
        tune_model('lr') 

        This will tune the hyperparameters of Linear Regression.

        tune_model('lr', ensemble = True, method = 'Bagging') 

        This will tune the hyperparameters of Linear Regression wrapped around 
        Bagging Regressor. 


    Parameters
    ----------

    estimator : string, default = None

    Enter abbreviated name of the estimator class. List of estimators supported:

    Estimator                     Abbreviated String     Original Implementation 
    ---------                     ------------------     -----------------------
    Linear Regression             'lr'                   linear_model.LinearRegression
    Lasso Regression              'lasso'                linear_model.Lasso
    Ridge Regression              'ridge'                linear_model.Ridge
    Elastic Net                   'en'                   linear_model.ElasticNet
    Least Angle Regression        'lar'                  linear_model.Lars
    Lasso Least Angle Regression  'llar'                 linear_model.LassoLars
    Orthogonal Matching Pursuit   'omp'                  linear_model.OMP
    Bayesian Ridge                'br'                   linear_model.BayesianRidge
    Automatic Relevance Determ.   'ard'                  linear_model.ARDRegression
    Passive Aggressive Regressor  'par'                  linear_model.PAR
    Random Sample Consensus       'ransac'               linear_model.RANSACRegressor
    TheilSen Regressor            'tr'                   linear_model.TheilSenRegressor
    Huber Regressor               'huber'                linear_model.HuberRegressor 
    Kernel Ridge                  'kr'                   kernel_ridge.KernelRidge
    Support Vector Machine        'svm'                  svm.SVR
    K Neighbors Regressor         'knn'                  neighbors.KNeighborsRegressor 
    Decision Tree                 'dt'                   tree.DecisionTreeRegressor
    Random Forest                 'rf'                   ensemble.RandomForestRegressor
    Extra Trees Regressor         'et'                   ensemble.ExtraTreesRegressor
    AdaBoost Regressor            'ada'                  ensemble.AdaBoostRegressor
    Gradient Boosting             'gbr'                  ensemble.GradientBoostingRegressor 
    Multi Level Perceptron        'mlp'                  neural_network.MLPRegressor
    Extreme Gradient Boosting     'xgboost'              xgboost.readthedocs.io
    Light Gradient Boosting       'lightgbm'             github.com/microsoft/LightGBM

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places metrics in score grid will be rounded to. 

    n_iter: integer, default = 10
    Number of iterations within the Random Grid Search. For every iteration, 
    the model randomly selects one value from the pre-defined grid of hyperparameters.

    optimize: string, default = 'r2'
    Measure used to select the best model through the hyperparameter tuning.
    The default scoring measure is 'mae'. Other common measures include
    'mae', 'mse' 'me'. Complete list available at:
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    ensemble: Boolean, default = None
    True would enable ensembling of model through method defined in 'method' param.

    method: String, 'Bagging' or 'Boosting', default = None
    method comes into effect only when ensemble = True. Default is set to None. 

    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are MAE, MSE, RMSE, R2 and ME. Mean and 
                  standard deviation of the scores across the folds is also 
                  returned.

    model:        trained model object
    -----------

    Warnings:
    ---------
    
    - estimator parameter takes an abbreviated string. passing a trained model object
      returns an error. tune_model('lr') function internally calls create_model() before
      tuning the hyperparameters.
   
     
    
  """
 


    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #exception checking   
    import sys
    
    #checking error for estimator (string)
    available_estimators = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 
                            'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 
                            'mlp', 'xgboost', 'lightgbm']
    
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
    allowed_optimize = ['mae', 'mse', 'r2', 'me']
    if optimize not in allowed_optimize:
        sys.exit('(Value Error): Optimization method not supported. See docstring for list of available parameters.')
    
    if type(n_iter) is not int:
        sys.exit('(Type Error): n_iter parameter only accepts integer value.')
        
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.') 
    
    
    '''
    
    ERROR HANDLING ENDS HERE
    
    '''
    
    
    #pre-load libraries
    import pandas as pd
    import time, datetime
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    
    #progress bar
    progress = ipw.IntProgress(value=0, min=0, max=fold+5, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['MAE','MSE','RMSE', 'R2', 'ME'])
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
    data_X = X_train
    data_y = y_train

    #define optimizer
      #defining optimizer
    if optimize == 'mae':
        optimize = 'neg_mean_absolute_error'
    elif optimize == 'mse':
        optimize = 'neg_mean_squared_error'
    elif optimize == 'me':
        optimize = 'max_error'
    elif optimize == 'r2':
        optimize = 'r2'
    
    progress.value += 1
    
    
    #general dependencies
    import random
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import KFold
    from sklearn.model_selection import RandomizedSearchCV
    
    progress.value += 1
    
    kf = KFold(fold, random_state=seed)

    score_mae =np.empty((0,0))
    score_mse =np.empty((0,0))
    score_rmse =np.empty((0,0))
    score_r2 =np.empty((0,0))
    score_max_error =np.empty((0,0))
    avgs_mae =np.empty((0,0))
    avgs_mse =np.empty((0,0))
    avgs_rmse =np.empty((0,0))
    avgs_r2 =np.empty((0,0))
    avgs_max_error =np.empty((0,0))
    
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
    
    if estimator == 'lr':

        from sklearn.linear_model import LinearRegression
        param_grid = {'fit_intercept': [True, False],
                     'normalize' : [True, False]
                    }        
        model_grid = RandomizedSearchCV(estimator=LinearRegression(), param_distributions=param_grid, 
                                        scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                        n_jobs=-1, iid=False)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'lasso':

        from sklearn.linear_model import Lasso

        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                      'fit_intercept': [True, False],
                      'normalize' : [True, False],
                     }
        model_grid = RandomizedSearchCV(estimator=Lasso(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv, 
                                        random_state=seed, iid=False,n_jobs=-1)
        
        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'ridge':

        from sklearn.linear_model import Ridge

        param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                      "fit_intercept": [True, False],
                      "normalize": [True, False],
                      }

        model_grid = RandomizedSearchCV(estimator=Ridge(random_state=seed), param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       iid=False, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'en':

        from sklearn.linear_model import ElasticNet

        param_grid = {'alpha': [0.0001,0.001,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                      'l1_ratio' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                      'fit_intercept': [True, False],
                      'normalize': [True, False]
                     } 

        model_grid = RandomizedSearchCV(estimator=ElasticNet(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv, 
                                        random_state=seed, iid=False, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'lar':

        from sklearn.linear_model import Lars

        param_grid = {'fit_intercept':[True, False],
                     'normalize' : [True, False],
                     'eps': [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.0005, 0.005, 0.00005, 0.02, 0.007]}

        model_grid = RandomizedSearchCV(estimator=Lars(), param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_  

    elif estimator == 'llar':

        from sklearn.linear_model import LassoLars

        param_grid = {'alpha': [0.0001,0.001,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                     'fit_intercept':[True, False],
                     'normalize' : [True, False],
                     'eps': [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.0005, 0.005, 0.00005, 0.02, 0.007]}

        model_grid = RandomizedSearchCV(estimator=LassoLars(), param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_    

    elif estimator == 'omp':

        from sklearn.linear_model import OrthogonalMatchingPursuit
        import random

        param_grid = {'n_nonzero_coefs': range(1,len(X_train.columns)+1),
                      'fit_intercept' : [True, False],
                      'normalize': [True, False]}

        model_grid = RandomizedSearchCV(estimator=OrthogonalMatchingPursuit(), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_        

    elif estimator == 'br':

        from sklearn.linear_model import BayesianRidge

        param_grid = {'alpha_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                      'alpha_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                      'lambda_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                      'lambda_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                      'compute_score': [True, False],
                      'fit_intercept': [True, False],
                      'normalize': [True, False]
                     }    

        model_grid = RandomizedSearchCV(estimator=BayesianRidge(), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_    

    elif estimator == 'ard':

        from sklearn.linear_model import ARDRegression

        param_grid = {'alpha_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                      'alpha_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                      'lambda_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                      'lambda_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                      'threshold_lambda' : [5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000],
                      'compute_score': [True, False],
                      'fit_intercept': [True, False],
                      'normalize': [True, False]
                     }    

        model_grid = RandomizedSearchCV(estimator=ARDRegression(), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_       

    elif estimator == 'par':

        from sklearn.linear_model import PassiveAggressiveRegressor

        param_grid = {'C': [0.01, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                      'fit_intercept': [True, False],
                      'early_stopping' : [True, False],
                      #'validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                      'loss' : ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                      'epsilon' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                      'shuffle' : [True, False]
                     }    

        model_grid = RandomizedSearchCV(estimator=PassiveAggressiveRegressor(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_         

    elif estimator == 'ransac':

        from sklearn.linear_model import RANSACRegressor

        param_grid = {'min_samples': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                      'max_trials': [1,2,3,4,5,6,7,8,9,10],
                      'max_skips': [1,2,3,4,5,6,7,8,9,10],
                      'stop_n_inliers': [1,2,3,4,5,6,7,8,9,10],
                      'stop_probability': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                      'loss' : ['absolute_loss', 'squared_loss'],
                     }    

        model_grid = RandomizedSearchCV(estimator=RANSACRegressor(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_         

    elif estimator == 'tr':

        from sklearn.linear_model import TheilSenRegressor

        param_grid = {'fit_intercept': [True, False],
                      'max_subpopulation': [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000]
                     }    

        model_grid = RandomizedSearchCV(estimator=TheilSenRegressor(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_    

    elif estimator == 'huber':

        from sklearn.linear_model import HuberRegressor

        param_grid = {'epsilon': [1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9],
                      'alpha': [0.00001, 0.0001, 0.0003, 0.005, 0.05, 0.1, 0.0005, 0.15],
                      'fit_intercept' : [True, False]
                     }    

        model_grid = RandomizedSearchCV(estimator=HuberRegressor(), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_        

    elif estimator == 'kr':

        from sklearn.kernel_ridge import KernelRidge

        param_grid = {'alpha': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] }    

        model_grid = RandomizedSearchCV(estimator=KernelRidge(), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_       

    elif estimator == 'svm':

        from sklearn.svm import SVR

        param_grid = {#'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                      #'float' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                      'C' : [0.01, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                      'epsilon' : [1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9],
                      'shrinking': [True, False]
                     }    

        model_grid = RandomizedSearchCV(estimator=SVR(), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_     

    elif estimator == 'knn':

        from sklearn.neighbors import KNeighborsRegressor

        param_grid = {'n_neighbors': range(1,51),
                     'weights' :  ['uniform', 'distance'],
                     'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                     'leaf_size': [10,20,30,40,50,60,70,80,90]
                     } 

        model_grid = RandomizedSearchCV(estimator=KNeighborsRegressor(), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_         

    elif estimator == 'dt':

        from sklearn.tree import DecisionTreeRegressor

        param_grid = {"max_depth": np.random.randint(3, (len(X_train.columns)*.85),4),
                      "max_features": np.random.randint(3, len(X_train.columns),4),
                      "min_samples_leaf": [0.1,0.2,0.3,0.4,0.5],
                      "min_samples_split" : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                      "min_weight_fraction_leaf" : [0.1,0.2,0.3,0.4,0.5],
                      "min_impurity_decrease" : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                      "criterion": ["mse", "mae", "friedman_mse"],
                      #"max_leaf_nodes" : [1,2,3,4,5,6,7,8,9,10,None]
                     } 

        model_grid = RandomizedSearchCV(estimator=DecisionTreeRegressor(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_         

    elif estimator == 'rf':

        from sklearn.ensemble import RandomForestRegressor


        param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                      'criterion': ['mse', 'mae'],
                      'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                      'min_samples_split': [2, 5, 7, 9, 10],
                      'min_samples_leaf' : [1, 2, 4],
                      'max_features' : ['auto', 'sqrt', 'log2'],
                      'bootstrap': [True, False]
                      }

        model_grid = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_       


    elif estimator == 'et':

        from sklearn.ensemble import ExtraTreesRegressor

        param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                      'criterion': ['mse', 'mae'],
                      'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                      'min_samples_split': [2, 5, 7, 9, 10],
                      'min_samples_leaf' : [1, 2, 4],
                      'max_features' : ['auto', 'sqrt', 'log2'],
                      'bootstrap': [True, False]
                      }  

        model_grid = RandomizedSearchCV(estimator=ExtraTreesRegressor(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_       

    elif estimator == 'ada':

        from sklearn.ensemble import AdaBoostRegressor

        param_grid = {'n_estimators': [10, 40, 70, 80, 90, 100, 120, 140, 150],
                      'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                      'loss' : ["linear", "square", "exponential"]
                     }    

        model_grid = RandomizedSearchCV(estimator=AdaBoostRegressor(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_ 

    elif estimator == 'gbr':

        from sklearn.ensemble import GradientBoostingRegressor

        param_grid = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                      'n_estimators': [10, 40, 70, 80, 90, 100, 120, 140, 150],
                      'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                      'subsample' : [0.1,0.3,0.5,0.7,0.9,1],
                      'criterion' : ['friedman_mse', 'mse', 'mae'],
                      'min_samples_split' : [2,4,5,7,9,10],
                      'min_samples_leaf' : [1,2,3,4,5],
                      'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                      'max_features' : ['auto', 'sqrt', 'log2']
                     }     

        model_grid = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_         

    elif estimator == 'mlp':

        from sklearn.neural_network import MLPRegressor

        param_grid = {'learning_rate': ['constant', 'invscaling', 'adaptive'],
                      'solver' : ['lbfgs', 'adam'],
                      'alpha': [0.0001, 0.001, 0.01, 0.00001, 0.003, 0.0003, 0.0005, 0.005, 0.05],
                      'hidden_layer_sizes': np.random.randint(50,150,10),
                      'activation': ["tanh", "identity", "logistic","relu"]
                      }    

        model_grid = RandomizedSearchCV(estimator=MLPRegressor(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)    

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_   
        
        
    elif estimator == 'xgboost':
        
        from xgboost import XGBRegressor
        
        param_grid = {'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                      'n_estimators':[10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
                      'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                      'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
                      'colsample_bytree': [0.5, 0.7, 0.9, 1],
                      'min_child_weight': [1, 2, 3, 4]
                     }

        model_grid = RandomizedSearchCV(estimator=XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0), 
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
            
        model_grid = RandomizedSearchCV(estimator=lgb.LGBMRegressor(random_state=seed), 
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
    
    #when using normal BaggingRegressor() DT estimator raise's an exception for max_features parameter. Hence a separate 
    #call has been made for estimator='dt' and method = 'Bagging' where max_features has been removed from param_grid_dt.
    
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import BaggingRegressor

        param_grid = {'n_estimators': [10,15,20,25,30],
                     'max_samples': [0.3,0.5,0.6,0.7,0.8,0.9],
                     'max_features':[0.3,0.5,0.6,0.7,0.8,0.9],
                     'bootstrap': [True, False],
                     'bootstrap_features': [True, False],
                     }

        param_grid_dt = {"max_depth": np.random.randint(3, (len(X_train.columns)*.85),4),
                         "min_samples_leaf": [2,3,4],
                         "min_samples_leaf": [0.1,0.2,0.3,0.4,0.5],
                         "min_samples_split" : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                         "min_weight_fraction_leaf" : [0.1,0.2,0.3,0.4,0.5],
                         "min_impurity_decrease" : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                         "criterion": ["mse", "mae", "friedman_mse"]}


        model_grid = RandomizedSearchCV(estimator=DecisionTreeRegressor(random_state=seed), param_distributions=param_grid_dt,
                                       scoring=optimize, n_iter=n_iter, cv=fold, random_state=seed,
                                       iid=False, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

        best_model = BaggingRegressor(best_model, random_state=seed)

        model_grid = RandomizedSearchCV(estimator=best_model, 
                                        param_distributions=param_grid, n_iter=n_iter, 
                                        cv=fold, random_state=seed, iid=False, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_    
  
    elif ensemble and method == 'Bagging':
    
        from sklearn.ensemble import BaggingRegressor

        param_grid = {'n_estimators': [10,15,20,25,30],
                     'max_samples': [0.3,0.5,0.6,0.7,0.8,0.9],
                     'max_features':[0.3,0.5,0.6,0.7,0.8,0.9],
                     'bootstrap': [True, False],
                     'bootstrap_features': [True, False],
                     }

        best_model = BaggingRegressor(best_model, random_state=seed)

        model_grid = RandomizedSearchCV(estimator=best_model, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=fold, random_state=seed, iid=False, n_jobs=-1)

        model_grid.fit(X_train,y_train)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_    
  
      
    elif ensemble and method =='Boosting':
    
        from sklearn.ensemble import AdaBoostRegressor

        param_grid = {'n_estimators': [10, 40, 70, 80, 90, 100, 120, 140, 150],
                      'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                      'loss' : ["linear", "square", "exponential"]
                     }          

        best_model = AdaBoostRegressor(best_model, random_state=seed)

        model_grid = RandomizedSearchCV(estimator=best_model, 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=fold, random_state=seed, iid=False, n_jobs=-1)
    
    
   
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
        model.fit(Xtrain,ytrain)
        pred_ = model.predict(Xtest)
        mae = metrics.mean_absolute_error(ytest,pred_)
        mse = metrics.mean_squared_error(ytest,pred_)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(ytest,pred_)
        max_error_ = metrics.max_error(ytest,pred_)
        score_mae = np.append(score_mae,mae)
        score_mse = np.append(score_mse,mse)
        score_rmse = np.append(score_rmse,rmse)
        score_r2 =np.append(score_r2,r2)
        score_max_error = np.append(score_max_error,max_error_)
            
        progress.value += 1
            
            
        '''
        
        This section is created to update_display() as code loops through the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'MAE':[mae], 'MSE': [mse], 'RMSE': [rmse], 
                                     'R2': [r2], 'ME': [max_error_]}).round(round)
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
    
    mean_mae=np.mean(score_mae)
    mean_mse=np.mean(score_mse)
    mean_rmse=np.mean(score_rmse)
    mean_r2=np.mean(score_r2)
    mean_me=np.mean(score_max_error)
    std_mae=np.std(score_mae)
    std_mse=np.std(score_mse)
    std_rmse=np.std(score_rmse)
    std_r2=np.std(score_r2)
    std_me=np.std(score_max_error)

    avgs_mae = np.append(avgs_mae, mean_mae)
    avgs_mae = np.append(avgs_mae, std_mae) 
    avgs_mse = np.append(avgs_mse, mean_mse)
    avgs_mse = np.append(avgs_mse, std_mse)
    avgs_rmse = np.append(avgs_rmse, mean_rmse)
    avgs_rmse = np.append(avgs_rmse, std_rmse)
    avgs_r2 = np.append(avgs_r2, mean_r2)
    avgs_r2 = np.append(avgs_r2, std_r2)
    avgs_max_error = np.append(avgs_max_error, mean_me)
    avgs_max_error = np.append(avgs_max_error, std_me)
    

    progress.value += 1
    
    model_results = pd.DataFrame({'MAE': score_mae, 'MSE': score_mse, 'RMSE' : score_rmse, 'R2' : score_r2 , 
                     'ME' : score_max_error})
    model_avgs = pd.DataFrame({'MAE': avgs_mae, 'MSE': avgs_mse, 'RMSE' : avgs_rmse, 'R2' : avgs_r2, 
                     'ME' : avgs_max_error},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)

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

def stack_models(estimator_list, 
                 meta_model = None, 
                 fold = 10,
                 round = 4, 
                 restack = False, 
                 plot = False,
                 verbose = True):
    
    """
            
    Description:
    ------------
    This function creates a meta model and scores it using Kfold Cross Validation,
    the prediction from base level models passed as estimator_list parameter is used
    as input feature for meta model. Restacking parameter control the ability to expose
    raw features to meta model when set to True (default = False). 

    The output prints the score grid that shows MAE, MSE, RMSE, R2 and ME by fold 
    (default = 10 Folds). Function returns a container which is the list of all models. 

    This is an original implementation of pycaret.

        Example:
        --------

        lasso = create_model('lasso')
        rf = create_model('rf')
        ada = create_model('ada')
        ridge = create_model('ridge')
        knn = create_model('knn')

        stack_models( [ nb, rf, ada, ridge, knn ] )

        This will result in creation of meta model that will use the predictions of 
        all the models provided as an input feature of meta model By default meta model 
        is Linear Regression but can be changed with meta_model param.

    Parameters
    ----------

    estimator_list : list of object

    meta_model : object, default = None
    if set to None, Linear Regression is used as a meta model.

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places metrics in score grid will be rounded to.

    restack: Boolean, default = False
    When restack is set to True, raw data will be exposed to meta model when
    making predictions, otherwise when False, only predicted values are 
    passed to meta model when making final predictions.

    plot: Boolean, default = False
    When plot is set to True, it will return the correlation plot of prediction
    from all base models provided in estimator_list.
    
    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are MAE, MSE, RMSE, R2 and ME. Mean and
                  standard deviation of the scores across the folds is also 
                  returned.

    container:    list of all models where last element is meta model.
    ----------

    Warnings:
    ---------
    
    None
        
           
          
    """
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #exception checking   
    import sys
    
    #checking error for estimator_list
    for i in estimator_list:
        if 'sklearn' not in str(type(i)):
            sys.exit("(Value Error): estimator_list parameter only trained model object")
            
    #checking meta model
    if meta_model is not None:
        if 'sklearn' not in str(type(meta_model)):
            sys.exit("(Value Error): estimator_list parameter only trained model object")
    
    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    
    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')

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
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import time, datetime
    
    #progress bar
    max_progress = len(estimator_list) + fold + 4
    progress = ipw.IntProgress(value=0, min=0, max=max_progress, step=1 , description='Processing: ')
    master_display = pd.DataFrame(columns=['MAE','MSE','RMSE', 'R2', 'ME'])
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
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_predict
    import seaborn as sns
    
    progress.value += 1
    
    #Defining meta model. Linear Regression hardcoded for now
    if meta_model == None:
        from sklearn.linear_model import LinearRegression
        meta_model = LinearRegression()
    else:
        meta_model = meta_model
    
    #defining model_library model names
    model_names = np.zeros(0)
    for item in estimator_list:
        model_names = np.append(model_names, str(item).split("(")[0])
    
    base_array = np.zeros((0,0))
    base_prediction = pd.DataFrame(y_train)
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
        
        progress.value += 1
        
        base_array = cross_val_predict(model,X_train,y_train,cv=fold, method='predict')
        base_array = base_array
        base_array_df = pd.DataFrame(base_array)
        base_prediction = pd.concat([base_prediction,base_array_df],axis=1)
        base_array = np.empty((0,0))
        
        counter += 1
        
    #defining column names now
    target_col_name = np.array(base_prediction.columns[0])
    model_names = np.append(target_col_name, model_names)
    base_prediction.columns = model_names #defining colum names now
    
    #defining data_X and data_y dataframe to be used in next stage.
    
    if restack:
        data_X_ = X_train
        data_X_ = data_X_.reset_index(drop=True)
        data_X = base_prediction.drop(base_prediction.columns[0],axis=1)
        data_X = pd.concat([data_X_,data_X],axis=1)
        
    elif restack == False:
        data_X = base_prediction.drop(base_prediction.columns[0],axis=1)
        
    data_y = base_prediction[base_prediction.columns[0]]
    
    #Correlation matrix of base_prediction
    base_prediction_cor = base_prediction.drop(base_prediction.columns[0],axis=1)
    base_prediction_cor = base_prediction_cor.corr()
    
    #Meta Modeling Starts Here
    
    model = meta_model #this defines model to be used below as model = meta_model (as captured above)
    
    kf = KFold(fold, random_state=seed) #capturing fold requested by user

    score_mae =np.empty((0,0))
    score_mse =np.empty((0,0))
    score_rmse =np.empty((0,0))
    score_r2 =np.empty((0,0))
    score_max_error =np.empty((0,0))
    avgs_mae =np.empty((0,0))
    avgs_mse =np.empty((0,0))
    avgs_rmse =np.empty((0,0))
    avgs_r2 =np.empty((0,0))
    avgs_max_error =np.empty((0,0))  
    
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
        pred_ = model.predict(Xtest)
        mae = metrics.mean_absolute_error(ytest,pred_)
        mse = metrics.mean_squared_error(ytest,pred_)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(ytest,pred_)
        max_error_ = metrics.max_error(ytest,pred_)
        score_mae = np.append(score_mae,mae)
        score_mse = np.append(score_mse,mse)
        score_rmse = np.append(score_rmse,rmse)
        score_r2 =np.append(score_r2,r2)
        score_max_error = np.append(score_max_error,max_error_)
        
        
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'MAE':[mae], 'MSE': [mse], 'RMSE': [rmse], 
                                     'R2': [r2], 'ME': [max_error_]}).round(round)
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
            ETC = 'Time to Completion : ' + tt + ' Seconds Remaining'
                
        else:
            tt = str (tt)
            ETC = 'Time to Completion : ' + tt + ' Minutes Remaining'
        
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
     
    mean_mae=np.mean(score_mae)
    mean_mse=np.mean(score_mse)
    mean_rmse=np.mean(score_rmse)
    mean_r2=np.mean(score_r2)
    mean_max_error=np.mean(score_max_error)
    std_mae=np.std(score_mae)
    std_mse=np.std(score_mse)
    std_rmse=np.std(score_rmse)
    std_r2=np.std(score_r2)
    std_max_error=np.std(score_max_error)
    
    
    avgs_mae = np.append(avgs_mae, mean_mae)
    avgs_mae = np.append(avgs_mae, std_mae) 
    avgs_mse = np.append(avgs_mse, mean_mse)
    avgs_mse = np.append(avgs_mse, std_mse)
    avgs_rmse = np.append(avgs_rmse, mean_rmse)
    avgs_rmse = np.append(avgs_rmse, std_rmse)
    avgs_r2 = np.append(avgs_r2, mean_r2)
    avgs_r2 = np.append(avgs_r2, std_r2)
    avgs_max_error = np.append(avgs_max_error, mean_max_error)
    avgs_max_error = np.append(avgs_max_error, std_max_error)
      
    model_results = pd.DataFrame({'MAE': score_mae, 'MSE': score_mse, 'RMSE' : score_rmse, 'R2' : score_r2 , 
                     'ME' : score_max_error})
    model_avgs = pd.DataFrame({'MAE': avgs_mae, 'MSE': avgs_mse, 'RMSE' : avgs_rmse, 'R2' : avgs_r2, 
                     'ME' : avgs_max_error},index=['Mean', 'SD'])
  
    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)  
    
    progress.value += 1
    
    models = []
    for i in estimator_list:
        models.append(i)
    
    models.append(meta_model)
    
    
    #storing into experiment
    model_name = 'Stacking Regressor (Single Layer)'
    tup = (model_name,models)
    experiment__.append(tup)
    nam = str(model_name) + ' Score Grid'
    tup = (nam, model_results)
    experiment__.append(tup)
    
    if plot:
        clear_output()
        ax = sns.heatmap(base_prediction_cor, vmin=1, vmax=1, center=0,cmap='magma', square=True, annot=True, 
                         linewidths=1)
    
    if verbose:
        clear_output()
        display(model_results)
        return models
    else:
        clear_output()
        return models

def create_stacknet(estimator_list,
                    meta_model = None,
                    fold = 10,
                    round = 4,
                    restack = False,
                    verbose = True):
    
    """
         
    Description:
    ------------
    This function creates a sequential stack net using cross validated predictions at
    each layer. The final score grid is predictions from meta model using Kfold 
    Cross Validation. Base level models can be passed as estimator_list param, the
    layers can be organized as a sub list within the estimator_list object. 
    Restacking param control the ability to expose raw features to meta model.

        Example:
        --------

        lasso = create_model( 'lasso' )
        rf = create_model( 'rf' )
        ada = create_model( 'ada' )
        ridge = create_model( 'ridge' )
        knn = create_model( 'knn' )

        create_stacknet( [ [ lasso, rf ], [ ada, ridge, knn] ] )

        This will result in stacking of models in multiple layers. The first layer 
        contains lasso and rf, the predictions of which is used by models in second layer
        to produce predictions which is used by meta model to generate final predictions.
        By default meta model is Linear Regression but can be changed with meta_model
        param.

    Parameters
    ----------

    estimator_list : nested list of objects

    meta_model : object, default = None
    if set to None, Linear Regression is used as a meta model.

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places metrics in score grid will be rounded to.
  
    restack: Boolean, default = False
    When restack is set to True, raw data will be exposed to meta model when
    making predictions, otherwise when False, only the predicted values are
    passed to meta model when making final predictions.

    verbose: Boolean, default = True
    Score grid is not printed when verbose is set to False.

    Returns:
    --------

    score grid:   A table containing the scores of the model across the kfolds. 
    -----------   Scoring metrics used are MAE, MSE, RMSE, R2 and ME. Mean and 
                  standard deviation of the scores across the folds is also 
                  returned.

    container:    list of all models where last element is meta model.
    ----------

    Warnings:
    ---------
    
    None
    
    
    """

    
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #for checking only
    #No active test
    
    #exception checking   
    import sys
    
    #checking error for estimator_list
    for i in estimator_list:
        for j in i:
            if 'sklearn' not in str(type(j)):
                sys.exit("(Value Error): estimator_list parameter only trained model object")
            
    #checking meta model
    if meta_model is not None:
        if 'sklearn' not in str(type(meta_model)):
            sys.exit("(Value Error): estimator_list parameter only trained model object")
    
    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    
    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')
 
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
        master_display = pd.DataFrame(columns=['MAE','MSE','RMSE', 'R2', 'ME'])
        display_ = display(master_display, display_id=True)
        display_id = display_.display_id
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    #general dependencies
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_predict

    progress.value += 1
    
    #global base_array_df
    global base_level_names, base_level
    base_level = estimator_list[0]
    base_level_names = []
    
    #defining base_level_names
    for item in base_level:
            base_level_names = np.append(base_level_names, str(item).split("(")[0])
    
    inter_level = estimator_list[1:]
    inter_level_names = []
   
    #defining inter_level names
    for item in inter_level:
        for m in item:
            inter_level_names = np.append(inter_level_names, str(m).split("(")[0])    
    
    #defining data_X and data_y
    data_X = X_train
    data_y = y_train
    
    #defining meta model
    
    if meta_model == None:
        from sklearn.linear_model import LinearRegression
        meta_model = LinearRegression()
    else:
        meta_model = meta_model
        
    base_array = np.zeros((0,0))
    base_array_df = pd.DataFrame()
    base_prediction = pd.DataFrame(y_train)
    base_prediction = base_prediction.reset_index(drop=True)
    
    base_counter = 0
    
    for model in base_level:
        
        '''
        MONITOR UPDATE STARTS
        '''

        monitor.iloc[1,1:] = 'Evaluating ' + base_level_names[base_counter]
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        progress.value += 1
                     
        base_array = cross_val_predict(model,X_train,y_train,cv=fold, method='predict')
        base_array = base_array
        base_array = pd.DataFrame(base_array)
        base_array_df = pd.concat([base_array_df, base_array], axis=1)
        base_array = np.empty((0,0))
        
        #changing column names to avoid xgboost failure
        name_col = []
        for i in range(0,len(base_array_df.columns)):
            n = 'model_' + str(i)
            name_col.append(n)

        base_array_df.columns = name_col
        
        base_counter += 1
    
    inter_counter = 0
    
    for level in inter_level:
        
        for model in level:
            
            '''
            MONITOR UPDATE STARTS
            '''

            monitor.iloc[1,1:] = 'Evaluating ' + inter_level_names[inter_counter]
            update_display(monitor, display_id = 'monitor')

            '''
            MONITOR UPDATE ENDS
            '''
        
            base_array = cross_val_predict(model,base_array_df,base_prediction,cv=fold, method='predict')
            base_array = base_array
            base_array = pd.DataFrame(base_array)
            base_array_df = pd.concat([base_array, base_array_df], axis=1)
            base_array = np.empty((0,0))
            
            #changing column names to avoid xgboost failure
            name_col = []
            for i in range(0,len(base_array_df.columns)):
                n = 'model_' + str(i)
                name_col.append(n)
                
            base_array_df.columns = name_col
    
            inter_counter += 1
        
        if restack == False:
            base_array_df = base_array_df.iloc[:,:len(level)]
        else:
            base_array_df = base_array_df
    
    model = meta_model
    
    kf = KFold(fold, random_state=seed) #capturing fold requested by user

    score_mae =np.empty((0,0))
    score_mse =np.empty((0,0))
    score_rmse =np.empty((0,0))
    score_r2 =np.empty((0,0))
    score_max_error =np.empty((0,0))
    avgs_mae =np.empty((0,0))
    avgs_mse =np.empty((0,0))
    avgs_rmse =np.empty((0,0))
    avgs_r2 =np.empty((0,0))
    avgs_max_error =np.empty((0,0))  
    
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
        pred_ = model.predict(Xtest)
        mae = metrics.mean_absolute_error(ytest,pred_)
        mse = metrics.mean_squared_error(ytest,pred_)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(ytest,pred_)
        max_error_ = metrics.max_error(ytest,pred_)
        score_mae = np.append(score_mae,mae)
        score_mse = np.append(score_mse,mse)
        score_rmse = np.append(score_rmse,rmse)
        score_r2 =np.append(score_r2,r2)
        score_max_error = np.append(score_max_error,max_error_)

        progress.value += 1
        
        '''
        
        This section handles time calculation and is created to update_display() as code loops through 
        the fold defined.
        
        '''
        
        fold_results = pd.DataFrame({'MAE':[mae], 'MSE': [mse], 'RMSE': [rmse], 
                                     'R2': [r2], 'ME': [max_error_]}).round(round)
        
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
    
    mean_mae=np.mean(score_mae)
    mean_mse=np.mean(score_mse)
    mean_rmse=np.mean(score_rmse)
    mean_r2=np.mean(score_r2)
    mean_max_error=np.mean(score_max_error)
    std_mae=np.std(score_mae)
    std_mse=np.std(score_mse)
    std_rmse=np.std(score_rmse)
    std_r2=np.std(score_r2)
    std_max_error=np.std(score_max_error)
    
    
    avgs_mae = np.append(avgs_mae, mean_mae)
    avgs_mae = np.append(avgs_mae, std_mae) 
    avgs_mse = np.append(avgs_mse, mean_mse)
    avgs_mse = np.append(avgs_mse, std_mse)
    avgs_rmse = np.append(avgs_rmse, mean_rmse)
    avgs_rmse = np.append(avgs_rmse, std_rmse)
    avgs_r2 = np.append(avgs_r2, mean_r2)
    avgs_r2 = np.append(avgs_r2, std_r2)
    avgs_max_error = np.append(avgs_max_error, mean_max_error)
    avgs_max_error = np.append(avgs_max_error, std_max_error)
      
    model_results = pd.DataFrame({'MAE': score_mae, 'MSE': score_mse, 'RMSE' : score_rmse, 'R2' : score_r2 , 
                     'ME' : score_max_error})
    model_avgs = pd.DataFrame({'MAE': avgs_mae, 'MSE': avgs_mse, 'RMSE' : avgs_rmse, 'R2' : avgs_r2, 
                     'ME' : avgs_max_error},index=['Mean', 'SD'])
  
    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)      
    
    progress.value += 1
        
    models_ = []
    
    for i in estimator_list:
        models_.append(i)
        
    models_.append(meta_model)
    
    #storing into experiment
    model_name = 'Stacking Regressor (Multi Layer)'
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

def plot_model(estimator, 
               plot = 'residuals'): 
    
    
    """
          
    Description:
    ------------
    This function takes a trained model object and returns the plot on test set.
    Model may get re-trained in the process, as maybe required in certain cases.
    See list of plots supported below. 

        Example:
        --------

        plot_model(lr)

        This will return residual plot of trained Linear Regression.
        variable 'lr' is created used lr = create_model('lr')


    Parameters
    ----------

    estimator : object, default = none

    A trained model object should be passed as an estimator. 
    Model must be created using create_model() or tune_model() in pycaret or using any
    other package that returns sklearn object.

    plot : string, default = residual
    Enter abbreviation of type of plot. The current list of plots supported are:

    Name                        Abbreviated String     Original Implementation 
    ---------                   ------------------     -----------------------
    Residuals Plot               'residuals'           .. / residuals.html
    Prediction Error Plot        'error'               .. / peplot.html
    Cooks Distance Plot          'cooks'               .. / influence.html
    Recursive Feat. Selection    'rfe'                 .. / rfecv.html
    Learning Curve               'learning'            .. / learning_curve.html
    Validation Curve             'vc'                  .. / validation_curve.html
    Manifold Learning            'manifold'            .. / manifold.html
    Feature Importance           'feature'                   N/A 
    Model Hyperparameter         'parameter'                 N/A 

    ** https://www.scikit-yb.org/en/latest/api/regressor/<reference>

    Returns:
    --------

    Visual Plot:  Prints the visual plot. 
    ------------

    Warnings:
    ---------
    
    None
              
       

    """  
    
    
    '''
    
    ERROR HANDLING STARTS HERE
    
    '''
    
    #for testing
    #No active testing
    
    #exception checking   
    import sys
    
    #checking plots (string)
    available_plots = ['residuals', 'error', 'cooks', 'feature', 'parameter', 'rfe', 'learning', 'manifold', 'vc']
    
    if plot not in available_plots:
        sys.exit('(Value Error): Plot Not Available. Please see docstring for list of available Plots.')
        
    #checking for feature plot
    if not ( hasattr(estimator, 'coef_') or hasattr(estimator,'feature_importances_') ) and (plot == 'feature' or plot == 'rfe'):
        sys.exit('(Type Error): Feature Importance plot not available for estimators with coef_ attribute.')
    
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
    
    if plot == 'residuals':
        
        from yellowbrick.regressor import ResidualsPlot
        progress.value += 1
        visualizer = ResidualsPlot(model)
        progress.value += 1
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        progress.value += 1
        clear_output()
        visualizer.show()
        
        
    elif plot == 'error':
        from yellowbrick.regressor import PredictionError
        progress.value += 1
        visualizer = PredictionError(model)
        progress.value += 1
        visualizer.score(X_test, y_test)
        progress.value += 1
        clear_output()
        visualizer.show()
        
    elif plot == 'cooks':
        from yellowbrick.regressor import CooksDistance
        progress.value += 1
        visualizer = CooksDistance()
        progress.value += 1
        visualizer.fit(X, y)
        progress.value += 1
        clear_output()
        visualizer.show() 
        
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

    elif plot == 'vc':
        
        model_name = str(model).split("(")[0]
        
        not_allowed = ['LinearRegression', 'PassiveAggressiveRegressor']
        
        if model_name in not_allowed:
            clear_output()
            sys.exit('(Value Error): Estimator not supported in Validation Curve Plot.')
        
        elif model_name == 'GradientBoostingRegressor':
            param_name='alpha'
            param_range = np.arange(0.1,1,0.1)
        
        #lasso/ridge/en/llar/huber/kr/mlp/br/ard
        elif hasattr(model, 'alpha'):
            param_name='alpha'
            param_range = np.arange(0,1,0.1)
            
        elif hasattr(model, 'alpha_1'):
            param_name='alpha_1'
            param_range = np.arange(0,1,0.1)
            
        #par/svm
        elif hasattr(model, 'C'):
            param_name='C'
            param_range = np.arange(1,11)
            
        #tree based models (dt/rf/et)
        elif hasattr(model, 'max_depth'):
            param_name='max_depth'
            param_range = np.arange(1,11)
        
        #knn
        elif hasattr(model, 'n_neighbors'):
            param_name='n_neighbors'
            param_range = np.arange(1,11)         
            
        #Bagging / Boosting (ada/gbr)
        elif hasattr(model, 'n_estimators'):
            param_name='n_estimators'
            param_range = np.arange(1,100,10)   

        #Bagging / Boosting (ada/gbr)
        elif hasattr(model, 'n_nonzero_coefs'):
            param_name='n_nonzero_coefs'
            if len(X_train.columns) >= 10:
                param_max = 11
            else:
                param_max = len(X_train.columns)+1
            param_range = np.arange(1,param_max,1) 
            
        elif hasattr(model, 'eps'):
            param_name='eps'
            param_range = np.arange(0,1,0.1)   
            
        elif hasattr(model, 'max_subpopulation'):
            param_name='max_subpopulation'
            param_range = np.arange(1000,20000,2000)   

        elif hasattr(model, 'min_samples'):
            param_name='min_samples'
            param_range = np.arange(0.01,1,0.1)  
            
        else: 
            clear_output()
            sys.exit('(Value Error): Estimator not supported in Validation Curve Plot.')
        
            
        progress.value += 1
            
        from yellowbrick.model_selection import ValidationCurve
        viz = ValidationCurve(model, param_name=param_name, param_range=param_range,cv=10, 
                              random_state=seed)
        viz.fit(X_train, y_train)
        progress.value += 1
        clear_output()
        viz.poof()
        
    elif plot == 'feature':
        if hasattr(estimator, 'coef_'):
            variables = abs(model.coef_)
        else:
            variables = abs(model.feature_importances_)
        col_names = np.array(X_train.columns)
        global coef_df
        coef_df = pd.DataFrame({'Variable': X_train.columns, 'Value': variables})
        progress.value += 1
        sorted_df = coef_df.sort_values(by='Value', ascending=False)
        sorted_df = sorted_df.head(10)
        sorted_df = sorted_df.sort_values(by='Value')
        my_range=range(1,len(sorted_df.index)+1)
        plt.figure(figsize=(8,5))
        plt.hlines(y=my_range, xmin=0, xmax=sorted_df['Value'], color='skyblue')
        plt.plot(sorted_df['Value'], my_range, "o")
        plt.yticks(my_range, sorted_df['Variable'])
        progress.value += 1
        plt.title("Feature Importance Plot")
        plt.xlabel('Variable Importance')
        plt.ylabel('Features') 
        #var_imp = sorted_df.reset_index(drop=True)
        #var_imp_array = np.array(var_imp['Variable'])
        progress.value += 1
        clear_output()
        #var_imp_array_top_n = var_imp_array[0:len(var_imp_array)]
   
    elif plot == 'parameter':
        
        clear_output()
        param_df = pd.DataFrame.from_dict(estimator.get_params(estimator), orient='index', columns=['Parameters'])
        display(param_df)


def interpret_model(estimator,
                   plot = 'summary',
                   feature = None, 
                   observation = None):
    
    
    """
          
    Description:
    ------------
    This function takes a trained model object and returns the interpretation plot on
    test set. This function only supports tree based algorithm. 

    This function is implemented based on original implementation in package 'shap'.
    SHAP (SHapley Additive exPlanations) is a unified approach to explain the output 
    of any machine learning model. SHAP connects game theory with local explanations.

    For more information : https://shap.readthedocs.io/en/latest/

        Example:
        --------

        dt = create_model('dt')
        interpret_model(dt)

        This will return the summary interpretation plot of Decision Tree model.

    Parameters
    ----------

    estimator : object, default = none
    A trained tree based model object should be passed as an estimator. 
    Model must be created using create_model() or tune_model() in pycaret or using 
    any other package that returns sklearn object.

    plot : string, default = 'summary'
    other available options are 'correlation' and 'reason'.

    feature: string, default = None
    This parameter is only needed when plot = 'correlation'. By default feature is set
    to None which means the first column of dataset will be used as a variable. 
    To change feature param must be passed. 

    observation: integer, default = None
    This parameter only comes in effect when plot is set to 'reason'. If no observation
    number is provided, by default the it will return the analysis of all observations 
    with option to select the feature  on x and y axis through drop down interactivity. 
    For analysis at sample level, observation parameter must be passed with index value 
    of observation in test set. 

    Returns:
    --------

    Visual Plot:  Returns the visual plot.
    -----------   Returns the interactive JS plot when plot = 'reason'.

    Warnings:
    ---------
    None    
       
         
    """
    
    
    
    '''
    Error Checking starts here
    
    '''
    
    import sys
    
    #allowed models
    allowed_models = ['RandomForestRegressor',
                      'DecisionTreeRegressor',
                      'ExtraTreesRegressor',
                      'GradientBoostingRegressor',
                      'XGBRegressor',
                      'LGBMRegressor']
    
    model_name = str(estimator).split("(")[0]
    
    if model_name not in allowed_models:
        sys.exit('(Type Error): This function only supports tree based models.')
        
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
    
    if plot == 'summary':
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test)
                              
    elif plot == 'correlation':
        
        if feature == None:
            
            dependence = X_test.columns[0]
            
        else:
            
            dependence = feature
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test) 
        shap.dependence_plot(dependence, shap_values, X_test)
        
    elif plot == 'reason':
     
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

def evaluate_model(estimator):
    
    
    """
          
    Description:
    ------------
    This function displays user interface for all the available plots for 
    a given estimator. It internally uses plot_model() function. 
    
        Example:
        --------
        
        dt = create_model('dt')
        evaluate_model(dt)
        
        This will display the User Interface for all the plots for given
        estimator, in this case decision tree passed as 'dt'.

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
                                     ('Residuals Plot', 'residuals'), 
                                     ('Prediction Error Plot', 'error'), 
                                     ('Cooks Distance Plot', 'cooks'),
                                     ('Recursive Feature Selection', 'rfe'),
                                     ('Learning Curve', 'learning'),
                                     ('Validation Curve', 'vc'),
                                     ('Manifold Learning', 'manifold'),
                                     ('Feature Importance', 'feature')
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
    This function fits the estimator on complete dataset as passed into setup() 
    stage. The purpose of this function is to prepare for deployment. After 
    experimentation, one should be able to choose the final model for deployment.
    
        Example:
        --------
        
        dt = create_model('dt')
        model_for_deployment = finalize_model(dt)
        
        This will return the final model object fitted to complete dataset. 

    Parameters
    ----------

    estimator : object, default = none
    A trained model object should be passed as an estimator. 
    Model must be created using any function in pycaret that returns trained model
    object. 

    Returns:
    --------

    Model:  Trained model object fitted on complete dataset.
    ------   

    Warnings:
    ---------
    None    
       
         
    """
    
    
    model = estimator.fit(X,y)
    
    #storing into experiment
    model_name = str(estimator).split("(")[0]
    model_name = 'Final ' + model_name
    tup = (model_name,model)
    experiment__.append(tup)
    
    return model

def save_model(model, model_name):
    
    """
          
    Description:
    ------------
    This function saves the trained model object in current active directory
    as a pickle file for later use. 
    
        Example:
        --------
        
        lr = create_model('lr')
        save_model(lr, 'lr_model_23122019')
        
        This will save the model as binary pickle file in current directory. 

    Parameters
    ----------

    model : object, default = none
    A trained model object should be passed as an estimator. 
    
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
        
        saved_lr = load_model('lr_model_23122019')
        
        This will call the trained model in saved_lr variable using model_name param.
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

def automl(qualifier = 5,
           target_metric = 'R2',
           fold = 10, 
           round = 4,
           turbo = True):
    
    
    """
         
    Description:
    ------------
    This function is an original implementation of pycaret. It sequentially creates
    various model and apply different techniques for Ensembling and Stacking. It 
    returns the best model based on 'target_metric' param defined. To limit the 
    processing time, 'qualifier' param can be reduced (by default = 5). Turbo param 
    is used for blacklisting certain models ('kr', 'ard', 'mlp') to become 
    part of automl(). By default turbo is set to True.

        Example:
        --------

        automl_1 = automl()

        ** All parameters are optional

    Parameters
    ----------

    qualifier : integer, default = None
    Number of top models considered for experimentation to return the best model.
    Higher number will result in longer training time.

    target_metric : String, default = 'R2'
    Metric to use for qualifying models and tuning the hyperparameters.
    Other available values are 'MAE', MSE', 'RMSE', 'ME'.

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    round: integer, default = 4
    Number of decimal places metrics in score grid will be rounded to.

    turbo: Boolean, default = True
    When turbo is set to True, it blacklists estimator that uses Radial Kernel.

    Returns:
    --------

    score grid:   A table containing the averaged Kfold scores of all the models
    -----------   Scoring metrics used are MAE, MSE, RMSE, R2 and ME. 
 
    model:        trained model object (best model selected using target_metric param)
    -------

    Warnings:
    ---------
    None
        
       
    """
    
    #for checking only
    #NO ACTIVE TEST
    
    #base dependencies
    from IPython.display import clear_output, update_display
    import time, datetime
    import numpy as np
    import pandas as pd
    import random
    import sys
    
    #master collector
    #This is being used for appending throughout the process
    global master, master_results, master_display, progress
    master = []
    master_results = pd.DataFrame(columns=['Model', 'MAE','MSE','RMSE', 'R2', 'ME'])
    master_display = pd.DataFrame(columns=['Model', 'MAE','MSE','RMSE', 'R2', 'ME']) 
    
    #progress bar
    import ipywidgets as ipw
    if turbo:
        cand_num = 21
    else:
        cand_num = 24
        
    max_progress = (cand_num*fold) + (7*qualifier*fold) +  30 
    progress = ipw.IntProgress(value=0, min=0, max=max_progress, step=1 , description='Processing: ')
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
    
    #automl parameters to be used in this function
    top_n = qualifier #top_n candidates for processing
    
    if target_metric == 'MAE':
        optimize = 'neg_mean_absolute_error'
        sort = 'MAE'
        
    elif target_metric == 'MSE':
        optimize = 'neg_mean_squared_error'
        sort = 'MSE'     
        
    elif target_metric == 'RMSE':
        optimize = 'neg_mean_squared_error'
        sort = 'RMSE'        

    elif target_metric == 'R2':
        optimize = 'r2'
        sort = 'R2'
   
    elif target_metric == 'ME':
        optimize = 'max_error'
        sort = 'ME'
        
    n_iter = 10 #number of iteration for tuning
    
    #PROGRESS # 1 : parameters defined
    progress.value += 1
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 

    #defining X_train and y_train
    data_X = X_train
    data_y=y_train
    
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Loading Estimator'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    #sklearn dependencies
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import Lars
    from sklearn.linear_model import LassoLars
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.linear_model import BayesianRidge
    from sklearn.linear_model import ARDRegression
    from sklearn.linear_model import PassiveAggressiveRegressor
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import TheilSenRegressor
    from sklearn.linear_model import HuberRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    import lightgbm as lgb
    
    #sklearn ensembling dependencies
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    
    #other imports from sklearn
    from sklearn.model_selection import KFold
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import cross_val_predict
    from sklearn import metrics
    
    #PROGRESS # 2 : Dependencies Loaded
    progress.value += 1
    
    #create sklearn model objects
    lr = LinearRegression()
    lasso = Lasso(random_state=seed)
    ridge = Ridge(random_state=seed)
    en = ElasticNet(random_state=seed)
    lar = Lars()
    llar = LassoLars()
    omp = OrthogonalMatchingPursuit()
    br = BayesianRidge()
    ard = ARDRegression()
    par = PassiveAggressiveRegressor(random_state=seed)
    ransac = RANSACRegressor(random_state=seed)
    tr = TheilSenRegressor(random_state=seed)
    huber = HuberRegressor()
    kr = KernelRidge()
    svm = SVR()
    knn = KNeighborsRegressor()
    dt = DecisionTreeRegressor(random_state=seed)
    rf = RandomForestRegressor(random_state=seed)
    et = ExtraTreesRegressor(random_state=seed)
    ada = AdaBoostRegressor(random_state=seed)
    gbr = GradientBoostingRegressor(random_state=seed)
    mlp = MLPRegressor(random_state=seed)
    xgboost = XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0)
    lightgbm = lgb.LGBMRegressor(random_state=seed)
    
    if turbo:
        
        model_library = [lr, lasso, ridge, en, lar, llar, omp, br, par, ransac, tr, huber, 
                         svm, knn, dt, rf, et, ada, gbr, xgboost, lightgbm]

        #defining model names
        model_names = ['Linear Regression',
                       'Lasso Regression',
                       'Ridge Regression',
                       'Elastic Net',
                       'Least Angle Regression',
                       'Lasso Least Angle Regression',
                       'Orthogonal Matching Pursuit',
                       'Bayesian Ridge',
                       'Passive Aggressive Regressor',
                       'Random Sample Consensus',
                       'TheilSen Regressor',
                       'Huber Regressor',
                       'Support Vector Machine',
                       'K Neighbors Regressor',
                       'Decision Tree',
                       'Random Forest',
                       'Extra Trees Regressor',
                       'AdaBoost Regressor',
                       'Gradient Boosting Regressor',
                       'Extreme Gradient Boosting',
                       'Light Gradient Boosting Machine']
          
    else:
        
        model_library = [lr, lasso, ridge, en, lar, llar, omp, br, ard, par, ransac, tr, huber, kr, 
                         svm, knn, dt, rf, et, ada, gbr, mlp, xgboost, lightgbm]

        #defining model names
        model_names = ['Linear Regression',
                       'Lasso Regression',
                       'Ridge Regression',
                       'Elastic Net',
                       'Least Angle Regression',
                       'Lasso Least Angle Regression',
                       'Orthogonal Matching Pursuit',
                       'Bayesian Ridge',
                       'Automatic Relevance Determination',
                       'Passive Aggressive Regressor',
                       'Random Sample Consensus',
                       'TheilSen Regressor',
                       'Huber Regressor',
                       'Kernel Ridge',
                       'Support Vector Machine',
                       'K Neighbors Regressor',
                       'Decision Tree',
                       'Random Forest',
                       'Extra Trees Regressor',
                       'AdaBoost Regressor',
                       'Gradient Boosting Regressor',
                       'Multi Level Perceptron',
                       'Extreme Gradient Boosting',
                       'Light Gradient Boosting Machine']      
        
        
    #PROGRESS # 3 : Models and name list compiled
    progress.value += 1
    
    
    '''
    Step 1 - Run all the models in model library.
    This function is equivalent to compare_models() without any blacklist model

    '''
    
    
    '''
    MONITOR UPDATE STARTS
    '''
    
    monitor.iloc[1,1:] = 'Initializing CV'
    update_display(monitor, display_id = 'monitor')
    
    '''
    MONITOR UPDATE ENDS
    '''
    
    #cross validation
    kf = KFold(fold, random_state=seed)

    score_mae =np.empty((0,0))
    score_mse =np.empty((0,0))
    score_rmse =np.empty((0,0))
    score_r2 =np.empty((0,0))
    score_max_error =np.empty((0,0))
    avg_mae =np.empty((0,0))
    avg_mse =np.empty((0,0))
    avg_rmse =np.empty((0,0))
    avg_r2 =np.empty((0,0))
    avg_max_error =np.empty((0,0))  
    
    name_counter = 0
    
    #PROGRESS # 4 : Process Initiated
    progress.value += 1
    
    for model in model_library:
        
        '''
        MONITOR UPDATE STARTS
        '''
        monitor.iloc[2,1:] = model_names[name_counter]
        monitor.iloc[3,1:] = 'Calculating ETC'
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        #PROGRESS # 5 : Loop Counter (15x)
        progress.value += 1
        
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
            
            #PROGRESS # 6 : Loop Counter (xfold)
            progress.value += 1
     
            Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
            ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
        
            model.fit(Xtrain,ytrain)
            pred_ = model.predict(Xtest)
            mae = metrics.mean_absolute_error(ytest,pred_)
            mse = metrics.mean_squared_error(ytest,pred_)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(ytest,pred_)
            max_error_ = metrics.max_error(ytest,pred_)
            score_mae = np.append(score_mae,mae)
            score_mse = np.append(score_mse,mse)
            score_rmse = np.append(score_rmse,rmse)
            score_r2 =np.append(score_r2,r2)
            score_max_error = np.append(score_max_error,max_error_)      
        
            t1 = time.time()
            
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
        
        avg_mae = np.append(avg_mae,np.mean(score_mae))
        avg_mse = np.append(avg_mse,np.mean(score_mse))
        avg_rmse = np.append(avg_rmse,np.mean(score_rmse))
        avg_r2 = np.append(avg_r2,np.mean(score_r2))
        avg_max_error = np.append(avg_max_error,np.mean(score_max_error))
        
        compare_models_ = pd.DataFrame({'Model':model_names[name_counter], 'MAE':avg_mae, 'MSE':avg_mse, 
                           'RMSE':avg_rmse, 'R2':avg_r2, 'ME':avg_max_error})
        
        master_display = pd.concat([master_display, compare_models_],ignore_index=True)
        master_display = master_display.round(round)
        
        if target_metric == 'R2':
            master_display = master_display.sort_values(by=sort,ascending=False)
        else:
            master_display = master_display.sort_values(by=sort,ascending=True)
        master_display.reset_index(drop=True, inplace=True)
        
        update_display(master_display, display_id = display_id)
        
        score_mae =np.empty((0,0))
        score_mse =np.empty((0,0))
        score_rmse =np.empty((0,0))
        score_r2 =np.empty((0,0))
        score_max_error =np.empty((0,0))
        
        avg_mae =np.empty((0,0))
        avg_mse =np.empty((0,0))
        avg_rmse =np.empty((0,0))
        avg_r2 =np.empty((0,0))
        avg_max_error =np.empty((0,0))  
        
        name_counter += 1
    
    if target_metric == 'R2':
        master_results = master_display.sort_values(by=sort, ascending=False).reset_index(drop=True)
    else:
        master_results = master_display.sort_values(by=sort, ascending=True).reset_index(drop=True)
        
    master_results = master_results.round(round)
    top_n_model_names = list(master_results.iloc[0:top_n]['Model'])
    top_n_model_results = master_results[:top_n]
    master_results = master_results[:top_n]

    #PROGRESS # 7 : Section Completed
    progress.value += 1    
    
    '''

    The section below is still part of Step 1. The purpose of this chunk is to 
    take the name string from 'top_n_model_names' and create a model that is being
    appended to master list. Models are re-created (In future, re-creation must be
    replaced by already created object for efficiency purpose).
    
    '''
    
    '''
    MONITOR UPDATE STARTS
    '''
    monitor.iloc[2,1:] = 'Compiling Top Models'
    monitor.iloc[3,1:] = 'Calculating ETC'
    update_display(monitor, display_id = 'monitor')

    '''
    MONITOR UPDATE ENDS
    '''
    
    top_n_models = []
    
    for i in top_n_model_names:
        
        #PROGRESS # 8 : Model Creation (qualifier x based on top_n_model parameter)
        progress.value += 1
        
        if i == 'Linear Regression':
            
            model = LinearRegression()
            top_n_models.append(model)
            
        elif i == 'Lasso Regression':
            
            model = Lasso(random_state=seed)
            top_n_models.append(model)
            
        elif i == 'Ridge Regression':
            
            model = Ridge(random_state=seed)
            top_n_models.append(model)
            
        elif i == 'Elastic Net':
            
            model =  ElasticNet(random_state=seed)           
            top_n_models.append(model)
            
        elif i == 'Least Angle Regression':
            
            model = Lars()
            top_n_models.append(model)
            
        elif i == 'Lasso Least Angle Regression':
            
            model = LassoLars()
            top_n_models.append(model)
            
        elif i == 'Orthogonal Matching Pursuit':
            
            model = OrthogonalMatchingPursuit()
            top_n_models.append(model)
            
        elif i == 'Bayesian Ridge':
            
            model = BayesianRidge()
            top_n_models.append(model)
            
        elif i == 'Automatic Relevance Determination':
            
            model = ARDRegression()
            top_n_models.append(model)
            
        elif i == 'Passive Aggressive Regressor':
            
            model = PassiveAggressiveRegressor(random_state=seed)
            top_n_models.append(model)
            
        elif i == 'Random Sample Consensus':
            
            model = RANSACRegressor(random_state=seed)
            top_n_models.append(model)
            
        elif i == 'TheilSen Regressor':
            
            model = TheilSenRegressor(random_state=seed)
            top_n_models.append(model)
            
        elif i == 'Huber Regressor':
            
            model = HuberRegressor()
            top_n_models.append(model)
            
        elif i == 'Kernel Ridge':
            
            model = KernelRidge()
            top_n_models.append(model)
            
        elif i == 'Support Vector Machine':
            
            model = SVR()
            top_n_models.append(model)
    
        elif i == 'K Neighbors Regressor':
            
            model = KNeighborsRegressor()
            top_n_models.append(model)
    
        elif i == 'Decision Tree':
            
            model = DecisionTreeRegressor(random_state=seed)
            top_n_models.append(model)
            
        elif i == 'Random Forest':
            
            model = RandomForestRegressor(random_state=seed)
            top_n_models.append(model)
            
        elif i == 'Extra Trees Regressor':
            
            model = ExtraTreesRegressor(random_state=seed)
            top_n_models.append(model)
            
        elif i == 'AdaBoost Regressor':
            
            model = AdaBoostRegressor(random_state=seed)
            top_n_models.append(model)
            
        elif i == 'Gradient Boosting Regressor':
            
            model = GradientBoostingRegressor(random_state=seed)
            top_n_models.append(model)            
            
        elif i == 'Multi Level Perceptron':
            
            model = MLPRegressor(random_state=seed)
            top_n_models.append(model)            
            
        elif i == 'Extreme Gradient Boosting':
            
            model = XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0)
            top_n_models.append(model) 

        elif i == 'Light Gradient Boosting Machine':
            
            model = lgb.LGBMRegressor(random_state=seed)
            top_n_models.append(model) 
            
            
    master.append(top_n_models) #appending top_n models to master list
    
    #PROGRESS # 9 : Sub-section completed
    progress.value += 1
    
    '''
    
    Step 2 - Create Ensemble Bagging using BaggingRegressor() from sklearn for all the 
    models in 'top_n_models' param defined above. Number of models at this stage in 
    'top_n_models' param is equal to # of models in 'master' param.
    
    This function is equivalent to ensemble_model().
    
    '''    
    top_n_bagged_models = []
    top_n_bagged_model_results = pd.DataFrame(columns=['Model', 'MAE','MSE','RMSE', 'R2', 'ME'])
    
    #defining names
    bagging_model_names = []
    for i in top_n_model_names:
        s = 'Ensemble ' + i + ' (Bagging)'
        bagging_model_names.append(s)
        
    #PROGRESS # 10 : Name Defined for Bagging
    progress.value += 1
    
    #counter for naming
    name_counter = 0 
    
    for i in top_n_models:
        
        '''
        MONITOR UPDATE STARTS
        '''
        monitor.iloc[2,1:] = bagging_model_names[name_counter]
        monitor.iloc[3,1:] = 'Calculating ETC'
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        #PROGRESS # 11 : Loop Counter (top_n X)
        progress.value += 1
       
        #from sklearn.ensemble import BaggingRegressor
        model = BaggingRegressor(i,bootstrap=True,n_estimators=10, random_state=seed)
        top_n_bagged_models.append(model)
    
        #setting cross validation
        kf = KFold(fold, random_state=seed)

        score_mae =np.empty((0,0))
        score_mse =np.empty((0,0))
        score_rmse =np.empty((0,0))
        score_r2 =np.empty((0,0))
        score_max_error =np.empty((0,0))
        
        avg_mae =np.empty((0,0))
        avg_mse =np.empty((0,0))
        avg_rmse =np.empty((0,0))
        avg_r2 =np.empty((0,0))
        avg_max_error =np.empty((0,0))
        
        
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
            
            
            #PROGRESS # 11 : Loop Counter (xfold)
            progress.value += 1
    
            Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
            ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
        
            model.fit(Xtrain,ytrain)
            pred_ = model.predict(Xtest)
            mae = metrics.mean_absolute_error(ytest,pred_)
            mse = metrics.mean_squared_error(ytest,pred_)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(ytest,pred_)
            max_error_ = metrics.max_error(ytest,pred_)
            score_mae = np.append(score_mae,mae)
            score_mse = np.append(score_mse,mse)
            score_rmse = np.append(score_rmse,rmse)
            score_r2 =np.append(score_r2,r2)
            score_max_error = np.append(score_max_error,max_error_) 
   
            t1 = time.time()
            
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
                
                
        mean_mae=np.mean(score_mae)
        mean_mse=np.mean(score_mse)
        mean_rmse=np.mean(score_rmse)
        mean_r2=np.mean(score_r2)
        mean_max_error=np.mean(score_max_error)

        avg_mae = np.append(avg_mae, mean_mae)
        avg_mse = np.append(avg_mse, mean_mse)
        avg_rmse = np.append(avg_rmse, mean_rmse)
        avg_r2 = np.append(avg_r2, mean_r2)
        avg_max_error = np.append(avg_max_error, mean_max_error)
        
        model_results = pd.DataFrame({'Model': bagging_model_names[name_counter], 'MAE': avg_mae, 'MSE': avg_mse, 
                                      'RMSE' : avg_rmse, 'R2' : avg_r2 , 'ME' : avg_max_error
                                      })
        model_results = model_results.round(round)
        master_display = pd.concat([master_display, model_results],ignore_index=True)
        master_display = master_display.round(round)
        
        if target_metric == 'R2':
            master_display = master_display.sort_values(by=sort,ascending=False)
        else:
            master_display = master_display.sort_values(by=sort,ascending=True)
            
        master_display.reset_index(drop=True, inplace=True)
        
        update_display(master_display, display_id = display_id)
        
        avg_mae = np.empty((0,0))
        avg_mse = np.empty((0,0))
        avg_rmse = np.empty((0,0))
        avg_r2 = np.empty((0,0))
        avg_max_error = np.empty((0,0))
        
        name_counter += 1

        top_n_bagged_model_results = pd.concat([top_n_bagged_model_results, model_results],ignore_index=True)  
        
    master_results = master_results.append(top_n_bagged_model_results)
    master.append(top_n_bagged_models) 
    
    #PROGRESS # 12 : Section Completed
    progress.value += 1
    
    '''
    
    Step 3 - Create Ensemble Boosting using AdaBoostRegressor() from sklearn for all the 
    models in 'top_n_models' param defined above. 
    
    This function is equivalent to ensemble_model(method = 'Boosting').
    
    '''        
    
    top_n_boosted_models = []
    top_n_boosted_model_results = pd.DataFrame(columns=['Model','MAE','MSE','RMSE', 'R2', 'ME'])
    
    boosting_model_names = []
    for i in top_n_model_names:
        s = 'Ensemble ' + i + ' (Boosting)'
        boosting_model_names.append(s)
   
    #PROGRESS # 13 : Name Defined for Boosting
    progress.value += 1
    
    #counter for naming
    name_counter = 0 
        
    for i in top_n_models:
        
        '''
        MONITOR UPDATE STARTS
        '''
        monitor.iloc[2,1:] = boosting_model_names[name_counter]
        monitor.iloc[3,1:] = 'Calculating ETC'
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        #PROGRESS # 14 : Model Creation (qualifier x based on top_n_model parameter)
        progress.value += 1
       
        model = AdaBoostRegressor(i, random_state=seed)
        top_n_boosted_models.append(model)
            
        #setting cross validation
        kf = KFold(fold, random_state=seed)

        score_mae =np.empty((0,0))
        score_mse =np.empty((0,0))
        score_rmse =np.empty((0,0))
        score_r2 =np.empty((0,0))
        score_max_error =np.empty((0,0))
    
        avg_mae =np.empty((0,0))
        avg_mse =np.empty((0,0))
        avg_rmse =np.empty((0,0))
        avg_r2 =np.empty((0,0))
        avg_max_error =np.empty((0,0))
        
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
            
            #PROGRESS # 15 : Loop Counter (xfold)
            progress.value += 1
    
            Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
            ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
            
            model.fit(Xtrain,ytrain)
            pred_ = model.predict(Xtest)
            mae = metrics.mean_absolute_error(ytest,pred_)
            mse = metrics.mean_squared_error(ytest,pred_)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(ytest,pred_)
            max_error_ = metrics.max_error(ytest,pred_)
            score_mae = np.append(score_mae,mae)
            score_mse = np.append(score_mse,mse)
            score_rmse = np.append(score_rmse,rmse)
            score_r2 =np.append(score_r2,r2)
            score_max_error = np.append(score_max_error,max_error_)             
                
            t1 = time.time()
            
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

        mean_mae=np.mean(score_mae)
        mean_mse=np.mean(score_mse)
        mean_rmse=np.mean(score_rmse)
        mean_r2=np.mean(score_r2)
        mean_max_error=np.mean(score_max_error)

        avg_mae = np.append(avg_mae, mean_mae)
        avg_mse = np.append(avg_mse, mean_mse)
        avg_rmse = np.append(avg_rmse, mean_rmse)
        avg_r2 = np.append(avg_r2, mean_r2)
        avg_max_error = np.append(avg_max_error, mean_max_error)
        
        model_results = pd.DataFrame({'Model': boosting_model_names[name_counter],'MAE': avg_mae, 
                                      'MSE': avg_mse, 'RMSE' : avg_rmse, 'R2' : avg_r2, 
                                      'ME' : avg_max_error})
        model_results = model_results.round(round)
        master_display = pd.concat([master_display, model_results],ignore_index=True)
        master_display = master_display.round(round)
        
        if target_metric == 'R2':
            master_display = master_display.sort_values(by=sort,ascending=False)
        else:
            master_display = master_display.sort_values(by=sort,ascending=True)
        
        master_display.reset_index(drop=True, inplace=True)
        
        update_display(master_display, display_id = display_id)
        
        avg_mae = np.empty((0,0))
        avg_mse = np.empty((0,0))
        avg_rmse = np.empty((0,0))
        avg_r2 = np.empty((0,0))
        avg_max_error = np.empty((0,0))
               
        name_counter += 1
        
        top_n_boosted_model_results = pd.concat([top_n_boosted_model_results, model_results],ignore_index=True)
        
    master_results = master_results.append(top_n_boosted_model_results)
    master.append(top_n_boosted_models)
    
    #PROGRESS # 16 : Section Completed
    progress.value += 1
 
    '''

    Step 4 - Tune all models in 'top_n_models' param defined in Step 1 above.
    This function is equivalent to tune_model().


    '''           
    
    #4.1 Store tuned model objects in the list 'top_n_tuned_models'
    
    cv = 3
    
    top_n_tuned_models = []
    
    name_counter = 0 
    
    for i in top_n_model_names:
        
        monitor.iloc[1,1:] = 'Hyperparameter Grid Search'
        monitor.iloc[2,1:] = top_n_model_names[name_counter]
        update_display(monitor, display_id = 'monitor')
        
        #PROGRESS # 17 : Model Creation (qualifier x based on top_n_model parameter)
        progress.value += 1
        
        if i == 'Linear Regression':

            from sklearn.linear_model import LinearRegression
            param_grid = {'fit_intercept': [True, False],
                         'normalize' : [True, False]
                        }        
            model_grid = RandomizedSearchCV(estimator=LinearRegression(), param_distributions=param_grid, 
                                            scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                            n_jobs=-1, iid=False)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_

        elif i == 'Lasso Regression':

            from sklearn.linear_model import Lasso

            param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                          'fit_intercept': [True, False],
                          'normalize' : [True, False],
                         }
            model_grid = RandomizedSearchCV(estimator=Lasso(random_state=seed), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv, 
                                            random_state=seed, iid=False,n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_

        elif i == 'Ridge Regression':

            from sklearn.linear_model import Ridge

            param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                          "fit_intercept": [True, False],
                          "normalize": [True, False],
                          }

            model_grid = RandomizedSearchCV(estimator=Ridge(random_state=seed), param_distributions=param_grid,
                                           scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                           iid=False, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_

        elif i == 'Elastic Net':

            from sklearn.linear_model import ElasticNet

            param_grid = {'alpha': [0.0001,0.001,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                          'l1_ratio' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                          'fit_intercept': [True, False],
                          'normalize': [True, False]
                         } 

            model_grid = RandomizedSearchCV(estimator=ElasticNet(random_state=seed), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv, 
                                            random_state=seed, iid=False, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_

        elif i == 'Least Angle Regression':

            from sklearn.linear_model import Lars

            param_grid = {'fit_intercept':[True, False],
                         'normalize' : [True, False],
                         'eps': [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.0005, 0.005, 0.00005, 0.02, 0.007]}

            model_grid = RandomizedSearchCV(estimator=Lars(), param_distributions=param_grid,
                                           scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                           n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_  

        elif i == 'Lasso Least Angle Regression':

            from sklearn.linear_model import LassoLars

            param_grid = {'alpha': [0.0001,0.001,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                         'fit_intercept':[True, False],
                         'normalize' : [True, False],
                         'eps': [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.0005, 0.005, 0.00005, 0.02, 0.007]}

            model_grid = RandomizedSearchCV(estimator=LassoLars(), param_distributions=param_grid,
                                           scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                           n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_    

        elif i == 'Orthogonal Matching Pursuit':

            from sklearn.linear_model import OrthogonalMatchingPursuit
            import random

            param_grid = {'n_nonzero_coefs': range(1,len(X_train.columns)+1),
                          'fit_intercept' : [True, False],
                          'normalize': [True, False]}

            model_grid = RandomizedSearchCV(estimator=OrthogonalMatchingPursuit(), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_        

        elif i == 'Bayesian Ridge':

            from sklearn.linear_model import BayesianRidge

            param_grid = {'alpha_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                          'alpha_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                          'lambda_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                          'lambda_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                          'compute_score': [True, False],
                          'fit_intercept': [True, False],
                          'normalize': [True, False]
                         }    

            model_grid = RandomizedSearchCV(estimator=BayesianRidge(), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_    

        elif i == 'Automatic Relevance Determination':

            from sklearn.linear_model import ARDRegression

            param_grid = {'alpha_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                          'alpha_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                          'lambda_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                          'lambda_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                          'threshold_lambda' : [5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000],
                          'compute_score': [True, False],
                          'fit_intercept': [True, False],
                          'normalize': [True, False]
                         }    

            model_grid = RandomizedSearchCV(estimator=ARDRegression(), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_       

        elif i == 'Passive Aggressive Regressor':

            from sklearn.linear_model import PassiveAggressiveRegressor

            param_grid = {'C': [0.01, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                          'fit_intercept': [True, False],
                          'early_stopping' : [True, False],
                          #'validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                          'loss' : ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                          'epsilon' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                          'shuffle' : [True, False]
                         }    

            model_grid = RandomizedSearchCV(estimator=PassiveAggressiveRegressor(random_state=seed), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_         

        elif i == 'Random Sample Consensus':

            from sklearn.linear_model import RANSACRegressor

            param_grid = {'min_samples': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                          'max_trials': [1,2,3,4,5,6,7,8,9,10],
                          'max_skips': [1,2,3,4,5,6,7,8,9,10],
                          'stop_n_inliers': [1,2,3,4,5,6,7,8,9,10],
                          'stop_probability': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                          'loss' : ['absolute_loss', 'squared_loss'],
                         }    

            model_grid = RandomizedSearchCV(estimator=RANSACRegressor(random_state=seed), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_         

        elif i == 'TheilSen Regressor':

            from sklearn.linear_model import TheilSenRegressor

            param_grid = {'fit_intercept': [True, False],
                          'max_subpopulation': [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000]
                         }    

            model_grid = RandomizedSearchCV(estimator=TheilSenRegressor(random_state=seed), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_    

        elif i == 'Huber Regressor':

            from sklearn.linear_model import HuberRegressor

            param_grid = {'epsilon': [1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9],
                          'alpha': [0.00001, 0.0001, 0.0003, 0.005, 0.05, 0.1, 0.0005, 0.15],
                          'fit_intercept' : [True, False]
                         }    

            model_grid = RandomizedSearchCV(estimator=HuberRegressor(), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_        

        elif i == 'Kernel Ridge':

            from sklearn.kernel_ridge import KernelRidge

            param_grid = {'alpha': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] }    

            model_grid = RandomizedSearchCV(estimator=KernelRidge(), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_       

        elif i == 'Support Vector Machine':

            from sklearn.svm import SVR

            param_grid = {#'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                          #'float' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                          'C' : [0.01, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                          'epsilon' : [1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9],
                          'shrinking': [True, False]
                         }    

            model_grid = RandomizedSearchCV(estimator=SVR(), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_     

        elif i == 'K Neighbors Regressor':

            from sklearn.neighbors import KNeighborsRegressor

            param_grid = {'n_neighbors': range(1,51),
                         'weights' :  ['uniform', 'distance'],
                         'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                         'leaf_size': [10,20,30,40,50,60,70,80,90]
                         } 

            model_grid = RandomizedSearchCV(estimator=KNeighborsRegressor(), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_         

        elif i == 'Decision Tree':

            from sklearn.tree import DecisionTreeRegressor

            param_grid = {"max_depth": np.random.randint(3, (len(X_train.columns)*.85),4),
                          "max_features": np.random.randint(3, len(X_train.columns),4),
                          "min_samples_leaf": [0.1,0.2,0.3,0.4,0.5],
                          "min_samples_split" : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                          "min_weight_fraction_leaf" : [0.1,0.2,0.3,0.4,0.5],
                          "min_impurity_decrease" : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                          "criterion": ["mse", "mae", "friedman_mse"],
                          #"max_leaf_nodes" : [1,2,3,4,5,6,7,8,9,10,None]
                         } 

            model_grid = RandomizedSearchCV(estimator=DecisionTreeRegressor(random_state=seed), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_         

        elif i == 'Random Forest':

            from sklearn.ensemble import RandomForestRegressor


            param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                          'criterion': ['mse', 'mae'],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                          'min_samples_split': [2, 5, 7, 9, 10],
                          'min_samples_leaf' : [1, 2, 4],
                          'max_features' : ['auto', 'sqrt', 'log2'],
                          'bootstrap': [True, False]
                          }

            model_grid = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=seed), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_       


        elif i == 'Extra Trees Regressor':

            from sklearn.ensemble import ExtraTreesRegressor

            param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                          'criterion': ['mse', 'mae'],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                          'min_samples_split': [2, 5, 7, 9, 10],
                          'min_samples_leaf' : [1, 2, 4],
                          'max_features' : ['auto', 'sqrt', 'log2'],
                          'bootstrap': [True, False]
                          }  

            model_grid = RandomizedSearchCV(estimator=ExtraTreesRegressor(random_state=seed), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_       

        elif i == 'AdaBoost Regressor':

            from sklearn.ensemble import AdaBoostRegressor

            param_grid = {'n_estimators': [10, 40, 70, 80, 90, 100, 120, 140, 150],
                          'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                          'loss' : ["linear", "square", "exponential"]
                         }    

            model_grid = RandomizedSearchCV(estimator=AdaBoostRegressor(random_state=seed), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_ 

        elif i == 'Gradient Boosting Regressor':

            from sklearn.ensemble import GradientBoostingRegressor

            param_grid = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                          'n_estimators': [10, 40, 70, 80, 90, 100, 120, 140, 150],
                          'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                          'subsample' : [0.1,0.3,0.5,0.7,0.9,1],
                          'criterion' : ['friedman_mse', 'mse', 'mae'],
                          'min_samples_split' : [2,4,5,7,9,10],
                          'min_samples_leaf' : [1,2,3,4,5],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                          'max_features' : ['auto', 'sqrt', 'log2']
                         }     

            model_grid = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=seed), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_         

        elif i == 'Multi Level Perceptron':

            from sklearn.neural_network import MLPRegressor

            param_grid = {'learning_rate': ['constant', 'invscaling', 'adaptive'],
                          'solver' : ['lbfgs', 'adam'],
                          'alpha': [0.0001, 0.001, 0.01, 0.00001, 0.003, 0.0003, 0.0005, 0.005, 0.05],
                          'hidden_layer_sizes': np.random.randint(50,150,10),
                          'activation': ["tanh", "identity", "logistic","relu"]
                          }    

            model_grid = RandomizedSearchCV(estimator=MLPRegressor(random_state=seed), 
                                            param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                            cv=cv, random_state=seed, n_jobs=-1)    

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_
            
            
        elif i == 'Extreme Gradient Boosting':
            
            param_grid = {'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                          'n_estimators':[10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
                          'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
                          'colsample_bytree': [0.5, 0.7, 0.9, 1],
                          'min_child_weight': [1, 2, 3, 4]
                         }

            model_grid = RandomizedSearchCV(estimator=XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_ 
            
            
        elif i == 'Light Gradient Boosting Machine':
            
        
            param_grid = {#'boosting_type' : ['gbdt', 'dart', 'goss', 'rf'],
                          'num_leaves': [10,20,30,40,50,60,70,80,90,100,150,200],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                          'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                          'n_estimators': [10, 30, 50, 70, 90, 100, 120, 150, 170, 200], 
                          'min_split_gain' : [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                          'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                          'reg_lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        }

            model_grid = RandomizedSearchCV(estimator=lgb.LGBMRegressor(random_state=seed), 
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, 
                                        cv=cv, random_state=seed, n_jobs=-1)

            model_grid.fit(X_train,y_train)
            model = model_grid.best_estimator_
            best_model = model_grid.best_estimator_
            best_model_param = model_grid.best_params_             
            
            
        top_n_tuned_models.append(best_model)
            
        name_counter += 1
            
    master.append(top_n_tuned_models)
    
    #PROGRESS # 18 : Sub Section Completed
    progress.value += 1 
    
    '''
    
    This section below is still continued from Step 4. In the part above tuned model
    object is stored in the list. In the part below the CV results are generated using
    stored objects in above step.
    
    '''
    
    tuning_model_names = []
    top_n_tuned_model_results = pd.DataFrame(columns=['Model', 'MAE','MSE','RMSE', 'R2', 'ME'])
    
    for i in top_n_model_names:
        s = 'Tuned ' + i
        tuning_model_names.append(s)
    
    #PROGRESS # 19 : Name Defined for Tuning
    progress.value += 1 
    
    #defining name counter
    name_counter = 0
    
    for i in top_n_tuned_models:
        
        '''
        MONITOR UPDATE STARTS
        '''
        monitor.iloc[2,1:] = tuning_model_names[name_counter]
        monitor.iloc[3,1:] = 'Calculating ETC'
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        #PROGRESS # 20 : Model Creation (qualifier x based on top_n_model parameter)
        progress.value += 1 
        
        model = i
    
        #setting cross validation
        kf = KFold(fold, random_state=seed)

        score_mae =np.empty((0,0))
        score_mse =np.empty((0,0))
        score_rmse =np.empty((0,0))
        score_r2 =np.empty((0,0))
        score_max_error =np.empty((0,0))
        
        avs_mae =np.empty((0,0))
        avg_mse =np.empty((0,0))
        avg_rmse =np.empty((0,0))
        avg_r2 =np.empty((0,0))
        avg_max_error =np.empty((0,0))
        
        
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
            
            
            #PROGRESS # 21 : Loop Counter (xfold)
            progress.value += 1
    
            Xtrain,Xtest = data_X.iloc[train_i], data_X.iloc[test_i]
            ytrain,ytest = data_y.iloc[train_i], data_y.iloc[test_i]
          
            model.fit(Xtrain,ytrain)
            pred_ = model.predict(Xtest)
            mae = metrics.mean_absolute_error(ytest,pred_)
            mse = metrics.mean_squared_error(ytest,pred_)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(ytest,pred_)
            max_error_ = metrics.max_error(ytest,pred_)
            score_mae = np.append(score_mae,mae)
            score_mse = np.append(score_mse,mse)
            score_rmse = np.append(score_rmse,rmse)
            score_r2 =np.append(score_r2,r2)
            score_max_error = np.append(score_max_error,max_error_)  
        
            t1 = time.time()
            
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

        mean_mae=np.mean(score_mae)
        mean_mse=np.mean(score_mse)
        mean_rmse=np.mean(score_rmse)
        mean_r2=np.mean(score_r2)
        mean_max_error=np.mean(score_max_error)
        
        avg_mae = np.append(avg_mae, mean_mae)
        avg_mse = np.append(avg_mse, mean_mse)
        avg_rmse = np.append(avg_rmse, mean_rmse)
        avg_r2 = np.append(avg_r2, mean_r2)
        avg_max_error = np.append(avg_max_error, mean_max_error)
        
        
        model_results = pd.DataFrame({'Model': tuning_model_names[name_counter], 'MAE': avg_mae, 
                                      'MSE': avg_mse, 'RMSE' : avg_rmse, 'R2' : avg_r2, 
                                      'ME' : avg_max_error})
        model_results = model_results.round(round)
        master_display = pd.concat([master_display, model_results],ignore_index=True)
        master_display = master_display.round(round)
        
        if target_metric == 'R2':
            master_display = master_display.sort_values(by=sort,ascending=False)
        else:
            master_display = master_display.sort_values(by=sort,ascending=True)
            
        master_display.reset_index(drop=True, inplace=True)
        
        update_display(master_display, display_id = display_id)
        
        name_counter += 1
        
        avg_mae = np.empty((0,0))
        avg_mse = np.empty((0,0))
        avg_rmse = np.empty((0,0))
        avg_r2 = np.empty((0,0))
        avg_max_error = np.empty((0,0))
        
        top_n_tuned_model_results = pd.concat([top_n_tuned_model_results, model_results],ignore_index=True)
        
    master_results = master_results.append(top_n_tuned_model_results)
    
    
    '''
    Step 5 - This section is for ensembling the tuned models that are tuned in the step above.
    
    '''
       
    ensemble_tuned_model_names = []
    top_n_ensemble_tuned_model_results = pd.DataFrame(columns=['Model', 'MAE','MSE','RMSE', 'R2', 'ME'])
    
    for i in top_n_model_names:
        s = 'Ensemble Tuned ' + i
        ensemble_tuned_model_names.append(s)
    
    #PROGRESS # 22 : Name Defined for Ensembled Tuned Models
    progress.value += 1 
    
    #defining name counter
    name_counter = 0
    
    top_n_ensemble_tuned_models = []
    
    for i in top_n_tuned_models:
        
        '''
        MONITOR UPDATE STARTS
        '''
        monitor.iloc[2,1:] = ensemble_tuned_model_names[name_counter]
        monitor.iloc[3,1:] = 'Calculating ETC'
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        #PROGRESS # 23 : Model Creation (qualifier x based on top_n_model parameter)
        progress.value += 1 
        
        model = BaggingRegressor(i,bootstrap=True,n_estimators=10, random_state=seed)
        top_n_ensemble_tuned_models.append(model)
            
        #setting cross validation
        kf = KFold(fold, random_state=seed)

        score_mae =np.empty((0,0))
        score_mse =np.empty((0,0))
        score_rmse =np.empty((0,0))
        score_r2 =np.empty((0,0))
        score_max_error =np.empty((0,0))
        
        avg_mae =np.empty((0,0))
        avg_mse =np.empty((0,0))
        avg_rmse =np.empty((0,0))
        avg_r2 =np.empty((0,0))
        avg_max_error =np.empty((0,0))
        
        fold_num = 1
        
        for train_i , test_i in kf.split(data_X,data_y):
            
            #PROGRESS # 24 : Loop Counter (xfold)
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
        
            model.fit(Xtrain,ytrain)
            pred_ = model.predict(Xtest)
            mae = metrics.mean_absolute_error(ytest,pred_)
            mse = metrics.mean_squared_error(ytest,pred_)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(ytest,pred_)
            max_error_ = metrics.max_error(ytest,pred_)
            score_mae = np.append(score_mae,mae)
            score_mse = np.append(score_mse,mse)
            score_rmse = np.append(score_rmse,rmse)
            score_r2 =np.append(score_r2,r2)
            score_max_error = np.append(score_max_error,max_error_)  
                    
            t1 = time.time()
            
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

        mean_mae=np.mean(score_mae)
        mean_mse=np.mean(score_mse)
        mean_rmse=np.mean(score_rmse)
        mean_r2=np.mean(score_r2)
        mean_max_error=np.mean(score_max_error)
        
        avg_mae = np.append(avg_mae, mean_mae)
        avg_mse = np.append(avg_mse, mean_mse)
        avg_rmse = np.append(avg_rmse, mean_rmse)
        avg_r2 = np.append(avg_r2, mean_r2)
        avg_max_error = np.append(avg_max_error, mean_max_error)
        
        
        model_results = pd.DataFrame({'Model': ensemble_tuned_model_names[name_counter], 'MAE': avg_mae, 
                                      'MSE': avg_mse, 'RMSE' : avg_rmse, 'R2' : avg_r2, 
                                      'ME' : avg_max_error})
        model_results = model_results.round(round)
        master_display = pd.concat([master_display, model_results],ignore_index=True)
        master_display = master_display.round(round)
        
        if target_metric == 'R2':
            master_display = master_display.sort_values(by=sort,ascending=False)
        else:
            master_display = master_display.sort_values(by=sort,ascending=True)
            
        master_display.reset_index(drop=True, inplace=True)
        
        update_display(master_display, display_id = display_id)
        
        name_counter += 1
        
        avg_mae = np.empty((0,0))
        avg_mse = np.empty((0,0))
        avg_rmse = np.empty((0,0))
        avg_r2 = np.empty((0,0))
        avg_max_error = np.empty((0,0))
        
        top_n_ensemble_tuned_model_results = pd.concat([top_n_ensemble_tuned_model_results, model_results],ignore_index=True)
        
    master_results = master_results.append(top_n_ensemble_tuned_model_results)
    master.append(top_n_ensemble_tuned_models)
    
    
    #PROGRESS # 25 : Section Completed
    progress.value += 1
    
    '''
    
    Unpacking Master into master_unpack so it can be used for sampling 
    for VotingRegressor and Stacking in Step 5 and Step 6 below. Note that
    master_unpack is not the most updated list by the end of code as the 
    models created in Step 5 and 6 below are not unpacked into master_unpack.
    Last part of code used object 'master_final' to unpack all the models from
    object 'master'.
    
    '''
                
    '''
    MONITOR UPDATE STARTS
    '''

    monitor.iloc[1,1:] = 'Unpacking Master List'
    monitor.iloc[2,1:] = 'Compiling'
    monitor.iloc[3,1:] = 'Calculating ETC'
    update_display(monitor, display_id = 'monitor')

    '''
    MONITOR UPDATE ENDS
    '''  
            
    master_unpack = []
    for i in master:
        for k in i:
            master_unpack.append(k)
    
    #PROGRESS # 26 : Master Unpacking before Section 4
    progress.value += 1
    
    '''
    
    This is the loop created for random sampling index numbers in master_unpack list
    for models that can be used in VotingRegressor in Step 5 below. Same sampling i.e.
    variable mix and mix_names is used in Stacking in Step 6 below.
    
    
    '''
    
    count_while = 0
    
    mix = []
    mix_names = []
    while count_while < top_n:
        sub_list = []
        sub_list_names = []
        generator = random.sample(range(len(master_results)-1), random.randint(3,len(master_results)-1))
        for r in generator:
            sub_list.append(master_unpack[r])
            sub_list_names.append(master_results.iloc[r]['Model'])
        mix.append(sub_list)
        mix_names.append(sub_list_names)
        count_while += 1
    
    #PROGRESS # 27 : Sampling Completed
    progress.value += 1
    
    '''

    Step 6 - Using mix and mix_names created above, build voting regressor n # of times.
    This is equivalent to blend_models()

    '''    
    
    top_n_voting_models = []
    top_n_voting_model_results = pd.DataFrame(columns=['Model', 'MAE','MSE','RMSE', 'R2', 'ME'])
    
    voting_counter = 1
    
    for i,j in zip(mix,mix_names):
        
        #PROGRESS # 28 : Model Creation (qualifier x based on top_n_model parameter)
        progress.value += 1 
        
        
        '''
        MONITOR UPDATE STARTS
        '''
        monitor.iloc[2,1:] = 'Voting Regressor # ' + str(voting_counter)
        monitor.iloc[3,1:] = 'Calculating ETC'
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        voting_counter += 1
        
        estimator_list = zip(j, i)
        estimator_list = list(estimator_list) 
        
        try:
            model = VotingRegressor(estimators=estimator_list, n_jobs=-1)
            model.fit(Xtrain,ytrain)
        except:
            model = VotingRegressor(estimators=estimator_list)
            
        top_n_voting_models.append(model)
    
        #setting cross validation
        kf = KFold(fold, random_state=seed)

        score_mae =np.empty((0,0))
        score_mse =np.empty((0,0))
        score_rmse =np.empty((0,0))
        score_r2 =np.empty((0,0))
        score_max_error =np.empty((0,0))
        
        avg_mae =np.empty((0,0))
        avg_mse =np.empty((0,0))
        avg_rmse =np.empty((0,0))
        avg_r2 =np.empty((0,0))
        avg_max_error =np.empty((0,0)) 
        
        fold_num = 1
        
        for train_i , test_i in kf.split(data_X,data_y):
            
            #PROGRESS # 29 : Loop Counter (xfold)
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
            
            model.fit(Xtrain,ytrain)
            pred_ = model.predict(Xtest)
            mae = metrics.mean_absolute_error(ytest,pred_)
            mse = metrics.mean_squared_error(ytest,pred_)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(ytest,pred_)
            max_error_ = metrics.max_error(ytest,pred_)
            score_mae = np.append(score_mae,mae)
            score_mse = np.append(score_mse,mse)
            score_rmse = np.append(score_rmse,rmse)
            score_r2 =np.append(score_r2,r2)
            score_max_error = np.append(score_max_error,max_error_)   
            
            t1 = time.time()
            
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
        
        mean_mae=np.mean(score_mae)
        mean_mse=np.mean(score_mse)
        mean_rmse=np.mean(score_rmse)
        mean_r2=np.mean(score_r2)
        mean_max_error=np.mean(score_max_error)

        avg_mae = np.append(avg_mae, mean_mae)
        avg_mse = np.append(avg_mse, mean_mse)
        avg_rmse = np.append(avg_rmse, mean_rmse)
        avg_r2 = np.append(avg_r2, mean_r2)
        avg_max_error = np.append(avg_max_error, mean_max_error)

        model_results = pd.DataFrame({'Model': 'Voting Regressor', 'MAE': avg_mae, 'MSE': avg_mse, 
                                      'RMSE' : avg_rmse, 'R2' : avg_r2 , 'ME' : avg_max_error
                                     })
        model_results = model_results.round(round)
        master_display = pd.concat([master_display, model_results],ignore_index=True)
        master_display = master_display.round(round)
        
        if target_metric == 'R2':
            master_display = master_display.sort_values(by=sort,ascending=False)
        else:
            master_display = master_display.sort_values(by=sort,ascending=True)
            
        master_display.reset_index(drop=True, inplace=True)

        update_display(master_display, display_id = display_id)
        
        avg_mae = np.empty((0,0))
        avg_mse = np.empty((0,0))
        avg_rmse = np.empty((0,0))
        avg_r2 = np.empty((0,0))
        avg_max_error = np.empty((0,0))
        
        top_n_voting_model_results = pd.concat([top_n_voting_model_results, model_results],ignore_index=True)
        
    master_results = master_results.append(top_n_voting_model_results)
    master_results = master_results.reset_index(drop=True)
    master.append(top_n_voting_models)
    
    #PROGRESS # 30 : Section Completed
    progress.value += 1
    
    '''

    Step 7 - Stacking for all the models using same sample as above that are stored in
    mix and mix_names. 
    
    This is equivalent to stack_models()

    '''    
        
    top_n_stacking_models = []
    top_n_stacking_model_results = pd.DataFrame(columns=['Model', 'MAE','MSE','RMSE', 'R2', 'ME'])
    
    meta_model = LinearRegression()
    
    #PROGRESS # 31 : Meta Model Defined for Stacking
    progress.value += 1

    stack_counter = 1
    
    for i in mix:
            
        '''
        MONITOR UPDATE STARTS
        '''
        monitor.iloc[1,1:] = 'Compiling Base Estimators'
        monitor.iloc[2,1:] = 'Stacking Regressor # ' + str(stack_counter)
        monitor.iloc[3,1:] = 'Calculating ETC'
        update_display(monitor, display_id = 'monitor')

        '''
        MONITOR UPDATE ENDS
        '''
        
        stack_counter += 1
        
        #PROGRESS # 32 : Model Creation (qualifier x based on top_n_model parameter)
        progress.value += 1
        
        estimator_list = i
        top_n_stacking_models.append(i)
        
        #defining model_library model names
        model_names = np.zeros(0)
        for item in estimator_list:
            model_names = np.append(model_names, str(item).split("(")[0])
    
        base_array = np.zeros((0,0))
        base_prediction = pd.DataFrame(y_train)
        base_prediction = base_prediction.reset_index(drop=True)
    
        for model in estimator_list:
            base_array = cross_val_predict(model,X_train,y_train,cv=fold, method='predict')
            base_array = base_array
            base_array_df = pd.DataFrame(base_array)
            base_prediction = pd.concat([base_prediction,base_array_df],axis=1)
            base_array = np.empty((0,0))
        
        #defining column names now
        target_col_name = np.array(base_prediction.columns[0])
        model_names = np.append(target_col_name, model_names)
        base_prediction.columns = model_names #defining colum names now
        data_X = base_prediction.drop(base_prediction.columns[0],axis=1)
        data_y = base_prediction[base_prediction.columns[0]]

        #Meta Modeling Starts Here

        model = meta_model 
        
        kf = KFold(fold, random_state=seed)

        score_mae =np.empty((0,0))
        score_mse =np.empty((0,0))
        score_rmse =np.empty((0,0))
        score_r2 =np.empty((0,0))
        score_max_error =np.empty((0,0))
        
        avg_mae =np.empty((0,0))
        avg_mse =np.empty((0,0))
        avg_rmse =np.empty((0,0))
        avg_r2 =np.empty((0,0))
        avg_max_error =np.empty((0,0))
        
        fold_num = 1
        
        for train_i , test_i in kf.split(data_X,data_y):
            
            #PROGRESS # 33 : Loop Counter (xfold)
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

            model.fit(Xtrain,ytrain)
            pred_ = model.predict(Xtest)
            mae = metrics.mean_absolute_error(ytest,pred_)
            mse = metrics.mean_squared_error(ytest,pred_)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(ytest,pred_)
            max_error_ = metrics.max_error(ytest,pred_)
            score_mae = np.append(score_mae,mae)
            score_mse = np.append(score_mse,mse)
            score_rmse = np.append(score_rmse,rmse)
            score_r2 =np.append(score_r2,r2)
            score_max_error = np.append(score_max_error,max_error_) 
            
            
            t1 = time.time()
            
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
            
        mean_mae=np.mean(score_mae)
        mean_mse=np.mean(score_mse)
        mean_rmse=np.mean(score_rmse)
        mean_r2=np.mean(score_r2)
        mean_max_error=np.mean(score_max_error)
        
        #std_acc=np.std(score_acc)
        #std_auc=np.std(score_auc)
        #std_recall=np.std(score_recall)
        #std_precision=np.std(score_precision)
        #std_f1=np.std(score_f1)
        #std_kappa=np.std(score_kappa)

        avg_mae = np.append(avg_mae, mean_mae)
        avg_mse = np.append(avg_mse, mean_mse)
        avg_rmse = np.append(avg_rmse, mean_rmse)
        avg_r2 = np.append(avg_r2, mean_r2)
        avg_max_error = np.append(avg_max_error, mean_max_error)

        model_results = pd.DataFrame({'Model': 'Stacking Regressor',  'MAE': avg_mae, 'MSE': avg_mse, 
                                      'RMSE' : avg_rmse, 'R2' : avg_r2 , 'ME' : avg_max_error 
                                      })
        model_results = model_results.round(round)
        master_display = pd.concat([master_display, model_results],ignore_index=True)
        master_display = master_display.round(round)
        
        if target_metric == 'R2':
            master_display = master_display.sort_values(by=sort,ascending=False)
        else:
            master_display = master_display.sort_values(by=sort,ascending=True)
            
        master_display.reset_index(drop=True, inplace=True)
        
        update_display(master_display, display_id = display_id)
        
        avg_mae = np.empty((0,0))
        avg_mse = np.empty((0,0))
        avg_rmse = np.empty((0,0))
        avg_r2 = np.empty((0,0))
        avg_max_error = np.empty((0,0))
        
        top_n_stacking_model_results = pd.concat([top_n_stacking_model_results, model_results],ignore_index=True)
        top_n_stacking_model_results = top_n_stacking_model_results.round(round)  

    master_results = master_results.append(top_n_stacking_model_results)
    master_results = master_results.reset_index(drop=True)
    master.append(top_n_stacking_models)
    
    #PROGRESS # 34 : Section Completed
    progress.value += 1
    
    '''

    Step 7 - Unpacking final master list stored in object 'master'. The one unpacked
    before step 4 was used for sampling in Step 5 and 6.
    
    THIS IS THE FINAL UNPACKING.
    
    '''
    
    
    '''
    MONITOR UPDATE STARTS
    '''
    monitor.iloc[1,1:] = 'Finalizing Results'
    monitor.iloc[2,1:] = 'Compiling Masterlist'
    monitor.iloc[3,1:] = 'Calculating ETC'
    update_display(monitor, display_id = 'monitor')

    '''
    MONITOR UPDATE ENDS
    '''

    #global master_final
    master_final = []
    for i in master:
        for k in i:
            master_final.append(k)
    
    #renaming
    master = master_final
    del(master_final) #remove master_final
    
    #PROGRESS # 35 : Final unpacking completed
    progress.value += 1
    
    '''
    
    Step 8 - This is the final step in which master_results is sorted based on defined metric
    to get the index of best model so that master can return the final best model.
    also master_results is sorted and index is reset before display.
    
    ''' 
    best_model_position = master_results.sort_values(by=sort,ascending=False).index[0]
    best_model = master[best_model_position]

    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: yellow' if v else '' for v in is_min]
    
    if target_metric == 'R2':
        master_display = master_display.sort_values(by=sort,ascending=False).reset_index(drop=True)
    else:
        master_display = master_display.sort_values(by=sort,ascending=True).reset_index(drop=True)
            
    master_display_ = master_display.style.apply(highlight_min,subset=['MAE','MSE','RMSE','ME'])
    master_display_ = master_display_.set_properties(**{'text-align': 'left'})
    master_display_ = master_display_.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    
    #PROGRESS # 36 : Final Sorting completed
    progress.value += 1
    
    #storing into experiment
    model_name = 'AutoML (best model)'
    tup = (model_name,best_model)
    experiment__.append(tup)
    
    model_name = 'AutoML Results'
    tup = (model_name,master_display)
    experiment__.append(tup)
 
    clear_output()    
    display(master_display_)
    return best_model