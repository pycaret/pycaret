# Module: Utility
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT

version_ = "2.0"
nightly_version_ = "2.1"

def version():
    return version_

def nightly_version():
    return nightly_version_

def __version__():
    return version_

def check_metric(actual, prediction, metric, round=4):
    
    """
    Function to evaluate classification and regression metrics.
    """
    
    #general dependencies
    import numpy as np
    from sklearn import metrics

    #metric calculation starts here
    
    if metric == 'Accuracy':
        
        result = metrics.accuracy_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'Recall':
        
        result = metrics.recall_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'Precision':
        
        result = metrics.precision_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'F1':
        
        result = metrics.f1_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'Kappa':
        
        result = metrics.cohen_kappa_score(actual,prediction)
        result = result.round(round)
       
    elif metric == 'AUC':
        
        result = metrics.roc_auc_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'MCC':
        
        result = metrics.matthews_corrcoef(actual,prediction)
        result = result.round(round)

    elif metric == 'MAE':

        result = metrics.mean_absolute_error(actual,prediction)
        result = result.round(round)
        
    elif metric == 'MSE':

        result = metrics.mean_squared_error(actual,prediction)
        result = result.round(round)        
        
    elif metric == 'RMSE':

        result = metrics.mean_squared_error(actual,prediction)
        result = np.sqrt(result)
        result = result.round(round)     
        
    elif metric == 'R2':

        result = metrics.r2_score(actual,prediction)
        result = result.round(round)    
        
    elif metric == 'RMSLE':

        result = np.sqrt(np.mean(np.power(np.log(np.array(abs(prediction))+1) - np.log(np.array(abs(actual))+1), 2)))
        result = result.round(round)

    elif metric == 'MAPE':

        mask = actual.iloc[:,0] != 0
        result = (np.fabs(actual.iloc[:,0] - prediction.iloc[:,0])/actual.iloc[:,0])[mask].mean()
        result = result.round(round)
       
    return float(result)


def _train_test_split(data,
                      target,
                      train_size,
                      random_sample_size = None,
                      split_type='random',
                      data_split_shuffle=True,
                      split_stratify_feature=None,
                      split_groups=None,
                      split_test_fold=None,
                      random_state=None):

    """

    Parameters
    ----------
    data : pandas.DataFrame
        Shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.

    target: string
        Name of the target column to be passed in as a string. The target variable could
        be binary or multiclass.

    train_size: float
        Size of the training set. The remaining data will be used for a test / hold-out set.

    random_sample_size: float, default=None
        Size of sample of the data to take before splitting. If split_type='random', the observations are randomly sampled.
        If split_type='group', the groups are randomly sampled.

    split_type: string, default = 'random'
        Defines the method to be used for the train/test split. By default, sklearn.model_selection.train_test_split
        is used, which splits the data into random train and test subsets. The other available options are:

        'group'         : Splits the data according to group labels using sklearn.model_selection.GroupShuffleSplit.

        'predefined'    : Splits the data according to predefined labels specified by the user.
                          Uses sklearn.model_selection.PredefinedSplit.

    data_split_shuffle: bool, default = True
        If set to False, prevents shuffling of rows when splitting data. Only used if split_type='random'.

    split_stratify_feature: str, default = None
        If None, stratifies using the target variable. If string, it is the name of the column in data to be used in
        stratification. Only used if split_type='random'.

    split_groups: string, default = None
        Name of the column in data containing the group labels to use in splitting the dataset
        into train/test set. Only used if split_type='group'.

    split_test_fold: string, default = None
            Name of the column in data containing the predefined labels specified by the user to use in splitting the dataset
            into train/test set. Only used if split_type='predfined'. If test_fold[i] = -1, observation i will be
            included in the train set; if test_fold[i] = 0, observation i will be included in the test set. No other
            values for test_fold[i] are allowed.

    random_state: int, default = None
        Controls the randomness of the training and testing indices produced.

    Returns
    -------

    """

    from sklearn.model_selection import GroupShuffleSplit, train_test_split, PredefinedSplit

    if split_type=='group':

        if random_sample_size is not None:
            gss = GroupShuffleSplit(n_splits=1, train_size=random_sample_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(data, data[split_groups]))
            data_ = data.iloc[train_idx]
        else:
            data_ = data.copy()

        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)

        train_idx, test_idx = next(gss.split(data_, data_[split_groups]))

        X_train, y_train = data_.iloc[train_idx].drop(target, axis=1), data_.iloc[train_idx].loc[:,target]
        y_train, y_test = data_.iloc[train_idx].loc[:,target], data_.iloc[test_idx].loc[:,target]

    elif split_type=='random':

        X = data.drop(target, axis=1)
        y = data[target]

        if (random_sample_size is not None) and (split_stratify_feature is not None):
            X_, X__, y_, y__ = train_test_split(X, y, train_size=random_sample_size, stratify=X[split_stratify_feature], random_state=random_state, shuffle=data_split_shuffle)
            X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=train_size, stratify=X_[split_stratify_feature], random_state=random_state, shuffle=data_split_shuffle)

        elif (random_sample_size is None) and (split_stratify_feature is not None):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=X[split_stratify_feature], random_state=random_state, shuffle=data_split_shuffle)

        elif (random_sample_size is not None) and (split_stratify_feature is None):
            X_, X__, y_, y__ = train_test_split(X, y, train_size=random_sample_size, stratify=y, random_state=random_state, shuffle=data_split_shuffle)
            X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=train_size,stratify=y_, random_state=random_state, shuffle=data_split_shuffle)

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=random_state, shuffle=data_split_shuffle)

    elif split_type=='predefined':
        ps = PredefinedSplit(data[split_test_fold])
        train_idx, test_idx = next(ps.split())

        data = data.copy().drop(split_test_fold, axis=1)

        X_train, X_test = data.iloc[train_idx].drop(target, axis=1), data.iloc[test_idx].drop(target, axis=1)
        y_train, y_test = data.iloc[train_idx].loc[:,target], data.iloc[test_idx].loc[:,target]

    return (X_train, X_test, y_train, y_test)


def enable_colab():
    
    """
    Function to render plotly visuals in colab.
    """
    
    def configure_plotly_browser_state():
        
        import IPython
        display(IPython.core.display.HTML('''
            <script src="/static/components/requirejs/require.js"></script>
            <script>
              requirejs.config({
                paths: {
                  base: '/static/base',
                  plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
                },
              });
            </script>
            '''))
  
    import IPython
    IPython.get_ipython().events.register('pre_run_cell', configure_plotly_browser_state)
    print('Colab mode activated.')
