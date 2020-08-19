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
