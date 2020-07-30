# Module: Utility
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT

version_ = "pycaret-nightly-0.40"

def version():
    print(version_)

def __version__():
    return version_

def check_metric(actual, prediction, metric, round=4):
    
    """
    reserved for docstring
    """
    
    #general dependencies
    import numpy as np

    #metric calculation starts here
    
    if metric == 'Accuracy':
        
        from sklearn import metrics
        result = metrics.accuracy_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'Recall':
        
        from sklearn import metrics
        result = metrics.recall_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'Precision':
        
        from sklearn import metrics
        result = metrics.precision_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'F1':
        
        from sklearn import metrics
        result = metrics.f1_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'Kappa':
        
        from sklearn import metrics
        result = metrics.cohen_kappa_score(actual,prediction)
        result = result.round(round)
       
    elif metric == 'AUC':
        
        from sklearn import metrics
        result = metrics.roc_auc_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'MCC':
        
        from sklearn import metrics
        result = metrics.matthews_corrcoef(actual,prediction)
        result = result.round(round)

    elif metric == 'MAE':

        from sklearn import metrics
        result = metrics.mean_absolute_error(actual,prediction)
        result = result.round(round)
        
    elif metric == 'MSE':

        from sklearn import metrics
        result = metrics.mean_squared_error(actual,prediction)
        result = result.round(round)        
        
    elif metric == 'RMSE':

        from sklearn import metrics
        result = metrics.mean_squared_error(actual,prediction)
        result = np.sqrt(result)
        result = result.round(round)     
        
    elif metric == 'R2':

        from sklearn import metrics
        result = metrics.r2_score(actual,prediction)
        result = result.round(round)    
        
    elif metric == 'RMSLE':

        result = np.sqrt(np.mean(np.power(np.log(np.array(abs(prediction))+1) - np.log(np.array(abs(actual))+1), 2)))
        result = result.round(round)

    elif metric == 'MAPE':

        mask = actual != 0
        result = (np.fabs(actual - prediction)/actual)[mask].mean()
        result = result.round(round)
       
    return result


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