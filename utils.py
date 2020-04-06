# Module: Utility
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT

def version():
    print("1.0.0")


def check_metric(actual, prediction, metric, round=4):
    
    """
    reserved for docstring
    """
    
    #general dependencies
    import numpy as np

    #metric calculation starts here
    
    if metric == 'accuracy':
        
        from sklearn import metrics
        result = metrics.accuracy_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'recall':
        
        from sklearn import metrics
        result = metrics.recall_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'precision':
        
        from sklearn import metrics
        result = metrics.precision_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'f1':
        
        from sklearn import metrics
        result = metrics.f1_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'kappa':
        
        from sklearn import metrics
        result = metrics.cohen_kappa_score(actual,prediction)
        result = result.round(round)
       
    elif metric == 'auc':
        
        from sklearn import metrics
        result = metrics.roc_auc_score(actual,prediction)
        result = result.round(round)
        
    elif metric == 'mae':

        from sklearn import metrics
        result = metrics.mean_absolute_error(actual,prediction)
        result = result.round(round)
        
    elif metric == 'mse':

        from sklearn import metrics
        result = metrics.mean_squared_error(actual,prediction)
        result = result.round(round)        
        
    elif metric == 'rmse':

        from sklearn import metrics
        result = metrics.mean_squared_error(actual,prediction)
        result = np.sqrt(result)
        result = result.round(round)     
        
    elif metric == 'r2':

        from sklearn import metrics
        result = metrics.r2_score(actual,prediction)
        result = result.round(round)    
        
    elif metric == 'rmsle':

        result = np.sqrt(np.mean(np.power(np.log(np.array(abs(prediction))+1) - np.log(np.array(abs(actual))+1), 2)))
        result = result.round(round)

    elif metric == 'mape':

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