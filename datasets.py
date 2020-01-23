# Module: Datasets
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT


def get_data(dataset, save_copy=False, profile = False):
    
    """
      
    Description:
    ------------
    This function loads the sample dataset available at pycaret git repository. To view
    the full list of available datasets and their description, index can be called.

        Example
        -------
        data = get_data('index')

        This will display the list of available datasets that can be loaded using 
        get_data() function. For example to load credit dataset:
        
        credit_data = get_data('credit')
        
    Parameters
    ----------
    
    dataset : string 
    index value of dataset
    
    save_copy : bool, default = False
    When set to true, it saves the copy of dataset in your local active directory.
    
    profile: bool, default = False
    If set to true, it will display data profile for Exploratory Data Analysis in 
    interactive HTML report. 
    
    
    Returns:
    --------

    DataFrame:    Pandas dataframe is returned. 
    ----------      

    Warnings:
    ---------
    - Use of get_data() requires internet connection.
    
  
       
    """
    
    import pandas as pd
    
    address = 'https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/'
    extension = '.csv'
    filename = dataset
    
    complete_address = address + str(dataset) + extension
    
    data = pd.read_csv(complete_address)
    
    if save_copy:
        save_name = str(dataset) + str(extension)
        data.to_csv(save_name)
        
    if dataset == 'index':
        display(data)
    
    else:
        if profile:
            import pandas_profiling
            pf = pandas_profiling.ProfileReport(data)
            display(pf)
            
        else:
            display(data.head())
        
        return data