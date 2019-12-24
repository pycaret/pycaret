def get_data(dataset):
    
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
        
    
    Returns:
    --------

    DataFrame:    Pandas dataframe is returned. 
    ----------      

    Warnings:
    ---------
    Use of get_date() requires internet connection.
    
  
       
    """
    
    import pandas as pd
    
    address = 'https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/'
    extension = '.csv'
    filename = dataset
    
    complete_address = address + str(dataset) + extension
    
    data = pd.read_csv(complete_address)
    
    if dataset == 'index':
        display(data)
    
    else:
        display(data.head())
        return data