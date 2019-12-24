def get_data(data):
    
    """
      
    Description:
    ------------
    This function loads the sample dataset available at pycaret git repository.

        Example
        -------
        data = get_data(data, 'target')

        data is a pandas DataFrame and 'target' is the name of the column in dataframe.
        
        
    Available datasets
    ------------------
    
    data        type              target         shape
    -------     -------           ------         -----    
    juice       classification    Purchase       1070 x 15
    credit      classification    default        24000 x 24
    cancer      classification    Class          683 x 10
    iris        classification    Class          100 x 5
    gold        regression        Gold_T+22      2558 x 121
    boston      regression        medv           506 x 14
    bike        regression        cnt            17379 x 15
    diamond     regression        Price          6000 x 8
    kiva        NLP / classf.     en / status    6818 x 7

    
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
    filename = data
    
    complete_address = address + str(data) + extension
    
    data = pd.read_csv(complete_address)
    
    display(data.head())
    
    return data