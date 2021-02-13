# Module: Datasets
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT


def get_data(dataset="index", save_copy=False, profile=False, verbose=True):

    """
    This function loads sample datasets from git repository. List of available
    datasets can be checked using ``get_data('index')``.
    

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> all_datasets = get_data('index')
    >>> juice = get_data('juice')

        
    dataset: str, default = 'index'
        Index value of dataset.
    

    save_copy: bool, default = False
        When set to true, it saves a copy in current working directory.
    

    profile: bool, default = False
        When set to true, an interactive EDA report is displayed. 


    verbose: bool, default = True
        When set to False, head of data is not displayed.


    Returns:
        pandas.DataFrame
        

    Warnings
    --------
    - Use of ``get_data`` requires internet connection.
         
    """

    import pandas as pd
    import os.path
    from IPython.display import display, HTML, clear_output, update_display

    address = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/"
    extension = ".csv"
    filename = str(dataset) + extension

    complete_address = address + filename

    if os.path.isfile(filename):
        data = pd.read_csv(filename)
    else:
        data = pd.read_csv(complete_address)

    # create a copy for pandas profiler
    data_for_profiling = data.copy()

    if save_copy:
        save_name = filename
        data.to_csv(save_name, index=False)

    if dataset == "index":
        display(data)

    else:
        if profile:
            import pandas_profiling

            pf = pandas_profiling.ProfileReport(data_for_profiling)
            display(pf)

        else:
            if verbose:
                display(data.head())

    return data
