# Module: Datasets
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT


def get_data(
    dataset: str = "index",
    folder: str = "common",
    index: bool = False,
    save_copy: bool = False,
    profile=False,
    verbose=True,
    address="https://raw.githubusercontent.com/pycaret/datasets/main/",
):

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

    address: string, default = "https://raw.githubusercontent.com/pycaret/datasets/main/"
        Download url of dataset. For people have difficulty linking to github, they can change
        the default address to their own (e.g. "https://gitee.com/IncubatorShokuhou/pycaret/raw/master/datasets/")


    Returns:
        pandas.DataFrame


    Warnings
    --------
    - Use of ``get_data`` requires internet connection.


    Raises
    ------
    ImportError
        When trying to import time series datasets that require sktime,
        but sktime has not been installed.
    """

    import pandas as pd
    import os.path
    from IPython.display import display

    root = address
    data_dir = "data/"
    meta_dir = "meta/"

    extension = ".csv"

    if index:
        filename = "index.csv"
        complete_address = root + meta_dir + folder + "/" + filename
    else:
        filename = str(dataset) + extension
        complete_address = root + data_dir + folder + "/" + filename

    sktime_datasets = ["airline", "lynx", "uschange"]

    if os.path.isfile(filename) and not index:
        data = pd.read_csv(filename)
    elif dataset in sktime_datasets:
        try:
            from sktime.datasets import load_airline, load_lynx, load_uschange
        except ImportError as e:
            print(e)
            raise ImportError(
                f"Dataset '{dataset}' is meant for time series analysis and needs"
                " the sktime library to be installed."
            )

        ts_dataset_mapping = {
            "airline": load_airline,
            "lynx": load_lynx,
            "uschange": load_uschange,
        }
        data = ts_dataset_mapping.get(dataset)()
        if isinstance(data, tuple):
            data = data[0]  # ignoring X for now (univariate only)
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
