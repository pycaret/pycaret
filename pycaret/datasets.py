"""Module to get datasets in pycaret
"""
from typing import Optional

import requests

from pycaret.utils._dependencies import _check_soft_dependencies


def get_data(
    dataset: str = "index",
    folder: Optional[str] = None,
    save_copy: bool = False,
    profile: bool = False,
    verbose: bool = True,
    address: Optional[str] = None,
):

    """
    Function to load sample datasets.

    Order of read:
    (1) Tries to read dataset from local folder first.
    (2) Then tries to read dataset from folder in GitHub "address" (see below)
    (3) Then tries to read from sktime (if installed)
    (4) Raises error if none exist

    List of available datasets on GitHub can be checked using
    (1) ``get_data('index')`` or
    (2) ``get_data('index', folder='time_series/seasonal)``
    (see available "folder" options below)


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> all_datasets = get_data('index')
    >>> juice = get_data('juice')


    dataset: str, default = 'index'
        Index value of dataset.


    folder: Optional[str], default = None
        The folder from which to get the data.
        If 'None', gets it from the "common" folder. Other options are:
            - time_series/seasonal
            - time_series/random_walk
            - time_series/white_noise


    save_copy: bool, default = False
        When set to true, it saves a copy in current working directory.


    profile: bool, default = False
        When set to true, an interactive EDA report is displayed.


    verbose: bool, default = True
        When set to False, head of data is not displayed.

    address: string, default = None
        Download url of dataset. Defaults to None which fetches the dataset from
        "https://raw.githubusercontent.com/pycaret/datasets/main/". For people
        having difficulty linking to github, they can change the default address
        to their own
        (e.g. "https://gitee.com/IncubatorShokuhou/pycaret/raw/master/datasets/")


    Returns:
        pandas.DataFrame


    Warnings
    --------
    - Use of ``get_data`` requires internet connection.


    Raises
    ------
    ImportError
        (1) When trying to import time series datasets that require sktime,
        but sktime has not been installed.
        (2) If the data does not exist
    """

    import os.path

    import pandas as pd

    from pycaret.internal.display import CommonDisplay

    extension = ".csv"
    filename = str(dataset) + extension
    if address is None:
        root = "https://raw.githubusercontent.com/pycaret/datasets/main/"
        data_dir, meta_dir = "data/", "meta/"

        folder = "common" if folder is None else folder

        if dataset == "index":
            complete_address = root + meta_dir + folder + "/" + filename
        else:
            complete_address = root + data_dir + folder + "/" + filename
    else:
        complete_address = address + "/" + filename

    sktime_datasets = ["airline", "lynx", "uschange"]

    # Read the file name from local folder first
    # If it does not exist, then read the file from GitHub
    # If that does not exist then read sktime datasets
    if os.path.isfile(filename):
        data = pd.read_csv(filename)
    elif requests.get(complete_address).status_code == 200:
        data = pd.read_csv(complete_address)
    elif dataset in sktime_datasets:
        from sktime.datasets import load_airline, load_lynx, load_uschange

        ts_dataset_mapping = {
            "airline": load_airline,
            "lynx": load_lynx,
            "uschange": load_uschange,
        }
        data = ts_dataset_mapping.get(dataset)()
        if isinstance(data, tuple):
            y = data[0]
            X = data[1]
            data = pd.concat([y, X], axis=1)
    else:
        raise ValueError("Data could not be read. Please check your inputs...")

    # create a copy for pandas profiler
    data_for_profiling = data.copy()

    if save_copy:
        save_name = filename
        data.to_csv(save_name, index=False)

    display = CommonDisplay(
        verbose=True,
        html_param=True,
    )

    if dataset == "index":
        display.display(data)

    else:
        if profile:
            _check_soft_dependencies(
                "pandas_profiling",
                extra="analysis",
                severity="error",
                install_name="pandas-profiling",
            )
            import pandas_profiling

            pf = pandas_profiling.ProfileReport(data_for_profiling)
            display.display(pf)

        else:
            if verbose:
                display.display(data.head())

    return data
