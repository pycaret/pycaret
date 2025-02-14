import pandas as pd
import polars as pl
import pytest

from pycaret.datasets import get_data
from pycaret.internal.preprocess import Preprocessor


def test_datasets():
    #########################
    # Load Local File ####
    #########################

    # # loading dataset
    # os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # data = get_data("test_files/dummy_dataset")
    # assert isinstance(data, pd.DataFrame)
    # rows, cols = data.shape
    # assert rows >= 1
    # assert cols >= 1

    ##############################
    # GitHub Common folder ####
    ##############################

    # loading list of datasets
    index = get_data("index")
    assert isinstance(index, pd.DataFrame)
    rows, cols = index.shape
    assert rows > 1
    assert cols == 8

    # loading dataset
    data = get_data("credit")
    assert isinstance(data, pd.DataFrame)
    rows, cols = data.shape
    assert rows == 24000
    assert cols == 24
    assert data.size == 576000

    ################################
    # Conversion from Polars to Pandas ####
    ################################

    # creating a Polars DataFrame manually
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        }
    )

    # testing conversion
    preprocessor = Preprocessor()
    data_converted = preprocessor._prepare_dataset(df)
    assert isinstance(data_converted, pd.DataFrame)
    rows, cols = data_converted.shape
    assert rows == 5
    assert cols == 4

    ################################
    # GitHub Specific folder ####
    ################################

    folder = "time_series/seasonal"
    # loading list of datasets
    index = get_data("index", folder=folder)
    assert isinstance(index, pd.DataFrame)
    rows, cols = index.shape
    assert rows > 1
    assert cols == 12

    # loading dataset
    data = get_data("1", folder=folder)
    assert isinstance(data, pd.DataFrame)
    rows, cols = data.shape
    assert rows >= 1
    assert cols >= 1

    ###########################
    # `sktime` datasets ####
    ###########################

    # loading dataset
    data = get_data("airline")
    assert isinstance(data, pd.Series)
    rows = len(data)
    assert rows >= 1

    ###########################
    # Incorrect dataset ####
    ###########################

    with pytest.raises(ValueError) as errmsg:
        _ = get_data("wrong")

    exceptionmsg = errmsg.value.args[0]
    assert exceptionmsg == "Data could not be read. Please check your inputs..."


if __name__ == "__main__":
    test_datasets()
