import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.arules
import pycaret.datasets


def test():
    # loading dataset
    data = pycaret.datasets.get_data("france")
    assert isinstance(data, pd.core.frame.DataFrame)
    row, col = data.shape
    assert row == 8557
    assert col == 8

    # init setup
    arul101 = pycaret.arules.setup(
        data=data,
        transaction_id="InvoiceNo",
        item_id="Description",
        session_id=123,
    )
    assert isinstance(arul101, tuple)
    assert isinstance(arul101[0], pd.core.frame.DataFrame)
    row, col = arul101[0].shape
    assert row == 8557
    assert col == 8
    assert isinstance(arul101[1], str)
    assert isinstance(arul101[2], str)
    assert isinstance(arul101[4], int)
    assert isinstance(arul101[5], list)

    # create model
    model = pycaret.arules.create_model()
    assert isinstance(model, pd.core.frame.DataFrame)
    row, col = model.shape
    assert row == 141
    assert col == 9

    # get rules
    rules = pycaret.arules.get_rules(
        data=data, transaction_id="InvoiceNo", item_id="Description"
    )
    assert isinstance(rules, pd.core.frame.DataFrame)
    row, col = rules.shape
    assert row == 141
    assert col == 9

    assert 1 == 1


if __name__ == "__main__":
    test()
