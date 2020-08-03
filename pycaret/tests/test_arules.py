import os, sys
sys.path.insert(0, os.path.abspath(".."))

#compare_models_test
import pytest
import pycaret.arules
import pycaret.datasets

def test():
    # loading dataset
    data = pycaret.datasets.get_data("france")

    # init setup
    arul101 = pycaret.arules.setup(data = data, 
                    transaction_id = "InvoiceNo",
                    item_id = "Description", session_id=123)

    # create model
    model = pycaret.arules.create_model()

    assert 1 == 1

if __name__ == "__main__":
    test()
