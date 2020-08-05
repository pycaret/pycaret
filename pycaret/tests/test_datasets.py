import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pytest
import pycaret.datasets

def test():
    # loading list of datasets
    data = pycaret.datasets.get_data('index')

    # loading dataset
    credit = pycaret.datasets.get_data('credit')

    assert 1 == 1
    
if __name__ == "__main__":
    test()
