import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pytest
import pycaret.forecast 
import pycaret.datasets

def test_compare_models():
    data = pycaret.datasets.get_data('air_passangers')
    data = data.tail(100)
    foc1 = pycaret.forecast.setup(data, target='#Passengers',silent=True, session_id=786)
    models = pycaret.forecast.auto_select(splits=2, verbose=True)
    results = len(models)
    assert results == 2