#compare_models_test
import pytest

def test_compare_models():
    from pycaret.datasets import get_data
    data = get_data('boston')
    from pycaret.regression import *
    reg1 = setup(data, target='medv',silent=True, html=False, session_id=123)
    models = compare_models(n_select=3)
    top_3 = len(models)
    assert top_3 == 3