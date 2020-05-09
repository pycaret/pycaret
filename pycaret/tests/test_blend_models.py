import os, sys
sys.path.insert(0, os.path.abspath(".."))

#blend_models_test
import pytest
import pycaret.regression
import pycaret.datasets

def test_blend_models():
    data = pycaret.datasets.get_data('boston')
    data = data.head(50)
    reg1 = pycaret.regression.setup(data, target='medv',silent=True, html=False, session_id=123)
    estimator_list = pycaret.regression.compare_models(n_select=5, verbose=False) #select top 5
    blender = pycaret.regression.blend_models(estimator_list=estimator_list, improve_only=False, verbose=False)
    assert hasattr(blender, 'predict')