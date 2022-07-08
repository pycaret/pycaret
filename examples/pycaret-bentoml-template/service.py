import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import bentoml
import numpy as np
import pandas as pd
from conf.config_core import config
from bentoml.io import NumpyNdarray, JSON
from pycaret.regression import load_model, predict_model


class PyCaretRunnable(bentoml.Runnable):
    """https://docs.bentoml.org/en/latest/concepts/runner.html"""
    SUPPORTS_CPU_MULTI_THREADING = True
    
    def __init__(self):
        self.pipeline = load_model(config.pipeline_config.path, verbose=False)
        
    
    @bentoml.Runnable.method(batchable=True)
    def run_model(self, sample: np.ndarray):
        """
        This method will take a real world sample and then
        estimate something.
        """
        data = pd.DataFrame([sample])
        data.columns = config.pipeline_config.features
        
        predictions = predict_model(self.pipeline, data=data)
        predictions.Label = np.exp(predictions.Label)
        
        return {"predictions": list(predictions.Label)}
    

# Initiating Runner
pycare_pipeline_runnable = bentoml.Runner(PyCaretRunnable)


# Setting Up the Service
svc = bentoml.Service('pycaret_estimator_bentoml_service', runners=[pycare_pipeline_runnable])

@svc.api(input=NumpyNdarray(), output=JSON())
def predict(sample: np.ndarray):
    """
    This function takes the input values and then
    run the loaded pycaret model.
    """
    predictions = pycare_pipeline_runnable.run_model.run(sample)
    return predictions