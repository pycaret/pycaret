import os
from copy import deepcopy
from ctypes.wintypes import PPOINT
from xml.etree.ElementTree import PI
from pycaret.loggers import BaseLogger
import wandb
import pandas as pd
from pickle import dumps

class wandbLogger(BaseLogger):
    def __init__(self) -> None:
        super().__init__()
        self.run = None
    
    def init_experiment(self, exp_name_log, full_name=None, **kwargs):
        self.run = wandb.init(
            project=exp_name_log,
            name=full_name,
            **kwargs
        ) if not wandb.run else wandb.run

        return self.run
    
    def log_params(self, params, model_name=None):
        if model_name:
            params = {model_name: params}
        self.run.config.update(params)

    def log_metrics(self, metrics):
        self.run.log(metrics)
    
    def log_artifact(self, file, type=None):
        file_name, extension = file.split('.')
        art = wandb.Artifact(name=file_name.replace(" ", "_"), type=type or "exp_data")
        art.add_file(file)
        self.run.log_artifact(art)

        if extension=="html":
            self.run.log({file_name: wandb.Html(file)})
        elif extension=="csv":
            self.run.log({file_name: pd.read_csv(file)})


    def log_sklearn_pipeline(self, prep_pipe, model):
        pipeline = deepcopy(prep_pipe)
        pipeline.steps.append(["trained_model", model])
        dumps(pipeline, open("pipline.pkl","wb"))

        art = wandb.Artifact("pipeline", type="model")
        art.add_file("pipeline.pkl")
        self.run.log_artifact(art)
        os.remove("pipeline.pkl")
    
    def log_model_comparison(self, model_result):
        if "Object" in model_result:
            model_result = model_result.drop(columns=["Object"])
        self.run.log({"compare_models": model_result})
