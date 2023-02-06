import os
import uuid
from copy import deepcopy
from pathlib import Path

import pandas as pd

from pycaret.loggers.base_logger import BaseLogger

try:
    import wandb
except ImportError:
    wandb = None


class WandbLogger(BaseLogger):
    def __init__(self) -> None:
        if wandb is None:
            raise ImportError(
                "WandbLogger requires wandb. Install using `pip install wandb`"
            )
        super().__init__()
        self.run = None

    def init_experiment(self, exp_name_log, full_name=None, **kwargs):
        self.run = (
            wandb.init(project=exp_name_log, name=full_name, **kwargs)
            if not wandb.run
            else wandb.run
        )

        return self.run

    def log_params(self, params, model_name=None):
        if model_name:
            params = {model_name: params}
        self.run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics, source=None):
        if source:
            prefixed_metrics = {}
            for metric in metrics:
                prefixed_metrics[source + "/" + metric] = metrics[metric]
            metrics = prefixed_metrics
        self.run.log(metrics)

    def log_artifact(self, file, type=None):
        file_name, extension = None, ""
        file_pathlib = Path(file)
        file_name = file_pathlib.stem.replace(" ", "_") + str(uuid.uuid1())[:8]
        extension = file_pathlib.suffix
        art = wandb.Artifact(name=file_name, type=type or "exp_data")
        art.add_file(file)
        self.run.log_artifact(art)

        if extension == "html":
            self.run.log({file_name: wandb.Html(file)})
        elif extension == "csv":
            self.run.log({file_name: pd.read_csv(file)})

    def log_sklearn_pipeline(self, experiment, prep_pipe, model, path=None):
        path = path or ""
        pipeline = deepcopy(prep_pipe)
        pipeline.steps.append(["trained_model", model])
        art = wandb.Artifact("pipeline", type="model")
        with art.new_file(os.path.join(path, "pipeline.pkl")) as f:
            f.write(pipeline)
        self.run.log_artifact(art)

    def log_model_comparison(self, model_result, source):
        result_copy = deepcopy(model_result)
        if "Object" in result_copy:
            result_copy["Object"] = result_copy["Object"].apply(
                lambda obj: str(type(obj).__name__)
            )
        self.run.log({source: result_copy})

    def log_plot(self, plot, title):
        self.run.log({title: wandb.Image(plot)})

    def log_hpram_grid(self, html_file, title="hpram_grid"):
        self.run.log({title: wandb.Html(html_file)})

    def finish_experiment(self):
        if self.run:
            self.run.finish()
