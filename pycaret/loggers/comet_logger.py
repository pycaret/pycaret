from copy import deepcopy
from pathlib import Path

import joblib

from pycaret.loggers.base_logger import BaseLogger


class CometLogger(BaseLogger):
    def __init__(self) -> None:
        # lazy import to avoid comet logging
        try:
            import comet_ml
        except ImportError:
            comet_ml = None

        if comet_ml is None:
            raise ImportError(
                "CometLogger requires Comet. Install using `pip install comet_ml`"
            )
        super().__init__()
        self.run = None

    def init_experiment(self, exp_name_log, full_name=None, setup=True, **kwargs):
        import comet_ml

        self.run = comet_ml.Experiment(project_name=exp_name_log, **kwargs)
        self.run.set_name(full_name)
        self.run.log_other("Created from", "pycaret")
        return self.run

    def log_params(self, params, model_name=None):
        self.run.log_parameters(params, prefix=model_name)

    def set_tags(self, source, experiment_custom_tags, runtime, USI=None):
        tags = [source, runtime]
        self.run.add_tags(tags)
        if experiment_custom_tags:
            self.run.log_others(experiment_custom_tags)

    def log_sklearn_pipeline(self, experiment, prep_pipe, model, path=None):
        pipeline = self._construct_pipeline_if_needed(model, prep_pipe)
        joblib.dump(pipeline, "pipeline.pkl")
        self.run.log_model(name="model", file_or_folder="pipeline.pkl")

    def log_model_comparison(self, model_result, source):
        result_copy = deepcopy(model_result)
        if "Object" in result_copy:
            result_copy["Object"] = result_copy["Object"].apply(
                lambda obj: str(type(obj).__name__)
            )
        self.run.log_table("compare.csv", result_copy)

    def log_metrics(self, metrics, source=None):
        self.run.log_metrics(metrics)

    def log_plot(self, plot, title):
        self.run.log_figure(figure=plot, figure_name=title)

    def log_hpram_grid(self, html_file, title="hpram_grid"):
        self.run.log_html(html_file)

    def log_artifact(self, file, type="artifact"):
        _, extension = None, ""
        file_pathlib = Path(file)
        extension = file_pathlib.suffix

        if extension == "html":
            self.run.log_html(file)
        elif extension == "csv":
            self.run.log_table(file)
        else:
            self.run.log_asset(file)

    def finish_experiment(self):
        if self.run is None:
            return
        self.run.end()
