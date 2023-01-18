from pycaret.loggers.base_logger import BaseLogger
from pathlib import Path
import uuid
from copy import deepcopy
import joblib

try:
    import comet_ml
except ImportError:
    comet_ml = None

class CometLogger(BaseLogger):
    def __init__(self) -> None:
        if comet_ml is None:
            raise ImportError(
                "CometLogger requires Comet. Install using `pip install comet_ml`"
            )
        super().__init__()
        self.run = None
    
    def init_experiment(self, exp_name_log, full_name=None, **kwargs):
        self.run = comet_ml.Experiment(project_name=exp_name_log, **kwargs)
        self.run.set_name(full_name)
        self.run.log_other('Created from', 'pycaret')
        return self.run
    
    def log_params(self, params, model_name=None):
        self.run.log_parameters(params)
    
    def set_tags(self, source, experiment_custom_tags, runtime):
        tags = [source, experiment_custom_tags, runtime]
        self.run.add_tags(tags)

    def log_sklearn_pipeline(self, experiment, prep_pipe, model, path=None):
        
        pipeline = deepcopy(prep_pipe)
        pipeline.steps.append(["trained_model", model])
        joblib.dump(pipeline, 'pipeline.pkl')
        self.run.log_model(name='model', file_or_folder='pipeline.pkl')

    def log_model_comparison(self, model_result, source):
        result_copy = deepcopy(model_result)
        if "Object" in result_copy:
            result_copy["Object"] = result_copy["Object"].apply(
                lambda obj: str(type(obj).__name__)
            )
        self.run.log_metrics({source: result_copy})

    def log_metrics(self, metrics, source=None):
        self.run.log_metrics(metrics)

    def log_plot(self, plot, title):
        self.run.log_figure(figure=plot, figure_name=title)

    def log_hpram_grid(self, html_file, title="hpram_grid"):
        self.run.log_html(html_file)

    def log_artifact(self, file, type="artifact"):
        file_name, extension = None, ""
        file_pathlib = Path(file)
        file_name = file_pathlib.stem.replace(" ", "_") + str(uuid.uuid1())[:8]
        artifact = comet_ml.Artifact(name=file_name, artifact_type=type)
        artifact.add(file)
        self.run.log_artifact(artifact)

    def finish_experiment(self):
        if self.run:
            self.run.end()