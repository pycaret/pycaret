import warnings
from typing import List, Tuple

import pycaret.containers.metrics.anomaly
import pycaret.containers.models.anomaly
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
import pycaret.internal.persistence
import pycaret.internal.preprocess
from pycaret.internal.logging import get_logger
from pycaret.internal.pycaret_experiment.unsupervised_experiment import (
    _UnsupervisedExperiment,
)
from pycaret.internal.pycaret_experiment.utils import MLUsecase

warnings.filterwarnings("ignore")
LOGGER = get_logger()


class AnomalyExperiment(_UnsupervisedExperiment):
    def __init__(self):
        super().__init__()
        self._ml_usecase = MLUsecase.ANOMALY
        self.exp_name_log = "anomaly-default-name"
        self._available_plots = {
            "tsne": "t-SNE (3d) Dimension Plot",
            "umap": "UMAP Dimensionality Plot",
        }

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in pycaret.containers.models.anomaly.get_all_model_containers(
                self, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = (
            pycaret.containers.models.anomaly.get_all_model_containers(
                self, raise_errors=raise_errors
            )
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return pycaret.containers.metrics.anomaly.get_all_metric_containers(
            self.variables, raise_errors=raise_errors
        )

    def _get_default_plots_to_log(self) -> List[str]:
        return ["tsne"]
