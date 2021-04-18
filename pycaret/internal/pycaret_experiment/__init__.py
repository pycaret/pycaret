from .anomaly_experiment import AnomalyExperiment
from .clustering_experiment import ClusteringExperiment

from .classification_experiment import ClassificationExperiment
from .regression_experiment import RegressionExperiment
from .time_series_experiment import TimeSeriesExperiment

from .utils import MLUsecase


def experiment_factory(usecase: MLUsecase):
    switch = {
        MLUsecase.CLASSIFICATION: ClassificationExperiment,
        MLUsecase.REGRESSION: RegressionExperiment,
        MLUsecase.CLUSTERING: ClusteringExperiment,
        MLUsecase.ANOMALY: AnomalyExperiment,
        MLUsecase.TIME_SERIES: TimeSeriesExperiment,
    }
    return switch[usecase]()
