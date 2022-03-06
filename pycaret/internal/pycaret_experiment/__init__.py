from .anomaly_experiment import AnomalyExperiment
from .clustering_experiment import ClusteringExperiment

from .classification_experiment import ClassificationExperiment
from .regression_experiment import RegressionExperiment

# Already refactored, hence not needed here.
# from pycaret.time_series import TSForecastingExperiment

# from .utils import MLUsecase

################################################################################
#### NOTE: experiment_factory is not being used anymore but was causing     ####
#### circular import issues. Hence removing.                                ####
################################################################################
# def experiment_factory(usecase: MLUsecase):
#     switch = {
#         MLUsecase.CLASSIFICATION: ClassificationExperiment,
#         MLUsecase.REGRESSION: RegressionExperiment,
#         MLUsecase.CLUSTERING: ClusteringExperiment,
#         MLUsecase.ANOMALY: AnomalyExperiment,
#         MLUsecase.TIME_SERIES: TSForecastingExperiment,
#     }
#     return switch[usecase]()
