# Module: containers.models.clustering
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# The purpose of this module is to serve as a central repository of clustering models. The `clustering` module will
# call `get_all_model_containers()`, which will return instances of all classes in this module that have `ClassifierContainer`
# as a base (but not `ClassifierContainer` itself). In order to add a new model, you only need to create a new class that has
# `ClassifierContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

from typing import Any, Dict, Optional

import numpy as np

import pycaret.containers.base_container
import pycaret.internal.cuml_wrappers
from pycaret.containers.models.base_model import ModelContainer
from pycaret.internal.cuml_wrappers import get_dbscan, get_kmeans
from pycaret.internal.distributions import Distribution
from pycaret.utils._dependencies import _check_soft_dependencies
from pycaret.utils.generic import get_logger, param_grid_to_lists

_DEFAULT_N_CLUSTERS = 4


class ClusterContainer(ModelContainer):
    """
    Base clustering model container class, for easier definition of containers. Ensures consistent format
    before being turned into a dataframe row.

    Parameters
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    eq_function : type, default = None
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. If None, will be ``is_instance(x, class_def)`` where x is the object.
    args : dict, default = {} (empty dict)
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool, default = False
        Is the model special (not intended to be used on its own, eg. VotingClassifier).
    tune_grid : dict of str : list, default = {} (empty dict)
        The hyperparameters tuning grid for random and grid search.
    tune_distribution : dict of str : Distribution, default = {} (empty dict)
        The hyperparameters tuning grid for other types of searches.
    tune_args : dict, default = {} (empty dict)
        The arguments to always pass to the tuner.
    is_gpu_enabled : bool, default = None
        If None, will try to automatically determine.

    Attributes
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    eq_function : type
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. If None, will be ``is_instance(x, class_def)`` where x is the object.
    args : dict
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool
        Is the model special (not intended to be used on its own, eg. VotingClassifier).
    tune_grid : dict of str : list
        The hyperparameters tuning grid for random and grid search.
    tune_distribution : dict of str : Distribution
        The hyperparameters tuning grid for other types of searches.
    tune_args : dict
        The arguments to always pass to the tuner.
    is_gpu_enabled : bool
        If None, will try to automatically determine.

    """

    def __init__(
        self,
        id: str,
        name: str,
        class_def: type,
        eq_function: Optional[type] = None,
        args: Dict[str, Any] = None,
        is_special: bool = False,
        tune_grid: Dict[str, list] = None,
        tune_distribution: Dict[str, Distribution] = None,
        tune_args: Dict[str, Any] = None,
        is_gpu_enabled: Optional[bool] = None,
    ) -> None:

        if not args:
            args = {}

        if not tune_grid:
            tune_grid = {}

        if not tune_distribution:
            tune_distribution = {}

        if not tune_args:
            tune_args = {}

        super().__init__(
            id=id,
            name=name,
            class_def=class_def,
            eq_function=eq_function,
            args=args,
            is_special=is_special,
        )
        self.tune_grid = param_grid_to_lists(tune_grid)
        self.tune_distribution = tune_distribution
        self.tune_args = tune_args

        if is_gpu_enabled is not None:
            self.is_gpu_enabled = is_gpu_enabled
        else:
            self.is_gpu_enabled = bool(self.get_package_name() == "cuml")

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:
        """
        Returns a dictionary of the model properties, to
        be turned into a pandas DataFrame row.

        Parameters
        ----------
        internal : bool, default = True
            If True, will return all properties. If False, will only
            return properties intended for the user to see.

        Returns
        -------
        dict of str : Any

        """
        d = [
            ("ID", self.id),
            ("Name", self.name),
            ("Reference", self.reference),
        ]

        if internal:
            d += [
                ("Special", self.is_special),
                ("Class", self.class_def),
                ("Equality", self.eq_function),
                ("Args", self.args),
                ("Tune Grid", self.tune_grid),
                ("Tune Distributions", self.tune_distribution),
                ("Tune Args", self.tune_args),
                ("GPU Enabled", self.is_gpu_enabled),
            ]

        return dict(d)


class KMeansClusterContainer(ClusterContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False
        from sklearn.cluster import KMeans

        if experiment.gpu_param == "force":
            from cuml.cluster import KMeans

            logger.info("Imported cuml.cluster.KMeans")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.cluster import KMeans

                logger.info("Imported cuml.cluster.KMeans")
                gpu_imported = True

        args = {
            "n_clusters": _DEFAULT_N_CLUSTERS,
            "random_state": experiment.seed,
        }
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        if gpu_imported:
            KMeans = get_kmeans()

        super().__init__(
            id="kmeans",
            name="K-Means Clustering",
            class_def=KMeans,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
        )


class AffinityPropagationClusterContainer(ClusterContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.cluster import AffinityPropagation

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        super().__init__(
            id="ap",
            name="Affinity Propagation",
            class_def=AffinityPropagation,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
        )


class MeanShiftClusterContainer(ClusterContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.cluster import MeanShift

        args = {
            "n_jobs": experiment.n_jobs_param,
        }
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        super().__init__(
            id="meanshift",
            name="Mean Shift Clustering",
            class_def=MeanShift,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
        )


class SpectralClusteringClusterContainer(ClusterContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.cluster import SpectralClustering

        args = {
            "n_clusters": _DEFAULT_N_CLUSTERS,
            "random_state": experiment.seed,
            "n_jobs": experiment.n_jobs_param,
        }
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        super().__init__(
            id="sc",
            name="Spectral Clustering",
            class_def=SpectralClustering,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
        )


class AgglomerativeClusteringClusterContainer(ClusterContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.cluster import AgglomerativeClustering

        args = {
            "n_clusters": _DEFAULT_N_CLUSTERS,
        }
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        super().__init__(
            id="hclust",
            name="Agglomerative Clustering",
            class_def=AgglomerativeClustering,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
        )


class DBSCANClusterContainer(ClusterContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False
        from sklearn.cluster import DBSCAN

        if experiment.gpu_param == "force":
            from cuml.cluster import DBSCAN

            logger.info("Imported cuml.cluster.DBSCAN")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.cluster import DBSCAN

                logger.info("Imported cuml.cluster.DBSCAN")
                gpu_imported = True

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        if not gpu_imported:
            args["n_jobs"] = experiment.n_jobs_param
        else:
            DBSCAN = get_dbscan()

        super().__init__(
            id="dbscan",
            name="Density-Based Spatial Clustering",
            class_def=DBSCAN,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
        )


class OPTICSClusterContainer(ClusterContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.cluster import OPTICS

        args = {"n_jobs": experiment.n_jobs_param}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        super().__init__(
            id="optics",
            name="OPTICS Clustering",
            class_def=OPTICS,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
        )


class BirchClusterContainer(ClusterContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.cluster import Birch

        args = {"n_clusters": _DEFAULT_N_CLUSTERS}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        super().__init__(
            id="birch",
            name="Birch Clustering",
            class_def=Birch,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
        )


class KModesClusterContainer(ClusterContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        if not _check_soft_dependencies("kmodes", extra="models", severity="warning"):
            self.active = False
            return

        from kmodes.kmodes import KModes

        args = {
            "n_clusters": _DEFAULT_N_CLUSTERS,
            "random_state": experiment.seed,
            "n_jobs": experiment.n_jobs_param,
        }
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        super().__init__(
            id="kmodes",
            name="K-Modes Clustering",
            class_def=KModes,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
        )


def get_all_model_containers(
    experiment: Any, raise_errors: bool = True
) -> Dict[str, ClusterContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), experiment, ClusterContainer, raise_errors
    )
