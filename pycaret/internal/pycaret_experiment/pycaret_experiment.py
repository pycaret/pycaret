from collections import defaultdict
from typing import Any, Dict, Optional

import joblib
import pandas as pd

import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
import pycaret.internal.persistence
from pycaret import show_versions
from pycaret.internal.logging import get_logger
from pycaret.utils.generic import MLUsecase
from pycaret.utils.time_series.forecasting.pipeline import _pipeline_transform

LOGGER = get_logger()


class _PyCaretExperiment:
    def __init__(self) -> None:
        self._ml_usecase = None
        self._available_plots = {}
        self.variable_keys = set()
        self.exp_id = None
        self.gpu_param = False
        self.n_jobs_param = -1
        self.logger = LOGGER
        self.master_model_container = []

        # Data attrs
        self.data = None
        self.target_param = None
        self.idx = [None, None]  # Train and test indices

        # Setup attrs
        self.fold_generator = None
        self.pipeline = None
        self.display_container = None
        self._fxs = defaultdict(list)
        self._setup_ran = False
        self._setup_params = None

        self._remote = False

    def _pack_for_remote(self) -> dict:
        """Serialize local member variables and send to remote. Note we should not use
        ``__getstate__`` here because it will be hard to maintain.
        We are using a different mechanism that is more resistant to further
        code change. This private method is for parallel processing.
        """
        return {"_setup_params": self._setup_params, "_remote": True}

    def _unpack_at_remote(self, data: dict) -> None:
        """Deserialize member variables at remote to reconstruct the experiment.
        This private method is for parallel processing.
        """
        for k, v in data.items():
            setattr(self, k, v)

    def _register_setup_params(self, params: dict) -> None:
        """Register the parameters used to call ``setup`` at local machine.
        This information will be sent to remote workers to re-setup the experiments.
        This private method is for parallel processing.
        """
        self._setup_params = {
            k: v for k, v in params.items() if k != "self" and v is not None
        }

    @property
    def _gpu_n_jobs_param(self) -> int:
        return self.n_jobs_param if not self.gpu_param else 1

    @property
    def variables(self) -> dict:
        return {k: getattr(self, k, None) for k in self.variable_keys}

    @property
    def _is_multiclass(self) -> bool:
        """
        Method to check if the problem is multiclass.
        """
        return False

    def _check_environment(self) -> None:
        # logging environment and libraries
        self.logger.info("Checking environment")

        from platform import machine, platform, python_build, python_version

        self.logger.info(f"python_version: {python_version()}")
        self.logger.info(f"python_build: {python_build()}")
        self.logger.info(f"machine: {machine()}")
        self.logger.info(f"platform: {platform()}")

        import psutil

        self.logger.info(f"Memory: {psutil.virtual_memory()}")
        self.logger.info(f"Physical Core: {psutil.cpu_count(logical=False)}")
        self.logger.info(f"Logical Core: {psutil.cpu_count(logical=True)}")

        self.logger.info("Checking libraries")
        self.logger.info(show_versions(logger=self.logger))

    def setup(self, *args, **kwargs) -> None:
        return

    def _check_setup_ran(self):
        """Checks to see if setup has been run or not. If it has not been run, then
        an error is raised. Useful for operations that require setup to be run before
        they can be executed. e.g. in some experiments, setup must be run first before
        plotting can be done.

        Raises
        ------
        RuntimeError
            If setup has not been run.
        """
        if not self._setup_ran:
            raise RuntimeError(
                "This function/method requires the users to run setup() first."
                "\nMore info: https://pycaret.gitbook.io/docs/get-started/quickstart"
            )

    def deploy_model(
        self,
        model,
        model_name: str,
        authentication: dict,
        platform: str = "aws",  # added gcp and azure support in pycaret==2.1
    ):
        return None

    def save_model(
        self,
        model,
        model_name: str,
        model_only: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        return None

    def load_model(
        self,
        model_name: str,
        platform: Optional[str] = None,
        authentication: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ):

        """
        This function loads a previously saved transformation pipeline and model
        from the current active directory into the current python environment.
        Load object must be a pickle file.

        Example
        -------
        >>> saved_lr = load_model('lr_model_23122019')

        This will load the previously saved model in saved_lr variable. The file
        must be in the current directory.

        Parameters
        ----------
        model_name : str, default = none
            Name of pickle file to be passed as a string.

        platform: str, default = None
            Name of platform, if loading model from cloud. Current available options are:
            'aws', 'gcp' and 'azure'.

        authentication : dict
            dictionary of applicable authentication tokens.

            When platform = 'aws':
            {'bucket' : 'Name of Bucket on S3'}

            When platform = 'gcp':
            {'project': 'gcp_pycaret', 'bucket' : 'pycaret-test'}

            When platform = 'azure':
            {'container': 'pycaret-test'}

        verbose: bool, default = True
            Success message is not printed when verbose is set to False.

        Returns
        -------
        Model Object

        """

        return pycaret.internal.persistence.load_model(
            model_name, platform, authentication, verbose
        )

    def get_logs(
        self, experiment_name: Optional[str] = None, save: bool = False
    ) -> pd.DataFrame:

        """
        Returns a table with experiment logs consisting
        run details, parameter, metrics and tags.

        Example
        -------
        >>> logs = get_logs()

        This will return pandas dataframe.

        Parameters
        ----------
        experiment_name : str, default = None
            When set to None current active run is used.

        save : bool, default = False
            When set to True, csv file is saved in current directory.

        Returns
        -------
        pandas.DataFrame

        """

        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        if experiment_name is None:
            exp_id = self.exp_id
            experiment = client.get_experiment(exp_id)
            if experiment is None:
                raise ValueError(
                    "No active run found. Check logging parameter in setup "
                    "or to get logs for inactive run pass experiment_name."
                )

            exp_name_log_ = experiment.name
        else:
            exp_name_log_ = experiment_name
            experiment = client.get_experiment_by_name(exp_name_log_)
            if experiment is None:
                raise ValueError(
                    "No active run found. Check logging parameter in setup "
                    "or to get logs for inactive run pass experiment_name."
                )

            exp_id = client.get_experiment_by_name(exp_name_log_).experiment_id

        runs = mlflow.search_runs(exp_id)

        if save:
            file_name = f"{exp_name_log_}_logs.csv"
            runs.to_csv(file_name, index=False)

        return runs

    def get_config(self, variable: str) -> Any:
        """
        This function is used to access global environment variables.

        Example
        -------
        >>> X_train = get_config('X_train')

        This will return X_train transformed dataset.

        Returns
        -------
        variable

        """
        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
        )

        self.logger.info("Initializing get_config()")
        self.logger.info(f"get_config({function_params_str})")

        if variable not in self.variables:
            raise ValueError(
                f"Variable {variable} not found. Possible variables are: {list(self.variables)}"
            )

        if any(variable.endswith(attr) for attr in ("train", "test", "dataset")):
            variable += "_transformed"

        var = getattr(self, variable)

        self.logger.info(f"Variable: {variable[:-12]} returned as {var}")
        self.logger.info(
            "get_config() successfully completed......................................"
        )

        return var

    def set_config(
        self, variable: Optional[str] = None, value: Optional[Any] = None, **kwargs
    ) -> None:
        """
        This function is used to reset global environment variables.

        Example
        -------
        >>> set_config('seed', 123)

        This will set the global seed to '123'.

        """
        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
        )

        self.logger.info("Initializing set_config()")
        self.logger.info(f"set_config({function_params_str})")

        if kwargs and variable:
            raise ValueError(
                "variable parameter cannot be used together with keyword arguments."
            )

        variables = kwargs if kwargs else {variable: value}

        for k, v in variables.items():
            if k.startswith("_"):
                raise ValueError(f"Variable {k} is read only ('_' prefix).")

            if k not in self.variables:
                raise ValueError(
                    f"Variable {k} not found. Possible variables are: {list(self.variables)}"
                )

            setattr(self, k, v)
            self.logger.info(f"Global variable: {k} updated to {v}")
        self.logger.info(
            "set_config() successfully completed......................................"
        )
        return

    def save_config(self, file_name: str) -> None:
        """
        This function save all global variables to a pickle file, allowing to
        later resume without rerunning the ``setup``.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> save_config('myvars.pkl')


        Returns:
            None

        """
        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
        )

        self.logger.info("Initializing save_config()")
        self.logger.info(f"save_config({function_params_str})")

        globals_to_ignore = {
            "_all_models",
            "_all_models_internal",
            "_all_metrics",
            "master_model_container",
            "display_container",
        }

        globals_to_dump = {
            k: v
            for k, v in self.variables.items()
            if k not in globals_to_ignore
            and not isinstance(getattr(self.__class__, k, None), property)
            and not k.startswith("_")
        }

        joblib.dump(globals_to_dump, file_name)

        self.logger.info(f"Global variables dumped to {file_name}")
        self.logger.info(
            "save_config() successfully completed......................................"
        )
        return

    def load_config(self, file_name: str) -> None:
        """
        This function loads global variables from a pickle file into Python
        environment.


        Example
        -------
        >>> from pycaret.classification import load_config
        >>> load_config('myvars.pkl')


        Returns:
            Global variables

        """
        self._check_setup_ran()

        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
        )

        self.logger.info("Initializing load_config()")
        self.logger.info(f"load_config({function_params_str})")

        loaded_globals = joblib.load(file_name)

        self.logger.info(f"Global variables loaded from {file_name}")

        for k, v in loaded_globals.items():
            self.set_config(k, v)
            self.logger.info(f"Global variable: {k} updated to {v}")

        self.logger.info(f"Global variables set to match those in {file_name}")

        self.logger.info(
            "load_config() successfully completed......................................"
        )
        return

    def pull(self, pop=False) -> pd.DataFrame:  # added in pycaret==2.2.0
        """
        Returns the latest displayed table.

        Parameters
        ----------
        pop : bool, default = False
            If true, will pop (remove) the returned dataframe from the
            display container.

        Returns
        -------
        pandas.DataFrame
            Equivalent to get_config('display_container')[-1]

        """
        return self.display_container.pop(-1) if pop else self.display_container[-1]

    @property
    def dataset(self):
        """Complete dataset without ignored columns."""
        return self.data[[c for c in self.data.columns if c not in self._fxs["Ignore"]]]

    @property
    def train(self):
        """Training set."""
        return self.dataset.loc[self.idx[0], :]

    @property
    def test(self):
        """Test set."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            return self.dataset.loc[self.idx[1], :]
        else:
            # Return the y_test indices not X_test indices.
            # X_test indices are expanded indices for handling FH with gaps.
            # But if we return X_test indices, then we will get expanded test
            # indices even for univariate time series without exogenous variables
            # which would be confusing. Hence, we return y_test indices here and if
            # we want to get X_test indices, then we use self.X_test directly.
            # Refer:
            # https://github.com/alan-turing-institute/sktime/issues/2598#issuecomment-1203308542
            # https://github.com/alan-turing-institute/sktime/blob/4164639e1c521b112711c045d0f7e63013c1e4eb/sktime/forecasting/model_evaluation/_functions.py#L196
            return self.dataset.loc[self.idx[1], :]

    @property
    def X(self):
        """Feature set."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            if self.target_param:
                return self.dataset.drop(self.target_param, axis=1)
            else:
                return self.dataset  # For unsupervised: dataset == X
        else:
            X = self.dataset.drop(self.target_param, axis=1)
            if X.empty:
                return None
            else:
                return X

    @property
    def y(self):
        """Target column."""
        if self.target_param:
            return self.dataset[self.target_param]

    @property
    def X_train(self):
        """Feature set of the training set."""
        if self.target_param is not None:
            # Supervised Learning
            X_train = self.train.drop(self.target_param, axis=1)
        else:
            # Unsupervised Learning
            X_train = self.train
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            return X_train
        else:
            if X_train.empty:
                return None
            else:
                return X_train

    @property
    def X_test(self):
        """Feature set of the test set."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            if self.target_param is not None:
                X_test = self.test.drop(self.target_param, axis=1)
            else:
                # Unsupervised Learning
                X_test = self.test
        else:
            # Use index for y_test (idx 2) to get the data
            test = self.dataset.loc[self.idx[2], :]
            X_test = test.drop(self.target_param, axis=1)

        if self._ml_usecase != MLUsecase.TIME_SERIES:
            return X_test
        else:
            if X_test.empty:
                return None
            else:
                return X_test

    @property
    def y_train(self):
        """Target column of the training set."""
        if self.target_param:
            return self.train[self.target_param]

    @property
    def y_test(self):
        """Target column of the test set."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            if self.target_param:
                return self.test[self.target_param]
        else:
            if self.target_param:
                # Use index for y_test (idx 1) to get the data
                test = self.dataset.loc[self.idx[1], :]
                return test[self.target_param]

    @property
    def dataset_transformed(self):
        """Transformed dataset."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            if self.target_param:
                return pd.concat([self.train_transformed, self.test_transformed])
            else:
                return self.train_transformed
        else:
            # Use fully trained pipeline to get the requested data
            return pd.concat(
                [
                    *_pipeline_transform(
                        pipeline=self.pipeline_fully_trained, y=self.y, X=self.X
                    )
                ],
                axis=1,
            )

    @property
    def train_transformed(self):
        """Transformed training set."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            if self.target_param:
                return pd.concat(
                    [self.X_train_transformed, self.y_train_transformed],
                    axis=1,
                )
            else:
                return self.X_train_transformed
        else:
            # Use pipeline trained on training data only to get the requested data
            # In time series, the order of arguments and returns may be reversed.
            return pd.concat(
                [
                    *_pipeline_transform(
                        pipeline=self.pipeline, y=self.y_train, X=self.X_train
                    )
                ],
                axis=1,
            )

    @property
    def test_transformed(self):
        """Transformed test set."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            return pd.concat(
                [self.X_test_transformed, self.y_test_transformed],
                axis=1,
            )
        else:
            # When transforming the test set, we can and should use all data before that
            # In time series, the order of arguments and returns may be reversed.
            all_data = pd.concat(
                [
                    *_pipeline_transform(
                        pipeline=self.pipeline_fully_trained,
                        y=self.y,
                        X=self.X,
                    )
                ],
                axis=1,
            )
            # Return the y_test indices not X_test indices.
            # X_test indices are expanded indices for handling FH with gaps.
            # But if we return X_test indices, then we will get expanded test
            # indices even for univariate time series without exogenous variables
            # which would be confusing. Hence, we return y_test indices here and if
            # we want to get X_test indices, then we use self.X_test directly.
            # Refer:
            # https://github.com/alan-turing-institute/sktime/issues/2598#issuecomment-1203308542
            # https://github.com/alan-turing-institute/sktime/blob/4164639e1c521b112711c045d0f7e63013c1e4eb/sktime/forecasting/model_evaluation/_functions.py#L196
            return all_data.loc[self.idx[1]]

    @property
    def X_transformed(self):
        """Transformed feature set."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            if self.target_param:
                return pd.concat([self.X_train_transformed, self.X_test_transformed])
            else:
                return self.X_train_transformed
        else:
            # Use fully trained pipeline to get the requested data
            # In time series, the order of arguments and returns may be reversed.
            return _pipeline_transform(
                pipeline=self.pipeline_fully_trained, y=self.y, X=self.X
            )[1]

    @property
    def y_transformed(self):
        """Transformed target column."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            return pd.concat([self.y_train_transformed, self.y_test_transformed])
        else:
            # Use fully trained pipeline to get the requested data
            # In time series, the order of arguments and returns may be reversed.
            return _pipeline_transform(
                pipeline=self.pipeline_fully_trained, y=self.y, X=self.X
            )[0]

    @property
    def X_train_transformed(self):
        """Transformed feature set of the training set."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            if self.target_param:
                return self.pipeline.transform(
                    X=self.X_train,
                    y=self.y_train,
                    filter_train_only=False,
                )[0]
            else:
                return self.pipeline.transform(self.X_train, filter_train_only=False)
        else:
            # Use pipeline trained on training data only to get the requested data
            # In time series, the order of arguments and returns may be reversed.
            return _pipeline_transform(
                pipeline=self.pipeline, y=self.y_train, X=self.X_train
            )[1]

    @property
    def X_test_transformed(self):
        """Transformed feature set of the test set."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            return self.pipeline.transform(self.X_test)
        else:
            # In time series, the order of arguments and returns may be reversed.
            # When transforming the test set, we can and should use all data before that
            _, X = _pipeline_transform(
                pipeline=self.pipeline_fully_trained, y=self.y, X=self.X
            )

            if X is None:
                return None
            else:
                return X.loc[self.idx[2]]

    @property
    def y_train_transformed(self):
        """Transformed target column of the training set."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            return self.pipeline.transform(
                X=self.X_train,
                y=self.y_train,
                filter_train_only=False,
            )[1]
        else:
            # Use pipeline trained on training data only to get the requested data
            # In time series, the order of arguments and returns may be reversed.
            return _pipeline_transform(
                pipeline=self.pipeline, y=self.y_train, X=self.X_train
            )[0]

    @property
    def y_test_transformed(self):
        """Transformed target column of the test set."""
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            return self.pipeline.transform(y=self.y_test)
        else:
            # In time series, the order of arguments and returns may be reversed.
            # When transforming the test set, we can and should use all data before that
            y, _ = _pipeline_transform(
                pipeline=self.pipeline_fully_trained, y=self.y, X=self.X
            )
            return y.loc[self.idx[1]]
