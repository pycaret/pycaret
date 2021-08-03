from pycaret.internal.pycaret_experiment.utils import highlight_setup, MLUsecase
from pycaret.internal.pycaret_experiment.supervised_experiment import (
    _SupervisedExperiment,
)
from pycaret.internal.utils import color_df
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
from pycaret.internal.logging import get_logger
from pycaret.internal.Display import Display
from pycaret.internal.distributions import *
from pycaret.internal.validation import *
import pycaret.containers.metrics.classification
import pycaret.containers.models.classification
import pycaret.internal.preprocess
import pycaret.internal.persistence
import pandas as pd  # type ignore
from pandas.io.formats.style import Styler
import numpy as np  # type: ignore
import datetime
import time
import gc
from typing import List, Tuple, Any, Union, Optional, Dict
import warnings
import traceback
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import logging


warnings.filterwarnings("ignore")
LOGGER = get_logger()


class ClassificationExperiment(_SupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.CLASSIFICATION
        self.exp_name_log = "clf-default-name"
        self.variable_keys = self.variable_keys.union(
            {"fix_imbalance_param", "fix_imbalance_method_param"}
        )
        self._available_plots = {
            "parameter": "Hyperparameters",
            "auc": "AUC",
            "confusion_matrix": "Confusion Matrix",
            "threshold": "Threshold",
            "pr": "Precision Recall",
            "error": "Prediction Error",
            "class_report": "Class Report",
            "rfe": "Feature Selection",
            "learning": "Learning Curve",
            "manifold": "Manifold Learning",
            "calibration": "Calibration Curve",
            "vc": "Validation Curve",
            "dimension": "Dimensions",
            "feature": "Feature Importance",
            "feature_all": "Feature Importance (All)",
            "boundary": "Decision Boundary",
            "lift": "Lift Chart",
            "gain": "Gain Chart",
            "tree": "Decision Tree",
        }

    def _get_setup_display(self, **kwargs) -> Styler:
        """Display overview of the setup."""
        overview = pd.DataFrame(columns=["Description", "Value"])
        overview = overview.append(
            [
                ["session_id", self.seed],
                ["Target", self.target_param],
                ["Target type", kwargs["target_type"]],
                ["Data shape", self.data.shape],
                ["Ordinal features", kwargs["ordinal_features"]],
                ["Numerical features", kwargs["numerical_features"]],
                ["Categorical features", kwargs["categorical_features"]],
                ["Date features", kwargs["date_features"]],
                ["Ignored features", kwargs["ignore_features"]],
                ["Missing Values", kwargs["missing_values"]],
                ["Fold Generator", type(self.fold_generator).__name__],
                ["Fold Number", self.fold_param],
                ["CPU Jobs", self.n_jobs_param],
                ["Use GPU", self.gpu_param],
                ["Log Experiment", self.logging_param],
                ["Experiment Name", self.exp_name_log],
                ["USI", self.USI],
            ],
            ignore_index=True,
        )

        return overview.style.apply(highlight_setup)

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in pycaret.containers.models.classification.get_all_model_containers(
                self.variables, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = pycaret.containers.models.classification.get_all_model_containers(
            self.variables, raise_errors=raise_errors
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return pycaret.containers.metrics.classification.get_all_metric_containers(
            self.variables, raise_errors=raise_errors
        )

    def _is_multiclass(self) -> bool:
        """
        Method to check if the problem is multiclass.
        """
        try:
            return self.y.value_counts().count() > 2
        except Exception:
            return False

    def setup(self, *args, **kwargs):
        if kwargs["log_plots"] is True:
            kwargs["log_plots"] = ["auc", "confusion_matrix", "feature"]
        return super().setup(*args, **kwargs)

    def compare_models(
        self,
        include: Optional[List[Union[str, Any]]] = None,
        exclude: Optional[List[str]] = None,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        sort: str = "Accuracy",
        n_select: int = 1,
        budget_time: Optional[float] = None,
        turbo: bool = True,
        errors: str = "ignore",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
    ) -> Union[Any, List[Any]]:

        """
        This function trains and evaluates performance of all estimators available in the
        model library using cross validation. The output of this function is a score grid
        with average cross validated scores. Metrics evaluated during CV can be accessed
        using the ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> best_model = compare_models()


        include: list of str or scikit-learn compatible object, default = None
            To train and evaluate select models, list containing model ID or scikit-learn
            compatible object can be passed in include param. To see a list of all models
            available in the model library use the ``models`` function.


        exclude: list of str, default = None
            To omit certain models from training and evaluation, pass a list containing
            model id in the exclude parameter. To see a list of all models available
            in the model library use the ``models`` function.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        cross_validation: bool, default = True
            When set to False, metrics are evaluated on holdout set. ``fold`` param
            is ignored when cross_validation is set to False.


        sort: str, default = 'Accuracy'
            The sort order of the score grid. It also accepts custom metrics that are
            added through the ``add_metric`` function.


        n_select: int, default = 1
            Number of top_n models to return. For example, to select top 3 models use
            n_select = 3.


        budget_time: int or float, default = None
            If not None, will terminate execution of the function after budget_time
            minutes have passed and return results up to that point.


        turbo: bool, default = True
            When set to True, it excludes estimators with longer training times. To
            see which algorithms are excluded use the ``models`` function.


        errors: str, default = 'ignore'
            When set to 'ignore', will skip the model with exceptions and continue.
            If 'raise', will break the function when exceptions are raised.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when 'GroupKFold' is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in the training dataset. When string is passed, it is interpreted
            as the column name in the dataset containing group labels.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained model or list of trained models, depending on the ``n_select`` param.

        Warnings
        --------
        - Changing turbo parameter to False may result in very high training times with
        datasets exceeding 10,000 rows.

        - AUC for estimators that does not support 'predict_proba' is shown as 0.0000.

        - No models are logged in ``MLFlow`` when ``cross_validation`` parameter is False.
        """

        return super().compare_models(
            include=include,
            exclude=exclude,
            fold=fold,
            round=round,
            cross_validation=cross_validation,
            sort=sort,
            n_select=n_select,
            budget_time=budget_time,
            turbo=turbo,
            errors=errors,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
        )

    def create_model(
        self,
        estimator: Union[str, Any],
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Any:

        """
        This function trains and evaluates the performance of a given estimator
        using cross validation. The output of this function is a score grid with
        CV scores by fold. Metrics evaluated during CV can be accessed using the
        ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function. All the available models
        can be accessed using the ``models`` function.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')


        estimator: str or scikit-learn compatible object
            ID of an estimator available in model library or pass an untrained
            model object consistent with scikit-learn API. Estimators available
            in the model library (ID - Name):

            * 'lr' - Logistic Regression
            * 'knn' - K Neighbors Classifier
            * 'nb' - Naive Bayes
            * 'dt' - Decision Tree Classifier
            * 'svm' - SVM - Linear Kernel
            * 'rbfsvm' - SVM - Radial Kernel
            * 'gpc' - Gaussian Process Classifier
            * 'mlp' - MLP Classifier
            * 'ridge' - Ridge Classifier
            * 'rf' - Random Forest Classifier
            * 'qda' - Quadratic Discriminant Analysis
            * 'ada' - Ada Boost Classifier
            * 'gbc' - Gradient Boosting Classifier
            * 'lda' - Linear Discriminant Analysis
            * 'et' - Extra Trees Classifier
            * 'xgboost' - Extreme Gradient Boosting
            * 'lightgbm' - Light Gradient Boosting Machine
            * 'catboost' - CatBoost Classifier


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        cross_validation: bool, default = True
            When set to False, metrics are evaluated on holdout set. ``fold`` param
            is ignored when cross_validation is set to False.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        **kwargs:
            Additional keyword arguments to pass to the estimator.


        Returns:
            Trained Model


        Warnings
        --------
        - AUC for estimators that does not support 'predict_proba' is shown as 0.0000.

        - Models are not logged on the ``MLFlow`` server when ``cross_validation`` param
        is set to False.

        """

        return super().create_model(
            estimator=estimator,
            fold=fold,
            round=round,
            cross_validation=cross_validation,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
            **kwargs,
        )

    def tune_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        n_iter: int = 10,
        custom_grid: Optional[Union[Dict[str, list], Any]] = None,
        optimize: str = "Accuracy",
        custom_scorer=None,
        search_library: str = "scikit-learn",
        search_algorithm: Optional[str] = None,
        early_stopping: Any = False,
        early_stopping_max_iters: int = 10,
        choose_better: bool = False,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        return_tuner: bool = False,
        verbose: bool = True,
        tuner_verbose: Union[int, bool] = True,
        **kwargs,
    ) -> Any:

        """
        This function tunes the hyperparameters of a given estimator. The output of
        this function is a score grid with CV scores by fold of the best selected
        model based on ``optimize`` parameter. Metrics evaluated during CV can be
        accessed using the ``get_metrics`` function. Custom metrics can be added
        or removed using ``add_metric`` and ``remove_metric`` function.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> tuned_lr = tune_model(lr)


        estimator: scikit-learn compatible object
            Trained model object


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        n_iter: int, default = 10
            Number of iterations in the grid search. Increasing 'n_iter' may improve
            model performance but also increases the training time.


        custom_grid: dictionary, default = None
            To define custom search space for hyperparameters, pass a dictionary with
            parameter name and values to be iterated. Custom grids must be in a format
            supported by the defined ``search_library``.


        optimize: str, default = 'Accuracy'
            Metric name to be evaluated for hyperparameter tuning. It also accepts custom
            metrics that are added through the ``add_metric`` function.


        custom_scorer: object, default = None
            custom scoring strategy can be passed to tune hyperparameters of the model.
            It must be created using ``sklearn.make_scorer``. It is equivalent of adding
            custom metric using the ``add_metric`` function and passing the name of the
            custom metric in the ``optimize`` parameter.
            Will be deprecated in future.


        search_library: str, default = 'scikit-learn'
            The search library used for tuning hyperparameters. Possible values:

            - 'scikit-learn' - default, requires no further installation
                https://github.com/scikit-learn/scikit-learn

            - 'scikit-optimize' - ``pip install scikit-optimize``
                https://scikit-optimize.github.io/stable/

            - 'tune-sklearn' - ``pip install tune-sklearn ray[tune]``
                https://github.com/ray-project/tune-sklearn

            - 'optuna' - ``pip install optuna``
                https://optuna.org/


        search_algorithm: str, default = None
            The search algorithm depends on the ``search_library`` parameter.
            Some search algorithms require additional libraries to be installed.
            If None, will use search library-specific default algorithm.

            - 'scikit-learn' possible values:
                - 'random' : random grid search (default)
                - 'grid' : grid search

            - 'scikit-optimize' possible values:
                - 'bayesian' : Bayesian search (default)

            - 'tune-sklearn' possible values:
                - 'random' : random grid search (default)
                - 'grid' : grid search
                - 'bayesian' : ``pip install scikit-optimize``
                - 'hyperopt' : ``pip install hyperopt``
                - 'bohb' : ``pip install hpbandster ConfigSpace``

            - 'optuna' possible values:
                - 'random' : randomized search
                - 'tpe' : Tree-structured Parzen Estimator search (default)


        early_stopping: bool or str or object, default = False
            Use early stopping to stop fitting to a hyperparameter configuration
            if it performs poorly. Ignored when ``search_library`` is scikit-learn,
            or if the estimator does not have 'partial_fit' attribute. If False or
            None, early stopping will not be used. Can be either an object accepted
            by the search library or one of the following:

            - 'asha' for Asynchronous Successive Halving Algorithm
            - 'hyperband' for Hyperband
            - 'median' for Median Stopping Rule
            - If False or None, early stopping will not be used.


        early_stopping_max_iters: int, default = 10
            Maximum number of epochs to run for each sampled configuration.
            Ignored if ``early_stopping`` is False or None.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the tuner.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        return_tuner: bool, default = False
            When set to True, will return a tuple of (model, tuner_object).


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        tuner_verbose: bool or in, default = True
            If True or above 0, will print messages from the tuner. Higher values
            print more messages. Ignored when ``verbose`` parameter is False.


        **kwargs:
            Additional keyword arguments to pass to the optimizer.


        Returns:
            Trained Model and Optional Tuner Object when ``return_tuner`` is True.


        Warnings
        --------
        - Using 'grid' as ``search_algorithm`` may result in very long computation.
        Only recommended with smaller search spaces that can be defined in the
        ``custom_grid`` parameter.

        - ``search_library`` 'tune-sklearn' does not support GPU models.

        """

        return super().tune_model(
            estimator=estimator,
            fold=fold,
            round=round,
            n_iter=n_iter,
            custom_grid=custom_grid,
            optimize=optimize,
            custom_scorer=custom_scorer,
            search_library=search_library,
            search_algorithm=search_algorithm,
            early_stopping=early_stopping,
            early_stopping_max_iters=early_stopping_max_iters,
            choose_better=choose_better,
            fit_kwargs=fit_kwargs,
            groups=groups,
            return_tuner=return_tuner,
            verbose=verbose,
            tuner_verbose=tuner_verbose,
            **kwargs,
        )

    def ensemble_model(
        self,
        estimator,
        method: str = "Bagging",
        fold: Optional[Union[int, Any]] = None,
        n_estimators: int = 10,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = "Accuracy",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
    ) -> Any:

        """
        This function ensembles a given estimator. The output of this function is
        a score grid with CV scores by fold. Metrics evaluated during CV can be
        accessed using the ``get_metrics`` function. Custom metrics can be added
        or removed using ``add_metric`` and ``remove_metric`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> dt = create_model('dt')
        >>> bagged_dt = ensemble_model(dt, method = 'Bagging')


        estimator: scikit-learn compatible object
            Trained model object


        method: str, default = 'Bagging'
            Method for ensembling base estimator. It can be 'Bagging' or 'Boosting'.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        n_estimators: int, default = 10
            The number of base estimators in the ensemble. In case of perfect fit, the
            learning procedure is stopped early.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        optimize: str, default = 'Accuracy'
            Metric to compare for model selection when ``choose_better`` is True.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained Model


        Warnings
        --------
        - Method 'Boosting' is not supported for estimators that do not have 'class_weights'
        or 'predict_proba' attributes.

        """

        return super().ensemble_model(
            estimator=estimator,
            method=method,
            fold=fold,
            n_estimators=n_estimators,
            round=round,
            choose_better=choose_better,
            optimize=optimize,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
        )

    def blend_models(
        self,
        estimator_list: list,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = "Accuracy",
        method: str = "auto",
        weights: Optional[List[float]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
    ) -> Any:

        """
        This function trains a Soft Voting / Majority Rule classifier for select
        models passed in the ``estimator_list`` param. The output of this function
        is a score grid with CV scores by fold. Metrics evaluated during CV can be
        accessed using the ``get_metrics`` function. Custom metrics can be added
        or removed using ``add_metric`` and ``remove_metric`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> top3 = compare_models(n_select = 3)
        >>> blender = blend_models(top3)


        estimator_list: list of scikit-learn compatible objects
            List of trained model objects


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        optimize: str, default = 'Accuracy'
            Metric to compare for model selection when ``choose_better`` is True.


        method: str, default = 'auto'
            'hard' uses predicted class labels for majority rule voting. 'soft', predicts
            the class label based on the argmax of the sums of the predicted probabilities,
            which is recommended for an ensemble of well-calibrated classifiers. Default
            value, 'auto', will try to use 'soft' and fall back to 'hard' if the former is
            not supported.


        weights: list, default = None
            Sequence of weights (float or int) to weight the occurrences of predicted class
            labels (hard voting) or class probabilities before averaging (soft voting). Uses
            uniform weights when None.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained Model

        """

        return super().blend_models(
            estimator_list=estimator_list,
            fold=fold,
            round=round,
            choose_better=choose_better,
            optimize=optimize,
            method=method,
            weights=weights,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
        )

    def stack_models(
        self,
        estimator_list: list,
        meta_model=None,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        method: str = "auto",
        restack: bool = False,
        choose_better: bool = False,
        optimize: str = "Accuracy",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
    ) -> Any:

        """
        This function trains a meta model over select estimators passed in
        the ``estimator_list`` parameter. The output of this function is a
        score grid with CV scores by fold. Metrics evaluated during CV can
        be accessed using the ``get_metrics`` function. Custom metrics
        can be added or removed using ``add_metric`` and ``remove_metric``
        function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> top3 = compare_models(n_select = 3)
        >>> stacker = stack_models(top3)


        estimator_list: list of scikit-learn compatible objects
            List of trained model objects


        meta_model: scikit-learn compatible object, default = None
            When None, Logistic Regression is trained as a meta model.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        method: str, default = 'auto'
            When set to 'auto', it will invoke, for each estimator, 'predict_proba',
            'decision_function' or 'predict' in that order. Other, manually pass one
            of the value from 'predict_proba', 'decision_function' or 'predict'.


        restack: bool, default = False
            When set to False, only the predictions of estimators will be used as
            training data for the ``meta_model``.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        optimize: str, default = 'Accuracy'
            Metric to compare for model selection when ``choose_better`` is True.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained Model


        Warnings
        --------
        - When ``method`` is not set to 'auto', it will check if the defined method
        is available for all estimators passed in ``estimator_list``. If the method is
        not implemented by any estimator, it will raise an error.

        """

        return super().stack_models(
            estimator_list=estimator_list,
            meta_model=meta_model,
            fold=fold,
            round=round,
            method=method,
            restack=restack,
            choose_better=choose_better,
            optimize=optimize,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
        )

    def plot_model(
        self,
        estimator,
        plot: str = "auc",
        scale: float = 1,
        save: bool = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        use_train_data: bool = False,
        verbose: bool = True,
        display_format: Optional[str] = None,
    ) -> str:

        """
        This function analyzes the performance of a trained model on holdout set.
        It may require re-training the model in certain cases.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> plot_model(lr, plot = 'auc')


        estimator: scikit-learn compatible object
            Trained model object


        plot: str, default = 'auc'
            List of available plots (ID - Name):

            * 'residuals_interactive' - Interactive Residual plots
            * 'auc' - Area Under the Curve
            * 'threshold' - Discrimination Threshold
            * 'pr' - Precision Recall Curve
            * 'confusion_matrix' - Confusion Matrix
            * 'error' - Class Prediction Error
            * 'class_report' - Classification Report
            * 'boundary' - Decision Boundary
            * 'rfe' - Recursive Feature Selection
            * 'learning' - Learning Curve
            * 'manifold' - Manifold Learning
            * 'calibration' - Calibration Curve
            * 'vc' - Validation Curve
            * 'dimension' - Dimension Learning
            * 'feature' - Feature Importance
            * 'feature_all' - Feature Importance (All)
            * 'parameter' - Model Hyperparameter
            * 'lift' - Lift Curve
            * 'gain' - Gain Chart
            * 'tree' - Decision Tree


        scale: float, default = 1
            The resolution scale of the figure.


        save: bool, default = False
            When set to True, plot is saved in the current working directory.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.


        verbose: bool, default = True
            When set to False, progress bar is not displayed.


        display_format: str, default = None
            To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.
            Currently, not all plots are supported.


        Returns:
            None


        Warnings
        --------
        -   Estimators that does not support 'predict_proba' attribute cannot be used for
            'AUC' and 'calibration' plots.

        -   When the target is multiclass, 'calibration', 'threshold', 'manifold' and 'rfe'
            plots are not available.

        -   When the 'max_features' parameter of a trained model object is not equal to
            the number of samples in training set, the 'rfe' plot is not available.

        """

        return super().plot_model(
            estimator=estimator,
            plot=plot,
            scale=scale,
            save=save,
            fold=fold,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
            use_train_data=use_train_data,
            system=True,
            display_format=display_format,
        )

    def evaluate_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        use_train_data: bool = False,
    ):

        """
        This function displays a user interface for analyzing performance of a trained
        model. It calls the ``plot_model`` function internally.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> evaluate_model(lr)


        estimator: scikit-learn compatible object
            Trained model object


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.


        Returns:
            None


        Warnings
        --------
        -   This function only works in IPython enabled Notebook.

        """

        return super().evaluate_model(
            estimator=estimator,
            fold=fold,
            fit_kwargs=fit_kwargs,
            groups=groups,
            use_train_data=use_train_data,
        )

    def interpret_model(
        self,
        estimator,
        plot: str = "summary",
        feature: Optional[str] = None,
        observation: Optional[int] = None,
        use_train_data: bool = False,
        **kwargs,
    ):

        """
        This function analyzes the predictions generated from a tree-based model. It is
        implemented based on the SHAP (SHapley Additive exPlanations). For more info on
        this, please see https://shap.readthedocs.io/en/latest/

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> xgboost = create_model('xgboost')
        >>> interpret_model(xgboost)


        estimator: scikit-learn compatible object
            Trained model object


        plot: str, default = 'summary'
            Type of plot. Available options are: 'summary', 'correlation', and 'reason'.


        feature: str, default = None
            Feature to check correlation with. This parameter is only required when ``plot``
            type is 'correlation'. When set to None, it uses the first column in the train
            dataset.


        observation: int, default = None
            Observation index number in holdout set to explain. When ``plot`` is not
            'reason', this parameter is ignored.


        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.


        **kwargs:
            Additional keyword arguments to pass to the plot.


        Returns:
            None

        """

        return super().interpret_model(
            estimator=estimator,
            plot=plot,
            feature=feature,
            observation=observation,
            use_train_data=use_train_data,
            **kwargs,
        )

    def calibrate_model(
        self,
        estimator,
        method: str = "sigmoid",
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
        display: Optional[Display] = None,  # added in pycaret==2.2.0
    ) -> Any:

        """
        This function takes the input of trained estimator and performs probability
        calibration with sigmoid or isotonic regression. The output prints a score
        grid that shows Accuracy, AUC, Recall, Precision, F1, Kappa and MCC by fold
        (default = 10 Fold). The ouput of the original estimator and the calibrated
        estimator (created using this function) might not differ much. In order
        to see the calibration differences, use 'calibration' plot in plot_model to
        see the difference before and after.

        This function returns a trained model object.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> dt_boosted = create_model('dt', ensemble = True, method = 'Boosting')
        >>> calibrated_dt = calibrate_model(dt_boosted)

        This will return Calibrated Boosted Decision Tree Model.

        Parameters
        ----------
        estimator : object

        method : str, default = 'sigmoid'
            The method to use for calibration. Can be 'sigmoid' which corresponds to Platt's
            method or 'isotonic' which is a non-parametric approach. It is not advised to use
            isotonic calibration with too few calibration samples

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.

        Returns
        -------
        score_grid
            A table containing the scores of the model across the kfolds.
            Scoring metrics used are Accuracy, AUC, Recall, Precision, F1,
            Kappa and MCC. Mean and standard deviation of the scores across
            the folds are also returned.

        model
            trained and calibrated model object.

        Warnings
        --------
        - Avoid isotonic calibration with too few calibration samples (<1000) since it
        tends to overfit.

        - calibration plot not available for multiclass problems.


        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing calibrate_model()")
        self.logger.info(f"calibrate_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

        """

        ERROR HANDLING ENDS HERE

        """

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        self.logger.info("Preloading libraries")

        # pre-load libraries

        self.logger.info("Preparing display monitor")

        if not display:
            progress_args = {"max": 2 + 4}
            master_display_columns = [
                v.display_name for k, v in self._all_metrics.items()
            ]
            timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
            monitor_rows = [
                ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
                [
                    "Status",
                    ". . . . . . . . . . . . . . . . . .",
                    "Loading Dependencies",
                ],
                [
                    "Estimator",
                    ". . . . . . . . . . . . . . . . . .",
                    "Compiling Library",
                ],
            ]
            display = Display(
                verbose=verbose,
                html_param=self.html_param,
                progress_args=progress_args,
                master_display_columns=master_display_columns,
                monitor_rows=monitor_rows,
            )

            display.display_progress()
            display.display_monitor()
            display.display_master_display()

        np.random.seed(self.seed)

        self.logger.info("Getting model name")

        full_name = self._get_model_name(estimator)

        self.logger.info(f"Base model : {full_name}")

        display.update_monitor(2, full_name)
        display.display_monitor()

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Selecting Estimator")
        display.display_monitor()

        """
        MONITOR UPDATE ENDS
        """

        # calibrating estimator

        self.logger.info("Importing untrained CalibratedClassifierCV")

        calibrated_model_definition = self._all_models_internal["CalibratedCV"]
        model = calibrated_model_definition.class_def(
            base_estimator=estimator,
            method=method,
            cv=fold,
            **calibrated_model_definition.args,
        )

        display.move_progress()

        self.logger.info(
            "SubProcess create_model() called =================================="
        )
        model, model_fit_time = self.create_model(
            estimator=model,
            system=False,
            display=display,
            fold=fold,
            round=round,
            fit_kwargs=fit_kwargs,
            groups=groups,
        )
        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

        model_results = model_results.round(round)

        display.move_progress()

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # mlflow logging
        if self.logging_param:

            avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

            try:
                self._mlflow_log_model(
                    model=model,
                    model_results=model_results,
                    score_dict=avgs_dict_log,
                    source="calibrate_models",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=self.prep_pipe,
                    log_plots=self.log_plots_param,
                    display=display,
                )
            except:
                self.logger.error(
                    f"_mlflow_log_model() for {model} raised an exception:"
                )
                self.logger.error(traceback.format_exc())

        model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
        model_results = model_results.set_precision(round)
        display.display(model_results, clear=True)

        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "calibrate_model() succesfully completed......................................"
        )

        gc.collect()
        return model

    def optimize_threshold(
        self,
        estimator,
        true_positive: int = 0,
        true_negative: int = 0,
        false_positive: int = 0,
        false_negative: int = 0,
    ):

        """
        This function optimizes probability threshold for a trained model using custom cost
        function that can be defined using combination of True Positives, True Negatives,
        False Positives (also known as Type I error), and False Negatives (Type II error).

        This function returns a plot of optimized cost as a function of probability
        threshold between 0 to 100.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> optimize_threshold(lr, true_negative = 10, false_negative = -100)

        This will return a plot of optimized cost as a function of probability threshold.

        Parameters
        ----------
        estimator : object
            A trained model object should be passed as an estimator.

        true_positive : int, default = 0
            Cost function or returns when prediction is true positive.

        true_negative : int, default = 0
            Cost function or returns when prediction is true negative.

        false_positive : int, default = 0
            Cost function or returns when prediction is false positive.

        false_negative : int, default = 0
            Cost function or returns when prediction is false negative.


        Returns
        -------
        Visual_Plot
            Prints the visual plot.

        Warnings
        --------
        - This function is not supported for multiclass problems.


        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing optimize_threshold()")
        self.logger.info(f"optimize_threshold({function_params_str})")

        self.logger.info("Importing libraries")

        # import libraries

        np.random.seed(self.seed)

        """
        ERROR HANDLING STARTS HERE
        """

        self.logger.info("Checking exceptions")

        # exception 1 for multi-class
        if self._is_multiclass():
            raise TypeError(
                "optimize_threshold() cannot be used when target is multi-class."
            )

        # check predict_proba value
        if type(estimator) is not list:
            if not hasattr(estimator, "predict_proba"):
                raise TypeError(
                    "Estimator doesn't support predict_proba function and cannot be used in optimize_threshold()."
                )

        # check cost function type
        allowed_types = [int, float]

        if type(true_positive) not in allowed_types:
            raise TypeError(
                "true_positive parameter only accepts float or integer value."
            )

        if type(true_negative) not in allowed_types:
            raise TypeError(
                "true_negative parameter only accepts float or integer value."
            )

        if type(false_positive) not in allowed_types:
            raise TypeError(
                "false_positive parameter only accepts float or integer value."
            )

        if type(false_negative) not in allowed_types:
            raise TypeError(
                "false_negative parameter only accepts float or integer value."
            )

        """
        ERROR HANDLING ENDS HERE
        """

        # define model as estimator
        model = estimator

        model_name = self._get_model_name(model)

        # generate predictions and store actual on y_test in numpy array
        actual = self.y_test.values

        predicted = model.predict_proba(self.X_test.values)
        predicted = predicted[:, 1]

        """
        internal function to calculate loss starts here
        """

        self.logger.info("Defining loss function")

        def calculate_loss(
            actual,
            predicted,
            tp_cost=true_positive,
            tn_cost=true_negative,
            fp_cost=false_positive,
            fn_cost=false_negative,
        ):

            # true positives
            tp = predicted + actual
            tp = np.where(tp == 2, 1, 0)
            tp = tp.sum()

            # true negative
            tn = predicted + actual
            tn = np.where(tn == 0, 1, 0)
            tn = tn.sum()

            # false positive
            fp = (predicted > actual).astype(int)
            fp = np.where(fp == 1, 1, 0)
            fp = fp.sum()

            # false negative
            fn = (predicted < actual).astype(int)
            fn = np.where(fn == 1, 1, 0)
            fn = fn.sum()

            total_cost = (
                (tp_cost * tp) + (tn_cost * tn) + (fp_cost * fp) + (fn_cost * fn)
            )

            return total_cost

        """
        internal function to calculate loss ends here
        """

        grid = np.arange(0, 1, 0.0001)

        # loop starts here

        cost = []
        # global optimize_results

        self.logger.info("Iteration starts at 0")

        for i in grid:

            pred_prob = (predicted >= i).astype(int)
            cost.append(calculate_loss(actual, pred_prob))

        optimize_results = pd.DataFrame(
            {"Probability Threshold": grid, "Cost Function": cost}
        )
        fig = px.line(
            optimize_results,
            x="Probability Threshold",
            y="Cost Function",
            line_shape="linear",
        )
        fig.update_layout(plot_bgcolor="rgb(245,245,245)")
        title = f"{model_name} Probability Threshold Optimization"

        # calculate vertical line
        y0 = optimize_results["Cost Function"].min()
        y1 = optimize_results["Cost Function"].max()
        x0 = optimize_results.sort_values(by="Cost Function", ascending=False).iloc[0][
            0
        ]
        x1 = x0

        t = x0
        if self.html_param:

            fig.add_shape(
                dict(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color="red", width=2),
                )
            )
            fig.update_layout(
                title={
                    "text": title,
                    "y": 0.95,
                    "x": 0.45,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            )
            self.logger.info("Figure ready for render")
            fig.show()
        print(f"Optimized Probability Threshold: {t} | Optimized Cost Function: {y1}")
        self.logger.info(
            "optimize_threshold() succesfully completed......................................"
        )

        return float(t)

    def predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
        probability_threshold: Optional[float] = None,
        encoded_labels: bool = False,
        raw_score: bool = False,
        round: int = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:

        """
        This function predicts ``Label`` and ``Score`` (probability of predicted
        class) using a trained model. When ``data`` is None, it predicts label and
        score on the holdout set.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> pred_holdout = predict_model(lr)
        >>> pred_unseen = predict_model(lr, data = unseen_dataframe)


        estimator: scikit-learn compatible object
            Trained model object


        data: pandas.DataFrame
            Shape (n_samples, n_features). All features used during training
            must be available in the unseen dataset.


        probability_threshold: float, default = None
            Threshold for converting predicted probability to class label.
            It defaults to 0.5 for all classifiers unless explicitly defined
            in this parameter.


        encoded_labels: bool, default = False
            When set to True, will return labels encoded as an integer.


        raw_score: bool, default = False
            When set to True, scores for all labels will be returned.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        verbose: bool, default = True
            When set to False, holdout score grid is not printed.


        Returns:
            pandas.DataFrame


        Warnings
        --------
        - The behavior of the ``predict_model`` is changed in version 2.1 without backward
        compatibility. As such, the pipelines trained using the version (<= 2.0), may not
        work for inference with version >= 2.1. You can either retrain your models with a
        newer version or downgrade the version for inference.

        """

        return super().predict_model(
            estimator=estimator,
            data=data,
            probability_threshold=probability_threshold,
            encoded_labels=encoded_labels,
            raw_score=raw_score,
            round=round,
            verbose=verbose,
        )

    def finalize_model(
        self,
        estimator,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        model_only: bool = True,
    ) -> Any:

        """
        This function trains a given estimator on the entire dataset including the
        holdout set.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> final_lr = finalize_model(lr)


        estimator: scikit-learn compatible object
            Trained model object


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        model_only: bool, default = True
            When set to False, only model object is re-trained and all the
            transformations in Pipeline are ignored.


        Returns:
            Trained Model

        """

        return super().finalize_model(
            estimator=estimator,
            fit_kwargs=fit_kwargs,
            groups=groups,
            model_only=model_only,
        )

    def deploy_model(
        self, model, model_name: str, authentication: dict, platform: str = "aws",
    ):

        """
        This function deploys the transformation pipeline and trained model on cloud.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> deploy_model(model = lr, model_name = 'lr-for-deployment', platform = 'aws', authentication = {'bucket' : 'S3-bucket-name'})


        Amazon Web Service (AWS) users:
            To deploy a model on AWS S3 ('aws'), environment variables must be set in your
            local environment. To configure AWS environment variables, type ``aws configure``
            in the command line. Following information from the IAM portal of amazon console
            account is required:

            - AWS Access Key ID
            - AWS Secret Key Access
            - Default Region Name (can be seen under Global settings on your AWS console)

            More info: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html


        Google Cloud Platform (GCP) users:
            To deploy a model on Google Cloud Platform ('gcp'), project must be created
            using command line or GCP console. Once project is created, you must create
            a service account and download the service account key as a JSON file to set
            environment variables in your local environment.

            More info: https://cloud.google.com/docs/authentication/production


        Microsoft Azure (Azure) users:
            To deploy a model on Microsoft Azure ('azure'), environment variables for connection
            string must be set in your local environment. Go to settings of storage account on
            Azure portal to access the connection string required.

            More info: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json


        model: scikit-learn compatible object
            Trained model object


        model_name: str
            Name of model.


        authentication: dict
            Dictionary of applicable authentication tokens.

            When platform = 'aws':
            {'bucket' : 'S3-bucket-name'}

            When platform = 'gcp':
            {'project': 'gcp-project-name', 'bucket' : 'gcp-bucket-name'}

            When platform = 'azure':
            {'container': 'azure-container-name'}


        platform: str, default = 'aws'
            Name of the cloud platform. Currently supported platforms: 'aws', 'gcp' and 'azure'.


        Returns:
            None

        """

        return super().deploy_model(
            model=model,
            model_name=model_name,
            authentication=authentication,
            platform=platform,
        )

    def save_model(
        self,
        model,
        model_name: str,
        model_only: bool = False,
        verbose: bool = True,
        **kwargs,
    ):

        """
        This function saves the transformation pipeline and trained model object
        into the current working directory as a pickle file for later use.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> save_model(lr, 'saved_lr_model')


        model: scikit-learn compatible object
            Trained model object


        model_name: str
            Name of the model.


        model_only: bool, default = False
            When set to True, only trained model object is saved instead of the
            entire pipeline.


        verbose: bool, default = True
            Success message is not printed when verbose is set to False.


        Returns:
            Tuple of the model object and the filename.

        """

        return super().save_model(
            model=model,
            model_name=model_name,
            model_only=model_only,
            verbose=verbose,
            **kwargs,
        )

    def load_model(
        self,
        model_name,
        platform: Optional[str] = None,
        authentication: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ):

        """
        This function loads a previously saved pipeline.


        Example
        -------
        >>> from pycaret.classification import load_model
        >>> saved_lr = load_model('saved_lr_model')


        model_name: str
            Name of the model.


        platform: str, default = None
            Name of the cloud platform. Currently supported platforms:
            'aws', 'gcp' and 'azure'.


        authentication: dict, default = None
            dictionary of applicable authentication tokens.

            when platform = 'aws':
            {'bucket' : 'S3-bucket-name'}

            when platform = 'gcp':
            {'project': 'gcp-project-name', 'bucket' : 'gcp-bucket-name'}

            when platform = 'azure':
            {'container': 'azure-container-name'}


        verbose: bool, default = True
            Success message is not printed when verbose is set to False.


        Returns:
            Trained Model

        """

        return super().load_model(
            model_name=model_name,
            platform=platform,
            authentication=authentication,
            verbose=verbose,
        )

    def automl(
        self, optimize: str = "Accuracy", use_holdout: bool = False, turbo: bool = True
    ) -> Any:

        """
        This function returns the best model out of all trained models in
        current session based on the ``optimize`` parameter. Metrics
        evaluated can be accessed using the ``get_metrics`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> top3 = compare_models(n_select = 3)
        >>> tuned_top3 = [tune_model(i) for i in top3]
        >>> blender = blend_models(tuned_top3)
        >>> stacker = stack_models(tuned_top3)
        >>> best_auc_model = automl(optimize = 'AUC')


        optimize: str, default = 'Accuracy'
            Metric to use for model selection. It also accepts custom metrics
            added using the ``add_metric`` function.


        use_holdout: bool, default = False
            When set to True, metrics are evaluated on holdout set instead of CV.


        turbo: bool, default = True
            When set to True and use_holdout is False, only models created with default fold
            parameter will be considered. If set to False, models created with a non-default
            fold parameter will be scored again using default fold settings, so that they can be
            compared.


        Returns:
            Trained Model

        """
        return super().automl(optimize=optimize, use_holdout=use_holdout, turbo=turbo)

    def pull(self, pop: bool = False) -> pd.DataFrame:

        """
        Returns last printed score grid. Use ``pull`` function after
        any training function to store the score grid in pandas.DataFrame.


        pop: bool, default = False
            If True, will pop (remove) the returned dataframe from the
            display container.


        Returns:
            pandas.DataFrame

        """
        return super().pull(pop=pop)

    def models(
        self,
        type: Optional[str] = None,
        internal: bool = False,
        raise_errors: bool = True,
    ) -> pd.DataFrame:

        """
        Returns table of models available in the model library.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> all_models = models()


        type: str, default = None
            - linear : filters and only return linear models
            - tree : filters and only return tree based models
            - ensemble : filters and only return ensemble models


        internal: bool, default = False
            When True, will return extra columns and rows used internally.


        raise_errors: bool, default = True
            When False, will suppress all exceptions, ignoring models
            that couldn't be created.


        Returns:
            pandas.DataFrame

        """
        return super().models(type=type, internal=internal, raise_errors=raise_errors)

    def get_metrics(
        self,
        reset: bool = False,
        include_custom: bool = True,
        raise_errors: bool = True,
    ) -> pd.DataFrame:

        """
        Returns table of available metrics used for CV.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> all_metrics = get_metrics()


        reset: bool, default = False
            When True, will reset all changes made using the ``add_metric``
            and ``remove_metric`` function.


        include_custom: bool, default = True
            Whether to include user added (custom) metrics or not.


        raise_errors: bool, default = True
            If False, will suppress all exceptions, ignoring models that
            couldn't be created.


        Returns:
            pandas.DataFrame

        """

        return super().get_metrics(
            reset=reset, include_custom=include_custom, raise_errors=raise_errors,
        )

    def add_metric(
        self,
        id: str,
        name: str,
        score_func: type,
        target: str = "pred",
        greater_is_better: bool = True,
        multiclass: bool = True,
        **kwargs,
    ) -> pd.Series:

        """
        Adds a custom metric to be used for CV.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> from sklearn.metrics import log_loss
        >>> add_metric('logloss', 'Log Loss', log_loss, greater_is_better = False)


        id: str
            Unique id for the metric.


        name: str
            Display name of the metric.


        score_func: type
            Score function (or loss function) with signature ``score_func(y, y_pred, **kwargs)``.


        target: str, default = 'pred'
            The target of the score function.

            - 'pred' for the prediction table
            - 'pred_proba' for pred_proba
            - 'threshold' for decision_function or predict_proba


        greater_is_better: bool, default = True
            Whether ``score_func`` is higher the better or not.


        multiclass: bool, default = True
            Whether the metric supports multiclass target.


        **kwargs:
            Arguments to be passed to score function.


        Returns:
            pandas.Series

        """

        return super().add_metric(
            id=id,
            name=name,
            score_func=score_func,
            target=target,
            greater_is_better=greater_is_better,
            multiclass=multiclass,
            **kwargs,
        )

    def remove_metric(self, name_or_id: str):

        """
        Removes a metric from CV.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> remove_metric('MCC')


        name_or_id: str
            Display name or ID of the metric.


        Returns:
            None

        """
        return super().remove_metric(name_or_id=name_or_id)

    def get_logs(
        self, experiment_name: Optional[str] = None, save: bool = False
    ) -> pd.DataFrame:

        """
        Returns a table of experiment logs. Only works when ``log_experiment``
        is True when initializing the ``setup`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase', log_experiment = True)
        >>> best = compare_models()
        >>> exp_logs = get_logs()


        experiment_name: str, default = None
            When None current active run is used.


        save: bool, default = False
            When set to True, csv file is saved in current working directory.


        Returns:
            pandas.DataFrame

        """

        return super().get_logs(experiment_name=experiment_name, save=save)

