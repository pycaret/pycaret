from typing import Optional, Union

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sktime.transformations.compose import ColumnwiseTransformer
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.boxcox import BoxCoxTransformer, LogTransformer
from sktime.transformations.series.cos import CosineTransformer
from sktime.transformations.series.exponent import ExponentTransformer, SqrtTransformer
from sktime.transformations.series.impute import Imputer

from pycaret.utils.time_series import TSExogenousPresent


class TSForecastingPreprocessor:
    """Class for preprocessing Time Series Forecasting Experiments."""

    def _imputation(
        self,
        numeric_imputation_target: Optional[Union[str, int, float]],
        numeric_imputation_exogenous: Optional[Union[str, int, float]],
        exogenous_present: TSExogenousPresent,
    ):
        # Impute target ----
        if numeric_imputation_target is not None:
            self._add_imputation_steps(
                numeric_imputation=numeric_imputation_target, target=True
            )

        # Impute Exogenous ----
        # Only add exogenous pipeline steps if exogenous variables are present.
        if (
            exogenous_present == TSExogenousPresent.YES
            and numeric_imputation_exogenous is not None
        ):
            self._add_imputation_steps(
                numeric_imputation=numeric_imputation_exogenous, target=False
            )

    def _add_imputation_steps(
        self, numeric_imputation: Union[str, int, float], target: bool = True
    ):
        """Perform numeric imputation of missing values.

        Parameters
        ----------
        numeric_imputation : Union[str, int, float]
            The method to be used for imputation.
            If str, then passed as is to the underlying `sktime` imputer.
            Allowed values are:
                "drift", "linear", "nearest", "mean", "median", "backfill",
                "bfill", "pad", "ffill", "random"
            If int or float, imputation method is set to "constant" with the given value.
        target : bool, optional
            If True, imputation is added to the target variable steps
            If False, imputation is added to the exogenous variable steps,
            by default True

        Raises
        ------
        ValueError
            (1) `numeric_imputation` is of type str but not of one of the allowed values.
            (2) `numeric_imputation` is not of one of the allowed types.
        """
        type_ = "Target" if target else "Exogenous"
        self.logger.info(f"Set up imputation for {type_} variable(s).")

        if isinstance(numeric_imputation, str):
            allowed_values = [
                "drift",
                "linear",
                "nearest",
                "mean",
                "median",
                "backfill",
                "bfill",
                "pad",
                "ffill",
                "random",
            ]
            if numeric_imputation not in allowed_values:
                raise ValueError(
                    f"{type_} Imputation Type '{numeric_imputation}' not allowed."
                )
            num_estimator = Imputer(method=numeric_imputation, random_state=self.seed)
        elif isinstance(numeric_imputation, (int, float)):
            num_estimator = Imputer(method="constant", value=numeric_imputation)
        else:
            raise ValueError(
                f"{type_} Imputation Type '{type(numeric_imputation)}' is not of allowed type."
            )

        if target:
            self.transformer_steps_target.extend([("numerical_imputer", num_estimator)])
        else:
            self.transformer_steps_exogenous.extend(
                [("numerical_imputer", num_estimator)]
            )

    def _transformation(
        self,
        transform_target: Optional[Union[str, int, float]],
        transform_exogenous: Optional[Union[str, int, float]],
        exogenous_present: TSExogenousPresent,
    ):
        # Impute target ----
        if transform_target is not None:
            self._add_transformation_steps(transform=transform_target, target=True)

        # Impute Exogenous ----
        # Only add exogenous pipeline steps if exogenous variables are present.
        if (
            exogenous_present == TSExogenousPresent.YES
            and transform_exogenous is not None
        ):
            self._add_transformation_steps(transform=transform_exogenous, target=False)

    def _add_transformation_steps(self, transform: str, target: bool = True):
        """Power transform the data to be more Gaussian-like.

        Parameters
        ----------
        transform : str
            The method to be used for transformation. Allowed values and
            corresponding transformers are:
                "box-cox": BoxCoxTransformer(),
                "log": LogTransformer(),
                "sqrt": SqrtTransformer(),
                "exp": ExponentTransformer(),
                "cos": CosineTransformer(),
        target : bool, optional
            If True, transformation is added to the target variable steps
            If False, transformation is added to the exogenous variable steps,
            by default True

        Raises
        ------
        ValueError
            (1) `transform` is not of one of the allowed values.
            (2) `transform` is not of one of the allowed types.
        """
        type_ = "Target" if target else "Exogenous"
        self.logger.info(f"Set up transformation for {type_} variable(s).")

        if isinstance(transform, str):
            transform_dict = {
                "box-cox": BoxCoxTransformer(),
                "log": LogTransformer(),
                "sqrt": SqrtTransformer(),
                "exp": ExponentTransformer(),
                "cos": CosineTransformer(),
            }

            if transform not in transform_dict:
                raise ValueError(
                    f"{type_} transformation method '{transform}' not allowed."
                )

        else:
            raise ValueError(
                f"{type_} transformation method '{type(transform)}' is not of allowed type."
            )

        if target:
            transformer = transform_dict[transform]
            self.transformer_steps_target.extend([("transformer", transformer)])
        else:
            transformer = ColumnwiseTransformer(transform_dict[transform])
            self.transformer_steps_exogenous.extend([("transformer", transformer)])

    def _scaling(
        self,
        scale_target: Optional[Union[str, int, float]],
        scale_exogenous: Optional[Union[str, int, float]],
        exogenous_present: TSExogenousPresent,
    ):
        # Scale target ----
        if scale_target:
            self._add_scaling_steps(scale=scale_target, target=True)

        # Scale Exogenous ----
        # Only add exogenous pipeline steps if exogenous variables are present.
        if exogenous_present == TSExogenousPresent.YES and scale_exogenous is not None:
            self._add_scaling_steps(scale=scale_exogenous, target=False)

    def _add_scaling_steps(self, scale: str, target: bool = True):
        """Scale the data.

        Parameters
        ----------
        scale : str
            The method to be used for scaling. Allowed values and corresponding scalers are:
                "zscore": StandardScaler(),
                "minmax": MinMaxScaler(),
                "maxabs": MaxAbsScaler(),
                "robust": RobustScaler(),
        target : bool, optional
            If True, scaling is added to the target variable steps
            If False, scaling is added to the exogenous variable steps,
            by default True

        Raises
        ------
        ValueError
            (1) `scale` is not of one of the allowed values.
            (2) `scale` is not of one of the allowed types.
        """
        type_ = "Target" if target else "Exogenous"
        self.logger.info(f"Set up scaling for {type_} variable(s).")

        if isinstance(scale, str):
            scale_dict = {
                "zscore": StandardScaler(),
                "minmax": MinMaxScaler(),
                "maxabs": MaxAbsScaler(),
                "robust": RobustScaler(),
            }

            if scale not in scale_dict:
                raise ValueError(f"{type_} scale method '{scale}' not allowed.")

            scaler = TabularToSeriesAdaptor(scale_dict[scale])

        else:
            raise ValueError(
                f"{type_} transformation method '{type(scale)}' is not of allowed type."
            )

        if target:
            self.transformer_steps_target.extend([("scaler", scaler)])
        else:
            self.transformer_steps_exogenous.extend([("scaler", scaler)])

    def _feature_engineering(
        self,
        fe_exogenous: Optional[list],
        exogenous_present: TSExogenousPresent,
    ):
        """Add feature engineering steps to the pipeline.
        NOTE: Only Applied to Reduced regression models (see note in Setup).
        But in these models, target feature engineering is done internal to the
        model. Hence target feature engineering is not done here.

        Parameters
        ----------
        fe_exogenous : Optional[list]
            Feature Engineering Transformer to apply to exogenous variables.
        exogenous_present : TSExogenousPresent
            TSExogenousPresent.YES if exogenous variables are present in the data,
            TSExogenousPresent.NO otherwise. Exogenous feature transformers are
            only added if this is set to TSExogenousPresent.YES.
        """
        # Transform exogenous variables ----
        # Only add exogenous pipeline steps if exogenous variables are present,
        # but this is an exception. We may not have exogenous variables, but
        # these could be created using fe_exogenous, so we do not explicitly check
        # for the presence of exogenous variables to add this to the pipeline.
        # if exogenous_present == TSExogenousPresent.YES and fe_exogenous is not None:
        if fe_exogenous is not None:
            self._add_feat_eng_steps(
                fe_exogenous=fe_exogenous,
                target=False,
            )

    def _add_feat_eng_steps(self, fe_exogenous: list, target: bool = True):
        """Add feature engineering steps for the data.

        Parameters
        ----------
        fe_exogenous : list
            Feature engineering transformations to be applied to exogenous variables.
        target : bool, optional
            If True, feature engineering steps are added to the target variable steps
            (This is not implemented yet - see note in _feature_engineering)
            If False, feature engineering steps are added to the exogenous variable steps,
            by default True

        Raises
        ------
        ValueError
            `fe_exogenous` is not of the correct type.
        """
        type_ = "Target" if target else "Exogenous"
        self.logger.info(f"Set up feature engineering for {type_} variable(s).")

        if not isinstance(fe_exogenous, list):
            raise ValueError(
                f"{type_} Feature Engineering input must be of a List or sktime transformers."
                f"You provided {fe_exogenous}"
            )

        if target:
            # This is not implemented yet - see note in _feature_engineering
            pass
        else:
            self.transformer_steps_exogenous.extend(fe_exogenous)

    # def _feature_selection(
    #     self,
    #     feature_selection_method,
    #     feature_selection_estimator,
    #     n_features_to_select,
    # ):
    #     """Select relevant features."""
    #     self.logger.info("Set up feature selection.")
    #     # TODO: Maybe implement https://github.com/pycaret/pycaret/issues/2230
    #     # self.pipeline.steps.append(("feature_selection", feature_selector))

    # def _add_custom_pipeline(self, custom_pipeline):
    #     """Add custom transformers to the pipeline."""
    #     self.logger.info("Set up custom pipeline.")
    #     for name, estimator in normalize_custom_transformers(custom_pipeline):
    #         self.pipeline.steps.append((name, TransformerWrapper(estimator)))
