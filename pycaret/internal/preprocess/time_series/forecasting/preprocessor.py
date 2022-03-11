import numpy as np
import pandas as pd

from sktime.transformations.series.impute import Imputer

from pycaret.internal.utils import (
    to_df,
    get_columns_to_stratify_by,
    df_shrink_dtypes,
    check_features_exist,
    normalize_custom_transformers,
)


class TSForecastingPreprocessor:
    """Class for preprocessing Time Series Forecasting Experiments."""

    def _imputation(self, numeric_imputation, target: bool = True):
        """Perform simple imputation of missing values."""
        type = "Target" if target else "Exogenous"
        self.logger.info(f"Set up imputation for {type} variable(s).")

        if isinstance(numeric_imputation, str):
            allowed_values = [
                "drift",
                "linear",
                "nearest",
                "constant",
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
                    f"{target} Imputation Type '{numeric_imputation}' not allowed."
                )
            num_estimator = Imputer(method=numeric_imputation, random_state=self.seed)
        elif isinstance(numeric_imputation, (int, float)):
            num_estimator = Imputer(method="constant", value=numeric_imputation)
        else:
            raise ValueError(
                f"{target} Imputation Type '{numeric_imputation}' is not of allowed type."
            )

        if target:
            self.pipe_steps_target.extend([("numerical_imputer", num_estimator)])
        else:
            self.pipe_steps_exogenous.extend([("numerical_imputer", num_estimator)])

    # def _transformation(self, transformation_method):
    #     """Power transform the data to be more Gaussian-like."""
    #     self.logger.info("Set up column transformation.")

    #     if transformation_method == "yeo-johnson":
    #         transformation_estimator = PowerTransformer(
    #             method="yeo-johnson", standardize=False, copy=True
    #         )
    #     elif transformation_method == "quantile":
    #         transformation_estimator = QuantileTransformer(
    #             random_state=self.seed,
    #             output_distribution="normal",
    #         )
    #     else:
    #         raise ValueError(
    #             "Invalid value for the transformation_method parameter. "
    #             "The value should be either yeo-johnson or quantile, "
    #             f"got {transformation_method}."
    #         )

    #     self.pipeline.steps.append(
    #         ("transformation", TransfomerWrapper(transformation_estimator))
    #     )

    # def _normalization(self, normalize_method):
    #     """Scale the features."""
    #     self.logger.info("Set up feature normalization.")

    #     norm_dict = {
    #         "zscore": StandardScaler(),
    #         "minmax": MinMaxScaler(),
    #         "maxabs": MaxAbsScaler(),
    #         "robust": RobustScaler(),
    #     }
    #     if normalize_method in norm_dict:
    #         normalize_estimator = TransfomerWrapper(norm_dict[normalize_method])
    #     else:
    #         raise ValueError(
    #             "Invalid value for the normalize_method parameter, got "
    #             f"{normalize_method}. Possible values are: {' '.join(norm_dict)}."
    #         )

    #     self.pipeline.steps.append(("normalize", normalize_estimator))

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
    #         self.pipeline.steps.append((name, TransfomerWrapper(estimator)))
