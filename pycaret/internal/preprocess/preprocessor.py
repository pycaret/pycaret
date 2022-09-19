# Author: Mavs (m.524687@gmail.com)
# License: MIT

from copy import deepcopy

import numpy as np
import pandas as pd
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTEN,
    SMOTENC,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    AllKNN,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from pycaret.containers.models import (
    get_all_class_model_containers,
    get_all_reg_model_containers,
)
from pycaret.internal.preprocess.iterative_imputer import IterativeImputer
from pycaret.internal.preprocess.transformers import (
    DropImputer,
    EmbedTextFeatures,
    ExtractDateTimeFeatures,
    FixImbalancer,
    GroupFeatures,
    RareCategoryGrouping,
    RemoveMulticollinearity,
    RemoveOutliers,
    TargetTransformer,
    TransformerWrapper,
    TransformerWrapperWithInverse,
)
from pycaret.utils.constants import SEQUENCE
from pycaret.utils.generic import (
    MLUsecase,
    check_features_exist,
    df_shrink_dtypes,
    get_columns_to_stratify_by,
    normalize_custom_transformers,
    to_df,
    to_series,
)


class Preprocessor:
    """Class for all standard transformation steps."""

    def _prepare_dataset(self, X, y=None):
        """Prepare the input data.

        Convert X and y to pandas (if not already) and perform standard
        compatibility checks (dimensions, length, indices, etc...).
        From https://github.com/tvdboom/ATOM/blob/master/atom/basetransformer.py#L211

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        """
        self.logger.info("Set up data.")

        # Make copy to not overwrite mutable arguments
        X = to_df(deepcopy(X))

        # Prepare target column
        if isinstance(y, (list, tuple, np.ndarray, pd.Series)):
            if not isinstance(y, pd.Series):
                # Check that y is one-dimensional
                ndim = np.array(y).ndim
                if ndim != 1:
                    raise ValueError(f"y should be one-dimensional, got ndim={ndim}.")

                # Check X and y have the same number of rows
                if len(X) != len(y):
                    raise ValueError(
                        "X and y don't have the same number of rows,"
                        f" got len(X)={len(X)} and len(y)={len(y)}."
                    )

                y = to_series(y, index=X.index)

            elif not X.index.equals(y.index):
                raise ValueError("X and y don't have the same indices!")

        elif isinstance(y, str):
            if y not in X.columns:
                raise ValueError(
                    "Invalid value for the target parameter. "
                    f"Column {y} not found in the data."
                )

            X, y = X.drop(y, axis=1), X[y]

        elif isinstance(y, int):
            X, y = X.drop(X.columns[y], axis=1), X[X.columns[y]]

        else:  # y=None
            return df_shrink_dtypes(X)

        # Check that y has no missing values
        if y.isna().any():
            raise ValueError(
                f"{y.isna().sum()} missing values found in the target column: "
                f"{y.name}. To proceed, remove the respective rows from the data. "
            )

        return df_shrink_dtypes(
            X.merge(y.to_frame(), left_index=True, right_index=True)
        )

    def _set_index(self, df):
        """Assign an index to the dataframe."""
        self.logger.info("Set up index.")

        target = df.columns[-1]

        if getattr(self, "index", True) is True:  # True gets caught by isinstance(int)
            return df
        elif self.index is False:
            df = df.reset_index(drop=True)
        elif isinstance(self.index, int):
            if -df.shape[1] <= self.index <= df.shape[1]:
                df = df.set_index(df.columns[self.index], drop=True)
            else:
                raise ValueError(
                    f"Invalid value for the index parameter. Value {self.index} "
                    f"is out of range for a dataset with {df.shape[1]} columns."
                )
        elif isinstance(self.index, str):
            if self.index in df:
                df = df.set_index(self.index, drop=True)
            else:
                raise ValueError(
                    "Invalid value for the index parameter. "
                    f"Column {self.index} not found in the dataset."
                )

        if df.index.name == target:
            raise ValueError(
                "Invalid value for the index parameter. The index column "
                f"can not be the same as the target column, got {target}."
            )

        return df

    def _prepare_train_test(
        self,
        train_size,
        test_data,
        data_split_stratify,
        data_split_shuffle,
    ):
        """Make the train/test split."""
        self.logger.info("Set up train/test split.")

        if test_data is None:
            if isinstance(self.index, SEQUENCE):
                if len(self.index) != len(self.data):
                    raise ValueError(
                        "Invalid value for the index parameter. Length of "
                        f"index ({len(self.index)}) doesn't match that of "
                        f"the dataset ({len(self.data)})."
                    )
                self.data.index = self.index

            # self.data is already prepared here
            train, test = train_test_split(
                self.data,
                test_size=1 - train_size,
                stratify=get_columns_to_stratify_by(
                    self.X, self.y, data_split_stratify
                ),
                random_state=self.seed,
                shuffle=data_split_shuffle,
            )
            self.data = self._set_index(pd.concat([train, test]))
            self.idx = [self.data.index[: len(train)], self.data.index[-len(test) :]]

        else:  # test_data is provided
            test_data = self._prepare_dataset(test_data, self.target_param)

            if isinstance(self.index, SEQUENCE):
                if len(self.index) != len(self.data) + len(test_data):
                    raise ValueError(
                        "Invalid value for the index parameter. Length of "
                        f"index ({len(self.index)}) doesn't match that of "
                        f"the data sets ({len(self.data) + len(test_data)})."
                    )
                self.data.index = self.index[: len(self.data)]
                test_data.index = self.index[-len(test_data) :]

            self.data = self._set_index(pd.concat([self.data, test_data]))
            self.idx = [
                self.data.index[: -len(test_data)],
                self.data.index[-len(test_data) :],
            ]

    def _prepare_column_types(
        self,
        ordinal_features,
        numeric_features,
        categorical_features,
        date_features,
        text_features,
        ignore_features,
        keep_features,
    ):
        """Assign the types of every column in the dataset."""
        self.logger.info("Assigning column types.")

        # Features to be ignored (are not read by self.dataset, self.X, etc...)
        self._fxs["Ignore"] = ignore_features or []

        # Ordinal features
        if ordinal_features:
            check_features_exist(ordinal_features.keys(), self.X)
            self._fxs["Ordinal"] = ordinal_features
        else:
            self._fxs["Ordinal"] = {}

        # Numerical features
        if numeric_features:
            check_features_exist(numeric_features, self.X)
            self._fxs["Numeric"] = numeric_features
        else:
            self._fxs["Numeric"] = list(self.X.select_dtypes(include="number").columns)

        # Date features
        if date_features:
            check_features_exist(date_features, self.X)
            self._fxs["Date"] = date_features
        else:
            self._fxs["Date"] = list(self.X.select_dtypes(include="datetime").columns)

        # Text features
        if text_features:
            check_features_exist(text_features, self.X)
            self._fxs["Text"] = text_features

        # Categorical features
        if categorical_features:
            check_features_exist(categorical_features, self.X)
            self._fxs["Categorical"] = categorical_features
        else:
            # Default should exclude datetime and text columns
            self._fxs["Categorical"] = [
                col
                for col in self.X.select_dtypes(include=["object", "category"]).columns
                if col not in self._fxs["Date"] + self._fxs["Text"]
            ]

        # Features to keep during all preprocessing
        self._fxs["Keep"] = keep_features or []

    def _prepare_folds(self, fold_strategy, fold, fold_shuffle, fold_groups):
        """Assign the fold strategy."""
        self.logger.info("Set up folding strategy.")
        allowed_fold_strategy = ["kfold", "stratifiedkfold", "groupkfold", "timeseries"]

        if isinstance(fold_strategy, str):
            if fold_strategy == "groupkfold":
                if fold_groups is None or len(fold_groups) == 0:
                    raise ValueError(
                        "Invalid value for the fold_strategy parameter. 'groupkfold' "
                        "requires 'fold_groups' to be a non-empty array-like object."
                    )
            elif fold_strategy not in allowed_fold_strategy:
                raise ValueError(
                    "Invalid value for the fold_strategy parameter. "
                    f"Choose from: {', '.join(allowed_fold_strategy)}."
                )

        if fold_strategy == "timeseries" or isinstance(fold_strategy, TimeSeriesSplit):
            if fold_shuffle:
                raise ValueError(
                    "Invalid value for the fold_strategy parameter. 'timeseries' "
                    "requires 'data_split_shuffle' to be False as it can lead to "
                    "unexpected data split."
                )

        if isinstance(fold_groups, str):
            if fold_groups in self.X.columns:
                if pd.isna(self.X[fold_groups]).any():
                    raise ValueError("The 'fold_groups' column cannot contain NaNs.")
                else:
                    self.fold_groups_param = self.X[fold_groups]
            else:
                raise ValueError(
                    "Invalid value for the fold_groups parameter. "
                    f"Column {fold_groups} is not present in the dataset."
                )

        if fold_strategy == "kfold":
            self.fold_generator = KFold(
                fold,
                shuffle=fold_shuffle,
                random_state=self.seed if fold_shuffle else None,
            )
        elif fold_strategy == "stratifiedkfold":
            self.fold_generator = StratifiedKFold(
                fold,
                shuffle=fold_shuffle,
                random_state=self.seed if fold_shuffle else None,
            )
        elif fold_strategy == "groupkfold":
            self.fold_generator = GroupKFold(fold)
        elif fold_strategy == "timeseries":
            self.fold_generator = TimeSeriesSplit(fold)
        else:
            self.fold_generator = fold_strategy

    def _encode_target_column(self):
        """Add LabelEncoder to the pipeline."""
        self.logger.info("Set up label encoding.")
        self.pipeline.steps.append(
            ("label_encoding", TransformerWrapperWithInverse(LabelEncoder()))
        )

    def _target_transformation(self, transformation_method):
        """Power transform the data to be more Gaussian-like."""
        self.logger.info("Set up target transformation.")

        if transformation_method == "yeo-johnson":
            transformation_estimator = PowerTransformer(
                method="yeo-johnson", standardize=False, copy=True
            )
        elif transformation_method == "quantile":
            transformation_estimator = QuantileTransformer(
                random_state=self.seed,
                output_distribution="normal",
            )
        else:
            raise ValueError(
                "Invalid value for the transform_target_method parameter. "
                "The value should be either yeo-johnson or quantile, "
                f"got {transformation_method}."
            )

        self.pipeline.steps.append(
            (
                "target_transformation",
                TransformerWrapperWithInverse(
                    TargetTransformer(transformation_estimator)
                ),
            )
        )

    def _date_feature_engineering(self, create_date_columns):
        """Convert date features to numerical values."""
        self.logger.info("Set up date feature engineering.")
        # TODO: Could be improved allowing the user to choose which features to add
        date_estimator = TransformerWrapper(
            transformer=ExtractDateTimeFeatures(create_date_columns),
            include=self._fxs["Date"],
        )
        self.pipeline.steps.append(
            ("date_feature_extractor", date_estimator),
        )

    def _simple_imputation(self, numeric_imputation, categorical_imputation):
        """Perform simple imputation of missing values."""
        self.logger.info("Set up simple imputation.")

        # Numerical imputation
        num_dict = {"mode": "most_frequent", "mean": "mean", "median": "median"}
        if isinstance(numeric_imputation, str):
            if numeric_imputation.lower() == "drop":
                num_estimator = TransformerWrapper(
                    transformer=DropImputer(columns=self._fxs["Numeric"])
                )
            elif numeric_imputation.lower() == "knn":
                num_estimator = TransformerWrapper(
                    transformer=KNNImputer(),
                    include=self._fxs["Numeric"],
                )
            elif numeric_imputation.lower() in num_dict:
                num_estimator = TransformerWrapper(
                    SimpleImputer(strategy=num_dict[numeric_imputation.lower()]),
                    include=self._fxs["Numeric"],
                )
            else:
                raise ValueError(
                    "Invalid value for the numeric_imputation parameter, got "
                    f"{numeric_imputation}. Choose from: drop, mean, median, mode, knn."
                )
        else:
            num_estimator = TransformerWrapper(
                SimpleImputer(strategy="constant", fill_value=numeric_imputation),
                include=self._fxs["Numeric"],
            )

        if categorical_imputation.lower() == "drop":
            cat_estimator = TransformerWrapper(
                transformer=DropImputer(columns=self._fxs["Categorical"])
            )
        elif categorical_imputation.lower() == "mode":
            cat_estimator = TransformerWrapper(
                transformer=SimpleImputer(strategy="most_frequent"),
                include=self._fxs["Categorical"],
            )
        else:
            cat_estimator = TransformerWrapper(
                SimpleImputer(strategy="constant", fill_value=categorical_imputation),
                include=self._fxs["Categorical"],
            )

        self.pipeline.steps.extend(
            [
                ("numerical_imputer", num_estimator),
                ("categorical_imputer", cat_estimator),
            ],
        )

    def _iterative_imputation(
        self,
        iterative_imputation_iters,
        numeric_iterative_imputer,
        categorical_iterative_imputer,
    ):
        """Perform iterative imputation of missing values."""
        self.logger.info("Set up iterative imputation.")

        # Dict of all regressor models available
        regressors = {
            k: v
            for k, v in get_all_reg_model_containers(self).items()
            if not v.is_special
        }
        # Dict of all classifier models available
        classifiers = {
            k: v
            for k, v in get_all_class_model_containers(self).items()
            if not v.is_special
        }

        if isinstance(numeric_iterative_imputer, str):
            if numeric_iterative_imputer not in regressors:
                raise ValueError(
                    "Invalid value for the numeric_iterative_imputer "
                    f"parameter, got {numeric_iterative_imputer}. "
                    f"Allowed estimators are: {', '.join(regressors)}."
                )
            numeric_iterative_imputer = regressors[numeric_iterative_imputer].class_def(
                **regressors[numeric_iterative_imputer].args
            )
        elif not hasattr(numeric_iterative_imputer, "predict"):
            raise ValueError(
                "Invalid value for the numeric_iterative_imputer "
                "parameter. The provided estimator does not adhere "
                "to sklearn's API."
            )

        if isinstance(categorical_iterative_imputer, str):
            if categorical_iterative_imputer not in classifiers:
                raise ValueError(
                    "Invalid value for the categorical_iterative_imputer "
                    "parameter, got {categorical_iterative_imputer}. "
                    f"Allowed estimators are: {', '.join(classifiers)}."
                )
            categorical_iterative_imputer = classifiers[
                categorical_iterative_imputer
            ].class_def(**classifiers[categorical_iterative_imputer].args)
        elif not hasattr(categorical_iterative_imputer, "predict"):
            raise ValueError(
                "Invalid value for the categorical_iterative_imputer "
                "parameter. The provided estimator does not adhere "
                "to sklearn's API."
            )

        categorical_indices = [
            i
            for i in range(len(self.X.columns))
            if self.X.columns[i] in self._fxs["Categorical"]
        ]

        def get_prepare_estimator_for_categoricals_type(estimator, estimators_dict):
            # See pycaret.internal.preprocess.iterative_imputer
            fit_params = {}
            if not categorical_indices:
                return estimator, fit_params
            if isinstance(estimator, estimators_dict["lightgbm"].class_def):
                return "fit_params_categorical_feature"
            elif isinstance(estimator, estimators_dict["catboost"].class_def):
                return "params_cat_features"
            elif isinstance(
                estimator,
                (
                    estimators_dict["xgboost"].class_def,
                    estimators_dict["rf"].class_def,
                    estimators_dict["et"].class_def,
                    estimators_dict["dt"].class_def,
                    estimators_dict["ada"].class_def,
                    estimators_dict.get(
                        "gbr",
                        estimators_dict.get("gbc", estimators_dict["xgboost"]),
                    ).class_def,
                ),
            ):
                return "ordinal"
            else:
                return "one_hot"

        imputer = TransformerWrapper(
            transformer=IterativeImputer(
                num_estimator=numeric_iterative_imputer,
                cat_estimator=categorical_iterative_imputer,
                skip_complete=True,
                max_iter=iterative_imputation_iters,
                random_state=self.seed,
                categorical_indices=categorical_indices,
                num_estimator_prepare_for_categoricals_type=get_prepare_estimator_for_categoricals_type(
                    estimator=numeric_iterative_imputer,
                    estimators_dict=regressors,
                ),
                cat_estimator_prepare_for_categoricals_type=get_prepare_estimator_for_categoricals_type(
                    estimator=categorical_iterative_imputer,
                    estimators_dict=classifiers,
                ),
            ),
        )
        self.pipeline.steps.append(("iterative_imputer", imputer))

    def _text_embedding(self, text_features_method):
        """Convert text features to meaningful vectors."""
        self.logger.info("Set up text embedding.")

        if text_features_method.lower() in ("bow", "tfidf", "tf-idf"):
            embed_estimator = TransformerWrapper(
                transformer=EmbedTextFeatures(method=text_features_method),
                include=self._fxs["Text"],
            )
        else:
            raise ValueError(
                "Invalid value for the text_features_method "
                "parameter. Choose between bow (Bag of Words) "
                f"or tf-idf, got {text_features_method}."
            )

        self.pipeline.steps.append(("text_embedding", embed_estimator))

    def _encoding(self, max_encoding_ohe, encoding_method, rare_to_value, rare_value):
        """Encode categorical columns."""
        if rare_to_value:
            self.logger.info("Set up grouping of rare categories.")

            if rare_to_value < 0 or rare_to_value >= 1:
                raise ValueError(
                    "Invalid value for the rare_to_value parameter. "
                    f"The value must lie between 0 and 1, got {rare_to_value}."
                )

            rare_estimator = TransformerWrapper(
                transformer=RareCategoryGrouping(rare_to_value, rare_value),
                include=self._fxs["Categorical"],
            )

            self.pipeline.steps.append(("rare_category_grouping", rare_estimator))

            # To select the encoding type for every column,
            # we first need to run the grouping
            X_transformed = rare_estimator.fit_transform(self.X_train)
        else:
            X_transformed = self.X_train

        # Select columns for different encoding types
        one_hot_cols, rest_cols = [], []
        for name, column in X_transformed[self._fxs["Categorical"]].items():
            n_unique = column.nunique()
            if n_unique == 2:
                self._fxs["Ordinal"][name] = list(sorted(column.dropna().unique()))
            elif max_encoding_ohe < 0 or n_unique <= max_encoding_ohe:
                one_hot_cols.append(name)
            else:
                rest_cols.append(name)

        if self._fxs["Ordinal"]:
            self.logger.info("Set up encoding of ordinal features.")

            # Check provided features and levels are correct
            mapping = {}
            for key, value in self._fxs["Ordinal"].items():
                if self.X[key].nunique() != len(value):
                    self.logger.warning(
                        f"The number of classes passed to feature {key} in the "
                        f"ordinal_features parameter ({len(value)}) don't match "
                        f"with the number of classes in the data ({self.X[key].nunique()})."
                    )

                # Encoder always needs mapping of NaN value
                mapping[key] = {v: i for i, v in enumerate(value)}
                mapping[key].setdefault(np.NaN, -1)

            ord_estimator = TransformerWrapper(
                transformer=OrdinalEncoder(
                    mapping=[{"col": k, "mapping": val} for k, val in mapping.items()],
                    cols=list(
                        self._fxs["Ordinal"].keys()
                    ),  # Specify to not skip bool columns
                    handle_missing="return_nan",
                    handle_unknown="value",
                ),
                include=list(self._fxs["Ordinal"].keys()),
            )

            self.pipeline.steps.append(("ordinal_encoding", ord_estimator))

        if self._fxs["Categorical"]:
            self.logger.info("Set up encoding of categorical features.")

            if len(one_hot_cols) > 0:
                onehot_estimator = TransformerWrapper(
                    transformer=OneHotEncoder(
                        use_cat_names=True,
                        handle_missing="return_nan",
                        handle_unknown="value",
                    ),
                    include=one_hot_cols,
                )

                self.pipeline.steps.append(("onehot_encoding", onehot_estimator))

            # Encode the rest of the categorical columns
            if len(rest_cols) > 0:
                if not encoding_method:
                    encoding_method = LeaveOneOutEncoder(
                        handle_missing="return_nan",
                        handle_unknown="value",
                        random_state=self.seed,
                    )

                rest_estimator = TransformerWrapper(
                    transformer=encoding_method,
                    include=rest_cols,
                )

                self.pipeline.steps.append(("rest_encoding", rest_estimator))

    def _polynomial_features(self, polynomial_degree):
        """Create polynomial features from the existing ones."""
        self.logger.info("Set up polynomial features.")

        polynomial = TransformerWrapper(
            transformer=PolynomialFeatures(
                degree=polynomial_degree,
                interaction_only=False,
                include_bias=False,
                order="C",
            ),
        )

        self.pipeline.steps.append(("polynomial_features", polynomial))

    def _low_variance(self, low_variance_threshold):
        """Drop features with too low variance."""
        self.logger.info("Set up variance threshold.")

        if low_variance_threshold < 0:
            raise ValueError(
                "Invalid value for the ignore_low_variance parameter. "
                f"The value should be >0, got {low_variance_threshold}."
            )
        else:
            variance_estimator = TransformerWrapper(
                transformer=VarianceThreshold(low_variance_threshold),
                exclude=self._fxs["Keep"],
            )

        self.pipeline.steps.append(("low_variance", variance_estimator))

    def _group_features(self, group_features, group_names):
        """Get statistical properties of a group of features."""
        self.logger.info("Set up feature grouping.")

        # Convert a single group to sequence
        if np.array(group_features).ndim == 1:
            group_features = [group_features]

        if group_names:
            if isinstance(group_names, str):
                group_names = [group_names]

            if len(group_names) != len(group_features):
                raise ValueError(
                    "Invalid value for the group_names parameter. Length "
                    f"({len(group_names)}) does not match with length of "
                    f"group_features ({len(group_features)})."
                )

        grouping_estimator = TransformerWrapper(
            transformer=GroupFeatures(group_features, group_names),
            exclude=self._fxs["Keep"],
        )

        self.pipeline.steps.append(("group_features", grouping_estimator))

    def _remove_multicollinearity(self, multicollinearity_threshold):
        """Drop features that are collinear with other features."""
        self.logger.info("Set up removing multicollinearity.")

        if 0 > multicollinearity_threshold or multicollinearity_threshold > 1:
            raise ValueError(
                "Invalid value for the multicollinearity_threshold "
                "parameter. Value should lie between 0 and 1, got "
                f"{multicollinearity_threshold}."
            )

        multicollinearity = TransformerWrapper(
            transformer=RemoveMulticollinearity(multicollinearity_threshold),
            exclude=self._fxs["Keep"],
        )

        self.pipeline.steps.append(("remove_multicollinearity", multicollinearity))

    def _bin_numerical_features(self, bin_numeric_features):
        """Bin numerical features to 5 clusters."""
        self.logger.info("Set up binning of numerical features.")

        check_features_exist(bin_numeric_features, self.X)
        binning_estimator = TransformerWrapper(
            transformer=KBinsDiscretizer(encode="ordinal", strategy="kmeans"),
            include=bin_numeric_features,
        )

        self.pipeline.steps.append(("bin_numeric_features", binning_estimator))

    def _remove_outliers(self, outliers_method, outliers_threshold):
        """Remove outliers from the dataset."""
        self.logger.info("Set up removing outliers.")

        if outliers_method.lower() not in ("iforest", "ee", "lof"):
            raise ValueError(
                "Invalid value for the outliers_method parameter, "
                f"got {outliers_method}. Possible values are: "
                "'iforest', 'ee' or 'lof'."
            )

        outliers = TransformerWrapper(
            RemoveOutliers(
                method=outliers_method,
                threshold=outliers_threshold,
            ),
        )

        self.pipeline.steps.append(("remove_outliers", outliers))

    def _balance(self, fix_imbalance_method):
        """Balance the classes in the target column."""
        self.logger.info("Set up imbalanced handling.")

        strategies = dict(
            # clustercentroids=ClusterCentroids,  # Has no sample_indices_
            condensednearestneighbour=CondensedNearestNeighbour,
            editednearestneighborus=EditedNearestNeighbours,
            repeatededitednearestneighbours=RepeatedEditedNearestNeighbours,
            allknn=AllKNN,
            instancehardnessthreshold=InstanceHardnessThreshold,
            nearmiss=NearMiss,
            neighbourhoodcleaningrule=NeighbourhoodCleaningRule,
            onesidedselection=OneSidedSelection,
            randomundersampler=RandomUnderSampler,
            tomeklinks=TomekLinks,
            randomoversampler=RandomOverSampler,
            smote=SMOTE,
            smotenc=SMOTENC,
            smoten=SMOTEN,
            adasyn=ADASYN,
            borderlinesmote=BorderlineSMOTE,
            kmeanssmote=KMeansSMOTE,
            svmsmote=SVMSMOTE,
            smoteenn=SMOTEENN,
            smotetomek=SMOTETomek,
        )

        if isinstance(fix_imbalance_method, str):
            fix_imbalance_method = fix_imbalance_method.lower()
            if fix_imbalance_method not in strategies:
                raise ValueError(
                    "Invalid value for the strategy parameter, got "
                    f"{fix_imbalance_method}. Choose from: {', '.join(strategies)}."
                )
            balance_estimator = FixImbalancer(strategies[fix_imbalance_method]())
        elif not hasattr(fix_imbalance_method, "fit_resample"):
            raise TypeError(
                "Invalid value for the fix_imbalance_method parameter. "
                "The provided value must be a imblearn estimator, got "
                f"{fix_imbalance_method.__class__.__name_}."
            )
        else:
            balance_estimator = FixImbalancer(fix_imbalance_method)

        self.pipeline.steps.append(("balance", TransformerWrapper(balance_estimator)))

    def _transformation(self, transformation_method):
        """Power transform the data to be more Gaussian-like."""
        self.logger.info("Set up column transformation.")

        if transformation_method == "yeo-johnson":
            transformation_estimator = PowerTransformer(
                method="yeo-johnson", standardize=False, copy=True
            )
        elif transformation_method == "quantile":
            transformation_estimator = QuantileTransformer(
                random_state=self.seed,
                output_distribution="normal",
            )
        else:
            raise ValueError(
                "Invalid value for the transformation_method parameter. "
                "The value should be either yeo-johnson or quantile, "
                f"got {transformation_method}."
            )

        self.pipeline.steps.append(
            ("transformation", TransformerWrapper(transformation_estimator))
        )

    def _normalization(self, normalize_method):
        """Scale the features."""
        self.logger.info("Set up feature normalization.")

        norm_dict = {
            "zscore": StandardScaler(),
            "minmax": MinMaxScaler(),
            "maxabs": MaxAbsScaler(),
            "robust": RobustScaler(),
        }
        if normalize_method in norm_dict:
            normalize_estimator = TransformerWrapper(norm_dict[normalize_method])
        else:
            raise ValueError(
                "Invalid value for the normalize_method parameter, got "
                f"{normalize_method}. Possible values are: {' '.join(norm_dict)}."
            )

        self.pipeline.steps.append(("normalize", normalize_estimator))

    def _pca(self, pca_method, pca_components):
        """Apply Principal Component Analysis."""
        self.logger.info("Set up PCA.")

        pca_dict = {
            "linear": PCA(n_components=pca_components),
            "kernel": KernelPCA(n_components=pca_components, kernel="rbf"),
            "incremental": IncrementalPCA(n_components=pca_components),
        }
        if pca_method in pca_dict:
            pca_estimator = TransformerWrapper(
                transformer=pca_dict[pca_method],
                exclude=self._fxs["Keep"],
            )
        else:
            raise ValueError(
                "Invalid value for the pca_method parameter, got "
                f"{pca_method}. Possible values are: {' '.join(pca_dict)}."
            )

        self.pipeline.steps.append(("pca", pca_estimator))

    def _feature_selection(
        self,
        feature_selection_method,
        feature_selection_estimator,
        n_features_to_select,
    ):
        """Select relevant features."""
        self.logger.info("Set up feature selection.")

        if self._ml_usecase == MLUsecase.CLASSIFICATION:
            func = get_all_class_model_containers
        else:
            func = get_all_reg_model_containers

        models = {k: v for k, v in func(self).items() if not v.is_special}
        if isinstance(feature_selection_estimator, str):
            if feature_selection_estimator not in models:
                raise ValueError(
                    "Invalid value for the feature_selection_estimator "
                    f"parameter, got {feature_selection_estimator}. Allowed "
                    f"estimators are: {', '.join(models)}."
                )
            fs_estimator = models[feature_selection_estimator].class_def()
        elif not hasattr(feature_selection_estimator, "predict"):
            raise ValueError(
                "Invalid value for the feature_selection_estimator parameter. "
                "The provided estimator does not adhere to sklearn's API."
            )

        if feature_selection_method.lower() == "univariate":
            if self._ml_usecase == MLUsecase.CLASSIFICATION:
                func = f_classif
            else:
                func = f_regression
            feature_selector = TransformerWrapper(
                transformer=SelectKBest(score_func=func, k=n_features_to_select),
                exclude=self._fxs["Keep"],
            )
        elif feature_selection_method.lower() == "classic":
            feature_selector = TransformerWrapper(
                transformer=SelectFromModel(
                    estimator=fs_estimator,
                    threshold=-np.inf,
                    max_features=n_features_to_select,
                ),
                exclude=self._fxs["Keep"],
            )
        elif feature_selection_method.lower() == "sequential":
            feature_selector = TransformerWrapper(
                transformer=SequentialFeatureSelector(
                    estimator=fs_estimator,
                    n_features_to_select=n_features_to_select,
                    n_jobs=self.n_jobs_param,
                ),
                exclude=self._fxs["Keep"],
            )
        else:
            raise ValueError(
                "Invalid value for the feature_selection_method parameter, "
                f"got {feature_selection_method}. Possible values are: "
                "'classic', 'univariate' or 'sequential'."
            )

        self.pipeline.steps.append(("feature_selection", feature_selector))

    def _add_custom_pipeline(self, custom_pipeline, custom_pipeline_position):
        """Add custom transformers to the pipeline."""
        self.logger.info("Set up custom pipeline.")

        # Determine position to insert
        if custom_pipeline_position < 0:
            # -1 becomes last, etc...
            pos = len(self.pipeline.steps) + custom_pipeline_position + 1
        else:
            # +1 because of the placeholder
            pos = custom_pipeline_position + 1

        for name, estimator in normalize_custom_transformers(custom_pipeline):
            self.pipeline.steps.insert(pos, (name, TransformerWrapper(estimator)))
            pos += 1
