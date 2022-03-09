# Author: Mavs (m.524687@gmail.com)
# License: MIT

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.model_selection import (
    train_test_split,
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    KBinsDiscretizer,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    f_classif,
    f_regression,
    SelectKBest,
    SelectFromModel,
    SequentialFeatureSelector,
)
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from imblearn.over_sampling import SMOTE

# Own modules
from pycaret.internal.pycaret_experiment.utils import MLUsecase
from pycaret.containers.models.classification import (
    get_all_model_containers as get_classifiers,
)
from pycaret.containers.models.regression import (
    get_all_model_containers as get_regressors,
)
from pycaret.internal.preprocess.iterative_imputer import IterativeImputer
from pycaret.internal.preprocess.transformers import (
    TransfomerWrapper,
    ExtractDateTimeFeatures,
    DropImputer,
    EmbedTextFeatures,
    RemoveMulticollinearity,
    RemoveOutliers,
    FixImbalancer,
)
from pycaret.internal.utils import (
    to_df,
    get_columns_to_stratify_by,
    df_shrink_dtypes,
    check_features_exist,
    normalize_custom_transformers,
)


class Preprocessor:
    """Class for all standard transformation steps."""

    def _prepare_dataset(self, data):
        """Convert the dataset to a pd.DataFrame."""
        self.logger.info("Set up data.")
        self.data = df_shrink_dtypes(to_df(data))

    def _prepare_target(self, target):
        """Assign the target column."""
        self.logger.info("Set up target column.")
        if isinstance(target, str):
            if target not in self.data.columns:
                raise ValueError(
                    "Invalid value for the target parameter. "
                    f"Column {target} not found in the data."
                )
            self.target_param = target
        else:
            self.target_param = self.data.columns[target]

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
            train, test = train_test_split(
                self.data,
                test_size=1 - train_size,
                stratify=get_columns_to_stratify_by(self.X, self.y, data_split_stratify),
                random_state=self.seed,
                shuffle=data_split_shuffle,
            )
            self.data = pd.concat([train, test]).reset_index(drop=True)
            self.idx = [self.data.index[:len(train)], self.data.index[-len(test):]]

        else:  # test_data is provided
            self.data = pd.concat([self.data, test_data]).reset_index(drop=True)
            self.idx = [self.data.index[:len(self.data)], self.data.index[-len(test_data):]]

    def _prepare_folds(self, fold_strategy, fold, fold_shuffle, fold_groups):
        """Assign the fold strategy."""
        self.logger.info("Set up folding strategy.")
        allowed_fold_strategy = ["kfold", "stratifiedkfold", "groupkfold", "timeseries"]

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

        if isinstance(fold_groups, str):
            if fold_groups in self.X.columns:
                if pd.isna(fold_groups).any():
                    raise ValueError(f"The 'fold_groups' column cannot contain NaNs.")
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
        self.pipeline.steps.append(("label_encoding", TransfomerWrapper(LabelEncoder())))

    def _date_feature_engineering(self):
        """Convert date features to numerical values."""
        self.logger.info("Set up date feature engineering.")
        # TODO: Could be improved allowing the user to choose which features to add
        date_estimator = TransfomerWrapper(
            transformer=ExtractDateTimeFeatures(), include=self._fxs["Date"]
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
                num_estimator = TransfomerWrapper(
                    transformer=DropImputer(columns=self._fxs["Numeric"])
                )
            elif numeric_imputation.lower() == "knn":
                num_estimator = TransfomerWrapper(
                    transformer=KNNImputer(),
                    include=self._fxs["Numeric"],
                )
            elif numeric_imputation.lower() in num_dict:
                num_estimator = TransfomerWrapper(
                    SimpleImputer(strategy=num_dict[numeric_imputation.lower()]),
                    include=self._fxs["Numeric"],
                )
            else:
                raise ValueError(
                    "Invalid value for the numeric_imputation parameter, got "
                    f"{numeric_imputation}. Choose from: drop, mean, median, mode, knn."
                )
        else:
            num_estimator = TransfomerWrapper(
                SimpleImputer(strategy="constant", fill_value=numeric_imputation),
                include=self._fxs["Numeric"],
            )

        if categorical_imputation.lower() == "drop":
            cat_estimator = TransfomerWrapper(
                transformer=DropImputer(columns=self._fxs["Categorical"])
            )
        elif categorical_imputation.lower() == "mode":
            cat_estimator = TransfomerWrapper(
                transformer=SimpleImputer(strategy="most_frequent"),
                include=self._fxs["Categorical"],
            )
        else:
            cat_estimator = TransfomerWrapper(
                SimpleImputer(strategy="constant", fill_value=categorical_imputation)
                , include=self._fxs["Categorical"],
            )

        self.pipeline.steps.extend(
            [
                ("numerical_imputer", num_estimator),
                ("categorical_imputer", cat_estimator)
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
            for k, v in get_regressors(self).items()
            if not v.is_special
        }
        # Dict of all classifier models available
        classifiers = {
            k: v
            for k, v in get_classifiers(self).items()
            if not v.is_special
        }

        if isinstance(numeric_iterative_imputer, str):
            if numeric_iterative_imputer not in regressors:
                raise ValueError(
                    "Invalid value for the numeric_iterative_imputer "
                    f"parameter, got {numeric_iterative_imputer}. "
                    f"Allowed estimators are: {', '.join(regressors)}."
                )
            numeric_iterative_imputer = regressors[
                numeric_iterative_imputer
            ].class_def(**regressors[numeric_iterative_imputer].args)
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
            elif isinstance(
                    estimator, estimators_dict["catboost"].class_def
            ):
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
                                estimators_dict.get(
                                    "gbc", estimators_dict["xgboost"]
                                ),
                            ).class_def,
                    ),
            ):
                return "ordinal"
            else:
                return "one_hot"

        imputer = TransfomerWrapper(
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
            embed_estimator = TransfomerWrapper(
                transformer=EmbedTextFeatures(method=text_features_method),
                include=self._fxs["Text"],
            )
        else:
            raise ValueError(
                "Invalid value for the text_features_method "
                "parameter. Choose between bow (Bag of Words) "
                f"or tf-idf, got {text_features_method}."
            )

        self.pipeline.steps.append(
            ("text_embedding", embed_estimator)
        )

    def _encoding(self, max_encoding_ohe, encoding_method):
        """Encode categorical columns."""
        # Select columns for different encoding types
        one_hot_cols, rest_cols = [], []
        for col in self._fxs["Categorical"]:
            n_unique = self.X[col].nunique(dropna=False)
            if n_unique == 2:
                self._fxs["Ordinal"][col] = list(sorted(self.X[col].unique()))
            elif n_unique <= max_encoding_ohe:
                one_hot_cols.append(col)
            else:
                rest_cols.append(col)

        if self._fxs["Ordinal"]:
            self.logger.info("Set up encoding of ordinal features.")

            # Check provided features and levels are correct
            mapping = {}
            for key, value in self._fxs["Ordinal"].items():
                if self.X[key].nunique() != len(value):
                    raise ValueError(
                        "The levels passed to the ordinal_features parameter "
                        "doesn't match with the levels in the dataset."
                    )
                for elem in value:
                    if elem not in self.X[key].unique():
                        raise ValueError(
                            f"Feature {key} doesn't contain the {elem} element."
                        )
                mapping[key] = {v: i for i, v in enumerate(value)}

                # Encoder always needs mapping of NaN value
                mapping[key].setdefault(np.NaN, -1)

            ord_estimator = TransfomerWrapper(
                transformer=OrdinalEncoder(
                    mapping=[
                        {"col": k, "mapping": val} for k, val in mapping.items()
                    ],
                    handle_missing="return_nan",
                    handle_unknown="value",
                ),
                include=list(self._fxs["Ordinal"].keys()),
            )

            self.pipeline.steps.append(
                ("ordinal_encoding", ord_estimator)
            )

        if self._fxs["Categorical"]:
            self.logger.info("Set up encoding of categorical features.")
    
            if len(one_hot_cols) > 0:
                onehot_estimator = TransfomerWrapper(
                    transformer=OneHotEncoder(
                        use_cat_names=True,
                        handle_missing="return_nan",
                        handle_unknown="value",
                    ),
                    include=one_hot_cols,
                )
    
                self.pipeline.steps.append(
                    ("onehot_encoding", onehot_estimator)
                )
    
            # Encode the rest of the categorical columns
            if len(rest_cols) > 0:
                if not encoding_method:
                    encoding_method = LeaveOneOutEncoder(
                        handle_missing="return_nan",
                        handle_unknown="value",
                        random_state=self.seed,
                    )
    
                rest_estimator = TransfomerWrapper(
                    transformer=encoding_method, include=rest_cols,
                )
    
                self.pipeline.steps.append(
                    ("rest_encoding", rest_estimator)
                )

    def _polynomial_features(self, polynomial_degree):
        """Create polynomial features from the existing ones."""
        self.logger.info("Set up polynomial features.")

        polynomial = TransfomerWrapper(
            transformer=PolynomialFeatures(
                degree=polynomial_degree,
                interaction_only=False,
                include_bias=False,
                order="C",
            ),
        )

        self.pipeline.steps.append(
            ("polynomial_features", polynomial)
        )

    def _low_variance(self, low_variance_threshold):
        """Drop features with too low variance."""
        self.logger.info("Set up variance threshold.")

        if low_variance_threshold < 0:
            raise ValueError(
                "Invalid value for the ignore_low_variance parameter. "
                f"The value should be >0, got {low_variance_threshold}."
            )
        else:
            variance_estimator = TransfomerWrapper(
                transformer=VarianceThreshold(low_variance_threshold),
                exclude=self._fxs["Keep"],
            )

        self.pipeline.steps.append(
            ("low_variance", variance_estimator)
        )

    def _remove_multicollinearity(self, multicollinearity_threshold):
        """Drop features that are collinear with other features."""
        self.logger.info("Set up removing multicollinearity.")

        if 0 > multicollinearity_threshold or multicollinearity_threshold > 1:
            raise ValueError(
                "Invalid value for the multicollinearity_threshold "
                "parameter. Value should lie between 0 and 1, got "
                f"{multicollinearity_threshold}."
            )

        multicollinearity = TransfomerWrapper(
            transformer=RemoveMulticollinearity(multicollinearity_threshold),
            exclude=self._fxs["Keep"],
        )

        self.pipeline.steps.append(
            ("remove_multicollinearity", multicollinearity)
        )

    def _bin_numerical_features(self, bin_numeric_features):
        """Bin numerical features to 5 clusters."""
        self.logger.info("Set up binning of numerical features.")

        check_features_exist(bin_numeric_features, self.X)
        binning_estimator = TransfomerWrapper(
            transformer=KBinsDiscretizer(encode="ordinal", strategy="kmeans"),
            include=bin_numeric_features,
        )

        self.pipeline.steps.append(
            ("bin_numeric_features", binning_estimator)
        )

    def _remove_outliers(self, outliers_method, outliers_threshold):
        """Remove outliers from the dataset."""
        self.logger.info("Set up removing outliers.")

        if outliers_method.lower() not in ("iforest", "ee", "lof"):
            raise ValueError(
                "Invalid value for the outliers_method parameter, "
                f"got {outliers_method}. Possible values are: "
                "'iforest', 'ee' or 'lof'."
            )

        outliers = TransfomerWrapper(
            RemoveOutliers(
                method=outliers_method, threshold=outliers_threshold,
            ),
        )

        self.pipeline.steps.append(("remove_outliers", outliers))

    def _balance(self, fix_imbalance_method):
        """Balance the classes in the target column."""
        self.logger.info("Set up imbalanced handling.")

        if fix_imbalance_method is None:
            balance_estimator = FixImbalancer(SMOTE())
        elif not hasattr(fix_imbalance_method, "fit_resample"):
            raise ValueError(
                "Invalid value for the fix_imbalance_method parameter. "
                "The provided value must be a imblearn estimator, got "
                f"{fix_imbalance_method.__class__.__name_}."
            )
        else:
            balance_estimator = FixImbalancer(fix_imbalance_method)

        balance_estimator = TransfomerWrapper(balance_estimator)
        self.pipeline.steps.append(("balance", balance_estimator))

    def _transformation(self, transformation_method):
        """Power transform the data to be more Gaussian-like."""
        self.logger.info("Set up column transformation.")

        if transformation_method == "yeo-johnson":
            transformation_estimator = PowerTransformer(
                method="yeo-johnson", standardize=False, copy=True
            )
        elif transformation_method == "quantile":
            transformation_estimator = QuantileTransformer(
                random_state=self.seed, output_distribution="normal",
            )
        else:
            raise ValueError(
                "Invalid value for the transformation_method parameter. "
                "The value should be either yeo-johnson or quantile, "
                f"got {transformation_method}."
            )

        self.pipeline.steps.append(
            ("transformation", TransfomerWrapper(transformation_estimator))
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
            normalize_estimator = TransfomerWrapper(norm_dict[normalize_method])
        else:
            raise ValueError(
                "Invalid value for the normalize_method parameter, got "
                f"{normalize_method}. Possible values are: {' '.join(norm_dict)}."
            )

        self.pipeline.steps.append(("normalize", normalize_estimator))

    def _pca(self, pca_method, pca_components):
        """Apply Principal Component Analysis."""
        self.logger.info("Set up PCA.")

        if pca_components <= 0:
            raise ValueError(
                "Invalid value for the pca_components parameter. "
                f"The value should be >0, got {pca_components}."
            )
        elif pca_components <= 1:
            pca_components = int(pca_components * self.X.shape[1])
        elif pca_components <= self.X.shape[1]:
            pca_components = int(pca_components)
        else:
            raise ValueError(
                "Invalid value for the pca_components parameter. "
                "The value should be smaller than the number of "
                f"features, got {pca_components}."
            )

        pca_dict = {
            "linear": PCA(n_components=pca_components),
            "kernel": KernelPCA(n_components=pca_components, kernel="rbf"),
            "incremental": IncrementalPCA(n_components=pca_components),
        }
        if pca_method in pca_dict:
            pca_estimator = TransfomerWrapper(
                transformer=pca_dict[pca_method], exclude=self._fxs["Keep"],
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
            func = get_classifiers
        else:
            func = get_regressors

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
            feature_selector = TransfomerWrapper(
                transformer=SelectKBest(score_func=func, k=n_features_to_select),
                exclude=self._fxs["Keep"],
            )
        elif feature_selection_method.lower() == "classic":
            feature_selector = TransfomerWrapper(
                transformer=SelectFromModel(
                    estimator=fs_estimator,
                    threshold=-np.inf,
                    max_features=n_features_to_select,
                ),
                exclude=self._fxs["Keep"],
            )
        elif feature_selection_method.lower() == "sequential":
            feature_selector = TransfomerWrapper(
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
                "'classic' or 'boruta'."
            )

        self.pipeline.steps.append(
            ("feature_selection", feature_selector)
        )

    def _add_custom_pipeline(self, custom_pipeline):
        """Add custom transformers to the pipeline."""
        self.logger.info("Set up custom pipeline.")
        for name, estimator in normalize_custom_transformers(custom_pipeline):
            self.pipeline.steps.append(
                (name, TransfomerWrapper(estimator))
            )
