# Module: Classification & Regression
# Author: Mavs <m.524687@gmail.com>
# License: MIT

import time
import logging
import warnings
import numpy as np
import pandas as pd
from joblib.memory import Memory
from IPython.display import display
from typing import Optional, Union, Dict, List, Any

from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectFromModel,
    SequentialFeatureSelector,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
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

# Internal imports
from pycaret.internal.logging import get_logger
from pycaret.internal.pipeline import Pipeline as InternalPipeline
from pycaret.internal.pycaret_experiment.utils import MLUsecase, highlight_setup
from pycaret.internal.pycaret_experiment.supervised_experiment import (
    _SupervisedExperiment,
)
from pycaret.containers.models.classification import (
    get_all_model_containers as get_classifiers,
)
from pycaret.containers.models.regression import (
    get_all_model_containers as get_regressors,
)
from pycaret.internal.preprocess import (
    TransfomerWrapper,
    ExtractDateTimeFeatures,
    EmbedTextFeatures,
    RemoveMulticollinearity,
    RemoveOutliers,
    FixImbalancer,
    IterativeImputer,
)
from pycaret.internal.utils import (
    to_df,
    get_columns_to_stratify_by,
    df_shrink_dtypes,
    check_features_exist,
    normalize_custom_transformers,
)

warnings.filterwarnings("ignore")
LOGGER = get_logger()


class ClassRegExperiment(_SupervisedExperiment):
    def setup(
        self,
        data: pd.DataFrame,
        target: Union[int, str] = -1,
        train_size: float = 0.7,
        test_data: Optional[pd.DataFrame] = None,
        ordinal_features: Optional[Dict[str, list]] = None,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        date_features: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
        ignore_features: Optional[List[str]] = None,
        keep_features: Optional[List[str]] = None,
        preprocess: bool = True,
        imputation_type: str = "simple",
        numeric_imputation: str = "mean",
        categorical_imputation: str = "constant",
        iterative_imputation_iters: int = 5,
        numeric_iterative_imputer: Union[str, Any] = "lightgbm",
        categorical_iterative_imputer: Union[str, Any] = "lightgbm",
        text_features_method: str = "tf-idf",
        max_encoding_ohe: int = 5,
        encoding_method: Optional[Any] = None,
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        low_variance_threshold: float = 0,
        remove_multicollinearity: bool = False,
        multicollinearity_threshold: float = 0.9,
        bin_numeric_features: Optional[List[str]] = None,
        remove_outliers: bool = False,
        outliers_method: str = "iforest",
        outliers_threshold: float = 0.05,
        fix_imbalance: bool = False,
        fix_imbalance_method: Optional[Any] = None,
        transformation: bool = False,
        transformation_method: str = "yeo-johnson",
        normalize: bool = False,
        normalize_method: str = "zscore",
        pca: bool = False,
        pca_method: str = "linear",
        pca_components: Union[int, float] = 1.0,
        feature_selection: bool = False,
        feature_selection_method: str = "classic",
        feature_selection_estimator: Union[str, Any] = "lightgbm",
        n_features_to_select: int = 10,
        transform_target: bool = False,
        transform_target_method: str = "box-cox",
        custom_pipeline: Any = None,
        data_split_shuffle: bool = True,
        data_split_stratify: Union[bool, List[str]] = False,
        fold_strategy: Union[str, Any] = "stratifiedkfold",
        fold: int = 10,
        fold_shuffle: bool = False,
        fold_groups: Optional[Union[str, pd.DataFrame]] = None,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,
        html: bool = True,
        session_id: Optional[int] = None,
        system_log: Union[bool, logging.Logger] = True,
        log_experiment: bool = False,
        experiment_name: Optional[str] = None,
        log_plots: Union[bool, list] = False,
        log_profile: bool = False,
        log_data: bool = False,
        silent: bool = False,
        verbose: bool = True,
        memory: Union[bool, str, Memory] = True,
        profile: bool = False,
        profile_kwargs: Dict[str, Any] = None,
    ):
        # Setup initialization ===================================== >>

        runtime_start = time.time()

        # Define parameter attrs
        self.fold_shuffle_param = fold_shuffle
        self.fold_groups_param = fold_groups
        self.log_plots_param = log_plots

        self._initialize_setup(
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            html=html,
            session_id=session_id,
            system_log=system_log,
            log_experiment=log_experiment,
            experiment_name=experiment_name,
            memory=memory,
            verbose=verbose,
        )

        # Set up data ============================================== >>

        self.logger.info("Set up data.")
        data = to_df(data)

        # Checking target parameter
        if isinstance(target, str):
            if target not in data.columns:
                raise ValueError(
                    "Invalid value for the target parameter. "
                    f"Column {target} not found in the data."
                )
            self.target_param = target
        else:
            self.target_param = data.columns[target]

        if test_data is None:
            train, test = train_test_split(
                data,
                test_size=1 - train_size,
                stratify=get_columns_to_stratify_by(
                    X=data.drop(self.target_param, axis=1),
                    y=data[self.target_param],
                    stratify=data_split_stratify,
                ),
                random_state=self.seed,
                shuffle=data_split_shuffle,
            )
            self.data = pd.concat([train, test]).reset_index(drop=True)
            self.idx = [self.data.index[:len(train)], self.data.index[-len(test):]]

        else:  # test_data is provided
            self.data = pd.concat([data, test_data]).reset_index(drop=True)
            self.idx = [self.data.index[:len(data)], self.data.index[-len(test_data):]]

        # Set up folding strategy ================================== >>

        self.logger.info("Set up folding strategy.")

        allowed_fold_strategy = [
            "kfold",
            "stratifiedkfold",
            "groupkfold",
            "timeseries",
        ]

        if fold_strategy == "groupkfold":
            if fold_groups is None or len(fold_groups) == 0:
                raise ValueError(
                    "'groupkfold' fold_strategy requires 'fold_groups' "
                    "to be a non-empty array-like object."
                )
        elif fold_strategy not in allowed_fold_strategy:
            raise ValueError(
                "Invalid value for the fold_strategy parameter. "
                f"Choose from: {', '.join(allowed_fold_strategy)}."
            )

        if isinstance(fold_groups, str):
            if fold_groups not in self.X.columns:
                raise ValueError(
                    f"Column {fold_groups} used for fold_groups "
                    f"is not present in the dataset."
                )

        if fold_groups is not None:
            if isinstance(fold_groups, str):
                self.fold_groups_param = self.X[fold_groups]
            if pd.isna(fold_groups).any():
                raise ValueError(f"fold_groups cannot contain NaNs.")

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

        # Prepare pipeline ========================================= >>

        if self.log_plots_param is True:
            if self._ml_usecase == MLUsecase.CLASSIFICATION:
                self.log_plots_param = ["residuals", "error", "feature"]
            else:
                self.log_plots_param = ["auc", "confusion_matrix", "feature"]
        elif isinstance(self.log_plots_param, list):
            for i in self.log_plots_param:
                if i not in self._available_plots:
                    raise ValueError(
                        f"Incorrect value for log_plots '{i}'. Possible values "
                        f"are: {', '.join(self._available_plots.keys())}."
                    )

        # Check transform_target_method
        allowed_transform_target_method = ["box-cox", "yeo-johnson"]
        if transform_target_method not in allowed_transform_target_method:
            raise ValueError(
                "Invalid value for the transform_target_method parameter. "
                f"Choose from: {', '.join(allowed_transform_target_method)}."
            )
        self.transform_target_param = transform_target
        self.transform_target_method = transform_target_method

        # Data preparation ========================================= >>

        # Standardize dataframe types to save memory
        self.data = df_shrink_dtypes(self.data)

        # Features to be ignored (are not read by self.dataset, self.X, etc...)
        self._ign_cols = ignore_features or []

        # Ordinal features
        if ordinal_features:
            check_features_exist(ordinal_features.keys(), self.X)
            ordinal_features = ordinal_features
        else:
            ordinal_features = {}

        # Numerical features
        if numeric_features:
            check_features_exist(numeric_features, self.X)
            numeric_features = numeric_features
        else:
            numeric_features = list(self.X.select_dtypes(include="number").columns)

        # Date features
        if date_features:
            check_features_exist(date_features, self.X)
            date_features = date_features
        else:
            date_features = list(self.X.select_dtypes(include="datetime").columns)

        # Text features
        if text_features:
            check_features_exist(text_features, self.X)
            text_features = text_features
        else:
            text_features = []

        # Categorical features
        if categorical_features:
            check_features_exist(categorical_features, self.X)
            categorical_features = categorical_features
        else:
            # Default should exclude datetime and text columns
            categorical_features = [
                col
                for col in self.X.select_dtypes(include=["object", "category"]).columns
                if col not in date_features + text_features
            ]

        # Features to keep during all preprocessing
        keep_features = keep_features or []

        # Preprocessing ============================================ >>

        # Initialize empty pipeline
        self.pipeline = InternalPipeline(
            steps=[("placeholder", None)], memory=self.memory,
        )

        if preprocess:

            self.logger.info("Preparing preprocessing pipeline...")

            # Encode target variable =============================== >>

            if self.y.dtype.kind not in "ifu":
                self.pipeline.steps.append(
                    ("label_encoding", TransfomerWrapper(LabelEncoder()))
                )

            # Date feature engineering ============================= >>

            # TODO: Could be improved allowing the user to choose which features to add
            if date_features:
                self.logger.info("Extracting features from datetime columns")
                date_estimator = TransfomerWrapper(
                    transformer=ExtractDateTimeFeatures(), include=date_features,
                )

                self.pipeline.steps.append(
                    ("date_feature_extractor", date_estimator),
                )

            # Imputation =========================================== >>

            if self.data.isna().any().any():
                # Checking parameters
                num_dict = {"zero": "constant", "mean": "mean", "median": "median"}
                if numeric_imputation not in num_dict:
                    raise ValueError(
                        "Invalid value for the numeric_imputation parameter, "
                        f"got {numeric_imputation}. Possible values are "
                        f"{' '.join(num_dict)}."
                    )

                cat_dict = {"constant": "constant", "mode": "most_frequent"}
                if categorical_imputation not in cat_dict:
                    raise ValueError(
                        "Invalid value for the categorical_imputation "
                        f"parameter, got {categorical_imputation}. Possible "
                        f"values are {' '.join(cat_dict)}."
                    )

                if imputation_type == "simple":
                    self.logger.info("Setting up simple imputation")

                    num_estimator = TransfomerWrapper(
                        transformer=SimpleImputer(
                            strategy=num_dict[numeric_imputation], fill_value=0,
                        ),
                        include=numeric_features,
                    )
                    cat_estimator = TransfomerWrapper(
                        transformer=SimpleImputer(
                            strategy=cat_dict[categorical_imputation],
                            fill_value="not_available",
                        ),
                        include=categorical_features,
                    )
                    self.pipeline.steps.extend(
                        [
                            ("numerical_imputer", num_estimator),
                            ("categorical_imputer", cat_estimator),
                        ],
                    )
                elif imputation_type == "iterative":
                    self.logger.info("Setting up iterative imputation")

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
                        if self.X.columns[i] in categorical_features
                    ]

                    def get_prepare_estimator_for_categoricals_type(
                        estimator, estimators_dict
                    ):
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
                                numeric_iterative_imputer, estimators_dict=regressors
                            ),
                            cat_estimator_prepare_for_categoricals_type=get_prepare_estimator_for_categoricals_type(
                                categorical_iterative_imputer,
                                estimators_dict=classifiers,
                            ),
                        ),
                    )
                    self.pipeline.steps.extend(
                        [
                            ("imputer", imputer),
                        ],
                    )
                else:
                    raise ValueError(
                        "Invalid value for the imputation_type parameter, got "
                        f"{imputation_type}. Possible values are: simple, iterative."
                    )

            # Text embedding ======================================= >>

            if text_features:
                self.logger.info("Setting text embedding...")
                if text_features_method.lower() in ("bow", "tfidf", "tf-idf"):
                    embed_estimator = TransfomerWrapper(
                        transformer=EmbedTextFeatures(method=text_features_method),
                        include=text_features,
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

            # Encoding ============================================= >>

            self.logger.info("Setting up encoding")

            # Select columns for different encoding types
            one_hot_cols, rest_cols = [], []
            for col in categorical_features:
                n_unique = self.X[col].nunique()
                if n_unique == 2:
                    ordinal_features[col] = list(self.X[col].dropna().unique())
                elif n_unique <= max_encoding_ohe:
                    one_hot_cols.append(col)
                else:
                    rest_cols.append(col)

            if ordinal_features:
                self.logger.info("Setting up encoding of ordinal features")

                # Check provided features and levels are correct
                mapping = {}
                for key, value in ordinal_features.items():
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
                    if np.NaN not in mapping[key]:
                        mapping[key][np.NaN] = -1

                ord_estimator = TransfomerWrapper(
                    transformer=OrdinalEncoder(
                        mapping=[
                            {"col": k, "mapping": val} for k, val in mapping.items()
                        ],
                        handle_missing="return_nan",
                        handle_unknown="value",
                    ),
                    include=list(ordinal_features.keys()),
                )

                self.pipeline.steps.append(
                    ("ordinal_encoding", ord_estimator)
                )

            if categorical_features:
                self.logger.info("Setting up encoding of categorical features")

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

            # Polynomial features ================================== >>

            if polynomial_features:
                self.logger.info("Setting up polynomial features")
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

            # Low variance ========================================= >>

            if low_variance_threshold:
                self.logger.info("Setting up variance threshold")
                if low_variance_threshold < 0:
                    raise ValueError(
                        "Invalid value for the ignore_low_variance parameter. "
                        f"The value should be >0, got {low_variance_threshold}."
                    )
                else:
                    variance_estimator = TransfomerWrapper(
                        transformer=VarianceThreshold(low_variance_threshold),
                        exclude=keep_features,
                    )

                self.pipeline.steps.append(
                    ("low_variance", variance_estimator)
                )

            # Remove multicollinearity ============================= >>

            if remove_multicollinearity:
                self.logger.info("Setting up removing multicollinearity")
                if 0 > multicollinearity_threshold or multicollinearity_threshold > 1:
                    raise ValueError(
                        "Invalid value for the multicollinearity_threshold "
                        "parameter. Value should lie between 0 and 1, got "
                        f"{multicollinearity_threshold}."
                    )

                multicollinearity = TransfomerWrapper(
                    transformer=RemoveMulticollinearity(multicollinearity_threshold),
                    exclude=keep_features,
                )

                self.pipeline.steps.append(
                    ("remove_multicollinearity", multicollinearity)
                )

            # Bin numerical features =============================== >>

            if bin_numeric_features:
                self.logger.info("Setting up binning of numerical features")
                check_features_exist(bin_numeric_features, self.X)

                binning_estimator = TransfomerWrapper(
                    transformer=KBinsDiscretizer(encode="ordinal", strategy="kmeans"),
                    include=bin_numeric_features,
                )

                self.pipeline.steps.append(
                    ("bin_numeric_features", binning_estimator)
                )

            # Remove outliers ====================================== >>

            if remove_outliers:
                self.logger.info("Setting up removing outliers")
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

            # Balance the dataset ================================== >>

            if fix_imbalance:
                self.logger.info("Setting up imbalanced handling")
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

            # Transformation ======================================= >>

            if transformation:
                self.logger.info("Setting up column transformation")
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

            # Normalization ======================================== >>

            if normalize:
                self.logger.info("Setting up feature normalization")
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

            # PCA ================================================== >>

            if pca:
                self.logger.info("Setting up PCA")
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
                        transformer=pca_dict[pca_method], exclude=keep_features,
                    )
                else:
                    raise ValueError(
                        "Invalid value for the pca_method parameter, got "
                        f"{pca_method}. Possible values are: {' '.join(pca_dict)}."
                    )

                self.pipeline.steps.append(("pca", pca_estimator))

            # Feature selection ==================================== >>

            if feature_selection:
                self.logger.info("Setting up feature selection...")

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

                if feature_selection_method.lower() == "classic":
                    feature_selector = TransfomerWrapper(
                        transformer=SelectFromModel(
                            estimator=fs_estimator,
                            threshold=-np.inf,
                            max_features=n_features_to_select,
                        ),
                        exclude=keep_features,
                    )
                elif feature_selection_method.lower() == "sequential":
                    feature_selector = TransfomerWrapper(
                        transformer=SequentialFeatureSelector(
                            estimator=fs_estimator,
                            n_features_to_select=n_features_to_select,
                            n_jobs=self.n_jobs_param,
                        ),
                        exclude=keep_features,
                    )
                elif feature_selection_method.lower() == "boruta":
                    # TODO: Fix
                    feature_selector = TransfomerWrapper(
                        transformer=BorutaPy(
                            estimator=fs_estimator, n_estimators="auto",
                        ),
                        exclude=keep_features,
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

        # Custom transformers ====================================== >>

        if custom_pipeline:
            self.logger.info("Setting up custom pipeline")
            for name, estimator in normalize_custom_transformers(custom_pipeline):
                self.pipeline.steps.append(
                    (name, TransfomerWrapper(estimator))
                )

        # Remove placeholder step
        if len(self.pipeline) > 1:
            self.pipeline.steps.pop(0)

        self.pipeline.fit(self.X_train, self.y_train)

        self.logger.info(f"Finished creating preprocessing pipeline.")
        self.logger.info(f"Pipeline: {self.pipeline}")

        # Final display ============================================ >>

        self.logger.info("Creating final display dataframe.")

        if isinstance(numeric_iterative_imputer, str):
            num_imputer = numeric_iterative_imputer
        else:
            num_imputer = numeric_iterative_imputer.__class__.__name__

        if isinstance(categorical_iterative_imputer, str):
            cat_imputer = categorical_iterative_imputer
        else:
            cat_imputer = categorical_iterative_imputer.__class__.__name__

        display_container = [
            ["session_id", self.seed],
            ["Target", self.target_param],
            ["Target type", "Regression"],
            ["Data shape", self.data.shape],
            ["Train data shape", self.train.shape],
            ["Test data shape", self.test.shape],
            ["Ordinal features", len(ordinal_features)],
            ["Numerical features", len(numeric_features)],
            ["Categorical features", len(categorical_features)],
            ["Date features", len(date_features)],
            ["Text features", len(text_features)],
            ["Ignored features", len(self._ign_cols)],
            ["Kept features", len(keep_features)],
            ["Missing Values", self.data.isna().sum().sum()],
            ["Preprocess", preprocess],
        ]

        if preprocess:
            display_container.extend(
                [
                    ["Imputation type", imputation_type],
                    ["Numeric imputation", numeric_imputation],
                    ["Categorical imputation", categorical_imputation],
                    ["Iterative imputation iterations", iterative_imputation_iters],
                    ["Numeric iterative imputer", num_imputer],
                    ["Categorical iterative imputer", cat_imputer],
                    ["Text features embedding method", text_features_method],
                    ["Maximum one-hot encoding", max_encoding_ohe],
                    ["Encoding method", encoding_method],
                    ["Polynomial features", polynomial_features],
                    ["Polynomial degree", polynomial_degree],
                    ["Low variance threshold", low_variance_threshold],
                    ["Remove multicollinearity", remove_multicollinearity],
                    ["Multicollinearity threshold", multicollinearity_threshold],
                    ["Remove outliers", remove_outliers],
                    ["Outliers threshold", outliers_threshold],
                ]
            )

            if self._ml_usecase == MLUsecase.CLASSIFICATION:
                display_container.extend(
                    [
                        ["Fix imbalance", fix_imbalance],
                        ["Fix imbalance method", fix_imbalance_method],
                    ]
                )

            display_container.extend(
                [
                    ["Transformation", transformation],
                    ["Transformation method", transformation_method],
                    ["Normalize", normalize],
                    ["Normalize method", normalize_method],
                    ["PCA", pca],
                    ["PCA method", pca_method],
                    ["PCA components", pca_components],
                    ["Feature selection", feature_selection],
                    ["Feature selection method", feature_selection_method],
                    ["Feature selection estimator", feature_selection_estimator],
                    ["Number of features selected", n_features_to_select],
                ]
            )

            if self._ml_usecase == MLUsecase.REGRESSION:
                display_container.extend(
                    [
                        ["Transform target", transform_target],
                        ["Transform target method", transform_target_method],
                    ]
                )

            display_container.extend(
                [
                    ["Custom pipeline", "Yes" if custom_pipeline else "No"],
                    ["Fold Generator", self.fold_generator.__class__.__name__],
                    ["Fold Number", fold],
                    ["CPU Jobs", self.n_jobs_param],
                    ["Use GPU", self.gpu_param],
                    ["Log Experiment", self.logging_param],
                    ["Experiment Name", self.exp_name_log],
                    ["USI", self.USI],
                ]
            )

        self.display_container = [
            pd.DataFrame(display_container, columns=["Description", "Value"])
        ]
        self.logger.info(f"Setup display_container: {self.display_container[0]}")
        if self.verbose:
            pd.set_option("display.max_rows", 100)
            display(self.display_container[0].style.apply(highlight_setup))
            pd.reset_option("display.max_rows")  # Reset option

        # Wrap-up ================================================== >>

        # Create a profile report
        self._profile(profile, profile_kwargs)

        # Define models and metrics
        self._all_models, self._all_models_internal = self._get_models()
        self._all_metrics = self._get_metrics()

        runtime = np.array(time.time() - runtime_start).round(2)
        self._set_up_mlflow(runtime, log_data, log_profile)

        self._setup_ran = True
        self.logger.info(f"setup() successfully completed in {runtime}s...............")

        return self
