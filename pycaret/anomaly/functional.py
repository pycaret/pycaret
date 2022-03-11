import logging
import warnings
import pandas as pd
from joblib.memory import Memory

from pycaret.anomaly import AnomalyExperiment
from pycaret.internal.utils import check_if_global_is_not_none

from typing import List, Any, Union, Optional, Dict

warnings.filterwarnings("ignore")

_EXPERIMENT_CLASS = AnomalyExperiment
_CURRENT_EXPERIMENT = None
_CURRENT_EXPERIMENT_EXCEPTION = (
    "_CURRENT_EXPERIMENT global variable is not set. Please run setup() first."
)
_CURRENT_EXPERIMENT_DECORATOR_DICT = {
    "_CURRENT_EXPERIMENT": _CURRENT_EXPERIMENT_EXCEPTION
}


def setup(
    data,
    ordinal_features: Optional[Dict[str, list]] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    date_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
    ignore_features: Optional[List[str]] = None,
    keep_features: Optional[List[str]] = None,
    preprocess: bool = True,
    imputation_type: Optional[str] = "simple",
    numeric_imputation: str = "mean",
    categorical_imputation: str = "constant",
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
    transformation: bool = False,
    transformation_method: str = "yeo-johnson",
    normalize: bool = False,
    normalize_method: str = "zscore",
    pca: bool = False,
    pca_method: str = "linear",
    pca_components: Union[int, float] = 1.0,
    custom_pipeline: Any = None,
    n_jobs: Optional[int] = -1,
    use_gpu: bool = False,
    html: bool = True,
    session_id: Optional[int] = None,
    system_log: Union[bool, logging.Logger] = True,
    log_experiment: bool = False,
    experiment_name: Optional[str] = None,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    log_plots: Union[bool, list] = False,
    log_profile: bool = False,
    log_data: bool = False,
    silent: bool = False,
    verbose: bool = True,
    memory: Union[bool, str, Memory] = True,
    profile: bool = False,
    profile_kwargs: Dict[str, Any] = None,
):

    """
    This function initializes the training environment and creates the transformation
    pipeline. Setup function must be called before executing any other function. It
    takes one mandatory parameter: ``data``. All the other parameters are optional.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)


    data: dataframe-like
        Shape (n_samples, n_features), where n_samples is the number of samples and
        n_features is the number of features.


    ordinal_features: dict, default = None
        Categorical features to be encoded ordinally. For example, a categorical
        feature with 'low', 'medium', 'high' values where low < medium < high can
        be passed as ordinal_features = {'column_name' : ['low', 'medium', 'high']}.


    numeric_features: list of str, default = None
        If the inferred data types are not correct, the numeric_features param can
        be used to define the data types. It takes a list of strings with column
        names that are numeric.


    categorical_features: list of str, default = None
        If the inferred data types are not correct, the categorical_features param
        can be used to define the data types. It takes a list of strings with column
        names that are categorical.


    date_features: list of str, default = None
        If the inferred data types are not correct, the date_features param can be
        used to overwrite the data types. It takes a list of strings with column
        names that are DateTime.


    text_features: list of str, default = None
        Column names that contain a text corpus. If None, no text features are
        selected.


    ignore_features: list of str, default = None
        ignore_features param can be used to ignore features during preprocessing
        and model training. It takes a list of strings with column names that are
        to be ignored.


    keep_features: list of str, default = None
        keep_features param can be used to always keep specific features during
        preprocessing, i.e. these features are never dropped by any kind of
        feature selection. It takes a list of strings with column names that are
        to be kept.


    preprocess: bool, default = True
        When set to False, no transformations are applied except for train_test_split
        and custom transformations passed in ``custom_pipeline`` param. Data must be
        ready for modeling (no missing values, no dates, categorical data encoding),
        when preprocess is set to False.


    imputation_type: str or None, default = 'simple'
        The type of imputation to use. Can be either 'simple' or 'iterative'.
        If None, no imputation of missing values is performed.


    numeric_imputation: str, default = 'mean'
        Missing values in numeric features are imputed with 'mean' value of the feature
        in the training dataset. The other available option is 'median' or 'zero'.


    categorical_imputation: str, default = 'constant'
        Missing values in categorical features are imputed with a constant 'not_available'
        value. The other available option is 'mode'.


    text_features_method: str, default = "tf-idf"
        Method with which to embed the text features in the dataset. Choose
        between "bow" (Bag of Words - CountVectorizer) or "tf-idf" (TfidfVectorizer).
        Be aware that the sparse matrix output of the transformer is converted
        internally to its full array. This can cause memory issues for large
        text embeddings.


    max_encoding_ohe: int, default = 5
        Categorical columns with `max_encoding_ohe` or less unique values are
        encoded using OneHotEncoding. If more, the `encoding_method` estimator
        is used. Note that columns with exactly two classes are always encoded
        ordinally.


    encoding_method: category-encoders estimator, default = None
        A `category-encoders` estimator to encode the categorical columns
        with more than `max_encoding_ohe` unique values. If None,
        `category_encoders.leave_one_out.LeaveOneOutEncoder` is used.


    polynomial_features: bool, default = False
        When set to True, new features are derived using existing numeric features.


    polynomial_degree: int, default = 2
        Degree of polynomial features. For example, if an input sample is two dimensional
        and of the form [a, b], the polynomial features with degree = 2 are:
        [1, a, b, a^2, ab, b^2]. Ignored when ``polynomial_features`` is not True.


    low_variance_threshold: float or None, default = 0
        Remove features with a training-set variance lower than the provided
        threshold. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples. If
        None, skip this treansformation step.


    remove_multicollinearity: bool, default = False
        When set to True, features with the inter-correlations higher than the defined
        threshold are removed. When two features are highly correlated with each other,
        the feature that is less correlated with the target variable is removed. Only
        considers numeric features.


    multicollinearity_threshold: float, default = 0.9
        Threshold for correlated features. Ignored when ``remove_multicollinearity``
        is not True.


    bin_numeric_features: list of str, default = None
        To convert numeric features into categorical, bin_numeric_features parameter can
        be used. It takes a list of strings with column names to be discretized. It does
        so by using 'sturges' rule to determine the number of clusters and then apply
        KMeans algorithm. Original values of the feature are then replaced by the
        cluster label.


    remove_outliers: bool, default = False
        When set to True, outliers from the training data are removed using an
        Isolation Forest.


    outliers_method: str, default = "iforest"
        Method with which to remove outliers. Possible values are:
            - 'iforest': Uses sklearn's IsolationForest.
            - 'ee': Uses sklearn's EllipticEnvelope.
            - 'lof': Uses sklearn's LocalOutlierFactor.


    outliers_threshold: float, default = 0.05
        The percentage outliers to be removed from the dataset. Ignored
        when ``remove_outliers=False``.


    transformation: bool, default = False
        When set to True, it applies the power transform to make data more Gaussian-like.
        Type of transformation is defined by the ``transformation_method`` parameter.


    transformation_method: str, default = 'yeo-johnson'
        Defines the method for transformation. By default, the transformation method is
        set to 'yeo-johnson'. The other available option for transformation is 'quantile'.
        Ignored when ``transformation`` is not True.


    normalize: bool, default = False
        When set to True, it transforms the features by scaling them to a given
        range. Type of scaling is defined by the ``normalize_method`` parameter.


    normalize_method: str, default = 'zscore'
        Defines the method for scaling. By default, normalize method is set to 'zscore'
        The standard zscore is calculated as z = (x - u) / s. Ignored when ``normalize``
        is not True. The other options are:

        - minmax: scales and translates each feature individually such that it is in
          the range of 0 - 1.
        - maxabs: scales and translates each feature individually such that the
          maximal absolute value of each feature will be 1.0. It does not
          shift/center the data, and thus does not destroy any sparsity.
        - robust: scales and translates each feature according to the Interquartile
          range. When the dataset contains outliers, robust scaler often gives
          better results.


    pca: bool, default = False
        When set to True, dimensionality reduction is applied to project the data into
        a lower dimensional space using the method defined in ``pca_method`` parameter.


    pca_method: str, default = 'linear'
        Method with which to apply PCA. Possible values are:
            - 'linear': Uses Singular Value  Decomposition.
            - kernel: Dimensionality reduction through the use of RBF kernel.
            - incremental: Similar to 'linear', but more efficient for large datasets.


    pca_components: int or float, default = 1.0
        Number of components to keep. If >1, it selects that number of
        components. If <= 1, it selects that fraction of components from
        the original features. The value must be smaller than the number
        of original features. This parameter is ignored when `pca=False`.


    custom_pipeline: (str, transformer), list of (str, transformer) or dict, default = None
        Addidiotnal custom transformers. If passed, they are applied to the
        pipeline last, after all the build-in transformers.


    n_jobs: int, default = -1
        The number of jobs to run in parallel (for functions that supports parallel
        processing) -1 means using all processors. To run all functions on single
        processor set n_jobs to None.


    use_gpu: bool or str, default = False
        When set to True, it will use GPU for training with algorithms that support it,
        and fall back to CPU if they are unavailable. When set to 'force', it will only
        use GPU-enabled algorithms and raise exceptions when they are unavailable. When
        False, all algorithms are trained using CPU only.

        GPU enabled algorithms:

        - None at this moment.


    html: bool, default = True
        When set to False, prevents runtime display of monitor. This must be set to False
        when the environment does not support IPython. For example, command line terminal,
        Databricks Notebook, Spyder and other similar IDEs.


    session_id: int, default = None
        Controls the randomness of experiment. It is equivalent to 'random_state' in
        scikit-learn. When None, a pseudo random number is generated. This can be used
        for later reproducibility of the entire experiment.


    system_log: bool or logging.Logger, default = True
        Whether to save the system logging file (as logs.log). If the input
        already is a logger object, that one is used instead.


    log_experiment: bool, default = False
        When set to True, all metrics and parameters are logged on the ``MLFlow`` server.


    experiment_name: str, default = None
        Name of the experiment for logging. Ignored when ``log_experiment`` is not True.


    experiment_custom_tags: dict or None, default = None
        Dictionary of tag_name: String -> value: (String, but will be string-ified
        if not) passed to the mlflow.set_tags to add new custom tags for the experiment.


    log_plots: bool or list, default = False
        When set to True, certain plots are logged automatically in the ``MLFlow`` server.
        To change the type of plots to be logged, pass a list containing plot IDs. Refer
        to documentation of ``plot_model``. Ignored when ``log_experiment`` is not True.


    log_profile: bool, default = False
        When set to True, data profile is logged on the ``MLflow`` server as a html file.
        Ignored when ``log_experiment`` is not True.


    log_data: bool, default = False
        When set to True, dataset is logged on the ``MLflow`` server as a csv file.
        Ignored when ``log_experiment`` is not True.


    silent: bool, default = False
        Controls the confirmation input of data types when ``setup`` is executed. When
        executing in completely automated mode or on a remote kernel, this must be True.


    verbose: bool, default = True
        When set to False, Information grid is not printed.


    memory: str, bool or Memory, default=True
        Used to cache the fitted transformers of the pipeline.
            If False: No caching is performed.
            If True: A default temp directory is used.
            If str: Path to the caching directory.


    profile: bool, default = False
        When set to True, an interactive EDA report is displayed.


    profile_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the ProfileReport method used
        to create the EDA report. Ignored if ``profile`` is False.


    Returns:
        Global variables that can be changed using the ``set_config`` function.

    """

    exp = _EXPERIMENT_CLASS()
    set_current_experiment(exp)
    return exp.setup(
        data=data,
        ordinal_features=ordinal_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        date_features=date_features,
        text_features=text_features,
        ignore_features=ignore_features,
        keep_features=keep_features,
        preprocess=preprocess,
        imputation_type=imputation_type,
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
        text_features_method=text_features_method,
        max_encoding_ohe=max_encoding_ohe,
        encoding_method=encoding_method,
        polynomial_features=polynomial_features,
        polynomial_degree=polynomial_degree,
        low_variance_threshold=low_variance_threshold,
        remove_multicollinearity=remove_multicollinearity,
        multicollinearity_threshold=multicollinearity_threshold,
        bin_numeric_features=bin_numeric_features,
        remove_outliers=remove_outliers,
        outliers_method=outliers_method,
        outliers_threshold=outliers_threshold,
        transformation=transformation,
        transformation_method=transformation_method,
        normalize=normalize,
        normalize_method=normalize_method,
        pca=pca,
        pca_method=pca_method,
        pca_components=pca_components,
        custom_pipeline=custom_pipeline,
        n_jobs=n_jobs,
        use_gpu=use_gpu,
        html=html,
        session_id=session_id,
        system_log=system_log,
        log_experiment=log_experiment,
        experiment_name=experiment_name,
        experiment_custom_tags=experiment_custom_tags,
        log_plots=log_plots,
        log_profile=log_profile,
        log_data=log_data,
        silent=silent,
        verbose=verbose,
        memory=memory,
        profile=profile,
        profile_kwargs=profile_kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def create_model(
    model: Union[str, Any],
    fraction: float = 0.05,
    verbose: bool = True,
    fit_kwargs: Optional[dict] = None,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    **kwargs,
):

    """
    This function trains a given model from the model library. All available
    models can be accessed using the ``models`` function.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> knn = create_model('knn')


    model: str or scikit-learn compatible object
        ID of an model available in the model library or pass an untrained
        model object consistent with scikit-learn API. Estimators available
        in the model library (ID - Name):

        * 'abod' - Angle-base Outlier Detection
        * 'cluster' - Clustering-Based Local Outlier
        * 'cof' - Connectivity-Based Outlier Factor
        * 'histogram' - Histogram-based Outlier Detection
        * 'knn' - k-Nearest Neighbors Detector
        * 'lof' - Local Outlier Factor
        * 'svm' - One-class SVM detector
        * 'pca' - Principal Component Analysis
        * 'mcd' - Minimum Covariance Determinant
        * 'sod' - Subspace Outlier Detection
        * 'sos' - Stochastic Outlier Selection


    fraction: float, default = 0.05
        The amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. Used when fitting to define the threshold on
        the decision function.


    verbose: bool, default = True
        Status update is not printed when verbose is set to False.


    experiment_custom_tags: dict, default = None
        Dictionary of tag_name: String -> value: (String, but will be string-ified
        if not) passed to the mlflow.set_tags to add new custom tags for the experiment.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    **kwargs:
        Additional keyword arguments to pass to the estimator.


    Returns:
        Trained Model

    """

    return _CURRENT_EXPERIMENT.create_model(
        estimator=model,
        fraction=fraction,
        fit_kwargs=fit_kwargs,
        verbose=verbose,
        experiment_custom_tags=experiment_custom_tags,
        **kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def assign_model(
    model, transformation: bool = False, score: bool = True, verbose: bool = True
) -> pd.DataFrame:

    """
    This function assigns anomaly labels to the dataset for a given model.
    (1 = outlier, 0 = inlier).


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> knn = create_model('knn')
    >>> knn_df = assign_model(knn)


    model: scikit-learn compatible object
        Trained model object


    transformation: bool, default = False
        Whether to apply anomaly labels on the transformed dataset.


    score: bool, default = True
        Whether to show outlier score or not.


    verbose: bool, default = True
        Status update is not printed when verbose is set to False.


    Returns:
        pandas.DataFrame

    """
    return _CURRENT_EXPERIMENT.assign_model(
        model, transformation=transformation, score=score, verbose=verbose
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def plot_model(
    model,
    plot: str = "tsne",
    feature: Optional[str] = None,
    label: bool = False,
    scale: float = 1,
    save: bool = False,
    display_format: Optional[str] = None,
):

    """
    This function analyzes the performance of a trained model.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> knn = create_model('knn')
    >>> plot_model(knn, plot = 'tsne')


    model: scikit-learn compatible object
        Trained Model Object


    plot: str, default = 'tsne'
        List of available plots (ID - Name):

        * 'tsne' - t-SNE (3d) Dimension Plot
        * 'umap' - UMAP Dimensionality Plot


    feature: str, default = None
        Feature to be used as a hoverover tooltip and/or label when the ``label``
        param is set to True. When feature is None, first column of the dataset
        is used.


    label: bool, default = False
        Name of column to be used as data labels.


    scale: float, default = 1
        The resolution scale of the figure.


    save: bool, default = False
        When set to True, plot is saved in the current working directory.


    display_format: str, default = None
        To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.
        Currently, not all plots are supported.


    Returns:
        None

    """
    return _CURRENT_EXPERIMENT.plot_model(
        model,
        plot=plot,
        feature_name=feature,
        label=label,
        scale=scale,
        save=save,
        display_format=display_format,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def evaluate_model(
    model,
    feature: Optional[str] = None,
    fit_kwargs: Optional[dict] = None,
):

    """
    This function displays a user interface for analyzing performance of a trained
    model. It calls the ``plot_model`` function internally.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> knn = create_model('knn')
    >>> evaluate_model(knn)


    model: scikit-learn compatible object
        Trained model object


    feature: str, default = None
        Feature to be used as a hoverover tooltip and/or label when the ``label``
        param is set to True. When feature is None, first column of the dataset
        is used by default.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.

    Returns:
        None


    Warnings
    --------
    -   This function only works in IPython enabled Notebook.

    """

    return _CURRENT_EXPERIMENT.evaluate_model(
        estimator=model, feature_name=feature, fit_kwargs=fit_kwargs
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def tune_model(
    model,
    supervised_target: str,
    supervised_type: Optional[str] = None,
    supervised_estimator: Union[str, Any] = "lr",
    method: str = "drop",
    optimize: Optional[str] = None,
    custom_grid: Optional[List[int]] = None,
    fold: int = 10,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    round: int = 4,
    verbose: bool = True,
):

    """
    This function tunes the ``fraction`` parameter of a given model.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = juice)
    >>> tuned_knn = tune_model(model = 'knn', supervised_target = 'Purchase')


    model: str
        ID of an model available in the model library. Models that can be
        tuned in this function (ID - Model):

        * 'abod' - Angle-base Outlier Detection
        * 'cluster' - Clustering-Based Local Outlier
        * 'cof' - Connectivity-Based Outlier Factor
        * 'histogram' - Histogram-based Outlier Detection
        * 'knn' - k-Nearest Neighbors Detector
        * 'lof' - Local Outlier Factor
        * 'svm' - One-class SVM detector
        * 'pca' - Principal Component Analysis
        * 'mcd' - Minimum Covariance Determinant
        * 'sod' - Subspace Outlier Detection
        * 'sos' - Stochastic Outlier Selection


    supervised_target: str
        Name of the target column containing labels.


    supervised_type: str, default = None
        Type of task. 'classification' or 'regression'. Automatically inferred
        when None.


    supervised_estimator: str, default = None
        Classification (ID - Name):
            * 'lr' - Logistic Regression (Default)
            * 'knn' - K Nearest Neighbour
            * 'nb' - Naive Bayes
            * 'dt' - Decision Tree Classifier
            * 'svm' - SVM - Linear Kernel
            * 'rbfsvm' - SVM - Radial Kernel
            * 'gpc' - Gaussian Process Classifier
            * 'mlp' - Multi Level Perceptron
            * 'ridge' - Ridge Classifier
            * 'rf' - Random Forest Classifier
            * 'qda' - Quadratic Discriminant Analysis
            * 'ada' - Ada Boost Classifier
            * 'gbc' - Gradient Boosting Classifier
            * 'lda' - Linear Discriminant Analysis
            * 'et' - Extra Trees Classifier
            * 'xgboost' - Extreme Gradient Boosting
            * 'lightgbm' - Light Gradient Boosting
            * 'catboost' - CatBoost Classifier

        Regression (ID - Name):
            * 'lr' - Linear Regression (Default)
            * 'lasso' - Lasso Regression
            * 'ridge' - Ridge Regression
            * 'en' - Elastic Net
            * 'lar' - Least Angle Regression
            * 'llar' - Lasso Least Angle Regression
            * 'omp' - Orthogonal Matching Pursuit
            * 'br' - Bayesian Ridge
            * 'ard' - Automatic Relevance Determ.
            * 'par' - Passive Aggressive Regressor
            * 'ransac' - Random Sample Consensus
            * 'tr' - TheilSen Regressor
            * 'huber' - Huber Regressor
            * 'kr' - Kernel Ridge
            * 'svm' - Support Vector Machine
            * 'knn' - K Neighbors Regressor
            * 'dt' - Decision Tree
            * 'rf' - Random Forest
            * 'et' - Extra Trees Regressor
            * 'ada' - AdaBoost Regressor
            * 'gbr' - Gradient Boosting
            * 'mlp' - Multi Level Perceptron
            * 'xgboost' - Extreme Gradient Boosting
            * 'lightgbm' - Light Gradient Boosting
            * 'catboost' - CatBoost Regressor


    method: str, default = 'drop'
        When method set to drop, it will drop the outliers from training dataset.
        When 'surrogate', it uses decision function and label as a feature during
        training.


    optimize: str, default = None
        For Classification tasks:
            Accuracy, AUC, Recall, Precision, F1, Kappa (default = 'Accuracy')

        For Regression tasks:
            MAE, MSE, RMSE, R2, RMSLE, MAPE (default = 'R2')


    custom_grid: list, default = None
        By default, a pre-defined list of fraction values is iterated over to
        optimize the supervised objective. To overwrite default iteration,
        pass a list of fraction value to iterate over in custom_grid param.


    fold: int, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2.


    verbose: bool, default = True
        Status update is not printed when verbose is set to False.


    Returns:
        Trained Model with optimized ``fraction`` parameter.

    """

    return _CURRENT_EXPERIMENT.tune_model(
        model=model,
        supervised_target=supervised_target,
        supervised_type=supervised_type,
        supervised_estimator=supervised_estimator,
        method=method,
        optimize=optimize,
        custom_grid=custom_grid,
        fold=fold,
        fit_kwargs=fit_kwargs,
        groups=groups,
        round=round,
        verbose=verbose,
    )


# not using check_if_global_is_not_none on purpose
def predict_model(model, data: pd.DataFrame) -> pd.DataFrame:

    """
    This function generates anomaly labels on using a trained model.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> knn = create_model('knn')
    >>> knn_predictions = predict_model(model = knn, data = unseen_data)


    model: scikit-learn compatible object
        Trained Model Object.


    data : pandas.DataFrame
        Shape (n_samples, n_features) where n_samples is the number of samples and
        n_features is the number of features.


    Returns:
        pandas.DataFrame


    Warnings
    --------
    - The behavior of the predict_model is changed in version 2.1 without backward compatibility.
      As such, the pipelines trained using the version (<= 2.0), may not work for inference
      with version >= 2.1. You can either retrain your models with a newer version or downgrade
      the version for inference.


    """

    experiment = _CURRENT_EXPERIMENT
    if experiment is None:
        experiment = _EXPERIMENT_CLASS()

    return experiment.predict_model(estimator=model, data=data)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def deploy_model(
    model,
    model_name: str,
    authentication: dict,
    platform: str = "aws",
):

    """
    This function deploys the transformation pipeline and trained model on cloud.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> knn = create_model('knn')
    >>> # sets appropriate credentials for the platform as environment variables
    >>> import os
    >>> os.environ["AWS_ACCESS_KEY_ID"] = str("foo")
    >>> os.environ["AWS_SECRET_ACCESS_KEY"] = str("bar")
    >>> deploy_model(model = knn, model_name = 'knn-for-deployment', platform = 'aws', authentication = {'bucket' : 'S3-bucket-name'})


    Amazon Web Service (AWS) users:
        To deploy a model on AWS S3 ('aws'), the credentials have to be passed. The easiest way is to use environment
        variables in your local environment. Following information from the IAM portal of amazon console account
        are required:

        - AWS Access Key ID
        - AWS Secret Key Access


        More info: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#environment-variables


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

        - AZURE_STORAGE_CONNECTION_STRING (required as environment variable)

        More info: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json


    model: scikit-learn compatible object
        Trained model object


    model_name: str
        Name of model.


    authentication: dict
        Dictionary of applicable authentication tokens.

        When platform = 'aws':
        {'bucket' : 'S3-bucket-name', 'path': (optional) folder name under the bucket}

        When platform = 'gcp':
        {'project': 'gcp-project-name', 'bucket' : 'gcp-bucket-name'}

        When platform = 'azure':
        {'container': 'azure-container-name'}


    platform: str, default = 'aws'
        Name of the platform. Currently supported platforms: 'aws', 'gcp' and 'azure'.


    Returns:
        None

    """

    return _CURRENT_EXPERIMENT.deploy_model(
        model=model,
        model_name=model_name,
        authentication=authentication,
        platform=platform,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def save_model(
    model, model_name: str, model_only: bool = False, verbose: bool = True, **kwargs
):

    """
    This function saves the transformation pipeline and trained model object
    into the current working directory as a pickle file for later use.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> knn = create_model('knn')
    >>> save_model(knn, 'saved_knn_model')


    model: scikit-learn compatible object
        Trained model object


    model_name: str
        Name of the model.


    model_only: bool, default = False
        When set to True, only trained model object is saved instead of the
        entire pipeline.


    verbose: bool, default = True
        Success message is not printed when verbose is set to False.


    **kwargs:
        Additional keyword arguments to pass to joblib.dump().


    Returns:
        Tuple of the model object and the filename.

    """

    return _CURRENT_EXPERIMENT.save_model(
        model=model,
        model_name=model_name,
        model_only=model_only,
        verbose=verbose,
        **kwargs,
    )


# not using check_if_global_is_not_none on purpose
def load_model(
    model_name,
    platform: Optional[str] = None,
    authentication: Optional[Dict[str, str]] = None,
    verbose: bool = True,
):

    """
    This function loads a previously saved pipeline.


    Example
    -------
    >>> from pycaret.anomaly import load_model
    >>> saved_knn = load_model('saved_knn_model')


    model_name: str
        Name of the model.


    platform: str, default = None
        Name of the cloud platform. Currently supported platforms:
        'aws', 'gcp' and 'azure'.


    authentication: dict, default = None
        dictionary of applicable authentication tokens.

        when platform = 'aws':
        {'bucket' : 'Name of Bucket on S3', 'path': (optional) folder name under the bucket}

        when platform = 'gcp':
        {'project': 'gcp-project-name', 'bucket' : 'gcp-bucket-name'}

        when platform = 'azure':
        {'container': 'azure-container-name'}


    verbose: bool, default = True
        Success message is not printed when verbose is set to False.


    Returns:
        Trained Model

    """

    experiment = _CURRENT_EXPERIMENT
    if experiment is None:
        experiment = _EXPERIMENT_CLASS()

    return experiment.load_model(
        model_name=model_name,
        platform=platform,
        authentication=authentication,
        verbose=verbose,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def models(
    internal: bool = False,
    raise_errors: bool = True,
) -> pd.DataFrame:

    """
    Returns table of models available in the model library.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> all_models = models()


    internal: bool, default = False
        If True, will return extra columns and rows used internally.


    raise_errors: bool, default = True
        If False, will suppress all exceptions, ignoring models
        that couldn't be created.


    Returns:
        pandas.DataFrame

    """
    return _CURRENT_EXPERIMENT.models(internal=internal, raise_errors=raise_errors)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_logs(experiment_name: Optional[str] = None, save: bool = False) -> pd.DataFrame:

    """
    Returns a table of experiment logs. Only works when ``log_experiment``
    is True when initializing the ``setup`` function.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly,  log_experiment = True)
    >>> knn = create_model('knn')
    >>> exp_logs = get_logs()


    experiment_name: str, default = None
        When None current active run is used.


    save: bool, default = False
        When set to True, csv file is saved in current working directory.


    Returns:
        pandas.DataFrame

    """

    return _CURRENT_EXPERIMENT.get_logs(experiment_name=experiment_name, save=save)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_config(variable: str):

    """
    This function retrieves the global variables created when initializing the
    ``setup`` function. Following variables are accessible:

    - dataset: Transformed dataset
    - train: Transformed training set
    - test: Transformed test set
    - X: Transformed feature set
    - y: Transformed target column
    - X_train, X_test, y_train, y_test: Subsets of the train and test sets.
    - seed: random state set through session_id
    - pipeline: Transformation pipeline configured through setup
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - master_model_container: model storage container
    - display_container: results display container
    - exp_name_log: Name of experiment set through setup
    - logging_param: log_experiment param set through setup
    - log_plots_param: log_plots param set through setup
    - USI: Unique session ID parameter set through setup
    - gpu_param: use_gpu param configured through setup


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> X = get_config('X')


    Returns:
        Global variable

    """

    return _CURRENT_EXPERIMENT.get_config(variable=variable)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def set_config(variable: str, value):

    """
    This function resets the global variables. Following variables are
    accessible:

    - X: Transformed dataset (X)
    - data_before_preprocess: data before preprocessing
    - seed: random state set through session_id
    - prep_pipe: Transformation pipeline configured through setup
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - master_model_container: model storage container
    - display_container: results display container
    - exp_name_log: Name of experiment set through setup
    - logging_param: log_experiment param set through setup
    - log_plots_param: log_plots param set through setup
    - USI: Unique session ID parameter set through setup
    - gpu_param: use_gpu param configured through setup


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> set_config('seed', 123)


    Returns:
        None

    """

    return _CURRENT_EXPERIMENT.set_config(variable=variable, value=value)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def save_config(file_name: str):

    """
    This function save all global variables to a pickle file, allowing to
    later resume without rerunning the ``setup``.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> anomaly = get_data('anomaly')
    >>> from pycaret.anomaly import *
    >>> exp_name = setup(data = anomaly)
    >>> save_config('myvars.pkl')


    Returns:
        None

    """

    return _CURRENT_EXPERIMENT.save_config(file_name=file_name)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def load_config(file_name: str):

    """
    This function loads global variables from a pickle file into Python
    environment.


    Example
    -------
    >>> from pycaret.anomaly import load_config
    >>> load_config('myvars.pkl')


    Returns:
        Global variables

    """

    return _CURRENT_EXPERIMENT.load_config(file_name=file_name)


def get_outliers(
    data,
    model: Union[str, Any] = "knn",
    fraction: float = 0.05,
    fit_kwargs: Optional[dict] = None,
    preprocess: bool = True,
    imputation_type: str = "simple",
    iterative_imputation_iters: int = 5,
    categorical_features: Optional[List[str]] = None,
    categorical_imputation: str = "mode",
    categorical_iterative_imputer: Union[str, Any] = "lightgbm",
    ordinal_features: Optional[Dict[str, list]] = None,
    high_cardinality_features: Optional[List[str]] = None,
    high_cardinality_method: str = "frequency",
    numeric_features: Optional[List[str]] = None,
    numeric_imputation: str = "mean",  # method 'zero' added in pycaret==2.1
    numeric_iterative_imputer: Union[str, Any] = "lightgbm",
    date_features: Optional[List[str]] = None,
    ignore_features: Optional[List[str]] = None,
    normalize: bool = False,
    normalize_method: str = "zscore",
    transformation: bool = False,
    transformation_method: str = "yeo-johnson",
    handle_unknown_categorical: bool = True,
    unknown_categorical_method: str = "least_frequent",
    pca: bool = False,
    pca_method: str = "linear",
    pca_components: Union[int, float] = 1.0,
    low_variance_threshold: float = 0,
    combine_rare_levels: bool = False,
    rare_level_threshold: float = 0.10,
    bin_numeric_features: Optional[List[str]] = None,
    remove_multicollinearity: bool = False,
    multicollinearity_threshold: float = 0.9,
    remove_perfect_collinearity: bool = False,
    group_features: Optional[List[str]] = None,
    group_names: Optional[List[str]] = None,
    n_jobs: Optional[int] = -1,
    session_id: Optional[int] = None,
    system_log: Union[bool, logging.Logger] = True,
    log_experiment: bool = False,
    experiment_name: Optional[str] = None,
    log_plots: Union[bool, list] = False,
    log_profile: bool = False,
    log_data: bool = False,
    profile: bool = False,
    **kwargs,
) -> pd.DataFrame:

    """
    Callable from any external environment without requiring setup initialization.
    """
    exp = _EXPERIMENT_CLASS()
    exp.setup(
        data=data,
        preprocess=preprocess,
        imputation_type=imputation_type,
        iterative_imputation_iters=iterative_imputation_iters,
        categorical_features=categorical_features,
        categorical_imputation=categorical_imputation,
        categorical_iterative_imputer=categorical_iterative_imputer,
        ordinal_features=ordinal_features,
        high_cardinality_features=high_cardinality_features,
        high_cardinality_method=high_cardinality_method,
        numeric_features=numeric_features,
        numeric_imputation=numeric_imputation,
        numeric_iterative_imputer=numeric_iterative_imputer,
        date_features=date_features,
        ignore_features=ignore_features,
        normalize=normalize,
        normalize_method=normalize_method,
        transformation=transformation,
        transformation_method=transformation_method,
        handle_unknown_categorical=handle_unknown_categorical,
        unknown_categorical_method=unknown_categorical_method,
        pca=pca,
        pca_method=pca_method,
        pca_components=pca_components,
        low_variance_threshold=low_variance_threshold,
        combine_rare_levels=combine_rare_levels,
        rare_level_threshold=rare_level_threshold,
        bin_numeric_features=bin_numeric_features,
        remove_multicollinearity=remove_multicollinearity,
        multicollinearity_threshold=multicollinearity_threshold,
        remove_perfect_collinearity=remove_perfect_collinearity,
        group_features=group_features,
        group_names=group_names,
        n_jobs=n_jobs,
        html=False,
        session_id=session_id,
        system_log=system_log,
        log_experiment=log_experiment,
        experiment_name=experiment_name,
        log_plots=log_plots,
        log_profile=log_profile,
        log_data=log_data,
        silent=True,
        verbose=False,
        profile=profile,
    )

    c = exp.create_model(
        model=model,
        fraction=fraction,
        fit_kwargs=fit_kwargs,
        verbose=False,
        **kwargs,
    )
    return exp.assign_model(c, verbose=False)


def set_current_experiment(experiment: AnomalyExperiment):
    global _CURRENT_EXPERIMENT

    if not isinstance(experiment, AnomalyExperiment):
        raise TypeError(
            f"experiment must be a PyCaret AnomalyExperiment object, got {type(experiment)}."
        )
    _CURRENT_EXPERIMENT = experiment
