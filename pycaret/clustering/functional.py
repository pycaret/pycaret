import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from joblib.memory import Memory

from pycaret.clustering.oop import ClusteringExperiment
from pycaret.loggers.base_logger import BaseLogger
from pycaret.utils.constants import DATAFRAME_LIKE, SEQUENCE_LIKE
from pycaret.utils.generic import check_if_global_is_not_none

_EXPERIMENT_CLASS = ClusteringExperiment
_CURRENT_EXPERIMENT: Optional[ClusteringExperiment] = None
_CURRENT_EXPERIMENT_EXCEPTION = (
    "_CURRENT_EXPERIMENT global variable is not set. Please run setup() first."
)
_CURRENT_EXPERIMENT_DECORATOR_DICT = {
    "_CURRENT_EXPERIMENT": _CURRENT_EXPERIMENT_EXCEPTION
}


def setup(
    data: DATAFRAME_LIKE,
    index: Union[bool, int, str, SEQUENCE_LIKE] = False,
    ordinal_features: Optional[Dict[str, list]] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    date_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
    ignore_features: Optional[List[str]] = None,
    keep_features: Optional[List[str]] = None,
    preprocess: bool = True,
    create_date_columns: List[str] = ["day", "month", "year"],
    imputation_type: Optional[str] = "simple",
    numeric_imputation: str = "mean",
    categorical_imputation: str = "constant",
    text_features_method: str = "tf-idf",
    max_encoding_ohe: int = -1,
    encoding_method: Optional[Any] = None,
    rare_to_value: Optional[float] = None,
    rare_value: str = "rare",
    polynomial_features: bool = False,
    polynomial_degree: int = 2,
    low_variance_threshold: Optional[float] = 0,
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
    pca_components: Optional[Union[int, float, str]] = None,
    custom_pipeline: Optional[Any] = None,
    custom_pipeline_position: int = -1,
    n_jobs: Optional[int] = -1,
    use_gpu: bool = False,
    html: bool = True,
    session_id: Optional[int] = None,
    system_log: Union[bool, str, logging.Logger] = True,
    log_experiment: Union[bool, str, BaseLogger, List[Union[str, BaseLogger]]] = False,
    experiment_name: Optional[str] = None,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    log_plots: Union[bool, list] = False,
    log_profile: bool = False,
    log_data: bool = False,
    verbose: bool = True,
    memory: Union[bool, str, Memory] = True,
    profile: bool = False,
    profile_kwargs: Optional[Dict[str, Any]] = None,
):

    """

    This function initializes the training environment and creates the transformation
    pipeline. Setup function must be called before executing any other function. It
    takes one mandatory parameter: ``data``. All the other parameters are optional.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)


    data: dataframe-like
        Data set with shape (n_samples, n_features), where n_samples is the
        number of samples and n_features is the number of features. If data
        is not a pandas dataframe, it's converted to one using default column
        names.


    index: bool, int, str or sequence, default = False
        Handle indices in the `data` dataframe.
            - If False: Reset to RangeIndex.
            - If True: Keep the provided index.
            - If int: Position of the column to use as index.
            - If str: Name of the column to use as index.
            - If sequence: Array with shape=(n_samples,) to use as index.


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


    create_date_columns: list of str, default = ["day", "month", "year"]
        Columns to create from the date features. Note that created features
        with zero variance (e.g. the feature hour in a column that only contains
        dates) are ignored. Allowed values are datetime attributes from
        `pandas.Series.dt`. The datetime format of the feature is inferred
        automatically from the first non NaN value.


    imputation_type: str or None, default = 'simple'
        The type of imputation to use. Unsupervised learning only supports
        'imputation_type=simple'. If None, no imputation of missing values
        is performed.


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


    max_encoding_ohe: int, default = -1
        Categorical columns with `max_encoding_ohe` or less unique values are
        encoded using OneHotEncoding. If more, the `encoding_method` estimator
        is used. Note that columns with exactly two classes are always encoded
        ordinally. Set to below 0 to always use OneHotEncoding.


    encoding_method: category-encoders estimator, default = None
        A `category-encoders` estimator to encode the categorical columns
        with more than `max_encoding_ohe` unique values. If None,
        `category_encoders.leave_one_out.LeaveOneOutEncoder` is used.


    rare_to_value: float or None, default=None
        Minimum fraction of category occurrences in a categorical column.
        If a category is less frequent than `rare_to_value * len(X)`, it is
        replaced with the string in `rare_value`. Use this parameter to group
        rare categories before encoding the column. If None, ignores this step.


    rare_value: str, default="rare"
        Value with which to replace rare categories. Ignored when
        ``rare_to_value`` is None.


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
        None, skip this transformation step.


    remove_multicollinearity: bool, default = False
        When set to True, features with the inter-correlations higher than
        the defined threshold are removed. For each group, it removes all
        except the first feature.


    multicollinearity_threshold: float, default = 0.9
        Minimum absolute Pearson correlation to identify correlated
        features. The default value removes equal columns. Ignored when
        ``remove_multicollinearity`` is not True.


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
            - 'kernel': Dimensionality reduction through the use of RBF kernel.
            - 'incremental': Similar to 'linear', but more efficient for large datasets.


    pca_components: int, float, str or None, default = None
        Number of components to keep. This parameter is ignored when `pca=False`.
            - If None: All components are kept.
            - If int: Absolute number of components.
            - If float: Such an amount that the variance that needs to be explained
                        is greater than the percentage specified by `n_components`.
                        Value should lie between 0 and 1 (ony for pca_method='linear').
            - If "mle": Minka’s MLE is used to guess the dimension (ony for pca_method='linear').


    custom_pipeline: list of (str, transformer), dict or Pipeline, default = None
        Addidiotnal custom transformers. If passed, they are applied to the
        pipeline last, after all the build-in transformers.


    custom_pipeline_position: int, default = -1
        Position of the custom pipeline in the overal preprocessing pipeline.
        The default value adds the custom pipeline last.


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


    system_log: bool or str or logging.Logger, default = True
        Whether to save the system logging file (as logs.log). If the input
        is a string, use that as the path to the logging file. If the input
        already is a logger object, use that one instead.


    log_experiment: bool, default = False
        A (list of) PyCaret ``BaseLogger`` or str (one of 'mlflow', 'wandb')
        corresponding to a logger to determine which experiment loggers to use.
        Setting to True will use just MLFlow.
        If ``wandb`` (Weights & Biases) is installed, will also log there.


    experiment_name: str, default = None
        Name of the experiment for logging. Ignored when ``log_experiment`` is False.


    experiment_custom_tags: dict, default = None
        Dictionary of tag_name: String -> value: (String, but will be string-ified
        if not) passed to the mlflow.set_tags to add new custom tags for the experiment.


    log_plots: bool or list, default = False
        When set to True, certain plots are logged automatically in the ``MLFlow`` server.
        To change the type of plots to be logged, pass a list containing plot IDs. Refer
        to documentation of ``plot_model``. Ignored when ``log_experiment`` is False.


    log_profile: bool, default = False
        When set to True, data profile is logged on the ``MLflow`` server as a html file.
        Ignored when ``log_experiment`` is False.


    log_data: bool, default = False
        When set to True, dataset is logged on the ``MLflow`` server as a csv file.
        Ignored when ``log_experiment`` is False.


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
        index=index,
        ordinal_features=ordinal_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        date_features=date_features,
        text_features=text_features,
        ignore_features=ignore_features,
        keep_features=keep_features,
        preprocess=preprocess,
        create_date_columns=create_date_columns,
        imputation_type=imputation_type,
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
        text_features_method=text_features_method,
        max_encoding_ohe=max_encoding_ohe,
        encoding_method=encoding_method,
        rare_to_value=rare_to_value,
        rare_value=rare_value,
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
        custom_pipeline_position=custom_pipeline_position,
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
        verbose=verbose,
        memory=memory,
        profile=profile,
        profile_kwargs=profile_kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def create_model(
    model: Union[str, Any],
    num_clusters: int = 4,
    ground_truth: Optional[str] = None,
    round: int = 4,
    fit_kwargs: Optional[dict] = None,
    verbose: bool = True,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    **kwargs,
):

    """
    This function trains and evaluates the performance of a given model.
    Metrics evaluated can be accessed using the ``get_metrics`` function.
    Custom metrics can be added or removed using the ``add_metric`` and
    ``remove_metric`` function. All the available models can be accessed
    using the ``models`` function.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')


    model: str or scikit-learn compatible object
        ID of an model available in the model library or pass an untrained
        model object consistent with scikit-learn API. Models available
        in the model library (ID - Name):

        * 'kmeans' - K-Means Clustering
        * 'ap' - Affinity Propagation
        * 'meanshift' - Mean shift Clustering
        * 'sc' - Spectral Clustering
        * 'hclust' - Agglomerative Clustering
        * 'dbscan' - Density-Based Spatial Clustering
        * 'optics' - OPTICS Clustering
        * 'birch' - Birch Clustering
        * 'kmodes' - K-Modes Clustering


    num_clusters: int, default = 4
        The number of clusters to form.


    ground_truth: str, default = None
        ground_truth to be provided to evaluate metrics that require true labels.
        When None, such metrics are returned as 0.0. All metrics evaluated can
        be accessed using ``get_metrics`` function.


    round: int, default = 4
        Number of decimal places the metrics in the score grid will be rounded to.


    fit_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the fit method of the model.


    verbose: bool, default = True
        Status update is not printed when verbose is set to False.


    experiment_custom_tags: dict, default = None
        Dictionary of tag_name: String -> value: (String, but will be string-ified
        if not) passed to the mlflow.set_tags to add new custom tags for the experiment.


    **kwargs:
        Additional keyword arguments to pass to the estimator.


    Returns:
        Trained Model


    Warnings
    --------
    - ``num_clusters`` param not required for Affinity Propagation ('ap'),
      Mean shift ('meanshift'), Density-Based Spatial Clustering ('dbscan')
      and OPTICS Clustering ('optics').

    - When fit doesn't converge in Affinity Propagation ('ap') model, all
      datapoints are labelled as -1.

    - Noisy samples are given the label -1, when using Density-Based Spatial
      ('dbscan') or OPTICS Clustering ('optics').

    - OPTICS ('optics') clustering may take longer training times on large
      datasets.


    """

    return _CURRENT_EXPERIMENT.create_model(
        estimator=model,
        num_clusters=num_clusters,
        ground_truth=ground_truth,
        round=round,
        fit_kwargs=fit_kwargs,
        verbose=verbose,
        experiment_custom_tags=experiment_custom_tags,
        **kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def assign_model(
    model, transformation: bool = False, verbose: bool = True
) -> pd.DataFrame:

    """
    This function assigns cluster labels to the dataset for a given model.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> kmeans_df = assign_model(kmeans)



    model: scikit-learn compatible object
        Trained model object


    transformation: bool, default = False
        Whether to apply cluster labels on the transformed dataset.


    verbose: bool, default = True
        Status update is not printed when verbose is set to False.


    Returns:
        pandas.DataFrame

    """

    return _CURRENT_EXPERIMENT.assign_model(
        model, transformation=transformation, verbose=verbose
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def plot_model(
    model,
    plot: str = "cluster",
    feature: Optional[str] = None,
    label: bool = False,
    scale: float = 1,
    save: bool = False,
    display_format: Optional[str] = None,
) -> Optional[str]:

    """
    This function analyzes the performance of a trained model.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> plot_model(kmeans, plot = 'cluster')


    model: scikit-learn compatible object
        Trained Model Object


    plot: str, default = 'cluster'
        List of available plots (ID - Name):

        * 'cluster' - Cluster PCA Plot (2d)
        * 'tsne' - Cluster t-SNE (3d)
        * 'elbow' - Elbow Plot
        * 'silhouette' - Silhouette Plot
        * 'distance' - Distance Plot
        * 'distribution' - Distribution Plot


    feature: str, default = None
        Feature to be evaluated when plot = 'distribution'. When ``plot`` type is
        'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or
        label when the ``label`` param is set to True. When the ``plot`` type is
        'cluster' or 'tsne' and feature is None, first column of the dataset is
        used.


    label: bool, default = False
        Name of column to be used as data labels. Ignored when ``plot`` is not
        'cluster' or 'tsne'.


    scale: float, default = 1
        The resolution scale of the figure.


    save: bool, default = False
        When set to True, plot is saved in the current working directory.


    display_format: str, default = None
        To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.
        Currently, not all plots are supported.


    Returns:
        Path to saved file, if any.

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
    --------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> evaluate_model(kmeans)


    model: scikit-learn compatible object
        Trained model object


    feature: str, default = None
        Feature to be evaluated when plot = 'distribution'. When ``plot`` type is
        'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or
        label when the ``label`` param is set to True. When the ``plot`` type is
        'cluster' or 'tsne' and feature is None, first column of the dataset is
        used.


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
    optimize: Optional[str] = None,
    custom_grid: Optional[List[int]] = None,
    fold: int = 10,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    round: int = 4,
    verbose: bool = True,
):

    """
    This function tunes the ``num_clusters`` parameter of a given model.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = juice)
    >>> tuned_kmeans = tune_model(model = 'kmeans', supervised_target = 'Purchase')


    model: str
        ID of an model available in the model library. Models that can be
        tuned in this function (ID - Model):

        * 'kmeans' - K-Means Clustering
        * 'sc' - Spectral Clustering
        * 'hclust' - Agglomerative Clustering
        * 'birch' - Birch Clustering
        * 'kmodes' - K-Modes Clustering


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


    optimize: str, default = None
        For Classification tasks:
            Accuracy, AUC, Recall, Precision, F1, Kappa (default = 'Accuracy')

        For Regression tasks:
            MAE, MSE, RMSE, R2, RMSLE, MAPE (default = 'R2')


    custom_grid: list, default = None
        By default, a pre-defined number of clusters is iterated over to
        optimize the supervised objective. To overwrite default iteration,
        pass a list of num_clusters to iterate over in custom_grid param.


    fold: int, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2.


    verbose: bool, default = True
        Status update is not printed when verbose is set to False.


    Returns:
        Trained Model with optimized ``num_clusters`` parameter.


    Warnings
    --------
    - Affinity Propagation, Mean shift, Density-Based Spatial Clustering
      and OPTICS Clustering cannot be used in this function since they donot
      support the ``num_clusters`` param.


    """
    return _CURRENT_EXPERIMENT.tune_model(
        model=model,
        supervised_target=supervised_target,
        supervised_type=supervised_type,
        supervised_estimator=supervised_estimator,
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
    This function generates cluster labels using a trained model.

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> kmeans_predictions = predict_model(model = kmeans, data = unseen_data)


    model: scikit-learn compatible object
        Trained Model Object.


    data : pandas.DataFrame
        Shape (n_samples, n_features) where n_samples is the number of samples and
        n_features is the number of features.


    Returns:
        pandas.DataFrame


    Warnings
    --------
    - Models that do not support 'predict' method cannot be used in the ``predict_model``.

    - The behavior of the predict_model is changed in version 2.1 without backward compatibility.
      As such, the pipelines trained using the version (<= 2.0), may not work for inference
      with version >= 2.1. You can either retrain your models with a newer version or downgrade
      the version for inference.


    """

    experiment = _CURRENT_EXPERIMENT
    if experiment is None:
        experiment = _EXPERIMENT_CLASS()

    return experiment.predict_model(
        estimator=model,
        data=data,
    )


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
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> # sets appropriate credentials for the platform as environment variables
    >>> import os
    >>> os.environ["AWS_ACCESS_KEY_ID"] = str("foo")
    >>> os.environ["AWS_SECRET_ACCESS_KEY"] = str("bar")
    >>> deploy_model(model = kmeans, model_name = 'kmeans-for-deployment', platform = 'aws', authentication = {'bucket' : 'S3-bucket-name'})


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
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> kmeans = create_model('kmeans')
    >>> save_model(lr, 'saved_kmeans_model')


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
    model_name: str,
    platform: Optional[str] = None,
    authentication: Optional[Dict[str, str]] = None,
    verbose: bool = True,
):

    """
    This function loads a previously saved pipeline.


    Example
    -------
    >>> from pycaret.clustering import load_model
    >>> saved_kmeans = load_model('saved_kmeans_model')


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
def pull(pop: bool = False) -> pd.DataFrame:
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
    return _CURRENT_EXPERIMENT.pull(pop=pop)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def models(internal: bool = False, raise_errors: bool = True) -> pd.DataFrame:

    """
    Returns table of models available in the model library.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
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
def get_metrics(
    reset: bool = False,
    include_custom: bool = True,
    raise_errors: bool = True,
) -> pd.DataFrame:

    """
    Returns table of metrics available.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> all_metrics = get_metrics()


    reset: bool, default = False
        If True, will reset all changes made using add_metric() and get_metric().


    include_custom: bool, default = True
        Whether to include user added (custom) metrics or not.


    raise_errors: bool, default = True
        If False, will suppress all exceptions, ignoring models
        that couldn't be created.


    Returns:
        pandas.DataFrame

    """

    return _CURRENT_EXPERIMENT.get_metrics(
        reset=reset,
        include_custom=include_custom,
        raise_errors=raise_errors,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def add_metric(
    id: str,
    name: str,
    score_func: type,
    target: str = "pred",
    greater_is_better: bool = True,
    multiclass: bool = True,
    **kwargs,
) -> pd.Series:
    """
    Adds a custom metric to be used in all functions.


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
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.


    multiclass: bool, default = True
        Whether the metric supports multiclass problems.


    **kwargs:
        Arguments to be passed to score function.

    Returns:
        pandas.Series

    """

    return _CURRENT_EXPERIMENT.add_metric(
        id=id,
        name=name,
        score_func=score_func,
        target=target,
        greater_is_better=greater_is_better,
        multiclass=multiclass,
        **kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def remove_metric(name_or_id: str):
    """
    Removes a metric used for evaluation.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
    >>> remove_metric('cs')


    name_or_id: str
        Display name or ID of the metric.


    Returns:
        None

    """

    return _CURRENT_EXPERIMENT.remove_metric(name_or_id=name_or_id)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_logs(experiment_name: Optional[str] = None, save: bool = False) -> pd.DataFrame:

    """
    Returns a table of experiment logs. Only works when ``log_experiment``
    is True when initializing the ``setup`` function.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery,  log_experiment = True)
    >>> kmeans = create_model('kmeans')
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
    This function is used to access global environment variables.

    Example
    -------
    >>> X_train = get_config('X_train')

    This will return X_train transformed dataset.

    Returns
    -------
    variable

    """

    return _CURRENT_EXPERIMENT.get_config(variable=variable)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def set_config(variable: str, value):

    """
    This function is used to reset global environment variables.

    Example
    -------
    >>> set_config('seed', 123)

    This will set the global seed to '123'.

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
    >>> jewellery = get_data('jewellery')
    >>> from pycaret.clustering import *
    >>> exp_name = setup(data = jewellery)
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
    >>> from pycaret.clustering import load_config
    >>> load_config('myvars.pkl')


    Returns:
        Global variables

    """

    return _CURRENT_EXPERIMENT.load_config(file_name=file_name)


def set_current_experiment(experiment: ClusteringExperiment):
    """
    Set the current experiment to be used with the functional API.

    experiment: ClusteringExperiment
        Experiment object to use.

    Returns:
        None
    """
    global _CURRENT_EXPERIMENT

    if not isinstance(experiment, ClusteringExperiment):
        raise TypeError(
            f"experiment must be a PyCaret ClusteringExperiment object, got {type(experiment)}."
        )
    _CURRENT_EXPERIMENT = experiment
