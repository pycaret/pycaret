import pandas as pd
from pycaret.internal.pipeline import Pipeline


def create_classification_drift_report(
    estimator_name: str,
    prep_pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    unprocessed_data: pd.DataFrame=None,
) -> str:

    try:
        import evidently
    except ImportError:
        raise ImportError(
            "It appears that evidently (required for `drift_report=True`) is not installed. "
            "Do: pip install evidently"
        )

    from evidently.dashboard import Dashboard
    from evidently.tabs import DataDriftTab, CatTargetDriftTab
    from evidently.pipeline.column_mapping import ColumnMapping

    p = prep_pipe.steps[0][1].learned_dtypes
    numeric_features = list(p[(p == "float32") | (p == "float64")].index)
    categorical_features = list(p[p == "object"].index)
    if not unprocessed_data is None: # When ,model_predict data is None
        #filter out cases with object dtype
        categorical_features= unprocessed_data[categorical_features].select_dtypes(exclude=["object"]).columns.to_list()
        reference_data = unprocessed_data.iloc[X_train.index]
        current_data = unprocessed_data.iloc[X_test.index]
    else:
        target=[prep_pipe.steps[0][1].target]
        numeric_features=list(set(X_train.columns.to_list())&set(numeric_features)&set(X_test.columns.to_list()))
        categorical_features=list(set(X_train.columns.to_list())&set(categorical_features)&set(X_test.columns.to_list()))
        categorical_features= X_train[categorical_features].select_dtypes(exclude=["object"]).columns.to_list()
        reference_data=X_train[categorical_features+numeric_features+target]
        current_data=X_test[categorical_features+numeric_features+target]

    column_mapping = ColumnMapping()
    column_mapping.target = prep_pipe.steps[0][1].target
    column_mapping.prediction = None
    column_mapping.datetime = None
    column_mapping.numerical_features = numeric_features
    column_mapping.categorical_features = categorical_features

    dashboard = Dashboard(tabs=[DataDriftTab(), CatTargetDriftTab()])
    dashboard.calculate(reference_data, current_data, column_mapping=column_mapping)
    report_name = f"{estimator_name}_Drift_Report_Classification.html"
    dashboard.save(report_name)
    print(f"{report_name} saved successfully.")
    return report_name


def create_regression_drift_report(
    estimator_name: str,
    prep_pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    unprocessed_data: pd.DataFrame=None,
) -> str:

    try:
        import evidently
    except ImportError:
        raise ImportError(
            "It appears that evidently (required for `drift_report=True`) is not installed. "
            "Do: pip install evidently"
        )

    from evidently.dashboard import Dashboard
    from evidently.tabs import DataDriftTab, NumTargetDriftTab
    from evidently.pipeline.column_mapping import ColumnMapping

    p = prep_pipe.steps[0][1].learned_dtypes
    numeric_features = list(p[(p == "float32") | (p == "float64")].index)
    categorical_features = list(p[p == "object"].index)
    
    #filter out cases with object dtype
    categorical_features= unprocessed_data[categorical_features].select_dtypes(exclude=["object"]).columns.to_list()
    
    reference_data = unprocessed_data.iloc[X_train.index]
    current_data = unprocessed_data.iloc[X_test.index]

    column_mapping = ColumnMapping()
    column_mapping.target = prep_pipe.steps[0][1].target
    column_mapping.prediction = None
    column_mapping.datetime = None

    column_mapping.numerical_features = numeric_features
    column_mapping.categorical_features = categorical_features

    dashboard = Dashboard(tabs=[DataDriftTab(), NumTargetDriftTab()])
    dashboard.calculate(reference_data, current_data, column_mapping=column_mapping)
    report_name = f"{estimator_name}_Drift_Report_Regression.html"
    dashboard.save(report_name)
    print(f"{report_name} saved successfully.")
    return report_name
