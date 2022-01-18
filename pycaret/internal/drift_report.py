import pandas as pd
from pycaret.internal.pipeline import Pipeline


def create_classification_drift_report(
    estimator_name: str,
    prep_pipe: Pipeline,
    unprocessed_data: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
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

    reference_data = unprocessed_data.iloc[X_train.index]
    current_data = unprocessed_data.iloc[X_test.index]

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
    unprocessed_data: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
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
