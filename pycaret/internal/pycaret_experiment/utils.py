from enum import Enum, auto


class MLUsecase(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()
    CLUSTERING = auto()
    ANOMALY = auto()
    TIME_SERIES = auto()


def get_ml_task(y):
    c1 = y.dtype == "int64"
    c2 = y.nunique() <= 20
    c3 = y.dtype.name in ["object", "bool", "category"]
    if (c1 & c2) | c3:
        ml_usecase = MLUsecase.CLASSIFICATION
    else:
        ml_usecase = MLUsecase.REGRESSION
    return ml_usecase


def highlight_setup(column):
    return [
        "background-color: lightgreen" if v is True or v == "Yes" else ""
        for v in column
    ]
