"""Generate (and optionally execute) the 5 canonical PyCaret 4.0 notebooks.

Run:
    uv run python scripts/build_notebooks.py           # generate only
    uv run python scripts/build_notebooks.py --run     # generate + execute

Produces:
    notebooks/01_classification.ipynb
    notebooks/02_regression.ipynb
    notebooks/03_clustering.ipynb
    notebooks/04_anomaly_detection.ipynb
    notebooks/05_time_series.ipynb

These are the authoritative demo notebooks for PyCaret 4.0. Keep them short
and focused — full coverage lives in the OOP test suite. The goal is for a
new user (or an LLM agent looking for an example) to see the canonical
pattern for each task in < 50 cells.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"


# -----------------------------------------------------------------------------
# Cell-builder helpers
# -----------------------------------------------------------------------------


def _cells(*parts: Any) -> list:
    cells = []
    for p in parts:
        if isinstance(p, tuple):
            kind, text = p
            if kind == "md":
                cells.append(new_markdown_cell(text))
            elif kind == "code":
                cells.append(new_code_cell(text))
            else:
                raise ValueError(f"Unknown cell kind: {kind!r}")
        else:
            raise ValueError(f"Expected (kind, text) tuple, got {p!r}")
    return cells


def _write(name: str, cells: Iterable) -> Path:
    nb = new_notebook(
        cells=list(cells),
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": sys.version.split()[0],
            },
        },
    )
    dest = NOTEBOOK_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return dest


# -----------------------------------------------------------------------------
# Shared preamble
# -----------------------------------------------------------------------------


PREAMBLE = ("code", """# PyCaret 4.0 — OOP-only engine
# https://github.com/pycaret/pycaret
import warnings
warnings.filterwarnings("ignore")
import pycaret
print("PyCaret", pycaret.__version__)
""")


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------


def classification_notebook() -> list:
    return _cells(
        ("md", """# Classification — PyCaret 4.0

Predict a binary / multi-class label. This notebook shows the canonical
4.0 OOP pattern end-to-end:

1. Load data
2. Construct and fit a `ClassificationExperiment`
3. Compare models → get a typed `CompareResult`
4. Tune the best
5. Predict on test / new data
6. Save & load the fitted pipeline

Every verb returns a typed dataclass with fitted pipeline, metrics, and an
event trace. Nothing is implicit / global / stateful.
"""),
        PREAMBLE,
        ("md", "## 1. Load data"),
        ("code", """from pycaret.datasets import get_data
data = get_data("juice")
data.head()
"""),
        ("md", "## 2. Fit an experiment\n\nThe `ClassificationExperiment` is a `sklearn.base.BaseEstimator` subclass.\n`fit(data)` runs the full preprocessing pipeline and train/test split."),
        ("code", """from pycaret.tasks import ClassificationExperiment

exp = ClassificationExperiment(
    target="Purchase",
    session_id=42,
    n_jobs=1,            # set to -1 for all cores
    log_experiment=True, # capture a typed event stream
).fit(data)

print("is_fitted:", exp.__sklearn_is_fitted__())
print("train size:", exp.X_train.shape, "test size:", exp.X_test.shape)
"""),
        ("md", "## 3. Compare models\n\nReturns a `CompareResult` dataclass: `.best`, `.models`, `.leaderboard`, `.ranked_ids`, `.events`."),
        ("code", """result = exp.compare_models(include=["lr", "dt", "rf"])
print(type(result).__name__)
result.leaderboard
"""),
        ("code", """best = result.best
type(best).__name__
"""),
        ("md", "## 4. Tune the best model\n\nReturns a `TuneResult` with `.pipeline`, `.best_params`, `.cv_results`."),
        ("code", """tuned_result = exp.tune_model(best, n_iter=5)
tuned = tuned_result.pipeline
tuned_result.cv_results
"""),
        ("md", "## 5. Predict on the held-out test set\n\n`predict_model` returns a `PredictResult` with `.predictions` (DataFrame) and `.metrics` (DataFrame)."),
        ("code", """preds = exp.predict_model(tuned)
preds.predictions.head()
"""),
        ("md", "## 6. Persist the fitted pipeline\n\n`save_model` / `load_model` are stateless top-level utilities — no Experiment required to load."),
        ("code", """from pycaret import save_model, load_model
from pathlib import Path

out = save_model(tuned, Path("artifacts") / "juice_classifier")
print("saved to:", out)

restored = load_model(out)
print("restored:", type(restored).__name__)
"""),
        ("md", "## 7. Inspect the event stream\n\nThis is what a React UI / LLM agent subscribes to."),
        ("code", """events = exp.events
for e in events:
    dur = f"{e.duration_ms:7.1f} ms" if e.duration_ms else " " * 10
    print(f"{e.kind.value:30s} {dur}  {e.message}")
"""),
    )


# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------


def regression_notebook() -> list:
    return _cells(
        ("md", "# Regression — PyCaret 4.0\n\nPredict a continuous target. Same shape as the classification notebook, different `Experiment` subclass."),
        PREAMBLE,
        ("md", "## 1. Load data"),
        ("code", """from pycaret.datasets import get_data
data = get_data("boston")
data.head()
"""),
        ("md", "## 2. Fit an experiment"),
        ("code", """from pycaret.tasks import RegressionExperiment

exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1).fit(data)
exp.X_train.shape, exp.X_test.shape
"""),
        ("md", "## 3. Compare models"),
        ("code", """result = exp.compare_models(include=["lr", "ridge", "rf"])
result.leaderboard
"""),
        ("md", "## 4. Predict on the test set"),
        ("code", """preds = exp.predict_model(result.best)
preds.predictions.head()
"""),
        ("md", "## 5. Persist"),
        ("code", """from pycaret import save_model
save_model(result.best, "artifacts/boston_regressor")
"""),
    )


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------


def clustering_notebook() -> list:
    return _cells(
        ("md", "# Clustering — PyCaret 4.0\n\nUnsupervised: no `target`, no `compare_models` / `tune_model`. You pick an algorithm with `create_model`, then call `assign_model` to label every row."),
        PREAMBLE,
        ("md", "## 1. Load data"),
        ("code", """from pycaret.datasets import get_data
data = get_data("jewellery")
data.head()
"""),
        ("md", "## 2. Fit the experiment"),
        ("code", """from pycaret.tasks import ClusteringExperiment

exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(data)
exp.X.shape
"""),
        ("md", "## 3. Create a clustering model\n\n`create_model` returns a `CreateResult` with the fitted `.pipeline` and CV metrics."),
        ("code", """km = exp.create_model("kmeans", num_clusters=4)
km.metrics
"""),
        ("md", "## 4. Assign cluster labels to every row"),
        ("code", """labelled = exp.assign_model(km.pipeline)
labelled.head()
"""),
        ("md", "## 5. Predict clusters on new data"),
        ("code", """new_data = data.sample(5, random_state=42)
exp.predict_model(km.pipeline, data=new_data).predictions
"""),
    )


# -----------------------------------------------------------------------------
# Anomaly detection
# -----------------------------------------------------------------------------


def anomaly_notebook() -> list:
    return _cells(
        ("md", "# Anomaly detection — PyCaret 4.0\n\nUnsupervised: `create_model` picks the algorithm, `assign_model` labels every row with an anomaly score."),
        PREAMBLE,
        ("md", "## 1. Load data"),
        ("code", """from pycaret.datasets import get_data
data = get_data("anomaly")
data.head()
"""),
        ("md", "## 2. Fit the experiment"),
        ("code", """from pycaret.tasks import AnomalyExperiment

exp = AnomalyExperiment(session_id=42, n_jobs=1).fit(data)
exp.X.shape
"""),
        ("md", "## 3. Create an anomaly-detection model"),
        ("code", """iforest = exp.create_model("iforest")
iforest.params
"""),
        ("md", "## 4. Label every row"),
        ("code", """labelled = exp.assign_model(iforest.pipeline)
labelled.head()
"""),
        ("md", "Rows with `Anomaly == 1` are flagged; `Anomaly_Score` is the raw score."),
    )


# -----------------------------------------------------------------------------
# Time-series forecasting
# -----------------------------------------------------------------------------


def time_series_notebook() -> list:
    return _cells(
        ("md", "# Time-series forecasting — PyCaret 4.0\n\nInput is a univariate `pandas.Series` (or a DataFrame with a named target column). `fh` is the forecast horizon."),
        PREAMBLE,
        ("md", "## 1. Load data"),
        ("code", """from pycaret.datasets import get_data
y = get_data("airline")
y.head()
"""),
        ("md", "## 2. Fit the experiment\n\n`fh=12` means forecast 12 steps ahead."),
        ("code", """from pycaret.tasks import TimeSeriesExperiment

exp = TimeSeriesExperiment(fh=12, session_id=42, n_jobs=1).fit(y)
"""),
        ("md", "## 3. Compare forecasters"),
        ("code", """result = exp.compare_models(include=["naive", "arima", "exp_smooth"])
result.leaderboard
"""),
        ("md", "## 4. Forecast"),
        ("code", """best = result.best
forecast = exp.predict_model(best)
forecast.predictions
"""),
    )


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def build() -> list[Path]:
    paths: list[Path] = []
    for name, cells_fn in [
        ("01_classification.ipynb", classification_notebook),
        ("02_regression.ipynb", regression_notebook),
        ("03_clustering.ipynb", clustering_notebook),
        ("04_anomaly_detection.ipynb", anomaly_notebook),
        ("05_time_series.ipynb", time_series_notebook),
    ]:
        paths.append(_write(name, cells_fn()))
    return paths


def execute(paths: list[Path]) -> None:
    from nbclient import NotebookClient

    for p in paths:
        print(f"executing {p.name} ...", flush=True)
        nb = nbformat.read(p, as_version=4)
        client = NotebookClient(nb, timeout=600, kernel_name="python3")
        client.execute()
        with p.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"  done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Execute notebooks after generation")
    args = ap.parse_args()
    paths = build()
    for p in paths:
        print(f"wrote {p}")
    if args.run:
        execute(paths)
