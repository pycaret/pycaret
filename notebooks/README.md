# PyCaret 4.0 tutorials

**The `.ipynb` files are in [`legacy_v3/`](legacy_v3/) and use the PyCaret 3.x functional API, which was removed in 4.0.** They will be re-authored in OOP style for the 4.0 release.

If you want to run PyCaret 4.0 today, use the examples below or the [README quickstart](../README.md).

---

## The PyCaret 4.0 pattern

One shape works for every task. Pick the right `Experiment` subclass, construct it with your config, call `fit(data)`, then drive it.

```python
from pycaret.tasks import (
    ClassificationExperiment, RegressionExperiment,
    ClusteringExperiment, AnomalyExperiment, TimeSeriesExperiment,
)
from pycaret import save_model, load_model
from pycaret.datasets import get_data
```

## Classification

```python
df = get_data("juice")

exp = ClassificationExperiment(target="Purchase", session_id=42).fit(df)
result = exp.compare_models()                 # -> CompareResult
tuned  = exp.tune_model(result.best).pipeline # -> Pipeline
preds  = exp.predict_model(tuned).predictions # -> DataFrame

save_model(tuned, "artifacts/juice")
```

## Regression

```python
df = get_data("boston")
exp = RegressionExperiment(target="medv", session_id=42).fit(df)
best = exp.compare_models().best
exp.predict_model(best).predictions
```

## Clustering

```python
df = get_data("jewellery")
exp = ClusteringExperiment(session_id=42).fit(df)
kmeans = exp.create_model("kmeans", num_clusters=4).pipeline
labelled = exp.assign_model(kmeans)   # DataFrame with a "Cluster" column
```

## Anomaly detection

```python
df = get_data("anomaly")
exp = AnomalyExperiment(session_id=42).fit(df)
iforest = exp.create_model("iforest").pipeline
labelled = exp.assign_model(iforest)  # DataFrame with "Anomaly" + score
```

## Time-series forecasting

```python
y = get_data("airline")
exp = TimeSeriesExperiment(fh=12, session_id=42).fit(y)
best = exp.compare_models().best
forecast = exp.predict_model(best).predictions
```

---

## Agent / UI introspection

Every task exposes a typed, JSON-serializable description:

```python
from pycaret.api import list_models, describe_model, describe_setup_params

list_models("classification")              # -> list[ModelCard]
describe_model("classification", "rf")     # -> ModelCard(id='rf', name='Random Forest Classifier', …)

schema = describe_setup_params("classification")
import json
json.dumps(schema.to_dict())               # -> renders directly as a React form
```

## Event stream (for UIs and agents)

```python
from pycaret.logging import MemoryLogger

log = MemoryLogger()
log.subscribe(lambda event: print(event.kind.value, event.message))

exp = ClassificationExperiment(target="y", logger=log).fit(df)
exp.compare_models()
# Events: experiment.started, experiment.fitted,
#         model.compare.started, model.compare.finished
```

---

## What changed from 3.x

| 3.x (functional)                      | 4.0 (OOP)                                              |
|---------------------------------------|--------------------------------------------------------|
| `setup(data, target='y')`             | `exp = ClassificationExperiment(target='y').fit(data)` |
| `compare_models()`                    | `exp.compare_models()` → `CompareResult`               |
| `pull()`                              | `result.leaderboard`                                   |
| `tune_model(model)`                   | `exp.tune_model(model).pipeline`                       |
| `predict_model(m, data=new_df)`       | `exp.predict_model(m, data=new_df).predictions`        |
| `save_model(m, 'f')`                  | `from pycaret import save_model; save_model(m, 'f')`   |
| `set_current_experiment(exp)`         | — (implicit state removed)                             |
