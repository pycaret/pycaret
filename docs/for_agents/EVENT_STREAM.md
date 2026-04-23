# Event stream

The engine emits structured `Event` dataclasses through `BaseLogger.log(...)`. Consumers (React UI, LLM agents, notebook cells) subscribe with `logger.subscribe(callback)`. Every event is JSON-serializable via `event.to_dict()`.

## `Event`

```python
@dataclass(frozen=True)
class Event:
    kind: EventKind                 # str-enum, JSON-friendly
    message: str                    # short human summary (may be empty)
    payload: dict[str, Any]         # structured fields specific to this kind
    duration_ms: float | None       # wall-clock for "finished" events; None otherwise
    timestamp: float                # time.time() when emitted
    experiment_id: str | None       # stamped on every event from a given Experiment
```

## `EventKind` — all 22 canonical kinds

| Kind | Phase | Payload (typical) |
|---|---|---|
| `experiment.started` | `fit` begins | `{target, rows}` |
| `experiment.fitted` | `fit` ends | — |
| `experiment.finished` | (reserved) | — |
| `preprocessor.started` | (reserved, native-preprocessor work) | — |
| `preprocessor.fitted` | (reserved) | — |
| `data.split` | (reserved) | `{train, test}` |
| `model.create.started` | `create_model` start | `{estimator}` |
| `model.created` | `create_model` end | `{estimator}` |
| `model.compare.started` | `compare_models` start | — |
| `model.compared` | per-model during compare | `{model_id, score}` (Phase 5 — currently only emitted at finish) |
| `model.compare.finished` | `compare_models` end | `{n_select}` |
| `model.tune.started` | `tune_model` start | — |
| `model.tuned` | `tune_model` end | — |
| `model.ensemble.started` | `ensemble_model` start | — |
| `model.ensembled` | `ensemble_model` end | — |
| `model.blend.started` | `blend_models` start | `{n_models}` |
| `model.blended` | `blend_models` end | — |
| `model.stack.started` | `stack_models` start | `{n_models}` |
| `model.stacked` | `stack_models` end | — |
| `model.calibrate.started` | `calibrate_model` start | — |
| `model.calibrated` | `calibrate_model` end | — |
| `model.finalized` | `finalize_model` end | — |
| `model.predicted` | `predict_model` end | `{n_rows}` |
| `model.saved` | `save_model` via Experiment | `{path}` |
| `model.loaded` | (reserved; top-level `load_model` is stateless) | — |
| `warning` | non-fatal | kind-specific |
| `error` | fatal | kind-specific |

## Subscribing

```python
from pycaret.logging import MemoryLogger

log = MemoryLogger()
unsub = log.subscribe(lambda event: print(event.kind.value, event.message))

exp = ClassificationExperiment(target="y", logger=log).fit(df)
exp.compare_models()

# Stop listening:
unsub()

# Replay after the fact:
for e in log.events:
    ...

# File-tee for the React UI to tail:
log2 = MemoryLogger(file="run.jsonl")     # writes one JSON line per event
```

## Writing a custom logger

```python
from pycaret.logging import BaseLogger

class DatabaseLogger(BaseLogger):
    def __init__(self, conn):
        super().__init__()
        self.conn = conn

    def emit(self, event):
        self.conn.execute("INSERT INTO events (...) VALUES (...)", event.to_dict())
```

Pass it as `Experiment(logger=DatabaseLogger(conn))`.

## Subscriber guarantees

- Subscribers are called synchronously in registration order.
- Exceptions inside a subscriber are swallowed — one bad subscriber does not break emission or other subscribers.
- `subscribe(callback)` returns an **unsubscribe function** — capture it to avoid memory leaks in long-running processes.
- `MemoryLogger.events` is a thread-safe snapshot (copy) at read time.
