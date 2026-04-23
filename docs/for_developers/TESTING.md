# Testing

## Test layout

```
tests/
├── conftest.py                  <- minimal (no implicit-state fixtures)
├── test_core_architecture.py    <- fast unit tests for pycaret.core / pycaret.api / pycaret.logging
├── test_e2e_oop.py              <- end-to-end smoke tests, one per task
├── test_datasets.py             <- dataset loaders
└── test_models.py               <- model-registry equality (OOP)
```

All tests run under the OOP API. There is no longer a test file per functional-API module — the 41 3.x test files that coupled to the module-level functional API were deleted in the 4.0 revamp (see session 3 in `docs/revamp/release_notes_pycaret4.md`).

## Run

```bash
# Fast — unit + cheap integration:
uv run pytest tests/test_core_architecture.py tests/test_datasets.py -q    # ~5s

# Full e2e (slow — actually trains models):
uv run pytest tests/ -q                                                    # ~2m

# Single test:
uv run pytest tests/test_e2e_oop.py::test_classification_e2e_oop -v

# Skip the slow end-to-end tests:
uv run pytest tests/ -q -m "not slow"
```

## Markers

| Marker | Meaning |
|---|---|
| `slow` | Tests that train real models; skipped with `-m "not slow"` |
| `plotting` | Plot rendering tests (currently minimal; expands in Phase 3) |

Markers are declared in `pyproject.toml` → `[tool.pytest.ini_options]` → `markers`.

## Writing a new test

### Unit test (fast)

Add to `tests/test_core_architecture.py`. Don't train models here — test data structures, enums, dataclass round-trips, factory functions.

```python
def test_my_new_dataclass_json_roundtrip():
    from pycaret.core.results import CompareResult
    # ... construct with dummy fields, assert to_dict() works
```

### End-to-end test (slow)

Add to `tests/test_e2e_oop.py`. Use `@pytest.mark.slow`. One dataset per task, kept small.

```python
@pytest.mark.slow
def test_my_new_verb_e2e():
    from pycaret.datasets import get_data
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="y").fit(get_data("juice"))
    result = exp.my_new_verb(...)
    assert result.pipeline is not None
```

### Do NOT write

- Tests that exercise the 3.x functional API (`pycaret.classification.setup(...)`). It's gone.
- Tests that depend on module-level mutable state (`_CURRENT_EXPERIMENT`). It's gone.
- Tests that exercise killed dependencies (mlflow, boto3, yellowbrick, etc.). See `KILL_LIST.md`.

## CI matrix

CI runs on every push + PR:

- Ubuntu + Windows
- Python 3.11 + 3.12 + 3.13
- `uv sync --all-extras` + `uv run pytest tests/ -m "not slow"` + `uv run pytest tests/ -m "slow"`

The slow suite is gated on nightly / main-branch pushes in the future; PR checks currently run both.

## Coverage

```bash
uv run pytest --cov=pycaret --cov-report=term-missing tests/
```

`pytest-cov` is in the `test` extra.

## What to do when a test fails

1. **Reproduce locally.** `uv run pytest path/to/test_x.py::test_fn -v`.
2. **Check if it's environmental.** Re-run `uv sync --all-extras` to be sure the lockfile is current.
3. **Read the failure cluster taxonomy** in `docs/revamp/thinking/phase0_failure_landscape.md` — most test failures on this codebase fall into one of 5 root-cause buckets with known fixes.
4. **If it's a regression from your change**, fix it; if it's pre-existing, file an issue with the label `bug` and reference the test path.
