---
description: Scaffold a new task module (classification/regression/clustering/...)
---

You're adding a new task type to PyCaret 4.0. Tasks are first-class:
each one has its own `Experiment` subclass, plot module, and docs
page.

## 1. Gather inputs

Ask the user for (if not already supplied):

- **Task name** (snake_case, e.g. `survival`)
- **Display name** (Title Case, e.g. `Survival Analysis`)
- **Primary metric** (e.g. `concordance_index`)
- **Default model** (e.g. `coxph`)

Don't proceed without all four.

## 2. Read the prior art

Skim how an existing task is wired up — `classification` is the
canonical example:

- `packages/engine/pycaret/classification/__init__.py`
- `packages/engine/pycaret/classification/_experiment.py`
- `packages/engine/pycaret/plots/classification.py`
- `packages/engine/tests/test_classification.py`
- `apps/site/content/docs/getting-started/modules.mdx`

## 3. Create the engine files

Mirror the classification pattern at `packages/engine/pycaret/<task>/`:

- `__init__.py` — re-exports `<Task>Experiment`
- `_experiment.py` — the class itself, subclassing the right base
  in `pycaret/core/experiment.py`

And the plot module:

- `packages/engine/pycaret/plots/<task>.py` — at minimum, define the
  task's primary diagnostic plots (use Plotly, return `Figure`s)

## 4. Wire it into the registry

- `packages/engine/pycaret/tasks/__init__.py` — register the new
  experiment class so `from pycaret.tasks import <Task>Experiment` works

## 5. Tests

- `packages/engine/tests/test_<task>.py` — at minimum, an E2E test
  that calls `.fit()` → `.compare_models()` → `.predict_model()`
  on a bundled dataset

Run it:

```bash
uv run pytest packages/engine/tests/test_<task>.py -v
```

## 6. Docs

- `apps/site/content/docs/getting-started/modules.mdx` — add a row
  to the modules table
- `apps/site/scripts/gen_api_tree.py` — confirm the new module is
  included by `PUBLIC_ROOTS`; if not, add it
- Run `cd apps/site && npm run sync` to regenerate the API tree

## 7. Lint, commit, PR

Same flow as `/work-on-approved-issue` step 6+ — open a PR against
`main` titled `feat: <Task> task module`.
