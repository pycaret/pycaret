# Coding style

Enforced by `ruff`. Config in `pyproject.toml` → `[tool.ruff]` / `[tool.ruff.lint]`. Run `uv run ruff check pycaret/ tests/` before every commit.

## Formatting

- Line length: **100 characters**.
- Use `ruff format` (no black).
- Use trailing commas in multi-line collections.
- Double quotes for strings; single quotes only inside f-strings where needed.

## Imports

- Absolute imports inside `pycaret/`. No relative imports.
- Standard lib → third-party → `pycaret.*`, with blank lines between groups.
- No star imports (`from x import *`).
- Lazy-import heavy optional dependencies **inside the function** that needs them:

```python
def plot_calibration(...):
    from sklearn.calibration import calibration_curve      # OK: lazy
    ...
```

## Type hints

- **Every public function** has complete type hints. No `Any` on parameters unless genuinely untyped.
- `from __future__ import annotations` at the top of every module (lets you use `list[int]` etc. on Python 3.9+ syntax).
- Use `TYPE_CHECKING` guards for imports needed only for type hints:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd
    from sklearn.pipeline import Pipeline
```

## Docstrings

- Numpydoc style.
- First line: one-sentence summary.
- Describe **why**, not what (the code already says what).
- Always document parameters, return type, and raised exceptions for public functions.
- Private functions: optional docstring, but add one if the intent isn't obvious from the signature.

## Naming

- `snake_case` for functions, methods, variables, module files.
- `PascalCase` for classes.
- `SCREAMING_SNAKE_CASE` for module-level constants.
- Private names with a single leading underscore (`_internal`). Double-underscore only for name-mangling (rare).

## Data classes

- All result / card / event types are `@dataclass(frozen=True)`.
- Every field has a type hint. Defaults via `field(default_factory=list)` for mutable defaults.
- Add a `to_dict()` method if the dataclass is intended to round-trip through JSON.

## Errors

- Raise errors from the `PyCaretError` hierarchy (`pycaret.core.errors`).
- Error messages are a single sentence with a trailing period, include enough context to act on:

```python
raise ConfigurationError(
    f"target column {self.target!r} not found in the DataFrame."
)
```

- Never `pass` or swallow exceptions silently. If you must suppress, comment why and wrap only the minimal surface.

## Logging

- No `print` statements inside `pycaret/*`. Events go through `self.logger.log(EventKind.X, ...)` on an `Experiment`.
- Module-level `warnings.warn(...)` is acceptable for compat / deprecation warnings.
- The top-level stateless utilities (`save_model`, `load_model`) can `print` when `verbose=True` is passed.

## Dependency discipline

- Do not add a new runtime dependency without:
  1. Justifying it in `DECISIONS.md` with a new ADR entry.
  2. Updating `pyproject.toml` — decide whether it goes in core (`dependencies`) or an optional extra.
  3. Release-notes entry under `DEPS`.
- Do not re-introduce anything on the kill list (`docs/revamp/KILL_LIST.md`).
- No upper-bound version pins on NumPy, pandas, scipy, sklearn, joblib. (The sklearn upper-bound in 4.0 is a transitional artifact of the sktime constraint; remove when sktime catches up.)

## Git hygiene

- Small, focused commits with clear messages.
- One concern per commit.
- Commit message format: imperative mood, no trailing period, < 72 chars on the first line.

```
Fix np.product -> np.prod for NumPy 2 compat

np.product was removed in NumPy 2.0. Replaced the single call site in
pycaret/internal/patches/sklearn.py with the aliased np.prod.
```
