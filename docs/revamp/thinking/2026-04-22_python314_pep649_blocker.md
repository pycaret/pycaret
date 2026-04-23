# Python 3.14 PEP 649 — blocks joblib/cloudpickle serialisation

*2026-04-22 · discovered during Phase 0 smoke testing*

## What happened

After getting `import pycaret` and per-submodule imports working on Python 3.14, the first real end-to-end call (`setup → compare_models`) failed during pipeline hashing:

```
_pickle.PicklingError: Can't pickle <function __annotate__ at 0x...>:
it's not the same object as pycaret.internal.pipeline.__annotate__
```

## Root cause

Python 3.14 ships with PEP 649 (deferred evaluation of annotations) enabled by default. Every class and function now gets an `__annotate__` attribute synthesised at class/function creation. Python's stdlib `pickle` module has `whichmodule()` guards that require each picklable object to be the *same identity* as the attribute looked up by name on its declared module — and PEP 649's `__annotate__` functions fail that identity check because they are synthesised at attribute-access time rather than being bound module globals.

This affects **any** tool that pickles class/function graphs — joblib, cloudpickle, multiprocessing. PyCaret hits it because `joblib.Memory` hashes each pipeline step's code and metadata, which walks the `__annotate__` attribute and trips on the identity assertion.

## Why pycaret can't fix this alone

The offending path is pure `joblib.memory` → `pickle`. A pycaret-level workaround would need to monkey-patch `pickle.whichmodule` or suppress `__annotate__` at serialise time, both of which are worse than the problem.

## Upstream status (as of 2026-04-22)

Both joblib and cloudpickle have open tracking issues for PEP 649 compatibility. Fixes are in progress on cloudpickle main; a joblib release pinning a compatible cloudpickle is expected but not shipped.

## Decision

- **Primary dev target:** Python 3.13 (stable on current joblib/cloudpickle).
- **CI matrix:** 3.11 / 3.12 / 3.13.
- **Python 3.14:** tracked as a future target. Revisit when upstream joblib ships a release that declares 3.14 support in its classifiers. No pycaret-side workaround.
- `pyproject.toml` still declares `3.14` in classifiers (aspirational); CI will refuse to add a 3.14 row until upstream is fixed.

## What this means for the roadmap

Unchanged. Phase 2 (modernization) still targets current sklearn / NumPy 2.x / pandas 2.x, which is fully independent of PEP 649. The version-matrix tightening to 3.13 is recorded in `DECISIONS.md`.
