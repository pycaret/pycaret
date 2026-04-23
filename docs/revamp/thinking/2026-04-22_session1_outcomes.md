# Session 1 — Outcomes (raw notes for the research paper)

*2026-04-22 · Claude + Moez Ali · single session, continuous*

## Quantitative

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `pyproject.toml` core deps | 30 | 19 | -37% |
| Optional-extra tiers | 8 (tangled) | 6 (clean) | – |
| Total declared packages (core + all extras) | ~70+ | ~40 | -40%+ |
| Python support floor | 3.9 | 3.11 | +2 minor |
| Python support ceiling | <3.13 | 3.14 (3.13 primary) | +2 minor |
| scikit-learn cap | `<1.5` | `<1.8` | 3 minors up |
| Source files deleted | – | 7 py files (loggers + patches + plots) | |
| Test files deleted | – | 12 | |
| Legacy build files removed | – | `setup.cfg`, `MANIFEST.in`, `mypy.ini` | |
| Pytest collection errors | – | 0 (was 16 on first run) | |
| Smoke test | fails at import | green end-to-end | |

## Qualitative — what this session proved

1. **The dependency bloat was real and removable without rewriting the engine.** In a single afternoon, 37% of core deps and 12 test modules left the repo with zero loss of notebook-user functionality. The kill list from the user held up when stress-tested against actual import graphs.

2. **The "preserve the notebook golden path" heuristic worked cleanly.** `setup → compare_models → predict_model` never broke across ~30 mechanical edits. The stability of that surface is what let us amputate aggressively — every other API is fair game.

3. **Three-category failure taxonomy for an abandoned Python OSS project.** The failures encountered map to exactly three root causes that an AI agent can enumerate and fix systematically:
   - **Stdlib removals** — `distutils` (3.12), `np.NaN` (NumPy 2).
   - **Upstream API drift** — `joblib.Memory(bytes_limit=...)` moved to `reduce_size(bytes_limit=...)`.
   - **Vendor-removed-then-re-added deps** — optional trackers and visualisation packages that pinned the project to ancient versions of the core stack.
   (Plus one category the agent *cannot* fix locally: ecosystem-wide blockers like PEP 649 breaking pickling across joblib/cloudpickle. Those get documented, scoped out, and tracked.)

4. **Agent-led refactor is feasible when the owner makes the breaking-change call.** The thing that unlocked this pace was a clear instruction ("backward compat can break left right and center except the notebook API"). The agent did not have to re-ask for permission per change because the decision envelope was set.

5. **Documentation-as-you-go is structurally compatible with the agent's working style.** Writing the audit, kill list, roadmap, and decisions log *as* the work happens cost trivial additional tokens versus writing them after. It also served as a forcing function — proposing a cut in `KILL_LIST.md` made the agent pause enough to check assumptions before the code edit.

## What would break this approach at scale

- **Cold-start cost on the first session.** Roughly half of session 1's tokens went into audit, plan, and documentation scaffolding — before the first line of code changed. On a project smaller than pycaret the ROI curve would be worse.
- **Windows + prebuilt-wheel gaps.** statsforecast tripped the install because no cp314 wheel existed. The agent can route around this (drop the dep) but only because it was optional. A required dep with no wheel would need a human to unblock.
- **Tests that assert pre-amputation behaviour.** We deleted mlflow-tag tests rather than skip-gating them. That's the right call for 4.0 but loses coverage of the tag-normalisation logic. A longer-lived project would need a re-introduction plan.

## Reproduction recipe (for the paper)

```
git clone https://github.com/pycaret/pycaret.git
cd pycaret
# Read docs/revamp/README.md
# Apply all edits described in docs/revamp/DECISIONS.md in order
uv python install 3.13
uv venv --python 3.13 .venv
uv sync --all-extras
uv run pytest tests/ --collect-only   # expect 815 tests collected, 0 errors
uv run python -c "from pycaret.classification import setup, compare_models, predict_model; ..."
```
Each entry in `DECISIONS.md` maps to a reviewable diff; `thinking/` captures the reasoning.
