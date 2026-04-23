# Dev setup

Zero-to-running-tests in < 5 minutes.

## Prerequisites

- Git
- [`uv`](https://docs.astral.sh/uv/) — Astral's Python + package manager. `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) or `irm https://astral.sh/uv/install.ps1 | iex` (Windows PowerShell).

That's it. `uv` handles Python itself — you don't need a system `python` installed.

## Clone + bootstrap

```bash
git clone https://github.com/pycaret/pycaret.git
cd pycaret

# Install Python 3.13 if you don't have it yet (uv manages it):
uv python install 3.13

# Create a venv and install all deps including extras + dev/test groups:
uv sync --all-extras

# Verify:
uv run python -c "import pycaret; print(pycaret.__version__)"
# -> 4.0.0.dev0
```

## First green test

```bash
# Fast unit + e2e (< 2 min on a laptop):
uv run pytest tests/test_core_architecture.py tests/test_datasets.py -q

# End-to-end smoke across all 5 tasks (~5 min):
uv run pytest tests/test_e2e_oop.py -q

# Everything:
uv run pytest tests/ -q
```

If anything fails, open an issue with the full traceback and the output of `uv run python -m pycaret.utils._show_versions`.

## Lint + format

Ruff is the single linter + formatter.

```bash
uv run ruff check pycaret/ tests/          # lint
uv run ruff format pycaret/ tests/         # format (does not lint)
uv run ruff check pycaret/ tests/ --fix    # lint + apply autofixes
```

## Project layout

See `/AGENTS.md` → "Repo map" for the full layout.

## Python version matrix

| Version | Status | Notes |
|---|---|---|
| 3.11 | supported floor | Aligned with sklearn 1.7's floor |
| 3.12 | supported | — |
| 3.13 | **primary dev target** | What CI runs |
| 3.14 | **tracked, blocked** | Upstream joblib/cloudpickle need to ship PEP 649 support; see `docs/revamp/thinking/2026-04-22_python314_pep649_blocker.md` |

## Windows notes

- Use PowerShell or Git-Bash. `cmd.exe` works but some `uv` subcommands print ANSI codes that look ugly.
- Some optional deps (e.g. `statsforecast`) need a C compiler on Windows. They're deliberately excluded from the `timeseries` extra — install manually if you need them.

## IDE setup

- **VS Code:** the repo-root `pyproject.toml` is recognized. Set Python interpreter to `.venv\Scripts\python.exe`. Ruff extension picks up config automatically.
- **PyCharm:** import as an `existing venv` project, point to `.venv/`.

## Pre-commit hooks (optional)

```bash
uv run pre-commit install
```

See `.pre-commit-config.yaml` for what runs.
