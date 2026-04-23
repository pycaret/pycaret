"""PyCaret 4.0 pytest conftest.

The 3.x conftest wired up fixtures that depended on the module-level
functional API (`pycaret.*.functional._CURRENT_EXPERIMENT`). That API is
removed in 4.0 — experiments are OOP-only and stateless at the module level,
so no between-test reset is needed.

Remaining fixtures are minimal: a working directory sandbox and a couple of
dataset loaders for time-series tests that might be re-authored later. All the
state-reset machinery from the 3.x conftest is gone.
"""

from __future__ import annotations

import pytest


@pytest.fixture(name="change_test_dir", autouse=True)
def change_test_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
