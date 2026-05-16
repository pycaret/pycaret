"""Shared test-suite setup.

Auto-fixtures that run for every test module:

- ``_reset_singletons_between_tests`` — clears the process-wide caches that
  Phase 1 (crypto, object store, orchestrator) introduced so each test
  starts with a fresh resolution from its monkeypatched settings.

Individual test files still own their per-test fixtures (``client``,
``tmp_path``, etc.). This conftest only guards the global state.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_singletons_between_tests():
    """Drop cached singletons between tests so each fixture's env vars take effect.

    Without this, the FIRST test's ``settings.artifact_dir`` is captured
    by the ObjectStore singleton and every later test ends up writing
    pickles into a stale tmp directory (which the test runner then
    cleans up, breaking downstream reads).
    """
    # Pre-test: clear anything held from a prior module's fixture.
    try:
        from pycaret_server.storage import reset_for_tests as reset_storage

        reset_storage()
    except Exception:  # noqa: BLE001
        pass
    try:
        from pycaret_server.crypto import reset_for_tests as reset_crypto

        reset_crypto()
    except Exception:  # noqa: BLE001
        pass
    yield
    # Post-test: same drill so the next test starts clean.
    try:
        from pycaret_server.storage import reset_for_tests as reset_storage

        reset_storage()
    except Exception:  # noqa: BLE001
        pass
