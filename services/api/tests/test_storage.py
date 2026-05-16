"""Phase 2 ObjectStore tests.

Covers the LocalFs driver + the factory selection logic. S3 driver is
exercised via mocked boto3 in a separate test once boto3 lands in the
test extras.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pycaret_server.storage import ObjectStoreError, reset_for_tests
from pycaret_server.storage.local_fs import LocalFsObjectStore


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    monkeypatch.setenv("PYCARET_ARTIFACT_DIR", str(tmp_path / "art"))
    from pycaret_server.config import get_settings

    get_settings.cache_clear()
    reset_for_tests()
    yield
    reset_for_tests()


def test_local_fs_put_get_round_trip(tmp_path: Path) -> None:
    store = LocalFsObjectStore(root=tmp_path)
    uri = store.put_bytes("runs/abc/pipeline.pkl", b"hello")
    assert uri.startswith("file://")
    assert store.get_bytes(uri) == b"hello"
    assert store.exists(uri)
    assert store.size(uri) == 5


def test_local_fs_refuses_traversal(tmp_path: Path) -> None:
    """A malicious key with `..` segments mustn't escape the root."""
    store = LocalFsObjectStore(root=tmp_path)
    with pytest.raises(ObjectStoreError):
        store.put_bytes("../../etc/passwd", b"oops")


def test_local_fs_accepts_bare_path_for_back_compat(tmp_path: Path) -> None:
    """Pre-Phase-2 DB rows stored bare absolute paths; the driver still reads them."""
    p = tmp_path / "legacy.bin"
    p.write_bytes(b"legacy")
    store = LocalFsObjectStore(root=tmp_path)
    assert store.get_bytes(str(p)) == b"legacy"
    assert store.exists(str(p))


def test_local_fs_presigned_url_is_none(tmp_path: Path) -> None:
    """Local-fs has no concept of a presigned URL; caller streams via API."""
    store = LocalFsObjectStore(root=tmp_path)
    uri = store.put_bytes("x.bin", b"x")
    assert store.presigned_url(uri) is None


def test_factory_returns_local_fs_by_default() -> None:
    """Without env override, the factory hands back a LocalFsObjectStore."""
    from pycaret_server.storage import get_object_store
    from pycaret_server.storage.local_fs import LocalFsObjectStore

    s = get_object_store()
    assert isinstance(s, LocalFsObjectStore)
    assert s.scheme == "file"


def test_factory_singleton_is_stable() -> None:
    from pycaret_server.storage import get_object_store

    assert get_object_store() is get_object_store()


def test_factory_unknown_backend_raises(monkeypatch) -> None:
    monkeypatch.setenv("PYCARET_STORAGE_BACKEND", "exotic")
    from pycaret_server.config import get_settings
    from pycaret_server.storage import get_object_store

    get_settings.cache_clear()
    reset_for_tests()
    with pytest.raises(RuntimeError, match="unknown PYCARET_STORAGE_BACKEND"):
        get_object_store()
