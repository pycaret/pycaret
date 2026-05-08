"""Symmetric encryption for at-rest secrets (LLM API keys, future cloud creds).

Backed by ``cryptography.fernet.Fernet`` (AES-128-CBC + HMAC-SHA256). Keys
ride in env var ``PYCARET_SECRETS_KEY`` (base64-urlsafe-encoded 32 bytes —
the format ``Fernet.generate_key()`` produces).

Stored ciphertext is prefixed with ``ENC:v1:`` so reads can transparently
fall back to plaintext for rows written before this module existed. New
writes are always encrypted. Operators rotating the key should re-encrypt
existing rows; that's a future migration script.

Dev convenience: if no key is configured, we synthesize a per-process key
and log a loud warning. Encrypted values written under that key won't
survive a process restart, which is exactly the surfacing behaviour we
want — devs immediately notice and set ``PYCARET_SECRETS_KEY``.
"""

from __future__ import annotations

import logging
import threading

from cryptography.fernet import Fernet, InvalidToken

from pycaret_server.config import get_settings

_log = logging.getLogger(__name__)
_PREFIX = "ENC:v1:"

_lock = threading.Lock()
_fernet: Fernet | None = None
_using_ephemeral_key = False


def _resolve_fernet() -> Fernet:
    """Return the process-singleton Fernet instance, lazily constructed."""
    global _fernet, _using_ephemeral_key
    if _fernet is not None:
        return _fernet
    with _lock:
        if _fernet is not None:
            return _fernet
        settings = get_settings()
        key = settings.secrets_key
        if key:
            try:
                _fernet = Fernet(key.encode() if isinstance(key, str) else key)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "PYCARET_SECRETS_KEY is set but not a valid Fernet key "
                    "(must be base64-urlsafe-encoded 32 bytes — generate with "
                    "`python -c 'from cryptography.fernet import Fernet; "
                    "print(Fernet.generate_key().decode())'`). "
                    f"Original error: {exc}"
                ) from exc
        else:
            _fernet = Fernet(Fernet.generate_key())
            _using_ephemeral_key = True
            _log.warning(
                "PYCARET_SECRETS_KEY is not set. Using an EPHEMERAL key for "
                "this process — any encrypted secrets will be unreadable "
                "after restart. Set PYCARET_SECRETS_KEY in your environment "
                "before storing real credentials."
            )
    return _fernet


def encrypt(plaintext: str) -> str:
    """Encrypt ``plaintext`` and return the storage form (prefix + ciphertext)."""
    if plaintext is None:
        raise ValueError("encrypt() does not accept None")
    token = _resolve_fernet().encrypt(plaintext.encode("utf-8"))
    return _PREFIX + token.decode("ascii")


def decrypt(value: str) -> str:
    """Decrypt a stored value.

    For backwards compatibility, values without the ``ENC:v1:`` prefix are
    returned as-is on the assumption they were written before encryption
    was introduced. New writes always carry the prefix.
    """
    if value is None:
        raise ValueError("decrypt() does not accept None")
    if not value.startswith(_PREFIX):
        return value  # legacy plaintext
    token = value[len(_PREFIX) :].encode("ascii")
    try:
        return _resolve_fernet().decrypt(token).decode("utf-8")
    except InvalidToken as exc:
        raise RuntimeError(
            "Could not decrypt stored secret. The most likely cause is a "
            "PYCARET_SECRETS_KEY change (or restart with no key set). "
            "Re-enter the secret via the API."
        ) from exc


def is_encrypted(value: str | None) -> bool:
    """Return True if ``value`` is in the encrypted storage format."""
    return value is not None and value.startswith(_PREFIX)


def reset_for_tests() -> None:
    """Drop the cached Fernet so tests can swap the key at runtime."""
    global _fernet, _using_ephemeral_key
    with _lock:
        _fernet = None
        _using_ephemeral_key = False
