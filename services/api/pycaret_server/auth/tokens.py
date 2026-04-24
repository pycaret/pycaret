"""JWT access + refresh token helpers.

Access tokens are stateless JWTs with a short TTL (default 60 min). Refresh
tokens are random strings whose SHA-256 hash is stored in the ``sessions``
table; the client holds the plaintext. On refresh, the server looks up the
row by hash and issues a new access token.

Revocation: delete / mark revoked the ``sessions`` row for that hash.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from pycaret_server.config import get_settings

_settings = get_settings()


@dataclass(frozen=True)
class TokenPayload:
    """Decoded access-token contents."""

    sub: str  # user id
    email: str
    is_superuser: bool
    exp: int
    iat: int
    typ: str  # "access" | "refresh"


def _now_ts() -> int:
    return int(datetime.now(UTC).timestamp())


def create_access_token(
    *,
    user_id: str,
    email: str,
    is_superuser: bool = False,
    ttl_minutes: int | None = None,
) -> str:
    """Encode a signed access JWT for ``user_id``."""
    ttl_minutes = ttl_minutes or _settings.access_token_ttl_minutes
    now = _now_ts()
    claims: dict[str, Any] = {
        "sub": user_id,
        "email": email,
        "is_superuser": is_superuser,
        "typ": "access",
        "iat": now,
        "exp": now + ttl_minutes * 60,
    }
    return jwt.encode(claims, _settings.jwt_secret, algorithm=_settings.jwt_algorithm)


def create_refresh_token() -> tuple[str, str, datetime]:
    """Mint a new refresh token.

    Returns ``(plaintext_token, sha256_hash, expires_at)``. The server stores
    the hash in ``sessions.refresh_token_hash``; the client holds the plaintext.
    """
    plaintext = secrets.token_urlsafe(48)
    token_hash = hash_refresh_token(plaintext)
    expires_at = datetime.now(UTC) + timedelta(days=_settings.refresh_token_ttl_days)
    return plaintext, token_hash, expires_at


def hash_refresh_token(plaintext: str) -> str:
    """SHA-256 of a refresh token — stored server-side."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def decode_token(token: str) -> TokenPayload:
    """Decode and verify a JWT. Raises jwt.InvalidTokenError on any failure."""
    claims = jwt.decode(
        token,
        _settings.jwt_secret,
        algorithms=[_settings.jwt_algorithm],
    )
    return TokenPayload(
        sub=claims["sub"],
        email=claims["email"],
        is_superuser=bool(claims.get("is_superuser", False)),
        exp=int(claims["exp"]),
        iat=int(claims["iat"]),
        typ=claims.get("typ", "access"),
    )
