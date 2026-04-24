"""FastAPI dependencies for authenticated routes.

Two authentication shapes are accepted by every protected route:

1. **Bearer JWT** — the primary path used by the web UI. Short-lived access
   token in the ``Authorization: Bearer <jwt>`` header, with refresh rotation
   via ``/auth/refresh``.

2. **API key** — programmatic fallback for CI pipelines + scripts. A personal
   key (see ``api_keys.py``) passed in the ``X-PyCaret-Key`` header. We look
   up the SHA-256 hash, reject revoked / expired rows, bump ``last_used_at``.

Both routes resolve to the same `User` model. Route signatures don't care
which credential the caller used; everything after this file is credential-
agnostic.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.auth.tokens import TokenPayload, decode_token
from pycaret_server.db import ApiKey, User, get_db

_bearer = HTTPBearer(auto_error=False)


def _auth_error(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _resolve_api_key(db: Session, header_value: str) -> User:
    """Validate an ``X-PyCaret-Key`` and return the associated user.

    Checks (in order): key exists, not revoked, not expired, user exists
    and is active. Updates ``last_used_at`` on success.
    """
    token_hash = hashlib.sha256(header_value.encode("utf-8")).hexdigest()
    key = db.scalar(select(ApiKey).where(ApiKey.token_hash == token_hash))
    if key is None:
        raise _auth_error("invalid API key")
    if key.revoked_at is not None:
        raise _auth_error("API key revoked")
    if key.expires_at is not None:
        expires = key.expires_at
        # SQLite strips tz info on DateTime(timezone=True) round-trips. Coerce
        # back so the compare below is apples-to-apples.
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        if expires <= datetime.now(UTC):
            raise _auth_error("API key expired")

    user = db.get(User, key.user_id) if key.user_id else None
    if user is None or not user.is_active:
        raise _auth_error("API key's user not found or inactive")

    # Touch the last-used timestamp so operators can spot dormant keys.
    # Don't fail the request on a commit failure — the auth decision is made.
    try:
        key.last_used_at = datetime.now(UTC)
        db.commit()
    except Exception:
        db.rollback()
    return user


def get_current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: Annotated[Session, Depends(get_db)],
    x_pycaret_key: Annotated[str | None, Header(alias="X-PyCaret-Key")] = None,
) -> User:
    """Resolve the authenticated user.

    Accepts either ``Authorization: Bearer <jwt>`` (primary) or
    ``X-PyCaret-Key: <pck_…>`` (programmatic). If both are present, the
    bearer JWT takes precedence — matches the common developer pattern of
    having a long-lived key in their env plus a short-lived session token
    while they're clicking around in the UI.
    """
    # 1) Bearer JWT wins when present.
    if creds is not None:
        try:
            payload: TokenPayload = decode_token(creds.credentials)
        except jwt.ExpiredSignatureError as e:
            raise _auth_error("access token expired") from e
        except jwt.InvalidTokenError as e:
            raise _auth_error("invalid access token") from e
        if payload.typ != "access":
            raise _auth_error("wrong token type (expected access)")
        user = db.get(User, payload.sub)
        if user is None or not user.is_active:
            raise _auth_error("user not found or inactive")
        return user

    # 2) Fall back to the API key header.
    if x_pycaret_key:
        return _resolve_api_key(db, x_pycaret_key)

    raise _auth_error("missing Authorization or X-PyCaret-Key header")


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_admin(user: CurrentUser) -> User:
    """Dependency: resolve current user AND require superuser.

    Workspace-level `admin` role is enforced per-route inside the route body;
    this dependency is for cross-workspace / installation-admin checks only.
    """
    if not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin privilege required",
        )
    return user
