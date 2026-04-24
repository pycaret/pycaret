"""API key CRUD — personal programmatic-access tokens.

Routes under ``/api/v1/auth/api-keys``:

  POST   /auth/api-keys         mint a new key (plaintext returned ONCE)
  GET    /auth/api-keys         list this user's active keys (no plaintext)
  DELETE /auth/api-keys/{id}    revoke a key

Keys are scoped to the issuing user + optionally to a workspace. Revocation
sets ``revoked_at`` (soft delete — audit trail preserved).

Current auth stack (JWT) remains the primary credential. API keys are the
fallback for CI pipelines + scripts that can't do the login/refresh dance.
Middleware that accepts `X-PyCaret-Key` for programmatic traffic is a
session-20 concern; this session just ships the CRUD surface.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.auth import CurrentUser
from pycaret_server.db import ApiKey, get_db

router = APIRouter(tags=["auth"])

# Keys are 32 bytes of randomness url-safe-b64-encoded. That's ~43 chars
# plus the 4-char prefix = 47 chars. Recognisable `pck_` prefix so leaked
# keys are greppable in logs + GitHub secret scanners.
_KEY_PREFIX = "pck_"


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    workspace_id: str | None = None
    expires_in_days: int | None = Field(
        default=None, ge=1, le=3650, description="Optional expiry; None = never"
    )
    scopes: list[str] | None = None


class ApiKeyRead(BaseModel):
    id: str
    name: str
    prefix: str
    workspace_id: str | None
    scopes: list[str] | None
    expires_at: datetime | None
    last_used_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime


class ApiKeyCreateResponse(ApiKeyRead):
    """Returned once on creation — includes the plaintext token.

    The UI must warn the user to copy it now; subsequent reads never expose
    the plaintext (only the prefix).
    """

    token: str


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _mint_token() -> tuple[str, str, str]:
    """Return `(plaintext, prefix_for_display, sha256_hash)`."""
    body = secrets.token_urlsafe(32)
    plaintext = f"{_KEY_PREFIX}{body}"
    # First 12 chars after the prefix are the public display prefix.
    prefix = plaintext[: len(_KEY_PREFIX) + 8]  # e.g. "pck_abcd1234"
    return plaintext, prefix, _hash(plaintext)


def _serialise(key: ApiKey) -> ApiKeyRead:
    return ApiKeyRead(
        id=key.id,
        name=key.name,
        prefix=key.prefix,
        workspace_id=key.workspace_id,
        scopes=list(key.scopes) if key.scopes else None,
        expires_at=key.expires_at,
        last_used_at=key.last_used_at,
        revoked_at=key.revoked_at,
        created_at=key.created_at,
    )


# ─────────────────────────────────────────────────────────── routes


@router.post(
    "/auth/api-keys",
    response_model=ApiKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_api_key(
    payload: ApiKeyCreate,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ApiKeyCreateResponse:
    """Mint a new API key. The plaintext is returned **once** in the response."""
    plaintext, prefix, token_hash = _mint_token()
    expires_at = (
        datetime.now(UTC) + timedelta(days=payload.expires_in_days)
        if payload.expires_in_days
        else None
    )
    key = ApiKey(
        name=payload.name,
        token_hash=token_hash,
        prefix=prefix,
        user_id=user.id,
        workspace_id=payload.workspace_id,
        expires_at=expires_at,
        scopes=list(payload.scopes) if payload.scopes else None,
    )
    db.add(key)
    db.commit()
    db.refresh(key)

    base = _serialise(key).model_dump()
    return ApiKeyCreateResponse(**base, token=plaintext)


@router.get(
    "/auth/api-keys",
    response_model=list[ApiKeyRead],
)
def list_api_keys(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[ApiKeyRead]:
    """List the caller's API keys. Plaintext is never exposed."""
    rows = db.scalars(
        select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc())
    ).all()
    return [_serialise(k) for k in rows]


@router.delete("/auth/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_api_key(
    key_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Revoke (soft-delete) an API key. The row stays; ``revoked_at`` is set."""
    key = db.get(ApiKey, key_id)
    if key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "api key not found")
    if key.user_id != user.id and not user.is_superuser:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "can only revoke your own API keys")
    if key.revoked_at is None:
        key.revoked_at = datetime.now(UTC)
        db.commit()
