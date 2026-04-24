"""Authentication routes: login, refresh, logout, me."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.schemas import (
    LoginRequest,
    RefreshRequest,
    TokenPairResponse,
    UserResponse,
)
from pycaret_server.auth import (
    CurrentUser,
    create_access_token,
    create_refresh_token,
    hash_refresh_token,
    verify_password,
)
from pycaret_server.config import get_settings
from pycaret_server.db import (
    Session as UserSession,
)
from pycaret_server.db import (
    User,
    get_db,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenPairResponse)
def login(
    payload: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenPairResponse:
    """Email + password login. Returns access + refresh tokens."""
    settings = get_settings()

    user = db.scalar(select(User).where(User.email == str(payload.email).lower()))
    if user is None or user.password_hash is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "user inactive")

    access_token = create_access_token(
        user_id=user.id, email=user.email, is_superuser=user.is_superuser
    )
    refresh_plain, refresh_hash, expires_at = create_refresh_token()
    db.add(
        UserSession(
            user_id=user.id,
            refresh_token_hash=refresh_hash,
            expires_at=expires_at,
        )
    )
    db.commit()

    return TokenPairResponse(
        access_token=access_token,
        refresh_token=refresh_plain,
        expires_in=settings.access_token_ttl_minutes * 60,
    )


@router.post("/refresh", response_model=TokenPairResponse)
def refresh(
    payload: RefreshRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenPairResponse:
    """Exchange a refresh token for a new access + rotated refresh pair."""
    settings = get_settings()
    rt_hash = hash_refresh_token(payload.refresh_token)

    session: UserSession | None = db.scalar(
        select(UserSession).where(UserSession.refresh_token_hash == rt_hash)
    )
    now = datetime.now(UTC)
    # SQLite strips tz info when reading DateTime(timezone=True); coerce both
    # sides to naive-UTC for comparison so this works against every backend.
    expires = session.expires_at if session else None
    if expires is not None and expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    now_naive_safe = now.replace(tzinfo=now.tzinfo) if expires is None else now
    if session is None or session.revoked_at is not None or expires <= now_naive_safe:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid refresh token")

    user = db.get(User, session.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found")

    # Rotate — invalidate the old row and mint a fresh one.
    session.revoked_at = now
    new_plain, new_hash, new_expires = create_refresh_token()
    db.add(
        UserSession(
            user_id=user.id,
            refresh_token_hash=new_hash,
            expires_at=new_expires,
        )
    )
    db.commit()

    return TokenPairResponse(
        access_token=create_access_token(
            user_id=user.id, email=user.email, is_superuser=user.is_superuser
        ),
        refresh_token=new_plain,
        expires_in=settings.access_token_ttl_minutes * 60,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    payload: RefreshRequest,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Revoke a refresh token. The access token expires on its own."""
    rt_hash = hash_refresh_token(payload.refresh_token)
    session = db.scalar(select(UserSession).where(UserSession.refresh_token_hash == rt_hash))
    if session is not None and session.revoked_at is None:
        session.revoked_at = datetime.now(UTC)
        db.commit()


@router.get("/me", response_model=UserResponse)
def me(user: CurrentUser) -> UserResponse:
    """Return the authenticated user."""
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=user.created_at,
    )
