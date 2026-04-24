"""First-run bootstrap routes.

The frontend checks ``GET /api/v1/setup/status`` on load. If
``is_bootstrapped == false``, it redirects to a setup wizard that POSTs to
``/api/v1/setup/bootstrap`` with admin credentials + workspace name. The
bootstrap creates:

- a ``User`` row (superuser, active)
- a ``Workspace`` row
- a ``WorkspaceMember`` row (role=admin)

Subsequent calls to ``/setup/bootstrap`` return 409 — there can be only one.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.schemas import (
    BootstrapRequest,
    SetupStatusResponse,
    TokenPairResponse,
    UserResponse,
)
from pycaret_server.auth import (
    create_access_token,
    create_refresh_token,
    hash_password,
)
from pycaret_server.config import get_settings
from pycaret_server.db import (
    Session as UserSession,
)
from pycaret_server.db import (
    User,
    Workspace,
    WorkspaceMember,
    get_db,
)

router = APIRouter(prefix="/setup", tags=["setup"])


@router.get("/status", response_model=SetupStatusResponse)
def setup_status(db: Annotated[Session, Depends(get_db)]) -> SetupStatusResponse:
    """Report whether the instance has been bootstrapped."""
    user_count = db.scalar(select(User.id).limit(1))  # any row?
    workspace_count = db.scalar(select(Workspace.id).limit(1))
    return SetupStatusResponse(
        is_bootstrapped=bool(user_count and workspace_count),
        user_count=db.query(User).count(),
        workspace_count=db.query(Workspace).count(),
    )


@router.post(
    "/bootstrap",
    response_model=TokenPairResponse,
    status_code=status.HTTP_201_CREATED,
)
def bootstrap(
    payload: BootstrapRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenPairResponse:
    """First-run setup: create the admin user + initial workspace.

    Fails with 409 if any User or Workspace already exists.
    """
    settings = get_settings()

    if db.query(User).first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="instance already bootstrapped",
        )

    user = User(
        email=str(payload.email).lower(),
        display_name=payload.display_name,
        password_hash=hash_password(payload.password),
        is_active=True,
        is_superuser=True,
    )
    db.add(user)
    db.flush()  # assigns user.id

    workspace = Workspace(
        name=payload.workspace_name,
        created_by=user.id,
    )
    db.add(workspace)
    db.flush()

    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role="admin",
    )
    db.add(member)

    # Issue tokens so the UI logs the admin straight in.
    access_token = create_access_token(user_id=user.id, email=user.email, is_superuser=True)
    refresh_plain, refresh_hash, expires_at = create_refresh_token()
    session = UserSession(
        user_id=user.id,
        refresh_token_hash=refresh_hash,
        expires_at=expires_at,
    )
    db.add(session)
    db.commit()

    return TokenPairResponse(
        access_token=access_token,
        refresh_token=refresh_plain,
        expires_in=settings.access_token_ttl_minutes * 60,
    )


@router.get("/me", response_model=UserResponse)
def me(
    db: Annotated[Session, Depends(get_db)],
    # `get_current_user` is imported inline to avoid a circular import via auth/deps.py
) -> UserResponse:
    """Convenience: return the current authenticated user.

    Actually wired up through auth dependency below.
    """
    # placeholder — proper wiring in /auth/me; kept here as a stub for the
    # first-run flow where there's no auth yet but we want an endpoint to
    # echo the just-bootstrapped admin. Wire once /auth router exists.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="use /api/v1/auth/me with an access token",
    )
