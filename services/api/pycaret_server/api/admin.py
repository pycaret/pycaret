"""Superuser-only admin routes.

Routes (mounted under ``/api/v1``):

  GET    /admin/users         list all users (superuser-only)
  PATCH  /admin/users/{id}    toggle is_superuser / is_active

These are the cross-workspace controls — anything scoped to a single
workspace lives under ``/workspaces/{id}/...`` (see members.py, llm.py).

v1: only superusers can hit these. Multi-tier role enforcement (e.g. an
``operator`` role between user + superuser) is V2.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pycaret_server.auth import CurrentUser
from pycaret_server.db import User, WorkspaceMember, get_db

router = APIRouter(tags=["admin"])


class UserAdminRead(BaseModel):
    id: str
    email: str
    display_name: str | None
    is_superuser: bool
    is_active: bool
    workspace_count: int
    created_at: str | None


class UserAdminPatch(BaseModel):
    is_superuser: bool | None = None
    is_active: bool | None = None


def _require_superuser(user: User) -> None:
    if not user.is_superuser:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "this endpoint requires a platform superuser"
        )


@router.get("/admin/users")
def list_users(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """Return every user on the platform with workspace membership counts.

    Superuser-only. Use the ``WorkspaceMember`` join to display how many
    workspaces each user belongs to without N+1 queries.
    """
    _require_superuser(user)
    limit = max(1, min(500, limit))
    offset = max(0, offset)

    workspace_count_subq = (
        select(WorkspaceMember.user_id, func.count(WorkspaceMember.id).label("ws_count"))
        .group_by(WorkspaceMember.user_id)
        .subquery()
    )

    stmt = (
        select(User, func.coalesce(workspace_count_subq.c.ws_count, 0))
        .outerjoin(workspace_count_subq, User.id == workspace_count_subq.c.user_id)
        .order_by(User.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    items: list[UserAdminRead] = []
    for u, ws_count in db.execute(stmt).all():
        items.append(
            UserAdminRead(
                id=u.id,
                email=u.email,
                display_name=u.display_name,
                is_superuser=u.is_superuser,
                is_active=u.is_active,
                workspace_count=int(ws_count),
                created_at=u.created_at.isoformat() if u.created_at else None,
            )
        )
    return {"items": [it.model_dump() for it in items], "limit": limit, "offset": offset}


@router.patch("/admin/users/{user_id}")
def patch_user(
    user_id: str,
    payload: UserAdminPatch,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> UserAdminRead:
    """Update ``is_superuser`` and/or ``is_active``. Superuser-only.

    Guards:
    - You can't demote the last superuser (system would be uncontrollable).
    - You can't deactivate yourself (use a separate account to do that).
    """
    _require_superuser(user)
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")

    if payload.is_superuser is False and target.is_superuser:
        remaining = db.scalar(
            select(func.count(User.id)).where(
                User.is_superuser.is_(True), User.id != user_id, User.is_active.is_(True)
            )
        )
        if (remaining or 0) == 0:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "cannot demote the last active superuser",
            )

    if payload.is_active is False and target.id == user.id:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "cannot deactivate yourself"
        )

    if payload.is_superuser is not None:
        target.is_superuser = payload.is_superuser
    if payload.is_active is not None:
        target.is_active = payload.is_active

    db.commit()
    db.refresh(target)

    ws_count = db.scalar(
        select(func.count(WorkspaceMember.id)).where(WorkspaceMember.user_id == target.id)
    ) or 0
    return UserAdminRead(
        id=target.id,
        email=target.email,
        display_name=target.display_name,
        is_superuser=target.is_superuser,
        is_active=target.is_active,
        workspace_count=int(ws_count),
        created_at=target.created_at.isoformat() if target.created_at else None,
    )
