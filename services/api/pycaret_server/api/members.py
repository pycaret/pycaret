"""Workspace member CRUD — multi-user collaboration.

Routes under ``/api/v1/workspaces/{workspace_id}/members``:

  GET    /members             list members with role
  POST   /members             invite an existing user (by email) with a role
  PATCH  /members/{user_id}   change a member's role
  DELETE /members/{user_id}   remove a member

v1 semantics:
- Only workspace admins (or superusers) can mutate membership.
- "Invite" looks up an existing ``User`` by email — we don't create accounts.
  Email-invite-with-account-creation is V2 (needs a mail pipeline).
- Can't remove / demote the last admin. The UI reflects this by disabling
  the destructive action on the last-admin row.
- Role values are restricted to ``admin`` | ``member`` for v1 (SPEC § 17.2
  proposes a richer 6-role set; rolled forward when SSO lands).
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access, _require_admin
from pycaret_server.auth import CurrentUser
from pycaret_server.db import User, Workspace, WorkspaceMember, get_db

router = APIRouter(
    prefix="/workspaces/{workspace_id}/members",
    tags=["workspace-members"],
)

Role = Literal["admin", "member"]


class MemberRead(BaseModel):
    user_id: str
    email: EmailStr
    display_name: str | None
    role: Role
    is_active: bool
    created_at: str


class InviteRequest(BaseModel):
    email: EmailStr
    role: Role = Field(default="member")


class PatchRoleRequest(BaseModel):
    role: Role


def _serialise(m: WorkspaceMember, u: User) -> MemberRead:
    return MemberRead(
        user_id=u.id,
        email=u.email,
        display_name=u.display_name,
        role=m.role,  # type: ignore[arg-type]
        is_active=u.is_active,
        created_at=m.created_at.isoformat(),
    )


def _admin_count(db: Session, workspace_id: str) -> int:
    """Count currently-active admins in the workspace.

    "Last admin" guard reads this + refuses to drop below 1.
    """
    return int(
        db.scalar(
            select(func.count())
            .select_from(WorkspaceMember)
            .where(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.role == "admin",
            )
        )
        or 0
    )


# ──────────────────────────────────────────────────────────────── routes


@router.get("", response_model=list[MemberRead])
def list_members(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[MemberRead]:
    """List members + roles. Any workspace member can read this."""
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, workspace_id)

    rows = db.execute(
        select(WorkspaceMember, User)
        .join(User, User.id == WorkspaceMember.user_id)
        .where(WorkspaceMember.workspace_id == workspace_id)
        .order_by(WorkspaceMember.created_at.asc())
    ).all()
    return [_serialise(m, u) for (m, u) in rows]


@router.post("", response_model=MemberRead, status_code=status.HTTP_201_CREATED)
def invite_member(
    workspace_id: str,
    payload: InviteRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> MemberRead:
    """Add an existing user to the workspace with the given role.

    v1: looks up by email. Returns 404 if no such user exists. V2 will add
    an email-invite flow that creates a pending-account row.
    """
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_admin(user, db, workspace_id)

    target = db.scalar(select(User).where(User.email == payload.email))
    if target is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no user with email {payload.email!r}; ask them to sign up first "
            "(email invites arrive in V2).",
        )
    if not target.is_active:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "target user is deactivated")

    existing = db.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == target.id,
        )
    )
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "user is already a member of this workspace")

    m = WorkspaceMember(
        workspace_id=workspace_id,
        user_id=target.id,
        role=payload.role,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return _serialise(m, target)


@router.patch("/{user_id}", response_model=MemberRead)
def change_role(
    workspace_id: str,
    user_id: str,
    payload: PatchRoleRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> MemberRead:
    """Change a member's role. Last-admin protection applies."""
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_admin(user, db, workspace_id)

    m = db.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
    )
    if m is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "member not found")

    # If we're demoting an admin, refuse if that drops the workspace to zero admins.
    if m.role == "admin" and payload.role != "admin" and _admin_count(db, workspace_id) <= 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "cannot demote the last admin of this workspace",
        )

    m.role = payload.role
    db.commit()
    db.refresh(m)
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "user disappeared")
    return _serialise(m, target)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_member(
    workspace_id: str,
    user_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Remove a member. Last-admin protection applies."""
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_admin(user, db, workspace_id)

    m = db.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
    )
    if m is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "member not found")

    if m.role == "admin" and _admin_count(db, workspace_id) <= 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "cannot remove the last admin of this workspace",
        )

    db.delete(m)
    db.commit()
