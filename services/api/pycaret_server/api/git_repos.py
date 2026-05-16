"""Phase 5: GitRepository routes.

Endpoints:

- ``GET    /workspaces/{ws}/git-repositories``
- ``POST   /workspaces/{ws}/git-repositories``
- ``DELETE /workspaces/{ws}/git-repositories/{id}``
- ``POST   /git-repositories/{id}/publish``   ← does the actual push

The publish endpoint runs synchronously for v1 (small workspaces).
The Job-table integration (Phase 1) makes it trivial to flip this to
async when needed: produce a Job(kind="git_publish"), worker handler
calls the same service function.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import GitRepository, Project, get_db

router = APIRouter(tags=["git"])


def _serialise(r: GitRepository) -> dict:
    return {
        "id": r.id,
        "workspace_id": r.workspace_id,
        "project_id": r.project_id,
        "provider": r.provider,
        "url": r.url,
        "default_branch": r.default_branch,
        "path_prefix": r.path_prefix,
        "secret_id": r.secret_id,
        "enabled": r.enabled,
        "auto_publish": bool(r.auto_publish),
        "last_push_at": r.last_push_at.isoformat() if r.last_push_at else None,
        "last_push_status": r.last_push_status,
        "last_push_sha": r.last_push_sha,
        "last_push_error": r.last_push_error,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "created_by": r.created_by,
    }


@router.get("/workspaces/{workspace_id}/git-repositories")
def list_repos(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    _require_access(user, db, workspace_id)
    rows = db.scalars(
        select(GitRepository)
        .where(GitRepository.workspace_id == workspace_id)
        .order_by(GitRepository.created_at.desc())
    ).all()
    return [_serialise(r) for r in rows]


@router.post(
    "/workspaces/{workspace_id}/git-repositories",
    status_code=status.HTTP_201_CREATED,
)
def create_repo(
    workspace_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Body: ``{provider, url, project_id?, default_branch?, path_prefix?, secret_id?}``."""
    _require_access(user, db, workspace_id)
    provider = payload.get("provider")
    url = payload.get("url")
    if not provider or not url:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "provider + url are required"
        )
    if provider not in ("github", "gitlab", "gitea", "bitbucket"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "provider must be github | gitlab | gitea | bitbucket",
        )
    project_id = payload.get("project_id")
    if project_id:
        p = db.get(Project, project_id)
        if p is None or p.workspace_id != workspace_id:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "project_id doesn't belong to this workspace",
            )
    r = GitRepository(
        workspace_id=workspace_id,
        project_id=project_id,
        provider=str(provider),
        url=str(url),
        default_branch=str(payload.get("default_branch") or "main"),
        path_prefix=payload.get("path_prefix"),
        secret_id=payload.get("secret_id"),
        enabled=bool(payload.get("enabled", True)),
        auto_publish=bool(payload.get("auto_publish", False)),
        created_by=user.id,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return _serialise(r)


@router.patch(
    "/workspaces/{workspace_id}/git-repositories/{repo_id}",
)
def patch_repo(
    workspace_id: str,
    repo_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Body: any subset of ``{enabled, auto_publish, default_branch,
    path_prefix, project_id, secret_id}``. Returns the updated repo."""
    _require_access(user, db, workspace_id)
    r = db.get(GitRepository, repo_id)
    if r is None or r.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "repo not found")
    for field in ("default_branch", "path_prefix", "secret_id"):
        if field in payload:
            setattr(r, field, payload[field])
    if "enabled" in payload:
        r.enabled = bool(payload["enabled"])
    if "auto_publish" in payload:
        r.auto_publish = bool(payload["auto_publish"])
    if "project_id" in payload:
        pid = payload["project_id"]
        if pid:
            p = db.get(Project, pid)
            if p is None or p.workspace_id != workspace_id:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    "project_id doesn't belong to this workspace",
                )
        r.project_id = pid or None
    db.commit()
    db.refresh(r)
    return _serialise(r)


@router.delete(
    "/workspaces/{workspace_id}/git-repositories/{repo_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_repo(
    workspace_id: str,
    repo_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    _require_access(user, db, workspace_id)
    r = db.get(GitRepository, repo_id)
    if r is None or r.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "repo not found")
    db.delete(r)
    db.commit()
    return None


@router.post("/git-repositories/{repo_id}/publish")
def publish_repo(
    repo_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    payload: dict[str, Any] | None = None,
) -> dict:
    """Publish the linked Project's experiments/trials/runs to the repo.

    Body: ``{commit_message?, project_id?}``. ``project_id`` overrides
    ``repo.project_id`` (useful when a repo serves multiple projects).
    Runs synchronously; future cuts flip this to a Job.
    """
    r = db.get(GitRepository, repo_id)
    if r is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "repo not found")
    _require_access(user, db, r.workspace_id)
    if not r.enabled:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "repo is disabled — enable it first"
        )
    project_id = (payload or {}).get("project_id") or r.project_id
    if not project_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "repo isn't bound to a project; pass project_id in the body",
        )
    from pycaret_server.git.service import publish_project

    result = publish_project(
        db,
        r,
        project_id,
        commit_message=(payload or {}).get("commit_message"),
    )
    return result
