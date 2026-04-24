"""FastAPI dependencies for authenticated routes."""

from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from pycaret_server.auth.tokens import TokenPayload, decode_token
from pycaret_server.db import User, get_db

_bearer = HTTPBearer(auto_error=False)


def _auth_error(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Resolve the authenticated user from the ``Authorization: Bearer <jwt>`` header.

    Raises 401 on any failure. Use ``Annotated[User, Depends(get_current_user)]``
    in route signatures.
    """
    if creds is None:
        raise _auth_error("missing Authorization header")
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
