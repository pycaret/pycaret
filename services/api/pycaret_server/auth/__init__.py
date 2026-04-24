"""Authentication helpers.

- Password hashing (bcrypt).
- JWT access-token + refresh-token encoding / decoding.
- FastAPI dependencies for current-user resolution.

Trusted publisher / OAuth integrations come later as plugins.
"""

from pycaret_server.auth.deps import (
    CurrentUser,
    get_current_user,
    require_admin,
)
from pycaret_server.auth.passwords import hash_password, verify_password
from pycaret_server.auth.tokens import (
    TokenPayload,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_refresh_token,
)

__all__ = [
    "CurrentUser",
    "TokenPayload",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "get_current_user",
    "hash_password",
    "hash_refresh_token",
    "require_admin",
    "verify_password",
]
