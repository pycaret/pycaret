#!/bin/sh
# PyCaret API container entrypoint.
#
# Responsibilities:
#   1. Make PYCARET_SECRETS_KEY deterministic across restarts.
#      - If the user supplied one via env, use it.
#      - Else, generate one on first run, persist to /data/.secrets/fernet.key,
#        and reuse it on every subsequent start.
#      Without this step the api auto-rotates an ephemeral key each restart,
#      which silently breaks every previously-encrypted secret in the DB
#      (we lived this bug in session 55 — don't repeat it).
#   2. Exec the actual server. (Replace this process so docker stop signals
#      reach uvicorn directly — no "leaked PID 1" weirdness.)

set -eu

SECRETS_DIR="/data/.secrets"
KEY_FILE="$SECRETS_DIR/fernet.key"

if [ -z "${PYCARET_SECRETS_KEY:-}" ]; then
    if [ ! -f "$KEY_FILE" ]; then
        mkdir -p "$SECRETS_DIR"
        # Generate a Fernet key (base64-urlsafe 32 bytes). We use python's
        # cryptography (already a runtime dep — it powers crypto.py).
        python -c "from cryptography.fernet import Fernet; import sys; sys.stdout.write(Fernet.generate_key().decode())" > "$KEY_FILE"
        chmod 600 "$KEY_FILE"
        echo "[entrypoint] generated new PYCARET_SECRETS_KEY → $KEY_FILE (0600)"
    fi
    PYCARET_SECRETS_KEY="$(cat "$KEY_FILE")"
    export PYCARET_SECRETS_KEY
    echo "[entrypoint] using persisted PYCARET_SECRETS_KEY from $KEY_FILE"
else
    echo "[entrypoint] using PYCARET_SECRETS_KEY from environment"
fi

exec "$@"
