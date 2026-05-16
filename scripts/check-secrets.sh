#!/usr/bin/env bash
# check-secrets.sh — pre-push gate for accidental secret commits.
#
# Run manually:   bash scripts/check-secrets.sh
# As pre-push hook:
#   cp scripts/check-secrets.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push
#
# Scans the diff staged for push (or the working tree if no remote set) for
# patterns that look like real credentials. Designed to fail closed — false
# positives are preferred over a leaked key.
#
# Allow-list a single line by appending ``# pragma: allow-secret`` to it.
# Allow-list a whole file by listing its path in scripts/.secrets-allowlist.

set -euo pipefail

RED=$'\033[31m'; YELLOW=$'\033[33m'; GREEN=$'\033[32m'; RESET=$'\033[0m'

ROOT="$(git rev-parse --show-toplevel)"
ALLOWLIST="$ROOT/scripts/.secrets-allowlist"

# What we're scanning. If invoked as a git pre-push hook, git pipes us
# the refs being pushed on stdin (we use that to scan only the diff). For
# any other invocation (manual `bash scripts/check-secrets.sh`, CI), we
# default to scanning every tracked file — safer.
#
# Detecting "am I a pre-push hook?" via `[[ -t 0 ]]` is unreliable
# (subprocess stdin is often not a TTY even outside hooks). We use the
# script's own name: git names the hook ``pre-push``.

TARGETS=""
if [[ "${0##*/}" == "pre-push" ]]; then
    while read -r local_ref local_sha remote_ref remote_sha; do
        [[ "$local_sha" == "0000000000000000000000000000000000000000" ]] && continue
        if [[ "$remote_sha" == "0000000000000000000000000000000000000000" ]]; then
            range="$local_sha"
        else
            range="${remote_sha}..${local_sha}"
        fi
        TARGETS+=$'\n'$(git diff --name-only "$range" --diff-filter=AM)
    done
    TARGETS="${TARGETS#$'\n'}"
fi
if [[ -z "$TARGETS" ]]; then
    TARGETS=$(git ls-files)
fi

if [[ -z "${TARGETS:-}" ]]; then
    echo "${GREEN}check-secrets: nothing to scan${RESET}"
    exit 0
fi

# Build the allowlist filter (files we should skip entirely).
ALLOWED_FILES=()
if [[ -f "$ALLOWLIST" ]]; then
    while IFS= read -r line; do
        # Skip blank lines + comments.
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        ALLOWED_FILES+=("$line")
    done < "$ALLOWLIST"
fi

is_allowed() {
    local path="$1"
    for pat in "${ALLOWED_FILES[@]:-}"; do
        # shellcheck disable=SC2053
        [[ "$path" == $pat ]] && return 0
    done
    return 1
}

# Patterns to scan for. Anchored where possible to reduce false positives.
# Each pattern: <name>::<regex>
PATTERNS=(
    'Anthropic API key::sk-ant-[A-Za-z0-9_-]{20,}'
    'OpenAI API key (sk-)::\bsk-[A-Za-z0-9]{32,}\b'
    'OpenAI API key (sk-proj-)::sk-proj-[A-Za-z0-9_-]{40,}'
    'OpenAI API key (sk-svcacct-)::sk-svcacct-[A-Za-z0-9_-]{40,}'
    'Stripe live secret::sk_live_[A-Za-z0-9]{20,}'
    'Slack bot token::xox[abprs]-[A-Za-z0-9-]{20,}'
    'GitHub PAT::ghp_[A-Za-z0-9]{36,}'
    'GitHub fine-grained PAT::github_pat_[A-Za-z0-9_]{50,}'
    'AWS access key id::\bAKIA[0-9A-Z]{16}\b'
    'AWS secret access key (heuristic)::aws_secret_access_key[[:space:]]*=[[:space:]]*[A-Za-z0-9/+=]{40,}'
    'Google API key::\bAIza[A-Za-z0-9_-]{35}\b'
    'Fernet-encrypted blob in source::ENC:v1:[A-Za-z0-9_=-]{40,}'
    'PEM private key block::-----BEGIN [A-Z ]*PRIVATE KEY-----'
)

violations=0
echo "${GREEN}check-secrets: scanning $(echo "$TARGETS" | wc -l | tr -d ' ') files…${RESET}"

while IFS= read -r f; do
    [[ -z "$f" || ! -f "$ROOT/$f" ]] && continue
    if is_allowed "$f"; then
        continue
    fi
    for entry in "${PATTERNS[@]}"; do
        name="${entry%%::*}"
        regex="${entry#*::}"
        # -P would be cleaner but BSD grep doesn't have it. -E + sane regex.
        matches=$(grep -nE "$regex" "$ROOT/$f" 2>/dev/null || true)
        if [[ -n "$matches" ]]; then
            # Filter out lines explicitly allowed.
            filtered=$(echo "$matches" | grep -v '# pragma: allow-secret' || true)
            if [[ -n "$filtered" ]]; then
                echo "${RED}✗ ${name}${RESET} in ${YELLOW}${f}${RESET}:"
                echo "$filtered" | sed 's/^/    /'
                violations=$((violations + 1))
            fi
        fi
    done
done <<< "$TARGETS"

if (( violations > 0 )); then
    echo
    echo "${RED}check-secrets: ${violations} suspected secret(s) found — push BLOCKED.${RESET}"
    echo "Options:"
    echo "  1. Remove the secret + rotate the key it leaked from."
    echo "  2. If it's a legitimate test fixture, add"
    echo "     '# pragma: allow-secret' to that exact line."
    echo "  3. To allow-list a whole file, append its path to"
    echo "     scripts/.secrets-allowlist (one path per line, glob OK)."
    exit 1
fi

echo "${GREEN}check-secrets: clean.${RESET}"
exit 0
