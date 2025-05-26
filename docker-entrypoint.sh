#!/usr/bin/env bash
###############################################################################
# docker-entrypoint.sh - team-zeal-project
# --------------------------------------------------------------------------- #
# • Loads a Google-Drive service-account key (JSON) into GDRIVE_CREDENTIALS_DATA
#   so that DVC can authenticate to a GDrive remote.
# • Accepts either:
#     1) Path inside the container   ($GDRIVE_KEY_FILE_PATH_IN_CONTAINER), or
#     2) Raw JSON string             ($GDRIVE_CREDENTIALS_DATA_CONTENT).
# • Falls back to cached DVC tokens if neither is supplied.
###############################################################################
set -euo pipefail

log()   { echo "[entrypoint] $*"; }
fatal() { log "ERROR - $*" >&2; exit 1; }

###############################################
# 1. Load credentials                         #
###############################################

if [[ -n "${GDRIVE_KEY_FILE_PATH_IN_CONTAINER:-}" ]]; then
    KEY_PATH="$GDRIVE_KEY_FILE_PATH_IN_CONTAINER"
    [[ -f "$KEY_PATH"        ]] || fatal "File not found: $KEY_PATH"
    [[ -r "$KEY_PATH"        ]] || fatal "Cannot read file (check permissions): $KEY_PATH"
    GDRIVE_KEY_CONTENT="$(tr -d '\n\r' < "$KEY_PATH")"
elif [[ -n "${GDRIVE_CREDENTIALS_DATA_CONTENT:-}" ]]; then
    GDRIVE_KEY_CONTENT="$GDRIVE_CREDENTIALS_DATA_CONTENT"
else
    log "WARNING - No GDrive credentials provided. Cached tokens must exist for auth."
fi

# Validate JSON & export
if [[ -n "${GDRIVE_KEY_CONTENT:-}" ]]; then
    echo "$GDRIVE_KEY_CONTENT" | python -m json.tool >/dev/null 2>&1 \
        || fatal "Provided credentials are not valid JSON"
    export GDRIVE_CREDENTIALS_DATA="$GDRIVE_KEY_CONTENT"
    log "GDRIVE_CREDENTIALS_DATA environment variable populated. (Content hidden)"
fi

###############################################
# 2. Execute CMD / passed arguments           #
###############################################

log "Executing: $*"
exec "$@"
