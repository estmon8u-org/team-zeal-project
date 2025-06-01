#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status.

# Default port if PORT environment variable is not set
APP_PORT=${PORT:-8080}

echo "Starting Uvicorn server on host 0.0.0.0, port ${APP_PORT}..."

# Use exec to replace the shell process with the uvicorn process.
# This ensures uvicorn receives signals directly.
exec uvicorn main:app --host 0.0.0.0 --port "${APP_PORT}" --workers 1 "$@"
# "$@" passes any arguments given to the entrypoint script along to uvicorn
# (though in this CMD usage, there won't be any CMD args passed here)
