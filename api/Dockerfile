# api/Dockerfile
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    APP_HOME=/app

WORKDIR $APP_HOME

# Install any system dependencies ... (as before)

COPY ./requirements.txt .
# Ensure gunicorn is installed for production-grade ASGI server if preferred by some GCP setups,
# uvicorn with --workers 1 is also fine for many serverless scenarios.
# Sticking to uvicorn as per exercise example's CMD.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire API directory content (main.py, etc.) into the APP_HOME directory in the image
COPY . .

# Ensure the entrypoint script is executable
RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
# CMD is now optional, or can be used to pass default arguments to the entrypoint.
# For uvicorn started by entrypoint.sh, CMD is often left empty or as ["main:app"]
# if the entrypoint was more generic. Here, our entrypoint is specific to uvicorn.
# An empty CMD or a CMD like ["--log-level", "info"] could pass additional args to uvicorn via "$@" in entrypoint.
# Let's keep it simple for now:
CMD []
