# api/Dockerfile
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    APP_HOME=/app \
    # PORT will be set by Cloud Functions/Run environment. Default to 8080 if run directly.
    PORT=8080

WORKDIR $APP_HOME

# Install any system dependencies your API might need (e.g., for Pillow, opencv-python-headless)
# Example: RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
# For this project, Pillow (from torchvision) might need libjpeg-dev, libpng-dev etc.
# python:3.10-slim should have many common ones, but if issues arise, add them here.

COPY ./requirements.txt .
# Ensure gunicorn is installed for production-grade ASGI server if preferred by some GCP setups,
# uvicorn with --workers 1 is also fine for many serverless scenarios.
# Sticking to uvicorn as per exercise example's CMD.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire API directory content (main.py, etc.) into the APP_HOME directory in the image
COPY . .

# Make sure the port environment variable is used by uvicorn
# The default host 0.0.0.0 makes the app accessible from outside the container.
# Cloud Functions (2nd gen) / Cloud Run will set the PORT environment variable.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}", "--workers", "1"]
