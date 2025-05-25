
# DOCKERFILE

# Stage 1: Base Image and Environment Setup

# Ensure this Python version matches our project's requirements (pyproject.toml specifies >=3.10)
FROM python:3.13-slim AS base

# Set environment variables
ENV \
# Disable Python's output buffering to ensure logs are printed in real-time
PYTHONUNBUFFERED=1 \
# Disable writing .pyc files to disk
PYTHONDONTWRITEBYTECODE=1 \
# Configure pip to use cache (set to 'on' to disable cache)
PIP_NO_CACHE_DIR=off \
# Disable pip's version check
PIP_DISABLE_PIP_VERSION_CHECK=on \
# Set a default timeout for pip operations to avoid hanging indefinitely
PIP_DEFAULT_TIMEOUT=100

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file to the container.
COPY pyproject.toml README.md LICENSE Makefile ./

# Copy the rest of the application source code into the container
COPY drift_detector_pipeline/ ./drift_detector_pipeline/
COPY conf/ ./conf/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install system dependencies.
RUN pip install .

# Add labels for metadata (Optional but good practice)
LABEL maintainer="Team Zeal <EMONTEL1@depaul.edu, SBANDARA@depaul.edu, ASANKARC@depaul.edu>"
LABEL project="team-zeal-project"
LABEL description="MLOps pipeline for image classification with drift detection. Includes tools for data versioning, training, and experiment tracking."
LABEL version="1.0.0"