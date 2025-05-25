###############################################################################
# Dockerfile — team-zeal-project
###############################################################################
# Purpose   : Reproducible build for the MLOps image-classification pipeline
# Base image: python:3.10-slim (≈ 50 MB)
###############################################################################

###############################
# 1. Base image & env vars    #
###############################

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

###############################
# 2. System dependencies      #
###############################

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        make \
        python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

###############################
# 3. Python build tools       #
###############################

RUN pip install --upgrade pip setuptools wheel

###############################
# 4. Copy project metadata    #
#    (layer caching for deps) #
###############################

COPY pyproject.toml README.md LICENSE Makefile ./

###############################
# 5. Copy source code         #
###############################

COPY drift_detector_pipeline/ ./drift_detector_pipeline/
COPY conf/    ./conf/
COPY scripts/ ./scripts/
COPY tests/   ./tests/

###############################
# 6. Install project          #
###############################

RUN pip install .

###############################
# 7. Entrypoint & metadata    #
###############################

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

LABEL maintainer="Team Zeal <emontel1@depaul.edu, sbandara@depaul.edu, asankarc@depaul.edu>" \
      project="team-zeal-project" \
      description="MLOps pipeline for image classification with drift detection" \
      version="1.0.0"

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]
