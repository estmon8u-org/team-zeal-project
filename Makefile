################################################################################
# Makefile — team-zeal-project
################################################################################
#
# High‑level commands:
#   make docker_build        Build Docker image
#   make docker_shell        Interactive shell in container
#   make docker_train        Train model in container
#   make train               Train model on host
#   make test                Run tests
#   make lint / format       Static code checks / auto‑format
#   make clean               Remove caches & artifacts
#   make help                Show all rules
#
################################################################################

# ----------------------------------------------------------------------------- #
# Global settings
# ----------------------------------------------------------------------------- #

PROJECT_NAME       := team-zeal-project
PYTHON_VERSION     := 3.10
PYTHON_INTERPRETER := python
CI_MODE ?= false

# ----------------------------------------------------------------------------- #
# Docker image
# ----------------------------------------------------------------------------- #

ifeq ($(CI_MODE),true)
    # Container must fit into ~3.7 GiB host RAM
    DOCKER_MEMORY_OPTS := --memory=3500m --memory-swap=4g --shm-size=1g
    # Lean training/pull defaults for CI
    DEFAULT_HYDRA_ARGS := run.device=cpu data.dataloader_workers=1 training.batch_size=32
    DVC_PARALLEL_JOBS  := -j 2          # dvc pull concurrency
else
    # Local / self-hosted runners
    DOCKER_MEMORY_OPTS ?= --memory=8g --memory-swap=10g --shm-size=4g
    DEFAULT_HYDRA_ARGS :=
    DVC_PARALLEL_JOBS  :=
endif

# ----------------------------------------------------------------------------- #
# DVC cache
# ----------------------------------------------------------------------------- #

HOST_DVC_CACHE_DIR ?= $(HOME)/.cache/dvc      # default on Linux/macOS
ifeq ($(OS),Windows_NT)
	HOST_DVC_CACHE_DIR := $(USERPROFILE)/.cache/dvc
endif
CONTAINER_DVC_CACHE_PATH := /root/.dvc/cache   # DVC default inside image

# ----------------------------------------------------------------------------- #
# Google‑Drive service account (for DVC remote)
# ----------------------------------------------------------------------------- #

HOST_SERVICE_ACCOUNT_KEY_PATH ?= $(CURDIR)/.secrets/gdrive-dvc-service-account.json
CONTAINER_KEY_FILE_PATH       := /app/.secrets/gdrive-dvc-service-account.json

# ----------------------------------------------------------------------------- #
# Weights & Biases
# ----------------------------------------------------------------------------- #

WANDB_ARGS :=
ifdef WANDB_API_KEY
	WANDB_ARGS += -e WANDB_API_KEY=$(WANDB_API_KEY)
endif

# ----------------------------------------------------------------------------- #
# Docker run helper arguments
# ----------------------------------------------------------------------------- #

ifeq ($(OS),Windows_NT)
	USER_ARGS :=
	DOCKER_VOLUMES := -v "$(CURDIR):/app" \
					  -v "$(subst /,\,$(HOST_DVC_CACHE_DIR)):$(CONTAINER_DVC_CACHE_PATH)"
	GDRIVE_ENV_ARGS := -v "$(subst /,\,$(HOST_SERVICE_ACCOUNT_KEY_PATH)):$(CONTAINER_KEY_FILE_PATH):ro" \
					   -e GDRIVE_KEY_FILE_PATH_IN_CONTAINER=$(CONTAINER_KEY_FILE_PATH)
else
	USER_ARGS := --user "$$(id -u):$$(id -g)"
	DOCKER_VOLUMES := -v "$(CURDIR):/app" \
					  -v "$(HOST_DVC_CACHE_DIR):$(CONTAINER_DVC_CACHE_PATH)"
	GDRIVE_ENV_ARGS := -v "$(HOST_SERVICE_ACCOUNT_KEY_PATH):$(CONTAINER_KEY_FILE_PATH):ro" \
					   -e GDRIVE_KEY_FILE_PATH_IN_CONTAINER=$(CONTAINER_KEY_FILE_PATH)
endif

# ----------------------------------------------------------------------------- #
# Helper targets (internal)
# ----------------------------------------------------------------------------- #

.PHONY: ensure_host_dvc_cache
ensure_host_dvc_cache:
	@echo "Ensuring host DVC cache exists: $(HOST_DVC_CACHE_DIR)"
ifeq ($(OS),Windows_NT)
	@if not exist "$(subst /,\,$(HOST_DVC_CACHE_DIR))" mkdir "$(subst /,\,$(HOST_DVC_CACHE_DIR))"
else
	@mkdir -p "$(HOST_DVC_CACHE_DIR)"
endif

.PHONY: check_service_account_key
check_service_account_key:
ifeq ($(CI_MODE),true)
	@echo "CI Mode: Skipping physical service account key file check. Assuming GDRIVE_CREDENTIALS_DATA is set by entrypoint."
else
	@echo "Local Mode: Checking for host service-account key at $(HOST_SERVICE_ACCOUNT_KEY_PATH)..."
ifeq ($(OS),Windows_NT)
	@if not exist "$(subst /,\,$(HOST_SERVICE_ACCOUNT_KEY_PATH))" ( \
		echo "ERROR: Service account key NOT FOUND at $(HOST_SERVICE_ACCOUNT_KEY_PATH)!" >&2 && \
		echo "Ensure the key file exists or HOST_SERVICE_ACCOUNT_KEY_PATH is set correctly." >&2 && \
		exit /b 1 \
	)
else
	@if [ ! -f "$(HOST_SERVICE_ACCOUNT_KEY_PATH)" ]; then \
		echo "ERROR: Service account key NOT FOUND at $(HOST_SERVICE_ACCOUNT_KEY_PATH)!" >&2; \
		echo "Ensure the key file exists or HOST_SERVICE_ACCOUNT_KEY_PATH is set correctly." >&2; \
		exit 1; \
	fi
endif
	@echo "Host service-account key found."
endif

# ----------------------------------------------------------------------------- #
# Docker
# ----------------------------------------------------------------------------- #

## Build Docker image
.PHONY: docker_build
docker_build:
	@echo "Building Docker image $(IMAGE_NAME):$(IMAGE_TAG)"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

## Interactive shell inside Docker container
.PHONY: docker_shell
docker_shell: ensure_host_dvc_cache check_service_account_key
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(USER_ARGS) \
		$(DOCKER_MEMORY_OPTS) $(IMAGE_NAME):$(IMAGE_TAG) bash

## Pull DVC data inside Docker container
.PHONY: docker_dvc_pull
docker_dvc_pull: ensure_host_dvc_cache check_service_account_key docker_build
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(USER_ARGS) \
		$(DOCKER_MEMORY_OPTS) $(IMAGE_NAME):$(IMAGE_TAG) make dvc_pull

## Train model inside Docker container
.PHONY: docker_train
docker_train: ensure_host_dvc_cache check_service_account_key docker_build
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(USER_ARGS) \
		$(DOCKER_MEMORY_OPTS) $(IMAGE_NAME):$(IMAGE_TAG) make train HYDRA_ARGS="$(HYDRA_ARGS)"

## Run tests inside Docker container
.PHONY: docker_test
docker_test: ensure_host_dvc_cache check_service_account_key docker_build
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(USER_ARGS) \
		$(DOCKER_MEMORY_OPTS) $(IMAGE_NAME):$(IMAGE_TAG) make test

# ----------------------------------------------------------------------------- #
# Host‑side utilities
# ----------------------------------------------------------------------------- #

## Pull dataset with DVC on host
.PHONY: dvc_pull
dvc_pull: ensure_host_dvc_cache check_service_account_key
	@echo "Pulling DVC tracked data (python -m dvc pull)..."
ifeq ($(CI_MODE),true)
	@echo "CI Mode: Using DVC with GDRIVE_CREDENTIALS_DATA (set by entrypoint)."
else
# For host `dvc pull`, it needs to know about the service account.
# User should set GDRIVE_CREDENTIALS_DATA in host shell or have DVC configured for the key file.
	@echo "Host Mode: Ensure DVC is configured for GDrive auth (e.g., GDRIVE_CREDENTIALS_DATA set in shell)."
endif
	$(PYTHON_INTERPRETER) -m dvc pull data/raw/imagenette2-160.tgz.dvc -r gdrive

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -e .

## Remove Python & build caches
.PHONY: clean
clean:
	$(PYTHON_INTERPRETER) scripts/clean.py

## Static code analysis
.PHONY: lint
lint:
	$(PYTHON_INTERPRETER) -m ruff format --check .
	$(PYTHON_INTERPRETER) -m ruff check .

## Auto‑format code
.PHONY: format
format:
	$(PYTHON_INTERPRETER) -m ruff format .
	$(PYTHON_INTERPRETER) -m ruff check --fix .

## Run unit tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests

## Train model on host
.PHONY: train
train: process_data
	$(PYTHON_INTERPRETER) -m drift_detector_pipeline.modeling.train

## Create virtual environment (.venv)
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo "Activate with: source .venv/bin/activate (Linux/macOS) or .venv\\Scripts\\activate.bat (Windows)"

# ----------------------------------------------------------------------------- #
# Data pipeline
# ----------------------------------------------------------------------------- #

## Extract and preprocess dataset
.PHONY: process_data
process_data: dvc_pull data/raw/imagenette2-160.tgz
	$(PYTHON_INTERPRETER) -m drift_detector_pipeline.dataset

# Placeholder prerequisite — the tarball is obtained via dvc_pull
data/raw/imagenette2-160.tgz:
	@echo "Run 'make dvc_pull' first to download the dataset."

# ----------------------------------------------------------------------------- #
# Help (default target)
# ----------------------------------------------------------------------------- #

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

## Show this help
.PHONY: help
help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
