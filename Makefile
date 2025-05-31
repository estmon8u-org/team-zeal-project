################################################################################
# Makefile — team-zeal-project
################################################################################
#
# High‑level commands:
#   make docker_build        Build Docker image
#   make docker_shell        Interactive shell in container
#   make docker_train        Train model in container. Pass Hydra args via ARGS variable.
#                            e.g., make docker_train ARGS="training.epochs=1 training.profiler.enabled=true"
#   make train               Train model on host. Pass Hydra args via ARGS variable.
#                            e.g., make train ARGS="training.epochs=1 training.profiler.enabled=true"
#   make test                Run tests. Pass pytest args via ARGS variable.
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

# Users will pass Hydra arguments like: make train ARGS="training.epochs=1 foo=bar"
ARGS ?=

# ----------------------------------------------------------------------------- #
# Docker image
# ----------------------------------------------------------------------------- #

IMAGE_NAME ?= team-zeal-project
IMAGE_TAG ?= 1.0.0
FULL_IMAGE_NAME := $(IMAGE_NAME):$(IMAGE_TAG)


# Allows us to choose different memory settings based on the environment
# for now we will keep them the same for both CI and local runs
ifeq ($(CI_MODE),true)
# CI runners (GitHub Actions, etc.)
    DOCKER_MEMORY_OPTS := --memory=32g --memory-swap=32g --shm-size=16g  --ipc=host
else
# Local / self-hosted runners
    DOCKER_MEMORY_OPTS ?= --memory=32g --memory-swap=32g --shm-size=16g  --ipc=host
endif

# ----------------------------------------------------------------------------- #
# DVC cache
# ----------------------------------------------------------------------------- #

HOST_DVC_CACHE_DIR ?= $(HOME)/.cache/dvc
ifeq ($(OS),Windows_NT)
	HOST_DVC_CACHE_DIR := $(USERPROFILE)/.cache/dvc
endif
CONTAINER_DVC_CACHE_PATH := /root/.dvc/cache
DVC_PARALLEL_JOBS  := -j 2

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
DOCKER_ENV_FILE_ARG :=
ifneq ("$(wildcard .env)","")
    DOCKER_ENV_FILE_ARG := --env-file .env
endif

# ----------------------------------------------------------------------------- #
# Docker run helper arguments
# ----------------------------------------------------------------------------- #
OS_FAMILY := Unknown
ifeq ($(OS),Windows_NT)
    OS_FAMILY := Windows
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OS_FAMILY := Linux
    else ifeq ($(UNAME_S),Darwin)
        OS_FAMILY := Darwin
    else ifeq ($(findstring MINGW,$(UNAME_S)),MINGW)
        OS_FAMILY := Windows
    else ifeq ($(findstring CYGWIN,$(UNAME_S)),CYGWIN)
        OS_FAMILY := Windows
    endif
endif

ifeq ($(OS_FAMILY),Windows)
	USER_ARGS :=
	DOCKER_VOLUMES := -v "$(CURDIR):/app" \
					  -v "$(HOST_DVC_CACHE_DIR):$(CONTAINER_DVC_CACHE_PATH)" \
					  -v "$(HOST_SERVICE_ACCOUNT_KEY_PATH):$(CONTAINER_KEY_FILE_PATH):ro"
else # Linux or Darwin (macOS)
	USER_ARGS := --user "$$(id -u):$$(id -g)"
	DOCKER_VOLUMES := -v "$(CURDIR):/app" \
					  -v "$(HOST_DVC_CACHE_DIR):$(CONTAINER_DVC_CACHE_PATH)" \
					  -v "$(HOST_SERVICE_ACCOUNT_KEY_PATH):$(CONTAINER_KEY_FILE_PATH):ro"
endif
GDRIVE_ENV_ARGS := -e GDRIVE_KEY_FILE_PATH_IN_CONTAINER=$(CONTAINER_KEY_FILE_PATH)

# ----------------------------------------------------------------------------- #
# Helper targets (internal)
# ----------------------------------------------------------------------------- #

.PHONY: ensure_host_dvc_cache
ensure_host_dvc_cache:
	@echo "Ensuring host DVC cache exists: $(HOST_DVC_CACHE_DIR)"
ifeq ($(OS_FAMILY),Windows)
	@if not exist "$(HOST_DVC_CACHE_DIR)" mkdir "$(HOST_DVC_CACHE_DIR)"
else
	@mkdir -p "$(HOST_DVC_CACHE_DIR)"
endif

.PHONY: check_service_account_key
check_service_account_key:
ifeq ($(CI_MODE),true)
	@echo "CI Mode: Skipping physical service account key file check."
else
	@echo "Local Mode: Checking for host service-account key at $(HOST_SERVICE_ACCOUNT_KEY_PATH)..."
ifeq ($(OS_FAMILY),Windows)
	@if not exist "$(HOST_SERVICE_ACCOUNT_KEY_PATH)" ( \
		echo "ERROR: Service account key NOT FOUND at $(HOST_SERVICE_ACCOUNT_KEY_PATH)!" >&2 && \
		echo "Ensure key exists or HOST_SERVICE_ACCOUNT_KEY_PATH is set." >&2 && \
		exit /b 1 \
	)
else
	@if [ ! -f "$(HOST_SERVICE_ACCOUNT_KEY_PATH)" ]; then \
		echo "ERROR: Service account key NOT FOUND at $(HOST_SERVICE_ACCOUNT_KEY_PATH)!" >&2; \
		echo "Ensure key exists or HOST_SERVICE_ACCOUNT_KEY_PATH is set." >&2; \
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
	@echo "Building Docker image $(FULL_IMAGE_NAME)"
	docker build -t $(FULL_IMAGE_NAME) .

## Interactive shell inside Docker container
.PHONY: docker_shell
docker_shell: ensure_host_dvc_cache check_service_account_key
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(DOCKER_ENV_FILE_ARG) $(USER_ARGS) \
		$(DOCKER_MEMORY_OPTS) $(FULL_IMAGE_NAME) bash

## Pull DVC data inside Docker container
.PHONY: docker_dvc_pull
docker_dvc_pull: ensure_host_dvc_cache check_service_account_key docker_build
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(DOCKER_ENV_FILE_ARG) $(USER_ARGS) \
		$(DOCKER_MEMORY_OPTS) $(FULL_IMAGE_NAME) make dvc_pull

## Train model inside Docker container. Pass Hydra args via ARGS="arg1=val1 ..."
.PHONY: docker_train
docker_train: ensure_host_dvc_cache check_service_account_key docker_build
	@echo "Running 'make train ARGS=\"$(ARGS)\"' inside Docker container..."
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(DOCKER_ENV_FILE_ARG) $(USER_ARGS) \
		-e CI_MODE=$(CI_MODE) \
		$(DOCKER_MEMORY_OPTS) $(FULL_IMAGE_NAME) make train ARGS="$(ARGS)"

## Run tests inside Docker container. Pass pytest args via ARGS="..."
.PHONY: docker_test
docker_test: ensure_host_dvc_cache check_service_account_key docker_build
	@echo "Running 'make test ARGS=\"$(ARGS)\"' inside Docker container..."
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(DOCKER_ENV_FILE_ARG) $(USER_ARGS) \
		$(DOCKER_MEMORY_OPTS) $(FULL_IMAGE_NAME) make test ARGS="$(ARGS)"

## Train model with CML inside Docker container
.PHONY: docker_cml_train
docker_cml_train: ensure_host_dvc_cache check_service_account_key docker_build
	@echo "Running 'make train' with CML enabled inside Docker container..."
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(DOCKER_ENV_FILE_ARG) $(USER_ARGS) \
		-e CI_MODE=$(CI_MODE) \
		$(DOCKER_MEMORY_OPTS) $(FULL_IMAGE_NAME) make train ARGS="cml.enabled=true $(ARGS)"

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
	@echo "Host Mode: Ensure DVC is configured for GDrive auth (e.g., GDRIVE_CREDENTIALS_DATA set in shell or key file setup)."
endif
	$(PYTHON_INTERPRETER) -m dvc pull $(DVC_PARALLEL_JOBS) data/raw/imagenette2-160.tgz.dvc -r gdrive

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
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

## Run unit tests. Pass pytest args via ARGS="..."
.PHONY: test
test:
	@echo "Running tests on host with Pytest args: $(ARGS)..."
	$(PYTHON_INTERPRETER) -m pytest tests/ $(ARGS)

## Train model on host. Pass Hydra args via ARGS="..."
.PHONY: train
train: process_data
	@echo "Starting model training on host with Hydra args: $(ARGS)..."
	$(PYTHON_INTERPRETER) -m drift_detector_pipeline.modeling.train $(ARGS)

## Create virtual environment (.venv)
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo "Activate with: source .venv/bin/activate (Linux/macOS) or .venv\\Scripts\\activate.bat (Windows)"

## Train model with CML reporting enabled
.PHONY: cml_train
cml_train: process_data
	@echo "Starting model training with CML reporting enabled..."
	$(PYTHON_INTERPRETER) -m drift_detector_pipeline.modeling.train cml.enabled=true $(ARGS)


# ----------------------------------------------------------------------------- #
# Data pipeline
# ----------------------------------------------------------------------------- #

## Extract and preprocess dataset. Pass Hydra args via ARGS="..."
.PHONY: process_data
process_data: dvc_pull data/raw/imagenette2-160.tgz
	@echo "Processing raw data (extracting imagenette2-160.tgz) with Hydra args: $(ARGS)..."
	$(PYTHON_INTERPRETER) -m drift_detector_pipeline.dataset $(ARGS)

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
