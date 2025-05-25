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

# ----------------------------------------------------------------------------- #
# Docker image
# ----------------------------------------------------------------------------- #

IMAGE_NAME ?= $(PROJECT_NAME)
IMAGE_TAG  ?= 1.0.0
DOCKER_SHM_SIZE := --shm-size=10g

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
	@echo "Checking service‑account key: $(HOST_SERVICE_ACCOUNT_KEY_PATH)"
ifeq ($(OS),Windows_NT)
	@if not exist "$(subst /,\,$(HOST_SERVICE_ACCOUNT_KEY_PATH))" (echo "ERROR: key not found" && exit /b 1)
else
	@[ -f "$(HOST_SERVICE_ACCOUNT_KEY_PATH)" ] || (echo "ERROR: key not found" && exit 1)
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
		$(DOCKER_SHM_SIZE) $(IMAGE_NAME):$(IMAGE_TAG) bash

## Pull DVC data inside Docker container
.PHONY: docker_dvc_pull
docker_dvc_pull: ensure_host_dvc_cache check_service_account_key docker_build
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(USER_ARGS) \
		$(DOCKER_SHM_SIZE) $(IMAGE_NAME):$(IMAGE_TAG) make dvc_pull

## Train model inside Docker container
.PHONY: docker_train
docker_train: ensure_host_dvc_cache check_service_account_key docker_build
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(USER_ARGS) \
		$(DOCKER_SHM_SIZE) $(IMAGE_NAME):$(IMAGE_TAG) make train HYDRA_ARGS="$(HYDRA_ARGS)"

## Run tests inside Docker container
.PHONY: docker_test
docker_test: ensure_host_dvc_cache check_service_account_key docker_build
	docker run -it --rm \
		$(DOCKER_VOLUMES) $(GDRIVE_ENV_ARGS) $(WANDB_ARGS) $(USER_ARGS) \
		$(DOCKER_SHM_SIZE) $(IMAGE_NAME):$(IMAGE_TAG) make test

# ----------------------------------------------------------------------------- #
# Host‑side utilities
# ----------------------------------------------------------------------------- #

## Pull dataset with DVC on host
.PHONY: dvc_pull
dvc_pull: ensure_host_dvc_cache check_service_account_key
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
