#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = team-zeal-project
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies into the active virtual environment
# Ensures pyproject.toml is used for editable install.
.PHONY: requirements
requirements:
	@echo "Installing dependencies from pyproject.toml..."
	$(PYTHON_INTERPRETER) -m pip install -e .
	@echo "Dependencies installed."

## Pull DVC tracked data
# Assumes DVC is configured and remote is accessible.
.PHONY: dvc_pull
dvc_pull:
	@echo "Pulling DVC tracked data..."
	$(PYTHON_INTERPRETER) -m dvc pull
	@echo "DVC pull complete."


## Delete Python cache files and other generated artifacts (cross-platform)
.PHONY: clean
clean:
	@echo "Cleaning Python cache files and generated artifacts using scripts/clean.py..."
	$(PYTHON_INTERPRETER) scripts/clean.py
	@echo "Cleaning complete."


## Lint code using Ruff (checks formatting and style)
.PHONY: lint
lint:
	@echo "Running Ruff linter and formatter check..."
	$(PYTHON_INTERPRETER) -m ruff format --check .
	$(PYTHON_INTERPRETER) -m ruff check .
	@echo "Linting check complete."

## Format source code using Ruff
.PHONY: format
format:
	@echo "Formatting code with Ruff..."
	$(PYTHON_INTERPRETER) -m ruff format .
	$(PYTHON_INTERPRETER) -m ruff check --fix .
	@echo "Formatting complete."


## Run tests using Pytest
.PHONY: test
test:
	@echo "Running tests..."
	$(PYTHON_INTERPRETER) -m pytest tests
	@echo "Tests finished."

## Run the model training script
# Ensure data is processed before training.
.PHONY: train
train: process_data
	@echo "Starting model training..."
	$(PYTHON_INTERPRETER) -m drift_detector_pipeline.modeling.train
	@echo "Training finished."

## Create a Python virtual environment
# This command guides the user. Actual activation is manual.
.PHONY: create_environment
create_environment:
	@echo "Creating Python virtual environment in $(VENV_DIR)..."
	$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "Virtual environment created in $(VENV_DIR)."
	@echo "To activate it:"
ifeq ($(OS),Windows_NT)
	@echo "  On Windows (cmd): $(VENV_DIR)\\Scripts\\activate.bat"
	@echo "  On Windows (PowerShell): .\\$(VENV_DIR)\\Scripts\\Activate.ps1"
else
	@echo "  On Linux/macOS: source $(VENV_DIR)/bin/activate"
endif
	@echo "Then run 'make requirements' to install dependencies."


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Process raw data: ensures raw data is present (via DVC) and then extracts/processes it.
# The target 'data/processed/imagenette2-160' is a directory.
# The Python script itself handles idempotency (not re-extracting if already done).
.PHONY: process_data  # Marking as .PHONY as its state isn't well-tracked by a single file
process_data: dvc_pull data/raw/imagenette2-160.tgz
	@echo "Processing raw data (extracting imagenette2-160.tgz)..."
	$(PYTHON_INTERPRETER) -m drift_detector_pipeline.dataset
	@echo "Data processing complete."

# Prerequisite for data processing: the raw tarball.
# This rule doesn't do anything itself but declares the dependency.
# The actual fetching of this file is handled by 'make dvc_pull'.
data/raw/imagenette2-160.tgz:
	@echo "Raw data file data/raw/imagenette2-160.tgz is expected."
	@echo "Run 'make dvc_pull' to ensure it is downloaded via DVC."


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
