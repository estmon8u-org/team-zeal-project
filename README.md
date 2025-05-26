# Image Classification with Drift Detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps pipeline for image classification on Imagenette-160. This project utilizes Docker for containerization, DVC for data versioning with Google Drive (via Service Accounts), Hydra for configuration, Weights & Biases for experiment tracking, PyTorch Profiler for performance analysis, and GitHub Actions for CI/CD.

## Project Organization

```
├── .dockerignore        <- Specifies files to exclude from Docker image
├── .dvc/                <- DVC metadata and configuration
├── .github/workflows/   <- GitHub Actions CI/CD workflows
├── .gitignore           <- Specifies intentionally untracked files for Git
├── .secrets/            <- (Gitignored) Directory for sensitive keys like service accounts
├── LICENSE              <- Open-source license
├── Makefile             <- Convenience commands for building, training, testing, etc.
├── PHASE1.md            <- Documentation for Phase 1 deliverables
├── PHASE2.md            <- Documentation for Phase 2 deliverables
├── README.md            <- This file: project overview and instructions
├── Dockerfile           <- Instructions to build the project's Docker image
├── conf/                <- Hydra configuration files (e.g., config.yaml)
├── data/                <- (DVC-managed, gitignored) Project data (raw, processed)
├── docker-entrypoint.sh <- Script executed when Docker container starts
├── docs/                <- Project documentation (e.g., for MkDocs)
├── drift_detector_pipeline/ <- Python source code for the project
├── models/              <- (DVC-managed or gitignored) Trained models
├── notebooks/           <- Jupyter notebooks for exploration
├── outputs/             <- (Gitignored) Hydra outputs, logs, saved models from runs
├── pyproject.toml       <- Python project metadata and dependencies (PEP 517/518)
├── requirements.txt     <- Pinned Python dependencies (for reference or specific envs)
└── tests/               <- Unit and integration tests
```

---

## 1. Team Information
-   **Team Name:** Team Zeal
-   **Team Members:**
    -   Montelongo, Esteban - EMONTEL1@depaul.edu
    -   Bandara, Sajith - SBANDARA@depaul.edu
    -   Sankar Chandrasekar, Arjun Kumar - ASANKARC@depaul.edu
-   **Course & Section:** SE 489: ML Engineering For Production (MLOps)

## 2. Project Overview
-   **Summary:** This project implements an end-to-end MLOps pipeline to train an image classification model (ResNet-18 using `timm`) on the Imagenette-160 dataset. It incorporates containerization with Docker, data versioning with DVC and Google Drive, experiment tracking with Weights & Biases, configuration management with Hydra, performance profiling with PyTorch Profiler, and CI/CD with GitHub Actions.
-   **Problem Statement:** Machine learning models in production often suffer performance degradation due to data drift. This project aims to build robust, reproducible, and analyzable training pipelines as a foundation for systems that can automatically detect such drift and maintain model performance.
-   **Key Objectives:**
    -   **Phase 1:** Establish a reproducible baseline training pipeline, DVC, Hydra, WandB integration, and unit tests. *(Status: Completed)*
    -   **Phase 2:** Enhance with Docker containerization, advanced logging, code profiling, and CI/CD. *(Status: Docker & Profiling implemented)*

## 3. Project Architecture Diagram (Phase 1)
![Phase 1 Architecture Diagram](./phase1.jpg)
*(This diagram represents the components set up in Phase 1: Data acquisition/versioning with DVC/G-Drive, Code versioning with Git/GitHub, Training pipeline using PyTorch/timm, Configuration via Hydra, Experiment Tracking via WandB, Unit Testing via Pytest, and local execution via Make.)*

## 4. Phase Deliverables
-   [X] [PHASE1.md](./PHASE1.md): Project Design & Model Development
-   [X] [PHASE2.md](./PHASE2.md): Enhancing ML Operations with Containerization & Monitoring *(Docker & Profiling implemented)*
-   [ ] PHASE3.md: Continuous ML & Deployment *(Upcoming)*

## 5. Setup Instructions

This section guides you through setting up the project environment for both local host development and Docker-based execution.

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/estmon8u/team-zeal-project.git # Or your SSH URL
    cd team-zeal-project
    ```

2.  **Prerequisites:**
    *   **Python:** Version 3.10 or higher (as defined in `pyproject.toml`).
    *   **Make:** GNU Make.
        *   Linux/macOS: Usually pre-installed.
        *   Windows: Requires installation (e.g., via Chocolatey `choco install make`, GnuWin32, or use Make within WSL/Git Bash).
    *   **Docker Desktop:** Required for building and running Docker containers. Download from [docker.com](https://www.docker.com/products/docker-desktop/). Ensure it's running.
    *   **Git:** For version control.

3.  **Create & Activate Virtual Environment (Recommended for Host Development):**
    ```bash
    make create_environment # Uses PYTHON_INTERPRETER defined in Makefile
    ```
    Activate the environment:
    *   Linux/macOS: `source .venv/bin/activate`
    *   Windows CMD: `.venv\Scripts\activate.bat`
    *   Windows PowerShell: `.\.venv\Scripts\Activate.ps1`

4.  **Install Python Dependencies (for Host Development):**
    (Ensure virtual environment is activated)
    ```bash
    make requirements
    ```

5.  **Set up Weights & Biases (WandB):**
    *   **Account:** Sign up at [wandb.ai](https://wandb.ai).
    *   **API Key for Local Development:**
        1.  Get your API key from `wandb.ai/authorize`.
        2.  Create a `.env` file in the project root (it's gitignored):
            ```env
            # .env (Do NOT commit this file)
            WANDB_API_KEY="YOUR_ACTUAL_WANDB_API_KEY_HERE"
            ```
        3.  The `Makefile` passes this key to Docker containers if the `.env` file is present (via `--env-file .env`). For host execution, `wandb` CLI will prompt or use a globally saved key.
    *   **Team/Entity:** Configure in `conf/config.yaml` (`wandb.entity`).

6.  **Set up DVC Google Drive Authentication (Service Account Method):**
    A Google Cloud Service Account is used for non-interactive DVC authentication with Google Drive, crucial for Docker and CI/CD.

    *   **6.1. Create Google Cloud Service Account & Key:**
        1.  In the Google Cloud Platform (GCP) Console, navigate to "IAM & Admin" > "Service Accounts."
        2.  Create a new service account (e.g., `dvc-gdrive-accessor-YOUR_INITIALS`).
        3.  Download a JSON key for this service account.
    *   **6.2. Store the Service Account Key Securely:**
        1.  Create a directory named `.secrets` in the root of this project if it doesn't exist.
        2.  Rename the downloaded JSON key to `gdrive-dvc-service-account.json` and place it inside this `.secrets` directory. The path should be: `YOUR_PROJECT_ROOT/.secrets/gdrive-dvc-service-account.json`.
        3.  **CRITICAL:** The `.secrets/` directory is correctly listed in your `.gitignore` and `.dockerignore` files.
    *   **6.3. Share DVC Google Drive Folder:**
        *   Go to the Google Drive folder used as your DVC remote (the ID is in `.dvc/config`, e.g., `gdrive://YOUR_GDRIVE_FOLDER_ID`).
        *   Share this folder with the **service account's email address** (e.g., `your-sa-name@your-gcp-project-id.iam.gserviceaccount.com`), granting it **"Editor"** permissions.
    *   **6.4. DVC Configuration (`.dvc/config` and `.dvc/config.local`):**
        *   Your primary DVC remote configuration is in `.dvc/config`:
            ```ini
            [core]
                remote = gdrive
            ['remote "gdrive"']
                url = gdrive://19qyjvhry7pP9AF4q03hbKl4M5EWhrtk2 # Your actual folder ID
                gdrive_use_service_account = true
            ```
        *   For local host DVC usage when `GDRIVE_CREDENTIALS_DATA` env var is not set, `.dvc/config.local` can point to the key file:
            ```ini
            # .dvc/config.local (This file IS gitignored by .dvc/.gitignore)
            ['remote "gdrive"']
                gdrive_service_account_json_file_path = ../.secrets/gdrive-dvc-service-account.json
            ```
            *(Note: Inside Docker, the `docker-entrypoint.sh` script primarily relies on `GDRIVE_CREDENTIALS_DATA_CONTENT` or the mounted key file via `GDRIVE_KEY_FILE_PATH_IN_CONTAINER` for DVC authentication.)*
    *   **6.5. DVC Cache (Host Setup):**
        Run `make ensure_host_dvc_cache` on the host once to create the DVC cache directory if it doesn't exist.

## 6. Usage Instructions

All primary operations can be run via `make` commands from the project root directory.
Hydra configurations can be overridden by passing them via the `ARGS` variable to `make`.

**Example for passing Hydra/Script arguments using `ARGS`:**
*   `make train ARGS="training.epochs=5 model.name=resnet34"`
*   `make docker_train ARGS="training.batch_size=32 training.profiler.enabled=true"`
*   `make test ARGS="-k specific_test_name --verbose"` (for passing pytest specific args)

### 6.1. Working Locally on Your Host Machine
(Ensure your virtual environment is activated)

*   **Pull DVC Data:**
    *   For host DVC operations using a service account, either ensure the `GDRIVE_CREDENTIALS_DATA` environment variable is set with the JSON key content, or that `.dvc/config.local` points to your key file.
        ```bash
        # Example for setting env var (Linux/macOS):
        # export GDRIVE_CREDENTIALS_DATA=$(cat .secrets/gdrive-dvc-service-account.json | tr -d '\n\r')
        make dvc_pull
        ```
*   **Process Data (Extract, etc.):** (Requires DVC data to be pulled first)
    ```bash
    make process_data # ARGS can be passed if dataset.py uses Hydra for its parameters
    ```
*   **Train Model:**
    ```bash
    make train ARGS="training.epochs=10"
    # For a short run with specific learning rate:
    make train ARGS="training.epochs=1 training.learning_rate=0.0005"
    ```
*   **Tests, Linting, Formatting, Cleaning:**
    ```bash
    make test ARGS="-m 'not slow'" # Example: run tests not marked as 'slow'
    make lint
    make format
    make clean
    ```

### 6.2. Working with Docker (Recommended for Reproducible Environments)
The `docker-entrypoint.sh` script handles DVC authentication inside containers.

*   **Build Docker Image:** (Needed once, or after `Dockerfile`/dependency changes)
    ```bash
    make docker_build
    # Specify tag: make docker_build IMAGE_TAG=my-custom-tag
    ```
*   **Run Interactive Shell in Docker:**
    ```bash
    make docker_shell
    ```
*   **Pull DVC Data *Inside* Docker:**
    ```bash
    make docker_dvc_pull
    ```
*   **Train Model *Inside* Docker:**
    ```bash
    make docker_train ARGS="training.epochs=10"
    # Example with profiler enabled for a short run:
    make docker_train ARGS="training.epochs=1 training.profiler.enabled=true training.profiler.active=10"
    ```
*   **Run Tests *Inside* Docker:**
    ```bash
    make docker_test ARGS="-m 'not slow'"
    ```

### 6.3. Key Makefile Variables for Docker & Customization
Override these on the command line if needed:
*   `IMAGE_NAME`, `IMAGE_TAG`
*   `HOST_DVC_CACHE_DIR`, `HOST_SERVICE_ACCOUNT_KEY_PATH`
*   `DOCKER_MEMORY_OPTS`
*   `ARGS`: For passing Hydra or script-specific arguments (e.g., `ARGS="training.epochs=5"`).
*   `WANDB_API_KEY`: (Set in host env/.env) API key for Weights & Biases.
*   `CI_MODE`: (Set to `true` for CI-specific behaviors).

## 7. Performance Profiling

### 7.1. Performance Profiling with PyTorch Profiler
The PyTorch Profiler is integrated into `drift_detector_pipeline/modeling/train.py` to analyze performance.

*   **Configuration:** Enable and configure the profiler in `conf/config.yaml` under the `training.profiler` section:
    ```yaml
    # In conf/config.yaml
    training:
      # ... other training params ...
      profiler:
        enabled: false # Set to true to enable profiling
        wait: 1
        warmup: 1
        active: 3      # Number of batches to actively profile per cycle
        repeat: 1      # Number of profiling cycles
        log_dir: "pytorch_profiler_logs" # Subdirectory in Hydra's output for traces
        export_chrome_trace: true  # For TensorBoard/Perfetto
        record_shapes: false     # Optional: records shapes of operator inputs
        profile_memory: false    # Optional: tracks tensor memory allocations
        with_stack: false        # Optional: records source information (adds overhead)
        with_flops: true         # Optional: records FLOPs for operators
        with_modules: true       # Optional: correlates operators with model hierarchy
    ```
*   **Running a Profiled Session:**
    Pass Hydra overrides via the `ARGS` variable in `make`:
    *   **Locally:**
        ```bash
        make train ARGS="training.profiler.enabled=true training.epochs=2 training.profiler.active=10"
        ```
    *   **In Docker:**
        ```bash
        make docker_train ARGS="training.profiler.enabled=true training.epochs=2 training.profiler.active=10"
        ```
*   **Viewing Profiler Traces:**
    1.  Profiler traces (`.json` files) are saved in the run's output directory: `outputs/YOUR_RUN_ID/pytorch_profiler_logs/`.
    2.  Use **TensorBoard** to view them:
        ```bash
        pip install tensorboard # If not already installed
        tensorboard --logdir path/to/your/run/pytorch_profiler_logs/
        # Or point to the parent 'outputs/' directory: tensorboard --logdir outputs/
        ```
        Open the URL (default: `http://localhost:6006`) and go to the "PyTorch Profiler" tab.
    3.  Alternatively, open trace files with the **Perfetto UI** at `ui.perfetto.dev`.
*   **Interpreting Traces:**
    *   In TensorBoard, examine the "Overview" (step time breakdown), "Operator View" (expensive CPU/GPU ops), "Kernel View" (GPU kernel times), and "Trace View" (timeline).
    *   Look for: CPU vs. GPU imbalances, data loading bottlenecks, long CPU-GPU transfers (`aten::copy_`), inefficient GPU kernels.

## 8. Continuous Integration & Delivery (CI/CD) with GitHub Actions

This project includes a GitHub Actions workflow (`.github/workflows/docker_train_pipeline.yml`) to automate:
1.  Building the Docker image.
2.  Running a short, CPU-only training validation job inside the container (includes DVC pull).
3.  Running tests inside the container.

**Workflow Overview (`docker_train_pipeline.yml`):**
*   **Triggers:** Runs on pushes to `main`, `develop`, `phase2-test`; pull requests to `main`, `develop`; and can be manually dispatched.
*   **Secrets:** Requires GitHub repository secrets: `GDRIVE_SA_KEY_JSON_CONTENT` and `WANDB_API_KEY`.
*   **Operation:**
    *   Builds the Docker image.
    *   Runs `make train ARGS="..."` and `make test` inside the container. The `ARGS` for the CI `make train` call are typically hardcoded in the workflow for a short validation run (e.g., `ARGS="training.epochs=3 data.dataloader_workers=2 run.device=cpu"`).
    *   The `docker-entrypoint.sh` handles DVC authentication using the `GDRIVE_SA_KEY_JSON_CONTENT` secret.
    *   `CI_MODE=true` is passed to `docker run`, allowing the Makefile to adjust settings (e.g., `DVC_PARALLEL_JOBS`).

**To enable this CI pipeline:**
1.  Ensure `.github/workflows/docker_train_pipeline.yml` is committed.
2.  Configure `GDRIVE_SA_KEY_JSON_CONTENT` and `WANDB_API_KEY` secrets in your GitHub repository (Settings > Secrets and variables > Actions).

## 9. Contribution Summary

### PHASE 1 Contributions

-   **Esteban Montelongo:** DVC setup & data versioning, `dataset.py` (extraction, transforms, dataloaders), initial documentation structure (`README.md`, `PHASE1.md`), architecture diagram, model DVC tracking, unit test implementation.
-   **Sajith Bandara:** Hydra integration (`conf/config.yaml`, `train.py` decorator/config usage), `train.py` core structure (model loading, optimizer, scheduler, loop), Makefile setup (`train`, `process_data` rules), model saving path correction.
-   **Arjun Kumar Sankar Chandrasekar:** WandB integration (`wandb.init`, `wandb.log`), dependency management (`pyproject.toml`, `requirements.txt`), `ruff` configuration and code formatting, testing infrastructure setup and test contributions.

### PHASE 2 Contributions

-   **Esteban Montelongo:** Initial `Dockerfile` creation, `docker-entrypoint.sh` script for DVC authentication (GDRIVE_CREDENTIALS_DATA logic), `.dockerignore` setup, Docker build caching strategies, design and implementation of `docker_train_pipeline.yml` GitHub Actions workflow including secret management, significant refactoring of `Makefile` for Docker targets and OS-specific command handling.
-   **Sajith Bandara:** Cross-platform compatibility enhancements for Docker-related `Makefile` targets (OS detection, volume pathing), integration of Docker shared memory (`--shm-size`), CI pipeline debugging and refinement, DVC service account authentication implementation.
-   **Arjun Kumar Sankar Chandrasekar:** Testing and validation of Docker image builds and container execution across environments, ensuring dependency consistency within containers, updating `README.md` with detailed setup and usage instructions for Docker, DVC service account authentication, Makefile targets, GitHub Actions CI/CD pipeline, and Profiling.

## 10. References & Key Tools Used
-   **Dataset:** [Imagenette-160 (v2)](https://github.com/fastai/imagenette)
-   **ML Framework:** [PyTorch](https://pytorch.org/)
-   **Model Zoo:** [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
-   **Containerization:** [Docker](https://www.docker.com/)
-   **Data Versioning:** [DVC (Data Version Control)](https://dvc.org/) + Google Drive (via Service Account)
-   **Configuration Management:** [Hydra](https://hydra.cc/)
-   **Experiment Tracking:** [Weights & Biases (WandB)](https://wandb.ai/)
-   **Performance Profiling:** [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
-   **Code Quality:** [Ruff](https://github.com/astral-sh/ruff) (Linting & Formatting), [Pytest](https://pytest.org/) (Testing)
-   **Version Control:** [Git](https://git-scm.com/) & [GitHub](https://github.com/)
-   **CI/CD:** [GitHub Actions](https://github.com/features/actions)
-   **Build/Task Runner:** [GNU Make](https://www.gnu.org/software/make/)
-   **Python Environment:** `venv` + `pip`
-   **Project Template:** [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)