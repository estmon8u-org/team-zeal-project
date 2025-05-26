# Image Classification with Drift Detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps pipeline for image classification on Imagenette-160, featuring automated data drift detection and retraining. This project utilizes Docker for containerization, DVC for data versioning with Google Drive, Hydra for configuration, Weights & Biases for experiment tracking, and GitHub Actions for CI/CD

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         drift_detector_pipeline and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── drift_detector_pipeline   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes drift_detector_pipeline a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
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
-   **Summary:** This project implements an end-to-end MLOps pipeline to train an image classification model (ResNet-18 using `timm`) on the Imagenette-160 dataset, monitor for data drift, and eventually automate retraining.
-   **Problem Statement:** Machine learning models deployed in production often suffer performance degradation over time due to changes in the underlying data distribution (data drift). This project aims to build a system that can automatically detect such drift in an image classification task and maintain model performance through automated retraining.
-   **Main Objectives (Phase 1 Focus):**
    1.  Establish a reproducible baseline training pipeline for ResNet-18 on clean Imagenette-160 data, achieving ≥85% validation accuracy. *(Achieved: 96.76%)*
    2.  Set up version control for data (DVC) and code (Git). *(Completed)*
    3.  Integrate configuration management (Hydra) and experiment tracking (WandB). *(Completed)*
    4.  Structure the codebase for future drift simulation and detection implementation. *(Completed)*
    5.  Implement basic unit tests for data handling. *(Completed)*

## 3. Project Architecture Diagram (Phase 1)
![Phase 1 Architecture Diagram](./phase1.jpg)
*(This diagram represents the components set up in Phase 1: Data acquisition/versioning with DVC/G-Drive, Code versioning with Git/GitHub, Training pipeline using PyTorch/timm, Configuration via Hydra, Experiment Tracking via WandB, Unit Testing via Pytest, and local execution via Make.)*

## 4. Phase Deliverables
-   [X] [PHASE1.md](./PHASE1.md): Project Design & Model Development *(Completed)*
-   [ ] PHASE2.md: Enhancing ML Operations *(Upcoming)*
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
    # Activate:
    # Linux/macOS: source .venv/bin/activate
    # Windows CMD: .venv\Scripts\activate.bat
    # Windows PowerShell: .\.venv\Scripts\Activate.ps1
    ```

4.  **Install Python Dependencies (for Host Development):**
    (Ensure virtual environment is activated)
    ```bash
    make requirements
    ```

5.  **Set up Weights & Biases (WandB):**
    *   **Account:** Sign up at [wandb.ai](https://wandb.ai).
    *   **API Key for Local Development:**
        1.  Get your API key from wandb.ai/authorize.
        2.  Create a `.env` file in the project root (this is gitignored):
            ```env
            # .env (Do NOT commit this file)
            WANDB_API_KEY="YOUR_ACTUAL_WANDB_API_KEY_HERE"
            ```
        3.  The `Makefile` (if `WANDB_API_KEY` is set in the shell or `.env`) passes this key to Docker containers. For host execution, `wandb` CLI will prompt or use a globally saved key.
    *   **Team/Entity:** Configure in `conf/config.yaml` (`wandb.entity`).

6.  **Set up DVC Google Drive Authentication (Service Account Method):**
    A Google Cloud Service Account is used for non-interactive DVC authentication with Google Drive, crucial for Docker and CI/CD.

    *   **6.1. Create Google Cloud Service Account & Key:**
        1.  In the Google Cloud Platform (GCP) Console, navigate to "IAM & Admin" > "Service Accounts."
        2.  Create a new service account (e.g., `dvc-gdrive-access-YOUR_INITIALS`).
        3.  Download a JSON key for this service account.
    *   **6.2. Store the Service Account Key Securely:**
        1.  Create a directory named `.secrets` in the root of this project.
        2.  Rename the downloaded JSON key to `gdrive-dvc-service-account.json` and place it inside this `.secrets` directory. The full path should be: `YOUR_PROJECT_ROOT/.secrets/gdrive-dvc-service-account.json`.
        3.  **CRITICAL:** The `.secrets/` directory (and its contents) **MUST** be listed in your project's `.gitignore` file AND `.dockerignore` file to prevent the key from being committed to version control or included in Docker images. (Your current `.dockerignore` includes `.secrets/`).
    *   **6.3. Share DVC Google Drive Folder:**
        *   Go to the Google Drive folder used as your DVC remote (the ID is in `.dvc/config`, e.g., `gdrive://19qyjvhry7pP9AF4q03hbKl4M5EWhrtk2`).
        *   Share this folder with the **service account's email address** (e.g., `your-sa@your-project.iam.gserviceaccount.com`), granting it **"Editor"** permissions.
    *   **6.4. DVC Configuration (`.dvc/config.local`):**
        *   Ensure the file `.dvc/config.local` (which should be in `.dvc/.gitignore`) exists and contains the following to instruct DVC to use the service account (the path is a fallback, as the entrypoint script handles the primary logic via environment variables):
            ```ini
            # .dvc/config.local
            ['remote "gdrive"']
                url = gdrive://YOUR_GDRIVE_FOLDER_ID # Replace with your actual folder ID
                gdrive_use_service_account = true
                # Path used by entrypoint if GDRIVE_CREDENTIALS_DATA_CONTENT is not set,
                # or as a fallback. For local Docker, Makefile mounts key to this path.
                gdrive_service_account_json_file_path = /app/.secrets/gdrive-dvc-service-account.json
            ```
            *(Note: The `docker-entrypoint.sh` primarily relies on `GDRIVE_CREDENTIALS_DATA_CONTENT` or `GDRIVE_KEY_FILE_PATH_IN_CONTAINER` environment variables for auth in Docker. The `gdrive_service_account_json_file_path` in `config.local` serves as a fallback or can be used by DVC on the host if `GDRIVE_CREDENTIALS_DATA` is not set in the host environment.)*
    *   **6.5. DVC Cache (Host Setup):**
        *   The `Makefile` mounts a host DVC cache (default `~/.cache/dvc` or `%USERPROFILE%/.cache/dvc`) into Docker containers. Run `make ensure_host_dvc_cache` on the host once.

## 6. Usage Instructions

All primary operations can be run via `make` commands from the project root directory.

### 6.1. Working Locally on Your Host Machine

(Ensure your virtual environment is activated: `source .venv/bin/activate` or Windows equivalent)

*   **Pull DVC Data:**
    *   For host DVC operations needing GDrive auth, first set the `GDRIVE_CREDENTIALS_DATA` environment variable using your service account key:
        ```bash
        # Linux/macOS:
        export GDRIVE_CREDENTIALS_DATA=$(cat .secrets/gdrive-dvc-service-account.json | tr -d '\n\r')
        # Windows (PowerShell):
        $env:GDRIVE_CREDENTIALS_DATA = Get-Content -Raw -Path ".\.secrets\gdrive-dvc-service-account.json"
        ```
    *   Then run:
        ```bash
        make dvc_pull
        ```
*   **Process Data (Extract, etc.):**
    ```bash
    make process_data
    ```
*   **Train Model:**
    ```bash
    make train
    make train HYDRA_ARGS="training.epochs=1 model.name=resnet34" # With overrides
    ```
*   **Tests, Linting, Formatting, Cleaning:**
    ```bash
    make test
    make lint
    make format
    make clean
    ```

### 6.2. Working with Docker (Recommended for Reproducible Environments)
The `docker-entrypoint.sh` script handles DVC authentication inside containers using the service account key.

*   **Build Docker Image:**
    (Needed once, or after `Dockerfile`/dependency changes)
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
    make docker_train
    make docker_train HYDRA_ARGS="training.epochs=1" # With overrides
    ```
    *   **DVC Authentication in Docker:** The `Makefile` mounts your local `.secrets/gdrive-dvc-service-account.json` and sets `GDRIVE_KEY_FILE_PATH_IN_CONTAINER`. The `docker-entrypoint.sh` reads this, sets `GDRIVE_CREDENTIALS_DATA`, which DVC uses.
    *   **WandB in Docker:** If `WANDB_API_KEY` is in your host `.env` or shell environment, it's passed to the container.
*   **Run Tests *Inside* Docker:**
    ```bash
    make docker_test
    ```

### 6.3. Key Makefile Variables for Docker Customization
Override these on the command line if needed (e.g., `make docker_train IMAGE_TAG=0.2.0 ...`).
*   `IMAGE_NAME`: Docker image name (default: `team-zeal-project`).
*   `IMAGE_TAG`: Image tag (default: `1.0.0` - *ensure this is updated in `Makefile` as you version*).
*   `HOST_DVC_CACHE_DIR`: Path to shared DVC cache on host.
*   `HOST_SERVICE_ACCOUNT_KEY_PATH`: Path to GDrive SA key on host (default: `./.secrets/gdrive-dvc-service-account.json`).
*   `HYDRA_ARGS`: Arguments passed to Hydra for training runs.
*   `WANDB_API_KEY`: (Set in host env/.env) API key for Weights & Biases.

## 7. Continuous Integration & Delivery (CI/CD) with GitHub Actions

This project includes a GitHub Actions workflow (`.github/workflows/docker-train.yml`) to automate building the Docker image and running a short training validation pipeline.

**Workflow Overview (`docker-train.yml`):**
1.  **Triggers:** Runs on pushes to `main`, `develop`, `phase2-test`; pull requests to `main`, `develop`; and can be manually dispatched.
2.  **Environment:** Uses an `ubuntu-latest` GitHub-hosted runner.
3.  **Secrets:** Requires the following GitHub repository secrets to be configured (Settings > Secrets and variables > Actions):
    *   `GDRIVE_SA_KEY_JSON_CONTENT`: The **full JSON string content** of your `gdrive-dvc-service-account.json` file.
    *   `WANDB_API_KEY`: Your Weights & Biases API key.
4.  **Steps:**
    *   Checks out the repository code.
    *   Sets up Docker Buildx.
    *   Builds the Docker image using `make docker_build` (tagging it as `ghcr.io/OWNER/REPO:ci`).
    *   Runs the training process inside the built Docker container:
        *   Mounts the checked-out code (`$PWD`) into `/app`.
        *   Injects `GDRIVE_CREDENTIALS_DATA_CONTENT` (from the GitHub secret `GDRIVE_SA_KEY_JSON_CONTENT`) and `WANDB_API_KEY` (from the GitHub secret) as environment variables into the container.
        *   The `docker-entrypoint.sh` script inside the container detects `GDRIVE_CREDENTIALS_DATA_CONTENT` and sets up `GDRIVE_CREDENTIALS_DATA` for DVC.
        *   Executes `make train` with CI-specific Hydra arguments (e.g., fewer epochs, CPU-only) for validation.
        ```yaml
        # Excerpt from GitHub Actions workflow:
        # ...
        - name: Run DVC pull + training in Docker
          env:
            SA_KEY: ${{ secrets.GDRIVE_SA_KEY_JSON_CONTENT }}
            WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
          run: |
            docker run --rm \
              -v "$PWD:/app" \
              -e GDRIVE_CREDENTIALS_DATA_CONTENT="$SA_KEY" \
              -e WANDB_API_KEY="$WANDB_API_KEY" \
              ghcr.io/${{ github.repository_owner }}/${{ github.event.repository.name }}:ci \
              make train HYDRA_ARGS="training.epochs=3 data.dataloader_workers=2 run.device=cpu"
        ```

**To enable this CI pipeline:**
1.  Ensure the workflow file `.github/workflows/docker-train.yml` (as provided in project or your version) is committed to your repository.
2.  Configure the `GDRIVE_SA_KEY_JSON_CONTENT` and `WANDB_API_KEY` secrets in your GitHub repository settings.
3.  The workflow will then trigger automatically based on the defined `on:` events.

## 7. Contribution Summary

### PHASE 1 Contributions

-   **Esteban Montelongo:** DVC setup & data versioning, `dataset.py` (extraction, transforms, dataloaders), initial documentation structure (`README.md`, `PHASE1.md`), architecture diagram, model DVC tracking, unit test implementation.
-   **Sajith Bandara:** Hydra integration (`conf/config.yaml`, `train.py` decorator/config usage), `train.py` core structure (model loading, optimizer, scheduler, loop), Makefile setup (`train`, `process_data` rules), model saving path correction.
-   **Arjun Kumar Sankar Chandrasekar:** WandB integration (`wandb.init`, `wandb.log`), dependency management (`pyproject.toml`, `requirements.txt`), `ruff` configuration and code formatting, testing infrastructure setup and test contributions.

### PHASE 2 Contributions

-   **Esteban Montelongo:** Initial `Dockerfile` creation, `docker-entrypoint.sh` script for DVC authentication (GDRIVE_CREDENTIALS_DATA logic), `.dockerignore` setup, Docker build caching strategies, design and implementation of `docker_train_pipeline.yml` GitHub Actions workflow including secret management, significant refactoring of `Makefile` for Docker targets and OS-specific command handling.
-   **Sajith Bandara:** Cross-platform compatibility enhancements for Docker-related `Makefile` targets (OS detection, volume pathing), integration of Docker shared memory (`--shm-size`), CI pipeline debugging and refinement, DVC service account authentication implementation.
-   **Arjun Kumar Sankar Chandrasekar:** Testing and validation of Docker image builds and container execution across environments, ensuring dependency consistency within containers, updating `README.md` with detailed setup and usage instructions for Docker, DVC service account authentication, Makefile targets, and GitHub Actions CI/CD pipeline.



## 8. References & Key Tools Used
-   **Dataset:** [Imagenette-160 (v2)](https://github.com/fastai/imagenette)
-   **ML Framework:** [PyTorch](https://pytorch.org/)
-   **Model Zoo:** [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
-   **Data Versioning:** [DVC (Data Version Control)](https://dvc.org/) + Google Drive
-   **Configuration Management:** [Hydra](https://hydra.cc/)
-   **Experiment Tracking:** [Weights & Biases (WandB)](https://wandb.ai/)
-   **Code Quality:** [Ruff](https://github.com/astral-sh/ruff) (Linting & Formatting), [Pytest](https://pytest.org/) (Testing)
-   **Version Control:** [Git](https://git-scm.com/) & [GitHub](https://github.com/)
-   **Build/Task Runner:** [GNU Make](https://www.gnu.org/software/make/)
-   **Python Environment:** `venv` + `pip`
-   **Project Template:** [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)