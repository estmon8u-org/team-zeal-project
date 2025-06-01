# Image Classification with Drift Detection & MLOps Pipeline

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps pipeline for image classification on Imagenette-160. This project utilizes Docker for containerization, DVC for data versioning with Google Drive (via Service Accounts), Hydra for configuration, Weights & Biases for experiment tracking, PyTorch Profiler for performance analysis, CML for automated reporting, and GitHub Actions for CI/CD, including pushing Docker images to GCP Artifact Registry and models to GCP Cloud Storage.

## Project Organization

```plaintext
├── .dockerignore        <- Specifies files to exclude from Docker image
├── .dvc/                <- DVC metadata and configuration
├── .github/             <- GitHub Actions workflows and composite actions
│   ├── actions/gcp-setup/action.yml <- Composite action for GCP authentication
│   └── workflows/docker-train-lint.yml <- Main CI/CD and CML workflow
├── .gitignore           <- Specifies intentionally untracked files for Git
├── .pre-commit-config.yaml <- Configuration for pre-commit hooks
├── .secrets/            <- (Gitignored) Directory for sensitive keys (e.g., GCP service accounts)
├── LICENSE              <- Open-source license
├── Makefile             <- Convenience commands for building, training, testing, etc.
├── PHASE1.md            <- Documentation for Phase 1 deliverables
├── PHASE2.md            <- Documentation for Phase 2 deliverables
├── PHASE3.md            <- Documentation for Phase 3 deliverables
├── README.md            <- This file: project overview and instructions
├── Dockerfile           <- Instructions to build the project's Docker image
├── api/                 <- FastAPI application for model serving
│   ├── main.py          <- API endpoint logic
│   ├── requirements.txt <- API specific dependencies
│   └── .env             <- Environment variables for API (e.g., model GCS path)
├── best_model.pth       <- (Gitignored, example) Placeholder for best model from a run
├── cml_plots/           <- (Gitignored) Plots generated for CML reports
├── cml_metrics.json     <- (Gitignored) Metrics exported for CML reports
├── conf/                <- Hydra configuration files (e.g., config.yaml)
├── data/                <- (DVC-managed, gitignored) Project data (raw, processed)
├── docker-entrypoint.sh <- Script executed when Docker container starts
├── drift_detector_pipeline/ <- Python source code for the project
├── models/              <- (DVC-managed or gitignored) Trained models (DVC for baseline, GCS for CI)
├── notebooks/           <- Jupyter notebooks for exploration (if any)
├── outputs/             <- (Gitignored) Hydra outputs, logs, saved models from local runs
├── pyproject.toml       <- Python project metadata and dependencies (PEP 517/518)
├── requirements.txt     <- Pinned Python dependencies (for reference or specific envs)
└── tests/               <- Unit and integration tests
```

---

## 1. Team Information

- **Team Name:** Team Zeal
- **Team Members:**
  - Montelongo, Esteban - <EMONTEL1@depaul.edu>
  - Bandara, Sajith - <SBANDARA@depaul.edu>
  - Sankar Chandrasekar, Arjun Kumar - <ASANKARC@depaul.edu>
- **Course & Section:** SE 489: ML Engineering For Production (MLOps)

## 2. Project Overview

- **Summary:** This project implements an end-to-end MLOps pipeline to train an image classification model (ResNet-18 using `timm`) on the Imagenette-160 dataset. It incorporates containerization with Docker, data versioning with DVC and Google Drive, experiment tracking with Weights & Biases, configuration management with Hydra, performance profiling with PyTorch Profiler, Continuous Machine Learning (CML) for automated reporting, and CI/CD with GitHub Actions. Docker images are managed in GCP Artifact Registry, and trained model artifacts are stored in GCP Cloud Storage. An API using FastAPI is developed for model serving.
- **Problem Statement:** Machine learning models in production often suffer performance degradation due to data drift. This project aims to build robust, reproducible, and analyzable training pipelines as a foundation for systems that can automatically detect such drift and maintain model performance.
- **Key Objectives:**
  - **Phase 1:** Establish a reproducible baseline training pipeline, DVC, Hydra, WandB integration, and unit tests. *(Status: Completed)*
  - **Phase 2:** Enhance with Docker containerization, advanced logging, code profiling, and CI/CD foundations. *(Status: Completed)*
  - **Phase 3:** Implement full CI/CD automation including Docker image management in GCP Artifact Registry, model artifact storage in GCS, CML reporting, and API development for model serving. *(Status: CI/CD, CML, GCP AR/GCS model storage, API dev completed. API deployment planned.)*

## 3. Project Architecture Diagram (Phase 3)

[Diagram illustrating the MLOps pipeline](https://www.mermaidchart.com/app/projects/a791a045-ff95-417a-af0e-3ff4ad7cd646/diagrams/4300196f-397e-4681-886a-88e4fa6ba0e6/version/v0.1/edit)

This diagram illustrates the MLOps pipeline: Developers push code to GitHub, triggering GitHub Actions. The CI/CD pipeline lints, builds a Docker image (pushed to GCP Artifact Registry), and then uses this image to run training (pulling data via DVC from Google Drive, using Hydra for config, logging to WandB, and profiling) and tests. After training, CML generates a report, and the best model is uploaded to GCS. The FastAPI application (planned for GCP deployment) serves predictions using models from GCS.

## 4. Phase Deliverables

- [X] [PHASE1.md](./PHASE1.md): Project Design & Model Development
- [X] [PHASE2.md](./PHASE2.md): Enhancing ML Operations with Containerization & Monitoring
- [X] [PHASE3.md](./PHASE3.md): Continuous Machine Learning (CML) & Deployment *(Core CI/CD, CML, GCP AR/GCS integration, API dev completed)*

## 5. Setup Instructions

This section guides you through setting up the project environment.

1. **Clone Repository:**

    ```bash
    git clone https://github.com/estmon8u/team-zeal-project.git # Or your SSH URL
    cd team-zeal-project
    ```

2. **Prerequisites:**
    - **Python:** Version 3.10 or higher (as defined in `pyproject.toml`).
    - **Make:** GNU Make.
    - **Docker Desktop:** Required for building and running Docker containers. Ensure it's running.
    - **Git:** For version control.
    - **Google Cloud SDK (`gcloud`):** Optional for local interaction with GCP Artifact Registry or GCS. Install from [Google Cloud SDK documentation](https://cloud.google.com/sdk/docs/install).

3. **Create & Activate Virtual Environment (Recommended for Host Development):**

    ```bash
    make create_environment # Uses PYTHON_INTERPRETER defined in Makefile
    ```

    Activate the environment:
    - Linux/macOS: `source .venv/bin/activate`
    - Windows CMD: `.venv\Scripts\activate.bat`
    - Windows PowerShell: `.\.venv\Scripts\Activate.ps1`

4. **Install Python Dependencies (for Host Development):**
    (Ensure virtual environment is activated)

    ```bash
    make requirements
    ```

5. **Set up Weights & Biases (WandB):**
    - **Account:** Sign up at [wandb.ai](https://wandb.ai).
    - **API Key:**
        1. Get your API key from `wandb.ai/authorize`.
        2. **For Local Development:** Create a `.env` file in the project root (it's gitignored):

            ```env
            # .env (Do NOT commit this file)
            WANDB_API_KEY="YOUR_ACTUAL_WANDB_API_KEY_HERE"
            ```

            The `Makefile` passes this to Docker containers if `.env` is present. For host execution, `wandb login` or environment variables are used.
        3. **For CI/CD (GitHub Actions):** Add `WANDB_API_KEY` as a repository secret (Settings > Secrets and variables > Actions).
    - **Team/Entity:** Configure in `conf/config.yaml` (`wandb.entity`).

6. **Set up GCP Service Account(s) & Authentication:**
    Two main types of GCP access are needed: DVC access to Google Drive, and general GCP access for Artifact Registry & GCS.

    - **6.1. For DVC Google Drive Access (Service Account):**
        1. **Create GCP Service Account & Key for DVC:**
            - In GCP Console: "IAM & Admin" > "Service Accounts". Create a new SA (e.g., `dvc-gdrive-accessor-YOUR_INITIALS`).
            - Download its JSON key.
        2. **Store Key Securely (Local):**
            - Place the key as `.secrets/gdrive-dvc-service-account.json`. This path is used by `.dvc/config.local` and the `Makefile`.
        3. **Share Google Drive Folder:** Share your DVC remote Google Drive folder with this service account's email address (grant "Editor" permissions). The folder ID is in `.dvc/config`.
        4. **DVC Configuration:**
            - `.dvc/config`: Specifies `gdrive_use_service_account = true`.
            - `.dvc/config.local` (gitignored): Points to the local key file for host DVC operations: `gdrive_service_account_json_file_path = ../.secrets/gdrive-dvc-service-account.json`.
        5. **For CI/CD (GitHub Actions):** The content of this JSON key should be stored as a GitHub repository secret named `GDRIVE_SA_KEY_JSON_CONTENT`. The `docker-entrypoint.sh` script and CI workflow use this.

    - **6.2. For GCP Artifact Registry & GCS Access (Service Account):**
        1. **Create/Use GCP Service Account for CI/CD:**
            - This SA needs roles like "Artifact Registry Writer" (to push images) and "Storage Object Admin" (to write models to GCS buckets). Example: `github-actions-gcp-manager`.
            - Download its JSON key.
        2. **For CI/CD (GitHub Actions):** The content of this JSON key should be stored as a GitHub repository secret named `GCP_SA_KEY`. The `.github/actions/gcp-setup/action.yml` and main CI workflow use this.
        3. **GCP Project ID:** Store your GCP Project ID as a GitHub repository secret named `GCP_PROJECT_ID`.
        4. **Local GCP Access:** For local interaction with GCP services (e.g., pushing Docker images manually to AR, or accessing GCS buckets not via DVC), authenticate the `gcloud` CLI:

            ```bash
            gcloud auth login
            gcloud auth application-default login
            gcloud config set project YOUR_GCP_PROJECT_ID
            # Configure Docker for GCP AR (replace YOUR_REGION e.g., us-west2)
            gcloud auth configure-docker YOUR_REGION-docker.pkg.dev
            ```

7. **DVC Cache (Host Setup):**
    Run `make ensure_host_dvc_cache` on the host once to create the DVC cache directory (e.g., `~/.cache/dvc` or `%USERPROFILE%/.cache/dvc`) if it doesn't exist. This directory is mounted into Docker containers.

## 6. Usage Instructions

All primary operations can be run via `make` commands from the project root. Hydra configurations can be overridden by passing them via the `ARGS` variable.

**Example for passing Hydra/Script arguments using `ARGS`:**

- `make train ARGS="training.epochs=5 model.name=resnet34"`
- `make docker_train ARGS="training.batch_size=32 training.profiler.enabled=true"`
- `make test ARGS="-k specific_test_name --verbose"` (for pytest specific args)

### 6.1. Working Locally on Your Host Machine

(Ensure your virtual environment is activated and DVC/GCP auth is configured)

- **Pull DVC Data:**

    ```bash
    make dvc_pull
    ```

- **Process Data (Extract, etc.):**

    ```bash
    make process_data # ARGS can be passed if dataset.py uses Hydra
    ```

- **Train Model:**

    ```bash
    make train ARGS="training.epochs=10"
    ```

- **Tests, Linting, Formatting, Cleaning:**

    ```bash
    make test ARGS="-m 'not slow'"
    make lint
    make format
    make clean
    ```

- **Run API Locally:**

    ```bash
    # Ensure MODEL_GCS_PATH in api/.env points to a valid model in GCS
    # Or set it as an environment variable:
    # export MODEL_GCS_PATH="gs://team-zeal-models/your-branch/your-sha_model.pth"
    make api_run
    ```

### 6.2. Working with Docker (Recommended for Reproducible Environments)

The `docker-entrypoint.sh` handles DVC authentication inside containers using credentials passed as environment variables or mounted files.

- **Build Docker Image:**

    ```bash
    make docker_build
    # Specify tag: make docker_build IMAGE_TAG=my-custom-tag
    ```

- **Run Interactive Shell in Docker:**

    ```bash
    make docker_shell
    ```

- **Pull DVC Data *Inside* Docker:**

    ```bash
    make docker_dvc_pull
    ```

- **Train Model *Inside* Docker:**

    ```bash
    make docker_train ARGS="training.epochs=10"
    # With CML reporting enabled (generates plots/metrics locally inside container)
    make docker_cml_train ARGS="training.epochs=1"
    ```

- **Run Tests *Inside* Docker:**

    ```bash
    make docker_test ARGS="-m 'not slow'"
    ```

### 6.3. Key Makefile Variables for Customization

Override these on the command line if needed:

- `IMAGE_NAME`, `IMAGE_TAG`
- `HOST_DVC_CACHE_DIR`, `HOST_SERVICE_ACCOUNT_KEY_PATH` (for GDrive DVC SA key)
- `DOCKER_MEMORY_OPTS`
- `ARGS`: For passing Hydra or script-specific arguments.
- `CI_MODE`: Set to `true` for CI-specific behaviors (usually handled by workflow).

## 7. Performance Profiling with PyTorch Profiler

The PyTorch Profiler is integrated into `drift_detector_pipeline/modeling/train.py`.

- **Configuration:** Enable and configure in `conf/config.yaml` under `training.profiler`.
- **Running a Profiled Session:**
  - Locally: `make train ARGS="training.profiler.enabled=true training.epochs=2 ..."`
  - In Docker: `make docker_train ARGS="training.profiler.enabled=true training.epochs=2 ..."`
- **Viewing Traces:** Traces (`.json`) are saved in `outputs/YOUR_RUN_ID/pytorch_profiler_logs/`. Use TensorBoard (`tensorboard --logdir path/to/logs`) or Perfetto UI (`ui.perfetto.dev`).

## 8. Continuous Integration, Delivery & Machine Learning (CI/CD/CML)

A GitHub Actions workflow (`.github/workflows/docker-train-lint.yml`) automates:

1. **Code Linting & Formatting Checks:** Uses Ruff for code quality enforcement.
2. **Docker Image Build & Push:** Builds the project's Docker image and pushes it to GCP Artifact Registry, tagged with `latest`, `ci`, and the Git commit SHA.
3. **Automated Training & Testing:**
    - Pulls the Docker image from GCP Artifact Registry.
    - Runs a training job inside the container. For `main` branch, it runs more epochs; for `develop` or PRs, it runs a shorter validation.
    - Pulls data using DVC (authenticating to Google Drive via service account secrets).
    - Logs metrics to Weights & Biases.
    - Runs unit tests using Pytest within the container.
4. **Model Upload to GCS:** The best model from the training job is uploaded to a specified GCS bucket (e.g., `gs://team-zeal-models/BRANCH_NAME/COMMIT_SHA_model.pth`).
5. **CML Report Generation:**
    - After successful training, metrics (e.g., best validation accuracy, losses) and plots (loss curves, accuracy curves) are generated.
    - A CML report (`report.md`) is created and posted as a comment on the Pull Request or commit.

**Workflow Secrets:**
Ensure the following secrets are configured in your GitHub repository (Settings > Secrets and variables > Actions):

- `GCP_SA_KEY`: JSON content of the service account key for GCP Artifact Registry and GCS access.
- `GCP_PROJECT_ID`: Your Google Cloud Project ID.
- `GDRIVE_SA_KEY_JSON_CONTENT`: JSON content of the service account key for DVC/Google Drive access.
- `WANDB_API_KEY`: Your Weights & Biases API key.

## 9. Model & Image Management in GCP

- **Docker Images:** Stored in GCP Artifact Registry. The path is typically `YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO_NAME/IMAGE_NAME:TAG`.
- **Trained Models (.pth files):** Stored in Google Cloud Storage, typically in a bucket like `gs://team-zeal-models/`, organized by branch and commit SHA.

## 10. API (FastAPI)

A FastAPI application is available in the `api/` directory.

- **Functionality:** Provides an endpoint (`/predict/`) to classify uploaded images.
- **Model Loading:** Designed to load a trained model from a GCS path specified by the `MODEL_GCS_PATH` environment variable (see `api/.env.example`).
- **Local Execution:**

    ```bash
    # Set MODEL_GCS_PATH, e.g.:
    # export MODEL_GCS_PATH="gs://team-zeal-models/main/abcdef1_model.pth"
    make api_run
    ```

- **Dockerization:**

    ```bash
    make api_docker_build
    # docker run -p 8008:8008 -e MODEL_GCS_PATH="gs://..." team-zeal-api:latest
    ```

## 11. API Deployment to GCP Cloud Functions (2nd Gen)

This project includes a FastAPI application in the `api/` directory for model serving. It can be deployed as a containerized application to GCP Cloud Functions (2nd Generation).

### Prerequisites for API Deployment

1. GCP Project with billing enabled.
2. Enabled APIs: Cloud Functions, Cloud Build, Artifact Registry, Cloud Run.
3. `gcloud` CLI authenticated and configured.
4. Docker installed and authenticated with GCP Artifact Registry (`gcloud auth configure-docker ...`).
5. A trained model (`.pth` file) uploaded to a GCS bucket.

### Building and Pushing the API Docker Image

The `Makefile` provides targets to build the API's Docker image and push it to GCP Artifact Registry.

```bash
# Example: Build and push the API image tagged as v1.0.0
# Ensure GCP_PROJECT_ID_LOCAL and GCP_REGION_LOCAL are correctly set
# (e.g., by `gcloud config get-value project` or by overriding in the make command)
make api_docker_push_gcp API_IMAGE_TAG=v1.0.0 API_ARTIFACT_REGISTRY_REPO=team-zeal-project
```

This will build the image defined in api/Dockerfile and push it to YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/team-zeal-project/team-zeal-project-api:v1.0.0

### Deploying to Cloud Functions

Use the gcloud CLI to deploy the container image as a 2nd Generation Cloud Function.

```bash
# Define these variables or replace them directly in the command
export FUNCTION_NAME="team-zeal-classifier-api"
export GCP_REGION="us-west2" # Your GCP region
export API_IMAGE_TAG="v1.0.0" # The tag you pushed
export GCP_PROJECT_ID=$(gcloud config get-value project)
export API_ARTIFACT_REGISTRY_REPO="team-zeal-project" # Your AR repo name
export API_IMAGE_NAME="team-zeal-project-api"

# Construct the IMAGE_URI
export IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${API_ARTIFACT_REGISTRY_REPO}/${API_IMAGE_NAME}:${API_IMAGE_TAG}"

# IMPORTANT: Update this path to your actual model in GCS
export MODEL_GCS_PATH_FOR_FUNCTION="gs://team-zeal-models/main/your_model_commit_sha.pth"

gcloud functions deploy ${FUNCTION_NAME} \
  --gen2 \
  --region=${GCP_REGION} \
  --runtime=python310 \
  --source=. \
  --entry-point=http \
  --trigger-http \
  --allow-unauthenticated \
  --container-image=${IMAGE_URI} \
  --set-env-vars MODEL_GCS_PATH=${MODEL_GCS_PATH_FOR_FUNCTION} \
  --memory=2Gi \
  --timeout=300s
```

## 12. API Deployment to GCP Cloud Run

You can deploy the API container to Cloud Run for scalable serving:

```bash
make api_deploy_cloudrun
```

This Makefile target:

1. Creates a service account for the Cloud Run service if it doesn't exist
2. Grants necessary GCS permissions to access the model
3. Deploys the API container to Cloud Run with appropriate settings
4. Sets environment variables including the MODEL_GCS_PATH

### Customizing Cloud Run Deployment

The deployment can be customized with various parameters:

```bash
# Specify a different project or GCP region
make api_deploy_cloudrun GCP_PROJECT_ID_LOCAL=for-copying-332923 GCP_REGION_LOCAL=us-central1

# Use a different model path in GCS
make api_deploy_cloudrun MODEL_GCS_PATH=gs://team-zeal-models/dev/latest_model.pth

# Adjust Cloud Run resource allocation
make api_deploy_cloudrun CLOUD_RUN_MEMORY=4Gi CLOUD_RUN_CPU=2

# Configure scaling behavior
make api_deploy_cloudrun CLOUD_RUN_MIN_INSTANCES=1 CLOUD_RUN_MAX_INSTANCES=10

# Use a custom service account
make api_deploy_cloudrun CLOUD_RUN_SERVICE_ACCOUNT=custom-sa@project-id.iam.gserviceaccount.com

# Combine multiple options
make api_deploy_cloudrun GCP_PROJECT_ID_LOCAL=for-copying-332923 \
  MODEL_GCS_PATH=gs://team-zeal-models/main/special_model.pth \
  CLOUD_RUN_MEMORY=4Gi \
  CLOUD_RUN_MIN_INSTANCES=1
```

### Troubleshooting Deployment

If you encounter permission issues with the service account, you can deploy without specifying a custom service account:

```bash
# Deploy using the default Compute Engine service account
make api_deploy_cloudrun WITHOUT_SA=1 GCP_PROJECT_ID_LOCAL=for-copying-332923
```

Make sure to grant GCS access to whichever service account is being used:

```bash
gsutil iam ch serviceAccount:YOUR-COMPUTE-SA@your-project.iam.gserviceaccount.com:objectViewer gs://team-zeal-models
```

After deployment, your API will be accessible at a URL like `https://team-zeal-api-run-HASH.run.app`.

## 13. Interactive Web UI (Streamlit on Hugging Face Spaces)

An interactive web interface for this image classifier is built using Streamlit and deployed on Hugging Face Spaces. This UI allows users to upload images and get predictions by calling the backend API deployed on GCP Cloud Run.

- **Streamlit Application Code:** Located in the `ui_streamlit/` directory.
- **Live Demo URL:** [https://huggingface.co/spaces/estmon8u/team-zeal-image-classifier](https://huggingface.co/spaces/estmon8u/team-zeal-image-classifier)
- **Functionality:**
  - Upload JPG, PNG images.
  - Displays the uploaded image.
  - Sends the image to the backend API for classification.
  - Shows the predicted class, confidence score, and a bar chart of probabilities.

To run the Streamlit app locally (after setting up its requirements and ensuring the backend API is accessible):

```bash
cd ui_streamlit
pip install -r requirements.txt
# Ensure API_URL in streamlit_app.py is correct or PREDICTION_API_URL env var is set
streamlit run streamlit_app.py
```

## 14. Contribution Summary

### PHASE 1 Contributions

- **Esteban Montelongo:** DVC setup & data versioning, `dataset.py` (extraction, transforms, dataloaders), initial documentation structure (`README.md`, `PHASE1.md`), architecture diagram, model DVC tracking, unit test implementation.
- **Sajith Bandara:** Hydra integration (`conf/config.yaml`, `train.py` decorator/config usage), `train.py` core structure (model loading, optimizer, scheduler, loop), Makefile setup (`train`, `process_data` rules), model saving path correction.
- **Arjun Kumar Sankar Chandrasekar:** WandB integration (`wandb.init`, `wandb.log`), dependency management (`pyproject.toml`, `requirements.txt`), `ruff` configuration and code formatting, testing infrastructure setup and test contributions.

### PHASE 2 Contributions

- **Esteban Montelongo:** Initial `Dockerfile` creation, `docker-entrypoint.sh` script for DVC authentication, `.dockerignore` setup, Docker build caching strategies, design and implementation of `docker-train-lint.yml` GitHub Actions workflow (initial CI for Docker builds & tests), significant refactoring of `Makefile` for Docker targets and OS-specific command handling.
- **Sajith Bandara:** Cross-platform compatibility enhancements for Docker-related `Makefile` targets (OS detection, volume pathing), integration of Docker shared memory (`--shm-size`), CI pipeline debugging and refinement, DVC service account authentication logic integration.
- **Arjun Kumar Sankar Chandrasekar:** Testing and validation of Docker image builds and container execution, ensuring dependency consistency, updating `README.md` with detailed setup and usage for Docker, DVC service account, Makefile targets, initial CI pipeline, and Profiling integration.

### PHASE 3 Contributions

- **Esteban Montelongo:** Enhanced GitHub Actions workflow (`docker-train-lint.yml`) to include pushing Docker images to GCP Artifact Registry, uploading trained models to GCS, and full CML report generation. Developed `gcp-setup` composite action. Implemented logic for branch-specific training parameters in CI.
- **Sajith Bandara:** Refined CML reporting steps, including plot generation and metrics export within the training script. Debugged and optimized CI workflow steps, especially artifact handling and GCS interactions. Contributed to API (`api/main.py`) development for GCS model loading and local testing setup.
- **Arjun Kumar Sankar Chandrasekar:** Integrated pre-commit hooks (`.pre-commit-config.yaml`). Updated `Makefile` for API targets and refined Docker memory options. Updated project documentation (`README.md`, `PHASE*.md`) to reflect Phase 3 advancements, including CI/CD, CML, and GCP integration details.

## 15. References & Key Tools Used

- **Dataset:** [Imagenette-160 (v2)](https://github.com/fastai/imagenette)
- **ML Framework:** [PyTorch](https://pytorch.org/)
- **Model Zoo:** [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
- **Containerization:** [Docker](https://www.docker.com/)
- **Data Versioning:** [DVC (Data Version Control)](https://dvc.org/) + Google Drive (via Service Account)
- **Configuration Management:** [Hydra](https://hydra.cc/)
- **Experiment Tracking:** [Weights & Biases (WandB)](https://wandb.ai/)
- **Performance Profiling:** [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- **Code Quality:** [Ruff](https://github.com/astral-sh/ruff) (Linting & Formatting), [Pytest](https://pytest.org/) (Testing), [Pre-commit](https://pre-commit.com/)
- **Version Control:** [Git](https://git-scm.com/) & [GitHub](https://github.com/)
- **CI/CD & CML:** [GitHub Actions](https://github.com/features/actions), [CML (Continuous Machine Learning)](https://cml.dev/)
- **Cloud Platform:** [Google Cloud Platform (GCP)](https://cloud.google.com/)
  - [GCP Artifact Registry](https://cloud.google.com/artifact-registry) (for Docker images)
  - [GCP Cloud Storage](https://cloud.google.com/storage) (for model artifacts)
- **API Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **Build/Task Runner:** [GNU Make](https://www.gnu.org/software/make/)
- **Python Environment:** `venv` + `pip`
- **Project Template:** [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
