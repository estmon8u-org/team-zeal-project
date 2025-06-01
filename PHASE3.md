## 1. Continuous Integration & Testing

- [x] **1.1 Unit Testing with pytest**
  - [x] Test scripts for data processing, model training, and evaluation
  - [x] Documentation of the testing process and example test cases
- [x] **1.2 GitHub Actions Workflows**
  - [x] CI workflows for running tests, DVC, code checks (e.g., ruff), Docker builds
  - [x] Workflow YAML files included
- [x] **1.3 Pre-commit Hooks**
  - [x] Pre-commit config and setup instructions

## 2. Continuous Docker Building & CML

- [x] **2.1 Docker Image Automation**
  - [x] Automated Docker builds and pushes (GitHub Actions)
  - [x] Dockerfile and build/push instructions for Docker Hub and GCP Artifact Registry
- [x] **2.2 Continuous Machine Learning (CML)**
  - [x] CML integration for automated model training on PRs
  - [x] Example CML outputs (metrics, visualizations)
  - [x] Setup and usage documentation

## 3. Deployment on Google Cloud Platform (GCP)

- [x] **3.1 GCP Artifact Registry**
  - [x] Steps for creating and pushing Docker images to GCP
- [ ] **3.2 Custom Training Job on GCP**
  - [ ] Github Runner job setup and documentation
  - [x] Data storage in GCP bucket (Handled by DVC with GDrive, models to GCS)
- [X] **3.3 Deploying API with FastAPI & GCP Cloud Functions**
  - [X] API code (`api/main.py`) prepared for containerized deployment.
  - [X] `api/Dockerfile` created to containerize the FastAPI application.
  - [X] `Makefile` updated with targets (`api_docker_build_gcp`, `api_docker_push_gcp`) to build and push the API Docker image to GCP Artifact Registry.
  - [x] Step-by-step deployment guide to GCP Cloud Functions (2nd Gen)/Cloud Run using the container image.
  - [x] Instructions for setting environment variables (e.g., `MODEL_GCS_PATH`).
  - [x] API testing instructions after deployment.
- [X] **3.4 Dockerize & Deploy Model with GCP Cloud Run**
  - [X] API Docker image built and pushed to GCP Artifact Registry.
  - [x] Step-by-step deployment guide to GCP Cloud Run using the container image.
  - [x] Instructions for setting environment variables (e.g., `MODEL_GCS_PATH`, `PORT`).
  - [x] API testing instructions after deployment.
- [X] **3.5 Interactive UI Deployment**
  - [X] Streamlit app (`ui_streamlit/streamlit_app.py`) developed to call the Cloud Run API.
  - [X] `ui_streamlit/requirements.txt` created for UI dependencies.
  - [X] Streamlit app deployed to Hugging Face Spaces.
        * Hugging Face Space URL: [https://huggingface.co/spaces/estmon8u/team-zeal-image-classifier](https://huggingface.co/spaces/estmon8u/team-zeal-image-classifier)
  - [ ] Integration of UI deployment into GitHub Actions workflow (SKIPPING)
  - [x] Screenshots and usage examples

  ## 4. Documentation & Repository Updates

- [x] **4.1 Comprehensive README**
  - [x] Setup, usage, and documentation for all CI/CD, CML, and deployment steps
  - [x] Screenshots and results of deployments
- [x] **4.2 Resource Cleanup Reminder**
  - [x] Checklist for removing GCP resources to avoid charges
