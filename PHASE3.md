# PHASE 3: Continuous Machine Learning (CML) & Deployment

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
  - [ ] Example CML outputs (metrics, visualizations)
  - [ ] Setup and usage documentation

## 3. Deployment on Google Cloud Platform (GCP)

- [x] **3.1 GCP Artifact Registry**
  - [x] Steps for creating and pushing Docker images to GCP
- [ ] **3.2 Custom Training Job on GCP**
  - [ ] Github Runner job setup and documentation
  - [x] Data storage in GCP bucket
- [ ] **3.3 Deploying API with FastAPI & GCP Cloud Functions**
  - [x] API code (`api/main.py`) prepared for containerized deployment.
  - [x] `api/Dockerfile` created to containerize the FastAPI application.
  - [x] `Makefile` updated with targets (`api_docker_build_gcp`, `api_docker_push_gcp`) to build and push the API Docker image to GCP Artifact Registry.
  - [x] Step-by-step deployment guide to GCP Cloud Functions (2nd Gen) using the container image.
  - [x] Instructions for setting environment variables (e.g., `MODEL_GCS_PATH`).
  - [x] API testing instructions after deployment.

**Deployment Guide:**

Deploying your FastAPI application to GCP Cloud Functions (2nd Generation) involves containerizing your API, pushing the container image to GCP Artifact Registry, and then creating a Cloud Function that uses this image.

**Prerequisites:**

1. **Enable APIs in GCP:** Ensure the following APIs are enabled in your GCP project:
    - Cloud Functions API
    - Cloud Build API (used by Cloud Functions for deployments)
    - Artifact Registry API
    - Cloud Run API (2nd Gen Cloud Functions are built on Cloud Run)
    - Cloud Logging API
2. **`gcloud` CLI:** Authenticated and configured (`gcloud auth login`, `gcloud config set project YOUR_PROJECT_ID`).
3. **Docker:** Docker installed and running. Authenticate Docker with GCP Artifact Registry:

    ```bash
    gcloud auth configure-docker YOUR_GCP_REGION-docker.pkg.dev
    # Example: gcloud auth configure-docker us-west2-docker.pkg.dev
    ```

4. **GCP Artifact Registry Repository:** Create a Docker repository in Artifact Registry if you haven't already. Your CI pipeline might already do this. You can use the same repository (`team-zeal-project`) as your training images, or create a new one.
5. **Trained Model in GCS:** Ensure your trained model (`.pth` file) is uploaded to a GCS bucket. The path to this model will be passed as an environment variable. Example: `gs://team-zeal-models/main/your_model_commit_sha.pth`.

**Deployment Steps:**

**1. Prepare API Code and Dockerfile:**

- Your FastAPI application is in `api/main.py`.
- A dedicated `api/Dockerfile` is used to containerize this application. (See Step 1 in the response above for its content).
- Ensure `api/requirements.txt` lists all necessary dependencies (`fastapi`, `uvicorn`, `torch`, `timm`, `google-cloud-storage`, etc.).

**2. Build and Push API Docker Image to GCP Artifact Registry:**
You can use the Makefile targets:

```bash
# Ensure your GCP_PROJECT_ID_LOCAL and GCP_REGION_LOCAL are set correctly,
# either by gcloud config or by passing them to make:
# make api_docker_push_gcp GCP_PROJECT_ID_LOCAL=your-gcp-proj GCP_REGION_LOCAL=your-region API_ARTIFACT_REGISTRY_REPO=your-ar-repo

make api_docker_push_gcp API_IMAGE_TAG=v1.0.0 # Use a specific tag
```

This command will:

- Build the Docker image using `api/Dockerfile`, tagging it locally (e.g., `team-zeal-project-api:v1.0.0`).
- Tag the image for Artifact Registry (e.g., `us-west2-docker.pkg.dev/your-gcp-project/team-zeal-project/team-zeal-project-api:v1.0.0`).
- Push the image to Artifact Registry.

**3. Deploy to GCP Cloud Functions (2nd Generation):**
You can deploy using the `gcloud` CLI. Replace placeholders with your actual values.

```bash
# Variables
FUNCTION_NAME="team-zeal-classifier-api"
GCP_REGION="us-west2" # Choose the region for your function
IMAGE_URI="$(gcloud config get-value project)/$(API_ARTIFACT_REGISTRY_REPO)/$(API_IMAGE_NAME):$(API_IMAGE_TAG)" # Construct this from Makefile variables
# Example IMAGE_URI: us-west2-docker.pkg.dev/your-gcp-project-id/team-zeal-project/team-zeal-project-api:v1.0.0
# Retrieve the latest pushed image URI if using 'latest' tag or a dynamic tag:
# IMAGE_URI=$(gcloud artifacts docker images list $(GCP_REGION_LOCAL)-docker.pkg.dev/$(GCP_PROJECT_ID_LOCAL)/$(API_ARTIFACT_REGISTRY_REPO)/$(API_IMAGE_NAME) --sort-by=~UPDATE_TIME --limit=1 --format='value(IMAGE_PULL_URL_WITH_TAG)')

MODEL_GCS_PATH_FOR_FUNCTION="gs://team-zeal-models/main/your_model_commit_sha.pth" # IMPORTANT: Update this path

gcloud functions deploy $FUNCTION_NAME \
--gen2 \
--region=$GCP_REGION \
--runtime=python310 # For container deployments, runtime is less critical but good to specify
--source=. # Dummy source, not used for container deployment from AR
--entry-point=http # For container deployments, this isn't the Python function name
--trigger-http \
--allow-unauthenticated \
--container-image=$IMAGE_URI \
--set-env-vars MODEL_GCS_PATH=$MODEL_GCS_PATH_FOR_FUNCTION \
--memory=2Gi # Adjust memory as needed for your model
# --cpu=1 # Adjust CPU if needed
# --timeout=300s # Adjust timeout
# --service-account=YOUR_SERVICE_ACCOUNT_EMAIL # Optional: if specific SA needed for function runtime
```

- `--gen2`: Specifies a 2nd Generation function.
- `--region`: The GCP region for deployment.
- `--runtime`: While deploying a container, this is less critical but specifying it can help GCP optimize. Python 3.10 matches your project.
- `--source=.`: For container deployments from Artifact Registry, a source directory is still formally required by `gcloud` but isn't used to build the function if `--container-image` is provided. You can point to your project root or `api/`.
- `--entry-point=http`: For containerized HTTP functions (2nd Gen), the entry point is not a specific Python function name in the Cloud Functions UI sense. The container's `CMD` handles starting the server.
- `--trigger-http`: Makes the function invokable via HTTP.
- `--allow-unauthenticated`: Allows public access for testing. **For production, configure proper authentication.**
- `--container-image`: **Crucial.** This points to your API image in Artifact Registry.
- `--set-env-vars`: Sets environment variables for your function. `MODEL_GCS_PATH` is vital for your API to find the model.
- `--memory`, `--cpu`, `--timeout`: Adjust these based on your model's needs. Loading PyTorch models can be memory-intensive.
- `--service-account`: If your function needs specific permissions at runtime (e.g., to access other GCP services beyond what the default compute SA provides, or if you use a more restricted SA).

**4. Test the Deployed Function:**

- After deployment, the `gcloud` command will output an HTTPS Trigger URL.
- You can use `curl`, Postman, or your browser (for GET requests) to test the `/predict/` endpoint:

```bash
# Get the trigger URL from the deployment output or from the GCP Console
FUNCTION_URL="YOUR_DEPLOYED_FUNCTION_URL"

curl -X POST -F "file=@/path/to/your/test_image.jpg" $FUNCTION_URL/predict/
```

- Check the logs in GCP Console (Cloud Functions > Your Function > Logs) for any errors or output.

**Notes:**

- The first deployment or updates can take a few minutes.
- If you encounter "Container failed to start" errors, check the Cloud Function logs in GCP. Common issues include incorrect `PORT` handling in the Docker container, missing dependencies, or insufficient memory/CPU.
- The `MODEL_GCS_PATH` must be accurate and the Cloud Function's runtime service account must have permission to read from that GCS bucket. By default, 2nd gen functions use the Compute Engine default service account, which often has broad read access within the project, but it's good practice to use more granular permissions with dedicated service accounts in production.

- [ ] **3.4 Dockerize & Deploy Model with GCP Cloud Run**
  - [ ] Containerization and deployment steps
  - [ ] Testing and result documentation
- [ ] **3.5 Interactive UI Deployment**
  - [ ] Streamlit or Gradio app for model demonstration
  - [ ] Deployment on Hugging Face platform
  - [ ] Integration of UI deployment into GitHub Actions workflow
  - [ ] Screenshots and usage examples

## 4. Documentation & Repository Updates

- [ ] **4.1 Comprehensive README**
  - [ ] Setup, usage, and documentation for all CI/CD, CML, and deployment steps
  - [ ] Screenshots and results of deployments
- [ ] **4.2 Resource Cleanup Reminder**
  - [ ] Checklist for removing GCP resources to avoid charges

---

> **Checklist:** Use this as a guide for documenting your Phase 3 deliverables. Focus on automation, deployment, and clear, reproducible instructions for all steps.
