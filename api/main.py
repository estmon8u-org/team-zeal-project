# api/main.py
import io
import logging
import os
import sys
import tempfile
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from google.cloud import storage  # For GCS
from PIL import Image
import timm
import torch
from torchvision import transforms

load_dotenv()  # This will load variables from .env

# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Ensure logs are visible in Cloud Functions & local console
if not logger.hasHandlers():  # Add handler only if no handlers are configured
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False  # Avoid duplicate logs if root logger is also configured

# --- Configuration ---
MODEL_NAME = "resnet18"  # Should match the model architecture saved
NUM_CLASSES = 10  # Imagenette
IMG_SIZE = 224  # As used in your training

# Path for downloaded model in the Cloud Function's ephemeral storage
LOCAL_MODEL_DIR = os.path.join(tempfile.gettempdir(), "model_cache")
LOCAL_MODEL_FILENAME = "downloaded_model.pth"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, LOCAL_MODEL_FILENAME)

# --- Global Variables ---
model_loaded = False
model_instance = None  # Renamed to avoid conflict with 'model' module/variable
api_transforms = None

app = FastAPI(title="Team Zeal Image Classifier API (GCS Model)")


# Add this function after your imports or helper functions
def clear_model_cache():
    """Clear cached model files to force a fresh download from GCS."""
    try:
        if os.path.exists(LOCAL_MODEL_PATH):
            logger.info(f"Clearing cached model at {LOCAL_MODEL_PATH}")
            os.remove(LOCAL_MODEL_PATH)
            logger.info("Model cache cleared successfully")
        else:
            logger.info("No cached model found to clear")
    except Exception as e:
        logger.error(f"Failed to clear model cache: {e}")


def get_api_transforms(img_size: int) -> transforms.Compose:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def download_model_from_gcs_uri(gcs_uri: str, destination_file_name: str):
    """Downloads a model from a GCS URI (gs://bucket/blob)."""
    if not gcs_uri.startswith("gs://"):
        logger.error(f"Invalid GCS URI: {gcs_uri}. Must start with gs://")
        return False

    if os.path.exists(destination_file_name):
        logger.info(f"Model already cached at {destination_file_name}, skipping GCS download.")
        return True

    try:
        bucket_name = gcs_uri.split("/")[2]
        source_blob_name = "/".join(gcs_uri.split("/")[3:])

        storage_client = storage.Client()  # Assumes ADC or SA for the CF runtime
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        logger.info(
            f"Attempting to download {source_blob_name} from bucket {bucket_name} to {destination_file_name}..."
        )
        blob.download_to_filename(destination_file_name)
        logger.info(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
        return True
    except Exception as e:
        logger.error(f"Failed to download {gcs_uri}: {e}")
        return False


async def startup_event():
    global model_instance, model_loaded, api_transforms
    # Only attempt to load if not already loaded (important for Cloud Functions lifecycle)
    if model_loaded:
        logger.info("Model already loaded. Skipping startup actions.")
        return

    logger.info("API Startup: Attempting to load model from GCS...")

    model_gcs_uri = os.getenv(
        "MODEL_GCS_PATH"
    )  # e.g., gs://team-zeal-models/ci-models/sha_model.pth

    if not model_gcs_uri:
        logger.error("MODEL_GCS_PATH environment variable not set. Cannot load model.")
        return  # Model stays unloaded, endpoint will raise 503

    model_downloaded = download_model_from_gcs_uri(model_gcs_uri, LOCAL_MODEL_PATH)

    if model_downloaded and os.path.exists(LOCAL_MODEL_PATH):
        try:
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading model from {LOCAL_MODEL_PATH} onto device: {current_device}")

            temp_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
            temp_model.load_state_dict(torch.load(LOCAL_MODEL_PATH, map_location=current_device))

            model_instance = temp_model  # Assign to global
            model_instance.to(current_device)
            model_instance.eval()

            api_transforms = get_api_transforms(IMG_SIZE)
            model_loaded = True
            logger.info("Model loaded successfully from GCS.")
        except Exception as e:
            logger.exception(f"Error loading model from local GCS copy {LOCAL_MODEL_PATH}: {e}")
            model_loaded = False  # Ensure it's false on error
    else:
        logger.error(
            f"Failed to download or find model at {LOCAL_MODEL_PATH} from GCS URI {model_gcs_uri}."
        )
        model_loaded = False


app.add_event_handler("startup", startup_event)


@app.get("/")
async def read_root():  # Added async
    if model_loaded:
        return {
            "message": f"Team Zeal Image Classifier API ready. Model: {os.getenv('MODEL_GCS_PATH', 'N/A')}"
        }
    else:
        return {
            "message": "Team Zeal Image Classifier API starting or model load failed. Please check logs or try again shortly."
        }


@app.post("/predict/")
async def predict_image_endpoint(file: UploadFile = File(...)) -> Dict:
    global model_instance, model_loaded, api_transforms  # Use model_instance
    request_id = os.urandom(8).hex()
    logger.info(
        f"Request ID [{request_id}]: Received prediction request for file: {file.filename}"
    )

    if not model_loaded or model_instance is None or api_transforms is None:
        logger.error(f"Request ID [{request_id}]: Model not loaded, cannot process request.")
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Please try again later or check server logs.",
        )

    try:
        logger.info(f"Request ID [{request_id}]: Processing image.")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img_tensor = api_transforms(image).unsqueeze(0)
        current_device = next(model_instance.parameters()).device
        img_tensor = img_tensor.to(current_device)

        with torch.no_grad():
            outputs = model_instance(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top_prob, top_catid = torch.max(probabilities, dim=0)

        # Ensure class_names list matches your model's output
        class_names = [
            "tench",
            "English springer",
            "cassette player",
            "chain saw",
            "church",
            "French horn",
            "garbage truck",
            "gas pump",
            "golf ball",
            "parachute",
        ]  # Imagenette specific
        if top_catid.item() >= len(class_names):
            logger.error(
                f"Request ID [{request_id}]: Predicted class ID {top_catid.item()} is out of bounds for class_names list (len: {len(class_names)})."
            )
            raise HTTPException(
                status_code=500, detail="Model prediction error: class ID out of bounds."
            )

        predicted_class = class_names[top_catid.item()]
        confidence = top_prob.item()

        logger.info(
            f"Request ID [{request_id}]: Prediction: {predicted_class}, Confidence: {confidence:.4f}"
        )
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": {
                class_names[i]: float(probabilities[i].item()) for i in range(len(class_names))
            },
        }
    except Exception as e:
        logger.exception(f"Request ID [{request_id}]: Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        if file:
            await file.close()


# For local testing with uvicorn
if __name__ == "__main__":
    import argparse

    import uvicorn

    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Team Zeal Image Classifier API")
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear model cache before starting"
    )
    args = parser.parse_args()

    # Check if cache clearing is requested via command line or environment variable
    if args.clear_cache or os.getenv("CLEAR_MODEL_CACHE", "").lower() in ("true", "1", "yes"):
        logger.info("Clearing model cache as requested...")
        clear_model_cache()

    # For local test, manually set env vars if needed:
    # Example: export MODEL_GCS_PATH="gs://your-bucket/path/to/your_model.pth"
    # Ensure you have `gcloud auth application-default login` run for local GCS access
    if not os.getenv("MODEL_GCS_PATH"):
        logger.warning(
            "MODEL_GCS_PATH env var not set. API might not load model on startup for local run."
        )
        logger.warning(
            "For local testing, set it e.g.: export MODEL_GCS_PATH=gs://team-zeal-models/your-branch/your-sha_model.pth"
        )

    # Use PORT from env var if available (like in Cloud Run/Functions), otherwise default for local
    port = int(os.environ.get("PORT", 8008))  # Default to 8008 for local, Cloud Run provides PORT
    logger.info(f"Starting FastAPI server locally with Uvicorn on host 0.0.0.0 port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
