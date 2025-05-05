# drift_detector_pipeline/dataset.py
"""
Handles dataset extraction, transformation definition, and DataLoader creation
for the Imagenette-160 dataset.
"""

import logging
import os
import sys
import tarfile

from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

# Define constants for paths relative to the project root
# Assumes the script is run from the project root directory
RAW_DATA_PATH = "data/raw/imagenette2-160.tgz"
PROCESSED_DATA_PATH = (
    "data/processed/imagenette2-160"  # ImageFolder expects the parent containing train/val
)


def extract_data() -> None:
    """
    Extracts the raw Imagenette dataset archive (imagenette2-160.tgz)
    into the 'data/processed/' directory if it doesn't already exist.

    Checks for the raw archive in 'data/raw/'. If not found, attempts
    to pull it using 'dvc pull'. If the processed directory
    'data/processed/imagenette2-160' already exists, extraction is skipped.

    Raises:
        SystemExit: If the raw data cannot be found or pulled via DVC,
                    or if extraction fails.
    """
    abs_raw_path = os.path.abspath(RAW_DATA_PATH)
    # Target directory for extraction is the parent of the final expected path
    abs_processed_parent_path = os.path.abspath(os.path.dirname(PROCESSED_DATA_PATH))
    final_extracted_path = os.path.abspath(PROCESSED_DATA_PATH)

    log.info(f"Checking for dataset extraction: target {final_extracted_path}")

    # Ensure parent directory exists
    os.makedirs(abs_processed_parent_path, exist_ok=True)

    # Check if raw data exists, try DVC pull if not
    if not os.path.exists(abs_raw_path):
        log.warning(f"Raw data file not found at {abs_raw_path}.")
        log.info("Attempting to pull data with DVC ('dvc pull data/raw/imagenette2-160.tgz')...")
        # Use os.system for simplicity; subprocess is generally safer
        pull_result = os.system("dvc pull data/raw/imagenette2-160.tgz")
        if pull_result != 0 or not os.path.exists(abs_raw_path):
            log.error(
                f"DVC pull failed (exit code {pull_result}) or file still not found at {abs_raw_path}. "
                "Please ensure DVC is configured and run 'dvc pull' manually."
            )
            sys.exit(1)  # Exit if data cannot be obtained
        log.info("DVC pull successful or file already existed.")

    # Check if data is already extracted
    if os.path.exists(final_extracted_path):
        log.info(
            f"Processed data directory already found at {final_extracted_path}. Skipping extraction."
        )
        return  # Skip extraction

    # Proceed with extraction if raw data exists and processed data doesn't
    log.info(f"Extracting {abs_raw_path} to {abs_processed_parent_path}...")
    try:
        with tarfile.open(abs_raw_path, "r:gz") as tar:
            tar.extractall(path=abs_processed_parent_path)
        log.info(f"Extraction complete. Data available in {final_extracted_path}")
    except tarfile.ReadError:
        log.error(f"Error: Failed to read tar file {abs_raw_path}. It might be corrupted.")
        sys.exit(1)
    except PermissionError:
        log.error(f"Error: Permission denied during extraction to {abs_processed_parent_path}.")
        sys.exit(1)
    except Exception as e:
        log.error(f"An unexpected error occurred during extraction: {e}")
        sys.exit(1)


def get_transforms(cfg: DictConfig) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs PyTorch image transformations for training and validation.

    Uses ImageNet normalization statistics. Training includes standard
    augmentations (RandomResizedCrop, RandomHorizontalFlip). Validation
    uses deterministic resizing and center cropping.

    Args:
        cfg (DictConfig): Hydra configuration object containing data parameters
                          like `cfg.data.img_size`.

    Returns:
        tuple[transforms.Compose, transforms.Compose]: A tuple containing the
            training transforms and validation transforms, respectively.
    """
    img_size = cfg.data.img_size  # Target image size (e.g., 224)

    # Standard ImageNet normalization constants
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    # Define transformations for the training set (with augmentations)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size, scale=(0.8, 1.0)
            ),  # Crop a random portion and resize
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
            transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
            normalize,  # Normalize tensor values
        ]
    )

    # Define transformations for the validation set (deterministic)
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),  # Resize smaller edge to 256 maintaining aspect ratio
            transforms.CenterCrop(img_size),  # Crop the center 224x224 patch
            transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
            normalize,  # Normalize tensor values
        ]
    )
    log.info(f"Transforms created for image size {img_size}x{img_size}.")
    return train_transform, val_transform


def get_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for the training and validation sets.

    Assumes the Imagenette data has been extracted by `extract_data()` into
    'data/processed/imagenette2-160/' with 'train' and 'val' subdirectories.

    Args:
        cfg (DictConfig): Hydra configuration object containing data and training
                          parameters like `cfg.data.processed_path`,
                          `cfg.training.batch_size`, and `cfg.data.dataloader_workers`.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader
            and validation DataLoader.

    Raises:
        SystemExit: If the processed data directories ('train', 'val') are not found
                    or if dataset creation fails.
    """
    log.info("Creating DataLoaders...")

    # Ensure data is extracted before trying to load it
    extract_data()

    processed_data_dir = os.path.abspath(cfg.data.processed_path)
    train_dir = os.path.join(processed_data_dir, "train")
    val_dir = os.path.join(processed_data_dir, "val")

    # Verify that the expected train and val directories exist
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        log.error(
            f"Processed data directories ('train' and 'val') not found inside {processed_data_dir}. "
            "Ensure 'make process_data' completed successfully."
        )
        sys.exit(1)

    # Get the appropriate transformations
    train_transform, val_transform = get_transforms(cfg)

    # Create datasets using torchvision's ImageFolder
    try:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
        log.info(
            f"Found {len(train_dataset):,} training images and {len(val_dataset):,} validation images "
            f"in {processed_data_dir}"
        )
    except FileNotFoundError:
        log.error(
            f"Dataset creation failed: Could not find image folders in {train_dir} or {val_dir}."
        )
        sys.exit(1)
    except Exception as e:
        log.error(f"An unexpected error occurred during ImageFolder dataset creation: {e}")
        sys.exit(1)

    # Determine pin_memory based on device availability for efficiency
    pin_memory_flag = cfg.run.device == "cuda" and torch.cuda.is_available()

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=cfg.data.dataloader_workers,  # Use multiple workers to speed up data loading
        pin_memory=pin_memory_flag,  # Faster data transfer to GPU if True
        persistent_workers=bool(
            cfg.data.dataloader_workers > 0
        ),  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=cfg.data.dataloader_workers,
        pin_memory=pin_memory_flag,
        persistent_workers=bool(cfg.data.dataloader_workers > 0),
    )

    log.info(
        f"DataLoaders created with batch size {cfg.training.batch_size} and {cfg.data.dataloader_workers} workers."
    )
    return train_loader, val_loader


# This block allows testing the extraction logic by running the script directly.
# Example: python -m drift_detector_pipeline.dataset
if __name__ == "__main__":
    # This part is primarily for testing the extraction logic standalone.
    # Full dataloader testing requires Hydra config and is usually done
    # via integration testing or running the main training script.
    print("Running data extraction check...")
    extract_data()
    print("Data extraction script finished.")
