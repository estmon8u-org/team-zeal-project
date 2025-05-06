# drift_detector_pipeline/dataset.py
"""
Handles dataset extraction, transformation definition, and DataLoader creation
for the Imagenette-160 dataset, aligning with project MLOps pipeline requirements.
"""

import logging
import os
import subprocess  # For robust DVC command execution
import sys
import tarfile

from omegaconf import DictConfig, OmegaConf  # Added OmegaConf for standalone testing
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configure module-level logger
# This basicConfig might be overridden if the main application (e.g., train.py
# with Hydra) sets up logging. For standalone script runs or simple tests, this is sufficient.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def extract_data(cfg: DictConfig) -> None:
    """
    Extracts the raw Imagenette dataset archive specified in `cfg.data.raw_path`
    into the directory structure expected for `cfg.data.processed_path`.

    Checks for the raw archive. If not found, attempts to pull it using
    'dvc pull <raw_path>.dvc'. If the processed directory (e.g.,
    'data/processed/imagenette2-160') already exists and is not empty,
    extraction is skipped.

    Args:
        cfg (DictConfig): Hydra configuration object containing data paths:
                          `cfg.data.raw_path` (e.g., "data/raw/imagenette2-160.tgz")
                          `cfg.data.processed_path` (e.g., "data/processed/imagenette2-160")

    Raises:
        SystemExit: If the raw data cannot be found or pulled via DVC,
                    or if extraction fails.
    """
    raw_path = cfg.data.raw_path
    processed_path = cfg.data.processed_path

    # The actual extraction target is the parent of the 'imagenette2-160' folder.
    # e.g., if processed_path is "data/processed/imagenette2-160",
    # extraction_target_dir is "data/processed"
    extraction_target_dir = os.path.dirname(processed_path)

    abs_raw_path = os.path.abspath(raw_path)
    abs_processed_path = os.path.abspath(processed_path)
    abs_extraction_target_dir = os.path.abspath(extraction_target_dir)

    log.info(f"Target for extracted data: {abs_processed_path}")

    # Ensure the parent directory for extraction exists (e.g., 'data/processed/')
    try:
        os.makedirs(abs_extraction_target_dir, exist_ok=True)
    except OSError as e:
        log.error(f"Could not create directory {abs_extraction_target_dir}: {e}")
        sys.exit(1)

    # 1. Check if raw data exists, try DVC pull if not
    if not os.path.exists(abs_raw_path):
        log.warning(f"Raw data file not found at {abs_raw_path}.")
        # DVC expects the .dvc metafile to be named <tracked_file>.dvc
        dvc_meta_file = raw_path + ".dvc"
        log.info(f"Attempting to pull data with DVC ('dvc pull {dvc_meta_file}')...")

        if not os.path.exists(dvc_meta_file):
            log.error(
                f"DVC metafile {dvc_meta_file} not found. "
                "Cannot pull data. Ensure the .dvc file is present and committed."
            )
            sys.exit(1)

        try:
            # Using subprocess.run for better control and error handling
            pull_command = ["dvc", "pull", dvc_meta_file]

            result = subprocess.run(
                pull_command,
                check=True,  # Raises CalledProcessError on non-zero exit
                capture_output=True,
                text=True,
                cwd=os.getcwd(),  # Run DVC from the project root
            )
            log.info(f"DVC pull output:\n{result.stdout}")
            if result.stderr:  # DVC might output info to stderr too
                log.info(
                    f"DVC pull stderr:\n{result.stderr}"
                )  # Changed to info as DVC uses stderr for progress

            if not os.path.exists(abs_raw_path):  # Check again after pull
                log.error(
                    f"DVC pull reported success, but file still not found at {abs_raw_path}. "
                    "Please ensure DVC is configured correctly and the remote is accessible."
                )
                sys.exit(1)
            log.info(f"DVC pull successful. Raw data available at {abs_raw_path}.")

        except subprocess.CalledProcessError as e:
            log.error(
                f"DVC pull command ('{' '.join(e.cmd)}') failed with exit code {e.returncode}.\n"
                f"Stdout: {e.stdout}\nStderr: {e.stderr}"
                "Please ensure DVC is initialized and the remote is configured."
            )
            sys.exit(1)
        except FileNotFoundError:
            log.error("DVC command not found. Please ensure DVC is installed and in your PATH.")
            sys.exit(1)
        except Exception as e:  # Catch other potential errors during DVC pull
            log.error(f"An unexpected error occurred during DVC pull: {e}")
            sys.exit(1)

    # 2. Check if data is already extracted and not empty
    if os.path.exists(abs_processed_path) and os.listdir(abs_processed_path):
        log.info(
            f"Processed data directory {abs_processed_path} already exists and is not empty. Skipping extraction."
        )
        return

    # 3. Proceed with extraction
    log.info(f"Extracting {abs_raw_path} to {abs_extraction_target_dir}...")
    try:
        with tarfile.open(abs_raw_path, "r:gz") as tar:
            tar.extractall(path=abs_extraction_target_dir)
        # Verify that the expected directory (e.g., imagenette2-160) was created by extraction
        if not os.path.exists(abs_processed_path):
            log.error(
                f"Extraction completed, but the expected directory {abs_processed_path} was not found. "
                f"The archive might have an unexpected internal structure or name."
            )
            sys.exit(1)
        log.info(f"Extraction complete. Data available in {abs_processed_path}")
    except tarfile.ReadError:
        log.error(f"Failed to read tar file {abs_raw_path}. It might be corrupted.")
        sys.exit(1)
    except PermissionError:
        log.error(f"Permission denied during extraction to {abs_extraction_target_dir}.")
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
    try:
        img_size = cfg.data.img_size
    except AttributeError:  # More specific error for missing key
        log.error("Configuration error: 'data.img_size' not found in config.")
        sys.exit(1)
    except Exception as e:  # Catch other errors like wrong type
        log.error(f"Configuration error with 'data.img_size': {e}")
        sys.exit(1)

    # Standard ImageNet normalization constants
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    # Transformations for the training set
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Transformations for the validation set
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),  # Resize smaller edge to 256, maintaining aspect ratio
            transforms.CenterCrop(img_size),  # Crop the center to img_size x img_size
            transforms.ToTensor(),
            normalize,
        ]
    )
    log.info(f"Image transforms created for image size {img_size}x{img_size}.")
    return train_transform, val_transform


def get_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for the training and validation sets.

    Ensures data is extracted via `extract_data()`. Assumes data is in
    `cfg.data.processed_path` with 'train' and 'val' subdirectories.

    Args:
        cfg (DictConfig): Hydra configuration object containing data and training
                          parameters like `cfg.data.processed_path`,
                          `cfg.training.batch_size`, `cfg.data.dataloader_workers`,
                          and `cfg.run.device`.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader
            and validation DataLoader.

    Raises:
        SystemExit: If data directories are not found or dataset creation fails.
    """
    log.info("Attempting to create DataLoaders...")

    # 1. Ensure data is extracted
    extract_data(cfg)  # Pass the config

    # 2. Define paths for train and validation data
    processed_data_dir = os.path.abspath(cfg.data.processed_path)
    train_dir = os.path.join(processed_data_dir, "train")
    val_dir = os.path.join(processed_data_dir, "val")

    # Verify that the expected train and val directories exist
    if not os.path.isdir(train_dir):
        log.error(
            f"Training data directory not found at {train_dir}. "
            "Ensure 'extract_data' (or 'make process_data') completed successfully and "
            "the archive contains a 'train' subdirectory at the expected location."
        )
        sys.exit(1)
    if not os.path.isdir(val_dir):
        log.error(
            f"Validation data directory not found at {val_dir}. "
            "Ensure 'extract_data' (or 'make process_data') completed successfully and "
            "the archive contains a 'val' subdirectory at the expected location."
        )
        sys.exit(1)

    # 3. Get transformations
    train_transform, val_transform = get_transforms(cfg)

    # 4. Create datasets using torchvision's ImageFolder
    try:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
        log.info(
            f"Successfully created datasets: "
            f"{len(train_dataset):,} training images, {len(val_dataset):,} validation images "
            f"from {processed_data_dir}"
        )
    except FileNotFoundError:  # Should be caught by earlier checks, but good for defense
        log.error(
            f"ImageFolder dataset creation failed: Could not find image folders in {train_dir} or {val_dir}."
        )
        sys.exit(1)
    except Exception as e:
        log.error(f"An unexpected error occurred during ImageFolder dataset creation: {e}")
        sys.exit(1)

    # 5. Create DataLoaders
    pin_memory_flag = cfg.run.device == "cuda" and torch.cuda.is_available()
    num_workers = cfg.data.dataloader_workers

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory_flag,
        persistent_workers=bool(num_workers > 0),  # Keep workers alive if num_workers > 0
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory_flag,
        persistent_workers=bool(num_workers > 0),
    )

    log.info(
        f"DataLoaders created. Batch size: {cfg.training.batch_size}, "
        f"Num workers: {num_workers}, Pin memory: {pin_memory_flag}."
    )
    return train_loader, val_loader


# This block allows testing the extraction and dataloader logic by running the script directly.
# Example: python -m drift_detector_pipeline.dataset
if __name__ == "__main__":
    log.info("--- Running dataset.py standalone for testing ---")

    # Create a dummy configuration for standalone testing
    # This mimics the structure expected from conf/config.yaml
    # NOTE: For this test to fully pass, 'data/raw/imagenette2-160.tgz.dvc'
    #       must exist, and DVC must be able to pull 'data/raw/imagenette2-160.tgz'.
    #       Alternatively, place 'imagenette2-160.tgz' manually in 'data/raw/'.
    dummy_cfg_dict = {
        "data": {
            "raw_path": "data/raw/imagenette2-160.tgz",
            "processed_path": "data/processed/imagenette2-160",
            "img_size": 224,
            "dataloader_workers": 0,  # Set to 0 for simple test to avoid multiprocessing issues
        },
        "training": {
            "batch_size": 2,  # Small batch size for quick test
        },
        "run": {
            "device": "cpu",  # Default to CPU for standalone test
        },
    }
    # Ensure 'data/raw' directory exists for the dummy test if dvc pull is not used/fails
    if not os.path.exists(os.path.dirname(dummy_cfg_dict["data"]["raw_path"])):
        os.makedirs(os.path.dirname(dummy_cfg_dict["data"]["raw_path"]), exist_ok=True)

    dummy_cfg = OmegaConf.create(dummy_cfg_dict)

    # Test 1: Data Extraction
    print("\n--- Testing Data Extraction ---")
    try:
        extract_data(dummy_cfg)
        print("extract_data function completed.")
        # Check if the processed directory exists after extraction attempt
        if os.path.exists(dummy_cfg.data.processed_path):
            print(f"Processed data directory found at: {dummy_cfg.data.processed_path}")
            # Basic check for content (e.g., train/val folders)
            if os.path.exists(
                os.path.join(dummy_cfg.data.processed_path, "train")
            ) and os.path.exists(os.path.join(dummy_cfg.data.processed_path, "val")):
                print("Train and Val subdirectories found within processed data.")
            else:
                print("WARNING: Train or Val subdirectories MISSING in processed data path.")
        else:
            print(
                f"WARNING: Processed data directory NOT found at: {dummy_cfg.data.processed_path}"
            )

    except SystemExit as e:
        print(
            f"extract_data exited with code {e.code}. This might be expected if DVC or data is not set up."
        )
    except Exception as e:
        print(f"An error occurred during extract_data test: {e}")

    # Test 2: Dataloader Creation (only if extraction was successful or skipped)
    # This part will only work if the data is actually present and extracted.
    if (
        os.path.exists(dummy_cfg.data.processed_path)
        and os.path.exists(os.path.join(dummy_cfg.data.processed_path, "train"))
        and os.path.exists(os.path.join(dummy_cfg.data.processed_path, "val"))
    ):
        print("\n--- Testing Dataloader Creation ---")
        try:
            train_dl, val_dl = get_dataloaders(dummy_cfg)
            print("Successfully created DataLoaders.")
            print(f"  Number of training batches: {len(train_dl)}")
            print(f"  Number of validation batches: {len(val_dl)}")

            # Try to get one batch to see if it works
            if len(train_dl) > 0:
                print("  Attempting to load one batch from training loader...")
                train_features, train_labels = next(iter(train_dl))
                print(f"    Training batch features shape: {train_features.shape}")
                print(f"    Training batch labels shape: {train_labels.shape}")
            else:
                print("  Skipping loading from training_loader (empty).")

            if len(val_dl) > 0:
                print("  Attempting to load one batch from validation loader...")
                val_features, val_labels = next(iter(val_dl))
                print(f"    Validation batch features shape: {val_features.shape}")
                print(f"    Validation batch labels shape: {val_labels.shape}")
            else:
                print("  Skipping loading from validation_loader (empty).")

        except SystemExit as e:
            print(f"get_dataloaders exited with code {e.code}.")
        except Exception as e:
            print(f"An error occurred during get_dataloaders test: {e}")
    else:
        print("\n--- Skipping Dataloader Creation Test ---")
        print(
            f"Reason: Processed data (with train/val subdirs) not available at '{dummy_cfg.data.processed_path}'."
        )
        print("Ensure DVC is configured and 'dvc pull' has been run, or data is manually placed.")

    log.info("--- Finished dataset.py standalone test ---")
