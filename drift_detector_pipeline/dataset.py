# drift_detector_pipeline/dataset.py
import tarfile
import os
import logging
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from omegaconf import DictConfig # For type hinting cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Define paths relative to project root (assuming execution from root)
RAW_DATA_PATH = 'data/raw/imagenette2-160.tgz'
PROCESSED_DATA_PATH = 'data/processed/imagenette2-160' # Note: ImageFolder expects the parent dir

# --- Data Extraction Logic (from before) ---
def extract_data():
    """ Extracts raw data archive (imagenette2-160.tgz) to the
        processed data directory (data/processed/).
    """
    abs_raw_path = os.path.abspath(RAW_DATA_PATH)
    # Corrected: Target path for extraction is 'data/processed/'
    abs_processed_parent_path = os.path.abspath(os.path.dirname(PROCESSED_DATA_PATH))
    final_extracted_path = os.path.abspath(PROCESSED_DATA_PATH)

    log.info(f'Attempting to extract {abs_raw_path} to {abs_processed_parent_path}')
    os.makedirs(abs_processed_parent_path, exist_ok=True)

    if not os.path.exists(abs_raw_path):
         log.warning(f"Raw data file not found at {abs_raw_path}.")
         log.info("Attempting to pull data with DVC ('dvc pull data/raw/imagenette2-160.tgz')...")
         pull_result = os.system('dvc pull data/raw/imagenette2-160.tgz')
         if pull_result != 0 or not os.path.exists(abs_raw_path):
              log.error("DVC pull failed or file still not found. Please run 'dvc pull' manually.")
              sys.exit(1)

    # Check if already extracted
    if os.path.exists(final_extracted_path):
        log.info(f"Processed data already found at {final_extracted_path}. Skipping extraction.")
        return

    try:
        with tarfile.open(abs_raw_path, 'r:gz') as tar:
            # Extract into the parent 'processed' directory
            tar.extractall(path=abs_processed_parent_path)
        log.info(f'Extraction complete. Data available in {final_extracted_path}/')
    except tarfile.ReadError:
         log.error(f"Error: Failed to read tar file {abs_raw_path}. It might be corrupted.")
         sys.exit(1)
    except Exception as e:
        log.error(f"An error occurred during extraction: {e}")
        sys.exit(1)

# --- PyTorch Dataset and DataLoader Logic ---

def get_transforms(cfg: DictConfig) -> tuple[transforms.Compose, transforms.Compose]:
    """ Get training and validation transforms based on config. """
    img_size = cfg.data.img_size # e.g., 224

    # Normalization stats (ImageNet)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Transforms for training (include augmentation)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size), # Standard augmentation
        transforms.RandomHorizontalFlip(),      # Standard augmentation
        transforms.ToTensor(),
        normalize,
    ])

    # Transforms for validation (no augmentation, just resize/crop)
    val_transform = transforms.Compose([
        transforms.Resize(256), # Resize slightly larger then center crop
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform

def get_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """ Creates train and validation DataLoaders.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        tuple[DataLoader, DataLoader]: train_loader, val_loader
    """
    log.info("Creating DataLoaders...")

    # Ensure data is extracted first
    extract_data() # Call extraction logic

    processed_data_dir = os.path.abspath(cfg.data.processed_path)
    train_dir = os.path.join(processed_data_dir, 'train')
    val_dir = os.path.join(processed_data_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        log.error(f"Processed data directories not found after extraction check.")
        log.error(f"Looked for: {train_dir} and {val_dir}")
        sys.exit(1)

    train_transform, val_transform = get_transforms(cfg)

    # Create datasets using ImageFolder
    try:
        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            root=val_dir,
            transform=val_transform
        )
        log.info(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    except Exception as e:
        log.error(f"Error creating ImageFolder datasets: {e}")
        log.error(f"Please check the contents of {processed_data_dir}")
        sys.exit(1)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True, # Shuffle training data
        num_workers=cfg.data.dataloader_workers,
        pin_memory=True # Helps speed up data transfer to GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=cfg.data.dataloader_workers,
        pin_memory=True
    )

    log.info("DataLoaders created successfully.")
    return train_loader, val_loader


# ==============================================================
# This block is mainly for testing the extraction logic directly
# ==============================================================
if __name__ == '__main__':
    # This part is primarily for testing extraction.
    # To test dataloaders, you'd typically run the main training script.
    print("Running data extraction check...")
    extract_data()
    print("Data extraction script finished.")
    # You could add basic dataloader testing here if needed,
    # but it requires constructing a dummy cfg object.