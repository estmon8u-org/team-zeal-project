# drift_detector_pipeline/dataset.py
import tarfile
import os
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths relative to project root
# Assumes script execution happens from the project root directory (e.g., via 'make')
RAW_DATA_PATH = 'data/raw/imagenette2-160.tgz'
PROCESSED_DATA_PATH = 'data/processed/'

def extract_data():
    """ Extracts raw data archive (imagenette2-160.tgz) to the
        processed data directory (data/processed/).
    """
    logger = logging.getLogger(__name__)
    # Get absolute paths based on CWD (should be project root when run via make)
    abs_raw_path = os.path.abspath(RAW_DATA_PATH)
    abs_processed_path = os.path.abspath(PROCESSED_DATA_PATH)

    logger.info(f'Attempting to extract {abs_raw_path} to {abs_processed_path}')

    # Ensure processed directory exists
    os.makedirs(abs_processed_path, exist_ok=True)

    # Check if raw file exists, try dvc pull if not
    if not os.path.exists(abs_raw_path):
         logger.warning(f"Raw data file not found at {abs_raw_path}.")
         logger.info("Attempting to pull data with DVC ('dvc pull data/raw/imagenette2-160.tgz')...")
         # Try to run dvc pull - relies on dvc being in PATH
         pull_result = os.system('dvc pull data/raw/imagenette2-160.tgz')
         if pull_result != 0 or not os.path.exists(abs_raw_path):
              logger.error("DVC pull failed or file still not found. Please run 'dvc pull' manually in the project root before running this script.")
              sys.exit(1) # Exit if data can't be found/pulled

    # Extract the data
    try:
        with tarfile.open(abs_raw_path, 'r:gz') as tar:
            tar.extractall(path=abs_processed_path)
        logger.info(f'Extraction complete. Data available in {abs_processed_path}/imagenette2-160/')
    except tarfile.ReadError:
         logger.error(f"Error: Failed to read tar file {abs_raw_path}. It might be corrupted.")
         sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during extraction: {e}")
        sys.exit(1)

# --- PyTorch Dataset and DataLoader code will go below here ---
# class ImagenetteDataset(torch.utils.data.Dataset):
#     def __init__(self, ...):
#         pass
#     def __len__(self):
#         pass
#     def __getitem__(self, idx):
#         pass

# ==============================================================
# This allows running the extraction directly if needed,
# although 'make process_data' is the intended way.
# ==============================================================
if __name__ == '__main__':
    print("Running data extraction...")
    extract_data()
    print("Data extraction script finished.")