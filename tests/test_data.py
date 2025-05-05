# tests/test_dataset.py

"""
Unit tests for the dataset.py module in the drift_detector_pipeline package.

This module contains tests that verify the functionality of data extraction 
and transformation operations used in the drift detector pipeline. The tests
focus on verifying:
1. Image transformation pipelines for training and validation
2. Data extraction logic for handling the Imagenette dataset
3. Control flow for checking and extracting compressed data

Tests use pytest fixtures and unittest.mock to isolate functionality and
avoid actual file system operations during testing.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from torchvision import transforms

# Add the project root to the Python path to allow importing drift_detector_pipeline
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Now we can import from the package
from drift_detector_pipeline.dataset import get_transforms, extract_data # noqa: E402


# --- Fixtures ---
@pytest.fixture
def mock_cfg() -> OmegaConf:
    """Creates a mock OmegaConf object for testing."""
    # Create a base config structure similar to the yaml file
    conf = OmegaConf.create(
        {
            "data": {
                "img_size": 224,
                # Add other keys if needed by functions under test
                "raw_path": "data/raw/imagenette2-160.tgz",
                "processed_path": "data/processed/imagenette2-160",
            },
            # Add other top-level keys if needed
        }
    )
    return conf


# --- Tests for get_transforms ---
def test_get_transforms_returns_tuple(mock_cfg):
    """Verify get_transforms returns a tuple of two transforms."""
    train_tfm, val_tfm = get_transforms(mock_cfg)
    assert isinstance(train_tfm, transforms.Compose), "Train transform should be a Compose object"
    assert isinstance(val_tfm, transforms.Compose), "Validation transform should be a Compose object"


def test_get_transforms_validation_structure(mock_cfg):
    """Check the first few transforms in the validation pipeline."""
    _, val_tfm = get_transforms(mock_cfg)
    assert len(val_tfm.transforms) >= 3, "Validation transform should have at least 3 steps"
    assert isinstance(val_tfm.transforms[0], transforms.Resize), "First validation step should be Resize"
    assert isinstance(val_tfm.transforms[1], transforms.CenterCrop), "Second validation step should be CenterCrop"
    assert isinstance(val_tfm.transforms[2], transforms.ToTensor), "Third validation step should be ToTensor"


def test_get_transforms_train_structure(mock_cfg):
    """Check the first few transforms in the training pipeline."""
    train_tfm, _ = get_transforms(mock_cfg)
    assert len(train_tfm.transforms) >= 3, "Train transform should have at least 3 steps"
    assert isinstance(
        train_tfm.transforms[0], transforms.RandomResizedCrop
    ), "First train step should be RandomResizedCrop"
    assert isinstance(
        train_tfm.transforms[1], transforms.RandomHorizontalFlip
    ), "Second train step should be RandomHorizontalFlip"
    assert isinstance(train_tfm.transforms[2], transforms.ToTensor), "Third train step should be ToTensor"


# --- Tests for extract_data control flow ---

# Use patch from unittest.mock, works well with pytest
@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.os.makedirs")
def test_extract_data_skips_if_already_extracted(mock_makedirs, mock_tar_open, mock_exists):
    """Test that extraction is skipped if the target directory exists."""
    # Simulate that the processed data directory already exists
    mock_exists.return_value = True

    extract_data()

    # Assert that tarfile.open was NOT called because extraction should be skipped
    mock_tar_open.assert_not_called()
    # Assert makedirs was called (it's called before the check)
    mock_makedirs.assert_called_once()


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.system")
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.os.makedirs")
@patch("drift_detector_pipeline.dataset.log") # Mock logger to suppress output during test
def test_extract_data_pulls_if_raw_missing(mock_log, mock_makedirs, mock_tar_open, mock_os_system, mock_exists):
    """Test that 'dvc pull' is attempted if the raw file is missing."""
    # Simulate raw file missing, processed dir missing, then raw file exists after pull
    mock_exists.side_effect = [False, True, False] # 1st: raw missing, 2nd: raw exists (after pull), 3rd: processed missing
    mock_os_system.return_value = 0 # Simulate successful dvc pull

    # Mock the tarfile context manager
    mock_tar_context = MagicMock()
    mock_tar_open.return_value.__enter__.return_value = mock_tar_context

    extract_data()

    # Assert os.system was called with the dvc pull command
    mock_os_system.assert_called_once_with("dvc pull data/raw/imagenette2-160.tgz")
    # Assert tarfile.open was called (since the raw file "appeared" after pull)
    mock_tar_open.assert_called_once()
    # Assert extractall was called on the tar object
    mock_tar_context.extractall.assert_called_once()


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.system")
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.os.makedirs")
@patch("drift_detector_pipeline.dataset.log") # Mock logger
def test_extract_data_extracts_if_raw_present_processed_missing(mock_log, mock_makedirs, mock_tar_open, mock_os_system, mock_exists):
    """Test that extraction happens if raw exists but processed does not."""
    # Simulate raw file exists, processed dir missing
    mock_exists.side_effect = [True, False] # 1st: raw exists, 2nd: processed missing

    # Mock the tarfile context manager
    mock_tar_context = MagicMock()
    mock_tar_open.return_value.__enter__.return_value = mock_tar_context

    extract_data()

    # Assert os.system (dvc pull) was NOT called
    mock_os_system.assert_not_called()
    # Assert tarfile.open was called
    mock_tar_open.assert_called_once()
    # Assert extractall was called
    mock_tar_context.extractall.assert_called_once()


# --- Tests for extract_data error handling ---

@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.system")
@patch("drift_detector_pipeline.dataset.sys.exit")
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_exits_if_pull_fails(mock_log, mock_exit, mock_os_system, mock_exists):
    """
    Test that the system exits if 'dvc pull' fails to retrieve the raw data file.
    
    This test verifies the error handling when the raw data file doesn't exist
    and cannot be retrieved using DVC, ensuring the system exits with an appropriate
    error message.
    """
    # Simulate raw file missing and dvc pull failing
    mock_exists.return_value = False  # Raw file doesn't exist
    mock_os_system.return_value = 1   # Non-zero exit code means dvc pull failed
    
    extract_data()
    
    # Verify system exit was called when pull failed
    mock_exit.assert_called_once_with(1)
    # Verify error was logged
    mock_log.error.assert_called()


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.sys.exit")
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_handles_tarfile_error(mock_log, mock_exit, mock_tar_open, mock_exists):
    """
    Test that extraction errors are handled properly.
    
    This test simulates a corrupt or unreadable tar file and verifies
    that the system exits gracefully with appropriate error logging.
    """
    # Simulate raw file exists, processed missing, but tar extraction fails
    mock_exists.side_effect = [True, False]  # 1st: raw exists, 2nd: processed missing
    mock_tar_open.side_effect = Exception("Simulated tar extraction error")
    
    extract_data()
    
    # Verify system exit was called
    mock_exit.assert_called_once_with(1)
    # Verify error was logged
    mock_log.error.assert_called()


# --- Integration tests for transform behavior ---

def test_transform_dimensions(mock_cfg):
    """
    Test that image transformations produce the expected dimensions.
    
    This test verifies that both training and validation transforms resize images
    to the dimensions specified in the configuration.
    """
    import torch
    import numpy as np
    
    # Create a mock image (3 channels RGB, arbitrary dimensions)
    mock_img = torch.rand(3, 300, 400)  # Mock image with dimensions (C, H, W)
    
    # Get transforms from the function
    train_tfm, val_tfm = get_transforms(mock_cfg)
    
    # Apply transforms to mock image
    transformed_train = train_tfm(mock_img)
    transformed_val = val_tfm(mock_img)
    
    # Check dimensions match the config
    expected_size = mock_cfg.data.img_size
    assert transformed_train.shape == (3, expected_size, expected_size), \
        f"Train transform should output image of size (3, {expected_size}, {expected_size})"
    assert transformed_val.shape == (3, expected_size, expected_size), \
        f"Validation transform should output image of size (3, {expected_size}, {expected_size})"


def test_transform_normalization(mock_cfg):
    """
    Test that image normalization is applied correctly in transformations.
    
    This test verifies that the normalization step in both training and validation 
    transforms produces tensor values within the expected range (typically between -1 and 1)
    using the ImageNet normalization statistics.
    """
    import torch
    
    # Create a mock normalized image (all values = 0.5)
    # This should result in specific values after normalization
    mock_img = torch.ones(3, 224, 224) * 0.5  # Mid-grey image
    
    # Get transforms from the function
    train_tfm, val_tfm = get_transforms(mock_cfg)
    
    # Apply transforms to mock image
    transformed_train = train_tfm(mock_img)
    transformed_val = val_tfm(mock_img)
    
    # Check normalization was applied - values should be shifted by ImageNet means and scaled by std
    # If input is 0.5 and ImageNet mean is ~0.45 and std is ~0.225, output should be around 0.2
    # This is a rough check - actual values depend on exact normalization parameters
    assert -1.0 <= transformed_train.min() and transformed_train.max() <= 1.0, \
        "Normalized values should typically be between -1 and 1"
    assert -1.0 <= transformed_val.min() and transformed_val.max() <= 1.0, \
        "Normalized values should typically be between -1 and 1"


# --- Additional tests for configuration handling ---

def test_transform_respects_img_size_config():
    """
    Test that transforms correctly use the image size from configuration.
    
    This test verifies that get_transforms properly reads and applies 
    the image size parameter from the configuration.
    """
    # Create configs with different image sizes
    cfg_224 = OmegaConf.create({"data": {"img_size": 224}})
    cfg_160 = OmegaConf.create({"data": {"img_size": 160}})
    
    # Get transforms for both configs
    train_tfm_224, val_tfm_224 = get_transforms(cfg_224)
    train_tfm_160, val_tfm_160 = get_transforms(cfg_160)
    
    # Check that the CenterCrop size matches the config
    assert val_tfm_224.transforms[1].size == 224, "Validation transform should use img_size=224 for cropping"
    assert val_tfm_160.transforms[1].size == 160, "Validation transform should use img_size=160 for cropping"
    
    # Check that the RandomResizedCrop size matches the config
    assert train_tfm_224.transforms[0].size == (224, 224), "Train transform should use img_size=224 for cropping"
    assert train_tfm_160.transforms[0].size == (160, 160), "Train transform should use img_size=160 for cropping"