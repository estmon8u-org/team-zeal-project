# tests/test_dataset.py

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