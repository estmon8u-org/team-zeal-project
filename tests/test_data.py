# tests/test_data.py
"""
Unit tests for the dataset.py module in the drift_detector_pipeline package.

This module focuses on testing:
1.  `extract_data`: Control flow for data presence, DVC interaction, and extraction.
2.  `get_transforms`: Correctness of transformation pipelines for training and validation.
3.  `get_dataloaders`: Correct interaction with `extract_data`, `get_transforms`,
    and `torchvision.datasets.ImageFolder`, and proper DataLoader instantiation.

Tests use pytest fixtures and unittest.mock to isolate functionality and
avoid actual file system operations or external calls during testing.
"""

import os
import subprocess  # To mock subprocess.CalledProcessError
import sys
import tarfile  # To mock tarfile.ReadError
from unittest.mock import MagicMock, patch

from omegaconf import DictConfig, OmegaConf  # Added DictConfig for type hint
import pytest
import torch  # For DataLoader and tensor creation in potential future tests
from torchvision import transforms

# Add the project root to the Python path to allow importing drift_detector_pipeline
# Assumes tests/ is at the project root sibling to drift_detector_pipeline/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from drift_detector_pipeline.dataset import (  # noqa: E402
    extract_data,
    get_dataloaders,
    get_transforms,
)

# --- Fixtures ---


@pytest.fixture
def mock_base_cfg() -> DictConfig:
    """Creates a base mock OmegaConf DictConfig object for testing."""
    conf_dict = {
        "data": {
            "raw_path": "data/raw/mock_dataset.tgz",
            "processed_path": "data/processed/mock_dataset_extracted",
            "img_size": 224,
            "dataloader_workers": 0,  # Keep workers at 0 for easier testing
        },
        "training": {
            "batch_size": 2,
        },
        "run": {
            "device": "cpu",
        },
    }
    return OmegaConf.create(conf_dict)


# --- Tests for get_transforms ---


def test_get_transforms_returns_tuple_of_compose(mock_base_cfg):
    """Verify get_transforms returns a tuple of two torchvision.transforms.Compose objects."""
    train_tfm, val_tfm = get_transforms(mock_base_cfg)
    assert isinstance(train_tfm, transforms.Compose), "Train transform should be a Compose object."
    assert isinstance(val_tfm, transforms.Compose), (
        "Validation transform should be a Compose object."
    )


def test_get_transforms_validation_structure(mock_base_cfg):
    """Check the type and order of initial transforms in the validation pipeline."""
    _, val_tfm = get_transforms(mock_base_cfg)
    assert len(val_tfm.transforms) >= 3, "Validation transform should have at least 3 steps."
    assert isinstance(val_tfm.transforms[0], transforms.Resize)
    assert val_tfm.transforms[0].size == 256, "Resize should be to 256 for validation pre-crop."
    assert isinstance(val_tfm.transforms[1], transforms.CenterCrop)
    assert val_tfm.transforms[1].size == (
        mock_base_cfg.data.img_size,
        mock_base_cfg.data.img_size,
    ), "CenterCrop size mismatch."
    assert isinstance(val_tfm.transforms[2], transforms.ToTensor)
    assert isinstance(val_tfm.transforms[3], transforms.Normalize), (
        "Normalization should be the last default step."
    )


def test_get_transforms_train_structure(mock_base_cfg):
    """Check the type and order of initial transforms in the training pipeline."""
    train_tfm, _ = get_transforms(mock_base_cfg)
    assert len(train_tfm.transforms) >= 3, "Train transform should have at least 3 steps."
    assert isinstance(train_tfm.transforms[0], transforms.RandomResizedCrop)
    assert train_tfm.transforms[0].size == (
        mock_base_cfg.data.img_size,
        mock_base_cfg.data.img_size,
    ), "RandomResizedCrop size mismatch."
    assert isinstance(train_tfm.transforms[1], transforms.RandomHorizontalFlip)
    assert isinstance(train_tfm.transforms[2], transforms.ToTensor)
    assert isinstance(train_tfm.transforms[3], transforms.Normalize), (
        "Normalization should be the last default step."
    )


def test_get_transforms_respects_img_size_config():
    """Test that transforms correctly use the img_size from configuration."""
    cfg_160 = OmegaConf.create({"data": {"img_size": 160}})
    _, val_tfm_160 = get_transforms(cfg_160)
    assert val_tfm_160.transforms[1].size == (160, 160), (
        "Validation CenterCrop should use img_size=160."
    )

    train_tfm_160, _ = get_transforms(cfg_160)
    assert train_tfm_160.transforms[0].size == (160, 160), (
        "Train RandomResizedCrop should use img_size=160."
    )


# --- Tests for extract_data ---


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.listdir")
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.os.makedirs")
@patch("drift_detector_pipeline.dataset.subprocess.run")  # For DVC
@patch("drift_detector_pipeline.dataset.log")  # Suppress logs during test
def test_extract_data_skips_if_processed_exists_and_not_empty(
    mock_log,
    mock_subprocess_run,
    mock_makedirs,
    mock_tar_open,
    mock_listdir,
    mock_os_exists,
    mock_base_cfg,
):
    """Test extraction is skipped if processed directory exists and is not empty."""
    # os.path.exists for raw_path, then for processed_path
    # os.listdir for processed_path
    mock_os_exists.side_effect = [True, True]  # raw exists, processed_path exists
    mock_listdir.return_value = ["train", "val"]  # processed_path is not empty

    extract_data(mock_base_cfg)

    mock_makedirs.assert_called_once_with(
        os.path.abspath(os.path.dirname(mock_base_cfg.data.processed_path)), exist_ok=True
    )
    mock_subprocess_run.assert_not_called()  # DVC pull should not be called
    mock_tar_open.assert_not_called()  # Tar extraction should not be called
    mock_log.info.assert_any_call(
        f"Processed data directory {os.path.abspath(mock_base_cfg.data.processed_path)} "
        "already exists and is not empty. Skipping extraction."
    )


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.listdir")  # For processed_path check
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.os.makedirs")
@patch("drift_detector_pipeline.dataset.subprocess.run")
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_extracts_if_processed_exists_but_empty(
    mock_log,
    mock_subprocess_run,
    mock_makedirs,
    mock_tar_open,
    mock_listdir,
    mock_os_exists,
    mock_base_cfg,
):
    """Test extraction proceeds if processed directory exists but is empty."""
    # os.path.exists: raw_path, dvc_meta_file (not called), processed_path, processed_path (after extract)
    # os.listdir: processed_path (before extract)
    mock_os_exists.side_effect = [
        True,
        True,
        True,
    ]  # raw exists, processed_path exists (before extract), processed_path exists (after extract by tar)
    mock_listdir.return_value = []  # processed_path is empty

    mock_tar_context = MagicMock()
    mock_tar_open.return_value.__enter__.return_value = mock_tar_context

    extract_data(mock_base_cfg)

    mock_makedirs.assert_called_once()
    mock_subprocess_run.assert_not_called()  # DVC pull
    mock_tar_open.assert_called_once_with(os.path.abspath(mock_base_cfg.data.raw_path), "r:gz")
    mock_tar_context.extractall.assert_called_once_with(
        path=os.path.abspath(os.path.dirname(mock_base_cfg.data.processed_path))
    )
    mock_log.info.assert_any_call(
        f"Extracting {os.path.abspath(mock_base_cfg.data.raw_path)} to {os.path.abspath(os.path.dirname(mock_base_cfg.data.processed_path))}..."
    )


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.listdir")  # For processed_path check
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.os.makedirs")
@patch("drift_detector_pipeline.dataset.subprocess.run")
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_dvc_pull_if_raw_missing_and_extracts(
    mock_log,
    mock_subprocess_run,
    mock_makedirs,
    mock_tar_open,
    mock_listdir,
    mock_os_exists,
    mock_base_cfg,
):
    """Test DVC pull is called if raw data is missing, then extracts."""
    dvc_meta_file = mock_base_cfg.data.raw_path + ".dvc"
    # os.path.exists:
    # 1. raw_path (False)
    # 2. dvc_meta_file (True)
    # 3. raw_path (True, after simulated pull)
    # 4. processed_path (False, before extract)
    # 5. processed_path (True, after simulated extract)
    mock_os_exists.side_effect = [False, True, True, False, True]
    mock_listdir.return_value = []  # Not relevant here as processed_path initially doesn't exist

    mock_subprocess_run.return_value = MagicMock(
        stdout="Pulled.", stderr="", returncode=0
    )  # Simulate successful DVC pull
    mock_tar_context = MagicMock()
    mock_tar_open.return_value.__enter__.return_value = mock_tar_context

    extract_data(mock_base_cfg)

    mock_makedirs.assert_called_once()
    mock_subprocess_run.assert_called_once_with(
        ["dvc", "pull", dvc_meta_file], check=True, capture_output=True, text=True, cwd=os.getcwd()
    )
    mock_tar_open.assert_called_once_with(os.path.abspath(mock_base_cfg.data.raw_path), "r:gz")
    mock_tar_context.extractall.assert_called_once()
    mock_log.info.assert_any_call(
        f"DVC pull successful. Raw data available at {os.path.abspath(mock_base_cfg.data.raw_path)}."
    )


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.subprocess.run")
@patch("drift_detector_pipeline.dataset.sys.exit")  # Mock sys.exit to prevent test termination
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_exits_if_dvc_metafile_missing(
    mock_log, mock_sys_exit, mock_subprocess_run, mock_os_exists, mock_base_cfg
):
    """Test sys.exit is called if DVC metafile is missing when raw data is absent."""
    # os.path.exists:
    # 1. raw_path (False)
    # 2. dvc_meta_file (False)
    mock_os_exists.side_effect = [False, False]

    extract_data(mock_base_cfg)

    mock_subprocess_run.assert_not_called()
    mock_sys_exit.assert_called_once_with(1)
    mock_log.error.assert_any_call(
        f"DVC metafile {mock_base_cfg.data.raw_path + '.dvc'} not found. "
        "Cannot pull data. Ensure the .dvc file is present and committed."
    )


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.subprocess.run")
@patch("drift_detector_pipeline.dataset.sys.exit")
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_exits_if_dvc_pull_fails(
    mock_log, mock_sys_exit, mock_subprocess_run, mock_os_exists, mock_base_cfg
):
    """Test sys.exit is called if DVC pull command fails."""
    dvc_meta_file = mock_base_cfg.data.raw_path + ".dvc"
    # os.path.exists:
    # 1. raw_path (False)
    # 2. dvc_meta_file (True)
    mock_os_exists.side_effect = [False, True]
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["dvc", "pull", dvc_meta_file],
        output="Error output",
        stderr="Error stderr",
    )

    extract_data(mock_base_cfg)

    mock_sys_exit.assert_called_once_with(1)
    mock_log.error.assert_any_call(
        f"DVC pull command ('dvc pull {dvc_meta_file}') failed with exit code 1.\n"
        f"Stdout: Error output\nStderr: Error stderr"
        "Please ensure DVC is initialized and the remote is configured."
    )


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.listdir")
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.sys.exit")
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_exits_on_tar_read_error(
    mock_log, mock_sys_exit, mock_tar_open, mock_listdir, mock_os_exists, mock_base_cfg
):
    """Test sys.exit is called if tarfile.open raises a ReadError."""
    # os.path.exists: raw_path (True), processed_path (False)
    # os.listdir: processed_path (not called as it doesn't exist)
    mock_os_exists.side_effect = [True, False]
    mock_listdir.return_value = []  # Not strictly needed but good for consistency

    mock_tar_open.side_effect = tarfile.ReadError("Simulated tar read error")

    extract_data(mock_base_cfg)

    mock_sys_exit.assert_called_once_with(1)
    mock_log.error.assert_any_call(
        f"Failed to read tar file {os.path.abspath(mock_base_cfg.data.raw_path)}. It might be corrupted."
    )


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.listdir")
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.sys.exit")
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_exits_if_extracted_dir_not_found(
    mock_log, mock_sys_exit, mock_tar_open, mock_listdir, mock_os_exists, mock_base_cfg
):
    """Test sys.exit if tar extraction finishes but target dir is still missing."""
    # os.path.exists:
    # 1. raw_path (True)
    # 2. processed_path (False, before extract)
    # 3. processed_path (False, after extract - this is the error condition)
    mock_os_exists.side_effect = [True, False, False]
    mock_listdir.return_value = []

    mock_tar_context = MagicMock()
    mock_tar_open.return_value.__enter__.return_value = mock_tar_context

    extract_data(mock_base_cfg)

    mock_tar_context.extractall.assert_called_once()  # Extraction was attempted
    mock_sys_exit.assert_called_once_with(1)
    mock_log.error.assert_any_call(
        f"Extraction completed, but the expected directory {os.path.abspath(mock_base_cfg.data.processed_path)} was not found. "
        f"The archive might have an unexpected internal structure or name."
    )


# --- Tests for get_dataloaders ---


@patch("drift_detector_pipeline.dataset.extract_data")  # Mock the call to extract_data
@patch("drift_detector_pipeline.dataset.get_transforms")
@patch("drift_detector_pipeline.dataset.datasets.ImageFolder")
@patch("drift_detector_pipeline.dataset.DataLoader")  # Mock torch.utils.data.DataLoader
@patch("drift_detector_pipeline.dataset.os.path.isdir")
@patch("drift_detector_pipeline.dataset.log")
def test_get_dataloaders_successful_creation(
    mock_log,
    mock_os_isdir,
    mock_torch_dataloader,
    mock_imagefolder,
    mock_get_transforms,
    mock_extract_data,
    mock_base_cfg,
):
    """Test successful creation of DataLoaders."""
    mock_os_isdir.return_value = True  # Simulate train/val dirs exist

    # Mock get_transforms to return mock transform objects
    mock_train_tfm, mock_val_tfm = (
        MagicMock(spec=transforms.Compose),
        MagicMock(spec=transforms.Compose),
    )
    mock_get_transforms.return_value = (mock_train_tfm, mock_val_tfm)

    # Mock ImageFolder to return mock dataset objects with a __len__
    mock_train_dataset = MagicMock(spec=torch.utils.data.Dataset)
    mock_train_dataset.__len__.return_value = 100
    mock_val_dataset = MagicMock(spec=torch.utils.data.Dataset)
    mock_val_dataset.__len__.return_value = 20
    mock_imagefolder.side_effect = [mock_train_dataset, mock_val_dataset]

    # Mock DataLoader to return mock DataLoader objects
    mock_train_loader = MagicMock(spec=torch.utils.data.DataLoader)
    mock_val_loader = MagicMock(spec=torch.utils.data.DataLoader)
    mock_torch_dataloader.side_effect = [mock_train_loader, mock_val_loader]

    train_loader, val_loader = get_dataloaders(mock_base_cfg)

    mock_extract_data.assert_called_once_with(mock_base_cfg)
    mock_get_transforms.assert_called_once_with(mock_base_cfg)

    expected_train_dir = os.path.join(os.path.abspath(mock_base_cfg.data.processed_path), "train")
    expected_val_dir = os.path.join(os.path.abspath(mock_base_cfg.data.processed_path), "val")

    mock_imagefolder.assert_any_call(root=expected_train_dir, transform=mock_train_tfm)
    mock_imagefolder.assert_any_call(root=expected_val_dir, transform=mock_val_tfm)

    # Check DataLoader instantiation calls
    mock_torch_dataloader.assert_any_call(
        dataset=mock_train_dataset,
        batch_size=mock_base_cfg.training.batch_size,
        shuffle=True,
        num_workers=mock_base_cfg.data.dataloader_workers,
        pin_memory=False,  # device is CPU in mock_base_cfg
        persistent_workers=False,
    )
    mock_torch_dataloader.assert_any_call(
        dataset=mock_val_dataset,
        batch_size=mock_base_cfg.training.batch_size,
        shuffle=False,
        num_workers=mock_base_cfg.data.dataloader_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    assert train_loader is mock_train_loader
    assert val_loader is mock_val_loader
    mock_log.info.assert_any_call(
        f"DataLoaders created. Batch size: {mock_base_cfg.training.batch_size}, "
        f"Num workers: {mock_base_cfg.data.dataloader_workers}, Pin memory: False."
    )


@patch("drift_detector_pipeline.dataset.extract_data")
@patch("drift_detector_pipeline.dataset.os.path.isdir")
@patch("drift_detector_pipeline.dataset.sys.exit")
@patch("drift_detector_pipeline.dataset.log")
def test_get_dataloaders_exits_if_train_dir_missing(
    mock_log, mock_sys_exit, mock_os_isdir, mock_extract_data, mock_base_cfg
):
    """Test sys.exit if train directory is missing after extract_data call."""
    mock_os_isdir.side_effect = [False, True]  # Train dir missing, Val dir exists

    get_dataloaders(mock_base_cfg)

    mock_extract_data.assert_called_once_with(mock_base_cfg)
    mock_sys_exit.assert_called_once_with(1)
    expected_train_dir = os.path.join(os.path.abspath(mock_base_cfg.data.processed_path), "train")
    mock_log.error.assert_any_call(
        f"Training data directory not found at {expected_train_dir}. "
        "Ensure 'extract_data' (or 'make process_data') completed successfully and "
        "the archive contains a 'train' subdirectory at the expected location."
    )


# Consider adding a test for get_dataloaders if val_dir is missing as well.
