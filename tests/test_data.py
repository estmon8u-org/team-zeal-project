# tests/test_dataset.py

"""
Unit tests for the dataset.py module in the drift_detector_pipeline package.

This module contains tests that verify the functionality of data extraction
and transformation operations used in the drift detector pipeline. The tests
focus on verifying:
1. Image transformation pipelines for training and validation (`get_transforms`).
2. Data extraction logic for handling the Imagenette dataset (`extract_data`),
   including DVC interaction, file presence checks, and archive extraction.
3. Control flow for various scenarios within `extract_data`.
4. DataLoader creation logic (`get_dataloaders`), ensuring correct interaction
   with helper functions and PyTorch components.

Tests use pytest fixtures for test setup and `unittest.mock.patch` to isolate
functionality from the file system and external processes (like DVC).
"""

import os
import subprocess  # For simulating subprocess-specific exceptions
import sys
import tarfile  # For simulating tarfile-specific exceptions
from unittest.mock import MagicMock, call, patch  # `call` for asserting multiple mock calls

import numpy as np  # For creating mock image data
from omegaconf import DictConfig, OmegaConf  # For creating and type-hinting mock configurations
from PIL import Image  # For creating mock image inputs for transforms
import pytest
import torch  # For tensor operations and DataLoader types
from torchvision import transforms  # For asserting transform types

# Add the project root to the Python path to allow importing drift_detector_pipeline
# This ensures that the tests can find the module being tested.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:  # Add only if not already present
    sys.path.insert(0, project_root)

# Import the functions to be tested from the dataset module
from drift_detector_pipeline.dataset import (  # noqa: E402
    extract_data,
    get_dataloaders,
    get_transforms,
)


@pytest.fixture
def mock_cfg() -> DictConfig:
    """
    Pytest fixture to create a mock OmegaConf DictConfig object for testing.

    This configuration simulates the structure expected by the dataset functions,
    providing necessary paths and parameters.
    """
    conf = OmegaConf.create(
        {
            "data": {
                "img_size": 224,
                "raw_path": "data/raw/mock_imagenette.tgz",  # Relative path from project root
                "processed_path": "data/processed/mock_imagenette_extracted",  # Relative
                "dataloader_workers": 0,  # Set to 0 for simpler testing of DataLoaders
            },
            "training": {  # Needed for get_dataloaders
                "batch_size": 4,
            },
            "run": {  # Needed for get_dataloaders (pin_memory)
                "device": "cpu",
            },
        }
    )
    return conf


# --- Tests for get_transforms ---
def test_get_transforms_returns_tuple(mock_cfg):
    """
    Verify that `get_transforms` returns a tuple containing two
    `torchvision.transforms.Compose` objects (one for training, one for validation).
    """
    train_tfm, val_tfm = get_transforms(mock_cfg)
    assert isinstance(train_tfm, transforms.Compose), "Train transform should be a Compose object."
    assert isinstance(val_tfm, transforms.Compose), (
        "Validation transform should be a Compose object."
    )


def test_get_transforms_validation_structure(mock_cfg):
    """
    Check the expected structure (type and order of initial transforms)
    of the validation transformation pipeline.
    """
    _, val_tfm = get_transforms(mock_cfg)
    # Expecting Resize, CenterCrop, ToTensor, Normalize
    assert len(val_tfm.transforms) >= 4, "Validation transform should have at least 4 steps."
    assert isinstance(val_tfm.transforms[0], transforms.Resize), "First val step should be Resize."
    assert val_tfm.transforms[0].size == 256, "Validation Resize should be to 256."
    assert isinstance(val_tfm.transforms[1], transforms.CenterCrop), (
        "Second val step should be CenterCrop."
    )
    assert val_tfm.transforms[1].size == (
        mock_cfg.data.img_size,
        mock_cfg.data.img_size,
    ), "Val CenterCrop size mismatch."
    assert isinstance(val_tfm.transforms[2], transforms.ToTensor), (
        "Third val step should be ToTensor."
    )
    assert isinstance(val_tfm.transforms[3], transforms.Normalize), (
        "Fourth val step should be Normalize."
    )


def test_get_transforms_train_structure(mock_cfg):
    """
    Check the expected structure (type and order of initial transforms)
    of the training transformation pipeline.
    """
    train_tfm, _ = get_transforms(mock_cfg)
    # Expecting RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize
    assert len(train_tfm.transforms) >= 4, "Train transform should have at least 4 steps."
    assert isinstance(train_tfm.transforms[0], transforms.RandomResizedCrop), (
        "First train step should be RandomResizedCrop."
    )
    assert train_tfm.transforms[0].size == (
        mock_cfg.data.img_size,
        mock_cfg.data.img_size,
    ), "Train RandomResizedCrop size mismatch."
    assert isinstance(train_tfm.transforms[1], transforms.RandomHorizontalFlip), (
        "Second train step should be RandomHorizontalFlip."
    )
    assert isinstance(train_tfm.transforms[2], transforms.ToTensor), (
        "Third train step should be ToTensor."
    )
    assert isinstance(train_tfm.transforms[3], transforms.Normalize), (
        "Fourth train step should be Normalize."
    )


# --- Tests for extract_data control flow ---


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.listdir")
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.os.makedirs")
@patch("drift_detector_pipeline.dataset.subprocess.run")  # For mocking DVC calls
@patch("drift_detector_pipeline.dataset.log")  # To suppress actual logging during tests
def test_extract_data_skips_if_processed_exists_and_not_empty(
    mock_log,
    mock_subprocess_run,
    mock_makedirs,
    mock_tar_open,
    mock_listdir,
    mock_os_exists,
    mock_cfg,
):
    """
    Test that `extract_data` skips extraction if the target processed directory
    already exists and is not empty.
    """
    # Define paths based on the mock configuration for clarity in the side_effect
    raw_path_abs = os.path.abspath(mock_cfg.data.raw_path)
    processed_path_abs = os.path.abspath(mock_cfg.data.processed_path)
    extraction_parent_abs = os.path.abspath(os.path.dirname(mock_cfg.data.processed_path))

    # Configure mock for os.path.exists
    def os_exists_side_effect(path_arg):
        # print(f"DEBUG (skips_if_processed): os.path.exists called with '{path_arg}'") # Keep for debugging
        if path_arg == raw_path_abs:
            return True  # Simulate raw_path exists (though not strictly needed for this test path)
        elif path_arg == processed_path_abs:
            return True  # Simulate processed_path exists
        elif path_arg == extraction_parent_abs:  # For os.makedirs check
            return False  # Simulate parent of processed_path does not exist to trigger makedirs
        return False  # Default for any other unexpected paths

    mock_os_exists.side_effect = os_exists_side_effect
    mock_listdir.return_value = ["train", "val"]  # Simulate processed_path is not empty

    # Call the function under test
    extract_data(mock_cfg)

    # Assertions
    mock_makedirs.assert_called_once_with(extraction_parent_abs, exist_ok=True)
    mock_subprocess_run.assert_not_called()  # DVC pull should not have been called
    mock_tar_open.assert_not_called()  # Tar extraction should not have been called
    mock_log.info.assert_any_call(
        f"Processed data directory {processed_path_abs} "
        "already exists and is not empty. Skipping extraction."
    )


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.listdir")
@patch("drift_detector_pipeline.dataset.subprocess.run")
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.os.makedirs")
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_pulls_if_raw_missing(
    mock_log,
    mock_makedirs,
    mock_tar_open,
    mock_subprocess_run,
    mock_listdir,
    mock_os_exists,
    mock_cfg,
):
    """
    Test that `extract_data` attempts to pull data using DVC if the raw
    archive is missing, and then proceeds with extraction.
    """
    # Define paths based on config for clarity
    dvc_meta_file_rel = (
        mock_cfg.data.raw_path + ".dvc"
    )  # Relative path as used in dataset.py for dvc_meta_file
    raw_path_abs = os.path.abspath(mock_cfg.data.raw_path)
    processed_path_abs = os.path.abspath(mock_cfg.data.processed_path)
    extraction_parent_abs = os.path.abspath(os.path.dirname(mock_cfg.data.processed_path))

    # Counter for calls to os.path.exists specifically for raw_path_abs
    raw_path_abs_call_count = 0

    # Configure mock for os.path.exists to simulate different states
    def os_exists_side_effect(path_arg):
        nonlocal raw_path_abs_call_count
        # print(f"DEBUG (pulls_if_raw_missing): os.path.exists called with '{path_arg}' (Overall call: {mock_os_exists.call_count})")
        if path_arg == raw_path_abs:
            raw_path_abs_call_count += 1
            if raw_path_abs_call_count == 1:  # First check for raw_path_abs (CALL 1 in dataset.py)
                return False  # Simulate raw file is missing initially
            else:  # Second check for raw_path_abs (CALL 3, after mocked DVC pull)
                return True  # Simulate raw file exists after pull
        elif (
            path_arg == dvc_meta_file_rel
        ):  # CALL 2 for dvc_meta_file (dataset.py uses relative path)
            return True  # Simulate .dvc metafile exists
        elif path_arg == processed_path_abs:  # CALL 4 and CALL 5
            # Before extraction (CALL 4), it doesn't exist.
            # After extraction (CALL 5), it exists.
            return (
                mock_tar_open.call_count > 0
            )  # True if tar was "opened" (i.e., after extraction attempt)
        elif path_arg == extraction_parent_abs:  # For os.makedirs
            return False  # Assume parent directory needs to be created
        return False  # Default for other paths

    mock_os_exists.side_effect = os_exists_side_effect
    mock_listdir.return_value = []  # Simulate processed_path is initially empty or non-existent
    mock_subprocess_run.return_value = MagicMock(
        stdout="DVC pull successful.", stderr="", returncode=0
    )  # Simulate successful DVC pull

    # Mock the tarfile context manager for successful extraction
    mock_tar_context = MagicMock()
    mock_tar_open.return_value.__enter__.return_value = mock_tar_context

    # Call the function under test
    extract_data(mock_cfg)

    # Assertions
    mock_makedirs.assert_called_once_with(extraction_parent_abs, exist_ok=True)
    mock_subprocess_run.assert_called_once_with(
        ["dvc", "pull", dvc_meta_file_rel],  # DVC pull was called with the .dvc file
        check=True,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    mock_tar_open.assert_called_once_with(raw_path_abs, "r:gz")  # Tar extraction was called
    mock_tar_context.extractall.assert_called_once_with(
        path=extraction_parent_abs
    )  # Tar extracted to correct parent
    mock_log.info.assert_any_call(f"DVC pull successful. Raw data available at {raw_path_abs}.")
    mock_log.info.assert_any_call(f"Extraction complete. Data available in {processed_path_abs}")


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.listdir")
@patch("drift_detector_pipeline.dataset.subprocess.run")
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.os.makedirs")
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_extracts_if_raw_present_processed_missing(
    mock_log,
    mock_makedirs,
    mock_tar_open,
    mock_subprocess_run,
    mock_listdir,
    mock_os_exists,
    mock_cfg,
):
    """
    Test that `extract_data` proceeds with extraction if the raw archive exists
    but the processed data directory is missing.
    """
    raw_path_abs = os.path.abspath(mock_cfg.data.raw_path)
    processed_path_abs = os.path.abspath(mock_cfg.data.processed_path)
    extraction_parent_abs = os.path.abspath(os.path.dirname(mock_cfg.data.processed_path))

    # Configure mock for os.path.exists
    def os_exists_side_effect(path_arg):
        # print(f"DEBUG (extracts_if_raw): os.path.exists called with '{path_arg}'")
        if path_arg == raw_path_abs:
            return True  # Simulate raw file exists
        elif path_arg == processed_path_abs:
            # CALL 4 (before extraction): False
            # CALL 5 (after extraction): True
            return mock_tar_open.call_count > 0  # True if tar "opened" (extraction happened)
        elif path_arg == extraction_parent_abs:
            return False  # Assume parent needs creation
        return False

    mock_os_exists.side_effect = os_exists_side_effect
    mock_listdir.return_value = []  # Processed directory is empty or doesn't exist
    mock_tar_context = MagicMock()
    mock_tar_open.return_value.__enter__.return_value = mock_tar_context

    # Call the function under test
    extract_data(mock_cfg)

    # Assertions
    mock_makedirs.assert_called_once_with(extraction_parent_abs, exist_ok=True)
    mock_subprocess_run.assert_not_called()  # DVC pull should not be called
    mock_tar_open.assert_called_once_with(raw_path_abs, "r:gz")  # Tar extraction should be called
    mock_tar_context.extractall.assert_called_once_with(path=extraction_parent_abs)
    mock_log.info.assert_any_call(f"Extracting {raw_path_abs} to {extraction_parent_abs}...")
    mock_log.info.assert_any_call(f"Extraction complete. Data available in {processed_path_abs}")


# --- Tests for extract_data error handling ---


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.subprocess.run")
@patch("drift_detector_pipeline.dataset.sys.exit")  # Mock sys.exit to prevent test termination
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_exits_if_pull_fails(
    mock_log, mock_sys_exit, mock_subprocess_run, mock_os_exists, mock_cfg
):
    """
    Test that `extract_data` exits if the DVC pull command fails
    (e.g., due to network issues or misconfiguration).
    """
    dvc_meta_file_rel = mock_cfg.data.raw_path + ".dvc"
    raw_path_abs = os.path.abspath(mock_cfg.data.raw_path)
    extraction_parent_abs = os.path.abspath(os.path.dirname(mock_cfg.data.processed_path))

    raw_path_abs_call_count = 0

    # Configure mock for os.path.exists
    def os_exists_side_effect_pull_fail(path_arg):
        nonlocal raw_path_abs_call_count
        # print(f"DEBUG (pull_fail): os.path.exists called with '{path_arg}' (Overall call: {mock_os_exists.call_count})")
        if path_arg == raw_path_abs:
            raw_path_abs_call_count += 1
            # CALL 1 (before DVC pull) should be False.
            # CALL 3 (after DVC pull attempt, in `except subprocess.CalledProcessError`'s own check) should also be False.
            return False
        elif path_arg == dvc_meta_file_rel:  # CALL 2 (dataset.py uses relative for this)
            return True  # Simulate .dvc metafile exists to trigger pull attempt
        elif path_arg == extraction_parent_abs:  # For os.makedirs
            return False  # Assume parent dir for processed data needs creation
        return False  # Default for other paths

    mock_os_exists.side_effect = os_exists_side_effect_pull_fail

    # Simulate DVC pull failure by raising CalledProcessError
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["dvc", "pull", dvc_meta_file_rel],
        output="Simulated DVC Error Output",
        stderr="Simulated DVC Error Stderr",
    )
    mock_sys_exit.side_effect = SystemExit  # Make mock_sys_exit raise SystemExit

    # Expect SystemExit to be raised
    with pytest.raises(SystemExit):
        extract_data(mock_cfg)

    # Verify the correct error log message was generated
    expected_log = (
        f"DVC pull command ('dvc pull {dvc_meta_file_rel}') failed with exit code 1.\n"
        f"Stdout: Simulated DVC Error Output\nStderr: Simulated DVC Error Stderr"  # Match the mock error output
        "Please ensure DVC is initialized and the remote is configured."
    )
    # Check if any of the calls to mock_log.error match the expected log
    found_call = any(
        call_args[0][0] == expected_log for call_args in mock_log.error.call_args_list
    )
    assert found_call, (
        f"Expected log call not found.\nExpected:\n{expected_log}\nActual calls:\n{[c[0][0] for c in mock_log.error.call_args_list]}"
    )
    mock_sys_exit.assert_called_once()  # Verify sys.exit was indeed called


@patch("drift_detector_pipeline.dataset.os.path.exists")
@patch("drift_detector_pipeline.dataset.os.listdir")
@patch("drift_detector_pipeline.dataset.tarfile.open")
@patch("drift_detector_pipeline.dataset.sys.exit")
@patch("drift_detector_pipeline.dataset.log")
def test_extract_data_handles_tarfile_read_error(
    mock_log, mock_sys_exit, mock_tar_open, mock_listdir, mock_os_exists, mock_cfg
):
    """
    Test that `extract_data` handles errors during tar extraction (e.g., corrupted archive)
    and exits gracefully.
    """
    raw_path_abs = os.path.abspath(mock_cfg.data.raw_path)
    processed_path_abs = os.path.abspath(mock_cfg.data.processed_path)
    extraction_parent_abs = os.path.abspath(os.path.dirname(mock_cfg.data.processed_path))

    # Configure mock for os.path.exists
    def os_exists_side_effect_tar_fail(path_arg):
        # print(f"DEBUG (tar_fail): os.path.exists called with '{path_arg}' (Overall call: {mock_os_exists.call_count})")
        if path_arg == raw_path_abs:
            return True  # Simulate raw file exists
        elif path_arg == processed_path_abs:
            # CALL 4 (before extraction): False
            return False  # Simulate processed directory does not exist before extraction
        elif path_arg == extraction_parent_abs:
            return False  # Assume parent needs creation
        return False

    mock_os_exists.side_effect = os_exists_side_effect_tar_fail
    mock_listdir.return_value = []  # Processed directory is empty or doesn't exist
    mock_tar_open.side_effect = tarfile.ReadError("Simulated tar read error")  # Simulate tar error
    mock_sys_exit.side_effect = SystemExit  # Make mock_sys_exit raise SystemExit

    # Expect SystemExit to be raised
    with pytest.raises(SystemExit):
        extract_data(mock_cfg)

    # Verify the correct error log message
    expected_log = f"Failed to read tar file {raw_path_abs}. It might be corrupted."
    found_call = any(
        call_args[0][0] == expected_log for call_args in mock_log.error.call_args_list
    )
    assert found_call, (
        f"Expected log call not found.\nExpected:\n{expected_log}\nActual calls:\n{[c[0][0] for c in mock_log.error.call_args_list]}"
    )
    mock_sys_exit.assert_called_once()


# --- Integration tests for transform behavior ---


def test_transform_dimensions(mock_cfg):
    """
    Test that image transformations produce output tensors with the expected dimensions
    as specified in the configuration.
    """
    # Create a mock PIL Image as input for the transform pipeline
    mock_pil_img = Image.fromarray(np.uint8(np.random.rand(300, 400, 3) * 255))  # H, W, C format

    train_tfm, val_tfm = get_transforms(mock_cfg)

    # Apply transforms
    transformed_train = train_tfm(mock_pil_img)
    transformed_val = val_tfm(mock_pil_img)

    # Check dimensions (C, H, W after ToTensor)
    expected_size = mock_cfg.data.img_size
    assert transformed_train.shape == (
        3,
        expected_size,
        expected_size,
    ), "Train transform output dimension mismatch."
    assert transformed_val.shape == (
        3,
        expected_size,
        expected_size,
    ), "Validation transform output dimension mismatch."


def test_transform_normalization(mock_cfg):
    """
    Test that image normalization is applied correctly, resulting in values
    within a typical normalized range.
    """
    # Create a mock mid-gray PIL image
    img_array = np.ones((mock_cfg.data.img_size, mock_cfg.data.img_size, 3), dtype=np.uint8) * 127
    mock_pil_img_norm = Image.fromarray(img_array)

    train_tfm, val_tfm = get_transforms(mock_cfg)

    # Apply transforms
    transformed_train = train_tfm(mock_pil_img_norm)
    transformed_val = val_tfm(mock_pil_img_norm)

    # Create a reference tensor: what the image would be after ToTensor() but before Normalize()
    # ToTensor converts HWC [0,255] uint8 to CHW [0,1] float.
    ref_tensor_before_norm = transforms.ToTensor()(mock_pil_img_norm)

    # Assert that normalization changed the values from the [0,1] range of ToTensor
    assert not torch.allclose(transformed_val, ref_tensor_before_norm, atol=1e-5), (
        "Validation transform did not significantly alter image values after ToTensor (normalization likely failed or was trivial)."
    )
    # Assert values are within a plausible range for ImageNet normalization
    assert transformed_val.min() >= -3.0 and transformed_val.max() <= 3.0, (
        "Normalized validation values out of typical ImageNet range."
    )

    # Train transform includes random cropping, so direct comparison is harder,
    # but values should still be normalized.
    assert not torch.allclose(transformed_train, ref_tensor_before_norm, atol=1e-5), (
        "Train transform did not significantly alter image values after ToTensor (normalization likely failed or was trivial)."
    )
    assert transformed_train.min() >= -3.0 and transformed_train.max() <= 3.0, (
        "Normalized training values out of typical ImageNet range."
    )


# --- Additional tests for configuration handling ---


def test_transform_respects_img_size_config():
    """
    Test that `get_transforms` correctly uses the `img_size` parameter
    from the configuration for cropping and resizing operations.
    """
    cfg_224 = OmegaConf.create({"data": {"img_size": 224}})
    cfg_160 = OmegaConf.create({"data": {"img_size": 160}})

    train_tfm_224, val_tfm_224 = get_transforms(cfg_224)
    train_tfm_160, val_tfm_160 = get_transforms(cfg_160)

    # Check CenterCrop size in validation transforms
    assert val_tfm_224.transforms[1].size == (
        224,
        224,
    ), "Validation CenterCrop should use img_size=224."
    assert val_tfm_160.transforms[1].size == (
        160,
        160,
    ), "Validation CenterCrop should use img_size=160."

    # Check RandomResizedCrop size in training transforms
    assert train_tfm_224.transforms[0].size == (
        224,
        224,
    ), "Train RandomResizedCrop should use img_size=224."
    assert train_tfm_160.transforms[0].size == (
        160,
        160,
    ), "Train RandomResizedCrop should use img_size=160."


# --- Tests for get_dataloaders ---


@patch(
    "drift_detector_pipeline.dataset.extract_data"
)  # Mock the call to extract_data within get_dataloaders
@patch("drift_detector_pipeline.dataset.get_transforms")  # Mock get_transforms
@patch(
    "drift_detector_pipeline.dataset.datasets.ImageFolder"
)  # Mock torchvision.datasets.ImageFolder
@patch("drift_detector_pipeline.dataset.DataLoader")  # Mock torch.utils.data.DataLoader
@patch(
    "drift_detector_pipeline.dataset.os.path.isdir"
)  # Mock os.path.isdir for train/val directory checks
@patch("drift_detector_pipeline.dataset.log")  # Mock logger
def test_get_dataloaders_successful_creation(
    mock_log,
    mock_os_isdir,
    mock_torch_dataloader,
    mock_imagefolder,
    mock_get_transforms,
    mock_extract_data,
    mock_cfg,  # Use the fixture
):
    """
    Test the successful creation of training and validation DataLoaders,
    ensuring all helper functions and PyTorch components are called correctly.
    """
    # --- Setup Mocks ---
    mock_os_isdir.return_value = True  # Simulate that 'train' and 'val' directories exist

    # Mock get_transforms to return mock transform objects
    mock_train_tfm = MagicMock(spec=transforms.Compose)
    mock_val_tfm = MagicMock(spec=transforms.Compose)
    mock_get_transforms.return_value = (mock_train_tfm, mock_val_tfm)

    # Mock ImageFolder to return mock dataset objects with a __len__ method
    mock_train_dataset = MagicMock()  # Plain MagicMock, will attach __len__
    mock_train_dataset.__len__ = MagicMock(return_value=100)  # Dataset has 100 items
    mock_val_dataset = MagicMock()  # Plain MagicMock
    mock_val_dataset.__len__ = MagicMock(return_value=20)  # Dataset has 20 items
    mock_imagefolder.side_effect = [
        mock_train_dataset,
        mock_val_dataset,
    ]  # Return train then val dataset

    # Mock DataLoader to return distinct mock DataLoader objects
    mock_train_loader_obj = MagicMock(spec=torch.utils.data.DataLoader)
    mock_val_loader_obj = MagicMock(spec=torch.utils.data.DataLoader)
    mock_torch_dataloader.side_effect = [mock_train_loader_obj, mock_val_loader_obj]

    # --- Call Function Under Test ---
    train_loader, val_loader = get_dataloaders(mock_cfg)

    # --- Assertions ---
    # Check that extract_data and get_transforms were called with the config
    mock_extract_data.assert_called_once_with(mock_cfg)
    mock_get_transforms.assert_called_once_with(mock_cfg)

    # Check that ImageFolder was called correctly for train and val datasets
    expected_train_dir = os.path.join(os.path.abspath(mock_cfg.data.processed_path), "train")
    expected_val_dir = os.path.join(os.path.abspath(mock_cfg.data.processed_path), "val")
    mock_imagefolder.assert_has_calls(
        [
            call(root=expected_train_dir, transform=mock_train_tfm),
            call(root=expected_val_dir, transform=mock_val_tfm),
        ],
        any_order=False,
    )  # Ensure train is processed before val if side_effect order matters

    # Check that DataLoader was instantiated correctly for train and val
    # pin_memory is False because mock_cfg.run.device is "cpu"
    # persistent_workers is False because mock_cfg.data.dataloader_workers is 0
    mock_torch_dataloader.assert_has_calls(
        [
            call(
                dataset=mock_train_dataset,
                batch_size=mock_cfg.training.batch_size,
                shuffle=True,
                num_workers=mock_cfg.data.dataloader_workers,
                pin_memory=False,
                persistent_workers=False,
            ),
            call(
                dataset=mock_val_dataset,
                batch_size=mock_cfg.training.batch_size,
                shuffle=False,
                num_workers=mock_cfg.data.dataloader_workers,
                pin_memory=False,
                persistent_workers=False,
            ),
        ],
        any_order=False,
    )

    # Check that the returned objects are the mocked DataLoader objects
    assert train_loader is mock_train_loader_obj, "Returned train_loader is not the mocked object."
    assert val_loader is mock_val_loader_obj, "Returned val_loader is not the mocked object."

    # Check for expected log message
    mock_log.info.assert_any_call(
        f"DataLoaders created. Batch size: {mock_cfg.training.batch_size}, "
        f"Num workers: {mock_cfg.data.dataloader_workers}, Pin memory: False."
    )
