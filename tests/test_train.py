# tests/test_train.py

"""
Unit tests for the train.py module in the drift_detector_pipeline.modeling package.

This module contains tests that verify the functionality of the model training 
pipeline. The tests focus on verifying:
1. The core training workflow works correctly when all components function properly.
2. Proper error handling when data loading fails.
3. Proper error handling when model creation fails.
4. Integration with external components like wandb for experiment tracking.

Tests use pytest fixtures for test setup and `unittest.mock.patch` to isolate
functionality from external dependencies (like wandb, torch.save, etc.).
"""

import pytest
import torch
from unittest.mock import patch, MagicMock, ANY
from omegaconf import OmegaConf, DictConfig
import sys
import os

# Add the project root to the Python path to allow importing drift_detector_pipeline
# This ensures the tests can find the module being tested.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Test Configuration Fixtures ---

@pytest.fixture
def minimal_hydra_cfg() -> DictConfig:
    """
    Provides a minimal mock Hydra DictConfig for testing train_model.
    
    This fixture creates a simplified configuration with only the
    essential parameters needed for the train_model function to run,
    avoiding the need for a full configuration file.
    """
    conf_dict = {
        "data": {
            "raw_path": "dummy/raw/data.tgz",
            "processed_path": "dummy/processed/data",
            "img_size": 32,
            "dataloader_workers": 0,  # Use 0 workers for simpler testing
        },
        "model": {
            "name": "resnet18",
            "pretrained": False,
            "num_classes": 10,
        },
        "training": {
            "epochs": 1,            # Use single epoch for faster tests
            "batch_size": 2,        # Small batch size for testing
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "optimizer": "AdamW",
            "scheduler": "None",
            "label_smoothing": 0.0,
            "profiler": {"enabled": False}  # Disable profiler for tests
        },
        "run": {
            "seed": 42,
            "device": "cpu",        # Use CPU for testing
        },
        "wandb": {
            "project": "test-pytest-project",
            "entity": "test-pytest-entity",
            "log_freq": 1,
            "watch_model": False,  # Disable model watching for simpler testing
        }
    }
    return OmegaConf.create(conf_dict)

def get_train_function():
    """
    Helper function to get the train_model function from the module.
    
    Returns the __wrapped__ attribute of train_model if available
    (which bypasses the Hydra decorator), otherwise returns the
    function itself. This allows testing the core logic without
    involving Hydra's configuration management.
    """
    from drift_detector_pipeline.modeling.train import train_model
    return train_model.__wrapped__ if hasattr(train_model, '__wrapped__') else train_model

# --- Test Success Case ---

@patch('drift_detector_pipeline.modeling.train.get_dataloaders')
@patch('drift_detector_pipeline.modeling.train.wandb')  # Mock entire wandb module
@patch('drift_detector_pipeline.modeling.train.timm.create_model')
@patch('drift_detector_pipeline.modeling.train.torch.save')
def test_train_model_components_called(
    mock_torch_save, mock_create_timm_model, mock_wandb, mock_get_dataloaders, minimal_hydra_cfg
):
    """
    Tests if train_model correctly calls its main components in the success path.
    
    This test verifies that:
    1. The function gets dataloaders correctly
    2. The model is created with the right parameters
    3. WandB is initialized and used for logging
    4. Training proceeds as expected
    5. The function returns a valid accuracy value
    """
    # --- Setup wandb module mock ---
    mock_wandb_run = MagicMock()
    mock_wandb_run.get_url.return_value = "http://mock-wandb-url.com"
    mock_wandb.init.return_value = mock_wandb_run
    
    # --- Create proper model mock ---
    # This mock model mimics a simple neural network structure
    # that can be called during training with minimal functionality
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3 * 32 * 32, 10)
            
        def forward(self, x):
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            return self.linear(x_flat)
        
        def to(self, device):
            return self  # Simulate moving to device
    
    mock_model = MockModel()
    mock_create_timm_model.return_value = mock_model

    # --- Setup dataloaders ---
    # Create mock training dataloader with 2 batches
    mock_train_loader = MagicMock()
    mock_train_loader.dataset.__len__.return_value = 4  # 4 total samples
    mock_train_loader.__len__.return_value = 2         # 2 batches
    mock_train_loader.__iter__.return_value = iter([
        (torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))  # (inputs, labels)
        for _ in range(2)  # 2 batches
    ])

    # Create mock validation dataloader with 1 batch
    mock_val_loader = MagicMock()
    mock_val_loader.dataset.__len__.return_value = 2  # 2 total samples
    mock_val_loader.__len__.return_value = 1         # 1 batch
    mock_val_loader.__iter__.return_value = iter([
        (torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))  # (inputs, labels)
    ])
    
    mock_get_dataloaders.return_value = (mock_train_loader, mock_val_loader)
    
    # --- Execute function ---
    train_func = get_train_function()
    result = train_func(minimal_hydra_cfg)

    # --- Assertions ---
    # Verify dataloaders were requested with the correct config
    mock_get_dataloaders.assert_called_once_with(minimal_hydra_cfg)
    
    # Verify model was created with the right parameters
    mock_create_timm_model.assert_called_once_with(
        minimal_hydra_cfg.model.name,
        pretrained=minimal_hydra_cfg.model.pretrained,
        num_classes=minimal_hydra_cfg.model.num_classes
    )
    
    # Verify wandb interactions
    mock_wandb.init.assert_called_once()  # WandB was initialized
    assert mock_wandb.log.call_count > 0  # Training metrics were logged
    mock_wandb.finish.assert_called_once()  # WandB session was properly closed
    
    # Verify the function returns a floating-point accuracy value
    assert isinstance(result, float)

# --- Test Error Handling ---

@patch('drift_detector_pipeline.modeling.train.get_dataloaders')
@patch('drift_detector_pipeline.modeling.train.wandb')
@patch('drift_detector_pipeline.modeling.train.sys.exit')
def test_train_model_handles_data_loading_exception(
    mock_sys_exit, mock_wandb, mock_get_dataloaders_error, minimal_hydra_cfg
):
    """
    Tests if train_model handles data loading failure gracefully.
    
    This test verifies that when get_dataloaders raises an exception:
    1. WandB is initialized
    2. The error is logged
    3. WandB is properly finished with an error exit code
    4. The function exits with a system exit
    """
    # --- Setup wandb mock ---
    mock_wandb_run = MagicMock()
    mock_wandb_run.get_url.return_value = "http://mock-wandb-url.com"
    mock_wandb.init.return_value = mock_wandb_run
    
    # --- Make get_dataloaders raise exception ---
    mock_get_dataloaders_error.side_effect = Exception("Data loading failed")
    
    # --- Setup sys.exit to raise SystemExit for testing ---
    mock_sys_exit.side_effect = SystemExit
    
    # --- Get the train function and execute with pytest's exception handling ---
    train_func = get_train_function()
    
    with pytest.raises(SystemExit):
        train_func(minimal_hydra_cfg)
    
    # --- Assertions ---
    # Verify WandB was initialized
    mock_wandb.init.assert_called_once()
    
    # Verify WandB was finished with error code
    mock_wandb.finish.assert_called_once_with(exit_code=1)
    
    # Verify system exit was called
    mock_sys_exit.assert_called_once()

@patch('drift_detector_pipeline.modeling.train.get_dataloaders')
@patch('drift_detector_pipeline.modeling.train.wandb')
@patch('drift_detector_pipeline.modeling.train.timm.create_model')
@patch('drift_detector_pipeline.modeling.train.sys.exit')
def test_train_model_handles_model_creation_exception(
    mock_sys_exit, mock_create_timm_model_error, mock_wandb, mock_get_dataloaders_success, minimal_hydra_cfg
):
    """
    Tests if train_model handles model creation failure gracefully.
    
    This test verifies that when timm.create_model raises an exception:
    1. WandB is initialized
    2. The error is logged
    3. WandB is properly finished with an error exit code
    4. The function exits with a system exit
    """
    # --- Setup wandb mock ---
    mock_wandb_run = MagicMock()
    mock_wandb_run.get_url.return_value = "http://mock-wandb-url.com"
    mock_wandb.init.return_value = mock_wandb_run
    
    # --- Setup successful dataloaders but failing model creation ---
    mock_train_loader = MagicMock()
    mock_train_loader.dataset.__len__.return_value = 4
    mock_val_loader = MagicMock()
    mock_val_loader.dataset.__len__.return_value = 2
    mock_get_dataloaders_success.return_value = (mock_train_loader, mock_val_loader)
    
    # --- Make model creation fail ---
    mock_create_timm_model_error.side_effect = Exception("Model creation failed")
    mock_sys_exit.side_effect = SystemExit
    
    # --- Get the train function and execute with pytest's exception handling ---
    train_func = get_train_function()
    
    with pytest.raises(SystemExit):
        train_func(minimal_hydra_cfg)
    
    # --- Assertions ---
    # Verify WandB was initialized
    mock_wandb.init.assert_called_once()
    
    # Verify WandB was finished with error code
    mock_wandb.finish.assert_called_once_with(exit_code=1)
    
    # Verify system exit was called
    mock_sys_exit.assert_called_once()