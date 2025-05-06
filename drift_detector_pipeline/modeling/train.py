# drift_detector_pipeline/modeling/train.py
"""
Main script for training an image classification model using PyTorch, timm,
Hydra for configuration, and WandB for experiment tracking.
"""

import logging
import os
import sys
import time  # For basic timing

import hydra
from omegaconf import DictConfig, OmegaConf
import timm
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import wandb  # Weights & Biases for experiment tracking

# Attempt to import from the package structure.
# This handles cases where the script might be run directly or as part of the package.
try:
    from drift_detector_pipeline.dataset import get_dataloaders
except ImportError:
    # If direct import fails, adjust sys.path to allow finding the package.
    # This assumes the script is in drift_detector_pipeline/modeling/
    # and the project root is two levels up.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from drift_detector_pipeline.dataset import get_dataloaders

# Setup module-level logger. Hydra will typically manage handlers and formatting.
log = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def train_model(cfg: DictConfig) -> float:
    """
    Trains an image classification model based on the provided Hydra configuration.

    Key steps:
    1. Sets up device (CPU/GPU) and random seeds for reproducibility.
    2. Initializes Weights & Biases (WandB) for experiment tracking.
    3. Loads data using `get_dataloaders` from the dataset module.
    4. Loads the model architecture (from `timm`) and moves it to the device.
    5. Sets up the loss function, optimizer, and learning rate scheduler.
    6. Executes the training and validation loops for the specified number of epochs.
    7. Saves the best model checkpoint (based on validation accuracy) to Hydra's output directory.
    8. Logs metrics to WandB throughout training.

    Args:
        cfg (DictConfig): Configuration object managed by Hydra. Typically loaded
                          from 'conf/config.yaml' and command-line overrides.

    Returns:
        float: The best validation accuracy achieved during training.
               Returns 0.0 if training fails or is interrupted early.

    Raises:
        SystemExit: If critical errors occur (e.g., data/model loading failure,
                    unsupported configuration).
    """
    script_start_time = time.time()
    log.info("--- Initializing Model Training Script ---")
    log.info("Full configuration:\n%s", OmegaConf.to_yaml(cfg))

    # --- 1. Runtime Setup (Device, Seeds) ---
    try:
        seed = cfg.run.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            # For full reproducibility, though potentially slower:
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        log.info(f"Random seed set to: {seed}")

        if cfg.run.device.lower() == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            if cfg.run.device.lower() == "cuda":
                log.warning("CUDA selected in config but not available. Falling back to CPU.")
            device = torch.device("cpu")
        log.info(f"Using device: {device.type.upper()}")

    except AttributeError as e:
        log.error(f"Configuration error: Missing key in 'run' section - {e}. Aborting.")
        sys.exit(1)
    except Exception as e:  # Catch any other unexpected error during setup
        log.error(f"Unexpected error during runtime setup: {e}. Aborting.")
        sys.exit(1)

    # --- 2. Initialize Experiment Tracking (WandB) ---
    wandb_run = None
    try:
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,  # Can be None for default entity
            config=wandb_config,
            name=f"{cfg.model.name.replace('/', '_')}-run-{time.strftime('%Y%m%d-%H%M%S')}",  # More descriptive run name
            job_type="train",
            reinit=True,  # Allows re-initialization, useful in notebooks or repeated runs
        )
        log.info(f"WandB initialized. Run URL: {wandb_run.get_url() if wandb_run else 'N/A'}")
    except AttributeError as e:
        log.error(f"Configuration error: Missing key in 'wandb' section - {e}. WandB disabled.")
    except Exception as e:
        log.warning(
            f"Failed to initialize WandB: {e}. Training will continue without WandB logging."
        )

    # --- 3. Data Loading ---
    try:
        log.info("Loading data...")
        # get_dataloaders now handles its own call to extract_data(cfg)
        train_loader, val_loader = get_dataloaders(cfg)
        log.info("Data loaded successfully.")
        if wandb_run:
            wandb.config.update(
                {  # Log actual dataset sizes
                    "num_train_samples": len(train_loader.dataset),
                    "num_val_samples": len(val_loader.dataset),
                    "num_train_batches": len(train_loader),
                    "num_val_batches": len(val_loader),
                }
            )
    except SystemExit:  # Catch SystemExit from get_dataloaders
        log.error("Data loading failed as get_dataloaders exited. Aborting training.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)
    except Exception:
        log.exception("Critical error during data loading. Aborting training.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)

    # --- 4. Model Setup ---
    try:
        log.info(f"Loading model: {cfg.model.name} (pretrained={cfg.model.pretrained})")
        model = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes,  # Adapts the final classifier layer
        )
        model.to(device)
        log.info(f"Model '{cfg.model.name}' loaded and moved to {device.type.upper()}.")

        if wandb_run:  # Log model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb.config.update(
                {"total_model_params": total_params, "trainable_model_params": trainable_params}
            )
            # Consider wandb.watch(model) for gradient/parameter tracking, but can be verbose
            # wandb.watch(model, log="all", log_freq=cfg.wandb.log_freq * 10) # Example: log every 10 wandb log steps

    except AttributeError as e:
        log.error(f"Configuration error: Missing key in 'model' section - {e}. Aborting.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)
    except Exception:  # Catch errors from timm.create_model or .to(device)
        log.exception(f"Critical error loading model '{cfg.model.name}'. Aborting.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)

    # --- 5. Loss Function, Optimizer, Scheduler ---
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
        log.info(
            f"Loss function: CrossEntropyLoss (label_smoothing={cfg.training.label_smoothing})"
        )

        optimizer_name = cfg.training.optimizer.lower()
        if optimizer_name == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
            )
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=cfg.training.learning_rate,
                momentum=cfg.training.get("momentum", 0.9),  # Get with default
                weight_decay=cfg.training.weight_decay,
            )
        else:
            log.error(
                f"Unsupported optimizer: '{cfg.training.optimizer}'. Supported: 'AdamW', 'SGD'."
            )
            if wandb_run:
                wandb.finish(exit_code=1)
            sys.exit(1)
        log.info(
            f"Optimizer: {cfg.training.optimizer} (lr={cfg.training.learning_rate}, wd={cfg.training.weight_decay})"
        )

        scheduler = None
        scheduler_name = cfg.training.scheduler.lower()
        if scheduler_name == "cosineannealinglr":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
        elif scheduler_name == "steplr":
            scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.training.get("scheduler_step_size", 10),  # Get with default
                gamma=cfg.training.get("scheduler_gamma", 0.1),  # Get with default
            )
        elif scheduler_name != "none" and scheduler_name is not None:  # Handle None gracefully
            log.warning(
                f"Unsupported scheduler: '{cfg.training.scheduler}'. Proceeding without scheduler."
            )

        if scheduler:
            log.info(f"Learning rate scheduler: {cfg.training.scheduler}")
        else:
            log.info("No learning rate scheduler configured or 'none' specified.")

    except AttributeError as e:
        log.error(f"Configuration error: Missing key in 'training' section - {e}. Aborting.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)
    except Exception as e:
        log.exception(f"Error setting up optimizer/scheduler: {e}. Aborting.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)

    # --- 6. Training Loop ---
    log.info(f"Starting training for {cfg.training.epochs} epochs...")
    best_val_accuracy = 0.0
    best_epoch = -1

    # Hydra's CWD is the output directory for this run
    model_save_path = "best_model.pth"  # Saved within Hydra's output dir
    final_model_save_path = "final_model.pth"

    for epoch in range(cfg.training.epochs):
        epoch_start_time = time.time()
        log.info(f"--- Epoch {epoch + 1}/{cfg.training.epochs} ---")

        # -- Training Phase --
        model.train()
        running_train_loss = 0.0
        train_batches_processed = 0

        for i, (inputs, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)  # Weighted by batch size
            train_batches_processed += inputs.size(0)

            if wandb_run and (i + 1) % cfg.wandb.log_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                avg_batch_loss = loss.item()  # Current batch loss, not accumulated average
                batch_duration = time.time() - batch_start_time
                wandb.log(
                    {
                        "epoch_frac": epoch + (i + 1) / len(train_loader),
                        "train/batch_loss": avg_batch_loss,
                        "train/learning_rate": current_lr,
                        "perf/batch_time_sec": batch_duration,
                    }
                )

        avg_epoch_train_loss = (
            running_train_loss / train_batches_processed if train_batches_processed > 0 else 0.0
        )
        log.info(f"Epoch {epoch + 1} Training: Avg Loss = {avg_epoch_train_loss:.4f}")
        if wandb_run:
            wandb.log({"epoch": epoch + 1, "train/epoch_loss": avg_epoch_train_loss})

        # -- Validation Phase --
        model.eval()
        running_val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0
        validation_start_time = time.time()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)  # Weighted by batch size
                _, predicted_classes = torch.max(outputs, 1)
                correct_val_predictions += (predicted_classes == labels).sum().item()
                total_val_samples += labels.size(0)

        validation_duration = time.time() - validation_start_time
        avg_val_loss = running_val_loss / total_val_samples if total_val_samples > 0 else 0.0
        val_accuracy = (
            (100 * correct_val_predictions / total_val_samples) if total_val_samples > 0 else 0.0
        )

        log.info(
            f"Epoch {epoch + 1} Validation: Avg Loss = {avg_val_loss:.4f}, Accuracy = {val_accuracy:.2f}% ({correct_val_predictions}/{total_val_samples})"
        )
        log.info(f"Epoch {epoch + 1} Validation Time: {validation_duration:.2f} seconds")

        if wandb_run:
            wandb.log(
                {
                    "epoch": epoch + 1,  # Ensure epoch is logged with val metrics too
                    "val/epoch_loss": avg_val_loss,
                    "val/accuracy": val_accuracy,
                    "perf/validation_time_sec": validation_duration,
                }
            )

        # -- Checkpoint Saving --
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            try:
                torch.save(model.state_dict(), model_save_path)
                abs_save_path = os.path.abspath(model_save_path)  # Path within Hydra's run dir
                log.info(
                    f"[SUCCESS] New best model saved at Epoch {epoch + 1} with accuracy {best_val_accuracy:.2f}% to {abs_save_path}"
                )
            except Exception as e:
                log.error(f"Error saving best model checkpoint: {e}")

        # -- Learning Rate Scheduler Step --
        if scheduler:
            scheduler.step()  # Argument depends on scheduler type, some need metrics

        epoch_duration = time.time() - epoch_start_time
        log.info(f"--- Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds ---")

    # --- End of Training ---
    total_training_duration = time.time() - script_start_time
    log.info(f"[DONE] Training finished after {cfg.training.epochs} epochs.")
    log.info(
        f"[BEST] Best Validation Accuracy: {best_val_accuracy:.2f}% (achieved at epoch {best_epoch})"
    )
    log.info(f"[TIME] Total Training Script Time: {total_training_duration:.2f} seconds")

    # Save the final model state (regardless of performance)
    try:
        torch.save(model.state_dict(), final_model_save_path)
        log.info(f"Final model state saved to {os.path.abspath(final_model_save_path)}")
    except Exception as e:
        log.error(f"Error saving final model state: {e}")

    # --- Finish WandB Run ---
    if wandb_run:
        wandb.summary["best_val_accuracy"] = best_val_accuracy
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["total_training_time_sec"] = total_training_duration

        # Optionally, log the best model as an artifact if it was saved
        if os.path.exists(model_save_path) and best_epoch != -1:
            try:
                best_model_artifact = wandb.Artifact(
                    f"{cfg.model.name.replace('/', '_')}-best_model",
                    type="model",
                    description=f"Best model for {cfg.model.name} with val_acc {best_val_accuracy:.2f}% at epoch {best_epoch}",
                    metadata={
                        "epoch": best_epoch,
                        "val_accuracy": best_val_accuracy,
                        "config": OmegaConf.to_container(cfg),
                    },
                )
                best_model_artifact.add_file(model_save_path)
                wandb.log_artifact(best_model_artifact, aliases=["best", f"epoch_{best_epoch}"])
                log.info(f"Logged best model '{model_save_path}' to WandB Artifacts.")
            except Exception as e:
                log.error(f"Failed to log best model artifact to WandB: {e}")

        wandb.finish()
        log.info("WandB run finished.")

    return best_val_accuracy


# Entry point for the script
if __name__ == "__main__":
    try:
        train_model()
    except SystemExit:  # Catch sys.exit calls for cleaner termination message
        log.info("Training script exited.")
    except Exception:  # Catch any other unexpected top-level error
        log.exception("An uncaught exception occurred at the top level of the training script:")
        sys.exit(1)  # Ensure non-zero exit code for errors
