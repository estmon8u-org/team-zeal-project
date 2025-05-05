# drift_detector_pipeline/modeling/train.py
"""
Main script for training an image classification model using PyTorch, timm,
Hydra for configuration, and WandB for experiment tracking.

Handles model loading, data loading, optimizer/scheduler setup,
training/validation loops, checkpointing, and logging.
"""

import logging
import os
import sys
import time  # Import time for potential profiling

import hydra
from omegaconf import DictConfig, OmegaConf
import timm
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import wandb  # Weights & Biases for experiment tracking

# Ensure the package is importable (useful if running script directly)
try:
    from drift_detector_pipeline.dataset import get_dataloaders
except ImportError:
    # Add project root to path if running script directly assumes script is in drift_detector_pipeline/modeling
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from drift_detector_pipeline.dataset import get_dataloaders


# Setup module-level logger. Hydra will manage the handlers and formatting.
log = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def train_model(cfg: DictConfig) -> float:
    """
    Trains an image classification model based on the provided Hydra configuration.

    Sets up the device, seeds, WandB logging, data loaders, model architecture (from timm),
    optimizer, learning rate scheduler, and executes the training and validation loops.
    Saves the best model checkpoint based on validation accuracy to the Hydra output directory.

    Args:
        cfg (DictConfig): Configuration object managed by Hydra. Loaded from 'conf/config.yaml'
                          and command-line overrides.

    Returns:
        float: The best validation accuracy achieved during training. Returns 0.0 if training fails early.

    Raises:
        SystemExit: If configuration is invalid (e.g., unsupported optimizer) or
                    if data/model loading fails critically.
    """

    start_time = time.time()
    log.info("--- Starting Model Training Script ---")
    log.info("Configuration used:\n%s", OmegaConf.to_yaml(cfg))

    # --- Runtime Setup ---
    # Set random seeds for reproducibility across libraries
    seed = cfg.run.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # Potentially enable deterministic operations (can impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # Determine compute device
    if cfg.run.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if cfg.run.device == "cuda":
            log.warning("CUDA selected but not available. Falling back to CPU.")
        device = torch.device("cpu")
    log.info(f"Using device: {device}")

    # --- Initialize Experiment Tracking (WandB) ---
    wandb_run = None  # Flag to check if WandB is active
    try:
        # Convert OmegaConf to plain dict for WandB config compatibility
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,  # Can be None if logging to default entity
            config=wandb_config,
            name=f"{cfg.model.name}-run-{wandb.util.generate_id()}",  # Unique run name
            job_type="train",  # Add job type for better organization
            reinit=True,  # Allow re-initialization if running in interactive environments
        )
        log.info(f"WandB initialized successfully. Run URL: {wandb_run.get_url()}")
    except Exception as e:
        log.error(
            f"Failed to initialize WandB: {e}. Training will continue without WandB logging."
        )

    # --- Data Loading ---
    try:
        log.info("Loading data...")
        train_loader, val_loader = get_dataloaders(cfg)
        log.info("Data loaded successfully.")
    except Exception:
        log.exception("Critical error during data loading. Aborting training.")  # Log traceback
        if wandb_run:
            wandb.finish(exit_code=1)  # Ensure WandB run is marked as failed
        sys.exit(1)  # Exit if data can't be loaded

    # --- Model Setup ---
    try:
        log.info(f"Loading model: {cfg.model.name} (pretrained={cfg.model.pretrained})")
        model = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes,  # Adapts the final layer
        )
        model.to(device)
        log.info("Model loaded and moved to device.")
        # Log model architecture/parameters count to WandB if needed
        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # if wandb_run: wandb.config.update({"total_params": total_params, "trainable_params": trainable_params})
    except Exception:
        log.exception(f"Critical error loading model '{cfg.model.name}'. Aborting.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)

    # --- Loss Function, Optimizer, Scheduler ---
    # Using CrossEntropyLoss, suitable for multi-class classification
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
    log.info(
        f"Using loss function: CrossEntropyLoss (label_smoothing={cfg.training.label_smoothing})"
    )

    # Select optimizer based on config
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
            momentum=0.9,  # Common default for SGD
            weight_decay=cfg.training.weight_decay,
        )
    else:
        log.error(f"Unsupported optimizer specified in config: '{cfg.training.optimizer}'")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)
    log.info(f"Using optimizer: {cfg.training.optimizer}")

    # Select learning rate scheduler based on config
    scheduler = None
    scheduler_name = cfg.training.scheduler.lower()
    if scheduler_name == "cosineannealinglr":
        # Cosine annealing decays LR following a cosine curve over T_max epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    elif scheduler_name == "steplr":
        # StepLR decays LR by gamma every step_size epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Example defaults
    elif scheduler_name != "none":
        log.warning(
            f"Unsupported scheduler: '{cfg.training.scheduler}'. Proceeding without scheduler."
        )

    if scheduler:
        log.info(f"Using learning rate scheduler: {cfg.training.scheduler}")

    # --- Training Loop ---
    log.info(f"Starting training for {cfg.training.epochs} epochs...")
    best_val_accuracy = 0.0
    # Save model within the Hydra run directory (Hydra manages the CWD)
    model_save_path = "best_model.pth"
    final_model_save_path = "final_model.pth"

    for epoch in range(cfg.training.epochs):
        epoch_start_time = time.time()
        log.info(f"--- Starting Epoch {epoch + 1}/{cfg.training.epochs} ---")

        # --- Training Phase ---
        model.train()  # Set model to training mode (enables dropout, batch norm updates)
        running_loss = 0.0
        processed_batches_in_epoch = 0

        for i, (inputs, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients before the backward pass
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed_batches_in_epoch += 1

            # Log batch metrics periodically
            if wandb_run and (i + 1) % cfg.wandb.log_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]  # Get current learning rate
                avg_batch_loss = running_loss / processed_batches_in_epoch
                batch_time = time.time() - batch_start_time
                log_dict = {
                    # Log fractional epoch for finer granularity in plots
                    "epoch_frac": epoch + (i + 1) / len(train_loader),
                    "train/batch_loss": avg_batch_loss,
                    "train/learning_rate": current_lr,
                    "perf/batch_time_sec": batch_time,
                }
                wandb.log(log_dict)
                # Optional: log to console as well if verbose
                # log.debug(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}: Loss={avg_batch_loss:.4f}, LR={current_lr:.6f}")

                # Reset accumulator for the next logging interval
                running_loss = 0.0
                processed_batches_in_epoch = 0

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode (disables dropout, fixes batch norm)
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        validation_start_time = time.time()

        # Disable gradient calculations during validation for efficiency
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()  # Accumulate validation loss

                # Get predictions
                _, predicted_classes = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted_classes == labels).sum().item()

        validation_time = time.time() - validation_start_time
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_predictions / total_samples
        log.info(f"Epoch {epoch + 1} Validation Summary:")
        log.info(f"  Average Loss: {avg_val_loss:.4f}")
        log.info(f"  Accuracy: {val_accuracy:.2f}% ({correct_predictions}/{total_samples})")
        log.info(f"  Validation Time: {validation_time:.2f} seconds")

        # Log epoch summary metrics to WandB
        if wandb_run:
            wandb.log(
                {
                    "epoch": epoch + 1,  # Log integer epoch
                    "val/epoch_loss": avg_val_loss,
                    "val/accuracy": val_accuracy,
                    "perf/validation_time_sec": validation_time,
                }
            )

        # --- Checkpoint Saving ---
        # Save the model if validation accuracy has improved
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            try:
                # Save model state dict
                torch.save(model.state_dict(), model_save_path)
                # Use os.path.abspath to get the full path within the Hydra run directory
                abs_save_path = os.path.abspath(model_save_path)
                log.info(
                    f"✅ New best model saved at Epoch {epoch + 1} with accuracy: "
                    f"{best_val_accuracy:.2f}% to {abs_save_path}"
                )
                # Optionally log to wandb that a checkpoint was saved
                # if wandb_run: wandb.summary["best_val_accuracy"] = best_val_accuracy
            except Exception as e:
                log.error(f"Error saving model checkpoint: {e}")

        # --- Learning Rate Scheduler Step ---
        # Step the scheduler after the validation phase
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        log.info(
            f"--- Finished Epoch {epoch + 1}/{cfg.training.epochs} (Time: {epoch_time:.2f} seconds) ---"
        )

    # --- End of Training ---
    training_time = time.time() - start_time
    log.info(f"🏁 Training finished after {cfg.training.epochs} epochs.")
    log.info(f"🏆 Best Validation Accuracy achieved: {best_val_accuracy:.2f}%")
    log.info(f"⏱️ Total Training Time: {training_time:.2f} seconds")

    # Save the final model state (optional, could be different from the best)
    try:
        torch.save(model.state_dict(), final_model_save_path)
        log.info(f"Final model state saved to {os.path.abspath(final_model_save_path)}")
    except Exception as e:
        log.error(f"Error saving final model state: {e}")

    # --- Finish WandB Run ---
    if wandb_run:
        # Log the best accuracy as a summary metric
        wandb.summary["best_val_accuracy"] = best_val_accuracy
        wandb.summary["total_training_time_sec"] = training_time

        # Optionally: Link the best model artifact if saved via WandB
        # best_model_artifact = wandb.Artifact(f'{cfg.model.name}-best', type='model', metadata={"val_accuracy": best_val_accuracy, "epoch": best_epoch}) # Need to track best_epoch
        # best_model_artifact.add_file(model_save_path)
        # wandb.log_artifact(best_model_artifact, aliases=["best", f"epoch_{best_epoch}"])

        wandb.finish()
        log.info("WandB run finished.")

    return best_val_accuracy  # Return best accuracy for potential use in sweeps


# Entry point for running the script
if __name__ == "__main__":
    try:
        train_model()
    except Exception:
        log.exception("An uncaught exception occurred during script execution:")  # Log traceback
        sys.exit(1)  # Exit with error code
