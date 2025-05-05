# drift_detector_pipeline/modeling/train.py

import logging
import os
import sys
import torch
import timm
import wandb # Weights & Biases for experiment tracking
import hydra
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader # We will import actual loaders later
from drift_detector_pipeline.dataset import get_dataloaders

# Setup logging
log = logging.getLogger(__name__) # Hydra automatically configures this logger

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def train_model(cfg: DictConfig) -> None:
    """Main training function orchestrated by Hydra."""

    log.info("Starting Training Script...")
    log.info("Loaded Configuration:\n %s", OmegaConf.to_yaml(cfg))

    # --- Setup ---
    # Set random seed for reproducibility
    torch.manual_seed(cfg.run.seed)
    if cfg.run.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        # Add deterministic settings for reproducibility if needed (can slow down training)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
        log.warning("CUDA not available or not selected, using CPU.")
    log.info(f"Using device: {device}")

    # Initialize Weights & Biases
    wandb_run = None  # Create a separate variable to track if wandb is initialized
    try:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity if cfg.wandb.entity else None, # Use entity from config
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), # Log Hydra config
            name=f"{cfg.model.name}-run-{wandb.util.generate_id()}", # Example run name
            reinit=True # Allows re-running in notebooks/scripts
        )
        log.info("WandB initialized successfully.")
    except Exception as e:
        log.error(f"Failed to initialize WandB: {e}")

    # --- Data Loading ---
    log.info("Loading data...")
    train_loader, val_loader = get_dataloaders(cfg)
    log.info("Data loaded.")

    # --- Model Setup ---
    log.info(f"Loading model: {cfg.model.name}")
    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes
    )
    model.to(device)
    log.info("Model loaded successfully.")

    # Example: Layer Freezing (Uncomment and adapt if needed)
    # if cfg.training.get("freeze_layers", 0) > 0:
    #     log.info(f"Freezing first {cfg.training.freeze_layers} stages...")
    #     # Add logic here to freeze layers based on ResNet structure if needed

    # --- Loss Function, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    # Select optimizer based on config
    if cfg.training.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    elif cfg.training.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.training.learning_rate, momentum=0.9, weight_decay=cfg.training.weight_decay)
    else:
        log.error(f"Unsupported optimizer: {cfg.training.optimizer}")
        sys.exit(1)
    log.info(f"Using optimizer: {cfg.training.optimizer}")

    # Select scheduler based on config
    scheduler = None
    if cfg.training.scheduler.lower() == "cosineannealinglr":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs) # T_max could be total steps if preferred
    elif cfg.training.scheduler.lower() == "steplr":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Example parameters
    elif cfg.training.scheduler.lower() != "none":
         log.warning(f"Unsupported scheduler: {cfg.training.scheduler}. Training without scheduler.")
    if scheduler:
        log.info(f"Using scheduler: {cfg.training.scheduler}")

    # --- Training Loop ---
    log.info("Starting training loop...")
    best_val_accuracy = 0.0
    model_save_path = "best_model.pth" # Relative path, saves to Hydra output dir

    for epoch in range(cfg.training.epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        processed_batches = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed_batches += 1

            # Log training loss periodically to WandB
            if wandb_run and (i + 1) % cfg.wandb.log_freq == 0:
                current_batch_loss = running_loss / processed_batches
                wandb.log({
                    "epoch": epoch + (i + 1) / len(train_loader), # Log fractional epoch
                    "train_loss_batch": current_batch_loss,
                    "learning_rate": optimizer.param_groups[0]['lr'] # Log current LR
                 })
                running_loss = 0.0 # Reset loss accumulator for next log period
                processed_batches = 0

        # --- Validation Loop ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        log.info(f"Epoch {epoch+1}/{cfg.training.epochs} - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Log epoch metrics to WandB
        if wandb_run:
            wandb.log({
                "epoch": epoch + 1,
                "val_loss_epoch": avg_val_loss,
                "val_accuracy_epoch": val_accuracy
            })

        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            log.info(f"New best model saved with accuracy: {best_val_accuracy:.2f}% to {os.path.abspath(model_save_path)}")

        # Step the scheduler (if applicable)
        if scheduler:
            scheduler.step()

    log.info("Training finished.")
    log.info(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

    # Finish WandB run
    if wandb_run:
        # Optionally: Save the final model as a WandB artifact
        # artifact = wandb.Artifact(f'{cfg.model.name}-final', type='model')
        # artifact.add_file(model_save_path)
        # wandb.log_artifact(artifact)
        wandb.finish()

if __name__ == "__main__":
    train_model()