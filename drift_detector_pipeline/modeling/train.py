# drift_detector_pipeline/modeling/train.py
"""
Main script for training an image classification model using PyTorch, timm,
Hydra for configuration, WandB for experiment tracking, and PyTorch Profiler
for performance analysis.
"""

import contextlib  # For conditional context manager
import json  # For CML metrics export
import logging
import os
import sys
import time  # For basic timing

import hydra
import matplotlib.pyplot as plt  # For CML plots
from omegaconf import DictConfig, OmegaConf
import timm
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

# --- PyTorch Profiler Imports ---
from torch.profiler import ProfilerActivity, profile, record_function, schedule
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
    (Full docstring as before)
    """
    script_start_time = time.time()
    log.info("--- Initializing Model Training Script ---")
    log.info("Full configuration:\n%s", OmegaConf.to_yaml(cfg))

    # --- CML Setup ---
    cml_enabled = cfg.get("cml", {}).get("enabled", False) or cfg.run.get("ci_mode", False)
    if cml_enabled:
        log.info("CML reporting enabled - will export metrics and plots")
        # Create plots directory for CML
        plots_dir = cfg.get("cml", {}).get("plots_dir", "cml_plots")
        os.makedirs(plots_dir, exist_ok=True)
        log.info(f"CML plots will be saved to: {plots_dir}")

    # --- 1. Runtime Setup (Device, Seeds) ---
    try:
        seed = cfg.run.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
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
    except Exception as e:
        log.error(f"Unexpected error during runtime setup: {e}. Aborting.")
        sys.exit(1)

    # --- 2. Initialize Experiment Tracking (WandB) ---
    # Note: model for wandb.watch is defined later, so watch is called after model init
    wandb_run = None
    try:
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_config,
            name=f"{cfg.model.name.replace('/', '_')}-run-{time.strftime('%Y%m%d-%H%M%S')}",
            job_type="train",
            reinit=True,
        )
        log.info(f"WandB initialized. Run URL: {wandb_run.get_url() if wandb_run else 'N/A'}")
        # wandb.watch will be called after model initialization
    except AttributeError as e:
        log.error(f"Configuration error: Missing key in 'wandb' section - {e}. WandB disabled.")
    except Exception as e:
        log.warning(
            f"Failed to initialize WandB: {e}. Training will continue without WandB logging."
        )

    # --- 3. PyTorch Profiler Setup ---
    profiler_cfg = cfg.training.get("profiler", OmegaConf.create({}))
    profiler_enabled = profiler_cfg.get("enabled", False)
    pytorch_profiler_instance = None

    if profiler_enabled:
        log.info("PyTorch Profiler is ENABLED via configuration.")
        prof_activities = [ProfilerActivity.CPU]
        if device.type == "cuda" and torch.cuda.is_available():
            prof_activities.append(ProfilerActivity.CUDA)
            log.info("Profiling CPU and CUDA activities.")
        else:
            log.info("Profiling CPU activities only (CUDA not selected or not available).")

        prof_schedule_params = {
            "wait": profiler_cfg.get("wait", 1),
            "warmup": profiler_cfg.get("warmup", 1),
            "active": profiler_cfg.get("active", 3),
            "repeat": profiler_cfg.get("repeat", 1),
        }
        prof_schedule_instance = schedule(**prof_schedule_params)
        log.info(
            f"Profiler schedule: Wait={prof_schedule_params['wait']}, Warmup={prof_schedule_params['warmup']}, Active={prof_schedule_params['active']}, Repeat={prof_schedule_params['repeat']}"
        )

        trace_log_dir_abs = os.path.join(
            os.getcwd(), profiler_cfg.get("log_dir", "pytorch_profiler_logs")
        )
        os.makedirs(trace_log_dir_abs, exist_ok=True)
        log.info(f"PyTorch Profiler traces will be saved to: {trace_log_dir_abs}")

        def profiler_trace_handler(prof):
            key_averages_table = prof.key_averages().table(
                sort_by="self_cuda_time_total"
                if ProfilerActivity.CUDA in prof.activities
                else "self_cpu_time_total",
                row_limit=15,
            )
            log.info(
                f"\n--- PyTorch Profiler: Cycle {prof.step_num} Key Averages ---\n{key_averages_table}"
            )

            if profiler_cfg.get("export_chrome_trace", True):
                trace_file_path = os.path.join(
                    trace_log_dir_abs, f"trace_cycle_{prof.step_num}.json"
                )
                try:
                    prof.export_chrome_trace(trace_file_path)
                    log.info(
                        f"Exported Chrome trace for cycle {prof.step_num} to {trace_file_path}"
                    )
                except Exception as e_trace:
                    log.error(
                        f"Failed to export Chrome trace for cycle {prof.step_num}: {e_trace}"
                    )

        pytorch_profiler_instance = profile(
            activities=prof_activities,
            schedule=prof_schedule_instance,
            on_trace_ready=profiler_trace_handler,
            record_shapes=profiler_cfg.get("record_shapes", False),
            profile_memory=profiler_cfg.get("profile_memory", False),
            with_stack=profiler_cfg.get("with_stack", False),
            with_flops=profiler_cfg.get("with_flops", True),
            with_modules=profiler_cfg.get("with_modules", True),
        )
    else:
        log.info("PyTorch Profiler is DISABLED.")
        pytorch_profiler_instance = contextlib.nullcontext()

    # --- 4. Data Loading ---
    try:
        log.info("Loading data...")
        # get_dataloaders now handles its own call to extract_data(cfg)
        train_loader, val_loader = get_dataloaders(cfg)
        log.info("Data loaded successfully.")
        if wandb_run:
            wandb.config.update(
                {
                    "num_train_samples": len(train_loader.dataset),
                    "num_val_samples": len(val_loader.dataset),
                    "num_train_batches": len(train_loader),
                    "num_val_batches": len(val_loader),
                }
            )
            raw_dvc_meta_path = cfg.data.raw_path + ".dvc"
            if os.path.exists(raw_dvc_meta_path):
                try:
                    with open(raw_dvc_meta_path, "r") as f_dvc:
                        dvc_content = f_dvc.read()
                        md5_line = [line for line in dvc_content.splitlines() if "md5:" in line]
                        if md5_line:
                            raw_data_md5 = md5_line[0].split("md5:")[1].strip()
                            wandb.config.update({"raw_data_dvc_md5": raw_data_md5})
                except Exception as e_dvc_log:
                    log.warning(f"Could not log DVC metadata for raw data: {e_dvc_log}")
    except SystemExit:
        log.error("Data loading failed as get_dataloaders exited. Aborting training.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)
    except Exception:
        log.exception("Critical error during data loading. Aborting training.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)

    # --- 5. Model Setup ---
    model = None  # Initialize model to None
    try:
        log.info(f"Loading model: {cfg.model.name} (pretrained={cfg.model.pretrained})")
        model = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes,
        )
        model.to(device)
        log.info(f"Model '{cfg.model.name}' loaded and moved to {device.type.upper()}.")

        # MOVED wandb.watch here, after model is defined
        if wandb_run and model is not None and cfg.wandb.get("watch_model", False):
            wandb.watch(
                model,
                log=cfg.wandb.get("watch_log_type", "gradients"),
                log_freq=cfg.wandb.get("watch_log_freq", 1000),
            )
            log.info(
                f"WandB watching model (type: {cfg.wandb.get('watch_log_type', 'gradients')}, freq: {cfg.wandb.get('watch_log_freq', 1000)})"
            )

        if wandb_run:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb.config.update(
                {"total_model_params": total_params, "trainable_model_params": trainable_params}
            )
    except AttributeError as e:
        log.error(f"Configuration error: Missing key in 'model' section - {e}. Aborting.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)
    except Exception:
        log.exception(f"Critical error loading model '{cfg.model.name}'. Aborting.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)

    # --- 6. Loss Function, Optimizer, Scheduler ---
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.get("label_smoothing", 0.0))
        log.info(
            f"Loss function: CrossEntropyLoss (label_smoothing={cfg.training.get('label_smoothing', 0.0)})"
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
                momentum=cfg.training.get("momentum", 0.9),
                weight_decay=cfg.training.weight_decay,
            )
        else:
            log.error(f"Unsupported optimizer: '{cfg.training.optimizer}'. Aborting.")
            if wandb_run:
                wandb.finish(exit_code=1)
            sys.exit(1)
        log.info(
            f"Optimizer: {cfg.training.optimizer} (lr={cfg.training.learning_rate}, wd={cfg.training.weight_decay})"
        )

        scheduler_instance = None
        scheduler_name = cfg.training.get("scheduler", "none").lower()
        if scheduler_name == "cosineannealinglr":
            scheduler_instance = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.training.epochs
            )
        elif scheduler_name == "steplr":
            scheduler_instance = lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.training.get("scheduler_step_size", 10),
                gamma=cfg.training.get("scheduler_gamma", 0.1),
            )
        elif scheduler_name != "none":
            log.warning(
                f"Unsupported scheduler: '{cfg.training.scheduler}'. Proceeding without one."
            )

        if scheduler_instance:
            log.info(f"Learning rate scheduler: {cfg.training.scheduler}")
        else:
            log.info("No learning rate scheduler configured or 'none' specified.")
    except AttributeError as e:
        log.error(
            f"Configuration error: Missing key in 'training' section for optim/scheduler - {e}. Aborting."
        )
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)
    except Exception as e:
        log.exception(f"Error setting up optimizer/scheduler: {e}. Aborting.")
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)

    # --- 7. Training Loop ---
    log.info(f"Starting training for {cfg.training.epochs} epochs...")
    best_val_accuracy = 0.0
    best_epoch = -1
    model_save_path = "best_model.pth"
    final_model_save_path = "final_model.pth"

    # For CML tracking
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accuracies = []
    epoch_numbers = []

    with pytorch_profiler_instance as prof:
        for epoch in range(cfg.training.epochs):
            epoch_start_time = time.time()
            log.info(f"--- Epoch {epoch + 1}/{cfg.training.epochs} ---")

            model.train()
            running_train_loss = 0.0
            train_batches_processed = 0
            for i, (inputs, labels) in enumerate(train_loader):
                batch_start_time = time.time()  # DEFINED batch_start_time HERE
                with record_function("TRAIN_BATCH_PROCESSING"):
                    with record_function("data_to_device_train"):
                        inputs, labels = (
                            inputs.to(
                                device, non_blocking=True if device.type == "cuda" else False
                            ),
                            labels.to(
                                device, non_blocking=True if device.type == "cuda" else False
                            ),
                        )
                    with record_function("optimizer_zero_grad"):
                        optimizer.zero_grad(set_to_none=True)
                    with record_function("forward_pass_train"):
                        outputs = model(inputs)
                    with record_function("loss_calculation_train"):
                        loss = criterion(outputs, labels)
                    with record_function("backward_pass_train"):
                        loss.backward()
                    with record_function("optimizer_step_train"):
                        optimizer.step()

                running_train_loss += loss.item() * inputs.size(0)
                train_batches_processed += inputs.size(0)

                if wandb_run and (i + 1) % cfg.wandb.log_freq == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    wandb.log(
                        {
                            "epoch_frac": epoch + (i + 1) / len(train_loader),
                            "train/batch_loss": loss.item(),
                            "train/learning_rate": current_lr,
                            "perf/batch_time_sec": time.time()
                            - batch_start_time,  # Now batch_start_time is defined
                        }
                    )

                if (
                    profiler_enabled
                    and prof is not None
                    and not isinstance(prof, contextlib.nullcontext)
                ):
                    prof.step()

            avg_epoch_train_loss = (
                (running_train_loss / train_batches_processed)
                if train_batches_processed > 0
                else 0.0
            )
            log.info(f"Epoch {epoch + 1} Training: Avg Loss = {avg_epoch_train_loss:.4f}")
            if wandb_run:
                wandb.log({"epoch": epoch + 1, "train/epoch_loss": avg_epoch_train_loss})

            # Track for CML
            epoch_train_losses.append(avg_epoch_train_loss)
            epoch_numbers.append(epoch + 1)

            model.eval()
            running_val_loss = 0.0
            correct_val_predictions = 0
            total_val_samples = 0
            validation_start_time = time.time()
            with torch.no_grad():
                with record_function("VALIDATION_LOOP_TOTAL"):
                    for inputs, labels in val_loader:
                        with record_function("batch_processing_val"):
                            with record_function("data_to_device_val"):
                                inputs, labels = (
                                    inputs.to(
                                        device,
                                        non_blocking=True if device.type == "cuda" else False,
                                    ),
                                    labels.to(
                                        device,
                                        non_blocking=True if device.type == "cuda" else False,
                                    ),
                                )
                            with record_function("forward_pass_val"):
                                outputs = model(inputs)
                            with record_function("loss_calculation_val"):
                                loss = criterion(outputs, labels)
                        running_val_loss += loss.item() * inputs.size(0)
                        _, predicted_classes = torch.max(outputs, 1)
                        correct_val_predictions += (predicted_classes == labels).sum().item()
                        total_val_samples += labels.size(0)

            validation_duration = time.time() - validation_start_time
            avg_val_loss = running_val_loss / total_val_samples if total_val_samples > 0 else 0.0
            val_accuracy = (
                (100 * correct_val_predictions / total_val_samples)
                if total_val_samples > 0
                else 0.0
            )
            log.info(
                f"Epoch {epoch + 1} Validation: Avg Loss = {avg_val_loss:.4f}, Accuracy = {val_accuracy:.2f}% ({correct_val_predictions}/{total_val_samples})"
            )
            log.info(f"Epoch {epoch + 1} Validation Time: {validation_duration:.2f} seconds")
            if wandb_run:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "val/epoch_loss": avg_val_loss,
                        "val/accuracy": val_accuracy,
                        "perf/validation_time_sec": validation_duration,
                    }
                )

            # Track for CML
            epoch_val_losses.append(avg_val_loss)
            epoch_val_accuracies.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1
                try:
                    torch.save(model.state_dict(), model_save_path)
                    log.info(
                        f"[SUCCESS] New best model saved at Epoch {epoch + 1} to {os.path.abspath(model_save_path)}"
                    )
                except Exception as e_save:
                    log.error(f"Error saving best model: {e_save}")

            if scheduler_instance:
                scheduler_instance.step()
            log.info(
                f"--- Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f} seconds ---"
            )

    if (
        profiler_enabled
        and pytorch_profiler_instance is not None
        and not isinstance(pytorch_profiler_instance, contextlib.nullcontext)
    ):
        log.info(
            "PyTorch Profiler run processing complete. Traces should be saved by the handler."
        )

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
    except Exception as e_save_final:
        log.error(f"Error saving final model: {e_save_final}")

    # --- CML Reporting ---
    if cml_enabled:
        try:
            # Create plots directory for CML if not already created
            plots_dir = cfg.get("cml", {}).get("plots_dir", "cml_plots")
            os.makedirs(plots_dir, exist_ok=True)

            # Plot training and validation loss
            plt.figure(figsize=(10, 5))
            plt.plot(epoch_numbers, epoch_train_losses, "b-", label="Training Loss")
            plt.plot(epoch_numbers, epoch_val_losses, "r-", label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            loss_plot_path = os.path.join(plots_dir, "loss_plot.png")
            plt.savefig(loss_plot_path)
            plt.close()
            log.info(f"Saved loss plot to {loss_plot_path}")

            # Plot validation accuracy
            plt.figure(figsize=(10, 5))
            plt.plot(epoch_numbers, epoch_val_accuracies, "g-", label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Validation Accuracy")
            plt.legend()
            plt.grid(True)
            accuracy_plot_path = os.path.join(plots_dir, "accuracy_plot.png")
            plt.savefig(accuracy_plot_path)
            plt.close()
            log.info(f"Saved accuracy plot to {accuracy_plot_path}")

            # Export metrics to JSON for CML reporting
            metrics_file = cfg.get("cml", {}).get("metrics_file", "cml_metrics.json")
            metrics_data = {
                "best_val_accuracy": float(best_val_accuracy),
                "best_epoch": int(best_epoch),
                "final_train_loss": float(epoch_train_losses[-1]),
                "final_val_loss": float(epoch_val_losses[-1]),
                "final_val_accuracy": float(epoch_val_accuracies[-1]),
                "total_training_time_sec": float(total_training_duration),
                "model_name": cfg.model.name,
                "epochs_completed": len(epoch_numbers),
            }
            with open(metrics_file, "w") as f:
                json.dump(metrics_data, f, indent=2)
            log.info(f"Exported metrics to {metrics_file}")

        except Exception as e_cml:
            log.error(f"Error during CML reporting: {e_cml}")

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
                    description=f"Best model ({cfg.model.name}) ValAcc {best_val_accuracy:.2f}% @ Epoch {best_epoch}",
                    metadata={
                        "epoch": best_epoch,
                        "val_accuracy": best_val_accuracy,
                        "config": OmegaConf.to_container(cfg),
                    },
                )
                best_model_artifact.add_file(model_save_path)
                wandb.log_artifact(best_model_artifact, aliases=["best", f"epoch_{best_epoch}"])
                log.info(f"Logged '{model_save_path}' to WandB Artifacts.")
            except Exception as e_art:
                log.error(f"Failed to log best model artifact: {e_art}")
        wandb.finish()
        log.info("WandB run finished.")

    return best_val_accuracy


# Entry point for the script
if __name__ == "__main__":
    try:
        train_model()
    except SystemExit:
        log.info("Training script exited via sys.exit().")
    except Exception:
        log.exception("An uncaught exception occurred at the top level of the training script:")
        sys.exit(1)
