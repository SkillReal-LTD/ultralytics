# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import shutil
from pathlib import Path

from ultralytics.utils import LOGGER


def print_val_metrics(validator):
    """Print the validation metrics so that Sagemaker can see the metrics."""
    results_dict = validator.metrics.results_dict
    for key, value in results_dict.items():
        LOGGER.info(f"{key}: {value}")


def on_train_start(trainer):
    """Add validation callbacks when training starts."""
    LOGGER.info("SkillReal: Adding validation callbacks")
    trainer.validator.add_callback("on_val_end", print_val_metrics)


def on_model_save(trainer):
    """Callback function to sync ALL checkpoint files to sync folder during training.

    Syncs:
    - last.pt: Latest checkpoint (always synced to root)
    - best.pt: Best performing checkpoint (always synced to root)
    - epoch{N}.pt: Periodic checkpoints (synced to 'epochs/' subfolder, skip if exists)

    Folder structure: SYNC_CHECKPOINT_FOLDER/
    ├── last.pt
    ├── best.pt
    └── epochs/
        ├── epoch5.pt
        ├── epoch10.pt
        └── ...

    Configuration via environment variables:
    - SYNC_CHECKPOINT_FOLDER: Destination folder for checkpoints
    """
    # Get configuration from environment variables
    sync_checkpoint_folder = os.getenv("SYNC_CHECKPOINT_FOLDER")

    # Skip if sync folder not configured
    if not sync_checkpoint_folder:
        LOGGER.warning("SkillReal: SYNC_CHECKPOINT_FOLDER not set, skipping checkpoint sync")
        return

    # Ensure sync folder and epochs subfolder exist
    sync_folder_path = Path(sync_checkpoint_folder)
    sync_folder_path.mkdir(parents=True, exist_ok=True)

    epochs_folder = sync_folder_path / "epochs"
    epochs_folder.mkdir(parents=True, exist_ok=True)

    # Sync last.pt to root (always sync - it updates every epoch)
    if trainer.last.exists():
        LOGGER.info(f"SkillReal: Syncing last checkpoint to {sync_checkpoint_folder}")
        try:
            shutil.copy(trainer.last, sync_folder_path)
        except Exception as e:
            LOGGER.error(f"SkillReal: Failed to sync last checkpoint: {e}")

    # Sync best.pt to root (always sync - it updates when new best is found)
    if trainer.best.exists():
        LOGGER.info(f"SkillReal: Syncing best checkpoint to {sync_checkpoint_folder}")
        try:
            shutil.copy(trainer.best, sync_folder_path)
        except Exception as e:
            LOGGER.error(f"SkillReal: Failed to sync best checkpoint: {e}")

    # Sync epoch checkpoints to 'epochs/' subfolder (skip if already exists)
    epoch_checkpoints = sorted(trainer.wdir.glob("epoch*.pt"))
    for ckpt_path in epoch_checkpoints:
        dest_path = epochs_folder / ckpt_path.name

        # Skip if already synced
        if dest_path.exists():
            LOGGER.debug(f"SkillReal: Skipping {ckpt_path.name} (already synced)")
            continue

        LOGGER.info(f"SkillReal: Syncing {ckpt_path.name} to {epochs_folder}")
        try:
            shutil.copy(ckpt_path, epochs_folder)
        except Exception as e:
            LOGGER.error(f"SkillReal: Failed to sync {ckpt_path.name}: {e}")


# Export callbacks dictionary
callbacks = {
    "on_train_start": on_train_start,
    "on_model_save": on_model_save,
}
