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
    """
    Callback function to move checkpoint files to sync checkpoint folder during training.

    This callback copies last and best checkpoints to the sync folder.

    Configuration via environment variables:
    - SYNC_CHECKPOINT_FOLDER: Destination folder for checkpoints
    """
    # Get configuration from environment variables
    sync_checkpoint_folder = os.getenv("SYNC_CHECKPOINT_FOLDER")

    # Skip if sync folder not configured
    if not sync_checkpoint_folder:
        LOGGER.warning("SkillReal: SYNC_CHECKPOINT_FOLDER not set, skipping checkpoint sync")
        return

    # Ensure sync folder exists
    sync_folder_path = Path(sync_checkpoint_folder)
    sync_folder_path.mkdir(parents=True, exist_ok=True)

    # Save the last checkpoint file
    if trainer.last.exists():
        LOGGER.info(f"SkillReal: Copying last checkpoint {trainer.last} to {sync_checkpoint_folder}")
        try:
            shutil.copy(trainer.last, sync_checkpoint_folder)
        except Exception as e:
            LOGGER.error(f"SkillReal: Failed to copy last checkpoint: {e}")

    # Save the best checkpoint file
    if trainer.best.exists():
        LOGGER.info(f"SkillReal: Copying best checkpoint {trainer.best} to {sync_checkpoint_folder}")
        try:
            shutil.copy(trainer.best, sync_checkpoint_folder)
        except Exception as e:
            LOGGER.error(f"SkillReal: Failed to copy best checkpoint: {e}")


# Export callbacks dictionary
callbacks = {
    "on_train_start": on_train_start,
    "on_model_save": on_model_save,
}
