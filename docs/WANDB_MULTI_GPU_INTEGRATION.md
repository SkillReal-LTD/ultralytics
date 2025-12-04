# W&B Multi-GPU (DDP) Integration Guide

## Overview

This document explains how to integrate an external `WandbManager` with Ultralytics training in multi-GPU (DDP) environments. The Ultralytics W&B callback now supports resuming runs created by external managers.

## Problem

In multi-GPU training, Ultralytics uses `torch.distributed.run` to spawn separate subprocesses for each GPU. This creates process isolation:

1. **Parent process**: Your script runs here, creates `WandbManager`
2. **DDP spawning**: Ultralytics spawns new child processes
3. **Rank 0 subprocess**: Training happens here, needs W&B access
4. **Issue**: Parent's W&B run is not accessible to child processes

## Solution

Use environment variables to pass W&B run information from parent to child processes.

### Architecture

```
Parent Process                    Rank 0 Subprocess
─────────────────                 ─────────────────
1. WandbManager.initialize()
   - Creates W&B run
   - Sets WANDB_RUN_ID env var
   - Sets WANDB_PROJECT env var
   - Uploads pre-training artifacts
   - Closes run (wandb.finish())

2. model.train()
   ├──> torch.distributed.run
   │                               3. on_pretrain_routine_start()
   │                                  - Reads WANDB_RUN_ID
   │                                  - Reads WANDB_PROJECT
   │                                  - wb.init(..., id=run_id, resume="allow")
   │                                  - Resumes parent's run ✓
   │
   │                               4. Training runs
   │                                  - Logs metrics to resumed run
   │
   │                               5. on_train_end()
   │                                  - wb.run.finish()

6. WandbManager.resume_run()
   - Reopens run to log final artifacts
   - wandb.finish()
```

## Implementation

### 1. Ultralytics Callback (Already Updated)

The W&B callback in `ultralytics/utils/callbacks/wb.py` now checks for environment variables:

```python
def on_pretrain_routine_start(trainer):
    """Initialize and start wandb project if module is present."""
    if not wb.run:
        import os

        # Check for existing run ID from external WandbManager
        run_id = os.getenv("WANDB_RUN_ID")
        project = os.getenv("WANDB_PROJECT")

        if run_id and project:
            # Resume existing run created by external WandbManager
            wb.init(
                project=project,
                id=run_id,
                resume="allow",  # Resume if exists, create if not
                config=vars(trainer.args),
            )
        else:
            # Create new run (original behavior)
            wb.init(
                project=str(trainer.args.project).replace("/", "-") if trainer.args.project else "Ultralytics",
                name=str(trainer.args.name).replace("/", "-"),
                config=vars(trainer.args),
            )
```

### 2. Your WandbManager (External Project)

Update your `WandbManager` to set environment variables and close the run before training:

```python
import os
import wandb

class WandbManager:
    def __init__(self, cfg, job_name, experiment_group_name=None):
        self.cfg = cfg
        self.job_name = job_name
        self.experiment_group_name = experiment_group_name
        self.run = None
        self.wandb_url = None

    def initialize(self):
        """Initialize Wandb run and set env vars for DDP."""
        rank = get_rank()

        if not is_main_process():
            logger.info(f"Skipping wandb initialization on rank {rank}")
            self.run = None
            self.wandb_url = None
            return None, None

        logger.info(f"Initializing wandb on rank {rank}")

        # Set API key
        os.environ["WANDB_API_KEY"] = self.cfg.wandb.api_key

        # Set run group if provided
        if self.experiment_group_name:
            os.environ["EXPERIMENT_GROUP_NAME"] = self.experiment_group_name

        # Initialize W&B run
        self.run = wandb.init(
            project=self.cfg.wandb.project,
            name=self.job_name,
            job_type="training",
            group=os.getenv("EXPERIMENT_GROUP_NAME", None),
        )
        self.wandb_url = wandb.run.get_url()

        # ✅ Set env vars for DDP subprocesses to resume this run
        os.environ["WANDB_RUN_ID"] = self.run.id
        os.environ["WANDB_PROJECT"] = self.cfg.wandb.project

        logger.info(f"Wandb run URL: {self.wandb_url}")
        logger.info(f"Set WANDB_RUN_ID={self.run.id} for DDP subprocesses")

        return self.run, self.wandb_url

    def prepare_for_training(self):
        """Upload pre-training artifacts and close run so DDP can resume it.

        IMPORTANT: Call this BEFORE starting Ultralytics training!
        """
        if not is_main_process() or self.run is None:
            return

        logger.info("Closing W&B run - Ultralytics will resume it in DDP subprocess")
        wandb.finish()

    def resume_after_training(self):
        """Resume the run after training to log additional artifacts.

        Call this AFTER Ultralytics training completes.
        """
        if not is_main_process():
            return

        run_id = os.environ.get("WANDB_RUN_ID")
        project = os.environ.get("WANDB_PROJECT")

        if not run_id:
            logger.warning("No WANDB_RUN_ID found, cannot resume")
            return

        logger.info(f"Resuming W&B run {run_id} for post-training artifacts")
        self.run = wandb.init(
            project=project,
            id=run_id,
            resume="must",
        )
        self.wandb_url = self.run.get_url()
        logger.info(f"Resumed W&B run: {self.wandb_url}")

    def save_config(self, raw_cfg):
        """Save configuration to Wandb."""
        if not is_main_process() or self.run is None:
            return

        os.makedirs("/tmp/configs", exist_ok=True)
        config_path = "/tmp/configs/training-job-config.yaml"

        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(raw_cfg))

        self.run.save(config_path, base_path="/tmp")

    def finish(self):
        """Finish Wandb run."""
        if not is_main_process() or self.run is None:
            return
        logger.info("Finishing W&B run")
        wandb.finish()

    def log_s3_artifact(self, s3_path, artifact_name, artifact_type="model"):
        """Log S3 artifacts to the run."""
        if not is_main_process():
            return

        if not self.run:
            logger.warning("No active run, call resume_after_training() first")
            return

        if not s3_path or not s3_path.startswith("s3://"):
            logger.info(f"Skipping artifact logging: invalid S3 path '{s3_path}'")
            return

        try:
            artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
            artifact.add_reference(uri=s3_path)
            self.run.log_artifact(artifact)
            logger.info(f"Logged S3 artifact '{artifact_name}' referencing: {s3_path}")
        except Exception as e:
            logger.warning(f"Failed to log S3 artifact '{artifact_name}': {e}")
```

### 3. Usage Pattern

```python
from ultralytics import YOLO

# 1. Initialize WandbManager and upload pre-training artifacts
wandb_manager = WandbManager(cfg, job_name="my-training", experiment_group_name="exp-1")
wandb_manager.initialize()

# Upload configs and metadata
wandb_manager.save_config(raw_cfg)
wandb_manager.save_training_artifacts()
wandb_manager.save_training_job_metadata(metadata)

# 2. IMPORTANT: Close the run before training starts
wandb_manager.prepare_for_training()

# 3. Start Ultralytics training
# DDP subprocess will automatically resume the W&B run
model = YOLO('yolo11n.pt')
results = model.train(
    data='coco8.yaml',
    epochs=100,
    device=[0, 1, 2, 3],  # Multi-GPU
)

# 4. After training, resume run to log final artifacts
wandb_manager.resume_after_training()
wandb_manager.log_s3_artifact('s3://bucket/model.pt', 'final_model')

# 5. Finish the run
wandb_manager.finish()
```

## How It Works

### Environment Variable Inheritance

1. **Parent sets env vars** before spawning DDP:
   ```python
   os.environ["WANDB_RUN_ID"] = "abc123"
   os.environ["WANDB_PROJECT"] = "my-project"
   ```

2. **DDP spawning** via `torch.distributed.run`:
   - Child processes inherit all environment variables
   - Each rank gets the same `WANDB_RUN_ID`

3. **Rank 0 subprocess** reads env vars:
   ```python
   run_id = os.getenv("WANDB_RUN_ID")  # "abc123"
   wandb.init(id=run_id, resume="allow")
   ```

### Run Lifecycle

```
Timeline                 W&B Run State              Active Process
────────────────────────────────────────────────────────────────────
t1: initialize()         CREATED & OPEN             Parent
t2: save_config()        OPEN (logging artifacts)   Parent
t3: prepare_for_training() CLOSED                   Parent
t4: model.train()        [spawning DDP...]          -
t5: on_pretrain...()     RESUMED & OPEN             Rank 0 child
t6: training...          OPEN (logging metrics)     Rank 0 child
t7: on_train_end()       CLOSED                     Rank 0 child
t8: resume_after_training() RESUMED & OPEN          Parent
t9: log_s3_artifact()    OPEN (logging artifacts)   Parent
t10: finish()            CLOSED (final)             Parent
```

## Key Points

### ✅ DO

1. **Always call `prepare_for_training()`** before `model.train()`
2. **Set both env vars**: `WANDB_RUN_ID` and `WANDB_PROJECT`
3. **Close the run** before DDP spawning (via `wandb.finish()`)
4. **Resume after training** to log final artifacts
5. **Check `is_main_process()`** in all WandbManager methods

### ❌ DON'T

1. **Don't keep run open** during DDP spawning (causes conflicts)
2. **Don't forget to resume** after training if you need to log more
3. **Don't call `finish()`** from non-main processes
4. **Don't assume the run persists** across process boundaries

## Troubleshooting

### Issue: "Run abc123 not found"

**Cause**: Run was not created or env var not set correctly.

**Solution**:
```python
# Ensure initialize() returns successfully
run, url = wandb_manager.initialize()
assert run is not None, "W&B initialization failed"
```

### Issue: "wandb: ERROR Run abc123 is already running"

**Cause**: Parent didn't close run before DDP spawning.

**Solution**: Always call `prepare_for_training()`:
```python
wandb_manager.initialize()
wandb_manager.prepare_for_training()  # ← Don't forget this!
model.train()
```

### Issue: Multiple W&B runs created

**Cause**: Env vars not set or callback not updated.

**Solution**:
1. Verify callback was updated in `ultralytics/utils/callbacks/wb.py`
2. Check env vars are set: `echo $WANDB_RUN_ID`
3. Enable debug logging: `os.environ["WANDB_DEBUG"] = "true"`

### Issue: Metrics not logged

**Cause**: Run was closed and not resumed in subprocess.

**Solution**: Ensure callback uses `resume="allow"`:
```python
wb.init(id=run_id, resume="allow")  # ← "allow" is important
```

## Benefits

✅ **Single W&B run** for entire training session
✅ **No code duplication** - reuse existing WandbManager
✅ **Clean separation** - pre/post training artifacts in manager, training metrics in Ultralytics
✅ **Works in both modes** - single GPU and multi-GPU
✅ **Simple integration** - just set env vars

## Related Files

- **Callback**: [ultralytics/utils/callbacks/wb.py](../ultralytics/utils/callbacks/wb.py)
- **Trainer**: [ultralytics/engine/trainer.py](../ultralytics/engine/trainer.py)
- **DDP Utils**: [ultralytics/utils/dist.py](../ultralytics/utils/dist.py)

## See Also

- [W&B Documentation - Run Resuming](https://docs.wandb.ai/guides/runs/resuming)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Ultralytics Multi-GPU Training](https://docs.ultralytics.com/modes/train/#multi-gpu-training)
