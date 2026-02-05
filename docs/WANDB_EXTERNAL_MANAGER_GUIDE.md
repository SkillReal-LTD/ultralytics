# External WandbManager Integration with Ultralytics

## Quick Start Guide

This guide shows how to integrate your external `WandbManager` with Ultralytics training in multi-GPU environments.

## What Changed in Ultralytics

Ultralytics W&B callback now supports resuming runs created externally via environment variables:

- `WANDB_RUN_ID` - The W&B run ID to resume
- `WANDB_PROJECT` - The W&B project name

## What You Need to Do

### 1. Update Your WandbManager

Add these three key changes to your `WandbManager` class:

#### A. Set Environment Variables in `initialize()`

```python
def initialize(self):
    """Initialize Wandb run."""
    rank = get_rank()

    if not is_main_process():
        logger.info(f"Skipping wandb initialization on rank {rank}")
        self.run = None
        self.wandb_url = None
        return None, None

    logger.info(f"Initializing wandb on rank {rank}")

    # Set API key and group as before
    os.environ["WANDB_API_KEY"] = self.cfg.wandb.api_key
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

    # ✅ ADD THESE TWO LINES - Set env vars for Ultralytics to resume
    os.environ["WANDB_RUN_ID"] = self.run.id
    os.environ["WANDB_PROJECT"] = self.cfg.wandb.project

    logger.info(f"Wandb run URL: {self.wandb_url}")
    logger.info(f"Set WANDB_RUN_ID={self.run.id} for Ultralytics DDP")

    return self.run, self.wandb_url
```

#### B. Add `prepare_for_training()` Method

```python
def prepare_for_training(self):
    """Close W&B run so Ultralytics can resume it in DDP subprocess.

    IMPORTANT: Call this BEFORE model.train()!
    """
    if not is_main_process() or self.run is None:
        return

    logger.info("Closing W&B run - Ultralytics will resume it")
    wandb.finish()
```

#### C. Add `resume_after_training()` Method

```python
def resume_after_training(self):
    """Resume W&B run after training to log additional artifacts.

    Call this AFTER model.train() completes.
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
```

### 2. Update Your Training Script

Change your training flow to close the run before Ultralytics training:

**BEFORE:**

```python
# Old way - doesn't work in multi-GPU
wandb_manager = WandbManager(cfg, job_name, experiment_group_name)
wandb_manager.initialize()
wandb_manager.save_config(raw_cfg)
wandb_manager.save_training_artifacts()

model.train(...)  # ❌ Run still open, DDP creates separate run

wandb_manager.log_s3_artifact(s3_path, "model")  # ❌ Wrong run
wandb_manager.finish()
```

**AFTER:**

```python
# New way - works in multi-GPU ✅
wandb_manager = WandbManager(cfg, job_name, experiment_group_name)
wandb_manager.initialize()
wandb_manager.save_config(raw_cfg)
wandb_manager.save_training_artifacts()
wandb_manager.save_training_job_metadata(metadata)

# ✅ Close run before training
wandb_manager.prepare_for_training()

# Ultralytics will automatically resume the run in rank 0
model.train(...)

# ✅ Resume run to log final artifacts
wandb_manager.resume_after_training()
wandb_manager.log_s3_artifact(s3_path, "final_model")
wandb_manager.finish()
```

## Complete Example

```python
from ultralytics import YOLO

# 1. Initialize WandbManager and upload pre-training artifacts
wandb_manager = WandbManager(cfg=cfg, job_name="yolo-training-run-001", experiment_group_name="experiment-v1")

# Creates run, sets WANDB_RUN_ID and WANDB_PROJECT env vars
run, url = wandb_manager.initialize()
print(f"W&B Run: {url}")

# Upload configs before training
wandb_manager.save_config(raw_cfg)
wandb_manager.save_training_artifacts()
wandb_manager.save_training_job_metadata(
    {
        "git_commit": "abc123",
        "model_version": "v1.0.0",
    }
)

# 2. CRITICAL: Close run before Ultralytics training
wandb_manager.prepare_for_training()

# 3. Run Ultralytics training (works with single or multi-GPU)
model = YOLO("yolo11n.pt")
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    device=[0, 1, 2, 3],  # Multi-GPU
)

# 4. Resume run for post-training artifacts
wandb_manager.resume_after_training()

# Log final artifacts
wandb_manager.log_s3_artifact(
    s3_path="s3://my-bucket/models/final.pt", artifact_name="final_model", artifact_type="model"
)

# 5. Clean up
wandb_manager.finish()
print("Training complete!")
```

## How It Works

### Single Process Flow

```
1. initialize()          → Create W&B run (id=abc123)
2. save_config()         → Upload artifacts to run abc123
3. prepare_for_training() → Close run abc123
4. model.train()         → Ultralytics resumes run abc123
                           Logs training metrics
                           Closes run abc123
5. resume_after_training() → Reopen run abc123
6. log_s3_artifact()     → Upload final artifacts
7. finish()              → Close run abc123 (final)
```

### Multi-GPU Flow

```
Parent Process              Rank 0 Subprocess           Rank 1-N Subprocesses
──────────────              ──────────────────          ─────────────────────
1. initialize()
   Set WANDB_RUN_ID=abc123
   Set WANDB_PROJECT=my-proj

2. save_config()

3. prepare_for_training()
   wandb.finish()

4. model.train()
   ├─> spawn DDP ─────────> Read WANDB_RUN_ID          Read env vars
                            Read WANDB_PROJECT          (no W&B init)

                            wb.init(id=abc123,
                                    resume="allow")

                            Training + logging

                            wb.run.finish()

5. resume_after_training()
   wb.init(id=abc123,
           resume="must")

6. log_s3_artifact()

7. finish()
```

## Key Points

### ✅ Must Do

1. **Set both env vars** in `initialize()`:
    - `os.environ["WANDB_RUN_ID"]`
    - `os.environ["WANDB_PROJECT"]`

2. **Call `prepare_for_training()`** before `model.train()`
    - This closes the run so DDP can resume it
    - Forgetting this causes "run already running" errors

3. **Call `resume_after_training()`** after `model.train()`
    - Only if you need to log more artifacts
    - Skip this if you're done after training

### ❌ Common Mistakes

1. **Don't forget to close run** before training:

    ```python
    wandb_manager.initialize()
    model.train()  # ❌ Run still open, causes conflicts
    ```

2. **Don't log artifacts without resuming**:

    ```python
    model.train()
    wandb_manager.log_s3_artifact(...)  # ❌ No active run
    ```

3. **Don't call `finish()` in both places**:
    ```python
    wandb_manager.prepare_for_training()  # Calls finish()
    # ... training ...
    wandb_manager.finish()  # ❌ Already finished by Ultralytics
    wandb_manager.resume_after_training()  # Must resume first!
    ```

## Minimal Code Changes

If you only need pre-training artifacts, the minimal change is:

```python
# Add to initialize()
os.environ["WANDB_RUN_ID"] = self.run.id
os.environ["WANDB_PROJECT"] = self.cfg.wandb.project

# Add before model.train()
wandb.finish()  # Close so Ultralytics can resume
```

That's it! Ultralytics handles everything else automatically.

## Verification

Check that it's working:

1. **Single W&B run created** - Not multiple runs
2. **Run shows both phases**:
    - Pre-training: Your configs/artifacts
    - Training: Ultralytics metrics/plots
    - Post-training: Your final artifacts (if using resume)
3. **No "run already running" errors**
4. **Multi-GPU works** - Same run ID across all ranks

## Troubleshooting

### "Run abc123 not found"

- **Cause**: Env vars not set or wrong run ID
- **Fix**: Verify `os.environ["WANDB_RUN_ID"] = self.run.id` in `initialize()`

### "wandb: ERROR Run abc123 is already running"

- **Cause**: Didn't call `prepare_for_training()` before `model.train()`
- **Fix**: Add `wandb_manager.prepare_for_training()` before training

### Multiple W&B runs created

- **Cause**: Env vars not being read by Ultralytics
- **Fix**: Check Ultralytics version has the updated callback

### "No active run" when logging artifacts

- **Cause**: Forgot to call `resume_after_training()`
- **Fix**: Call `wandb_manager.resume_after_training()` after training

## Summary

Three simple changes to make it work:

1. **Set env vars** → `os.environ["WANDB_RUN_ID"]` and `os.environ["WANDB_PROJECT"]`
2. **Close before training** → `wandb_manager.prepare_for_training()`
3. **Resume after training** → `wandb_manager.resume_after_training()` (optional)

That's it! Your WandbManager now works seamlessly with Ultralytics multi-GPU training.
