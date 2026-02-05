# Best Model Selection Logging

## Overview

The Ultralytics trainer now provides detailed logging whenever a new best model is saved during training. This helps you understand exactly why a particular checkpoint was chosen as the best model based on your fitness configuration.

## Features

### Automatic Logging

Every time a new best model is saved (when fitness improves), the trainer logs:

1. **Epoch number** - Which epoch produced the best model
2. **Fitness score** - The current best fitness value
3. **Fitness weights** - The weights used to calculate fitness
4. **All relevant metrics** - Precision, Recall, mAP@0.5, mAP@0.5:0.95
5. **Fitness calculation breakdown** - Shows exactly how the fitness was computed

### Task-Specific Logging

The logging adapts based on your task type:

#### Detection Tasks (4-weight fitness)

Shows standard detection metrics with clear calculation:

```
================================================================================
🏆 New Best Model Saved at Epoch 43
================================================================================
Best fitness: 0.7555

Fitness weights: [0.0, 0.0, 0.1, 0.9]

📊 Metrics:
  Precision: 0.8500 | Recall: 0.8200 | mAP@0.5: 0.8800 | mAP@0.5:0.95: 0.7500

📈 Fitness Calculation:
  mAP@0.5(0.8800) × 0.1 + mAP@0.5:0.95(0.7500) × 0.9 = 0.7555
================================================================================
```

#### Pose Tasks (8-weight fitness)

Shows separate box and pose metrics with individual contributions:

```
================================================================================
🏆 New Best Model Saved at Epoch 88
================================================================================
Best fitness: 0.8320

Fitness weights: [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.1]

📊 Box Detection Metrics (weights: [0.0, 0.0, 0.0, 0.0]):
  metrics/precision(B): 0.7800 | metrics/recall(B): 0.7500 | metrics/mAP50(B): 0.8200 | metrics/mAP50-95(B): 0.7000
  Box fitness contribution: 0.0000

🎯 Pose Keypoint Metrics (weights: [0.0, 0.9, 0.0, 0.1]):
  metrics/precision(P): 0.8200 | metrics/recall(P): 0.8400 | metrics/mAP50(P): 0.8700 | metrics/mAP50-95(P): 0.7600
  Pose fitness contribution: 0.8320

📈 Total Fitness Calculation:
  0.0000 (box) + 0.8320 (pose) = 0.8320
================================================================================
```

#### Segmentation Tasks (8-weight fitness)

Shows separate box and mask metrics:

```
================================================================================
🏆 New Best Model Saved at Epoch 121
================================================================================
Best fitness: 1.5321

Fitness weights: [0.0, 0.0, 0.05, 0.45, 0.0, 0.0, 0.05, 0.45]

📊 Box Detection Metrics (weights: [0.0, 0.0, 0.05, 0.45]):
  metrics/precision(B): 0.8500 | metrics/recall(B): 0.8300 | metrics/mAP50(B): 0.8800 | metrics/mAP50-95(B): 0.7600
  Box fitness contribution: 0.3860

🎭 Mask Segmentation Metrics (weights: [0.0, 0.0, 0.05, 0.45]):
  metrics/precision(M): 0.9000 | metrics/recall(M): 0.8800 | metrics/mAP50(M): 0.9200 | metrics/mAP50-95(M): 0.8500
  Mask fitness contribution: 0.4285

📈 Total Fitness Calculation:
  0.3860 (box) + 0.4285 (mask) = 1.5321
================================================================================
```

## Configuration

### Using Custom Fitness Weights

Set custom fitness weights in your training configuration:

#### Detection/OBB Tasks (4 weights)

```yaml
# config.yaml
fitness_weight: [0.0, 0.9, 0.1, 0.0] # Optimize for recall
```

#### Pose/Segment Tasks (8 weights)

```yaml
# config.yaml
# Optimize for pose keypoint accuracy only
fitness_weight: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9]

# Balanced box and pose
fitness_weight: [0.0, 0.0, 0.05, 0.45, 0.0, 0.0, 0.05, 0.45]

# Box 80%, Pose 20%
fitness_weight: [0.0, 0.0, 0.08, 0.72, 0.0, 0.0, 0.02, 0.18]
```

### Programmatic Configuration

```python
from ultralytics import YOLO

# For pose task - optimize pose only
model = YOLO("yolo11n-pose.pt")
model.train(data="coco8-pose.yaml", epochs=100, fitness_weight=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9])
```

## Benefits

### 1. **Transparency**

See exactly why each model was chosen as best, making it clear which metrics drove the selection.

### 2. **Debugging**

Quickly identify if your fitness weights are working as intended.

### 3. **Optimization Verification**

For pose/segment tasks with 8-weight fitness, verify that your model is being optimized for the right metrics (box vs pose/mask).

### 4. **Training Insights**

Understand which metrics improve over time and which contribute most to fitness.

### 5. **Reproducibility**

Logs include complete fitness weight configuration, making experiments reproducible.

## Example Training Logs

During a typical training session, you'll see logs like this every time a new best model is found:

```
Epoch 1/100: ...
[validation metrics]

Epoch 2/100: ...
[validation metrics]

================================================================================
🏆 New Best Model Saved at Epoch 3
================================================================================
Best fitness: 0.6234

Fitness weights: [0.0, 0.0, 0.05, 0.45, 0.0, 0.0, 0.05, 0.45]

📊 Box Detection Metrics (weights: [0.0, 0.0, 0.05, 0.45]):
  metrics/precision(B): 0.7200 | metrics/recall(B): 0.6800 | metrics/mAP50(B): 0.7500 | metrics/mAP50-95(B): 0.6100
  Box fitness contribution: 0.3120

🎯 Pose Keypoint Metrics (weights: [0.0, 0.0, 0.05, 0.45]):
  metrics/precision(P): 0.7800 | metrics/recall(P): 0.7200 | metrics/mAP50(P): 0.8100 | metrics/mAP50-95(P): 0.6900
  Pose fitness contribution: 0.3510

📈 Total Fitness Calculation:
  0.3120 (box) + 0.3510 (pose) = 0.6630
================================================================================

Epoch 4/100: ...
[continues training]
```

## Technical Details

### Implementation

The logging is implemented in the `BaseTrainer` class:

- **Method**: `_log_best_model_selection()`
- **Trigger**: Called automatically when `best.pt` is saved
- **Location**: `ultralytics/engine/trainer.py`

### Metrics Available

#### Detection/OBB

- `metrics/precision(B)`
- `metrics/recall(B)`
- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`

#### Pose (additional)

- `metrics/precision(P)` - Pose keypoint precision
- `metrics/recall(P)` - Pose keypoint recall
- `metrics/mAP50(P)` - Pose mAP@0.5
- `metrics/mAP50-95(P)` - Pose mAP@0.5:0.95

#### Segmentation (additional)

- `metrics/precision(M)` - Mask precision
- `metrics/recall(M)` - Mask recall
- `metrics/mAP50(M)` - Mask mAP@0.5
- `metrics/mAP50-95(M)` - Mask mAP@0.5:0.95

## Demo Script

Run the demonstration to see example logs:

```bash
python test_best_model_logging.py
```

This shows what the logs look like for different task types and fitness configurations.

## Related Documentation

- [8-Weight Fitness Configuration](../ultralytics/cfg/default.yaml#L40)
- [PoseMetrics Implementation](../ultralytics/utils/metrics.py#L1255)
- [SegmentMetrics Implementation](../ultralytics/utils/metrics.py#L1119)
- [Fitness Weight Tests](../tests/test_fitness_weight.py)
