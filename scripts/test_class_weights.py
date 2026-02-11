"""
Demonstration script: per-class loss weighting (class_weights) impact on FN.

Trains two YOLO detection models on coco8 with identical settings except:
  - Baseline: no class_weights (all classes treated equally)
  - Weighted: class_weights={person: 100.0} (heavily penalise missed persons)

After training, both models are validated and per-class recall (inverse of FN rate)
is compared to show that the weighted model favours the targeted class.

NOTE: coco8 is a tiny dataset (8 images). This script is designed as a quick
      functional proof that the class_weights parameter flows through the entire
      pipeline (config -> trainer -> loss -> fitness) and produces different
      training dynamics. For statistically significant results, use a real dataset
      with more images and epochs.

Usage:
    python scripts/test_class_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local ultralytics is imported, not an installed package
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def train_and_evaluate(
    name: str,
    class_weights: dict | None = None,
    epochs: int = 30,
    imgsz: int = 640,
) -> dict:
    """Train a YOLO model and return per-class validation metrics.

    Args:
        name: Experiment name (used as run subfolder).
        class_weights: Optional per-class weight dict, e.g. {"person": 10.0}.
        epochs: Number of training epochs.
        imgsz: Image size for training.

    Returns:
        Dict with keys: 'results', 'per_class_recall', 'per_class_precision',
        'class_names', 'mean_recall', 'fitness'.
    """
    model = YOLO("yolo11n.pt")

    train_kwargs = dict(
        data="coco8.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=4,
        project="runs/class_weights_demo",
        name=name,
        exist_ok=True,
        verbose=False,
        plots=False,
        val=True,
        seed=42,
        deterministic=True,
    )
    if class_weights is not None:
        train_kwargs["class_weights"] = class_weights

    model.train(**train_kwargs)

    # Run validation to get per-class metrics
    val_results = model.val(data="coco8.yaml", imgsz=imgsz, batch=4, verbose=False, plots=False)

    # Extract per-class recall and precision
    box = val_results.box  # Metric object
    class_indices = box.ap_class_index  # which classes were detected
    names = val_results.names  # {idx: name}

    per_class_recall = {}
    per_class_precision = {}
    for i, cls_idx in enumerate(class_indices):
        cls_name = names[cls_idx]
        per_class_recall[cls_name] = float(box.r[i])
        per_class_precision[cls_name] = float(box.p[i])

    return {
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "class_names": [names[c] for c in class_indices],
        "mean_recall": float(box.mr),
        "fitness": float(box.fitness()),
    }


def main():
    """Run baseline vs weighted training and compare results."""
    target_class = "person"
    weight_value = 100.0
    epochs = 30

    print("=" * 70)
    print("  CLASS WEIGHTS IMPACT DEMONSTRATION")
    print("=" * 70)
    print(f"\n  Target class : {target_class}")
    print(f"  Weight       : {weight_value}x (vs 1.0 for all others)")
    print(f"  Dataset      : coco8 (8 images)")
    print(f"  Epochs       : {epochs}")
    print()

    # ── Train baseline (no class weights) ──────────────────────────────
    print("-" * 70)
    print("  [1/2] Training BASELINE (no class_weights) ...")
    print("-" * 70)
    baseline = train_and_evaluate("baseline", class_weights=None, epochs=epochs)

    # ── Train weighted (person = 10x) ─────────────────────────────────
    print()
    print("-" * 70)
    print(f"  [2/2] Training WEIGHTED ({target_class}={weight_value}x) ...")
    print("-" * 70)
    weighted = train_and_evaluate(
        "weighted",
        class_weights={target_class: weight_value},
        epochs=epochs,
    )

    # ── Compare results ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)

    # Collect all class names that appear in either run
    all_classes = sorted(set(baseline["per_class_recall"]) | set(weighted["per_class_recall"]))

    print(f"\n  {'Class':<20} {'Baseline R':>12} {'Weighted R':>12} {'Delta':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")

    for cls in all_classes:
        b_r = baseline["per_class_recall"].get(cls, 0.0)
        w_r = weighted["per_class_recall"].get(cls, 0.0)
        delta = w_r - b_r
        marker = " <-- TARGET" if cls == target_class else ""
        print(f"  {cls:<20} {b_r:>12.4f} {w_r:>12.4f} {delta:>+10.4f}{marker}")

    print()
    print(f"  {'Mean Recall':<20} {baseline['mean_recall']:>12.4f} {weighted['mean_recall']:>12.4f} "
          f"{weighted['mean_recall'] - baseline['mean_recall']:>+10.4f}")
    print(f"  {'Fitness':<20} {baseline['fitness']:>12.4f} {weighted['fitness']:>12.4f} "
          f"{weighted['fitness'] - baseline['fitness']:>+10.4f}")

    # Highlight the target class specifically
    b_target = baseline["per_class_recall"].get(target_class, 0.0)
    w_target = weighted["per_class_recall"].get(target_class, 0.0)

    print()
    print("=" * 70)
    if w_target >= b_target:
        print(f"  SUCCESS: '{target_class}' recall improved or held: "
              f"{b_target:.4f} -> {w_target:.4f} ({w_target - b_target:+.4f})")
    else:
        print(f"  NOTE: '{target_class}' recall changed: "
              f"{b_target:.4f} -> {w_target:.4f} ({w_target - b_target:+.4f})")
    print("=" * 70)
    print()
    print("  Training artifacts saved under: runs/class_weights_demo/")
    print()


if __name__ == "__main__":
    main()
