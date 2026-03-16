"""Demo: train YOLO classification with different loss types and compare results.

Uses a local dataset at data/classification/ with class subfolders (e.g. Approve/, Reject/).
Auto-splits into train/val (80/20) if no train/ subfolder exists.

Trains 9 models (50 epochs each) with aggressive Reject-boosting to show the effect of:
  1. ce              – standard cross-entropy (baseline)
  2. ce + w10        – weighted CE (Reject ×10)
  3. ce + w20        – weighted CE (Reject ×20)
  4. focal γ=2       – focal loss
  5. focal γ=5       – focal loss with stronger hard-example focus
  6. focal γ=2 + w10 – focal loss + heavy Reject weight
  7. cb_focal γ=2    – class-balanced focal (auto frequency weights)
  8. cb_focal + w10  – class-balanced focal + extra Reject weight
  9. ce + smoothing  – cross-entropy with label smoothing

Usage:
    python scripts/test_classification_loss.py
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

# Ensure local ultralytics is used
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "classification"
SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "classification_split"


def prepare_train_val(raw_dir: Path, split_dir: Path, val_ratio: float = 0.2, seed: int = 42):
    """Split a flat class-folder dataset into train/val structure.

    If split_dir already exists it is reused (no re-split).
    """
    if (split_dir / "train").exists() and (split_dir / "val").exists():
        print(f"[INFO] Using existing split at {split_dir}")
        return

    print(f"[INFO] Splitting {raw_dir} -> {split_dir}  (val_ratio={val_ratio})")
    random.seed(seed)
    split_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(d.name for d in raw_dir.iterdir() if d.is_dir() and d.name not in ("train", "val", "test"))
    for cls_name in classes:
        cls_dir = raw_dir / cls_name
        images = sorted(f for f in cls_dir.iterdir() if f.is_file())
        random.shuffle(images)
        n_val = max(1, int(len(images) * val_ratio))
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        for img in train_imgs:
            dst = split_dir / "train" / cls_name / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst)
        for img in val_imgs:
            dst = split_dir / "val" / cls_name / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst)

        print(f"  {cls_name}: {len(train_imgs)} train, {len(val_imgs)} val")


def train_and_evaluate(
    run_name: str,
    data_dir: str,
    cls_loss: str = "ce",
    class_weights: dict | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    cb_beta: float = 0.9999,
    arcface_margin: float = 0.5,
    arcface_scale: float = 30.0,
    epochs: int = 50,
) -> dict:
    """Train a classification model and return metrics."""
    print(f"\n{'=' * 70}")
    print(f"  Training: {run_name}")
    print(f"  cls_loss={cls_loss}, class_weights={class_weights}")
    print(f"  label_smoothing={label_smoothing}, focal_gamma={focal_gamma}, cb_beta={cb_beta}")
    if cls_loss == "arcface":
        print(f"  arcface_margin={arcface_margin}, arcface_scale={arcface_scale}")
    print(f"  epochs={epochs}")
    print(f"{'=' * 70}\n")

    model = YOLO("yolo11n-cls.pt")

    kwargs = dict(
        data=data_dir,
        epochs=epochs,
        imgsz=64,
        batch=32,
        project="runs/cls_loss_demo",
        name=run_name,
        exist_ok=True,
        verbose=False,
        cls_loss=cls_loss,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        cb_beta=cb_beta,
        arcface_margin=arcface_margin,
        arcface_scale=arcface_scale,
    )
    if class_weights is not None:
        kwargs["class_weights"] = class_weights

    model.train(**kwargs)

    metrics = model.val(data=data_dir, imgsz=64, batch=32, verbose=False)
    top1 = metrics.top1
    top5 = metrics.top5

    return {"run_name": run_name, "top1": top1, "top5": top5}


def main():
    # ── Validate & split ──
    if not RAW_DIR.exists():
        print(f"ERROR: Dataset not found at {RAW_DIR}")
        sys.exit(1)

    prepare_train_val(RAW_DIR, SPLIT_DIR, val_ratio=0.2)
    data_dir = str(SPLIT_DIR)

    # Discover classes
    classes = sorted(d.name for d in (SPLIT_DIR / "train").iterdir() if d.is_dir())
    counts = {c: len(list((SPLIT_DIR / "train" / c).iterdir())) for c in classes}
    print(f"\nDataset: {data_dir}")
    print(f"Classes: {classes}")
    print(f"Train counts: {counts}")

    # Find the rare class to boost — Reject is far more important
    rare_class = min(counts, key=counts.get)
    print(f"Critical class (must not miss): '{rare_class}' ({counts[rare_class]} samples)")

    epochs = 50
    results = []

    # 1. Baseline CE
    results.append(train_and_evaluate("baseline_ce", data_dir, cls_loss="ce", epochs=epochs))

    # 2. Weighted CE — Reject ×10
    results.append(
        train_and_evaluate(
            "weighted_ce_10x",
            data_dir,
            cls_loss="ce",
            class_weights={rare_class: 10.0},
            epochs=epochs,
        )
    )

    # 3. Weighted CE — Reject ×20 (extreme)
    results.append(
        train_and_evaluate(
            "weighted_ce_20x",
            data_dir,
            cls_loss="ce",
            class_weights={rare_class: 20.0},
            epochs=epochs,
        )
    )

    # 4. Focal loss γ=2.0
    results.append(train_and_evaluate("focal_g2", data_dir, cls_loss="focal", focal_gamma=2.0, epochs=epochs))

    # 5. Focal loss γ=5.0 — stronger hard-example focus
    results.append(train_and_evaluate("focal_g5", data_dir, cls_loss="focal", focal_gamma=5.0, epochs=epochs))

    # 6. Focal loss γ=2 + Reject ×10
    results.append(
        train_and_evaluate(
            "focal_g2_w10",
            data_dir,
            cls_loss="focal",
            focal_gamma=2.0,
            class_weights={rare_class: 10.0},
            epochs=epochs,
        )
    )

    # 7. Class-balanced focal γ=2 (auto frequency weights)
    results.append(
        train_and_evaluate("cb_focal_g2", data_dir, cls_loss="cb_focal", focal_gamma=2.0, cb_beta=0.999, epochs=epochs)
    )

    # 8. Class-balanced focal γ=2 + extra Reject ×10
    results.append(
        train_and_evaluate(
            "cb_focal_g2_w10",
            data_dir,
            cls_loss="cb_focal",
            focal_gamma=2.0,
            cb_beta=0.999,
            class_weights={rare_class: 10.0},
            epochs=epochs,
        )
    )

    # 9. CE + label smoothing
    results.append(train_and_evaluate("ce_label_smooth", data_dir, cls_loss="ce", label_smoothing=0.1, epochs=epochs))

    # 10. ArcFace — default margin=0.5, scale=30
    results.append(
        train_and_evaluate(
            "arcface_m05_s30", data_dir, cls_loss="arcface", arcface_margin=0.5, arcface_scale=30.0, epochs=epochs
        )
    )

    # 11. ArcFace — larger margin for stricter separation
    results.append(
        train_and_evaluate(
            "arcface_m10_s30", data_dir, cls_loss="arcface", arcface_margin=1.0, arcface_scale=30.0, epochs=epochs
        )
    )

    # 12. ArcFace + class_weights boosting Reject
    results.append(
        train_and_evaluate(
            "arcface_m05_w10",
            data_dir,
            cls_loss="arcface",
            arcface_margin=0.5,
            arcface_scale=30.0,
            class_weights={rare_class: 10.0},
            epochs=epochs,
        )
    )

    # ── Summary ──
    print("\n" + "=" * 80)
    print(f"  CLASSIFICATION LOSS COMPARISON SUMMARY  ({epochs} epochs)")
    print(f"  Dataset: {data_dir}")
    print(f"  Classes: {classes}  |  Train counts: {counts}")
    print(f"  Critical class: '{rare_class}' — boosted with weights 10x / 20x")
    print("=" * 80)
    print(f"  {'#':<4} {'Run':<28} {'Top-1 Acc':>10} {'Top-5 Acc':>10}")
    print(f"  {'─' * 4} {'─' * 28} {'─' * 10} {'─' * 10}")
    for i, r in enumerate(results, 1):
        print(f"  {i:<4} {r['run_name']:<28} {r['top1']:>10.4f} {r['top5']:>10.4f}")
    print("=" * 80)
    print("  Runs with higher Reject weight sacrifice some Approve accuracy")
    print("  to ensure Reject samples are rarely missed (lower false-negative rate).")
    print("  ArcFace learns angularly separated embeddings for stronger class boundaries.")
    print("=" * 80)


if __name__ == "__main__":
    main()
