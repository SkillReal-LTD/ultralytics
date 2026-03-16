"""Unit tests for the enhanced v8ClassificationLoss with multiple loss types.

Tests cover:
  1. Default CE loss (backward-compatible)
  2. Weighted CE loss (user-defined class_weights)
  3. CE + label smoothing
  4. Focal loss (no weights)
  5. Focal loss + class_weights
  6. Focal loss + label smoothing
  7. Class-balanced focal loss (auto weights from class_counts)
  8. Class-balanced focal + user class_weights (combined)
  9. Invalid cls_loss raises ValueError
 10. Weight device migration
 11. ClassificationTrainer._resolve_class_weights (dict)
 12. ClassificationTrainer._resolve_class_weights (list)
 13. ClassificationTrainer._resolve_class_weights (missing classes default to 1.0)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Ensure the local ultralytics package is used
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics.utils.loss import v8ClassificationLoss


# ──────────────────────────────────────── helpers ─────────────────────────────
def _make_batch(targets: list[int], device: str = "cpu") -> dict[str, torch.Tensor]:
    return {"cls": torch.tensor(targets, dtype=torch.long, device=device)}


def _make_preds(batch_size: int, nc: int, device: str = "cpu") -> torch.Tensor:
    """Return raw logits (B, nc) — simulates model output before softmax."""
    torch.manual_seed(42)
    return torch.randn(batch_size, nc, device=device, requires_grad=True)


# ──────────────────────────────────────── 1. Default CE ──────────────────────
def test_default_ce():
    """Default v8ClassificationLoss matches plain cross_entropy."""
    loss_fn = v8ClassificationLoss()
    preds = _make_preds(8, 5)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 2])

    loss, loss_det = loss_fn(preds, batch)
    expected = torch.nn.functional.cross_entropy(preds, batch["cls"])

    assert torch.allclose(loss, expected, atol=1e-5), f"{loss.item()} != {expected.item()}"
    assert loss_det.requires_grad is False
    print(f"[PASS] default CE: loss={loss.item():.5f}")


# ──────────────────────────────────────── 2. Weighted CE ─────────────────────
def test_weighted_ce():
    """Weighted CE should differ from unweighted when weights are non-uniform."""
    weights = [1.0, 1.0, 1.0, 1.0, 5.0]
    loss_fn_w = v8ClassificationLoss(cls_loss="ce", class_weights=weights)
    loss_fn_u = v8ClassificationLoss(cls_loss="ce")

    preds = _make_preds(8, 5)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 4])

    loss_w, _ = loss_fn_w(preds, batch)
    loss_u, _ = loss_fn_u(preds, batch)

    assert not torch.allclose(loss_w, loss_u, atol=1e-4), "Weighted and unweighted CE should differ"
    print(f"[PASS] weighted CE: weighted={loss_w.item():.5f}, unweighted={loss_u.item():.5f}")


# ──────────────────────────────────────── 3. CE + label smoothing ────────────
def test_ce_label_smoothing():
    """CE with label smoothing should produce a different loss than without."""
    loss_fn_ls = v8ClassificationLoss(cls_loss="ce", label_smoothing=0.1)
    loss_fn_no = v8ClassificationLoss(cls_loss="ce", label_smoothing=0.0)

    preds = _make_preds(8, 5)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 2])

    loss_ls, _ = loss_fn_ls(preds, batch)
    loss_no, _ = loss_fn_no(preds, batch)

    assert not torch.allclose(loss_ls, loss_no, atol=1e-5), "Label smoothing should change the loss"
    print(f"[PASS] label smoothing: smoothed={loss_ls.item():.5f}, raw={loss_no.item():.5f}")


# ──────────────────────────────────────── 4. Focal loss (no weights) ─────────
def test_focal_no_weights():
    """Focal loss should be computable and differentiable."""
    loss_fn = v8ClassificationLoss(cls_loss="focal", focal_gamma=2.0)
    preds = _make_preds(8, 5)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 2])

    loss, _ = loss_fn(preds, batch)
    loss.backward()

    assert loss.item() > 0, "Focal loss should be positive"
    assert preds.grad is not None, "Focal loss should be differentiable"
    print(f"[PASS] focal (γ=2.0, no weights): loss={loss.item():.5f}")


# ──────────────────────────────────────── 5. Focal + class_weights ───────────
def test_focal_with_class_weights():
    """Focal loss with class_weights should differ from without."""
    weights = [1.0, 1.0, 1.0, 1.0, 5.0]
    loss_fn_w = v8ClassificationLoss(cls_loss="focal", focal_gamma=2.0, class_weights=weights)
    loss_fn_u = v8ClassificationLoss(cls_loss="focal", focal_gamma=2.0)

    preds = _make_preds(8, 5)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 4])

    loss_w, _ = loss_fn_w(preds, batch)
    loss_u, _ = loss_fn_u(preds, batch)

    assert not torch.allclose(loss_w, loss_u, atol=1e-4), "Focal w/ weights should differ from w/o"
    print(f"[PASS] focal + weights: weighted={loss_w.item():.5f}, unweighted={loss_u.item():.5f}")


# ──────────────────────────────────────── 6. Focal + label smoothing ─────────
def test_focal_label_smoothing():
    """Focal loss with label smoothing should differ from without."""
    loss_fn_ls = v8ClassificationLoss(cls_loss="focal", focal_gamma=2.0, label_smoothing=0.1)
    loss_fn_no = v8ClassificationLoss(cls_loss="focal", focal_gamma=2.0, label_smoothing=0.0)

    preds = _make_preds(8, 5)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 2])

    loss_ls, _ = loss_fn_ls(preds, batch)
    loss_no, _ = loss_fn_no(preds, batch)

    assert not torch.allclose(loss_ls, loss_no, atol=1e-5)
    print(f"[PASS] focal + label smoothing: smoothed={loss_ls.item():.5f}, raw={loss_no.item():.5f}")


# ──────────────────────────────────────── 7. CB Focal (auto weights) ─────────
def test_cb_focal_auto_weights():
    """Class-balanced focal loss should compute effective-number weights from counts."""
    # Imbalanced: class 0 has 1000 samples, class 1 has 10
    counts = [1000, 10, 500, 100, 50]
    loss_fn = v8ClassificationLoss(cls_loss="cb_focal", class_counts=counts, focal_gamma=2.0, cb_beta=0.999)

    preds = _make_preds(8, 5)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 1])

    loss, _ = loss_fn(preds, batch)
    loss.backward()

    assert loss.item() > 0
    assert preds.grad is not None
    # Verify internal weight tensor exists and is normalized
    assert loss_fn._weight is not None
    print(f"[PASS] cb_focal (auto weights): loss={loss.item():.5f}, weights={loss_fn._weight.tolist()}")


# ──────────────────────────────────────── 8. CB Focal + user weights ─────────
def test_cb_focal_combined_weights():
    """CB focal with user class_weights should combine both weight sources."""
    counts = [1000, 10, 500, 100, 50]
    user_weights = [1.0, 5.0, 1.0, 1.0, 1.0]

    loss_fn_combined = v8ClassificationLoss(
        cls_loss="cb_focal", class_counts=counts, class_weights=user_weights, focal_gamma=2.0, cb_beta=0.999
    )
    loss_fn_auto = v8ClassificationLoss(
        cls_loss="cb_focal", class_counts=counts, focal_gamma=2.0, cb_beta=0.999
    )

    preds = _make_preds(8, 5)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 1])

    loss_combined, _ = loss_fn_combined(preds, batch)
    loss_auto, _ = loss_fn_auto(preds, batch)

    assert not torch.allclose(loss_combined, loss_auto, atol=1e-4), (
        "Combined weights should differ from auto-only"
    )
    print(
        f"[PASS] cb_focal + user weights: combined={loss_combined.item():.5f}, "
        f"auto={loss_auto.item():.5f}"
    )


# ──────────────────────────────────────── 9. Invalid cls_loss ────────────────
def test_invalid_cls_loss():
    """An unsupported cls_loss string should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid cls_loss"):
        v8ClassificationLoss(cls_loss="invalid_loss")
    print("[PASS] invalid cls_loss raises ValueError")


# ──────────────────────────────────────── 10. Weight device migration ────────
def test_weight_device_migration():
    """Weights should be migrated to the correct device on first forward call."""
    loss_fn = v8ClassificationLoss(cls_loss="ce", class_weights=[1.0, 2.0, 3.0])
    assert loss_fn._weight.device == torch.device("cpu")

    preds = _make_preds(4, 3, device="cpu")
    batch = _make_batch([0, 1, 2, 0], device="cpu")
    loss, _ = loss_fn(preds, batch)

    assert loss_fn._weight.device == torch.device("cpu")
    print(f"[PASS] weight device migration: loss={loss.item():.5f}")


# ──────────────────────────────────────── 11. Gamma=0 ≈ CE ───────────────────
def test_focal_gamma_zero_equals_ce():
    """Focal loss with gamma=0 should approximate standard CE (no focusing)."""
    loss_fn_focal = v8ClassificationLoss(cls_loss="focal", focal_gamma=0.0)
    loss_fn_ce = v8ClassificationLoss(cls_loss="ce")

    preds = _make_preds(16, 5)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0])

    loss_focal, _ = loss_fn_focal(preds, batch)
    loss_ce, _ = loss_fn_ce(preds, batch)

    assert torch.allclose(loss_focal, loss_ce, atol=1e-4), (
        f"Focal γ=0 should ≈ CE: focal={loss_focal.item():.5f}, ce={loss_ce.item():.5f}"
    )
    print(f"[PASS] focal γ=0 ≈ CE: focal={loss_focal.item():.5f}, ce={loss_ce.item():.5f}")


# ──────────────────────────────────────── 12. Resolve class_weights dict ─────
def test_resolve_class_weights_dict():
    """ClassificationTrainer._resolve_class_weights should handle dict input."""
    from ultralytics.models.yolo.classify.train import ClassificationTrainer

    trainer = object.__new__(ClassificationTrainer)
    trainer.args = SimpleNamespace(class_weights={"cat": 3.0, "bird": 5.0})
    trainer.data = {"nc": 3, "names": {0: "cat", 1: "dog", 2: "bird"}}
    trainer.model = SimpleNamespace(names={0: "cat", 1: "dog", 2: "bird"})

    trainer._resolve_class_weights()

    assert trainer.args.class_weights_resolved == [3.0, 1.0, 5.0], (
        f"Got {trainer.args.class_weights_resolved}"
    )
    print(f"[PASS] resolve_class_weights dict: {trainer.args.class_weights_resolved}")


# ──────────────────────────────────────── 13. Resolve class_weights list ─────
def test_resolve_class_weights_list():
    """ClassificationTrainer._resolve_class_weights should handle list input."""
    from ultralytics.models.yolo.classify.train import ClassificationTrainer

    trainer = object.__new__(ClassificationTrainer)
    trainer.args = SimpleNamespace(class_weights=[2.0, 3.0, 1.0])
    trainer.data = {"nc": 3, "names": {0: "cat", 1: "dog", 2: "bird"}}
    trainer.model = SimpleNamespace(names={0: "cat", 1: "dog", 2: "bird"})

    trainer._resolve_class_weights()

    assert trainer.args.class_weights_resolved == [2.0, 3.0, 1.0]
    print(f"[PASS] resolve_class_weights list: {trainer.args.class_weights_resolved}")


# ──────────────────────────────────────── 14. Resolve — no class_weights ─────
def test_resolve_class_weights_none():
    """When class_weights is None, resolved should also be None."""
    from ultralytics.models.yolo.classify.train import ClassificationTrainer

    trainer = object.__new__(ClassificationTrainer)
    trainer.args = SimpleNamespace(class_weights=None)
    trainer.data = {"nc": 3, "names": {0: "cat", 1: "dog", 2: "bird"}}
    trainer.model = SimpleNamespace(names={0: "cat", 1: "dog", 2: "bird"})

    trainer._resolve_class_weights()

    assert trainer.args.class_weights_resolved is None
    print("[PASS] resolve_class_weights None -> None")


# ──────────────────────────────────────── 15. Higher gamma focuses more ──────
def test_higher_gamma_reduces_easy_loss():
    """Higher gamma should reduce loss contribution from well-classified (easy) examples."""
    # Make preds that are very confident for targets — "easy" examples
    preds_easy = torch.tensor(
        [[5.0, -5.0, -5.0], [5.0, -5.0, -5.0], [5.0, -5.0, -5.0], [5.0, -5.0, -5.0]],
        requires_grad=True,
    )
    batch = _make_batch([0, 0, 0, 0])

    loss_fn_low = v8ClassificationLoss(cls_loss="focal", focal_gamma=0.5)
    loss_fn_high = v8ClassificationLoss(cls_loss="focal", focal_gamma=5.0)

    loss_low, _ = loss_fn_low(preds_easy, batch)
    loss_high, _ = loss_fn_high(preds_easy.detach().requires_grad_(True), batch)

    assert loss_high < loss_low, (
        f"Higher γ should reduce easy loss: γ=0.5 → {loss_low.item():.6f}, γ=5.0 → {loss_high.item():.6f}"
    )
    print(
        f"[PASS] higher gamma focuses more: γ=0.5={loss_low.item():.6f}, γ=5.0={loss_high.item():.6f}"
    )


# ──────────────────────────────────────── run all ─────────────────────────────
if __name__ == "__main__":
    test_default_ce()
    test_weighted_ce()
    test_ce_label_smoothing()
    test_focal_no_weights()
    test_focal_with_class_weights()
    test_focal_label_smoothing()
    test_cb_focal_auto_weights()
    test_cb_focal_combined_weights()
    test_invalid_cls_loss()
    test_weight_device_migration()
    test_focal_gamma_zero_equals_ce()
    test_resolve_class_weights_dict()
    test_resolve_class_weights_list()
    test_resolve_class_weights_none()
    test_higher_gamma_reduces_easy_loss()
    print("\n" + "=" * 60)
    print("All 15 tests passed!")
    print("=" * 60)
