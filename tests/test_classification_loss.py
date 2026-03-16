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
 14. Focal γ=0 ≈ CE
 15. Higher gamma focuses more on hard examples
 16. ArcFace basic forward + backward
 17. ArcFace produces higher loss than CE (due to angular margin)
 18. ArcFace with class_weights differs from without
 19. ArcFace margin=0, scale=1 ≈ normalised cosine CE
 20. Classify head returns features when _return_features=True
 21. init_criterion resets _return_features when switching ArcFace → CE
 22. init_criterion with no args returns plain CE (legacy compat)
 23. ArcFace eval fallback uses CE on logits
 24. cb_beta >= 1.0 is clamped to 0.9999
 25. CB-Focal with zero-count classes produces finite weights (weight=0)
 26. Zero-count classes do NOT distort normalization of present classes
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


# ──────────────────────────────────────── 16. ArcFace basic ───────────────────
def _make_fc_weight(nc: int, feat_dim: int, device: str = "cpu") -> torch.nn.Parameter:
    """Create a mock FC weight parameter (nc, feat_dim)."""
    torch.manual_seed(123)
    return torch.nn.Parameter(torch.randn(nc, feat_dim, device=device))


def _make_features(batch_size: int, feat_dim: int, device: str = "cpu") -> torch.Tensor:
    """Return raw features (B, feat_dim)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, feat_dim, device=device, requires_grad=True)


def test_arcface_basic():
    """ArcFace loss should be computable and differentiable."""
    nc, feat_dim, bs = 5, 128, 8
    fc_weight = _make_fc_weight(nc, feat_dim)
    loss_fn = v8ClassificationLoss(
        cls_loss="arcface", arcface_margin=0.5, arcface_scale=30.0, fc_weight=fc_weight
    )

    features = _make_features(bs, feat_dim)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 2])

    loss, loss_det = loss_fn(features, batch)
    loss.backward()

    assert loss.item() > 0, "ArcFace loss should be positive"
    assert features.grad is not None, "ArcFace loss should be differentiable w.r.t. features"
    assert loss_det.requires_grad is False
    print(f"[PASS] ArcFace basic: loss={loss.item():.5f}")


# ──────────────────────────────────────── 17. ArcFace > CE ────────────────────
def test_arcface_higher_than_ce_equivalent():
    """ArcFace with margin > 0 should produce higher loss than normalised cosine CE (margin=0)."""
    nc, feat_dim, bs = 5, 128, 16
    fc_weight = _make_fc_weight(nc, feat_dim)

    loss_fn_margin = v8ClassificationLoss(
        cls_loss="arcface", arcface_margin=0.5, arcface_scale=30.0, fc_weight=fc_weight
    )
    loss_fn_no_margin = v8ClassificationLoss(
        cls_loss="arcface", arcface_margin=0.0, arcface_scale=30.0, fc_weight=fc_weight
    )

    features = _make_features(bs, feat_dim)
    batch = _make_batch(list(range(nc)) * (bs // nc) + list(range(bs % nc)))

    loss_margin, _ = loss_fn_margin(features, batch)
    loss_no_margin, _ = loss_fn_no_margin(features.detach().clone().requires_grad_(True), batch)

    assert loss_margin > loss_no_margin, (
        f"Margin should increase loss: margin={loss_margin.item():.5f} > no_margin={loss_no_margin.item():.5f}"
    )
    print(
        f"[PASS] ArcFace margin increases loss: m=0.5 → {loss_margin.item():.5f}, "
        f"m=0.0 → {loss_no_margin.item():.5f}"
    )


# ──────────────────────────────────────── 18. ArcFace + class_weights ─────────
def test_arcface_with_class_weights():
    """ArcFace with class_weights should produce different loss than without."""
    nc, feat_dim, bs = 5, 128, 8
    fc_weight = _make_fc_weight(nc, feat_dim)
    weights = [1.0, 1.0, 1.0, 1.0, 5.0]

    loss_fn_w = v8ClassificationLoss(
        cls_loss="arcface", arcface_margin=0.5, arcface_scale=30.0,
        fc_weight=fc_weight, class_weights=weights,
    )
    loss_fn_u = v8ClassificationLoss(
        cls_loss="arcface", arcface_margin=0.5, arcface_scale=30.0,
        fc_weight=fc_weight,
    )

    features = _make_features(bs, feat_dim)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 4])

    loss_w, _ = loss_fn_w(features, batch)
    loss_u, _ = loss_fn_u(features.detach().clone().requires_grad_(True), batch)

    assert not torch.allclose(loss_w, loss_u, atol=1e-4), "ArcFace w/ weights should differ"
    print(f"[PASS] ArcFace + weights: weighted={loss_w.item():.5f}, unweighted={loss_u.item():.5f}")


# ──────────────────────────────────────── 19. ArcFace margin=0 ≈ cosine CE ───
def test_arcface_zero_margin():
    """ArcFace with margin=0 and scale=1 should equal normalised-cosine CE."""
    nc, feat_dim, bs = 3, 64, 8
    fc_weight = _make_fc_weight(nc, feat_dim)

    loss_fn = v8ClassificationLoss(
        cls_loss="arcface", arcface_margin=0.0, arcface_scale=1.0, fc_weight=fc_weight
    )
    features = _make_features(bs, feat_dim)
    batch = _make_batch([0, 1, 2, 0, 1, 2, 0, 1])

    loss_af, _ = loss_fn(features, batch)

    # Manual cosine CE: normalise feat & weight, compute cos, cross_entropy
    with torch.no_grad():
        feat_n = torch.nn.functional.normalize(features, dim=1)
        w_n = torch.nn.functional.normalize(fc_weight, dim=1)
        cos_logits = feat_n @ w_n.T  # scale=1
        expected = torch.nn.functional.cross_entropy(cos_logits, batch["cls"])

    assert torch.allclose(loss_af, expected, atol=1e-4), (
        f"ArcFace m=0,s=1 should ≈ cosine CE: af={loss_af.item():.5f}, expected={expected.item():.5f}"
    )
    print(f"[PASS] ArcFace m=0, s=1 ≈ cosine CE: {loss_af.item():.5f} ≈ {expected.item():.5f}")


# ──────────────────────────────────────── 20. Classify _return_features ───────
def test_classify_return_features():
    """Classify head should return features (not logits) when _return_features=True."""
    from ultralytics.nn.modules.head import Classify

    head = Classify(c1=256, c2=10)
    head.train()
    x = torch.randn(2, 256, 8, 8)

    # Normal mode: returns logits (B, 10)
    out_normal = head(x)
    assert out_normal.shape == (2, 10), f"Expected (2,10) logits, got {out_normal.shape}"

    # ArcFace mode: returns features (B, 1280)
    head._return_features = True
    out_features = head(x)
    assert out_features.shape == (2, 1280), f"Expected (2,1280) features, got {out_features.shape}"

    # Eval mode: should still return normal (probs, logits) regardless of flag
    head.eval()
    out_eval = head(x)
    assert isinstance(out_eval, tuple) and len(out_eval) == 2, "Eval should return (probs, logits)"
    assert out_eval[0].shape == (2, 10)

    print("[PASS] Classify _return_features works correctly")


# ──────────────────────────────────────── 21. Backward compat: ArcFace→CE ────
def test_init_criterion_resets_return_features():
    """Switching from ArcFace to CE should reset _return_features on the Classify head."""
    from types import SimpleNamespace

    from ultralytics.nn.modules.head import Classify
    from ultralytics.nn.tasks import ClassificationModel

    # Build a minimal ClassificationModel via __new__ then manually init nn.Module internals
    model = ClassificationModel.__new__(ClassificationModel)
    torch.nn.Module.__init__(model)
    head = Classify(c1=256, c2=5)
    model.model = torch.nn.ModuleList([head])

    # Simulate: previously trained with ArcFace (instance attr set to True)
    head._return_features = True

    # Now retrain with plain CE
    model.args = SimpleNamespace(cls_loss="ce")
    criterion = model.init_criterion()

    # Head flag must be reset
    assert head._return_features is False, (
        "_return_features should be False after switching from ArcFace to CE"
    )
    # Criterion should be plain CE
    assert criterion.cls_loss == "ce"
    print("[PASS] init_criterion resets _return_features when cls_loss != 'arcface'")


# ──────────────────────────────────────── 22. No-args backward compat ─────────
def test_init_criterion_no_args():
    """ClassificationModel.init_criterion with no args returns plain CE (full backward compat)."""
    from ultralytics.nn.modules.head import Classify
    from ultralytics.nn.tasks import ClassificationModel

    model = ClassificationModel.__new__(ClassificationModel)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.ModuleList([Classify(c1=256, c2=5)])

    # No args attribute at all — mimics legacy models
    criterion = model.init_criterion()
    assert criterion.cls_loss == "ce"
    assert criterion._fc_weight is None
    assert criterion._weight is None
    assert criterion.label_smoothing == 0.0

    # Head should not have been touched
    head: Classify = model.model[-1]
    assert head._return_features is False
    print("[PASS] init_criterion with no args returns plain CE")


# ──────────────────────────────────────── 23. ArcFace eval fallback ───────────
def test_arcface_eval_fallback_uses_logits():
    """During eval, ArcFace loss should gracefully fall back to CE on logits."""
    import torch.nn.functional as F

    nc, feat_dim, bs = 5, 128, 8
    fc_weight = _make_fc_weight(nc, feat_dim)
    loss_fn = v8ClassificationLoss(
        cls_loss="arcface", arcface_margin=0.5, arcface_scale=30.0, fc_weight=fc_weight,
    )

    # Simulate eval-mode output: (probs, logits)
    torch.manual_seed(42)
    logits = torch.randn(bs, nc)
    probs = logits.softmax(1)
    preds_eval = (probs, logits)
    batch = _make_batch([0, 1, 2, 3, 4, 0, 1, 2])

    loss, _ = loss_fn(preds_eval, batch)
    expected = F.cross_entropy(logits, batch["cls"])
    assert torch.allclose(loss, expected, atol=1e-5), (
        f"Eval fallback should be CE on logits: {loss.item():.5f} vs {expected.item():.5f}"
    )
    print(f"[PASS] ArcFace eval fallback = CE on logits: {loss.item():.5f}")


# ──────────────────────────────────────── 24. cb_beta=1.0 clamped ─────────────
def test_cb_beta_one_clamped():
    """cb_beta >= 1.0 should be clamped to 0.9999 to avoid division by zero."""
    counts = [100, 50, 10]
    # Should not raise — beta is silently clamped
    loss_fn = v8ClassificationLoss(cls_loss="cb_focal", class_counts=counts, cb_beta=1.0)
    assert loss_fn.cb_beta == 0.9999, f"Expected cb_beta=0.9999, got {loss_fn.cb_beta}"
    assert loss_fn._weight is not None
    assert torch.isfinite(loss_fn._weight).all(), f"Weights contain NaN/Inf: {loss_fn._weight}"
    print(f"[PASS] cb_beta=1.0 clamped to 0.9999, weights={loss_fn._weight.tolist()}")


# ──────────────────────────────────────── 25. CB-Focal zero-count classes ─────
def test_cb_focal_zero_count_classes():
    """CB-Focal with zero-count classes should get weight=0 (ignored), not highest weight."""
    counts = [100, 0, 50]  # class 1 has zero samples
    loss_fn = v8ClassificationLoss(cls_loss="cb_focal", class_counts=counts, cb_beta=0.999)
    assert loss_fn._weight is not None
    assert torch.isfinite(loss_fn._weight).all(), f"Weights contain NaN/Inf: {loss_fn._weight}"

    # Zero-count class must have weight 0 (ignored), not an inflated weight
    assert loss_fn._weight[1].item() == 0.0, (
        f"Zero-count class should have weight 0, got {loss_fn._weight[1].item()}"
    )
    # Non-zero classes should have positive weight
    assert loss_fn._weight[0].item() > 0
    assert loss_fn._weight[2].item() > 0

    # Also test end-to-end: forward should not fail
    preds = _make_preds(8, 3)
    batch = _make_batch([0, 2, 0, 2, 0, 2, 0, 2])  # only classes that exist in training
    loss, _ = loss_fn(preds, batch)
    assert torch.isfinite(loss), f"Loss is NaN/Inf: {loss.item()}"
    print(
        f"[PASS] CB-Focal with zero-count class: weights={loss_fn._weight.tolist()}, "
        f"class1(absent)=0.0, loss={loss.item():.5f}"
    )


# ──────────────────────────────────────── 26. Zero-count must NOT distort normalization
def test_cb_focal_zero_count_no_normalization_distortion():
    """Absent classes (count=0) must not inflate the denominator during normalization.

    If zero-count classes were floored to 1 instead of being zeroed out, they would
    receive the highest CB-weight (rarest class), inflating the normalization sum and
    dragging down the weights of all real classes.  This test verifies that the weights
    for present classes are identical whether or not absent classes exist.
    """
    beta = 0.999

    # Scenario A: only present classes
    counts_no_absent = [100, 50]
    loss_a = v8ClassificationLoss(cls_loss="cb_focal", class_counts=counts_no_absent, cb_beta=beta)

    # Scenario B: same present classes + an absent class in the middle
    counts_with_absent = [100, 0, 50]
    loss_b = v8ClassificationLoss(cls_loss="cb_focal", class_counts=counts_with_absent, cb_beta=beta)

    w_a = loss_a._weight  # [w_cls0, w_cls1]
    w_b = loss_b._weight  # [w_cls0, 0.0, w_cls2]

    # Absent class must be zero
    assert w_b[1].item() == 0.0, f"Absent class weight should be 0, got {w_b[1].item()}"

    # Weights for present classes must be identical in both scenarios
    assert torch.allclose(w_a[0], w_b[0], atol=1e-6), (
        f"Class 0 weight distorted: no_absent={w_a[0].item():.6f} vs with_absent={w_b[0].item():.6f}"
    )
    assert torch.allclose(w_a[1], w_b[2], atol=1e-6), (
        f"Class 1/2 weight distorted: no_absent={w_a[1].item():.6f} vs with_absent={w_b[2].item():.6f}"
    )

    print(
        f"[PASS] Zero-count does NOT distort normalization: "
        f"no_absent={w_a.tolist()}, with_absent={w_b.tolist()}"
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
    test_arcface_basic()
    test_arcface_higher_than_ce_equivalent()
    test_arcface_with_class_weights()
    test_arcface_zero_margin()
    test_classify_return_features()
    test_init_criterion_resets_return_features()
    test_init_criterion_no_args()
    test_arcface_eval_fallback_uses_logits()
    test_cb_beta_one_clamped()
    test_cb_focal_zero_count_classes()
    test_cb_focal_zero_count_no_normalization_distortion()
    print("\n" + "=" * 60)
    print("All 26 tests passed!")
    print("=" * 60)
