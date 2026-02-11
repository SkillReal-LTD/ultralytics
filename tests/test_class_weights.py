# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for per-class loss weighting (class_weights) feature."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure we import from local ultralytics, not installed package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from ultralytics.utils.metrics import DetMetrics, Metric, OBBMetrics, PoseMetrics, SegmentMetrics

# ─── Metric-level tests ──────────────────────────────────────────────────────


def test_metric_class_weights_init():
    """Metric stores class_weights as a numpy array when provided."""
    print("Testing Metric.__init__ with class_weights...")
    m = Metric(class_weights=[1.0, 5.0, 1.0])
    assert m.class_weights is not None
    assert len(m.class_weights) == 3
    assert m.class_weights[1] == 5.0
    print("  ✓ Metric init with class_weights OK")


def test_metric_no_class_weights():
    """Metric defaults to None when class_weights is omitted."""
    print("Testing Metric.__init__ without class_weights...")
    m = Metric()
    assert m.class_weights is None
    print("  ✓ Metric default (no weights) OK")


def test_metric_weighted_mean_results():
    """Mean_results() uses weighted averages when class_weights is set."""
    print("Testing Metric.mean_results() weighted path...")
    cw = [1.0, 5.0, 1.0]
    m = Metric(class_weights=cw)
    m.p = np.array([0.8, 0.9, 0.7])
    m.r = np.array([0.6, 0.95, 0.5])
    m.all_ap = np.random.rand(3, 10)
    m.ap_class_index = np.array([0, 1, 2])
    m.nc = 3

    results = m.mean_results()

    expected_mr = float(np.average(m.r, weights=cw))
    expected_mp = float(np.average(m.p, weights=cw))
    assert abs(results[0] - expected_mp) < 1e-10, f"Expected weighted mp {expected_mp}, got {results[0]}"
    assert abs(results[1] - expected_mr) < 1e-10, f"Expected weighted mr {expected_mr}, got {results[1]}"
    print(f"  Simple mean recall:   {np.mean(m.r):.4f}")
    print(f"  Weighted mean recall: {expected_mr:.4f}")
    print("  ✓ Weighted mean_results OK")


def test_metric_unweighted_mean_results():
    """Mean_results() uses simple averages when class_weights is None."""
    print("Testing Metric.mean_results() unweighted path...")
    m = Metric()
    m.p = np.array([0.8, 0.9, 0.7])
    m.r = np.array([0.6, 0.95, 0.5])
    m.all_ap = np.random.rand(3, 10)
    m.ap_class_index = np.array([0, 1, 2])
    m.nc = 3

    results = m.mean_results()
    assert abs(results[0] - np.mean(m.p)) < 1e-10
    assert abs(results[1] - np.mean(m.r)) < 1e-10
    print("  ✓ Unweighted mean_results still works")


def test_metric_fitness_uses_weighted():
    """Fitness() incorporates weighted mean_results when class_weights present."""
    print("Testing Metric.fitness() with class_weights...")
    cw = [1.0, 5.0, 1.0]
    m = Metric(class_weights=cw, fitness_weight=[0.0, 1.0, 0.0, 0.0])  # fitness = recall only
    m.p = np.array([0.8, 0.9, 0.7])
    m.r = np.array([0.6, 0.95, 0.5])
    m.all_ap = np.random.rand(3, 10)
    m.ap_class_index = np.array([0, 1, 2])
    m.nc = 3

    fitness = m.fitness()
    expected = float(np.average(m.r, weights=cw))
    assert abs(fitness - expected) < 1e-10, f"Expected fitness {expected}, got {fitness}"
    print(f"  Fitness (weighted recall only): {fitness:.4f}")
    print("  ✓ fitness with class_weights OK")


def test_metric_partial_class_detection():
    """Weighted mean works when not all classes are detected (ap_class_index subset)."""
    print("Testing partial class detection with class_weights...")
    cw = [1.0, 5.0, 1.0]  # 3 classes total
    m = Metric(class_weights=cw)
    # Only classes 0 and 2 detected (class 1 missing from val set)
    m.p = np.array([0.8, 0.7])
    m.r = np.array([0.6, 0.5])
    m.all_ap = np.random.rand(2, 10)
    m.ap_class_index = np.array([0, 2])  # indices into the full nc
    m.nc = 3

    results = m.mean_results()
    expected_weights = np.array([cw[0], cw[2]])  # [1.0, 1.0]
    expected_mr = float(np.average(m.r, weights=expected_weights))
    assert abs(results[1] - expected_mr) < 1e-10
    print("  ✓ Partial detection weighted mean OK")


# ─── Metrics class propagation tests ─────────────────────────────────────────


def test_det_metrics_propagation():
    """DetMetrics passes class_weights to its Metric instance."""
    print("Testing DetMetrics class_weights propagation...")
    dm = DetMetrics(class_weights=[1.0, 2.0, 3.0])
    assert dm.box.class_weights is not None
    assert list(dm.box.class_weights) == [1.0, 2.0, 3.0]
    print("  ✓ DetMetrics propagation OK")


def test_pose_metrics_propagation():
    """PoseMetrics passes class_weights to both box and pose Metric instances."""
    print("Testing PoseMetrics class_weights propagation...")
    pm = PoseMetrics(class_weights=[1.0, 2.0])
    assert pm.box.class_weights is not None
    assert pm.pose.class_weights is not None
    assert list(pm.box.class_weights) == [1.0, 2.0]
    assert list(pm.pose.class_weights) == [1.0, 2.0]
    print("  ✓ PoseMetrics propagation OK")


def test_segment_metrics_propagation():
    """SegmentMetrics passes class_weights to both box and seg Metric instances."""
    print("Testing SegmentMetrics class_weights propagation...")
    sm = SegmentMetrics(class_weights=[1.0, 2.0])
    assert sm.box.class_weights is not None
    assert sm.seg.class_weights is not None
    print("  ✓ SegmentMetrics propagation OK")


def test_obb_metrics_propagation():
    """OBBMetrics passes class_weights to its Metric instance."""
    print("Testing OBBMetrics class_weights propagation...")
    om = OBBMetrics(class_weights=[1.0, 2.0])
    assert om.box.class_weights is not None
    print("  ✓ OBBMetrics propagation OK")


def test_no_class_weights_unchanged():
    """All Metrics classes work identically when class_weights is None (no-op)."""
    print("Testing all Metrics classes without class_weights (backward compat)...")
    dm = DetMetrics()
    assert dm.box.class_weights is None
    pm = PoseMetrics()
    assert pm.box.class_weights is None and pm.pose.class_weights is None
    sm = SegmentMetrics()
    assert sm.box.class_weights is None and sm.seg.class_weights is None
    om = OBBMetrics()
    assert om.box.class_weights is None
    print("  ✓ All backward-compatible (no class_weights) OK")


# ─── Loss tensor tests ───────────────────────────────────────────────────────


def test_loss_class_weights_tensor():
    """V8DetectionLoss stores class_weights tensor when model.args has class_weights_resolved."""
    print("Testing v8DetectionLoss class_weights tensor init...")
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    # Build a minimal mock model
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])  # device = cpu
    model.args = SimpleNamespace(
        box=7.5,
        cls=0.5,
        dfl=1.5,
        class_weights_resolved=[1.0, 5.0, 1.0],
    )
    m = MagicMock()
    m.stride = torch.tensor([8.0, 16.0, 32.0])
    m.nc = 3
    m.reg_max = 16
    model.model.__getitem__ = lambda self, idx: m

    from ultralytics.utils.loss import v8DetectionLoss

    loss_fn = v8DetectionLoss(model)
    assert loss_fn.class_weights is not None
    assert loss_fn.class_weights.shape == (3,)
    assert loss_fn.class_weights[1].item() == 5.0
    print(f"  class_weights tensor: {loss_fn.class_weights}")
    print("  ✓ Loss class_weights tensor OK")


def test_loss_no_class_weights():
    """V8DetectionLoss sets class_weights to None when not configured."""
    print("Testing v8DetectionLoss without class_weights...")
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    m = MagicMock()
    m.stride = torch.tensor([8.0, 16.0, 32.0])
    m.nc = 3
    m.reg_max = 16
    model.model.__getitem__ = lambda self, idx: m

    from ultralytics.utils.loss import v8DetectionLoss

    loss_fn = v8DetectionLoss(model)
    assert loss_fn.class_weights is None
    print("  ✓ Loss no class_weights (None) OK")


# ─── Trainer resolve tests ────────────────────────────────────────────────────


def test_trainer_resolve_dict():
    """DetectionTrainer._resolve_class_weights resolves a name->weight dict to index list."""
    print("Testing _resolve_class_weights with dict...")
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    trainer = MagicMock()
    trainer.data = {"nc": 3, "names": {0: "cat", 1: "bird", 2: "dog"}}
    trainer.args = SimpleNamespace(class_weights={"dog": 5.0, "cat": 2.0})

    from ultralytics.models.yolo.detect.train import DetectionTrainer

    DetectionTrainer._resolve_class_weights(trainer)

    assert trainer.args.class_weights_resolved == [2.0, 1.0, 5.0]
    print(f"  Resolved: {trainer.args.class_weights_resolved}")
    print("  ✓ Dict resolve OK")


def test_trainer_resolve_list():
    """DetectionTrainer._resolve_class_weights accepts a list of per-class weights."""
    print("Testing _resolve_class_weights with list...")
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    trainer = MagicMock()
    trainer.data = {"nc": 3, "names": {0: "cat", 1: "bird", 2: "dog"}}
    trainer.args = SimpleNamespace(class_weights=[2.0, 1.0, 5.0])

    from ultralytics.models.yolo.detect.train import DetectionTrainer

    DetectionTrainer._resolve_class_weights(trainer)

    assert trainer.args.class_weights_resolved == [2.0, 1.0, 5.0]
    print("  ✓ List resolve OK")


def test_trainer_resolve_none():
    """DetectionTrainer._resolve_class_weights sets None when class_weights not provided."""
    print("Testing _resolve_class_weights with None...")
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    trainer = MagicMock()
    trainer.data = {"nc": 3, "names": {0: "cat", 1: "bird", 2: "dog"}}
    trainer.args = SimpleNamespace()  # no class_weights attribute

    from ultralytics.models.yolo.detect.train import DetectionTrainer

    DetectionTrainer._resolve_class_weights(trainer)

    assert trainer.args.class_weights_resolved is None
    print("  ✓ None resolve OK")


def test_trainer_resolve_missing_class_warns():
    """DetectionTrainer._resolve_class_weights warns for class names not in the dataset."""
    print("Testing _resolve_class_weights with unknown class name...")
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    trainer = MagicMock()
    trainer.data = {"nc": 2, "names": {0: "cat", 1: "dog"}}
    trainer.args = SimpleNamespace(class_weights={"cat": 2.0, "unicorn": 10.0})

    import ultralytics.models.yolo.detect.train as train_mod

    with patch.object(train_mod, "LOGGER") as mock_logger:
        from ultralytics.models.yolo.detect.train import DetectionTrainer

        DetectionTrainer._resolve_class_weights(trainer)
        mock_logger.warning.assert_called_once()

    assert trainer.args.class_weights_resolved == [2.0, 1.0]  # unicorn ignored, dog defaults to 1.0
    print("  ✓ Unknown class warning OK")


def test_trainer_resolve_wrong_list_length():
    """DetectionTrainer._resolve_class_weights raises ValueError for wrong-length list."""
    print("Testing _resolve_class_weights with wrong list length...")
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    trainer = MagicMock()
    trainer.data = {"nc": 3, "names": {0: "cat", 1: "bird", 2: "dog"}}
    trainer.args = SimpleNamespace(class_weights=[1.0, 2.0])  # only 2 but nc=3

    from ultralytics.models.yolo.detect.train import DetectionTrainer

    try:
        DetectionTrainer._resolve_class_weights(trainer)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✓ Wrong list length raises ValueError OK")


# ─── Run all tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {test_fn.__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        raise SystemExit(1)
    print("All class_weights tests passed! ✓")
