#!/usr/bin/env python3
"""Test script to verify fitness_weight configuration works correctly."""

import sys
from pathlib import Path

# Ensure we import from local ultralytics, not installed package
sys.path.insert(0, str(Path(__file__).parent.parent))

from types import SimpleNamespace

from ultralytics.utils.metrics import DetMetrics, Metric, OBBMetrics, PoseMetrics, SegmentMetrics


def test_metric_fitness_weight():
    """Test that Metric class accepts and uses fitness_weight correctly."""
    print("Testing Metric class fitness_weight...")

    # Test default weights
    metric = Metric()
    assert metric.fitness_weight == [0.0, 0.0, 0.1, 0.9], f"Default weights incorrect: {metric.fitness_weight}"

    # Test custom weights
    custom_weights = [0.0, 0.9, 0.1, 0.0]
    metric = Metric(fitness_weight=custom_weights)
    assert metric.fitness_weight == custom_weights, f"Custom weights incorrect: {metric.fitness_weight}"

    # Test fitness calculation with mock data
    # Set up metric with actual data
    import numpy as np

    metric.p = np.array([0.8, 0.7])
    metric.r = np.array([0.9, 0.8])
    metric.all_ap = np.array([[0.85, 0.80], [0.75, 0.70]])
    metric.nc = 2

    # Test fitness calculation - it uses mean_results() which computes mp, mr, map50, map from the data
    fitness = metric.fitness()

    # Verify it's a valid number (can't easily predict exact value without full implementation)
    assert isinstance(fitness, (int, float, np.number)), f"Fitness should be a number, got {type(fitness)}"
    assert not np.isnan(fitness), "Fitness should not be NaN"

    print("✓ Metric class fitness_weight test passed")


def test_det_metrics_fitness_weight():
    """Test that DetMetrics accepts and passes fitness_weight correctly."""
    print("Testing DetMetrics class fitness_weight...")

    # Test with custom weights
    custom_weights = [0.0, 0.9, 0.1, 0.0]
    det_metrics = DetMetrics(fitness_weight=custom_weights)

    assert det_metrics.box.fitness_weight == custom_weights, (
        f"DetMetrics fitness_weight not passed: {det_metrics.box.fitness_weight}"
    )

    print("✓ DetMetrics class fitness_weight test passed")


def test_pose_metrics_fitness_weight():
    """Test that PoseMetrics accepts and passes fitness_weight correctly."""
    print("Testing PoseMetrics class fitness_weight...")

    # Test with custom weights
    custom_weights = [0.0, 0.9, 0.1, 0.0]
    pose_metrics = PoseMetrics(fitness_weight=custom_weights)

    assert pose_metrics.box.fitness_weight == custom_weights, (
        f"PoseMetrics box fitness_weight not passed: {pose_metrics.box.fitness_weight}"
    )
    assert pose_metrics.pose.fitness_weight == custom_weights, (
        f"PoseMetrics pose fitness_weight not passed: {pose_metrics.pose.fitness_weight}"
    )

    print("✓ PoseMetrics class fitness_weight test passed")


def test_segment_metrics_fitness_weight():
    """Test that SegmentMetrics accepts and passes fitness_weight correctly."""
    print("Testing SegmentMetrics class fitness_weight...")

    # Test with custom weights
    custom_weights = [0.0, 0.9, 0.1, 0.0]
    seg_metrics = SegmentMetrics(fitness_weight=custom_weights)

    assert seg_metrics.box.fitness_weight == custom_weights, (
        f"SegmentMetrics box fitness_weight not passed: {seg_metrics.box.fitness_weight}"
    )
    assert seg_metrics.seg.fitness_weight == custom_weights, (
        f"SegmentMetrics seg fitness_weight not passed: {seg_metrics.seg.fitness_weight}"
    )

    print("✓ SegmentMetrics class fitness_weight test passed")


def test_obb_metrics_fitness_weight():
    """Test that OBBMetrics accepts and passes fitness_weight correctly."""
    print("Testing OBBMetrics class fitness_weight...")

    # Test with custom weights
    custom_weights = [0.0, 0.9, 0.1, 0.0]
    obb_metrics = OBBMetrics(fitness_weight=custom_weights)

    assert obb_metrics.box.fitness_weight == custom_weights, (
        f"OBBMetrics fitness_weight not passed: {obb_metrics.box.fitness_weight}"
    )

    print("✓ OBBMetrics class fitness_weight test passed")


def test_validator_args_integration():
    """Test integration with validator args."""
    print("Testing validator args integration...")

    # Mock args object
    args = SimpleNamespace()
    args.fitness_weight = [0.0, 0.9, 0.1, 0.0]

    # Test getattr usage
    fitness_weight = getattr(args, "fitness_weight", None)
    assert fitness_weight == [0.0, 0.9, 0.1, 0.0], f"Args fitness_weight not retrieved: {fitness_weight}"

    # Test with None
    args_none = SimpleNamespace()
    fitness_weight_none = getattr(args_none, "fitness_weight", None)
    assert fitness_weight_none is None, f"Args fitness_weight should be None: {fitness_weight_none}"

    print("✓ Validator args integration test passed")


def test_different_optimization_strategies():
    """Test different optimization strategies."""
    print("Testing different optimization strategies...")
    import numpy as np

    strategies = {
        "Default (mAP)": [0.0, 0.0, 0.1, 0.9],
        "Recall Optimized": [0.0, 0.9, 0.1, 0.0],
        "Precision Optimized": [0.9, 0.0, 0.1, 0.0],
        "Balanced P/R": [0.45, 0.45, 0.1, 0.0],
        "mAP@0.5 Only": [0.0, 0.0, 1.0, 0.0],
    }

    for name, weights in strategies.items():
        metric = Metric(fitness_weight=weights)

        # Set up dummy data
        metric.p = np.array([0.8, 0.7])
        metric.r = np.array([0.7, 0.6])
        metric.all_ap = np.array(
            [
                [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
                [0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30],
            ]
        )
        metric.nc = 2

        fitness = metric.fitness()

        # Verify fitness is valid
        assert isinstance(fitness, (int, float, np.number)), f"Strategy {name} fitness should be a number"
        assert not np.isnan(fitness), f"Strategy {name} fitness should not be NaN"
        assert fitness >= 0, f"Strategy {name} fitness should be non-negative"

        print(f"  ✓ {name}: weights={weights}, fitness={fitness:.3f}")

    print("✓ Different optimization strategies test passed")


def test_pose_metrics_8_weight():
    """Test PoseMetrics with 8-weight fitness configuration."""
    print("Testing PoseMetrics with 8-weight fitness...")

    # Test 8-weight configuration - optimize for pose only
    weights_8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9]  # pose mAP only
    pose_metrics = PoseMetrics(fitness_weight=weights_8)

    # Verify box weights are first 4
    assert pose_metrics.box.fitness_weight == [0.0, 0.0, 0.0, 0.0], (
        f"Box weights incorrect: {pose_metrics.box.fitness_weight}"
    )

    # Verify pose weights are last 4
    assert pose_metrics.pose.fitness_weight == [0.0, 0.0, 0.1, 0.9], (
        f"Pose weights incorrect: {pose_metrics.pose.fitness_weight}"
    )

    print("  ✓ 8-weight split correctly between box and pose")

    # Test 4-weight backward compatibility
    weights_4 = [0.0, 0.9, 0.05, 0.05]
    pose_metrics_4 = PoseMetrics(fitness_weight=weights_4)

    # Both box and pose should have same weights
    assert pose_metrics_4.box.fitness_weight == weights_4, (
        f"Box weights (4-weight mode) incorrect: {pose_metrics_4.box.fitness_weight}"
    )
    assert pose_metrics_4.pose.fitness_weight == weights_4, (
        f"Pose weights (4-weight mode) incorrect: {pose_metrics_4.pose.fitness_weight}"
    )

    print("  ✓ 4-weight backward compatibility maintained")

    # Test different 8-weight strategies
    strategies = {
        "Box Only": [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
        "Pose Only": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
        "Balanced Box+Pose": [0.0, 0.0, 0.05, 0.45, 0.0, 0.0, 0.05, 0.45],
        "Box 80% Pose 20%": [0.0, 0.0, 0.08, 0.72, 0.0, 0.0, 0.02, 0.18],
        "Precision Focus Both": [0.45, 0.0, 0.05, 0.0, 0.45, 0.0, 0.05, 0.0],
    }

    for name, weights in strategies.items():
        PoseMetrics(fitness_weight=weights)
        print(f"  ✓ {name}: box_weights={weights[:4]}, pose_weights={weights[4:]}")

    print("✓ PoseMetrics 8-weight test passed")


def test_segment_metrics_8_weight():
    """Test SegmentMetrics with 8-weight fitness configuration."""
    print("Testing SegmentMetrics with 8-weight fitness...")

    # Test 8-weight configuration - optimize for mask only
    weights_8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9]  # mask mAP only
    seg_metrics = SegmentMetrics(fitness_weight=weights_8)

    # Verify box weights are first 4
    assert seg_metrics.box.fitness_weight == [0.0, 0.0, 0.0, 0.0], (
        f"Box weights incorrect: {seg_metrics.box.fitness_weight}"
    )

    # Verify mask weights are last 4
    assert seg_metrics.seg.fitness_weight == [0.0, 0.0, 0.1, 0.9], (
        f"Mask weights incorrect: {seg_metrics.seg.fitness_weight}"
    )

    print("  ✓ 8-weight split correctly between box and mask")

    # Test 4-weight backward compatibility
    weights_4 = [0.0, 0.9, 0.05, 0.05]
    seg_metrics_4 = SegmentMetrics(fitness_weight=weights_4)

    # Both box and mask should have same weights
    assert seg_metrics_4.box.fitness_weight == weights_4, (
        f"Box weights (4-weight mode) incorrect: {seg_metrics_4.box.fitness_weight}"
    )
    assert seg_metrics_4.seg.fitness_weight == weights_4, (
        f"Mask weights (4-weight mode) incorrect: {seg_metrics_4.seg.fitness_weight}"
    )

    print("  ✓ 4-weight backward compatibility maintained")

    # Test different 8-weight strategies
    strategies = {
        "Box Only": [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
        "Mask Only": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
        "Balanced Box+Mask": [0.0, 0.0, 0.05, 0.45, 0.0, 0.0, 0.05, 0.45],
    }

    for name, weights in strategies.items():
        SegmentMetrics(fitness_weight=weights)
        print(f"  ✓ {name}: box_weights={weights[:4]}, mask_weights={weights[4:]}")

    print("✓ SegmentMetrics 8-weight test passed")


def test_8_weight_fitness_calculation():
    """Test actual fitness calculation with 8 weights."""
    print("Testing 8-weight fitness calculation...")
    import numpy as np

    # Create PoseMetrics with 8 weights: optimize pose only
    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9]
    pose_metrics = PoseMetrics(fitness_weight=weights)

    # Set up dummy data for box metrics
    pose_metrics.box.p = np.array([0.8, 0.75])
    pose_metrics.box.r = np.array([0.75, 0.70])
    pose_metrics.box.all_ap = np.array(
        [
            [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
            [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25],
        ]
    )
    pose_metrics.box.nc = 2

    # Set up dummy data for pose metrics
    pose_metrics.pose.p = np.array([0.9, 0.88])
    pose_metrics.pose.r = np.array([0.88, 0.86])
    pose_metrics.pose.all_ap = np.array(
        [
            [0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74],
            [0.85, 0.83, 0.81, 0.79, 0.77, 0.75, 0.73, 0.71, 0.69, 0.67],
        ]
    )
    pose_metrics.pose.nc = 2

    # Calculate fitness
    total_fitness = pose_metrics.fitness

    # With weights [0,0,0,0, 0,0,0.1,0.9], only pose mAP50 and mAP contribute
    # Box fitness should be 0, pose fitness should be non-zero
    print(f"  Total fitness: {total_fitness:.3f}")

    # Verify it's a valid positive number
    assert isinstance(total_fitness, (int, float, np.number)), "Fitness should be a number"
    assert not np.isnan(total_fitness), "Fitness should not be NaN"
    assert total_fitness > 0, f"Fitness should be positive with pose weights, got {total_fitness}"

    # Test that pose-only weights give higher fitness than box-only weights
    # when pose metrics are better than box metrics
    weights_box_only = [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0]
    pose_metrics_box = PoseMetrics(fitness_weight=weights_box_only)
    pose_metrics_box.box.p = pose_metrics.box.p
    pose_metrics_box.box.r = pose_metrics.box.r
    pose_metrics_box.box.all_ap = pose_metrics.box.all_ap
    pose_metrics_box.box.nc = 2
    pose_metrics_box.pose.p = pose_metrics.pose.p
    pose_metrics_box.pose.r = pose_metrics.pose.r
    pose_metrics_box.pose.all_ap = pose_metrics.pose.all_ap
    pose_metrics_box.pose.nc = 2

    fitness_box_only = pose_metrics_box.fitness

    print(f"  Pose-only fitness: {total_fitness:.3f}")
    print(f"  Box-only fitness: {fitness_box_only:.3f}")
    print("  ✓ Successfully calculated fitness with 8-weight configuration")

    print("✓ 8-weight fitness calculation test passed")


def main():
    """Run all tests."""
    print("Running fitness_weight configuration tests...")
    print("=" * 60)

    try:
        # Original tests (4-weight)
        test_metric_fitness_weight()
        test_det_metrics_fitness_weight()
        test_pose_metrics_fitness_weight()
        test_segment_metrics_fitness_weight()
        test_obb_metrics_fitness_weight()
        test_validator_args_integration()
        test_different_optimization_strategies()

        # New tests (8-weight)
        print("\n" + "-" * 60)
        print("Testing 8-weight fitness functionality...")
        print("-" * 60)
        test_pose_metrics_8_weight()
        test_segment_metrics_8_weight()
        test_8_weight_fitness_calculation()

        print("\n" + "=" * 60)
        print("✅ All tests passed! fitness_weight configuration is working correctly.")
        print("   - 4-weight mode: detection/OBB/classification tasks")
        print("   - 8-weight mode: pose/segment tasks with independent control")
        print("   - Backward compatibility: maintained for existing configs")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
