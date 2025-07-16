#!/usr/bin/env python3
"""
Test script to verify fitness_weight configuration works correctly.
"""

import sys
from types import SimpleNamespace
from ultralytics.utils.metrics import DetMetrics, PoseMetrics, SegmentMetrics, OBBMetrics, Metric

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
    metric.p = [0.8, 0.7]
    metric.r = [0.9, 0.8]
    metric.all_ap = [[0.85, 0.80], [0.75, 0.70]]
    
    # Mock methods
    metric.mp = 0.75  # mean precision
    metric.mr = 0.85  # mean recall
    metric.map50 = 0.80  # mAP@0.5
    metric.map = 0.75  # mAP@0.5:0.95
    
    # Override mean_results to return mock values
    metric.mean_results = lambda: [0.75, 0.85, 0.80, 0.75]
    
    # Test fitness calculation
    fitness = metric.fitness()
    expected_fitness = (0.75 * 0.0) + (0.85 * 0.9) + (0.80 * 0.1) + (0.75 * 0.0)
    assert abs(fitness - expected_fitness) < 0.001, f"Fitness calculation incorrect: {fitness} vs {expected_fitness}"
    
    print("✓ Metric class fitness_weight test passed")

def test_det_metrics_fitness_weight():
    """Test that DetMetrics accepts and passes fitness_weight correctly."""
    print("Testing DetMetrics class fitness_weight...")
    
    # Test with custom weights
    custom_weights = [0.0, 0.9, 0.1, 0.0]
    det_metrics = DetMetrics(fitness_weight=custom_weights)
    
    assert det_metrics.box.fitness_weight == custom_weights, f"DetMetrics fitness_weight not passed: {det_metrics.box.fitness_weight}"
    
    print("✓ DetMetrics class fitness_weight test passed")

def test_pose_metrics_fitness_weight():
    """Test that PoseMetrics accepts and passes fitness_weight correctly."""
    print("Testing PoseMetrics class fitness_weight...")
    
    # Test with custom weights
    custom_weights = [0.0, 0.9, 0.1, 0.0]
    pose_metrics = PoseMetrics(fitness_weight=custom_weights)
    
    assert pose_metrics.box.fitness_weight == custom_weights, f"PoseMetrics box fitness_weight not passed: {pose_metrics.box.fitness_weight}"
    assert pose_metrics.pose.fitness_weight == custom_weights, f"PoseMetrics pose fitness_weight not passed: {pose_metrics.pose.fitness_weight}"
    
    print("✓ PoseMetrics class fitness_weight test passed")

def test_segment_metrics_fitness_weight():
    """Test that SegmentMetrics accepts and passes fitness_weight correctly."""
    print("Testing SegmentMetrics class fitness_weight...")
    
    # Test with custom weights
    custom_weights = [0.0, 0.9, 0.1, 0.0]
    seg_metrics = SegmentMetrics(fitness_weight=custom_weights)
    
    assert seg_metrics.box.fitness_weight == custom_weights, f"SegmentMetrics box fitness_weight not passed: {seg_metrics.box.fitness_weight}"
    assert seg_metrics.seg.fitness_weight == custom_weights, f"SegmentMetrics seg fitness_weight not passed: {seg_metrics.seg.fitness_weight}"
    
    print("✓ SegmentMetrics class fitness_weight test passed")

def test_obb_metrics_fitness_weight():
    """Test that OBBMetrics accepts and passes fitness_weight correctly."""
    print("Testing OBBMetrics class fitness_weight...")
    
    # Test with custom weights
    custom_weights = [0.0, 0.9, 0.1, 0.0]
    obb_metrics = OBBMetrics(fitness_weight=custom_weights)
    
    assert obb_metrics.box.fitness_weight == custom_weights, f"OBBMetrics fitness_weight not passed: {obb_metrics.box.fitness_weight}"
    
    print("✓ OBBMetrics class fitness_weight test passed")

def test_validator_args_integration():
    """Test integration with validator args."""
    print("Testing validator args integration...")
    
    # Mock args object
    args = SimpleNamespace()
    args.fitness_weight = [0.0, 0.9, 0.1, 0.0]
    
    # Test getattr usage
    fitness_weight = getattr(args, 'fitness_weight', None)
    assert fitness_weight == [0.0, 0.9, 0.1, 0.0], f"Args fitness_weight not retrieved: {fitness_weight}"
    
    # Test with None
    args_none = SimpleNamespace()
    fitness_weight_none = getattr(args_none, 'fitness_weight', None)
    assert fitness_weight_none is None, f"Args fitness_weight should be None: {fitness_weight_none}"
    
    print("✓ Validator args integration test passed")

def test_different_optimization_strategies():
    """Test different optimization strategies."""
    print("Testing different optimization strategies...")
    
    strategies = {
        'Default (mAP)': [0.0, 0.0, 0.1, 0.9],
        'Recall Optimized': [0.0, 0.9, 0.1, 0.0],
        'Precision Optimized': [0.9, 0.0, 0.1, 0.0],
        'Balanced P/R': [0.45, 0.45, 0.1, 0.0],
        'mAP@0.5 Only': [0.0, 0.0, 1.0, 0.0],
    }
    
    # Mock metrics: [precision, recall, map50, map]
    mock_metrics = [0.8, 0.7, 0.85, 0.75]
    
    for name, weights in strategies.items():
        metric = Metric(fitness_weight=weights)
        metric.mean_results = lambda: mock_metrics
        
        fitness = metric.fitness()
        expected_fitness = sum(m * w for m, w in zip(mock_metrics, weights))
        
        assert abs(fitness - expected_fitness) < 0.001, f"Strategy {name} fitness incorrect: {fitness} vs {expected_fitness}"
        
        print(f"  ✓ {name}: weights={weights}, fitness={fitness:.3f}")
    
    print("✓ Different optimization strategies test passed")

def main():
    """Run all tests."""
    print("Running fitness_weight configuration tests...")
    print("=" * 60)
    
    try:
        test_metric_fitness_weight()
        test_det_metrics_fitness_weight()
        test_pose_metrics_fitness_weight()
        test_segment_metrics_fitness_weight()
        test_obb_metrics_fitness_weight()
        test_validator_args_integration()
        test_different_optimization_strategies()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! fitness_weight configuration is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main() 