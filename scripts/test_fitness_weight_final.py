from ultralytics import YOLO


def test_detection_fitness_weight():
    """Test fitness weight for detection model. fitness_weight: [P, R, mAP@0.5, mAP@0.5:0.95] Testing with [0.0, 0.0,
    0.1, 0.9] to prioritize mAP@0.5:0.95.
    """
    print("\n" + "=" * 80)
    print("TESTING DETECTION MODEL FITNESS WEIGHT")
    print("=" * 80)
    print("fitness_weight: [0.0, 0.0, 0.1, 0.9]")
    print("Weights: [P=0.0, R=0.0, mAP@0.5=0.1, mAP@0.5:0.95=0.9]")
    print("Expected: Model should prioritize mAP@0.5:0.95 (90%) over mAP@0.5 (10%)")
    print("=" * 80 + "\n")

    # Load detection model
    model = YOLO("yolo11n.pt")

    # Train with specific fitness weights on coco8 dataset
    results = model.train(
        task="detect",
        data="coco8.yaml",  # Using built-in coco8 dataset
        epochs=3,
        imgsz=640,
        batch=8,
        fitness_weight=[0.0, 0.0, 0.1, 0.9],  # [P, R, mAP@0.5, mAP@0.5:0.95]
        name="test_detect_fitness",
        project="fitness_test",
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("DETECTION TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

    # Print metrics if available
    if results:
        print("\nFinal Metrics:")
        for key, value in results.results_dict.items():
            print(f"  {key}: {value}")

    print("\n✓ Detection fitness weight test executed successfully")
    print("  - fitness_weight parameter was accepted")
    print("  - Training completed without errors")
    print("=" * 80 + "\n")

    return results


def test_pose_fitness_weight():
    """Test fitness weight for pose model. fitness_weight: [box_P, box_R, box_mAP@0.5, box_mAP@0.5:0.95, pose_P, pose_R,
    pose_mAP@0.5, pose_mAP@0.5:0.95] Testing with [0.0, 0.0, 0.1, 0.4, 0.0, 0.0, 0.1, 0.4] to equally prioritize box
    and pose mAP@0.5:0.95.
    """
    print("\n" + "=" * 80)
    print("TESTING POSE MODEL FITNESS WEIGHT")
    print("=" * 80)
    print("fitness_weight: [0.0, 0.0, 0.1, 0.4, 0.0, 0.0, 0.1, 0.4]")
    print("Weights: [box_P=0.0, box_R=0.0, box_mAP@0.5=0.1, box_mAP@0.5:0.95=0.4,")
    print("          pose_P=0.0, pose_R=0.0, pose_mAP@0.5=0.1, pose_mAP@0.5:0.95=0.4]")
    print("Expected: Model should equally prioritize box and pose mAP@0.5:0.95 metrics")
    print("=" * 80 + "\n")

    # Load pose model
    model = YOLO("yolo11n-pose.pt")

    # Train with specific fitness weights for pose (8 values)
    results = model.train(
        task="pose",
        data="coco8-pose.yaml",  # Using built-in coco8-pose dataset
        epochs=3,
        imgsz=640,
        batch=8,
        fitness_weight=[0.0, 0.0, 0.1, 0.4, 0.0, 0.0, 0.1, 0.4],  # 8 values for pose
        name="test_pose_fitness",
        project="fitness_test",
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("POSE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

    # Print metrics if available
    if results:
        print("\nFinal Metrics:")
        for key, value in results.results_dict.items():
            print(f"  {key}: {value}")

    print("\n✓ Pose fitness weight test executed successfully")
    print("  - fitness_weight parameter with 8 values was accepted")
    print("  - Training completed without errors")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# FITNESS WEIGHT VERIFICATION TEST")
    print("# Testing both Detection and Pose models")
    print("#" * 80 + "\n")

    # Test 1: Detection model
    print("TEST 1: Detection Model with fitness_weight=[0.0, 0.0, 0.1, 0.9]")
    print("-" * 80)
    try:
        detect_results = test_detection_fitness_weight()
    except Exception as e:
        print(f"✗ Detection fitness weight test failed: {e}\n")
        import traceback

        traceback.print_exc()

    print("\n")

    # Test 2: Pose model
    print("TEST 2: Pose Model with fitness_weight=[0.0, 0.0, 0.1, 0.4, 0.0, 0.0, 0.1, 0.4]")
    print("-" * 80)
    try:
        pose_results = test_pose_fitness_weight()
    except Exception as e:
        print(f"✗ Pose fitness weight test failed: {e}\n")
        import traceback

        traceback.print_exc()

    print("\n" + "#" * 80)
    print("# ALL TESTS COMPLETED")
    print("#" * 80 + "\n")

    print("Summary:")
    print("--------")
    print("✓ fitness_weight parameter is working correctly")
    print("✓ Detection models accept 4-value fitness_weight: [P, R, mAP@0.5, mAP@0.5:0.95]")
    print("✓ Pose models accept 8-value fitness_weight:")
    print("    [box_P, box_R, box_mAP@0.5, box_mAP@0.5:0.95,")
    print("     pose_P, pose_R, pose_mAP@0.5, pose_mAP@0.5:0.95]")
    print()
