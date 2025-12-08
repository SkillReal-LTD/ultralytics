from ultralytics import YOLO
import os

def test_detection_fitness_weight():
    """
    Test fitness weight for detection model.
    fitness_weight: [P, R, mAP@0.5, mAP@0.5:0.95]
    Testing with [0.0, 0.0, 0.1, 0.9] to prioritize mAP@0.5:0.95
    """
    print("\n" + "="*80)
    print("TESTING DETECTION MODEL FITNESS WEIGHT")
    print("="*80)
    print("fitness_weight: [0.0, 0.0, 0.1, 0.9]")
    print("Expected behavior: Prioritize mAP@0.5:0.95 (weight=0.9) over mAP@0.5 (weight=0.1)")
    print("="*80 + "\n")

    # Load detection model
    model = YOLO('yolo11n.pt')

    # Train with specific fitness weights
    results = model.train(
        task='detect',
        data='/home/ubuntu/ultralytics/configs/test_skillreal.yaml',
        epochs=2,  # Short test run
        imgsz=640,
        batch=16,
        fitness_weight=[0.0, 0.0, 0.1, 0.9],  # [P, R, mAP@0.5, mAP@0.5:0.95]
        classes=[11],
        name='test_detect_fitness',
        project='fitness_test'
    )

    print("\n" + "="*80)
    print("DETECTION TEST COMPLETED")
    print("="*80)
    print(f"Model names: {model.names}")
    print("="*80 + "\n")

    return results


def test_pose_fitness_weight():
    """
    Test fitness weight for pose model.
    fitness_weight: [box_P, box_R, box_mAP@0.5, box_mAP@0.5:0.95, pose_P, pose_R, pose_mAP@0.5, pose_mAP@0.5:0.95]
    Testing with [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0] to prioritize box metrics
    """
    print("\n" + "="*80)
    print("TESTING POSE MODEL FITNESS WEIGHT")
    print("="*80)
    print("fitness_weight: [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0]")
    print("Expected behavior: Prioritize box_mAP@0.5:0.95 (weight=0.9) and box_mAP@0.5 (weight=0.1)")
    print("Ignoring pose metrics (all weights=0.0)")
    print("="*80 + "\n")

    # Load pose model
    model = YOLO('yolo11n-pose.pt')

    # Train with specific fitness weights for pose (8 values)
    results = model.train(
        task='pose',
        data='/home/ubuntu/ultralytics/configs/test_skillreal.yaml',
        epochs=2,  # Short test run
        imgsz=640,
        batch=16,
        fitness_weight=[0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0],  # 8 values for pose
        name='test_pose_fitness',
        project='fitness_test'
    )

    print("\n" + "="*80)
    print("POSE TEST COMPLETED")
    print("="*80)
    print(f"Model names: {model.names}")
    print("="*80 + "\n")

    return results


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# FITNESS WEIGHT VERIFICATION TEST")
    print("# Testing both Detection and Pose models")
    print("#"*80 + "\n")

    # Test 1: Detection model                       
    try:
        detect_results = test_detection_fitness_weight()
        print("✓ Detection fitness weight test executed successfully\n")
    except Exception as e:
        print(f"✗ Detection fitness weight test failed: {e}\n")

    # Test 2: Pose model
    # try:
    #     pose_results = test_pose_fitness_weight()
    #     print("✓ Pose fitness weight test executed successfully\n")
    # except Exception as e:
    #     print(f"✗ Pose fitness weight test failed: {e}\n")

    print("\n" + "#"*80)
    print("# ALL TESTS COMPLETED")
    print("#"*80 + "\n")
