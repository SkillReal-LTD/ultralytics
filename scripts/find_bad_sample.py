"""Script to find problematic samples in GPU 3's partition."""

import signal
from pathlib import Path


# Timeout handler
class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Sample loading timed out!")


# Set up signal handler for timeout
signal.signal(signal.SIGALRM, timeout_handler)


def test_samples():
    from ultralytics.data import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    # Your config - adjust path as needed
    data_yaml = "/home/ubuntu/algo-ai-training/outputs/test_freezing/output/YOLO11n-pose.yaml"

    # If the above doesn't exist, try to find it
    if not Path(data_yaml).exists():
        print(f"Data YAML not found at {data_yaml}")
        print("Please update the data_yaml path in this script")
        return

    # Load dataset info
    data = check_det_dataset(data_yaml)

    # Create dataset (same as training would)
    dataset = YOLODataset(
        img_path=data["train"],
        imgsz=640,
        batch_size=16,
        augment=False,  # No augmentation for testing
        hyp=None,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix="test: ",
        task="pose",
        classes=None,
        data=data,
        fraction=1.0,
    )

    print(f"Dataset has {len(dataset)} samples")

    # GPU 3's range (from our calculation)
    start_idx = 4320
    end_idx = min(5737, len(dataset))

    print(f"Testing samples {start_idx} to {end_idx - 1} (GPU 3's partition)")
    print("=" * 60)

    bad_samples = []

    for idx in range(start_idx, end_idx):
        try:
            # Set 10 second timeout per sample
            signal.alarm(10)

            # Try to load the sample
            dataset[idx]

            # Cancel alarm
            signal.alarm(0)

            if idx % 100 == 0:
                print(f"Sample {idx}: OK")

        except TimeoutError:
            print(f"Sample {idx}: TIMEOUT - took more than 10 seconds!")
            bad_samples.append((idx, "timeout"))
            signal.alarm(0)

        except Exception as e:
            print(f"Sample {idx}: ERROR - {type(e).__name__}: {e}")
            bad_samples.append((idx, str(e)))
            signal.alarm(0)

    print()
    print("=" * 60)
    if bad_samples:
        print(f"Found {len(bad_samples)} problematic samples:")
        for idx, error in bad_samples:
            # Get the image path
            img_path = dataset.im_files[idx] if idx < len(dataset.im_files) else "unknown"
            print(f"  Index {idx}: {img_path}")
            print(f"    Error: {error}")
    else:
        print("All samples loaded successfully!")
        print("The issue might be in DDP synchronization, not data loading.")


if __name__ == "__main__":
    test_samples()
