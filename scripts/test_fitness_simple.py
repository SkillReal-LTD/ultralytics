from ultralytics import YOLO

print("Testing fitness_weight parameter...")

# Load model
model = YOLO("yolo11n.pt")

# Try to train with fitness_weight
try:
    results = model.train(
        task="detect",
        data="/home/ubuntu/ultralytics/configs/test_skillreal.yaml",
        epochs=1,
        imgsz=640,
        batch=4,
        fitness_weight=[0.0, 0.0, 0.1, 0.9],
        classes=[11],
        name="test_simple",
        project="test_fitness",
    )
    print("SUCCESS: Training completed with fitness_weight parameter")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
