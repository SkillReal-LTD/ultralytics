from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

# Train with custom fitness weights
results = model.train(
    data="/home/ubuntu/ultralytics/ultralytics/cfg/datasets/coco8-grayscale.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)
