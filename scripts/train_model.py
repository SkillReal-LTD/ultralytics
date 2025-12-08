from ultralytics import YOLO

# Load model
model = YOLO('yolo11n.pt')

results = model.train(
    task='detect',
    data='/home/ubuntu/ultralytics/configs/test_skillreal.yaml',
    epochs=5,
    imgsz=640,
    batch=16,
    fitness_weight=[0.0, 0.9, 0.1, 0.0],
)
print(model.names)