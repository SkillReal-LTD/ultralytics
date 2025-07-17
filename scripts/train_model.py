from ultralytics import YOLO

# Load model
model = YOLO('yolo11n.pt')

# results = model.train(
#     task='detect',
#     data='configs/coco8.yaml',
#     epochs=30,
#     imgsz=640,
#     batch=16,
#     fitness_weight=[0.0, 0.9, 0.1, 0.0]  
# )

# Train with custom fitness weights
results = model.train(
    task='pose',
    data='configs/coco8_pose.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    fitness_weight=[0.1, 0.0, 0.9, 0.0]
)