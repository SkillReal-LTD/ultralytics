from ultralytics import YOLO

model = YOLO("/home/ubuntu/algo-ai-training/runs/detect/train24/weights/best.pt", task="detect")

results = model.predict(
    "/home/ubuntu/algo-ai-training/data/test_27072025/processed/test/images/Task image03-01-2025 16-58-26-24919537_0_0_default.jpg",
    save=True,
    save_txt=True,
    save_conf=True,
    save_crop=True,
    classes=[11],
)
