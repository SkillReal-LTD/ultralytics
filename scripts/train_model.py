import logging

from ultralytics import YOLO, settings

settings.update({"wandb": True})

logging.getLogger("ultralytics").setLevel(logging.DEBUG)

# Load model
model = YOLO("yolo11n.pt")

results = model.train(
    project="ultralytics",
    name="test_artifacts",
    task="detect",
    data="/home/ubuntu/ultralytics/configs/test_skillreal.yaml",
    epochs=4,
    imgsz=640,
    batch=16,
    save_period=1,  # Save checkpoint every epoch (1, 2, 3, 4, 5)
    fitness_weight=[0.0, 0.9, 0.1, 0.0],
)
print(model.trainer.best_epoch)
