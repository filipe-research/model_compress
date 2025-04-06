from ultralytics import YOLO
import torch

model = YOLO("yolo11n.pt")

model.train(
        data=dataset_path + "/data.yaml",
        workers=4,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0 ,
        project="runs/train",
        name="yolo11_oxford_tower_custom_train"
    )

print("Training finished.")
   


