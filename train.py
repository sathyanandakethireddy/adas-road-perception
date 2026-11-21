
from ultralytics import YOLO

# Load pretrained YOLO26 Nano model
model = YOLO("yolo26n.pt")

# Train on road damage dataset
results = model.train(
    data="data/road_damage/dataset/data.yaml",
    epochs=30,
    imgsz=416,
    batch=2,
    device="cpu",
    workers=2,
    project="runs/road_damage",
    name="yolo26n_pothole",
    patience=10,
    plots=True
)

print("Training completed!")
