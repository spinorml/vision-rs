from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # or yolo26n
model.train(data="coco128.yaml", epochs=10, batch=16)
