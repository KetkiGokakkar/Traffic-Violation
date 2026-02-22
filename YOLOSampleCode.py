from ultralytics import YOLO
model = YOLO("yolov8n.pt")
result = model("street.jpg")[0]
result.show()