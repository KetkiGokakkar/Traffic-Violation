from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="/Users/ketkigokakkar/Documents/Ketki Gokakkar/old/college/projects/Project_2/vehicle_dataset/VehiclesDetectionDataset/dataset.yaml",
    epochs=2,
    imgsz=640,
    batch=16
)
results = model("/Users/ketkigokakkar/Documents/Ketki Gokakkar/old/college/projects/Project_2/vehicle_dataset/VehiclesDetectionDataset/train/images/0a03d85b3dcb2a1b_jpg.rf.a00ac9c0b1e0178bd393f049593c73c6.jpg")[0]
classes = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
count = {cls: 0 for cls in classes}
cls_ids = results.boxes.cls.cpu().numpy().astype(int)
for c in cls_ids:
    name = model.names[int(c)]
    if name in classes:
        count[name] += 1
print("Vehicle counts per type:")
for k, v in count.items():
    print(f"{k}: {v}")
results.show()
