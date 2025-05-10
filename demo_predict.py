from ultralytics import YOLO

yolo = YOLO("./yolov8n.pt",task="detect")

result = yolo(source="./ultralytics/assets/playground.jpg",save=True,conf=0.1)