from ultralytics import YOLO

model = YOLO(r'C:\Users\86131\ultralytics-main\ultralytics\cfg\models\v8\yolov8n-CBAM.yaml')  # Load a pretrained YOLOv8 model

model.train(data="yolo-bvn.yaml", workers=0,epochs=100,batch=16)  # Train the model on your dataset