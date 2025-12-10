import ultralytics

model = ultralytics.YOLO("yolo11n.pt")

results = model.train(data="raw_data_config.yaml", epochs=300, imgsz=224, device="mps")
