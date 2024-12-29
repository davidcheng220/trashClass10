from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Load a pretrained model
# check_train_batch_size(model, imgsz=640, amp=True, batch=-1, max_num_obj=1)
results = model.train(data=r"D:\egg\data.yaml", epochs=100, imgsz=224, batch=8, workers=0)