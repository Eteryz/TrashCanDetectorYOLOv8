from ultralytics import YOLO

# model = YOLO("yolov8m.yaml")

model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# После обучения модель сохраняется по пути runs/detect/train
if __name__ == '__main__':
    # Предварительно необходимо разметить датасет и загрузить к себе в проект
    model.train(
         data="datasets/data.yaml",
         epochs=100,
         imgsz=640
    )

    # val() выполняет валидацию модели после ее обучения и сохраняет метрики, полученные в процессе валидации
    metrics = model.val()

# Сохранение обученной модели в других форматах
# model.export(format="torchscript")
# model.export(format="onnx")
