from ultralytics import YOLO
import cv2


def cropped_image_detect_object(image_path):
    frame = cv2.imread(image_path)
    for res in results:
        print("RESULT")
        cropped_images = []
        for box in res.boxes:
            # Вырезаем найденные объекты
            print("BOX")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
            cropped_images.append(cropped_image)

        for i in cropped_images:
            cv2.imshow('frame', i)
            cv2.waitKey(0)


if __name__ == '__main__':
    # Путь до обученной модели .pt
    path = 'C:\\Users\\Vladislav\\PycharmProjects\\TrashCanDetectorYOLOv8\\runs\\detect\\train\\weights\\best.pt'
    model = YOLO(path)

    img = "resources/5.png"
    # Обнаружение объектов и сохранение результата в папке runs
    results = model(
        source=img,
        save=True,
        save_txt=True,
        conf=0.6
    )
    cropped_image_detect_object(image_path=img)


