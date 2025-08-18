from ultralytics import YOLO

if __name__ == "__main__":  # 👈 ОБЯЗАТЕЛЬНО на Windows
    model = YOLO(r"E:\_JOB_\_Python\Seeding\models\yolov8m-seg.pt")

    model.train(
        data=r"E:\_JOB_\_Python\Seeding\dataset\datasetSegV3\dataset.yaml",
        epochs=64,
        name="seeding-seg",
        device=0,
        lr0 = 0.001,
        batch = 4,  # Поставь поменьше, если "вылетает" по памяти!
        imgsz = 640, #1024
        project = "E:\\_JOB_\\_Python\\Seeding\\results",
        mosaic = 0.0,  # Стандартные аугментации лучше не отключать!
        mixup = 0.2
    )
