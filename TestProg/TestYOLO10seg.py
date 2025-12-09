from ultralytics import YOLO

if __name__ == "__main__":
    # Загружаем YOLOv10
    model = YOLO("yolov10s.pt")  # можешь заменить на yolov10m.pt

    model.train(
        data=r"E:\_JOB_\_Python\Seeding\dataset\datasetKlassV5\data.yaml",
        project=r"E:\_JOB_\_Python\Seeding\results",
        name="seeding-seg-yolo10",

        # ==== GPU ====
        device=0,
        imgsz=1024,
        batch=8,
        workers=2,
        amp=True,
        cache=True,
        rect=True,

        # ==== ОПТИМИЗАЦИЯ ====
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.01,
        momentum=0.9,
        cos_lr=True,
        warmup_epochs=3,

        # ==== АУГМЕНТАЦИИ ====
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
        degrees=0.0,
        translate=0.05,
        scale=0.1,
        shear=0.0,
        perspective=0.0,

        # ==== ОБУЧЕНИЕ ====
        epochs=300,
        verbose=True,
    )
