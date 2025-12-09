from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"/models/best.pt")

    metrics = model.val(
        data=r"E:\_JOB_\_Python\Seeding\dataset\datasetKlassV5Or\data.yaml",
        conf=0.25,
        iou=0.7,
        save_json=True,
        plots=True,  # 🎯 создаёт confusion matrix, PR, F1
        workers=0    # 🎯 важно для Windows
    )

    print(metrics)
