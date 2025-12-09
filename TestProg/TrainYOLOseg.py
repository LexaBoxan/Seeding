from ultralytics import YOLO
import os

if __name__ == "__main__":

    # === ЗАГРУЗКА ЧЕКПОИНТА ===
    # Если есть last.pt → продолжаем обучение
    # Если нет — начнёт с нуля.
    ckpt_path = r"E:\_JOB_\_Python\Seeding\results\seeding-klass200\weights\last.pt"

    if os.path.exists(ckpt_path):
        print("📌 Продолжаю обучение с:", ckpt_path)
        model = YOLO(ckpt_path)
    else:
        print("📌 Чекпоинт не найден — старт с нуля.")
        model = YOLO(r"E:\_JOB_\_Python\Seeding\models\yolov8m-seg.pt")

    model.train(
        cfg=r"E:\_JOB_\_Python\Seeding\TrainConfigs\seeding_config_for_seg.yaml",
        data=r"E:\_JOB_\_Python\Seeding\dataset\datasetSegV6\data.yaml",
        project=r"E:\_JOB_\_Python\Seeding\results",
        name="seeding-seg-new-test",
        device=0,
    )


