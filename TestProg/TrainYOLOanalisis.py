import cv2
import os
from pathlib import Path
from ultralytics import YOLO


def preprocess_dataset(input_root, output_root, size=1024):
    """ Делаем фото квадратными с белым фоном """
    input_root, output_root = Path(input_root), Path(output_root)
    valid_ext = ('.jpg', '.jpeg', '.png')
    for subdir, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(valid_ext):
                rel_path = Path(subdir).relative_to(input_root)
                save_dir = output_root / rel_path
                save_dir.mkdir(parents=True, exist_ok=True)

                img = cv2.imread(os.path.join(subdir, file))
                if img is None: continue

                h, w = img.shape[:2]
                max_side = max(h, w)
                top = (max_side - h) // 2
                bottom = max_side - h - top
                left = (max_side - w) // 2
                right = max_side - w - left

                padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                            cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.imwrite(os.path.join(save_dir, file), cv2.resize(padded, (size, size)))


# --- КОНФИГУРАЦИЯ ---
data_sq = r"E:\_JOB_\_Python\Seeding\dataset\dataset_SQUARED"
data_orig = r"E:\_JOB_\_Python\Seeding\dataset\datasetAnalisV1"

if __name__ == '__main__':
    if not os.path.exists(data_sq):
        preprocess_dataset(data_orig, data_sq, size=1024)

    model = YOLO("yolov8s-cls.pt")

    results = model.train(
        data=data_sq,
        epochs=100,
        imgsz=1024,
        batch=4,  # Уменьшили с 8 до 4, чтобы не вылетало CUDA OOM
        workers=2,  # Меньше потоков - стабильнее на ноутбуке
        device=0,
        project=r"E:\_JOB_\_Python\Seeding\results",
        name="root_clean_v2",
        exist_ok=True,

        # ОТКЛЮЧАЕМ "НЕГАТИВ" И ЦВЕТОВЫЕ ИСКАЖЕНИЯ
        hsv_h=0.0,  # Оттенок (Hue) - 0.0 выключает изменение цвета
        hsv_s=0.0,  # Насыщенность (Saturation)
        hsv_v=0.0,  # Яркость (Value)
        augment=False,  # Отключаем стандартный RandAugment (он делает негатив)

        # ГЕОМЕТРИЯ (сохраняем)
        crop_fraction=1.0,  # Не резать края
        fliplr=0.5,  # Отразить право-лево можно
        flipud=0.0,  # Не переворачиваем корень вверх ногами

        # Оптимизация памяти
        amp=True,  # Смешанная точность (помогает 3050)
        cache=False  # Не кэшируем в оперативку, чтобы не забить систему
    )