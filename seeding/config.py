"""Конфигурация приложения."""

from pathlib import Path
import os

# Путь к весам YOLOv8. Можно задать через переменную окружения YOLO_WEIGHTS_PATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WEIGHTS_PATH = Path(
    os.getenv(
        "YOLO_WEIGHTS_PATH",
        str(PROJECT_ROOT / "models" /  "bestCrop.pt")
    )
)

# Путь к весам модели классификации. Можно задать переменной YOLO_CLASSIFY_WEIGHTS_PATH
DEFAULT_CLASSIFY_WEIGHTS_PATH = Path(
    os.getenv(
        "YOLO_CLASSIFY_WEIGHTS_PATH",
        str(PROJECT_ROOT / "models" / "bestKlassSeg.pt"),
    )
)

# Новый: Путь к третьей модели для классификации жизнеспособности корня (good/bad)
DEFAULT_ROOT_CLASSIFY_WEIGHTS_PATH = Path(
    os.getenv(
        "YOLO_ROOT_CLASSIFY_WEIGHTS_PATH",
        str(PROJECT_ROOT / "models" / "best_root_cls.pt"),  # Замените на вашу обученную модель
    )
)

# Параметр поворота на 90 градусов: значение k для np.rot90
ROTATE_K = 1

SEG_MODEL_CLASSES = ["flower", "root", "stem"]