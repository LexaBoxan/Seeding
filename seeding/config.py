"""Конфигурация приложения."""

from pathlib import Path
import os

# Путь к весам YOLOv8. Можно задать через переменную окружения YOLO_WEIGHTS_PATH
DEFAULT_WEIGHTS_PATH = Path(os.getenv("YOLO_WEIGHTS_PATH", "weights/best.pt"))

# Параметр поворота на 90 градусов: значение k для np.rot90
ROTATE_K = 1
