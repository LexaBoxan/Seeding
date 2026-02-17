"""Точка входа приложения Seeding.

Запускает PyQt5 GUI для анализа изображений сеянцев с детекцией YOLOv8.
"""

import argparse
import logging
import os
import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMessageBox
import qt_material

from seeding.config import (
    DEFAULT_WEIGHTS_PATH,
    APP_FONT_FAMILY,
    APP_FONT_SIZE,
    QT_MATERIAL_THEME,
)
from seeding.ui.main_window import ImageEditor
from seeding.ui.styles import MAIN_STYLESHEET


def _validate_weights_path(path: str) -> bool:
    """Проверяет, что путь к весам корректен. Для локальных файлов — существование."""
    if not path:
        return False
    # Если путь выглядит как локальный файл — проверяем существование
    if os.sep in path or (len(path) > 1 and path[1] == ":"):
        return os.path.isfile(path)
    # Имя модели (yolov8n.pt и т.д.) — считаем валидным
    return True


def main() -> None:
    """Запускает графическое приложение."""

    parser = argparse.ArgumentParser(description="ImageEditor")
    parser.add_argument(
        "--weights",
        default=os.getenv("YOLO_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH)),
        help="Путь к весам YOLOv8",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
    )

    app = QApplication(sys.argv)
    app.setFont(QFont(APP_FONT_FAMILY, APP_FONT_SIZE))
    qt_material.apply_stylesheet(app, theme=QT_MATERIAL_THEME)
    app.setStyleSheet(app.styleSheet() + MAIN_STYLESHEET)

    if not _validate_weights_path(args.weights):
        QMessageBox.critical(
            None,
            "Ошибка",
            f"Файл весов не найден:\n{args.weights}\n\nУкажите путь через --weights или YOLO_WEIGHTS_PATH.",
        )
        sys.exit(1)

    window = ImageEditor(weights_path=args.weights)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
