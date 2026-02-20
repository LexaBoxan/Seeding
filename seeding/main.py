"""Точка входа в графическое приложение Seeding."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMessageBox

from seeding.config import (
    APP_FONT_FAMILY,
    APP_FONT_SIZE,
    DEFAULT_WEIGHTS_PATH,
    PROJECT_ROOT,
)
from seeding.path_utils import resolve_weights_path
from seeding.ui.preferences import load_ui_preferences
from seeding.ui.main_window import ImageEditor
from seeding.ui.theme_manager import apply_theme


def _resolve_weights_path(path_value: str) -> str | None:
    """Разрешает и проверяет путь/алиас весов модели."""
    resolved = resolve_weights_path(
        path_value,
        base_dirs=(PROJECT_ROOT, Path.cwd()),
    )
    if resolved is None:
        return None
    return str(resolved)


def _validate_weights_path(path_value: str) -> bool:
    """Возвращает ``True``, если путь к весам корректен."""
    return _resolve_weights_path(path_value) is not None


def main() -> None:
    """Запускает Qt-приложение."""
    parser = argparse.ArgumentParser(description="Seeding")
    parser.add_argument(
        "--weights",
        default=os.getenv("YOLO_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH)),
        help="Путь к весам YOLO (.pt) или алиас модели",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
    )

    app = QApplication(sys.argv)
    app.setFont(QFont(APP_FONT_FAMILY, APP_FONT_SIZE))
    ui_preferences = load_ui_preferences()
    apply_theme(app, ui_preferences.theme)

    resolved_weights = _resolve_weights_path(args.weights)
    if resolved_weights is None:
        QMessageBox.critical(
            None,
            "Ошибка пути к весам",
            (
                f"Не удалось найти веса модели:\n{args.weights}\n\n"
                "Укажите корректный путь через --weights или "
                "YOLO_WEIGHTS_PATH, либо существующий файл <name>.pt."
            ),
        )
        sys.exit(1)

    window = ImageEditor(weights_path=resolved_weights)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
