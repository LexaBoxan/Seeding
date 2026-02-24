"""Применение темы приложения через проектный QSS.

Опционально поддерживается ``qt-material``.
"""

from __future__ import annotations

import logging
import os

from PyQt5.QtWidgets import QApplication

from .styles import build_main_stylesheet

QT_MATERIAL_THEME_BY_NAME = {
    "dark": "dark_teal.xml",
    "light": "light_blue.xml",
}
_QT_MATERIAL_ENABLED = (
    os.getenv("SEEDING_USE_QT_MATERIAL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
)
logger = logging.getLogger(__name__)


def _get_qt_material():
    """Ленивая загрузка qt-material только при явном включении."""
    try:  # pragma: no cover - опциональная зависимость
        import qt_material

        return qt_material
    except ImportError:  # pragma: no cover - безопасный fallback
        return None


def apply_theme(app: QApplication, theme: str) -> None:
    """Применяет тему приложения.

    По умолчанию используется только проектный QSS, чтобы избежать проблем с
    внешними SVG-ресурсами qt-material. Включить qt-material можно через
    переменную окружения ``SEEDING_USE_QT_MATERIAL=1``.
    """
    project_stylesheet = build_main_stylesheet(theme)
    qt_material = _get_qt_material() if _QT_MATERIAL_ENABLED else None
    if qt_material is not None:
        material_theme = QT_MATERIAL_THEME_BY_NAME.get(
            theme,
            QT_MATERIAL_THEME_BY_NAME["dark"],
        )
        try:
            qt_material.apply_stylesheet(app, theme=material_theme)
            app.setStyleSheet(f"{app.styleSheet()}\n{project_stylesheet}")
            return
        except Exception:  # pragma: no cover
            logger.exception(
                "Не удалось применить qt-material, используется QSS.",
            )

    app.setStyleSheet(project_stylesheet)
