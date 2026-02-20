"""Применение темы приложения через qt-material и QSS."""

from __future__ import annotations

from PyQt5.QtWidgets import QApplication

from .styles import build_main_stylesheet

try:  # pragma: no cover - опциональная зависимость
    import qt_material
except ImportError:  # pragma: no cover - безопасный fallback
    qt_material = None

QT_MATERIAL_THEME_BY_NAME = {
    "dark": "dark_teal.xml",
    "light": "light_blue.xml",
}


def apply_theme(app: QApplication, theme: str) -> None:
    """Применяет тему qt-material и проектный QSS."""
    if qt_material is not None:
        material_theme = QT_MATERIAL_THEME_BY_NAME.get(
            theme,
            QT_MATERIAL_THEME_BY_NAME["dark"],
        )
        qt_material.apply_stylesheet(app, theme=material_theme)
        app.setStyleSheet(app.styleSheet() + build_main_stylesheet(theme))
        return

    app.setStyleSheet(build_main_stylesheet(theme))
