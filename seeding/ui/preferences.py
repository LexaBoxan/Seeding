"""Вспомогательные функции для настроек интерфейса в QSettings."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import QSettings

from seeding.config import QSETTINGS_APP, QSETTINGS_ORG

DEFAULT_UI_THEME = "dark"
DEFAULT_UI_LANGUAGE = "ru"


@dataclass(frozen=True)
class UiPreferences:
    """Пользовательские параметры интерфейса."""

    theme: str = DEFAULT_UI_THEME
    language: str = DEFAULT_UI_LANGUAGE


def load_ui_preferences() -> UiPreferences:
    """Читает тему и язык интерфейса из QSettings."""
    settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    return UiPreferences(
        theme=settings.value("ui_theme", DEFAULT_UI_THEME, type=str),
        language=settings.value("ui_language", DEFAULT_UI_LANGUAGE, type=str),
    )


def save_ui_preferences(*, theme: str, language: str) -> None:
    """Сохраняет тему и язык интерфейса в QSettings."""
    settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    settings.setValue("ui_theme", theme)
    settings.setValue("ui_language", language)
    settings.sync()
