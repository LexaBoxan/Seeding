"""Минимальный runtime-словарь переводов интерфейса."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "ru": {
        "window_title": "Анализ сеянцев",
        "menu_file": "Файл",
        "menu_view": "Вид",
        "menu_tools": "Анализ",
        "menu_open": "Открыть файл",
        "dock_layers": "Слои",
        "tab_layers": "Слои",
        "tab_properties": "Свойства",
        "group_layers": "Слои",
        "group_info": "Свойства",
        "tool_select": "Выбор (V)",
        "tool_hand": "Панорамирование (H, Space)",
        "tool_zoom": "Лупа (Z)",
        "status_model_loading": "Загрузка модели детекции...",
        "status_model_ready": "Модель загружена",
        "status_model_error": "Ошибка загрузки модели",
        "status_model_missing": "Модель не загружена",
        "empty_state_title": "Перетащите файл сюда или откройте через кнопку",
        "empty_state_hint": "Поддерживаются изображения и PDF-документы",
        "empty_state_open": "Открыть файл",
    },
    "en": {
        "window_title": "Seedling Analyzer",
        "menu_file": "File",
        "menu_view": "View",
        "menu_tools": "Analyze",
        "menu_open": "Open file",
        "dock_layers": "Layers",
        "tab_layers": "Layers",
        "tab_properties": "Properties",
        "group_layers": "Layers",
        "group_info": "Properties",
        "tool_select": "Select (V)",
        "tool_hand": "Pan (H, Space)",
        "tool_zoom": "Zoom (Z)",
        "status_model_loading": "Loading detection model...",
        "status_model_ready": "Model loaded",
        "status_model_error": "Model loading error",
        "status_model_missing": "Model is not loaded",
        "empty_state_title": "Drop files here or use the open button",
        "empty_state_hint": "Images and PDF documents are supported",
        "empty_state_open": "Open file",
    },
}


def tr(language: str, key: str, fallback: str = "") -> str:
    """Возвращает перевод по ключу и языку."""
    locale = TRANSLATIONS.get(language) or TRANSLATIONS["ru"]
    if key in locale:
        return locale[key]
    if fallback:
        return fallback
    return key
