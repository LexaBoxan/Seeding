"""Пакет UI с ленивыми экспортами для снижения побочных импортов."""

from __future__ import annotations

__all__ = ["ImageEditor", "LayerTreeWidget"]


def __getattr__(name: str):
    """Лениво возвращает UI-классы по имени атрибута пакета.

    Такой механизм уменьшает побочные импорты при старте и оставляет
    совместимый внешний API: ``from seeding.ui import ImageEditor``.
    """
    if name == "ImageEditor":
        from .main_window import ImageEditor

        return ImageEditor
    if name == "LayerTreeWidget":
        from .tree_widget import LayerTreeWidget

        return LayerTreeWidget
    raise AttributeError(f"Модуль {__name__!r} не содержит атрибут {name!r}")
