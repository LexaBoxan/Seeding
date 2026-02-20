"""Метрики интерфейса для единого визуального ритма."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UiMetrics:
    """Централизованные размеры и отступы UI."""

    toolbar_height: int = 42
    icon_size_toolbar: int = 24
    icon_size_toolbox: int = 24
    tool_button_size: int = 36
    toolbox_width: int = 64
    toolbar_spacing: int = 6
    padding_s: int = 6
    padding_m: int = 10
    padding_l: int = 14
    radius: int = 10
    panel_min_width: int = 260
