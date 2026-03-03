"""Модели данных и состояния приложения.

Пакет объединён в один модуль для упрощения структуры проекта.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

import numpy as np
from PIL import Image

BBox = tuple[int, int, int, int]


@dataclass
class AllClassImage:
    """Информация о выделенной части растения."""

    class_name: str
    confidence: float
    image: np.ndarray | Image.Image
    bbox: BBox | None = None


@dataclass
class ObjectImage:
    """Информация о найденном сеянце и его классификации."""

    class_name: str
    confidence: float
    image: list[np.ndarray | Image.Image] = field(default_factory=list)
    image_all_class: list[AllClassImage] | None = None
    bbox: BBox | None = None
    rotation_k: int = 0


@dataclass
class OriginalImage:
    """Контейнер для исходных изображений и результатов анализа."""

    file_path: str = ""
    source_files: list[str] = field(default_factory=list)
    images: list[np.ndarray | Image.Image] = field(default_factory=list)
    masks: list[np.ndarray | Image.Image] = field(default_factory=list)
    final_images: list[np.ndarray | Image.Image] = field(default_factory=list)
    class_object_image: list[list[ObjectImage]] | None = None


class SelectionPayload(TypedDict, total=False):
    """Данные выбранного элемента дерева для операций контроллера."""

    type: Literal["original", "pdf", "seeding", "class"]
    index: int
    parent_index: int
    seeding_index: int
    class_index: int


@dataclass
class RotateSelectionResult:
    """Результат операции поворота выбранного элемента."""

    target: Literal["page", "crop"]
    page_index: int
    image: np.ndarray
    crop_index: int | None = None


@dataclass
class MeasurementRecord:
    """Запись результата измерения объекта в пикселях и миллиметрах."""

    timestamp: str
    source_file: str
    page_index: int
    object_index: int
    width_px: int
    height_px: int
    diagonal_px: float
    pixels_per_mm: float
    width_mm: float | None = None
    height_mm: float | None = None
    diagonal_mm: float | None = None


@dataclass
class AppState:
    """Состояние приложения, разделяемое между UI и логикой."""

    image_storage: OriginalImage = field(default_factory=OriginalImage)
    active_image_index: int = 0
    selected_item: SelectionPayload | None = None
    zoom_factor: float = 1.0
    last_report_path: str = ""
    report_dir: str = ""
    pixels_per_mm: float = 0.0
    use_cache: bool = True


__all__ = [
    "AllClassImage",
    "AppState",
    "BBox",
    "MeasurementRecord",
    "ObjectImage",
    "OriginalImage",
    "RotateSelectionResult",
    "SelectionPayload",
]
