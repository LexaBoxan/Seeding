"""Dataclasses для хранения изображений и результатов детекции."""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
from PIL import Image


@dataclass
class AllClassImage:
    """
    Класс для хранения информации о выделенном классе на изображении.
    """
    class_name: str
    confidence: float
    image: Union[np.ndarray, Image.Image]
    bbox: tuple | None = None  # (x1, y1, x2, y2) относит. к кропу сеянца


@dataclass
class ObjectImage:
    """
    Класс для хранения информации о выделенном объекте на изображении.
    """
    class_name: str
    confidence: float
    image: List[Union[np.ndarray, Image.Image]] = field(default_factory=list)
    image_all_class: Optional[List[AllClassImage]] = None
    bbox: tuple = None  # (x1, y1, x2, y2)
    rotation_k: int = 0  # Поворот, применённый к crop


@dataclass
class OriginalImage:
    """
    Класс для хранения всей информации о загруженном исходном изображении.
    """
    file_path: str = ""
    images: List[Union[np.ndarray, Image.Image]] = field(default_factory=list)
    masks: List[Union[np.ndarray, Image.Image]] = field(default_factory=list)
    final_images: List[Union[np.ndarray, Image.Image]] = field(default_factory=list)
    class_object_image: Optional[List[List[ObjectImage]]] = None