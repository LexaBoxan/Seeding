"""Базовые заглушки модуля обработки изображений."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Класс-оболочка для будущих алгоритмов обработки изображений."""

    @staticmethod
    def kmeans_segmentation(image: Any, k: int = 3) -> tuple[None, None]:
        """Заглушка K-Means сегментации.

        Метод оставлен для совместимости API. До реализации возвращает
        ``(None, None)`` и пишет информационную запись в лог.
        """
        logger.info(
            "ImageProcessor.kmeans_segmentation: метод пока не реализован",
        )
        return None, None
