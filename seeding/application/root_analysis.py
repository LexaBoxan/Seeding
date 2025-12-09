"""Модуль оценки морфологии корневой системы.

В модуле реализован класс :class:`RootAnalyzer`, который принимает
сегментированные маски корней и рассчитывает базовые морфологические
характеристики: длину, среднюю толщину, ветвистость, форму и плотность.
Также на основе этих метрик вычисляется грубая оценка жизнеспособности
корня. Модель YOLOv8 используется только как источник масок, поэтому
анализ можно применять к любым бинарным сегментациям.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


class RootViability(str, Enum):
    """Категории жизнеспособности корня."""

    VIABLE = "корень жизнеспособен"
    CRITICAL = "корень на гране смерти"
    NOT_RECOGNIZED = "корень не распознан"


@dataclass(slots=True)
class RootMorphology:
    """Морфологические показатели корневой системы."""

    length: float
    mean_thickness: float
    branching_index: float
    curvature: float
    density: float


@dataclass(slots=True)
class RootAnalysisResult:
    """Результат анализа одного корня."""

    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    morphology: RootMorphology
    viability: RootViability
    score: float
    confidence: float


class RootAnalyzer:
    """Вычисляет морфологические показатели по бинарной маске корня."""

    def __init__(self, viability_threshold: float = 0.35) -> None:
        self.viability_threshold = viability_threshold

    def analyze_root(
        self, mask: np.ndarray, seedling_crop: Optional[np.ndarray] = None, confidence: float = 1.0
    ) -> RootAnalysisResult:
        """Рассчитать метрики и статус жизнеспособности корня.

        Args:
            mask: Бинарная маска корня (любого размера).
            seedling_crop: Исходный кроп сеянца. Используется только для валидации размеров.
            confidence: Уверенность детекции маски, переданная из модели сегментации.

        Returns:
            :class:`RootAnalysisResult` с заполненными полями.
        """

        binary_mask = self._prepare_mask(mask)
        bbox = self._get_bbox(binary_mask)
        morphology = self._compute_morphology(binary_mask)
        viability, score = self._estimate_viability(morphology, confidence)

        return RootAnalysisResult(
            mask=binary_mask,
            bbox=bbox,
            morphology=morphology,
            viability=viability,
            score=score,
            confidence=float(confidence),
        )

    @staticmethod
    def _prepare_mask(mask: np.ndarray) -> np.ndarray:
        if mask is None:
            return np.zeros((1, 1), dtype=np.uint8)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _get_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, mask.shape[1], mask.shape[0]
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return x, y, x + w, y + h

    def _compute_morphology(self, mask: np.ndarray) -> RootMorphology:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return RootMorphology(0.0, 0.0, 0.0, 0.0, 0.0)

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = max(float(w * h), 1.0)

        length_est = max(float(max(w, h)), perimeter / 2.0)
        mean_thickness = float(area / length_est) if length_est > 0 else 0.0

        skeleton = self._skeletonize(mask)
        branching_index = self._branching_index(skeleton)

        curvature = (perimeter ** 2) / (4.0 * np.pi * area) if area > 0 else 0.0
        density = float(area / rect_area)

        return RootMorphology(
            length=round(length_est, 2),
            mean_thickness=round(mean_thickness, 3),
            branching_index=round(branching_index, 3),
            curvature=round(curvature, 3),
            density=round(density, 3),
        )

    @staticmethod
    def _skeletonize(mask: np.ndarray) -> np.ndarray:
        img = mask.copy()
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, open_img)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

    @staticmethod
    def _branching_index(skeleton: np.ndarray) -> float:
        if skeleton.size == 0:
            return 0.0
        # Оцениваем количество узлов со степенью больше двух
        skeleton_points = np.argwhere(skeleton > 0)
        if skeleton_points.size == 0:
            return 0.0

        padded = np.pad((skeleton > 0).astype(np.uint8), pad_width=1, mode="constant")
        branches = 0
        endpoints = 0
        for (y, x) in skeleton_points:
            neighborhood = padded[y : y + 3, x : x + 3]
            neighbors = int(np.sum(neighborhood) - 1)
            if neighbors > 2:
                branches += 1
            elif neighbors == 1:
                endpoints += 1
        return branches / max(endpoints, 1)

    def _estimate_viability(self, morphology: RootMorphology, confidence: float) -> tuple[RootViability, float]:
        if morphology.length == 0 or morphology.mean_thickness == 0:
            return RootViability.NOT_RECOGNIZED, 0.0

        score = 0.0
        score += min(morphology.length / 50.0, 1.0) * 0.35
        score += min(morphology.density / 0.5, 1.0) * 0.25
        score += min(morphology.branching_index / 0.2, 1.0) * 0.2
        score += min(confidence, 1.0) * 0.2

        if score >= max(self.viability_threshold, 0.65):
            return RootViability.VIABLE, score
        if score >= self.viability_threshold:
            return RootViability.CRITICAL, score
        return RootViability.NOT_RECOGNIZED, score
