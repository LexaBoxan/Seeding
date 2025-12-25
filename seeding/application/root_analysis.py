"""Модуль оценки морфологии корневой системы."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from seeding.utils import logger


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
    """Анализ морфологии корня по бинарной маске."""

    def __init__(self, viability_threshold: float = 0.6) -> None:
        self.viability_threshold_high = viability_threshold
        self.viability_threshold_low = 0.4

    def analyze_root(
        self, mask: np.ndarray, seedling_crop: Optional[np.ndarray] = None, confidence: float = 1.0
    ) -> RootAnalysisResult:
        """Основной метод анализа."""
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
        """Подготовка маски: перевод в бинарный вид, очистка шума."""
        if mask is None or mask.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Сглаживание шума
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Морфологическое открытие для удаления мелкого шума
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    @staticmethod
    def _get_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Ограничивающий прямоугольник главного контура."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, mask.shape[1], mask.shape[0]

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, x + w, y + h

    def _compute_morphology(self, mask: np.ndarray) -> RootMorphology:
        """Расчёт морфологических метрик."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return RootMorphology(0.0, 0.0, 0.0, 0.0, 0.0)

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = max(float(w * h), 1.0)

        # Улучшенная длина: по количеству пикселей скелета
        skeleton = self._skeletonize(mask)
        length_est = float(cv2.countNonZero(skeleton))

        mean_thickness = area / length_est if length_est > 0 else 0.0
        branching_index = self._branching_index(skeleton)

        # Кривизна (компактность)
        curvature = (perimeter ** 2) / (4.0 * np.pi * area) if area > 0 else 10.0

        density = area / rect_area

        return RootMorphology(
            length=round(length_est, 2),
            mean_thickness=round(mean_thickness, 3),
            branching_index=round(branching_index, 3),
            curvature=round(curvature, 3),
            density=round(density, 3),
        )

    @staticmethod
    def _skeletonize(mask: np.ndarray) -> np.ndarray:
        """Тонкий скелет маски. Безопасен при пустой или нулевой маске."""
        if mask is None or mask.size == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
            return np.zeros_like(mask) if mask is not None else np.zeros((1, 1), dtype=np.uint8)

        if np.count_nonzero(mask) == 0:
            return np.zeros(mask.shape, dtype=np.uint8)

        img = mask.copy()
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        try:
            while True:
                non_zero = cv2.countNonZero(img)
                if non_zero == 0:
                    break

                open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
                temp = cv2.subtract(img, open_img)
                eroded = cv2.erode(img, element)
                skel = cv2.bitwise_or(skel, temp)
                img = eroded.copy()

                # Защита от бесконечного цикла (на всякий случай)
                if cv2.countNonZero(img) == non_zero:
                    break

            return skel

        except Exception as e:
            logger.warning(f"Ошибка в _skeletonize: {e}. Возвращаем пустой скелет.")
            return np.zeros(mask.shape, dtype=np.uint8)

    @staticmethod
    def _branching_index(skeleton: np.ndarray) -> float:
        """Индекс ветвистости: количество узлов ветвления / количество концов."""
        if skeleton.size == 0 or np.count_nonzero(skeleton) == 0:
            return 0.0

        points = np.argwhere(skeleton > 0)
        if len(points) == 0:
            return 0.0

        padded = np.pad(skeleton > 0, pad_width=1, mode="constant")
        branches = 0
        endpoints = 0

        for y, x in points:
            neighborhood = padded[y:y+3, x:x+3]
            neighbors = int(np.sum(neighborhood)) - 1
            if neighbors > 2:
                branches += 1
            elif neighbors == 1:
                endpoints += 1

        return branches / max(endpoints, 1)

    def _estimate_viability(
        self, morphology: RootMorphology, confidence: float
    ) -> tuple[RootViability, float]:
        """Оценка жизнеспособности на основе метрик."""
        if morphology.length < 20 or morphology.mean_thickness < 2 or morphology.density < 0.05:
            return RootViability.NOT_RECOGNIZED, 0.0

        # Нормализация метрик (можно подстроить под ваши сеянцы)
        norm_length = min(morphology.length / 100.0, 1.0)
        norm_thickness = min(morphology.mean_thickness / 8.0, 1.0)
        norm_branching = min(morphology.branching_index / 0.3, 1.0)
        norm_density = min(morphology.density / 0.4, 1.0)

        score = (
            0.4 * norm_length +
            0.3 * norm_thickness +
            0.2 * norm_branching +
            0.1 * norm_density
        )

        # Учёт уверенности модели сегментации
        final_score = 0.8 * score + 0.2 * confidence

        if final_score >= self.viability_threshold_high:
            return RootViability.VIABLE, round(final_score, 3)
        elif final_score >= self.viability_threshold_low:
            return RootViability.CRITICAL, round(final_score, 3)
        else:
            return RootViability.NOT_RECOGNIZED, round(final_score, 3)

    def visualize(self, result: RootAnalysisResult) -> np.ndarray:
        """Визуализация анализа корня.
        Абсолютно безопасна при любой маске (пустой, нулевой, None)."""
        mask = result.mask if result.mask is not None else np.zeros((1, 1), dtype=np.uint8)

        # Если маска пустая или имеет нулевой размер — возвращаем заглушку
        if mask.size == 0 or mask.shape[0] == 0 or mask.shape[1] == 0 or np.count_nonzero(mask) == 0:
            # Создаём фиксированное изображение-заглушку
            vis = np.zeros((400, 500, 3), dtype=np.uint8)
            vis[:] = (45, 45, 55)  # тёмно-синий фон
            cv2.putText(vis, "Корень не обнаружен", (70, 170),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (220, 220, 220), 3, cv2.LINE_AA)
            cv2.putText(vis, "в данном сеянце", (140, 230),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
            cv2.putText(vis, "(маска отсутствует или пустая)", (60, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 130, 130), 1, cv2.LINE_AA)
            return vis

        # Нормальный случай — есть содержимое
        try:
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Скелет — красный
            skeleton = self._skeletonize(mask)
            if skeleton.size > 0:
                vis[skeleton > 0] = (0, 0, 255)

            # Контур — зелёный
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                thickness = max(2, min(mask.shape[0], mask.shape[1]) // 150)
                cv2.drawContours(vis, contours, -1, (0, 255, 0), thickness)

            return vis

        except Exception as e:
            logger.warning(f"Критическая ошибка в visualize: {e}. Возвращаем заглушку.")
            # Абсолютная защита — даже если OpenCV упал
            vis = np.zeros((400, 500, 3), dtype=np.uint8)
            vis[:] = (30, 30, 100)
            cv2.putText(vis, "Ошибка визуализации", (80, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 100, 255), 3)
            return vis