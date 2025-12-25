"""Конвейер обработки сеянцев с анализом корневой системы."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

from seeding.utils import simple_nms
from .root_analysis import RootAnalyzer, RootAnalysisResult, RootViability, RootMorphology


@dataclass(slots=True)
class SeedlingDetection:
    """Данные по одному найденному сеянцу."""
    crop: np.ndarray
    bbox: tuple[int, int, int, int]
    confidence: float
    rotation_k: int
    roots: List[RootAnalysisResult] = field(default_factory=list)
    parts: List[dict] = field(default_factory=list)  # Все части: root, stem, flower


class SeedlingPipeline:
    """Оркестратор полного цикла: 1. детекция сеянцев → 2. сегментация частей → 3. классификация корня."""

    def __init__(
        self,
        detection_weights: str | Path = "yolov8m.pt",
        classify_weights: Optional[str | Path] = None,
        root_classify_weights: Optional[str | Path] = None,  # Третья модель (cls)
        detection_model: Optional[YOLO] = None,
        classify_model: Optional[YOLO] = None,
        root_classify_model: Optional[YOLO] = None,
        root_analyzer: Optional[RootAnalyzer] = None,
    ) -> None:
        self.detection_weights = Path(detection_weights)
        self.classify_weights = Path(classify_weights) if classify_weights else None
        self.root_classify_weights = Path(root_classify_weights) if root_classify_weights else None

        self.detection_model = detection_model or self._load_model(self.detection_weights)
        self.classify_model = classify_model or (
            self._load_model(self.classify_weights) if self.classify_weights else None
        )
        self.root_classify_model = root_classify_model or (
            self._load_model(self.root_classify_weights) if self.root_classify_weights else None
        )
        self.root_analyzer = root_analyzer or RootAnalyzer()

    def _load_model(self, weights: Path) -> Optional[YOLO]:
        """Безопасная загрузка модели с проверкой существования файла."""
        if YOLO is None:
            logger.warning("Ultralytics не установлен — модели не будут загружены.")
            return None

        if weights is None:
            logger.info("Путь к модели не указан — пропуск загрузки.")
            return None

        if not weights.exists():
            logger.warning(f"Файл модели не найден: {weights}. Модель не будет использована.")
            return None

        try:
            model = YOLO(str(weights))
            logger.info(f"Модель успешно загружена: {weights}")
            return model
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {weights}: {e}")
            return None

    def process(self, images: Iterable[np.ndarray]) -> List[List[SeedlingDetection]]:
        """Полный конвейер по всем изображениям."""
        results: List[List[SeedlingDetection]] = []
        for image in images:
            detections = self._detect_seedlings(image)
            results.append(detections)
        return results

    def _detect_seedlings(self, image: np.ndarray) -> List[SeedlingDetection]:
        """Детекция сеянцев первой моделью."""
        if image is None or self.detection_model is None:
            logger.warning("Изображение пустое или модель детекции не загружена.")
            return []

        try:
            yolo_results = self.detection_model(image)[0]
        except Exception as e:
            logger.error(f"Ошибка при детекции сеянцев: {e}")
            return []

        model_names = yolo_results.names

        boxes = []
        scores = []
        payload = []

        for box_idx, box in enumerate(yolo_results.boxes):
            class_id = int(box.cls)
            class_name = model_names[class_id].lower()
            if class_name != "seeding":
                continue

            score = float(box.conf)
            x_center, y_center, width, height = box.xywh[0].cpu().numpy()
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(x2, w)
            y1, y2 = max(0, y1), min(y2, h)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            payload.append((box_idx, (x1, y1, x2, y2)))

        indices = simple_nms(boxes, scores, iou_threshold=0.4)
        detections: List[SeedlingDetection] = []

        for idx in indices:
            _, bbox = payload[idx]
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2].copy()

            rotation_k = 0
            if crop.shape[1] > crop.shape[0]:
                crop = np.rot90(crop, k=1)
                rotation_k = 1

            roots = []
            parts = []
            if self.classify_model:
                try:
                    classify_results = self.classify_model(crop)[0]
                    roots, parts = self._extract_parts(classify_results, crop)
                except Exception as e:
                    logger.error(f"Ошибка при сегментации частей сеянца: {e}")

            detections.append(
                SeedlingDetection(
                    crop=crop,
                    bbox=bbox,
                    confidence=scores[idx],
                    rotation_k=rotation_k,
                    roots=roots,
                    parts=parts,
                )
            )
        return detections

    def _extract_parts(
            self, yolo_result, crop: np.ndarray
    ) -> tuple[List[RootAnalysisResult], List[dict]]:
        """Извлечение частей (root, stem, flower) и анализ корня.
        Приоритет — третья модель (best_root_cls) на ВЕСЬ crop сеянца."""
        roots: List[RootAnalysisResult] = []
        parts: List[dict] = []

        # Основная оценка от третьей модели (best_root_cls.pt) на полном изображении
        primary_viability = RootViability.NOT_RECOGNIZED
        primary_score = 0.0
        primary_conf = 0.0
        used_third_model = False

        if self.root_classify_model:
            try:
                cls_result = self.root_classify_model(crop)[0]
                top_class_id = int(cls_result.probs.top1)
                top_class_name = cls_result.names[top_class_id].lower()
                top_conf = float(cls_result.probs.top1conf)

                if top_class_name == "good":
                    primary_viability = RootViability.VIABLE
                elif top_class_name == "bad":
                    primary_viability = RootViability.CRITICAL

                primary_score = top_conf
                primary_conf = top_conf
                used_third_model = True

                logger.info(
                    f"Третья модель (best_root_cls) классифицировала весь сеянец как '{top_class_name}' "
                    f"с уверенностью {top_conf:.2f}"
                )
            except Exception as e:
                logger.warning(f"Ошибка третьей модели на полном сеянце: {e}. Используем fallback.")

        # Обработка сегментационных частей — для визуализации и метрик
        if yolo_result.masks is not None and len(yolo_result.masks) > 0:
            masks = yolo_result.masks.data.cpu().numpy()
            boxes = yolo_result.boxes

            for mask_idx, box in enumerate(boxes):
                class_id = int(box.cls)
                class_name = yolo_result.names[class_id].lower()
                conf = float(box.conf)

                mask = (masks[mask_idx] * 255).astype(np.uint8)
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                part_crop = crop[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else crop

                parts.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "mask": mask,
                    "image": part_crop,
                })

                if class_name == "root":
                    morph_result = self.root_analyzer.analyze_root(mask, crop, conf)

                    # Если третья модель не сработала — берём из морфологии
                    viability = primary_viability if used_third_model else morph_result.viability
                    score = primary_score if used_third_model else morph_result.score
                    confidence = primary_conf if used_third_model else conf

                    roots.append(
                        RootAnalysisResult(
                            mask=morph_result.mask,
                            bbox=morph_result.bbox,
                            morphology=morph_result.morphology,
                            viability=viability,
                            score=score,
                            confidence=confidence,
                        )
                    )
        else:
            # Нет масок — создаём пустой результат
            logger.info("Нет масок сегментации — создаём пустой анализ корня")

        # Если корень не найден — добавляем пустой результат (чтобы UI не падал)
        if not roots:
            empty_morphology = RootMorphology(
                length=0.0,
                mean_thickness=0.0,
                branching_index=0.0,
                curvature=0.0,
                density=0.0,
            )
            roots.append(
                RootAnalysisResult(
                    mask=np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8),
                    bbox=(0, 0, crop.shape[1], crop.shape[0]),
                    morphology=empty_morphology,
                    viability=primary_viability if used_third_model else RootViability.NOT_RECOGNIZED,
                    score=primary_score if used_third_model else 0.0,
                    confidence=primary_conf if used_third_model else 0.0,
                )
            )

        return roots, parts

    @staticmethod
    def _is_overlapping(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


__all__ = ["SeedlingPipeline", "SeedlingDetection"]