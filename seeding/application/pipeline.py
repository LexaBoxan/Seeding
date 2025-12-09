"""Конвейер обработки сеянцев с анализом корневой системы."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - зависит от среды
    from ultralytics import YOLO
except Exception:  # pragma: no cover - среда без ultralytics
    YOLO = None  # type: ignore

from seeding.utils import simple_nms
from .root_analysis import RootAnalyzer, RootAnalysisResult


@dataclass(slots=True)
class SeedlingDetection:
    """Данные по одному найденному сеянцу."""

    crop: np.ndarray
    bbox: tuple[int, int, int, int]
    confidence: float
    rotation_k: int
    roots: List[RootAnalysisResult]


class SeedlingPipeline:
    """Оркестратор полного цикла обработки изображений сеянцев."""

    def __init__(
        self,
        detection_weights: str | Path = "yolov8m.pt",
        detection_model: Optional[YOLO] = None,
        root_analyzer: Optional[RootAnalyzer] = None,
    ) -> None:
        self.detection_weights = Path(detection_weights)
        self.model = detection_model or self._load_model(self.detection_weights)
        self.root_analyzer = root_analyzer or RootAnalyzer()

    def _load_model(self, weights: Path):  # pragma: no cover - зависит от среды
        if YOLO is None:
            return None
        try:
            return YOLO(str(weights))
        except FileNotFoundError:
            return YOLO("yolov8m.pt")

    def process(self, images: Iterable[np.ndarray]) -> List[List[SeedlingDetection]]:
        """Запустить конвейер: детекция сеянцев и анализ корней."""

        results: List[List[SeedlingDetection]] = []
        for image in images:
            detections = self._detect_seedlings(image)
            results.append(detections)
        return results

    def _detect_seedlings(self, image: np.ndarray) -> List[SeedlingDetection]:
        if image is None or self.model is None:
            return []

        yolo_results = self.model(image)
        model_names = yolo_results[0].names

        boxes = []
        scores = []
        payload = []
        for box_idx, box in enumerate(yolo_results[0].boxes):
            class_id = int(box.cls)
            class_name = model_names[class_id]
            if class_name.lower() != "seeding":
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
            box_idx, bbox = payload[idx]
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2].copy()
            rotation_k = 0
            if crop.shape[1] > crop.shape[0]:
                crop = np.rot90(crop, k=1)
                rotation_k = 1

            roots = self._extract_roots(yolo_results[0], box_idx, crop, bbox)
            detections.append(
                SeedlingDetection(
                    crop=crop,
                    bbox=bbox,
                    confidence=scores[idx],
                    rotation_k=rotation_k,
                    roots=roots,
                )
            )
        return detections

    def _extract_roots(
        self,
        yolo_result,
        seeding_index: int,
        crop: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> List[RootAnalysisResult]:
        if yolo_result.masks is None:
            return []

        masks = yolo_result.masks.data.cpu().numpy()
        boxes = yolo_result.boxes
        roots: List[RootAnalysisResult] = []
        for mask_idx, box in enumerate(boxes):
            class_id = int(box.cls)
            class_name = yolo_result.names[class_id].lower()
            if class_name != "root":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            seeding_x1, seeding_y1, seeding_x2, seeding_y2 = bbox
            if not self._is_overlapping(
                (seeding_x1, seeding_y1, seeding_x2, seeding_y2), (x1, y1, x2, y2)
            ):
                continue
            full_mask = (masks[mask_idx] * 255).astype(np.uint8)
            cropped_mask = full_mask[seeding_y1:seeding_y2, seeding_x1:seeding_x2]
            roots.append(
                self.root_analyzer.analyze_root(cropped_mask, crop, float(box.conf))
            )
        return roots

    @staticmethod
    def _is_overlapping(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        return inter_x2 > inter_x1 and inter_y2 > inter_y1


__all__ = ["SeedlingPipeline", "SeedlingDetection"]
