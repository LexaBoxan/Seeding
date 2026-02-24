"""Локальное хранение кэша обработки и истории измерений."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from seeding.config import LOCAL_STORAGE_DIR
from seeding.models import AllClassImage, MeasurementRecord, ObjectImage
from seeding.utils import ensure_dir


class StorageService:
    """Сервис файлового хранения кэша и истории измерений."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        """Создаёт структуру локального хранилища.

        Параметры:
            root_dir: пользовательский путь для хранения кэша и истории.
                Если не задан, используется директория из конфигурации.
        """
        base_dir = (
            Path(root_dir) if root_dir is not None else LOCAL_STORAGE_DIR
        )
        self.root_dir = ensure_dir(base_dir)
        self.cache_dir = ensure_dir(self.root_dir / "cache")
        self.detection_cache_dir = ensure_dir(self.cache_dir / "detection")
        self.classification_cache_dir = ensure_dir(
            self.cache_dir / "classification"
        )
        self.history_path = self.root_dir / "measurements.jsonl"

    @staticmethod
    def build_detection_cache_key(
        *,
        source_file: str,
        page_index: int,
        image_shape: tuple[int, ...],
        image_checksum: int,
        detect_weights_path: str,
        conf_threshold: float,
        iou_threshold: float,
    ) -> str:
        """Формирует ключ кэша детекции на основе входных параметров."""
        payload = {
            "source_file": source_file,
            "page_index": page_index,
            "image_shape": list(image_shape),
            "image_checksum": int(image_checksum),
            "detect_weights_path": detect_weights_path,
            "conf_threshold": round(float(conf_threshold), 6),
            "iou_threshold": round(float(iou_threshold), 6),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    @staticmethod
    def build_classification_cache_key(
        *,
        source_file: str,
        page_index: int,
        object_index: int,
        object_bbox: tuple[int, int, int, int] | None,
        rotation_k: int,
        crop_checksum: int,
        classify_weights_path: str,
    ) -> str:
        """Формирует ключ кэша классификации для одного сеянца."""
        payload = {
            "source_file": source_file,
            "page_index": page_index,
            "object_index": object_index,
            "object_bbox": list(object_bbox) if object_bbox else None,
            "rotation_k": int(rotation_k),
            "crop_checksum": int(crop_checksum),
            "classify_weights_path": classify_weights_path,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def _detection_cache_path(self, cache_key: str) -> Path:
        """Возвращает путь к JSON-файлу кэша детекции по ключу."""
        return self.detection_cache_dir / f"{cache_key}.json"

    def _classification_cache_path(self, cache_key: str) -> Path:
        """Возвращает путь к JSON-файлу кэша классификации по ключу."""
        return self.classification_cache_dir / f"{cache_key}.json"

    def save_detection_objects(
        self,
        cache_key: str,
        objects: Iterable[ObjectImage],
    ) -> Path:
        """Сохраняет результаты детекции страницы в JSON-кэш."""
        payload = {
            "objects": [
                {
                    "class_name": obj.class_name,
                    "confidence": float(obj.confidence),
                    "bbox": list(obj.bbox) if obj.bbox else None,
                    "rotation_k": int(getattr(obj, "rotation_k", 0)),
                }
                for obj in objects
            ]
        }
        path = self._detection_cache_path(cache_key)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def load_detection_objects(
        self,
        cache_key: str,
    ) -> list[ObjectImage] | None:
        """Загружает результаты детекции страницы из JSON-кэша."""
        path = self._detection_cache_path(cache_key)
        if not path.is_file():
            return None

        payload = json.loads(path.read_text(encoding="utf-8"))
        objects: list[ObjectImage] = []
        for item in payload.get("objects", []):
            bbox_data = item.get("bbox")
            bbox = tuple(bbox_data) if bbox_data else None
            objects.append(
                ObjectImage(
                    class_name=str(item.get("class_name", "seeding")),
                    confidence=float(item.get("confidence", 0.0)),
                    image=[],
                    image_all_class=None,
                    bbox=bbox,
                    rotation_k=int(item.get("rotation_k", 0)),
                )
            )
        return objects

    def save_classification_parts(
        self,
        cache_key: str,
        parts: Iterable[AllClassImage],
    ) -> Path:
        """Сохраняет результаты классификации одного сеянца в JSON-кэш."""
        payload = {
            "parts": [
                {
                    "class_name": part.class_name,
                    "confidence": float(part.confidence),
                    "bbox": list(part.bbox) if part.bbox else None,
                }
                for part in parts
            ]
        }
        path = self._classification_cache_path(cache_key)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def load_classification_parts(
        self,
        cache_key: str,
    ) -> list[AllClassImage] | None:
        """Загружает результаты классификации одного сеянца из JSON-кэша."""
        path = self._classification_cache_path(cache_key)
        if not path.is_file():
            return None

        payload = json.loads(path.read_text(encoding="utf-8"))
        parts: list[AllClassImage] = []
        for item in payload.get("parts", []):
            bbox_data = item.get("bbox")
            bbox = tuple(bbox_data) if bbox_data else None
            parts.append(
                AllClassImage(
                    class_name=str(item.get("class_name", "")),
                    confidence=float(item.get("confidence", 0.0)),
                    image=np.empty((0, 0, 3), dtype=np.uint8),
                    bbox=bbox,
                )
            )
        return parts

    def clear_cache(self) -> int:
        """Удаляет все файлы кэша и возвращает их количество."""
        removed = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                path.unlink()
                removed += 1
        return removed

    def append_measurement(self, record: MeasurementRecord) -> Path:
        """Добавляет запись измерения в историю в формате JSONL."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False))
            fh.write("\n")
        return self.history_path

    def load_measurements(
        self,
        *,
        limit: int | None = None,
    ) -> list[MeasurementRecord]:
        """Загружает историю измерений.

        При передаче ``limit`` возвращаются только последние записи.
        """
        if not self.history_path.is_file():
            return []

        records: list[MeasurementRecord] = []
        with self.history_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                    records.append(MeasurementRecord(**payload))
                except (json.JSONDecodeError, TypeError):
                    continue

        if limit is not None and limit > 0:
            return records[-limit:]
        return records

    def export_measurements_csv(self, output_path: str | Path) -> Path:
        """Экспортирует историю измерений в CSV."""
        records = self.load_measurements()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "timestamp",
                    "source_file",
                    "page_index",
                    "object_index",
                    "width_px",
                    "height_px",
                    "diagonal_px",
                    "pixels_per_mm",
                    "width_mm",
                    "height_mm",
                    "diagonal_mm",
                ],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
        return path


__all__ = ["StorageService"]
