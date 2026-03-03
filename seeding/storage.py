"""Локальное хранение кэша обработки и истории измерений."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from seeding.config import LEGACY_LOCAL_STORAGE_DIR, LOCAL_STORAGE_DIR
from seeding.models import AllClassImage, MeasurementRecord, ObjectImage
from seeding.utils import ensure_dir

logger = logging.getLogger(__name__)


class StorageService:
    """Сервис файлового хранения кэша и истории измерений."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        """Создаёт структуру локального хранилища.

        Параметры:
            root_dir: пользовательский путь для хранения кэша и истории.
                Если не задан, используется директория из конфигурации.
        """
        base_dir = Path(root_dir) if root_dir is not None else LOCAL_STORAGE_DIR
        legacy_dir = None if root_dir is not None else LEGACY_LOCAL_STORAGE_DIR
        self.migrated_from_legacy = False
        self.migrated_files_count = 0
        self.root_dir = self._prepare_root_dir(base_dir, legacy_dir)
        self.cache_dir = ensure_dir(self.root_dir / "cache")
        self.detection_cache_dir = ensure_dir(self.cache_dir / "detection")
        self.classification_cache_dir = ensure_dir(
            self.cache_dir / "classification"
        )
        self.calibrations_path = self.root_dir / "calibrations.json"
        self.history_path = self.root_dir / "measurements.jsonl"

    @staticmethod
    def _directory_has_files(directory: Path) -> bool:
        """Возвращает ``True``, если в директории уже есть хотя бы один файл."""
        return directory.is_dir() and any(
            path.is_file() for path in directory.rglob("*")
        )

    def _prepare_root_dir(
        self,
        target_dir: Path,
        legacy_dir: Path | None,
    ) -> Path:
        """Подготавливает корневой каталог и при необходимости переносит старые данные."""
        root_dir = ensure_dir(target_dir)
        if legacy_dir is None:
            return root_dir

        migrated_files = self._migrate_legacy_storage(legacy_dir, root_dir)
        self.migrated_from_legacy = migrated_files > 0
        self.migrated_files_count = migrated_files
        return root_dir

    def _migrate_legacy_storage(
        self,
        legacy_dir: Path,
        target_dir: Path,
    ) -> int:
        """Копирует legacy-хранилище в новый user-data каталог, если он ещё пуст."""
        source_dir = legacy_dir.expanduser()
        if not source_dir.is_dir():
            return 0

        try:
            if source_dir.resolve() == target_dir.resolve():
                return 0
        except OSError:
            return 0

        if self._directory_has_files(target_dir):
            return 0

        migrated_files = 0
        try:
            for source_path in source_dir.rglob("*"):
                relative_path = source_path.relative_to(source_dir)
                target_path = target_dir / relative_path
                if source_path.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                migrated_files += 1
        except OSError:
            logger.exception(
                "Failed to migrate legacy storage from %s to %s",
                source_dir,
                target_dir,
            )
            return 0

        if migrated_files:
            logger.info(
                "Migrated %s storage files from %s to %s",
                migrated_files,
                source_dir,
                target_dir,
            )
        return migrated_files

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

    @staticmethod
    def _normalize_source_file(source_file: str | Path | None) -> str | None:
        """РџСЂРёРІРѕРґРёС‚ РїСѓС‚СЊ Рє РЅРѕСЂРјР°Р»РёР·РѕРІР°РЅРЅРѕРјСѓ РІРёРґСѓ РґР»СЏ РєР»СЋС‡РµР№ С…СЂР°РЅРµРЅРёСЏ."""
        if source_file is None:
            return None

        raw_path = str(source_file).strip()
        if not raw_path:
            return None

        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path.absolute()
        return os.path.normcase(str(resolved))

    def _load_calibrations_payload(self) -> dict[str, float]:
        """Р—Р°РіСЂСѓР¶Р°РµС‚ СЃР»РѕРІР°СЂСЊ РєР°Р»РёР±СЂРѕРІРѕРє РёР· JSON-С„Р°Р№Р»Р°."""
        if not self.calibrations_path.is_file():
            return {}

        try:
            payload = json.loads(
                self.calibrations_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            logger.exception(
                "Failed to load calibrations from %s",
                self.calibrations_path,
            )
            return {}

        if not isinstance(payload, dict):
            return {}

        calibrations: dict[str, float] = {}
        for raw_key, raw_value in payload.items():
            if not isinstance(raw_key, str):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                calibrations[raw_key] = value
        return calibrations

    def _write_calibrations_payload(
        self,
        calibrations: dict[str, float],
    ) -> Path | None:
        """РЎРѕС…СЂР°РЅСЏРµС‚ СЃР»РѕРІР°СЂСЊ РєР°Р»РёР±СЂРѕРІРѕРє РІ JSON."""
        if not calibrations:
            try:
                self.calibrations_path.unlink(missing_ok=True)
            except OSError:
                logger.exception(
                    "Failed to remove empty calibrations file %s",
                    self.calibrations_path,
                )
            return None

        self.calibrations_path.parent.mkdir(parents=True, exist_ok=True)
        self.calibrations_path.write_text(
            json.dumps(calibrations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.calibrations_path

    def save_calibration(
        self,
        source_file: str | Path,
        pixels_per_mm: float,
    ) -> Path | None:
        """РЎРѕС…СЂР°РЅСЏРµС‚ РєРѕСЌС„С„РёС†РёРµРЅС‚ РєР°Р»РёР±СЂРѕРІРєРё РґР»СЏ РєРѕРЅРєСЂРµС‚РЅРѕРіРѕ С„Р°Р№Р»Р°."""
        normalized_source = self._normalize_source_file(source_file)
        value = float(pixels_per_mm)
        if normalized_source is None or value <= 0:
            return None

        calibrations = self._load_calibrations_payload()
        calibrations[normalized_source] = value
        return self._write_calibrations_payload(calibrations)

    def load_calibration(self, source_file: str | Path) -> float | None:
        """Р’РѕР·РІСЂР°С‰Р°РµС‚ СЃРѕС…СЂР°РЅС‘РЅРЅСѓСЋ РєР°Р»РёР±СЂРѕРІРєСѓ РґР»СЏ С„Р°Р№Р»Р°."""
        normalized_source = self._normalize_source_file(source_file)
        if normalized_source is None:
            return None

        calibrations = self._load_calibrations_payload()
        value = calibrations.get(normalized_source)
        if value is None or value <= 0:
            return None
        return float(value)

    def clear_calibration(self, source_file: str | Path) -> bool:
        """РЈРґР°Р»СЏРµС‚ РєР°Р»РёР±СЂРѕРІРєСѓ РґР»СЏ С„Р°Р№Р»Р° Рё РІРѕР·РІСЂР°С‰Р°РµС‚ С„Р»Р°Рі РёР·РјРµРЅРµРЅРёСЏ."""
        normalized_source = self._normalize_source_file(source_file)
        if normalized_source is None:
            return False

        calibrations = self._load_calibrations_payload()
        if normalized_source not in calibrations:
            return False

        calibrations.pop(normalized_source, None)
        self._write_calibrations_payload(calibrations)
        return True

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
