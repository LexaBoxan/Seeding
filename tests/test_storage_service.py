from pathlib import Path

import numpy as np

import seeding.storage as storage_module
from seeding.models import AllClassImage, MeasurementRecord, ObjectImage
from seeding.storage import StorageService


def test_detection_cache_roundtrip(tmp_path):
    service = StorageService(tmp_path)
    key = service.build_detection_cache_key(
        source_file="sample.jpg",
        page_index=0,
        image_shape=(100, 120, 3),
        image_checksum=123456,
        detect_weights_path="models/bestCrop.pt",
        conf_threshold=0.25,
        iou_threshold=0.4,
    )

    source_objects = [
        ObjectImage(
            class_name="seeding",
            confidence=0.91,
            image=[],
            bbox=(10, 20, 40, 80),
            rotation_k=3,
        )
    ]
    service.save_detection_objects(key, source_objects)
    loaded = service.load_detection_objects(key)

    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].bbox == (10, 20, 40, 80)
    assert loaded[0].rotation_k == 3


def test_classification_cache_roundtrip(tmp_path):
    service = StorageService(tmp_path)
    key = service.build_classification_cache_key(
        source_file="sample.jpg",
        page_index=1,
        object_index=2,
        object_bbox=(5, 6, 20, 25),
        rotation_k=1,
        crop_checksum=9876,
        classify_weights_path="models/bestKlassSeg.pt",
    )

    parts = [
        AllClassImage(
            class_name="stem",
            confidence=0.83,
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            bbox=(1, 1, 3, 3),
        )
    ]
    service.save_classification_parts(key, parts)
    loaded = service.load_classification_parts(key)

    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].class_name == "stem"
    assert loaded[0].bbox == (1, 1, 3, 3)


def test_measurement_history_append_load_and_export(tmp_path):
    service = StorageService(tmp_path)
    record = MeasurementRecord(
        timestamp="2026-02-20T10:11:12",
        source_file="sample.jpg",
        page_index=0,
        object_index=0,
        width_px=10,
        height_px=20,
        diagonal_px=22.36,
        pixels_per_mm=5.0,
        width_mm=2.0,
        height_mm=4.0,
        diagonal_mm=4.47,
    )
    service.append_measurement(record)

    loaded_all = service.load_measurements()
    loaded_one = service.load_measurements(limit=1)

    assert len(loaded_all) == 1
    assert len(loaded_one) == 1
    assert loaded_all[0].diagonal_px == 22.36

    output_csv = Path(tmp_path) / "history.csv"
    saved = service.export_measurements_csv(output_csv)
    assert saved.is_file()
    content = saved.read_text(encoding="utf-8")
    assert "timestamp" in content
    assert "sample.jpg" in content


def test_calibration_roundtrip_uses_normalized_source_path(tmp_path):
    service = StorageService(tmp_path)
    source_file = tmp_path / "docs" / "sample.pdf"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("stub", encoding="utf-8")

    saved_path = service.save_calibration(source_file, 12.5)

    assert saved_path == service.calibrations_path
    assert service.load_calibration(source_file) == 12.5


def test_clear_calibration_preserves_other_entries(tmp_path):
    service = StorageService(tmp_path)
    first_file = tmp_path / "first.png"
    second_file = tmp_path / "second.png"
    first_file.write_text("a", encoding="utf-8")
    second_file.write_text("b", encoding="utf-8")

    service.save_calibration(first_file, 5.0)
    service.save_calibration(second_file, 7.0)

    assert service.clear_calibration(first_file) is True
    assert service.load_calibration(first_file) is None
    assert service.load_calibration(second_file) == 7.0
    assert service.clear_calibration(first_file) is False


def test_default_storage_migrates_legacy_files(monkeypatch, tmp_path):
    legacy_dir = tmp_path / "legacy-storage"
    detection_cache = legacy_dir / "cache" / "detection" / "sample.json"
    detection_cache.parent.mkdir(parents=True)
    detection_cache.write_text('{"objects": []}', encoding="utf-8")

    measurements = legacy_dir / "measurements.jsonl"
    measurements.write_text(
        '{"timestamp": "2026-03-03T10:00:00"}\n',
        encoding="utf-8",
    )

    user_data_dir = tmp_path / "user-data"
    monkeypatch.setattr(storage_module, "LEGACY_LOCAL_STORAGE_DIR", legacy_dir)
    monkeypatch.setattr(storage_module, "LOCAL_STORAGE_DIR", user_data_dir)

    service = storage_module.StorageService()

    assert service.root_dir == user_data_dir.resolve()
    assert service.migrated_from_legacy is True
    assert service.migrated_files_count == 2
    assert (user_data_dir / "cache" / "detection" / "sample.json").read_text(
        encoding="utf-8"
    ) == '{"objects": []}'
    assert (user_data_dir / "measurements.jsonl").read_text(
        encoding="utf-8"
    ) == '{"timestamp": "2026-03-03T10:00:00"}\n'


def test_default_storage_does_not_overwrite_existing_user_data(
    monkeypatch,
    tmp_path,
):
    legacy_dir = tmp_path / "legacy-storage"
    legacy_measurements = legacy_dir / "measurements.jsonl"
    legacy_measurements.parent.mkdir(parents=True)
    legacy_measurements.write_text("legacy\n", encoding="utf-8")

    user_data_dir = tmp_path / "user-data"
    current_measurements = user_data_dir / "measurements.jsonl"
    current_measurements.parent.mkdir(parents=True)
    current_measurements.write_text("current\n", encoding="utf-8")

    monkeypatch.setattr(storage_module, "LEGACY_LOCAL_STORAGE_DIR", legacy_dir)
    monkeypatch.setattr(storage_module, "LOCAL_STORAGE_DIR", user_data_dir)

    service = storage_module.StorageService()

    assert service.root_dir == user_data_dir.resolve()
    assert service.migrated_from_legacy is False
    assert service.migrated_files_count == 0
    assert current_measurements.read_text(encoding="utf-8") == "current\n"
