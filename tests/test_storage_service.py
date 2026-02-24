from pathlib import Path

import numpy as np

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
