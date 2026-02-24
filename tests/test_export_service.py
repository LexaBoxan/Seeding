import json

import numpy as np

from seeding.models import AllClassImage, ObjectImage, OriginalImage
from seeding.services import ExportService


def _build_sample_storage() -> OriginalImage:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    cls = AllClassImage(
        class_name="stem",
        confidence=0.8,
        image=np.zeros((5, 5, 3), dtype=np.uint8),
        bbox=(1, 1, 4, 4),
    )
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.9,
        image=[np.zeros((8, 6, 3), dtype=np.uint8)],
        image_all_class=[cls],
        bbox=(5, 6, 16, 18),
        rotation_k=0,
    )
    return OriginalImage(
        file_path="sample.jpg",
        images=[image],
        class_object_image=[[obj]],
    )


def test_export_service_creates_json_csv_and_coco(tmp_path):
    storage = _build_sample_storage()
    service = ExportService()

    json_path = service.export_json(storage, tmp_path)
    csv_path = service.export_csv(storage, tmp_path)
    coco_path = service.export_coco(storage, tmp_path)

    assert json_path.is_file()
    assert csv_path.is_file()
    assert coco_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["pages_count"] == 1
    assert len(payload["annotations"]) >= 1

    coco_payload = json.loads(coco_path.read_text(encoding="utf-8"))
    assert len(coco_payload["images"]) == 1
    assert len(coco_payload["categories"]) >= 1
    assert len(coco_payload["annotations"]) >= 1


def test_export_service_creates_yolo_and_annotated_images(tmp_path):
    storage = _build_sample_storage()
    service = ExportService()

    yolo_dir = service.export_yolo(storage, tmp_path)
    annotated_dir = service.export_annotated_images(storage, tmp_path)

    assert (yolo_dir / "classes.txt").is_file()
    assert any(path.suffix == ".txt" for path in yolo_dir.glob("page_*.txt"))
    assert any(
        path.suffix.lower() == ".jpg"
        for path in annotated_dir.glob("*.jpg")
    )
