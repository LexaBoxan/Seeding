import fitz
import numpy as np

from seeding.models import AllClassImage, ObjectImage, OriginalImage
from seeding.report import _annotate_image, create_pdf_report


def test_create_pdf_report(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    obj = ObjectImage(
        class_name="Seeding",
        confidence=0.9,
        image=[img],
        bbox=(1, 2, 3, 4),
    )
    data = OriginalImage(
        file_path=str(tmp_path / "image.png"),
        images=[img],
        class_object_image=[[obj]],
    )
    output = tmp_path / "report.pdf"
    create_pdf_report(data, str(output))
    assert output.is_file() and output.stat().st_size > 0


def test_annotate_image_with_class_bbox():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cls = AllClassImage("part", 0.8, img, bbox=(1, 1, 3, 3))
    obj = ObjectImage(
        class_name="Seeding",
        confidence=0.9,
        image=[img],
        bbox=(1, 1, 5, 5),
        image_all_class=[cls],
    )
    annotated = _annotate_image(img, [obj])
    assert (annotated[2, 2] == np.array([255, 0, 0])).all()


def test_create_pdf_report_without_objects_has_no_extra_blank_pages(tmp_path):
    img1 = np.zeros((20, 20, 3), dtype=np.uint8)
    img2 = np.zeros((20, 20, 3), dtype=np.uint8)
    data = OriginalImage(
        file_path=str(tmp_path / "input.png"),
        images=[img1, img2],
        class_object_image=[[], []],
    )
    output = tmp_path / "report_no_objects.pdf"

    create_pdf_report(data, str(output))

    doc = fitz.open(str(output))
    try:
        assert doc.page_count == 2
    finally:
        doc.close()
