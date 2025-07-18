import os
import numpy as np
from seeding.models.data_models import OriginalImage, ObjectImage
from seeding.report import create_pdf_report

def test_create_pdf_report(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    obj = ObjectImage(class_name="Seeding", confidence=0.9, image=[img], bbox=(1, 2, 3, 4))
    data = OriginalImage(file_path=str(tmp_path / "image.png"), images=[img], class_object_image=[[obj]])
    output = tmp_path / "report.pdf"
    create_pdf_report(data, str(output))
    assert output.is_file() and output.stat().st_size > 0
