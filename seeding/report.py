"""PDF report generation utilities."""

from __future__ import annotations

import io
from PIL import Image
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image as RLImage,
    Table,
    TableStyle,
    Spacer,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet

from .models.data_models import OriginalImage


def _np_to_pil(img: np.ndarray) -> Image.Image:
    """Convert a numpy image (BGR or grayscale) to PIL.Image."""
    if img is None:
        raise ValueError("Image is None")
    if img.ndim == 3:
        if img.shape[2] == 3:
            # convert BGR to RGB
            return Image.fromarray(img[:, :, ::-1])
        else:
            return Image.fromarray(img)
    else:
        return Image.fromarray(img)


def create_pdf_report(data: OriginalImage, output_path: str) -> None:
    """Create a PDF report with images and detected object tables."""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for idx, img in enumerate(data.images):
        pil_img = _np_to_pil(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        story.append(Paragraph(f"Page {idx + 1}", styles["Heading1"]))

        max_width = 160 * mm
        aspect = pil_img.height / float(pil_img.width)
        img_elem = RLImage(buf, width=max_width, height=max_width * aspect)
        story.append(img_elem)
        story.append(Spacer(1, 5 * mm))

        objs = []
        if data.class_object_image and len(data.class_object_image) > idx:
            objs = data.class_object_image[idx]

        table_data = [["Class", "Confidence", "BBox"]]
        for obj in objs:
            bbox = obj.bbox if obj.bbox else ("", "", "", "")
            table_data.append([
                obj.class_name,
                f"{obj.confidence:.2f}",
                str(bbox),
            ])

        table = Table(table_data, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(table)

        if idx < len(data.images) - 1:
            story.append(PageBreak())

    doc.build(story)
