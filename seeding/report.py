"""Утилиты для формирования PDF-отчёта."""

from __future__ import annotations

import io

from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .models.data_models import ObjectImage, OriginalImage


def _np_to_pil(img: np.ndarray) -> Image.Image:
    """Преобразует изображение NumPy (BGR или оттенки серого) в `PIL.Image`."""
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


def _annotate_image(img: np.ndarray, objects: list[ObjectImage]) -> np.ndarray:
    """Наносит на изображение рамки объектов с порядковыми номерами."""
    annotated = img.copy()
    for i, obj in enumerate(objects, start=1):
        if obj.bbox:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                str(i),
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
    return annotated


def _pil_to_buf(image: Image.Image, *, quality: int = 70) -> io.BytesIO:
    """Сохраняет изображение в буфер JPEG для вставки в PDF.

    Args:
        image: Изображение PIL.
        quality: Качество JPEG (1-95), чем ниже — тем меньше размер.

    Returns:
        Буфер с JPEG-данными.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return buffer


def _object_to_pil(obj: ObjectImage) -> Image.Image | None:
    """Преобразует объект в `PIL.Image`, если возможно."""
    if not obj.image:
        return None
    img = obj.image[0]
    if isinstance(img, np.ndarray):
        return _np_to_pil(img)
    if isinstance(img, Image.Image):
        return img
    return None

def create_pdf_report(data: OriginalImage, output_path: str) -> None:
    """Создаёт PDF-отчёт с результатами детекции.

    На каждой странице показывается исходное изображение с рамками и номерами,
    таблица характеристик и уменьшенные копии каждого найденного объекта.
    Изображения сохраняются в формате JPEG с качеством 70 для снижения размера
    итогового файла.
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for idx, img in enumerate(data.images):
        objs: list[ObjectImage] = []
        if data.class_object_image and len(data.class_object_image) > idx:
            objs = data.class_object_image[idx]

        annotated_img = _annotate_image(img, objs)
        pil_img = _np_to_pil(annotated_img)

        story.append(Paragraph(f"Page {idx + 1}", styles["Heading1"]))

        max_width = 160 * mm
        max_height = 200 * mm
        aspect = pil_img.height / float(pil_img.width)
        width_pt = min(max_width, max_height / aspect)
        img_elem = RLImage(
            _pil_to_buf(pil_img), width=width_pt, height=width_pt * aspect
        )
        story.append(img_elem)
        story.append(Spacer(1, 5 * mm))

        table_data = [["#", "Class", "Confidence", "BBox"]]
        for i, obj in enumerate(objs, start=1):
            bbox = obj.bbox if obj.bbox else ("", "", "", "")
            table_data.append([
                str(i),
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
        story.append(Spacer(1, 5 * mm))

        for i, obj in enumerate(objs, start=1):
            pil_obj = _object_to_pil(obj)
            if pil_obj is None:
                continue
            story.append(Paragraph(f"Объект {i}", styles["Heading3"]))
            crop_elem = RLImage(_pil_to_buf(pil_obj), width=60 * mm)
            story.append(crop_elem)
            story.append(Spacer(1, 2 * mm))

        story.append(Spacer(1, 8 * mm))

        if idx < len(data.images) - 1:
            story.append(PageBreak())

    doc.build(story)
