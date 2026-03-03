"""Сервисы бизнес-логики обработки изображений и отчётов.

Пакет объединён в один модуль для уменьшения дробности структуры.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from seeding.models import AllClassImage, ObjectImage, OriginalImage
from seeding.report import create_pdf_report
from seeding.utils import (
    clip_bbox_to_image,
    rotate_bbox,
    rotate_image_and_boxes,
    simple_nms,
)


class ImageService:
    """Доменная логика поворота, clipping и пересборки кропов."""

    @staticmethod
    def _iter_page_objects(
        image_storage: OriginalImage,
        page_index: int,
    ) -> list[ObjectImage]:
        """Возвращает список объектов выбранной страницы.

        Если контейнер детекций ещё не инициализирован или индекс страницы
        выходит за границы, возвращается пустой список.
        """
        if (
            not image_storage.class_object_image
            or page_index >= len(image_storage.class_object_image)
        ):
            return []
        return image_storage.class_object_image[page_index]

    def refresh_page_crops(
        self,
        image_storage: OriginalImage,
        page_index: int,
        *,
        rotate_k: int,
        clear_classification: bool,
    ) -> None:
        """Пересобирает кропы объектов после изменения bbox/изображения."""
        page_objects = self._iter_page_objects(image_storage, page_index)
        if not page_objects:
            return

        base_img = image_storage.images[page_index]
        if base_img is None:
            return
        height, width = base_img.shape[:2]

        for obj in page_objects:
            if not obj.bbox:
                continue

            clipped = clip_bbox_to_image(obj.bbox, width, height)
            if clipped is None:
                obj.bbox = None
                obj.image = []
                if clear_classification:
                    obj.image_all_class = None
                continue

            x1, y1, x2, y2 = clipped
            obj.bbox = clipped
            crop = base_img[y1:y2, x1:x2].copy()

            rotation_k = 0
            if crop.shape[1] > crop.shape[0]:
                crop = np.rot90(crop, k=rotate_k)
                rotation_k = rotate_k

            obj.rotation_k = rotation_k
            obj.image = [crop]
            if clear_classification:
                obj.image_all_class = None

    def rotate_page(
        self,
        image_storage: OriginalImage,
        page_index: int,
        *,
        angle: float,
        rotate_k: int,
    ) -> np.ndarray:
        """Поворачивает страницу и трансформирует bbox всех объектов."""
        page_image = image_storage.images[page_index]
        if page_image is None:
            raise ValueError("Изображение страницы пустое")

        page_objects = self._iter_page_objects(image_storage, page_index)
        object_map: list[int] = []
        boxes_to_rotate: list[tuple[int, int, int, int]] = []
        for obj_idx, obj in enumerate(page_objects):
            if obj.bbox:
                boxes_to_rotate.append(obj.bbox)
                object_map.append(obj_idx)

        rotated_image, rotated_boxes = rotate_image_and_boxes(
            page_image,
            boxes_to_rotate,
            angle,
        )
        image_storage.images[page_index] = rotated_image

        for mapped_idx, obj_idx in enumerate(object_map):
            page_objects[obj_idx].bbox = rotated_boxes[mapped_idx]

        self.refresh_page_crops(
            image_storage,
            page_index,
            rotate_k=rotate_k,
            clear_classification=True,
        )
        return rotated_image

    def rotate_crop(
        self,
        image_storage: OriginalImage,
        page_index: int,
        crop_index: int,
        *,
        angle: float,
        rotate_k: int,
    ) -> np.ndarray:
        """Поворачивает кроп и локальные bbox частей внутри кропа."""
        page_objects = self._iter_page_objects(image_storage, page_index)
        obj = page_objects[crop_index]
        if not obj.image or obj.image[0] is None:
            raise ValueError("Изображение кропа пустое")

        crop = obj.image[0]
        class_map: list[int] = []
        class_boxes: list[tuple[int, int, int, int]] = []
        if obj.image_all_class:
            for class_idx, cls_obj in enumerate(obj.image_all_class):
                if cls_obj.bbox:
                    class_boxes.append(cls_obj.bbox)
                    class_map.append(class_idx)

        rotated_crop, rotated_class_boxes = rotate_image_and_boxes(
            crop,
            class_boxes,
            angle,
        )
        obj.image[0] = rotated_crop
        obj.rotation_k = (obj.rotation_k + rotate_k) % 4

        for mapped_idx, class_idx in enumerate(class_map):
            obj.image_all_class[class_idx].bbox = (
                rotated_class_boxes[mapped_idx]
            )
        return rotated_crop

    @staticmethod
    def clip_local_bbox(
        crop_image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        """Ограничивает локальный bbox размерами кропа."""
        height, width = crop_image.shape[:2]
        return clip_bbox_to_image(bbox, width, height)

    @staticmethod
    def crop_region(
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Возвращает область изображения по валидированному bbox."""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]

    @staticmethod
    def clip_many(
        image: np.ndarray,
        boxes: Iterable[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int] | None]:
        """Пакетно ограничивает bbox размерами изображения."""
        height, width = image.shape[:2]
        return [clip_bbox_to_image(box, width, height) for box in boxes]

    def sync_crops_and_parts(self, image_storage: OriginalImage) -> None:
        """Синхронизирует кропы объектов и частей по актуальным bbox."""
        if not image_storage.images or not image_storage.class_object_image:
            return

        for page_index, objects in enumerate(image_storage.class_object_image):
            if page_index >= len(image_storage.images):
                continue
            base_img = image_storage.images[page_index]
            height, width = base_img.shape[:2]
            for obj in objects:
                if obj.bbox:
                    clipped_obj = clip_bbox_to_image(obj.bbox, width, height)
                    if clipped_obj is None:
                        obj.image = []
                        obj.bbox = None
                        continue
                    x1, y1, x2, y2 = clipped_obj
                    obj.bbox = clipped_obj
                    crop = base_img[y1:y2, x1:x2].copy()
                    if getattr(obj, "rotation_k", 0):
                        crop = np.rot90(crop, k=obj.rotation_k)
                    obj.image = [crop]

                if not obj.image_all_class:
                    continue

                rotation_k = getattr(obj, "rotation_k", 0) % 4
                crop_height, crop_width = (
                    obj.image[0].shape[:2] if obj.image else (0, 0)
                )
                for cls_obj in obj.image_all_class:
                    if not cls_obj.bbox:
                        continue

                    lx1, ly1, lx2, ly2 = cls_obj.bbox
                    if rotation_k and crop_height and crop_width:
                        ux1, uy1, ux2, uy2 = rotate_bbox(
                            lx1,
                            ly1,
                            lx2,
                            ly2,
                            crop_width,
                            crop_height,
                            (-rotation_k) % 4,
                        )
                    else:
                        ux1, uy1, ux2, uy2 = lx1, ly1, lx2, ly2

                    if obj.bbox:
                        gx1 = obj.bbox[0] + ux1
                        gy1 = obj.bbox[1] + uy1
                        gx2 = obj.bbox[0] + ux2
                        gy2 = obj.bbox[1] + uy2
                    else:
                        gx1, gy1, gx2, gy2 = ux1, uy1, ux2, uy2

                    clipped_part = clip_bbox_to_image(
                        (gx1, gy1, gx2, gy2),
                        width,
                        height,
                    )
                    if clipped_part is None:
                        cls_obj.image = np.empty(
                            (0, 0, 3),
                            dtype=base_img.dtype,
                        )
                        continue
                    gx1, gy1, gx2, gy2 = clipped_part
                    part = base_img[gy1:gy2, gx1:gx2].copy()
                    if rotation_k:
                        part = np.rot90(part, k=rotation_k)
                    cls_obj.image = part


class DetectionService:
    """Преобразует сырые предсказания модели в объекты домена."""

    @staticmethod
    def build_objects(
        image: np.ndarray,
        results,
        *,
        detection_class_name: str,
        iou_threshold: float,
        rotate_k: int,
    ) -> list[ObjectImage]:
        """Возвращает список детекций после clipping и NMS."""
        if image is None or results is None or not results:
            return []

        parsed: list[dict] = []
        boxes: list[list[int]] = []
        scores: list[float] = []
        height, width = image.shape[:2]

        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = str(results[0].names[class_id]).lower()
            if class_name != detection_class_name.lower():
                continue

            score = float(box.conf)
            x_center, y_center, box_width, box_height = (
                box.xywh[0].cpu().numpy()
            )
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            clipped = clip_bbox_to_image((x1, y1, x2, y2), width, height)
            if clipped is None:
                continue

            cx1, cy1, cx2, cy2 = clipped
            parsed.append(
                {
                    "class_name": class_name,
                    "score": score,
                    "bbox": (cx1, cy1, cx2, cy2),
                }
            )
            boxes.append([cx1, cy1, cx2, cy2])
            scores.append(score)

        if not boxes:
            return []

        kept_indices = simple_nms(boxes, scores, iou_threshold=iou_threshold)
        objects: list[ObjectImage] = []
        for idx in kept_indices:
            item = parsed[idx]
            x1, y1, x2, y2 = item["bbox"]
            crop = image[y1:y2, x1:x2].copy()

            rotation_k = 0
            if crop.shape[1] > crop.shape[0]:
                crop = np.rot90(crop, k=rotate_k)
                rotation_k = rotate_k

            objects.append(
                ObjectImage(
                    class_name=item["class_name"],
                    confidence=float(item["score"]),
                    image=[crop],
                    bbox=(x1, y1, x2, y2),
                    rotation_k=rotation_k,
                )
            )
        return objects


class ClassificationService:
    """Преобразует сырые предсказания модели в список частей растения."""

    @staticmethod
    def build_parts(crop_image: np.ndarray, results) -> list[AllClassImage]:
        """Возвращает части растения с локальными bbox внутри кропа."""
        if crop_image is None or results is None or not results:
            return []

        crop_height, crop_width = crop_image.shape[:2]
        parts: list[AllClassImage] = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                class_id = int(box.cls)
                class_name = result.names[class_id]
                coords = box.xyxy[0].cpu().numpy().astype(int)
                local_bbox = clip_bbox_to_image(
                    (
                        int(coords[0]),
                        int(coords[1]),
                        int(coords[2]),
                        int(coords[3]),
                    ),
                    crop_width,
                    crop_height,
                )
                if local_bbox is None:
                    continue

                lx1, ly1, lx2, ly2 = local_bbox
                part_image = crop_image[ly1:ly2, lx1:lx2].copy()
                parts.append(
                    AllClassImage(
                        class_name=class_name,
                        confidence=confidence,
                        image=part_image,
                        bbox=local_bbox,
                    )
                )

        return parts


class ReportService:
    """Операции формирования отчётов."""

    @staticmethod
    def generate_report(data: OriginalImage, output_path: str) -> str:
        """Формирует отчёт и возвращает итоговый путь."""
        create_pdf_report(data, output_path)
        return output_path


class ExportService:
    """Экспорт результатов анализа в разные форматы."""

    @staticmethod
    def _iter_global_annotations(
        image_storage: OriginalImage,
    ) -> list[dict]:
        """Плоский список аннотаций с глобальными bbox по страницам."""
        annotations: list[dict] = []
        if not image_storage.class_object_image:
            return annotations

        for page_index, objects in enumerate(image_storage.class_object_image):
            if page_index >= len(image_storage.images):
                continue
            page_image = image_storage.images[page_index]
            height, width = page_image.shape[:2]
            for obj_idx, obj in enumerate(objects):
                if obj.bbox:
                    clipped = clip_bbox_to_image(obj.bbox, width, height)
                    if clipped is not None:
                        annotations.append(
                            {
                                "page_index": page_index,
                                "object_index": obj_idx,
                                "class_name": "seeding",
                                "confidence": float(obj.confidence),
                                "bbox": clipped,
                            }
                        )

                if not obj.image_all_class or not obj.bbox:
                    continue

                rotation_k = int(getattr(obj, "rotation_k", 0)) % 4
                if obj.image:
                    crop_height, crop_width = obj.image[0].shape[:2]
                else:
                    crop_height, crop_width = 0, 0

                for part in obj.image_all_class:
                    if not part.bbox:
                        continue
                    lx1, ly1, lx2, ly2 = part.bbox
                    if rotation_k and crop_height and crop_width:
                        ux1, uy1, ux2, uy2 = rotate_bbox(
                            lx1,
                            ly1,
                            lx2,
                            ly2,
                            crop_width,
                            crop_height,
                            (-rotation_k) % 4,
                        )
                    else:
                        ux1, uy1, ux2, uy2 = lx1, ly1, lx2, ly2

                    gx1 = obj.bbox[0] + ux1
                    gy1 = obj.bbox[1] + uy1
                    gx2 = obj.bbox[0] + ux2
                    gy2 = obj.bbox[1] + uy2
                    clipped = clip_bbox_to_image(
                        (gx1, gy1, gx2, gy2),
                        width,
                        height,
                    )
                    if clipped is None:
                        continue
                    annotations.append(
                        {
                            "page_index": page_index,
                            "object_index": obj_idx,
                            "class_name": str(part.class_name),
                            "confidence": float(part.confidence),
                            "bbox": clipped,
                        }
                    )
        return annotations

    @staticmethod
    def export_json(
        image_storage: OriginalImage,
        output_dir: str | Path,
    ) -> Path:
        """Экспортирует результаты в JSON."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        annotations = ExportService._iter_global_annotations(image_storage)
        payload = {
            "source_file": image_storage.file_path,
            "pages_count": len(image_storage.images),
            "annotations": annotations,
        }
        path = out_dir / "results.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def export_metadata(
        metadata: dict[str, object],
        output_dir: str | Path,
    ) -> Path:
        """Сохраняет sidecar-файл с метаданными экспорта."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "export_metadata.json"
        path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def export_csv(
        image_storage: OriginalImage,
        output_dir: str | Path,
    ) -> Path:
        """Экспортирует результаты в CSV."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        annotations = ExportService._iter_global_annotations(image_storage)
        path = out_dir / "results.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=";")
            writer.writerow(
                [
                    "page_index",
                    "object_index",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ]
            )
            for ann in annotations:
                x1, y1, x2, y2 = ann["bbox"]
                writer.writerow(
                    [
                        ann["page_index"],
                        ann["object_index"],
                        ann["class_name"],
                        f"{ann['confidence']:.6f}",
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )
        return path

    @staticmethod
    def export_yolo(
        image_storage: OriginalImage,
        output_dir: str | Path,
    ) -> Path:
        """Экспортирует результаты в YOLO txt по страницам."""
        out_dir = Path(output_dir) / "yolo"
        out_dir.mkdir(parents=True, exist_ok=True)
        annotations = ExportService._iter_global_annotations(image_storage)

        classes = sorted({ann["class_name"] for ann in annotations})
        class_to_id = {name: idx for idx, name in enumerate(classes)}
        classes_path = out_dir / "classes.txt"
        classes_path.write_text("\n".join(classes), encoding="utf-8")

        by_page: dict[int, list[dict]] = {}
        for ann in annotations:
            by_page.setdefault(int(ann["page_index"]), []).append(ann)

        for page_idx, page_annotations in by_page.items():
            if page_idx >= len(image_storage.images):
                continue
            image = image_storage.images[page_idx]
            height, width = image.shape[:2]
            lines: list[str] = []
            for ann in page_annotations:
                class_id = class_to_id[ann["class_name"]]
                x1, y1, x2, y2 = ann["bbox"]
                bw = max(0.0, float(x2 - x1))
                bh = max(0.0, float(y2 - y1))
                cx = float(x1) + bw / 2.0
                cy = float(y1) + bh / 2.0
                lines.append(
                    (
                        f"{class_id} "
                        f"{cx / width:.6f} {cy / height:.6f} "
                        f"{bw / width:.6f} {bh / height:.6f}"
                    )
                )
            (out_dir / f"page_{page_idx + 1:04d}.txt").write_text(
                "\n".join(lines),
                encoding="utf-8",
            )
        return out_dir

    @staticmethod
    def export_coco(
        image_storage: OriginalImage,
        output_dir: str | Path,
    ) -> Path:
        """Экспортирует результаты в COCO JSON."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        annotations = ExportService._iter_global_annotations(image_storage)

        categories_names = sorted({ann["class_name"] for ann in annotations})
        categories = [
            {"id": idx + 1, "name": name}
            for idx, name in enumerate(categories_names)
        ]
        class_to_id = {cat["name"]: cat["id"] for cat in categories}

        images = []
        coco_annotations = []
        ann_id = 1
        for page_idx, image in enumerate(image_storage.images):
            height, width = image.shape[:2]
            images.append(
                {
                    "id": page_idx + 1,
                    "file_name": f"page_{page_idx + 1:04d}.jpg",
                    "width": width,
                    "height": height,
                }
            )
        for ann in annotations:
            x1, y1, x2, y2 = ann["bbox"]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": int(ann["page_index"]) + 1,
                    "category_id": class_to_id[ann["class_name"]],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": float(ann["confidence"]),
                }
            )
            ann_id += 1

        payload = {
            "images": images,
            "annotations": coco_annotations,
            "categories": categories,
        }
        path = out_dir / "results_coco.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def export_annotated_images(
        image_storage: OriginalImage,
        output_dir: str | Path,
    ) -> Path:
        """Сохраняет изображения с отрисованными bbox в отдельную папку."""
        out_dir = Path(output_dir) / "annotated"
        out_dir.mkdir(parents=True, exist_ok=True)
        annotations = ExportService._iter_global_annotations(image_storage)
        by_page: dict[int, list[dict]] = {}
        for ann in annotations:
            by_page.setdefault(int(ann["page_index"]), []).append(ann)

        for page_idx, image in enumerate(image_storage.images):
            rendered = image.copy()
            for ann in by_page.get(page_idx, []):
                x1, y1, x2, y2 = ann["bbox"]
                class_name = ann["class_name"]
                color = (0, 255, 0) if class_name == "seeding" else (255, 0, 0)
                cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    rendered,
                    class_name,
                    (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
            cv2.imwrite(
                str(out_dir / f"page_{page_idx + 1:04d}.jpg"),
                rendered,
            )
        return out_dir


__all__ = [
    "ImageService",
    "ReportService",
    "DetectionService",
    "ClassificationService",
    "ExportService",
]
