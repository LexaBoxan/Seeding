import numpy as np

from seeding.models import AllClassImage, ObjectImage, OriginalImage
from seeding.services import ImageService


def test_rotate_page_updates_image_bbox_and_crop():
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    obj = ObjectImage(
        class_name="Seeding",
        confidence=0.9,
        bbox=(2, 2, 8, 6),
    )
    storage = OriginalImage(images=[image], class_object_image=[[obj]])

    service = ImageService()
    rotated = service.rotate_page(
        storage,
        page_index=0,
        angle=-90,
        rotate_k=-1,
    )

    assert rotated.shape[:2] == (12, 8)
    assert storage.class_object_image[0][0].bbox is not None
    assert storage.class_object_image[0][0].image
    assert storage.class_object_image[0][0].image[0].size > 0


def test_rotate_crop_updates_local_class_bboxes():
    crop = np.zeros((6, 4, 3), dtype=np.uint8)
    cls = AllClassImage("part", 0.7, crop.copy(), bbox=(1, 1, 3, 5))
    obj = ObjectImage(
        class_name="Seeding",
        confidence=0.9,
        image=[crop],
        image_all_class=[cls],
        bbox=(0, 0, 4, 6),
    )
    storage = OriginalImage(
        images=[np.zeros((10, 10, 3), dtype=np.uint8)],
        class_object_image=[[obj]],
    )

    service = ImageService()
    rotated_crop = service.rotate_crop(
        storage,
        page_index=0,
        crop_index=0,
        angle=-90,
        rotate_k=-1,
    )

    assert rotated_crop.shape[:2] == (4, 6)
    assert obj.rotation_k == 3
    assert obj.image_all_class[0].bbox is not None
