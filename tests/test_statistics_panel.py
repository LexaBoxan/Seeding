import numpy as np

from seeding.models import AllClassImage, ObjectImage, OriginalImage
from seeding.ui.statistics_panel import StatisticsPanel


def _part(name: str, confidence: float = 0.9) -> AllClassImage:
    return AllClassImage(
        class_name=name,
        confidence=confidence,
        image=np.empty((0, 0, 3), dtype=np.uint8),
        bbox=None,
    )


def test_statistics_split_counts_by_categories():
    data = OriginalImage(
        images=[np.zeros((20, 20, 3), dtype=np.uint8)],
        class_object_image=[
            [
                ObjectImage(
                    class_name="seeding",
                    confidence=0.95,
                    image=[np.zeros((10, 10, 3), dtype=np.uint8)],
                    bbox=(0, 0, 10, 10),
                    image_all_class=[
                        _part("inflorescence"),
                        _part("stem"),
                        _part("root"),
                    ],
                ),
                ObjectImage(
                    class_name="seeding",
                    confidence=0.85,
                    image=[np.zeros((8, 8, 3), dtype=np.uint8)],
                    bbox=(2, 2, 9, 9),
                    image_all_class=[
                        _part("stem"),
                        _part("unknown_part"),
                    ],
                ),
            ]
        ],
    )

    summary = StatisticsPanel.build_summary(data)

    assert summary.pages_count == 1
    assert summary.objects_count == 2
    assert summary.seedlings_count == 2
    assert summary.inflorescences_count == 1
    assert summary.stems_count == 2
    assert summary.roots_count == 1
    assert summary.other_parts_count == 1
