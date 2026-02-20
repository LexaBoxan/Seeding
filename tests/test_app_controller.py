import numpy as np

from seeding.controllers import AppController
from seeding.models import AppState, ObjectImage, OriginalImage


class DummyReportService:
    def __init__(self):
        self.output_path = None

    def generate_report(self, data, output_path: str) -> str:
        self.output_path = output_path
        return output_path


class _TensorStub:
    def __init__(self, values):
        self._values = np.array(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _BoxStub:
    def __init__(self, cls_id: int, conf: float, xywh):
        self.cls = cls_id
        self.conf = conf
        self.xywh = [_TensorStub(xywh)]


class _ResultStub:
    def __init__(self):
        self.names = {0: "seeding", 1: "other"}
        self.boxes = [
            _BoxStub(0, 0.9, [6.0, 6.0, 4.0, 4.0]),
            _BoxStub(1, 0.95, [2.0, 2.0, 2.0, 2.0]),
        ]


class _ClassBoxStub:
    def __init__(self, cls_id: int, conf: float, xyxy):
        self.cls = cls_id
        self.conf = conf
        self.xyxy = [_TensorStub(xyxy)]


class _ClassResultStub:
    def __init__(self):
        self.names = {0: "stem"}
        self.boxes = [_ClassBoxStub(0, 0.7, [1.0, 1.0, 3.0, 4.0])]


def test_rotate_selection_for_page_returns_result():
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    obj = ObjectImage(class_name="Seeding", confidence=0.9, bbox=(2, 2, 8, 6))
    state = AppState(
        image_storage=OriginalImage(images=[image], class_object_image=[[obj]])
    )
    controller = AppController()

    result = controller.rotate_selection(
        state,
        {"type": "pdf", "index": 0},
        angle=-90,
        rotate_k=-1,
    )

    assert result is not None
    assert result.target == "page"
    assert state.active_image_index == 0
    assert result.image.shape[:2] == (12, 8)


def test_generate_report_updates_state_last_path():
    state = AppState(
        image_storage=OriginalImage(
            images=[np.zeros((5, 5, 3), dtype=np.uint8)],
            class_object_image=[[]],
        )
    )
    report_service = DummyReportService()
    controller = AppController(report_service=report_service)

    output = controller.generate_report(state, "out/report.pdf")

    assert output == "out/report.pdf"
    assert state.last_report_path == "out/report.pdf"
    assert report_service.output_path == "out/report.pdf"


def test_run_detection_updates_app_state_objects():
    state = AppState(
        image_storage=OriginalImage(
            images=[np.zeros((12, 12, 3), dtype=np.uint8)],
            class_object_image=[[]],
        )
    )
    controller = AppController()

    objects = controller.run_detection(state, 0, [_ResultStub()])

    assert len(objects) == 1
    assert objects[0].bbox == (4, 4, 8, 8)
    assert state.image_storage.class_object_image[0][0].confidence == 0.9


def test_save_crops_syncs_object_crop_from_bbox():
    base = np.zeros((10, 10, 3), dtype=np.uint8)
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.8,
        bbox=(2, 3, 7, 9),
        image=[],
    )
    state = AppState(
        image_storage=OriginalImage(
            images=[base],
            class_object_image=[[obj]],
        )
    )
    controller = AppController()

    controller.save_crops(state)

    assert obj.image
    assert obj.image[0].shape[:2] == (6, 5)


def test_run_classification_updates_selected_object_parts():
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.9,
        image=[crop],
        bbox=(0, 0, 8, 8),
    )
    state = AppState(
        image_storage=OriginalImage(
            images=[np.zeros((8, 8, 3), dtype=np.uint8)],
            class_object_image=[[obj]],
        )
    )
    controller = AppController()

    parts = controller.run_classification_for_selection(
        state,
        0,
        0,
        [_ClassResultStub()],
    )

    assert len(parts) == 1
    assert obj.image_all_class is not None
    assert obj.image_all_class[0].class_name == "stem"
