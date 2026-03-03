import numpy as np

from seeding.inference import (
    decode_onnx_outputs,
    infer_backend_kind,
    normalize_yolo_results,
)


class _TensorStub:
    def __init__(self, values):
        self._values = np.array(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _BoxStub:
    def __init__(self, cls_id: int, conf: float, xyxy):
        self.cls = cls_id
        self.conf = conf
        self.xyxy = [_TensorStub(xyxy)]


class _ResultStub:
    def __init__(self):
        self.names = {0: "seedling", 1: "root"}
        self.boxes = [
            _BoxStub(1, 0.82, [10.0, 20.0, 40.0, 60.0]),
        ]


def test_infer_backend_kind_uses_model_suffix():
    assert infer_backend_kind("models/detect.onnx") == "onnx"
    assert infer_backend_kind("models/detect.pt") == "torch"


def test_normalize_yolo_results_converts_boxes_to_ultralytics_like_adapter():
    results = normalize_yolo_results([_ResultStub()])

    assert len(results) == 1
    assert results[0].names == {0: "seedling", 1: "root"}
    assert len(results[0].boxes) == 1
    box = results[0].boxes[0]
    assert box.cls == 1
    assert abs(box.conf - 0.82) < 1e-9
    assert box.xyxy[0].cpu().numpy().tolist() == [10.0, 20.0, 40.0, 60.0]
    assert box.xywh[0].cpu().numpy().tolist() == [25.0, 40.0, 30.0, 40.0]


def test_decode_onnx_outputs_restores_original_coordinates():
    outputs = [
        np.array(
            [
                [
                    [320.0, 160.0],
                    [272.0, 160.0],
                    [320.0, 64.0],
                    [160.0, 64.0],
                    [0.10, 0.20],
                    [0.80, 0.10],
                ]
            ],
            dtype=np.float32,
        )
    ]

    results = decode_onnx_outputs(
        outputs,
        class_names={0: "seedling", 1: "root"},
        original_shape=(100, 200),
        scale=3.2,
        pad_x=0,
        pad_y=160,
        conf_threshold=0.25,
    )

    assert len(results) == 1
    assert len(results[0].boxes) == 1
    box = results[0].boxes[0]
    coords = box.xyxy[0].cpu().numpy()
    assert box.cls == 1
    assert abs(box.conf - 0.80) < 1e-6
    assert np.allclose(coords, [50.0, 10.0, 150.0, 60.0], atol=1e-4)
