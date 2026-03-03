"""Inference backends for PyTorch/Ultralytics and ONNX Runtime."""

from __future__ import annotations

import ast
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def infer_backend_kind(model_path: str | Path) -> str:
    """Returns the backend kind inferred from the model file suffix."""
    suffix = Path(model_path).suffix.strip().lower()
    if suffix == ".onnx":
        return "onnx"
    return "torch"


class _TensorAdapter:
    """Small adapter that mimics the subset of tensor API used by services."""

    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = np.asarray(values, dtype=float)

    def cpu(self) -> "_TensorAdapter":
        """Keeps API parity with torch tensors used by the legacy services."""
        return self

    def numpy(self) -> np.ndarray:
        """Returns the wrapped values as a NumPy array."""
        return self._values


@dataclass(frozen=True)
class InferenceBox:
    """Prediction box normalized for the UI/domain layer."""

    cls: int
    conf: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def xyxy(self) -> list[_TensorAdapter]:
        """Returns an Ultralytics-like ``xyxy`` representation."""
        return [_TensorAdapter(self.bbox_xyxy)]

    @property
    def xywh(self) -> list[_TensorAdapter]:
        """Returns an Ultralytics-like ``xywh`` representation."""
        x1, y1, x2, y2 = self.bbox_xyxy
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return [
            _TensorAdapter(
                (
                    x1 + width / 2.0,
                    y1 + height / 2.0,
                    width,
                    height,
                )
            )
        ]


@dataclass(frozen=True)
class InferenceResult:
    """Prediction result with class names and a list of normalized boxes."""

    names: dict[int, str]
    boxes: list[InferenceBox]


class InferenceBackend(ABC):
    """Common interface for inference backends."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.backend_kind = infer_backend_kind(model_path)

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float | None = None,
    ) -> list[InferenceResult]:
        """Runs inference on an image and returns normalized predictions."""


def _load_yolo_class():
    from ultralytics import YOLO

    return YOLO


def _load_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "ONNX Runtime is not installed. Install with: "
            "python -m pip install 'seeding[onnx]'"
        ) from error
    return ort


def _normalize_names_map(names: Any) -> dict[int, str]:
    """Normalizes class names from dict/list metadata to ``{id: name}``."""
    if isinstance(names, dict):
        normalized: dict[int, str] = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return normalized
    if isinstance(names, list):
        return {idx: str(value) for idx, value in enumerate(names)}
    return {}


def normalize_yolo_results(raw_results) -> list[InferenceResult]:
    """Converts Ultralytics results into normalized backend objects."""
    if raw_results is None:
        return []

    normalized_results: list[InferenceResult] = []
    for result in raw_results:
        names = _normalize_names_map(getattr(result, "names", {}))
        boxes: list[InferenceBox] = []
        for box in getattr(result, "boxes", []) or []:
            coords = box.xyxy[0].cpu().numpy()
            boxes.append(
                InferenceBox(
                    cls=int(box.cls),
                    conf=float(box.conf),
                    bbox_xyxy=tuple(float(value) for value in coords[:4]),
                )
            )
        normalized_results.append(InferenceResult(names=names, boxes=boxes))
    return normalized_results


def _parse_names_metadata(metadata_map: dict[str, str]) -> dict[int, str]:
    """Extracts class names from ONNX metadata exported by Ultralytics."""
    raw_names = metadata_map.get("names", "").strip()
    if not raw_names:
        return {}
    try:
        parsed = ast.literal_eval(raw_names)
    except (ValueError, SyntaxError):
        return {}
    return _normalize_names_map(parsed)


def _resolve_input_shape(
    input_shape: list[Any] | tuple[Any, ...],
) -> tuple[int, int]:
    """Returns ``(height, width)`` for the ONNX input, defaulting to 640."""
    default_shape = (640, 640)
    if len(input_shape) < 4:
        return default_shape
    raw_height = input_shape[2]
    raw_width = input_shape[3]
    height = (
        int(raw_height)
        if isinstance(raw_height, int)
        else default_shape[0]
    )
    width = (
        int(raw_width)
        if isinstance(raw_width, int)
        else default_shape[1]
    )
    return max(height, 1), max(width, 1)


def _letterbox_image(
    image: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, float, int, int]:
    """Resizes and pads an image to the target shape."""
    original_height, original_width = image.shape[:2]
    target_height, target_width = target_shape
    scale = min(target_width / original_width, target_height / original_height)

    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_x = max(0, (target_width - resized_width) // 2)
    pad_y = max(0, (target_height - resized_height) // 2)
    canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    canvas[
        pad_y:pad_y + resized_height,
        pad_x:pad_x + resized_width,
    ] = resized
    return canvas, float(scale), pad_x, pad_y


def _guess_mask_feature_count(outputs: list[np.ndarray]) -> int:
    """Returns the mask channel count for segmentation exports when present."""
    if len(outputs) < 2:
        return 0
    extra = np.asarray(outputs[1])
    if extra.ndim < 2:
        return 0
    return int(extra.shape[1])


def _normalize_prediction_rows(
    predictions: np.ndarray,
    *,
    expected_features: int | None = None,
) -> np.ndarray:
    """Normalizes raw ONNX predictions to ``(num_boxes, num_features)``."""
    rows = np.asarray(predictions)
    if rows.ndim == 3:
        rows = rows[0]
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2:
        raise ValueError("Unsupported ONNX prediction tensor shape")

    candidates = [rows, rows.T]
    if expected_features is not None:
        exact = [
            candidate
            for candidate in candidates
            if candidate.shape[1] == expected_features
        ]
        if exact:
            return exact[0]
        valid = [
            candidate
            for candidate in candidates
            if candidate.shape[1] >= expected_features
        ]
        if valid:
            return min(valid, key=lambda candidate: candidate.shape[1])

    valid = [
        candidate
        for candidate in candidates
        if 5 <= candidate.shape[1] <= 256
    ]
    if valid:
        return min(valid, key=lambda candidate: candidate.shape[1])
    return rows if rows.shape[1] <= rows.shape[0] else rows.T


def decode_onnx_outputs(
    outputs: list[np.ndarray],
    *,
    class_names: dict[int, str] | None,
    original_shape: tuple[int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
    conf_threshold: float,
) -> list[InferenceResult]:
    """Decodes raw ONNX outputs into normalized prediction objects."""
    if not outputs:
        return []

    detection_tensor = np.asarray(outputs[0])
    mask_feature_count = _guess_mask_feature_count(outputs)
    names = dict(class_names or {})
    expected_features = None
    if names:
        expected_features = 4 + len(names) + mask_feature_count

    rows = _normalize_prediction_rows(
        detection_tensor,
        expected_features=expected_features,
    )
    if rows.size == 0 or rows.shape[1] <= 4:
        return []

    feature_count = int(rows.shape[1])
    class_count = len(names)
    if class_count == 0:
        class_count = max(feature_count - 4 - mask_feature_count, 1)
        names = {idx: str(idx) for idx in range(class_count)}
    else:
        names = {
            idx: names.get(idx, str(idx))
            for idx in range(class_count)
        }

    score_slice_end = min(4 + class_count, feature_count)
    scores = rows[:, 4:score_slice_end]
    if scores.size == 0:
        return []

    boxes: list[InferenceBox] = []
    image_height, image_width = original_shape
    for row, class_scores in zip(rows, scores, strict=False):
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < conf_threshold:
            continue

        x_center, y_center, width, height = row[:4].astype(float)
        x1 = (x_center - width / 2.0 - pad_x) / scale
        y1 = (y_center - height / 2.0 - pad_y) / scale
        x2 = (x_center + width / 2.0 - pad_x) / scale
        y2 = (y_center + height / 2.0 - pad_y) / scale
        x1 = min(max(x1, 0.0), float(image_width))
        y1 = min(max(y1, 0.0), float(image_height))
        x2 = min(max(x2, 0.0), float(image_width))
        y2 = min(max(y2, 0.0), float(image_height))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append(
            InferenceBox(
                cls=class_id,
                conf=confidence,
                bbox_xyxy=(x1, y1, x2, y2),
            )
        )

    return [InferenceResult(names=names, boxes=boxes)]


class TorchYoloBackend(InferenceBackend):
    """Inference backend that delegates to Ultralytics/PyTorch."""

    def __init__(self, model_path: str | Path) -> None:
        super().__init__(model_path)
        yolo_class = _load_yolo_class()
        self._model = yolo_class(self.model_path)

    def predict(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float | None = None,
    ) -> list[InferenceResult]:
        kwargs = {}
        if conf_threshold is not None:
            kwargs["conf"] = float(conf_threshold)
        return normalize_yolo_results(self._model(image, **kwargs))


class OnnxYoloBackend(InferenceBackend):
    """Inference backend for exported YOLO models in ONNX Runtime."""

    def __init__(self, model_path: str | Path) -> None:
        super().__init__(model_path)
        ort = _load_onnxruntime()
        self._session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input = self._session.get_inputs()[0]
        self._input_name = self._input.name
        self._input_shape = _resolve_input_shape(self._input.shape)
        metadata = getattr(
            self._session.get_modelmeta(),
            "custom_metadata_map",
            {},
        )
        self._class_names = _parse_names_metadata(metadata)

    def predict(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float | None = None,
    ) -> list[InferenceResult]:
        threshold = 0.25 if conf_threshold is None else float(conf_threshold)
        letterboxed, scale, pad_x, pad_y = _letterbox_image(
            image,
            self._input_shape,
        )
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        outputs = self._session.run(None, {self._input_name: tensor})
        return decode_onnx_outputs(
            outputs,
            class_names=self._class_names,
            original_shape=image.shape[:2],
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            conf_threshold=threshold,
        )


def load_inference_backend(model_path: str | Path) -> InferenceBackend:
    """Loads the correct inference backend for the provided model path."""
    backend_kind = infer_backend_kind(model_path)
    if backend_kind == "onnx":
        return OnnxYoloBackend(model_path)
    return TorchYoloBackend(model_path)


def export_model_to_onnx(
    weights_path: str | Path,
    *,
    output_path: str | Path | None = None,
    dynamic: bool = True,
    imgsz: int = 640,
    simplify: bool = False,
    half: bool = False,
    opset: int | None = 20,
) -> Path:
    """Exports a YOLO model to ONNX and returns the final path."""
    yolo_class = _load_yolo_class()
    model = yolo_class(str(weights_path))

    export_kwargs: dict[str, Any] = {
        "format": "onnx",
        "dynamic": bool(dynamic),
        "imgsz": int(imgsz),
        "simplify": bool(simplify),
        "half": bool(half),
        "nms": False,
    }
    if opset is not None:
        export_kwargs["opset"] = int(opset)

    exported_path = Path(model.export(**export_kwargs)).expanduser().resolve()
    if output_path is None:
        return exported_path

    target_path = Path(output_path).expanduser()
    if target_path.suffix.lower() != ".onnx":
        target_path = target_path.with_suffix(".onnx")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path != target_path.resolve():
        shutil.move(str(exported_path), str(target_path))
    return target_path.resolve()


__all__ = [
    "InferenceBackend",
    "InferenceBox",
    "InferenceResult",
    "OnnxYoloBackend",
    "TorchYoloBackend",
    "decode_onnx_outputs",
    "export_model_to_onnx",
    "infer_backend_kind",
    "load_inference_backend",
    "normalize_yolo_results",
]
