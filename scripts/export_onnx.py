"""CLI helper for exporting Ultralytics models to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Parses CLI arguments and exports the model to ONNX."""
    from seeding.inference import export_model_to_onnx

    parser = argparse.ArgumentParser(description="Export a YOLO model to ONNX")
    parser.add_argument("weights", help="Path to the source .pt weights")
    parser.add_argument(
        "--output",
        help=(
            "Optional target .onnx path. "
            "Defaults to Ultralytics export path."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size used by Ultralytics export",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Disable dynamic axes in the exported ONNX model",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Request graph simplification during export",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Request FP16 export when supported by the environment",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="Optional ONNX opset version override (default: 20)",
    )
    args = parser.parse_args()

    output_path = export_model_to_onnx(
        args.weights,
        output_path=args.output,
        dynamic=not args.static,
        imgsz=args.imgsz,
        simplify=args.simplify,
        half=args.half,
        opset=args.opset,
    )
    print(output_path)


if __name__ == "__main__":
    main()
