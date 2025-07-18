import sys
import argparse
import os
import logging
from PyQt5.QtWidgets import QApplication
from .ui.main_window import ImageEditor
from .config import DEFAULT_WEIGHTS_PATH
import qt_material  # Импортируйте qt_material после PyQt5


def main() -> None:
    """Запускает графическое приложение."""

    parser = argparse.ArgumentParser(description="ImageEditor")
    parser.add_argument(
        "--weights",
        default=os.getenv("YOLO_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH)),
        help="Путь к весам YOLOv8",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
    )

    app = QApplication(sys.argv)
    qt_material.apply_stylesheet(app, theme="dark_blue.xml")
    window = ImageEditor(weights_path=args.weights)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
