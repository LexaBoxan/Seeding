import argparse
import logging
import os
import sys

from PyQt5.QtWidgets import QApplication

from seeding.config import DEFAULT_WEIGHTS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeding Analyzer")
    parser.add_argument(
        "--weights",
        default=os.getenv("YOLO_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH)),
        help="Путь к весам модели детекции",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
    )

    # Создаём приложение
    app = QApplication(sys.argv)

    # Применяем тему ПОСЛЕ создания приложения
    import qt_material
    qt_material.apply_stylesheet(app, theme="dark_blue.xml")

    # Импорт главного окна
    from seeding.ui.main_window import ImageEditor

    window = ImageEditor(weights_path=args.weights)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()