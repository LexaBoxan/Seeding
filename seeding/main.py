import sys
from PyQt5.QtWidgets import QApplication
from .ui.main_window import ImageEditor
import qt_material  # Импортируйте qt_material после PyQt5


def main():
    """Точка входа в приложение."""
    app = QApplication(sys.argv)
    qt_material.apply_stylesheet(app, theme="dark_blue.xml")
    window = ImageEditor()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
