import sys
from PyQt5.QtWidgets import QApplication
from .ui_main import ImageEditor
import qt_material  # Импортируйте qt_material после PyQt5


def main():
    app = QApplication(sys.argv)
    qt_material.apply_stylesheet(app, theme="dark_blue.xml")
    window = ImageEditor()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
