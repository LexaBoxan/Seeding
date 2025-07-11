import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QImage, QPixmap

app = QApplication(sys.argv)

# Попробуй загрузить самую простую картинку
image = cv2.imread('C:\\Users\\alesh\\Desktop\\SCAN0006\\SCAN0006_page-0001.jpg')  # Подставь путь к своему PNG/JPG
if image is None:
    print("Не удалось загрузить изображение!")
    sys.exit(1)

h, w = image.shape[:2]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bytes_per_line = 3 * w
q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
pixmap = QPixmap.fromImage(q_image)

label = QLabel()
label.setPixmap(pixmap)
label.show()

sys.exit(app.exec_())
