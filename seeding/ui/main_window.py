"""Основное окно приложения."""

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QToolBar,
    QAction,
    QFileDialog,
    QGroupBox,
    QScrollArea,
)
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
from ..models.data_models import OriginalImage, ObjectImage
from .tree_widget import LayerTreeWidget
from ..utils import simple_nms
import numpy as np
import cv2
import fitz
from ultralytics import YOLO


class DraggableScrollArea(QScrollArea):
    """
    ScrollArea c возможностью перетаскивания средней кнопкой мыши.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_active = False
        self._drag_start_pos = QPoint()
        self._scroll_start_pos = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._drag_active = True
            self.setCursor(Qt.ClosedHandCursor)
            self._drag_start_pos = event.pos()
            self._scroll_start_pos = QPoint(
                self.horizontalScrollBar().value(), self.verticalScrollBar().value()
            )
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_active:
            delta = event.pos() - self._drag_start_pos
            self.horizontalScrollBar().setValue(self._scroll_start_pos.x() - delta.x())
            self.verticalScrollBar().setValue(self._scroll_start_pos.y() - delta.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._drag_active = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)


class ImageEditor(QMainWindow):
    """
    Главное окно приложения для работы с изображениями и PDF.

    Позволяет загружать файлы, управлять слоями и искать сеянцы при помощи YOLOv8.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Современный UI для работы с изображениями")
        self.setGeometry(100, 100, 1200, 800)
        self.zoom_factor = 1.0
        self.image_storage = OriginalImage()
        self.model = YOLO(
            "E:/_JOB_/_Python/pythonProject/results/exp_yolov8_custom_best11/weights/best.pt"
        )

        self.init_ui()

    def init_ui(self):
        self.create_menu()
        self.create_toolbars()
        self.create_central_widget()
        self.create_right_panel()

    def create_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        open_action = QAction("Открыть файл", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

    def create_toolbars(self):
        toolbar = QToolBar("Toolbar", self)
        toolbar.setOrientation(Qt.Vertical)
        toolbar.setMovable(False)
        toolbar.setFixedWidth(150)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)

        self.mask_action = QAction(QIcon(), "Создать маску", self)
        self.mask_action.triggered.connect(self.create_mask)
        toolbar.addAction(self.mask_action)

        self.seedlings_action = QAction(QIcon(), "Найти сеянцы", self)
        self.seedlings_action.triggered.connect(self.find_seedlings)
        toolbar.addAction(self.seedlings_action)

        self.find_all_seedlings_action = QAction(QIcon(), "Найти все сеянцы", self)
        self.find_all_seedlings_action.triggered.connect(self.find_all_seedlings)
        toolbar.addAction(self.find_all_seedlings_action)

        self.classify_action = QAction(QIcon(), "Классификация", self)
        self.classify_action.triggered.connect(self.classify)
        toolbar.addAction(self.classify_action)

        self.rotate_action = QAction(QIcon(), "Повернуть на 90°", self)
        self.rotate_action.triggered.connect(self.rotate_image)
        toolbar.addAction(self.rotate_action)

        toolbar.addSeparator()

        self.report_action = QAction(QIcon(), "Создать отчет", self)
        self.report_action.triggered.connect(self.create_report)
        toolbar.addAction(self.report_action)

        toolbar.addSeparator()

        self.zoom_in_action = QAction("Приблизить", self)
        self.zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Отдалить", self)
        self.zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(self.zoom_out_action)

        self.fit_action = QAction("Вписать", self)
        self.fit_action.triggered.connect(self.fit_to_window)
        toolbar.addAction(self.fit_action)

    def create_central_widget(self):
        self.scroll_area = DraggableScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.image_label = QLabel("Тут будет изображение")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px dashed gray;")

        self.scroll_area.setWidget(self.image_label)
        self.main_layout.addWidget(self.scroll_area, 2)

    def create_right_panel(self):
        self.right_panel = QGroupBox("Слои")
        layout = QVBoxLayout()
        self.tree_widget = LayerTreeWidget()
        layout.addWidget(self.tree_widget)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.tree_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)

        self.right_panel.setLayout(layout)
        self.main_layout.addWidget(self.right_panel, 1)

    def on_tree_item_clicked(self, item, column):
        item_data = item.data(0, Qt.UserRole)
        if item_data:
            if item_data["type"] in ("original", "pdf"):
                idx = item_data["index"]
                self._active_image_index = idx
                self.display_image_with_boxes(idx)
            elif item_data["type"] == "seeding":
                parent_idx = item_data["parent_index"]
                seed_idx = item_data["index"]
                crop = self.image_storage.class_object_image[parent_idx][
                    seed_idx
                ].image[0]
                self._active_image_index = parent_idx
                self.display_image(crop)
            else:
                return

    def open_image(self):
        self.image_storage = OriginalImage()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть изображение или PDF",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;PDF Files (*.pdf);;All Files (*)",
        )
        if file_name:
            self.image_storage.file_path = file_name
            self.image_storage.images.clear()
            self.tree_widget.clear()

            if file_name.lower().endswith(".pdf"):
                self.load_pdf(file_name)
            else:
                image = self.load_image(file_name)
                if image is not None:
                    self.image_storage.images.append(image)

            # Обязательно инициализируем пустые списки для найденных объектов
            self.image_storage.class_object_image = [
                [] for _ in range(len(self.image_storage.images))
            ]

    def load_image(self, file_name):
        try:
            image = cv2.imread(file_name)
            return image
        except Exception as e:
            print("Ошибка при загрузке изображения:", e)
            return None

    def load_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                mat = fitz.Matrix(4, 4)  # 2x масштаб
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:
                    img = img[:, :, :3].copy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.image_storage.images.append(img)
                # Для первой страницы — показать в QLabel
                if page_num == 0:
                    self.display_image(img)
                # Добавить в дерево
                self.tree_widget.add_root_item(
                    f"Стр. {page_num + 1}", "Страница PDF", page_num, "pdf", img
                )
            doc.close()

            # Инициализация class_object_image для всех страниц
            self.image_storage.class_object_image = [
                [] for _ in range(len(self.image_storage.images))
            ]

        except Exception as e:
            print("Ошибка при загрузке PDF:", e)

    def display_image(self, image):
        if image is None or not isinstance(image, np.ndarray):
            return
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return

        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            qformat = QImage.Format_RGB888
        elif len(image.shape) == 2:
            image_rgb = image
            bytes_per_line = width
            qformat = QImage.Format_Grayscale8
        else:
            return

        image_rgb = np.ascontiguousarray(image_rgb)
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, qformat)
        self._original_image = image
        self._original_pixmap = QPixmap.fromImage(q_image)

        scroll_size = self.scroll_area.viewport().size()
        ratio_w = scroll_size.width() / self._original_pixmap.width()
        ratio_h = scroll_size.height() / self._original_pixmap.height()
        self.min_fit_zoom = min(ratio_w, ratio_h, 1.0)
        self.zoom_factor = self.min_fit_zoom
        self.update_image_zoom()

    def zoom_in(self):
        self.zoom_factor *= 1.25
        self.update_image_zoom()

    def zoom_out(self):
        self.zoom_factor /= 1.25
        if self.zoom_factor < self.min_fit_zoom:
            self.zoom_factor = self.min_fit_zoom
        self.update_image_zoom()

    def fit_to_window(self):
        self.zoom_factor = self.min_fit_zoom
        self.update_image_zoom()

    def update_image_zoom(self):
        if hasattr(self, "_original_pixmap"):
            pixmap = self._original_pixmap
            new_width = max(1, int(pixmap.width() * self.zoom_factor))
            new_height = max(1, int(pixmap.height() * self.zoom_factor))
            scaled_pixmap = pixmap.scaled(
                new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.adjustSize()

    def rotate_image(self):
        selected_item = self.tree_widget.currentItem()
        if selected_item is None:
            print("rotate_image: Нет выбранного элемента в дереве")
            return

        item_data = selected_item.data(0, Qt.UserRole)
        if not item_data:
            print("rotate_image: Нет данных для выбранного элемента")
            return

        if item_data["type"] in ("original", "pdf"):
            idx = item_data["index"]
            image = self.image_storage.images[idx]
            if image is None:
                print("rotate_image: Оригинал отсутствует")
                return
            rotated = np.rot90(image, k=-1)
            self.image_storage.images[idx] = rotated
            print(f"rotate_image: Изображение {idx} повернуто")
            self.display_image(rotated)

        elif item_data["type"] == "seeding":
            parent_idx = item_data["parent_index"]
            seed_idx = item_data["index"]
            obj = self.image_storage.class_object_image[parent_idx][seed_idx]
            if not obj.image or obj.image[0] is None:
                print("rotate_image: Crop пустой")
                return
            crop = obj.image[0]
            rotated = np.rot90(crop, k=-1)
            self.image_storage.class_object_image[parent_idx][seed_idx].image[
                0
            ] = rotated
            print(f"rotate_image: Crop {seed_idx} (от оригинала {parent_idx}) повернут")
            self.display_image(rotated)
        else:
            print("rotate_image: Неизвестный тип данных")
            return

    def create_mask(self):
        print("Создание маски — пока не реализовано")

    def find_seedlings(self):
        if self.image_storage.class_object_image is None:
            self.image_storage.class_object_image = [
                [] for _ in range(len(self.image_storage.images))
            ]

        print("find_seedlings: start")
        current_index = getattr(self, "_active_image_index", 0)
        print(f"find_seedlings: current_index = {current_index}")

        if not self.image_storage.images:
            print("find_seedlings: Нет изображений для обработки")
            return

        image = self.image_storage.images[current_index]
        if image is None:
            print("find_seedlings: Текущее изображение пустое")
            return

        try:
            results = self.model(image)
            print(f"find_seedlings: модель вернула {len(results[0].boxes)} боксов")
        except Exception as e:
            print(f"Ошибка при вызове модели: {e}")
            return

        try:
            # Здесь дальше по коду — NMS и обработка
            # Пример простой NMS, как я писал ранее, чтобы избежать cv2.dnn.NMSBoxes
            boxes = []
            scores = []
            class_boxes_data = []
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = results[0].names[class_id]
                if class_name == "Seeding":
                    score = float(box.conf)
                    x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    h, w = image.shape[:2]
                    x1, x2 = max(0, x1), min(x2, w)
                    y1, y2 = max(0, y1), min(y2, h)
                    if x2 <= x1 or y2 <= y1:
                        print(
                            f"find_seedlings: пропускаем некорректный bbox {x1, y1, x2, y2}"
                        )
                        continue

                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    class_boxes_data.append(
                        {
                            "class_name": class_name,
                            "score": score,
                            "coords": (x1, y1, x2, y2),
                        }
                    )

            print(f"find_seedlings: найдено {len(boxes)} боксов, запускаем NMS")
            indices = simple_nms(boxes, scores, iou_threshold=0.4)
            print(f"find_seedlings: после NMS осталось {len(indices)} боксов")

            # Добавляем в dataclass и дерево
            self.image_storage.class_object_image[current_index] = []
            parent_item = self.tree_widget.topLevelItem(current_index)
            for i_out, i in enumerate(indices):
                data = class_boxes_data[i]
                x1, y1, x2, y2 = data["coords"]
                crop = image[y1:y2, x1:x2].copy()
                obj = ObjectImage(
                    class_name=data["class_name"],
                    confidence=data["score"],
                    image=[crop],
                    bbox=(x1, y1, x2, y2),
                )
                self.image_storage.class_object_image[current_index].append(obj)
                self.tree_widget.add_child_item(
                    parent_item,
                    f"Seeding{i_out + 1}",  # вместо "Сеянец {i_out + 1}"
                    f"Уверенность: {data['score']:.2f}",
                    current_index,
                    i_out,
                    "seeding",
                    crop,
                )

            print("find_seedlings: завершено")

        except Exception as e:
            print(f"Ошибка во время NMS или обработки результатов: {e}")
            return

    def find_all_seedlings(self):
        """
        Находит сеянцы на всех изображениях (или страницах PDF).
        """
        if not self.image_storage.images:
            print("find_all_seedlings: Нет изображений")
            return

        for idx, image in enumerate(self.image_storage.images):
            self._active_image_index = (
                idx  # чтобы всё работало так же, как для find_seedlings
            )
            self.find_seedlings()
        print("find_all_seedlings: завершено")

    def display_image_with_boxes(self, idx):
        image = self.image_storage.images[idx].copy()
        if (
            self.image_storage.class_object_image
            and len(self.image_storage.class_object_image) > idx
        ):
            for i, obj in enumerate(
                self.image_storage.class_object_image[idx], start=1
            ):
                if obj.bbox:
                    x1, y1, x2, y2 = obj.bbox
                    cv2.rectangle(
                        image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2
                    )
                    cv2.putText(
                        image,
                        f"{i}",  # Нумерация
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
        self.display_image(image)

    def classify(self):
        print("Классификация — пока не реализовано")

    def create_report(self):
        print("Создание отчёта — пока не реализовано")
