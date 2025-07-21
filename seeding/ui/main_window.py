"""Основное окно приложения."""

import logging
import os

import cv2
import fitz
import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QGroupBox,
    QLabel,
    QHBoxLayout,
    QMainWindow,
    QScrollArea,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

from seeding.config import ROTATE_K
from seeding.models.data_models import ObjectImage, OriginalImage
from seeding.utils import simple_nms
from .tree_widget import LayerTreeWidget

logger = logging.getLogger(__name__)


class DraggableScrollArea(QScrollArea):
    """
    ScrollArea c возможностью перетаскивания средней кнопкой мыши.
    """

    def __init__(self, parent=None):
        """Конструктор виджета с поддержкой перетаскивания."""
        super().__init__(parent)
        self._drag_active = False
        self._drag_start_pos = QPoint()
        self._scroll_start_pos = QPoint()

    def mousePressEvent(self, event):
        """Начинает перетаскивание при нажатии средней кнопкой мыши."""
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
        """Перемещает содержимое при активном перетаскивании."""
        if self._drag_active:
            delta = event.pos() - self._drag_start_pos
            self.horizontalScrollBar().setValue(self._scroll_start_pos.x() - delta.x())
            self.verticalScrollBar().setValue(self._scroll_start_pos.y() - delta.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Завершает перетаскивание."""
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

    def __init__(self, weights_path: str):
        """Инициализирует окно и загружает модель.

        Args:
            weights_path: Путь к файлу весов YOLOv8.
        """
        super().__init__()
        self.setWindowTitle("Современный UI для работы с изображениями")
        self.setGeometry(100, 100, 1200, 800)
        self.zoom_factor = 1.0
        self.image_storage = OriginalImage()
        self.model = YOLO(weights_path)

        self.init_ui()

    def init_ui(self):
        """Создаёт все основные виджеты интерфейса."""
        self.create_menu()
        self.create_toolbars()
        self.create_central_widget()
        self.create_right_panel()

    def create_menu(self):
        """Создаёт меню приложения с пунктом открытия файла."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        open_action = QAction("Открыть файл", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

    def create_toolbars(self):
        """Создаёт боковую панель инструментов."""
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
        """Создаёт центральную область отображения изображений."""
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
        """Создаёт правую панель с деревом слоёв."""
        self.right_panel = QGroupBox("Слои")
        layout = QVBoxLayout()
        self.tree_widget = LayerTreeWidget()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.tree_widget)
        layout.addWidget(scroll_area)

        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)

        self.right_panel.setLayout(layout)
        self.main_layout.addWidget(self.right_panel, 1)

    def on_tree_item_clicked(self, item, column):
        """Обрабатывает выбор элемента в дереве слоёв."""
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

    def open_image(self) -> None:
        """Открывает диалог выбора файла и загружает изображение или PDF."""
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

    def load_image(self, file_name: str) -> np.ndarray | None:
        """Загружает изображение с диска."""
        try:
            image = cv2.imread(file_name)
            return image
        except Exception as e:
            logger.error("Ошибка при загрузке изображения: %s", e)
            return None

    def load_pdf(self, pdf_path: str) -> None:
        """Загружает все страницы PDF как изображения."""
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
            logger.error("Ошибка при загрузке PDF: %s", e)

    def display_image(self, image: np.ndarray) -> None:
        """Отображает переданное изображение в центральной области."""
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

    def zoom_in(self) -> None:
        """Увеличивает изображение."""
        self.zoom_factor *= 1.25
        self.update_image_zoom()

    def zoom_out(self) -> None:
        """Уменьшает изображение."""
        self.zoom_factor /= 1.25
        if self.zoom_factor < self.min_fit_zoom:
            self.zoom_factor = self.min_fit_zoom
        self.update_image_zoom()

    def fit_to_window(self) -> None:
        """Масштабирует изображение так, чтобы оно поместилось в окно."""
        self.zoom_factor = self.min_fit_zoom
        self.update_image_zoom()

    def update_image_zoom(self) -> None:
        """Применяет текущий масштаб к изображению."""
        if hasattr(self, "_original_pixmap"):
            pixmap = self._original_pixmap
            new_width = max(1, int(pixmap.width() * self.zoom_factor))
            new_height = max(1, int(pixmap.height() * self.zoom_factor))
            scaled_pixmap = pixmap.scaled(
                new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.adjustSize()

    def rotate_image(self) -> None:
        """Поворачивает выбранное изображение или crop на 90 градусов."""
        selected_item = self.tree_widget.currentItem()
        if selected_item is None:
            logger.warning("rotate_image: Нет выбранного элемента в дереве")
            return

        item_data = selected_item.data(0, Qt.UserRole)
        if not item_data:
            logger.warning("rotate_image: Нет данных для выбранного элемента")
            return

        if item_data["type"] in ("original", "pdf"):
            idx = item_data["index"]
            image = self.image_storage.images[idx]
            if image is None:
                logger.warning("rotate_image: Оригинал отсутствует")
                return
            rotated = np.rot90(image, k=ROTATE_K)
            self.image_storage.images[idx] = rotated
            logger.info("rotate_image: Изображение %s повернуто", idx)
            self.display_image(rotated)

        elif item_data["type"] == "seeding":
            parent_idx = item_data["parent_index"]
            seed_idx = item_data["index"]
            obj = self.image_storage.class_object_image[parent_idx][seed_idx]
            if not obj.image or obj.image[0] is None:
                logger.warning("rotate_image: Crop пустой")
                return
            crop = obj.image[0]
            rotated = np.rot90(crop, k=ROTATE_K)
            self.image_storage.class_object_image[parent_idx][seed_idx].image[
                0
            ] = rotated
            logger.info(
                "rotate_image: Crop %s (от оригинала %s) повернут",
                seed_idx,
                parent_idx,
            )
            self.display_image(rotated)
        else:
            logger.warning("rotate_image: Неизвестный тип данных")
            return

    def create_mask(self) -> None:
        """Создание маски (функциональность пока не реализована)."""
        logger.info("Создание маски — пока не реализовано")

    def find_seedlings(self) -> None:
        """Запускает модель YOLOv8 для поиска сеянцев на текущем изображении.

        Результаты проходят через простую процедуру NMS. Каждая найденная
        область добавляется в хранилище `image_storage` и отображается в дереве
        слоёв. Если ширина вырезанного участка больше его высоты, изображение
        поворачивается на 90 градусов для вертикальной ориентации.
        """
        if self.image_storage.class_object_image is None:
            self.image_storage.class_object_image = [
                [] for _ in range(len(self.image_storage.images))
            ]

        logger.info("find_seedlings: start")
        current_index = getattr(self, "_active_image_index", 0)
        logger.debug("find_seedlings: current_index = %s", current_index)

        if not self.image_storage.images:
            logger.warning("find_seedlings: Нет изображений для обработки")
            return

        image = self.image_storage.images[current_index]
        if image is None:
            logger.warning("find_seedlings: Текущее изображение пустое")
            return

        try:
            results = self.model(image)
            logger.debug("find_seedlings: модель вернула %s боксов", len(results[0].boxes))
        except Exception as e:
            logger.error("Ошибка при вызове модели: %s", e)
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
                        logger.debug(
                            "find_seedlings: пропускаем некорректный bbox %s",
                            (x1, y1, x2, y2),
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

            logger.info(
                "find_seedlings: найдено %s боксов, запускаем NMS", len(boxes)
            )
            indices = simple_nms(boxes, scores, iou_threshold=0.4)
            logger.info(
                "find_seedlings: после NMS осталось %s боксов", len(indices)
            )

            # Добавляем в dataclass и дерево
            self.image_storage.class_object_image[current_index] = []
            parent_item = self.tree_widget.topLevelItem(current_index)
            for i_out, i in enumerate(indices):
                data = class_boxes_data[i]
                x1, y1, x2, y2 = data["coords"]
                crop = image[y1:y2, x1:x2].copy()
                # Проверяем ориентацию crop'a. Если ширина больше высоты,
                # поворачиваем изображение на 90 градусов, чтобы оно стало вертикальным
                if crop.shape[1] > crop.shape[0]:
                    crop = np.rot90(crop, k=ROTATE_K)
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

            logger.info("find_seedlings: завершено")

        except Exception as e:
            logger.error(
                "Ошибка во время NMS или обработки результатов: %s", e
            )
            return

    def find_all_seedlings(self) -> None:
        """Последовательно запускает поиск сеянцев на всех изображениях."""
        if not self.image_storage.images:
            logger.warning("find_all_seedlings: Нет изображений")
            return

        for idx, image in enumerate(self.image_storage.images):
            self._active_image_index = (
                idx  # чтобы всё работало так же, как для find_seedlings
            )
            self.find_seedlings()
        logger.info("find_all_seedlings: завершено")

    def display_image_with_boxes(self, idx: int) -> None:
        """Отображает изображение с нанесёнными рамками объектов."""
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

    def classify(self) -> None:
        """Классифицирует найденные объекты (заглушка)."""
        logger.info("Классификация — пока не реализовано")

    def create_report(self) -> None:
        """Создаёт PDF-отчёт по текущим результатам детекции."""
        if not self.image_storage.images:
            logger.warning("create_report: Нет данных для отчёта")
            return

        base_path, _ = os.path.splitext(self.image_storage.file_path)
        output_path = base_path + "_report.pdf"
        try:
            from ..report import create_pdf_report

            create_pdf_report(self.image_storage, output_path)
            logger.info("Отчёт сохранён: %s", output_path)
        except Exception as e:
            logger.error("Ошибка при создании отчёта: %s", e)
