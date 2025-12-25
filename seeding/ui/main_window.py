"""Основное окно приложения."""

import logging
import os
import csv

import cv2
import fitz
import numpy as np
from PyQt5.QtCore import QPoint, Qt, QRectF
from PyQt5.QtGui import QImage, QPixmap, QTransform, QColor

from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QGroupBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHeaderView,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QDialog,
    QHBoxLayout,
)

from seeding.config import ROTATE_K, DEFAULT_CLASSIFY_WEIGHTS_PATH, DEFAULT_ROOT_CLASSIFY_WEIGHTS_PATH
from seeding.models.data_models import AllClassImage, ObjectImage, OriginalImage
from seeding.utils import simple_nms, rotate_bbox
from seeding.application.root_analysis import (
    RootAnalyzer,
    RootAnalysisResult,
    RootViability,
)
from .tree_widget import LayerTreeWidget
from .bbox_item import BBoxItem

from seeding.application.pipeline import SeedlingPipeline

logger = logging.getLogger(__name__)

EXPECTED_CLASSIFY_NAMES = ["flower", "root", "stem"]


class DraggableScrollArea(QScrollArea):
    """ScrollArea с перетаскиванием средней кнопкой мыши."""

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
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_active:
            delta = event.pos() - self._drag_start_pos
            self.horizontalScrollBar().setValue(self._scroll_start_pos.x() - delta.x())
            self.verticalScrollBar().setValue(self._scroll_start_pos.y() - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._drag_active = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)


class ImageEditor(QMainWindow):
    """Главное окно приложения."""

    def __init__(self, weights_path: str):
        super().__init__()
        self.setWindowTitle("Анализ сеянцев с оценкой корневой системы")
        self.setGeometry(100, 100, 1400, 900)
        self.zoom_factor = 1.0
        self.image_storage = OriginalImage()
        self.weights_path = weights_path

        # Конвейер с поддержкой трёх моделей
        self.pipeline = SeedlingPipeline(
            detection_weights=weights_path,
            classify_weights=str(DEFAULT_CLASSIFY_WEIGHTS_PATH),
            root_classify_weights=str(DEFAULT_ROOT_CLASSIFY_WEIGHTS_PATH),
            root_analyzer=RootAnalyzer()
        )

        self.root_analyzer = self.pipeline.root_analyzer

        self.init_ui()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    def export_to_csv(self) -> None:
        """Экспорт результатов анализа корневой системы в CSV-файл."""
        if not self.image_storage.class_object_image or not any(self.image_storage.class_object_image):
            logger.warning("Нет данных для экспорта в CSV")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not file_name:
            return

        try:
            with open(file_name, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Страница", "Сеянец", "Оценка", "Длина (px)", "Толщина (px)",
                    "Ветвистость", "Плотность", "Уверенность", "Жизнеспособность"
                ])

                for img_idx, objects in enumerate(self.image_storage.class_object_image):
                    for obj_idx, obj in enumerate(objects):
                        if obj.root_analysis:
                            result = max(obj.root_analysis, key=lambda r: r.score)
                            morph = result.morphology
                            writer.writerow([
                                img_idx + 1,
                                obj_idx + 1,
                                round(result.score, 3),
                                round(morph.length, 1),
                                round(morph.mean_thickness, 2),
                                round(morph.branching_index, 3),
                                round(morph.density, 3),
                                round(result.confidence, 2),
                                result.viability.value
                            ])
            logger.info(f"CSV успешно сохранён: {file_name}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении CSV: {e}")
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

    def _update_action_states(self) -> None:
        """Включает или отключает действия, зависящие от наличия изображения."""
        has_image = bool(self.image_storage.images)
        has_classes = self._has_classified_parts()
        for action in (
            self.rotate_action,
            self.seedlings_action,
            self.find_all_seedlings_action,
            self.classify_action,
            self.report_action,
            self.save_action,
        ):
            action.setEnabled(has_image)
        if hasattr(self, "root_analysis_action"):
            self.root_analysis_action.setEnabled(has_image and has_classes)

    def _has_classified_parts(self) -> bool:
        if not self.image_storage.class_object_image:
            return False
        for objects in self.image_storage.class_object_image:
            for obj in objects:
                if obj.image_all_class:
                    return True
        return False

    def create_toolbars(self):
        toolbar = QToolBar("Toolbar", self)
        toolbar.setOrientation(Qt.Vertical)
        toolbar.setMovable(False)
        toolbar.setFixedWidth(150)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        style = self.style()
        self.mask_action = QAction(
            style.standardIcon(QStyle.SP_FileDialogNewFolder), "Создать маску", self
        )
        self.mask_action.triggered.connect(self.create_mask)
        toolbar.addAction(self.mask_action)
        self.seedlings_action = QAction(
            style.standardIcon(QStyle.SP_MediaPlay), "Найти сеянцы", self
        )
        self.seedlings_action.triggered.connect(self.find_seedlings)
        toolbar.addAction(self.seedlings_action)
        self.find_all_seedlings_action = QAction(
            style.standardIcon(QStyle.SP_DialogYesButton), "Найти все сеянцы", self
        )
        self.find_all_seedlings_action.triggered.connect(self.find_all_seedlings)
        toolbar.addAction(self.find_all_seedlings_action)
        self.classify_action = QAction(
            style.standardIcon(QStyle.SP_FileDialogDetailedView), "Классификация", self
        )
        self.classify_action.triggered.connect(self.classify)
        toolbar.addAction(self.classify_action)
        self.root_analysis_action = QAction(
            style.standardIcon(QStyle.SP_DialogApplyButton),
            "Анализ корней",
            self,
        )

        self.full_report_action = QAction(
            style.standardIcon(QStyle.SP_FileDialogContentsView),
            "Отчёт по всем сеянцам",
            self,
        )
        self.full_report_action.triggered.connect(self.full_root_report)
        toolbar.addAction(self.full_report_action)
        self.root_analysis_action.triggered.connect(self.analyze_roots)
        toolbar.addAction(self.root_analysis_action)
        self.rotate_action = QAction(
            style.standardIcon(QStyle.SP_BrowserReload), "Повернуть на 90°", self
        )
        self.rotate_action.triggered.connect(self.rotate_image)
        toolbar.addAction(self.rotate_action)
        toolbar.addSeparator()
        self.report_action = QAction(
            style.standardIcon(QStyle.SP_FileDialogContentsView), "Создать отчет", self
        )
        self.report_action.triggered.connect(self.create_report)
        toolbar.addAction(self.report_action)
        toolbar.addSeparator()
        self.zoom_in_action = QAction(
            style.standardIcon(QStyle.SP_ArrowUp), "Приблизить", self
        )
        self.zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(self.zoom_in_action)
        self.zoom_out_action = QAction(
            style.standardIcon(QStyle.SP_ArrowDown), "Отдалить", self
        )
        self.zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(self.zoom_out_action)
        self.fit_action = QAction(
            style.standardIcon(QStyle.SP_DesktopIcon), "Вписать", self
        )
        self.fit_action.triggered.connect(self.fit_to_window)
        toolbar.addAction(self.fit_action)
        toolbar.addSeparator()
        self.save_action = QAction(
            style.standardIcon(QStyle.SP_DialogSaveButton),
            "Сохранить изменения",
            self,
        )
        self.save_action.triggered.connect(self.save_changes)
        toolbar.addAction(self.save_action)

        # Новый: Экспорт в CSV
        self.export_csv_action = QAction(
            style.standardIcon(QStyle.SP_FileDialogListView), "Экспорт в CSV", self
        )
        self.export_csv_action.triggered.connect(self.export_to_csv)
        toolbar.addAction(self.export_csv_action)

        self._update_action_states()

    def create_central_widget(self):
        """Создаёт центральную область отображения изображений."""
        self.scroll_area = DraggableScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        self.image_item = QGraphicsPixmapItem()
        self.graphics_scene.addItem(self.image_item)
        self.rect_items = {}
        self.scroll_area.setWidget(self.graphics_view)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.scroll_area)
        self.setCentralWidget(self.splitter)

    def create_right_panel(self):
        """Создаёт правую панель с деревом слоёв."""
        self.right_panel = QGroupBox("Слои")
        self.right_panel.setMinimumWidth(200)
        layout = QVBoxLayout()
        self.tree_widget = LayerTreeWidget()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.tree_widget)
        layout.addWidget(scroll_area)
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)
        self.right_panel.setLayout(layout)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setCollapsible(1, False)

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
                self._active_image_index = parent_idx
                self.display_seeding_with_boxes(parent_idx, seed_idx)
            elif item_data["type"] == "class":
                parent_idx = item_data["parent_index"]
                seed_idx = item_data["seeding_index"]
                class_idx = item_data["class_index"]
                self._active_image_index = parent_idx

                obj = self.image_storage.class_object_image[parent_idx][seed_idx]
                cls = obj.image_all_class[class_idx]

                class_name_lower = cls.class_name.lower()

                if class_name_lower == "root" and obj.root_analysis:
                    # Специальный случай для корня — показываем визуализацию анализа, а не сырую маску
                    result = max(obj.root_analysis, key=lambda r: r.score)
                    try:
                        vis = self.root_analyzer.visualize(result)
                        self.display_image(vis)
                        logger.info("Показана визуализация анализа корня")
                    except Exception as e:
                        logger.error(f"Ошибка визуализации корня: {e}")
                        self.display_image(obj.image[0] if obj.image else np.zeros((400, 400, 3), dtype=np.uint8))
                else:
                    # Для flower и stem — показываем обычное изображение части
                    if cls.image is not None:
                        self.display_image(cls.image)
                    else:
                        logger.warning("cls.image пустое для класса %s", cls.class_name)
                        # Fallback на crop сеянца
                        self.display_image(obj.image[0] if obj.image else np.zeros((400, 400, 3), dtype=np.uint8))
            else:
                return

    def open_image(self) -> None:
        """Открывает диалог выбора файла и загружает изображение или PDF."""
        self.image_storage = OriginalImage()
        self._update_action_states()
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
                    self.display_image(image)
                    self._active_image_index = 0
                    self.tree_widget.add_root_item(
                        "Оригинал", "Исходное изображение", 0, "original", image
                    )
            # Инициализация class_object_image
            self.image_storage.class_object_image = [
                [] for _ in range(len(self.image_storage.images))
            ]
        self._update_action_states()

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
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, doc.page_count)
            self.progress_bar.setValue(0)
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
                if page_num == 0:
                    self.display_image(img)
                self.tree_widget.add_root_item(
                    f"Стр. {page_num + 1}", "Страница PDF", page_num, "pdf", img
                )
                self.progress_bar.setValue(page_num + 1)
            doc.close()
            self.image_storage.class_object_image = [
                [] for _ in range(len(self.image_storage.images))
            ]
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
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
        self.graphics_scene.clear()
        self.image_item = self.graphics_scene.addPixmap(self._original_pixmap)
        self.rect_items = {}
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
            transform = QTransform()
            transform.scale(self.zoom_factor, self.zoom_factor)
            self.graphics_view.setTransform(transform)
            self.graphics_scene.setSceneRect(
                0,
                0,
                self._original_pixmap.width(),
                self._original_pixmap.height(),
            )

    def rotate_image(self) -> None:
        """Поворачивает выбранное изображение или crop на 90 градусов."""
        self._update_action_states()
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
            obj.rotation_k = (obj.rotation_k + ROTATE_K) % 4
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
        """Запускает полный конвейер на текущем изображении: детекция + сегментация + анализ корней."""
        self._update_action_states()
        current_index = getattr(self, "_active_image_index", 0)
        if not self.image_storage.images or current_index >= len(self.image_storage.images):
            logger.warning("find_seedlings: Нет изображений")
            return

        image = self.image_storage.images[current_index]
        if image is None:
            return

        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        # Полный конвейер
        detections = self.pipeline.process([image])[0]

        # Очистка предыдущих результатов
        self.image_storage.class_object_image[current_index] = []
        parent_item = self.tree_widget.topLevelItem(current_index)
        if parent_item is not None:
            for i in reversed(range(parent_item.childCount())):
                parent_item.takeChild(i)

        # Заполнение хранилища и дерева
        for i, det in enumerate(detections):
            obj = ObjectImage(
                class_name="seeding",
                confidence=det.confidence,
                image=[det.crop],
                bbox=det.bbox,
                rotation_k=det.rotation_k,
                root_analysis=det.roots,
            )

            # Сохраняем все части (flower, stem, root) с масками
            obj.image_all_class = []
            for part in det.parts:
                all_class = AllClassImage(
                    class_name=part["class_name"],
                    confidence=part["confidence"],
                    image=part["image"],
                    bbox=part["bbox"],
                    mask=part["mask"],
                )
                obj.image_all_class.append(all_class)

            self.image_storage.class_object_image[current_index].append(obj)

            child = self.tree_widget.add_child_item(
                parent_item,
                f"Seeding {i + 1}",
                f"Уверенность: {det.confidence:.2f}",
                current_index,
                i,
                "seeding",
                det.crop,
            )

            # Подпункты для частей
            for part_idx, part in enumerate(det.parts):
                self.tree_widget.add_class_item(
                    child,
                    part["class_name"].capitalize(),
                    f"Уверенность: {part['confidence']:.2f}",
                    current_index,
                    i,
                    part_idx,
                )

        self.display_image_with_boxes(current_index)
        self.progress_bar.setVisible(False)
        self._update_action_states()
        logger.info("find_seedlings: завершено с анализом корней")

    def find_all_seedlings(self) -> None:
        """Запускает полный конвейер на всех изображениях."""
        self._update_action_states()
        if not self.image_storage.images:
            logger.warning("find_all_seedlings: Нет изображений")
            return

        total = len(self.image_storage.images)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)

        all_detections = self.pipeline.process(self.image_storage.images)

        for idx, detections in enumerate(all_detections):
            self.progress_bar.setValue(idx)
            self._active_image_index = idx

            self.image_storage.class_object_image[idx] = []
            parent_item = self.tree_widget.topLevelItem(idx)
            if parent_item is not None:
                for i in reversed(range(parent_item.childCount())):
                    parent_item.takeChild(i)

            for i, det in enumerate(detections):
                obj = ObjectImage(
                    class_name="seeding",
                    confidence=det.confidence,
                    image=[det.crop],
                    bbox=det.bbox,
                    rotation_k=det.rotation_k,
                    root_analysis=det.roots,
                )
                obj.image_all_class = []
                for part in det.parts:
                    all_class = AllClassImage(
                        class_name=part["class_name"],
                        confidence=part["confidence"],
                        image=part["image"],
                        bbox=part["bbox"],
                        mask=part["mask"],
                    )
                    obj.image_all_class.append(all_class)

                self.image_storage.class_object_image[idx].append(obj)

                child = self.tree_widget.add_child_item(
                    parent_item,
                    f"Seeding {i + 1}",
                    f"Уверенность: {det.confidence:.2f}",
                    idx,
                    i,
                    "seeding",
                    det.crop,
                )
                for part_idx, part in enumerate(det.parts):
                    self.tree_widget.add_class_item(
                        child,
                        part["class_name"].capitalize(),
                        f"Уверенность: {part['confidence']:.2f}",
                        idx,
                        i,
                        part_idx,
                    )

            self.progress_bar.setValue(idx + 1)

        self.progress_bar.setVisible(False)
        self.display_image_with_boxes(getattr(self, "_active_image_index", 0))
        self._update_action_states()
        logger.info("find_all_seedlings: завершено")

    def display_image_with_boxes(self, idx: int) -> None:
        """Отображает изображение с нанесёнными рамками объектов — с полной защитой от некорректных bbox."""
        logger.info(f"display_image_with_boxes: начало для изображения {idx}")

        if idx >= len(self.image_storage.images):
            logger.warning("display_image_with_boxes: индекс вне диапазона")
            return

        image = self.image_storage.images[idx].copy()
        self.display_image(image)

        if not self.image_storage.class_object_image or idx >= len(self.image_storage.class_object_image):
            logger.info("display_image_with_boxes: нет классифицированных объектов")
            return

        objects = self.image_storage.class_object_image[idx]
        image_height, image_width = image.shape[:2]

        logger.info(f"display_image_with_boxes: {len(objects)} объектов для отрисовки")

        for obj_idx, obj in enumerate(objects):
            if not obj.bbox:
                logger.warning(f"Объект {obj_idx}: bbox отсутствует — пропускаем")
                continue

            x1, y1, x2, y2 = obj.bbox

            # Жёсткая нормализация bbox
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)

            width = max(30, right - left)  # минимум 30 пикселей
            height = max(30, bottom - top)

            # Обрезка по границам изображения
            left = max(0, left)
            top = max(0, top)
            right = min(image_width, left + width)
            bottom = min(image_height, top + height)

            if right <= left or bottom <= top:
                logger.warning(f"Объект {obj_idx}: bbox после нормализации пустой — пропускаем")
                continue

            rect = QRectF(left, top, right - left, bottom - top)

            try:
                rect_item = BBoxItem(rect, obj)
                rect_item.setEditable(True)
                self.graphics_scene.addItem(rect_item)
                self.rect_items[(idx, obj_idx)] = rect_item
                logger.info(f"Успешно добавлен bbox: ({left}, {top}, {right}, {bottom})")
            except Exception as e:
                logger.error(f"Ошибка при создании BBoxItem для объекта {obj_idx}: {e}")

        logger.info("display_image_with_boxes: завершено")

    def display_seeding_with_boxes(self, parent_idx: int, seed_idx: int) -> None:
        """Отображает crop сеянца с его классификационными боксами."""
        if (
            not self.image_storage.class_object_image
            or parent_idx >= len(self.image_storage.class_object_image)
            or seed_idx >= len(self.image_storage.class_object_image[parent_idx])
        ):
            return
        obj = self.image_storage.class_object_image[parent_idx][seed_idx]
        if not obj.image:
            return
        crop_img = obj.image[0].copy()
        self.display_image(crop_img)
        if obj.image_all_class:
            for cls_idx, cls_obj in enumerate(obj.image_all_class):
                if cls_obj.bbox:
                    lx1, ly1, lx2, ly2 = cls_obj.bbox
                    rect = QRectF(lx1, ly1, lx2 - lx1, ly2 - ly1)
                    rect_item = BBoxItem(rect, cls_obj, color=Qt.red)
                    rect_item.setEditable(True)
                    self.graphics_scene.addItem(rect_item)
                    self.rect_items[(parent_idx, seed_idx, cls_idx)] = rect_item

    def display_seeding_with_boxes(self, parent_idx: int, seed_idx: int) -> None:
        """Отображает crop сеянца с его частями — безопасно."""
        if (parent_idx >= len(self.image_storage.class_object_image) or
                seed_idx >= len(self.image_storage.class_object_image[parent_idx])):
            return

        obj = self.image_storage.class_object_image[parent_idx][seed_idx]
        if not obj.image or not obj.image[0].size:
            return

        crop_img = obj.image[0].copy()
        h, w = crop_img.shape[:2]
        self.display_image(crop_img)

        if not obj.image_all_class:
            return

        for cls_idx, cls_obj in enumerate(obj.image_all_class):
            if not cls_obj.bbox:
                continue

            x1, y1, x2, y2 = cls_obj.bbox
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)

            width = max(20, right - left)
            height = max(20, bottom - top)

            left = max(0, left)
            top = max(0, top)
            right = min(w, left + width)
            bottom = min(h, top + height)

            if right <= left or bottom <= top:
                continue

            rect = QRectF(left, top, right - left, bottom - top)

            try:
                rect_item = BBoxItem(rect, cls_obj, color=Qt.red)
                rect_item.setEditable(True)
                self.graphics_scene.addItem(rect_item)
                self.rect_items[(parent_idx, seed_idx, cls_idx)] = rect_item
            except Exception as e:
                logger.error(f"Ошибка отрисовки части {cls_idx}: {e}")

    def save_changes(self) -> None:
        """Пересохраняет crop-изображения после изменения рамок."""
        if not self.image_storage.images or not self.image_storage.class_object_image:
            return
        for img_idx, objects in enumerate(self.image_storage.class_object_image):
            if img_idx >= len(self.image_storage.images):
                continue
            base_img = self.image_storage.images[img_idx]
            for obj in objects:
                if obj.bbox:
                    x1, y1, x2, y2 = obj.bbox
                    crop = base_img[y1:y2, x1:x2].copy()
                    if getattr(obj, "rotation_k", 0):
                        crop = np.rot90(crop, k=obj.rotation_k)
                    obj.image = [crop]
                if obj.image_all_class:
                    k = getattr(obj, "rotation_k", 0) % 4
                    h_rot, w_rot = obj.image[0].shape[:2] if obj.image else (0, 0)
                    for cls in obj.image_all_class:
                        if cls.bbox:
                            lx1, ly1, lx2, ly2 = cls.bbox
                            if k and h_rot and w_rot:
                                ux1, uy1, ux2, uy2 = rotate_bbox(
                                    lx1, ly1, lx2, ly2, w_rot, h_rot, (-k) % 4
                                )
                            else:
                                ux1, uy1, ux2, uy2 = lx1, ly1, lx2, ly2
                            if obj.bbox:
                                gx1 = obj.bbox[0] + ux1
                                gy1 = obj.bbox[1] + uy1
                                gx2 = obj.bbox[0] + ux2
                                gy2 = obj.bbox[1] + uy2
                            else:
                                gx1, gy1, gx2, gy2 = ux1, uy1, ux2, uy2
                            part = base_img[gy1:gy2, gx1:gx2].copy()
                            if k:
                                part = np.rot90(part, k=k)
                            cls.image = part
        logger.info("save_changes: обновлённые координаты сохранены")

    def classify(self) -> None:
        """Классификация частей (fallback, если pipeline не использовался)."""
        logger.info("classify: уже выполнено в pipeline, пропускаем")
        self._update_action_states()

    def analyze_roots(self) -> None:
        """Открывает отчёт по корню для выбранного сеянца в дереве."""
        selected_item = self.tree_widget.currentItem()
        if not selected_item:
            logger.warning("Нет выбранного сеянца для анализа корня")
            return

        item_data = selected_item.data(0, Qt.UserRole)
        if not item_data or item_data["type"] != "seeding":
            logger.warning("Выбранный элемент не сеянец")
            return

        parent_idx = item_data["parent_index"]
        seed_idx = item_data["index"]

        obj = self.image_storage.class_object_image[parent_idx][seed_idx]

        if not obj.root_analysis:
            logger.warning("Нет анализа корня для этого сеянца")
            return

        self._show_root_report(obj)

    def _build_root_mask(
        self,
        seeding_obj: ObjectImage,
        cls_obj: AllClassImage,
    ) -> np.ndarray:
        """Генерирует маску корня по кропу класса и его bbox."""
        base_img = seeding_obj.image[0] if seeding_obj.image else None
        if base_img is None:
            return np.zeros((1, 1), dtype=np.uint8)
        h, w = base_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        local_mask = None
        if isinstance(cls_obj.image, np.ndarray):
            cls_img = cls_obj.image
            if cls_img.ndim == 3 and cls_img.shape[2] == 3:
                cls_img = cv2.cvtColor(cls_img, cv2.COLOR_BGR2GRAY)
            _, local_mask = cv2.threshold(
                cls_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        if cls_obj.bbox:
            x1, y1, x2, y2 = map(int, cls_obj.bbox)
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 > x1 and y2 > y1:
                if local_mask is not None:
                    resized = cv2.resize(local_mask, (x2 - x1, y2 - y1))
                    mask[y1:y2, x1:x2] = resized
                else:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        elif local_mask is not None:
            mask = cv2.resize(local_mask, (w, h))
        return mask

    @staticmethod
    def _array_to_qpixmap(image: np.ndarray, max_size: int = 220) -> QPixmap | None:
        if image is None or not isinstance(image, np.ndarray):
            return None
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image_rgb.shape
            bytes_per_line = 3 * w
            q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        elif image.ndim == 2:
            h, w = image.shape
            q_image = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            return None
        pixmap = QPixmap.fromImage(q_image)
        return pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _show_root_report(self, obj: ObjectImage) -> None:
        """Красивая форма отчёта по корню (как на твоём скриншоте)."""
        logger.info("=== Открытие отчёта по корню ===")

        if not obj.root_analysis or len(obj.root_analysis) == 0:
            logger.warning("Нет данных анализа корня")
            return

        result = obj.root_analysis[0]

        dialog = QDialog(self)
        dialog.setWindowTitle("Анализ корневой системы")
        dialog.resize(1100, 800)

        layout = QVBoxLayout(dialog)

        # Заголовок
        title = QLabel(
            "КОРЕНЬ НА ГРАНЕ СМЕРТИ" if result.viability == RootViability.CRITICAL else "Жизнеспособный корень")
        title.setStyleSheet(
            "font-size: 24pt; font-weight: bold; color: red;" if result.viability == RootViability.CRITICAL else "color: green;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Уверенность и источник
        info = QLabel(
            f"Уверенность модели: {result.score:.3f}<br>Источник оценки: Модель best_root_cls2.pt (по всему сеянцу)")
        info.setStyleSheet("font-size: 14pt;")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        # Морфология
        morph_label = QLabel("Морфологические метрики (справочно):")
        morph_label.setStyleSheet("font-size: 16pt; margin-top: 20px;")
        layout.addWidget(morph_label)

        morph_text = QLabel(
            f"• Длина корня: {result.morphology.length:.1f} px<br>"
            f"• Средняя толщина: {result.morphology.mean_thickness:.1f} px<br>"
            f"• Индекс ветвистости: {result.morphology.branching_index:.3f}<br>"
            f"• Плотность корня: {result.morphology.density:.3f}<br>"
            f"• Кривизна: {result.morphology.curvature:.3f}"
        )
        morph_text.setStyleSheet("font-size: 14pt;")
        layout.addWidget(morph_text)

        # Изображение сеянца
        if obj.image and obj.image[0] is not None:
            pix = self._array_to_qpixmap(obj.image[0], max_size=400)
            if pix and not pix.isNull():
                label = QLabel()
                label.setPixmap(pix)
                label.setAlignment(Qt.AlignCenter)
                layout.addWidget(label)

        # Визуализация корня
        try:
            vis_img = self.root_analyzer.visualize(result)
            vis_pix = self._array_to_qpixmap(vis_img, max_size=400)
            if vis_pix and not vis_pix.isNull():
                vis_label = QLabel()
                vis_label.setPixmap(vis_pix)
                vis_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(vis_label)
        except Exception as e:
            logger.error(f"Ошибка визуализации: {e}")

        # Кнопка закрытия
        close_btn = QPushButton("ЗАКРЫТЬ")
        close_btn.setStyleSheet("font-size: 14pt; padding: 10px;")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)

        dialog.exec_()
        logger.info("Отчёт закрыт")
    def create_report(self) -> None:
        """Создаёт PDF-отчёт по текущим результатам детекции."""
        self._update_action_states()
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

    def full_root_report(self) -> None:
        """Отчёт по всем сеянцам в одном диалоге."""
        if not self.image_storage.class_object_image:
            logger.warning("Нет данных для полного отчёта")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Полный отчёт по всем сеянцам")
        dialog.resize(1200, 800)

        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        total_bad = 0
        total = 0

        for img_idx, objects in enumerate(self.image_storage.class_object_image):
            for obj_idx, obj in enumerate(objects):
                total += 1
                if not obj.root_analysis:
                    continue

                result = obj.root_analysis[0]
                if result.viability == RootViability.CRITICAL:
                    total_bad += 1

                frame = QGroupBox(f"Сеянец {total} (страница {img_idx+1})")
                frame_layout = QVBoxLayout(frame)

                status = "НА ГРАНЕ СМЕРТИ" if result.viability == RootViability.CRITICAL else "Жизнеспособный"
                color = "red" if result.viability == RootViability.CRITICAL else "green"
                frame_layout.addWidget(QLabel(f"<h3 style='color:{color};'>{status}</h3>"))
                frame_layout.addWidget(QLabel(f"Уверенность: {result.score:.3f}"))

                # Изображение сеянца
                if obj.image and obj.image[0] is not None:
                    pix = self._array_to_qpixmap(obj.image[0], max_size=300)
                    if pix:
                        label = QLabel()
                        label.setPixmap(pix)
                        label.setAlignment(Qt.AlignCenter)
                        frame_layout.addWidget(label)

                # Визуализация корня
                try:
                    vis = self.root_analyzer.visualize(result)
                    vis_pix = self._array_to_qpixmap(vis, max_size=300)
                    if vis_pix:
                        vis_label = QLabel()
                        vis_label.setPixmap(vis_pix)
                        vis_label.setAlignment(Qt.AlignCenter)
                        frame_layout.addWidget(vis_label)
                except:
                    pass

                scroll_layout.addWidget(frame)

        summary = QLabel(f"<h2>Итог: {total_bad} из {total} сеянцев на гране смерти</h2>")
        summary.setAlignment(Qt.AlignCenter)
        layout.addWidget(summary)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec_()