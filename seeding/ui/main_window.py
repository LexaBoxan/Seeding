"""Главное окно приложения Seeding.

Реализует ImageEditor — окно с загрузкой изображений/PDF, детекцией сеянцев YOLOv8,
классификацией частей растения, деревом слоёв и генерацией PDF-отчётов.
"""

import logging
import os

import cv2
import fitz
import numpy as np
from PyQt5.QtCore import QPoint, QRectF, QSettings, QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPixmap, QTransform

from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QGroupBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QMainWindow,
    QMessageBox,
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
    QTextEdit,
    QShortcut,
)
from ultralytics import YOLO

import seeding.config as cfg
from seeding.config import (
    DETECTION_CLASS_NAME,
    MAIN_CONTENT_MARGINS,
    TOOLBAR_ICON_SIZE,
    DEFAULT_CLASSIFY_WEIGHTS_PATH,
    NMS_IOU_THRESHOLD,
    PANEL_INFO_MIN_WIDTH,
    PANEL_LAYERS_MAX_WIDTH,
    PANEL_LAYERS_MIN_WIDTH,
    PANEL_LAYOUT_MARGINS,
    PDF_RENDER_SCALE,
    QSETTINGS_APP,
    QSETTINGS_ORG,
    ROTATE_ANGLE_DEG,
    ROTATE_K,
    SPLITTER_SIZES,
    VIEW_BACKGROUND_B,
    VIEW_BACKGROUND_G,
    VIEW_BACKGROUND_R,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    WINDOW_X,
    WINDOW_Y,
    ZOOM_FACTOR_INCREMENT,
    ZOOM_FACTOR_INITIAL,
    CONF_DISPLAY_HIGH,
    CONF_DISPLAY_MEDIUM,
)
from seeding.models.data_models import AllClassImage, ObjectImage, OriginalImage
from seeding.utils import simple_nms, rotate_bbox

from .bbox_item import BBoxItem
from .settings_dialog import SettingsDialog
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


class ModelLoadWorker(QThread):
    """Загружает модель YOLO в фоновом потоке."""

    model_loaded = pyqtSignal(object)
    model_error = pyqtSignal(str)

    def __init__(self, weights_path: str):
        super().__init__()
        self.weights_path = weights_path

    def run(self) -> None:  # pragma: no cover - поток
        try:
            model = YOLO(self.weights_path)
            self.model_loaded.emit(model)
        except Exception as e:
            logger.exception("Ошибка загрузки модели")
            self.model_error.emit(str(e))


class DetectionWorker(QThread):
    """Worker для выполнения детекции одного изображения в отдельном потоке."""

    result_ready = pyqtSignal(int, object)

    def __init__(
        self,
        index: int,
        image: np.ndarray,
        model: YOLO | None = None,
        weights_path: str | None = None,
    ):
        super().__init__()
        self.index = index
        self.image = image
        self.model = model
        self.weights_path = weights_path

    def run(self) -> None:  # pragma: no cover - поток
        model = self.model
        if model is None and self.weights_path:
            model = YOLO(self.weights_path)
        if model is None:
            return
        results = model(self.image)
        self.result_ready.emit(self.index, results)


class FindAllWorker(QThread):
    """Worker для поиска сеянцев на всех изображениях в фоновом потоке."""

    result_ready = pyqtSignal(int, object)
    progress_updated = pyqtSignal(int, int)

    def __init__(self, images: list, model: YOLO):
        super().__init__()
        self.images = images
        self.model = model
        self._cancel = False

    def cancel(self) -> None:
        """Запросить отмену обработки."""
        self._cancel = True

    def run(self) -> None:  # pragma: no cover - поток
        total = len(self.images)
        for idx, image in enumerate(self.images):
            if self._cancel:
                return
            self.progress_updated.emit(idx, total)
            results = self.model(image)
            self.result_ready.emit(idx, results)
        self.progress_updated.emit(total, total)


class ImageEditor(QMainWindow):
    """
    Главное окно приложения для работы с изображениями и PDF.

    Позволяет загружать файлы, управлять слоями и искать сеянцы при помощи YOLOv8.
    """

    def __init__(self, weights_path: str):
        super().__init__()

        # Загрузка сохранённых настроек при старте
        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)

        # Обновляем глобальные переменные в config.py теми, что сохранил пользователь
        # Если настроек еще нет, останутся дефолтные (0.9 и 0.5)
        cfg.CONF_THRESHOLD_HIGH = float(settings.value("conf_high", cfg.CONF_THRESHOLD_HIGH))
        cfg.CONF_THRESHOLD_LOW = float(settings.value("conf_low", cfg.CONF_THRESHOLD_LOW))

        logger.info(f"Настройки загружены: High={cfg.CONF_THRESHOLD_HIGH}, Low={cfg.CONF_THRESHOLD_LOW}")


        self.setWindowTitle("Анализ сеянцев")
        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.zoom_factor = ZOOM_FACTOR_INITIAL
        self.image_storage = OriginalImage()
        self.weights_path = weights_path
        self.model = None
        self.classify_model = None
        self._find_all_worker = None

        self._active_image_index = 0

        self.init_ui()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

        self._start_model_loading()

    def init_ui(self):
        self.create_menu()
        self.create_top_toolbar()  # ← теперь сверху
        self.create_left_panel()  # ← новая левая панель
        self.create_central_widget()
        self.create_right_panel()

        # Разделитель: левая | центр | правая
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.scroll_area)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes(SPLITTER_SIZES)

        central_wrapper = QWidget()
        wrapper_layout = QVBoxLayout(central_wrapper)
        wrapper_layout.setContentsMargins(*MAIN_CONTENT_MARGINS)
        wrapper_layout.setSpacing(0)
        wrapper_layout.addWidget(self.splitter)
        self.setCentralWidget(central_wrapper)

        self._setup_shortcuts()

        self.action_find.setEnabled(False)
        self.action_find_all.setEnabled(False)

    def _start_model_loading(self) -> None:
        """Запускает загрузку модели детекции в фоновом потоке."""
        self.statusBar().showMessage("Загрузка модели детекции...")
        self._model_load_worker = ModelLoadWorker(self.weights_path)
        self._model_load_worker.model_loaded.connect(self._on_model_loaded)
        self._model_load_worker.model_error.connect(self._on_model_error)
        self._model_load_worker.finished.connect(self._on_model_load_finished)
        self._model_load_worker.start()

    def _on_model_loaded(self, model) -> None:
        """Обработка успешной загрузки модели."""
        self.model = model
        self.action_find.setEnabled(True)
        self.action_find_all.setEnabled(True)
        self.statusBar().showMessage("Модель загружена", 3000)
        logger.info("Модель детекции успешно загружена")

    def _on_model_error(self, error_msg: str) -> None:
        """Обработка ошибки загрузки модели."""
        QMessageBox.critical(
            self,
            "Ошибка загрузки модели",
            f"Не удалось загрузить модель:\n{self.weights_path}\n\n{error_msg}",
        )
        self.statusBar().showMessage("Ошибка загрузки модели", 5000)

    def _on_model_load_finished(self) -> None:
        """Скрывает индикатор загрузки после завершения worker."""
        if self.model is None:
            self.statusBar().showMessage("Модель не загружена")
        self._model_load_worker = None

    def create_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        open_action = QAction("Открыть файл", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

    # ====================== НОВАЯ ВЕРХНЯЯ ПАНЕЛЬ ======================
    def create_top_toolbar(self):
        """Создание панели инструментов."""
        self.toolbar = QToolBar("Инструменты", self)
        self.toolbar.setObjectName("mainToolbar")
        self.toolbar.setIconSize(QSize(TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        style = self.style()

        # --- АНАЛИЗ ---
        self.action_mask = QAction(style.standardIcon(QStyle.SP_FileDialogNewFolder), "", self)
        self.action_mask.setToolTip("Создать маску (в разработке)")
        self.action_mask.triggered.connect(self.create_mask)
        self.toolbar.addAction(self.action_mask)

        self.action_find = QAction(style.standardIcon(QStyle.SP_MediaPlay), "", self)
        self.action_find.setToolTip("Поиск сеянцев на текущем изображении (Ctrl+F)")
        self.action_find.triggered.connect(self.find_seedlings)
        self.toolbar.addAction(self.action_find)

        self.action_find_all = QAction(style.standardIcon(QStyle.SP_BrowserReload), "", self)
        self.action_find_all.setToolTip("Поиск сеянцев на всех изображениях (Ctrl+Shift+F)")
        self.action_find_all.triggered.connect(self.find_all_seedlings)
        self.toolbar.addAction(self.action_find_all)

        self.action_classify = QAction(style.standardIcon(QStyle.SP_FileDialogDetailedView), "", self)
        self.action_classify.setToolTip("Классификация частей растения (Ctrl+C)")
        self.action_classify.triggered.connect(self.classify)
        self.toolbar.addAction(self.action_classify)

        self.action_rotate = QAction(style.standardIcon(QStyle.SP_BrowserReload), "", self)
        self.action_rotate.setToolTip("Повернуть изображение (Ctrl+R)")
        self.action_rotate.triggered.connect(self.rotate_image)
        self.toolbar.addAction(self.action_rotate)

        self.toolbar.addSeparator()

        # --- СЕКЦИЯ: ОТЧЕТЫ ---
        self.action_report = QAction(style.standardIcon(QStyle.SP_FileDialogContentsView), "", self)
        self.action_report.setToolTip("Создать PDF-отчёт (Ctrl+P)")
        self.action_report.triggered.connect(self.create_report)
        self.toolbar.addAction(self.action_report)

        self.toolbar.addSeparator()

        # --- СЕКЦИЯ: НАВИГАЦИЯ ---
        self.action_zoom_in = QAction(style.standardIcon(QStyle.SP_ArrowUp), "", self)
        self.action_zoom_in.setToolTip("Приблизить (Ctrl++)")
        self.action_zoom_in.triggered.connect(self.zoom_in)
        self.toolbar.addAction(self.action_zoom_in)

        self.action_zoom_out = QAction(style.standardIcon(QStyle.SP_ArrowDown), "", self)
        self.action_zoom_out.setToolTip("Отдалить (Ctrl+-)")
        self.action_zoom_out.triggered.connect(self.zoom_out)
        self.toolbar.addAction(self.action_zoom_out)

        self.action_fit = QAction(style.standardIcon(QStyle.SP_DesktopIcon), "", self)
        self.action_fit.setToolTip("Вписать в окно (Ctrl+0)")
        self.action_fit.triggered.connect(self.fit_to_window)
        self.toolbar.addAction(self.action_fit)

        self.toolbar.addSeparator()

        # --- СЕКЦИЯ: ФАЙЛЫ ---
        self.action_open = QAction(style.standardIcon(QStyle.SP_DialogOpenButton), "", self)
        self.action_open.setToolTip("Открыть изображения или PDF (Ctrl+O)")
        self.action_open.triggered.connect(self.open_image)
        self.toolbar.addAction(self.action_open)

        self.action_add = QAction(style.standardIcon(QStyle.SP_FileIcon), "", self)
        self.action_add.setToolTip("Добавить файлы к проекту (Ctrl+Shift+O)")
        self.action_add.triggered.connect(self.add_files)
        self.toolbar.addAction(self.action_add)

        self.toolbar.addSeparator()

        # --- СЕКЦИЯ: СИСТЕМА ---
        self.action_save = QAction(style.standardIcon(QStyle.SP_DialogSaveButton), "", self)
        self.action_save.setToolTip("Сохранить изменения (Ctrl+S)")
        self.action_save.triggered.connect(self.save_changes)
        self.toolbar.addAction(self.action_save)

        self.action_settings = QAction(style.standardIcon(QStyle.SP_ComputerIcon), "", self)
        self.action_settings.setToolTip("Параметры порогов уверенности")
        self.action_settings.triggered.connect(self.open_settings)
        self.toolbar.addAction(self.action_settings)


    # ====================== ЛЕВАЯ ПАНЕЛЬ ======================
    def create_left_panel(self):
        self.left_panel = QGroupBox("Информация")
        self.left_panel.setMinimumWidth(PANEL_INFO_MIN_WIDTH)
        self.left_panel.setObjectName("infoGroup")

        layout = QVBoxLayout()
        layout.setContentsMargins(*PANEL_LAYOUT_MARGINS)
        self.info_text = QTextEdit()
        self.info_text.setObjectName("infoPanel")
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)

        self.left_panel.setLayout(layout)

    def update_left_info(self, item_data: dict = None):
        """Обновляет левую панель в зависимости от выбранного элемента."""
        if not item_data or "type" not in item_data:
            self.info_text.setHtml(
                '<p style="color:#9ca3af; margin:0;">Выберите элемент в дереве</p>'
            )
            return

        t = item_data["type"]

        if t in ("original", "pdf"):
            idx = item_data.get("index", 0)
            self._show_page_stats(idx)
        elif t == "seeding":
            self._show_seeding_info(
                item_data["parent_index"],
                item_data["index"]
            )
        elif t == "class":
            # для класса показываем информацию о сеянце
            self._show_seeding_info(
                item_data["parent_index"],
                item_data["seeding_index"]
            )
    def switch_image(self, direction: int):
        """Переключает на предыдущее/следующее изображение стрелками."""
        if not self.image_storage.images:
            return

        new_idx = self._active_image_index + direction
        new_idx = max(0, min(new_idx, len(self.image_storage.images) - 1))

        if new_idx == self._active_image_index:
            return

        self._active_image_index = new_idx
        self.display_image_with_boxes(new_idx)

        # Выделяем соответствующий элемент в дереве
        item = self.tree_widget.topLevelItem(new_idx)
        if item:
            self.tree_widget.setCurrentItem(item)

        self.update_left_info({"type": "pdf", "index": new_idx})

    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            dialog.save_settings()
            # После изменения настроек нужно перерисовать текущие боксы,
            # чтобы они обновили цвета под новые пороги
            self.display_image_with_boxes(self._active_image_index)
            logger.info("Настройки обновлены и применены.")

    def _setup_shortcuts(self):
        """Настраивает все горячие клавиши."""
        # Стрелки — переключение страниц
        QShortcut(QKeySequence(Qt.Key_Left),  self, lambda: self.switch_image(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.switch_image(1))

        # Основные действия
        QShortcut(QKeySequence("Ctrl+O"),       self, self.open_image)      # Открыть файлы
        QShortcut(QKeySequence("Ctrl+Shift+O"), self, self.add_files)       # Добавить файлы
        QShortcut(QKeySequence("Ctrl+F"),       self, self.find_seedlings)  # Найти сеянцы (текущая)
        QShortcut(QKeySequence("Ctrl+Shift+F"), self, self.find_all_seedlings)  # Найти все
        QShortcut(QKeySequence("Ctrl+C"),       self, self.classify)        # Классификация
        QShortcut(QKeySequence("Ctrl+R"),       self, self.rotate_image)    # Повернуть 90°
        QShortcut(QKeySequence("Ctrl+P"),       self, self.create_report)   # PDF-отчёт
        QShortcut(QKeySequence("Ctrl+S"),       self, self.save_changes)    # Сохранить

        # Зум
        QShortcut(QKeySequence("Ctrl++"),       self, self.zoom_in)
        QShortcut(QKeySequence("Ctrl+="),       self, self.zoom_in)   # для клавиш без NumPad
        QShortcut(QKeySequence("Ctrl+-"),       self, self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"),       self, self.fit_to_window)

    def create_central_widget(self):
        """Создаёт центральную область — изображение с возможностью зума и рамок."""
        self.scroll_area = DraggableScrollArea()
        self.scroll_area.setObjectName("centralScroll")
        self.scroll_area.setWidgetResizable(True)

        # Графическая сцена для отображения изображения и интерактивных боксов
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_scene.setBackgroundBrush(
            QColor(VIEW_BACKGROUND_R, VIEW_BACKGROUND_G, VIEW_BACKGROUND_B)
        )
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setObjectName("centralView")
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)  # перетаскивание рукой

        self.image_item = QGraphicsPixmapItem()
        self.graphics_scene.addItem(self.image_item)

        self.rect_items = {}                     # для хранения BBoxItem

        self.scroll_area.setWidget(self.graphics_view)

    def create_right_panel(self):
        """Создаёт правую панель с деревом слоёв."""
        self.right_panel = QGroupBox("Слои")
        self.right_panel.setObjectName("layersGroup")
        self.right_panel.setMinimumWidth(PANEL_LAYERS_MIN_WIDTH)
        self.right_panel.setMaximumWidth(PANEL_LAYERS_MAX_WIDTH)

        layout = QVBoxLayout()
        layout.setContentsMargins(*PANEL_LAYOUT_MARGINS)

        # Дерево
        self.tree_widget = LayerTreeWidget()
        self.tree_widget.setObjectName("layerTree")
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)

        # Скролл для дерева (если страниц много)
        tree_scroll = QScrollArea()
        tree_scroll.setWidgetResizable(True)
        tree_scroll.setWidget(self.tree_widget)

        layout.addWidget(tree_scroll)
        self.right_panel.setLayout(layout)



    def _has_classified_parts(self) -> bool:
        if not self.image_storage.class_object_image:
            return False
        for objects in self.image_storage.class_object_image:
            for obj in objects:
                if obj.image_all_class:
                    return True
        return False





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
                self._active_image_index = parent_idx
                self.display_seeding_with_boxes(parent_idx, seed_idx)
            elif item_data["type"] == "class":
                parent_idx = item_data["parent_index"]
                seed_idx = item_data["seeding_index"]
                class_idx = item_data["class_index"]
                self._active_image_index = parent_idx
                self.display_class_image(parent_idx, seed_idx, class_idx)

            self.update_left_info(item_data)   # ← обновляем левую панель

    def open_image(self) -> None:
        """Открывает один или несколько файлов (изображения + PDF)."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Открыть изображения или PDF",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;PDF Files (*.pdf);;All Files (*)",
        )
        if not files:
            return

        self.image_storage = OriginalImage()
        self.tree_widget.clear()
        self.image_storage.file_path = files[0]  # для отчёта берём первый файл

        for file_path in files:
            if file_path.lower().endswith(".pdf"):
                self._add_pdf(file_path)
            else:
                self._add_image(file_path)

        self._finalize_after_load()


    def add_files(self) -> None:
        """Добавляет новые файлы к уже открытому проекту."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Добавить изображения или PDF",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;PDF Files (*.pdf);;All Files (*)",
        )
        if not files:
            return

        for file_path in files:
            if file_path.lower().endswith(".pdf"):
                self._add_pdf(file_path)
            else:
                self._add_image(file_path)

        self._finalize_after_load()

    def _add_image(self, file_path: str):
        """Добавляет одно изображение."""
        image = self.load_image(file_path)
        if image is None:
            QMessageBox.warning(
                self,
                "Ошибка загрузки",
                f"Не удалось загрузить изображение:\n{file_path}",
            )
            return

        idx = len(self.image_storage.images)
        self.image_storage.images.append(image)

        if self.image_storage.class_object_image is None:
            self.image_storage.class_object_image = []
        self.image_storage.class_object_image.append([])

        name = os.path.basename(file_path)
        self.tree_widget.add_root_item(
            name, "Изображение", idx, "original", image
        )

    def _add_pdf(self, pdf_path: str):
        """Добавляет все страницы PDF."""
        try:
            doc = fitz.open(pdf_path)
            base_idx = len(self.image_storage.images)

            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, doc.page_count)

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                mat = fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:
                    img = img[:, :, :3].copy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                self.image_storage.images.append(img)

                if self.image_storage.class_object_image is None:
                    self.image_storage.class_object_image = []
                self.image_storage.class_object_image.append([])

                name = f"{os.path.basename(pdf_path)} — стр. {page_num + 1}"
                self.tree_widget.add_root_item(
                    name, "Страница PDF", base_idx + page_num, "pdf", img
                )

                self.progress_bar.setValue(page_num + 1)

            doc.close()
            self.progress_bar.setVisible(False)

        except Exception as e:
            logger.error("Ошибка при загрузке PDF %s: %s", pdf_path, e)
            QMessageBox.critical(
                self,
                "Ошибка загрузки PDF",
                f"Не удалось загрузить PDF:\n{pdf_path}\n\n{e}",
            )

    def _finalize_after_load(self):
        """Общие действия после открытия/добавления файлов."""
        if not self.image_storage.class_object_image:
            self.image_storage.class_object_image = [[] for _ in self.image_storage.images]



        if self.image_storage.images:
            self._active_image_index = 0
            self.display_image_with_boxes(0)
            self.update_left_info({"type": "original", "index": 0})

    def _show_page_stats(self, idx: int):
        if (not self.image_storage.class_object_image or
                idx >= len(self.image_storage.class_object_image)):
            self.info_text.setHtml(
                '<p style="font-size:14px; margin:0;">'
                f'<b>Страница {idx + 1}</b></p>'
                '<p style="color:#9ca3af; margin-top:12px;">Нет сеянцев</p>'
            )
            return

        objs: list[ObjectImage] = self.image_storage.class_object_image[idx]
        total = len(objs)

        high = sum(1 for o in objs if o.confidence >= cfg.CONF_THRESHOLD_HIGH)
        medium = sum(1 for o in objs if cfg.CONF_THRESHOLD_LOW <= o.confidence < cfg.CONF_THRESHOLD_HIGH)
        critical = total - high - medium

        html = f"""
        <p style="font-size:15px; font-weight:600; margin:0 0 16px 0;">Страница {idx + 1}</p>
        <p style="margin:8px 0;"><b>Всего сеянцев:</b> <span style="color:#22c55e;">{total}</span></p>
        <p style="margin:6px 0;"><span style="color:#4ade80;">●</span> Хорошо (≥{cfg.CONF_THRESHOLD_HIGH}): <b>{high}</b></p>
        <p style="margin:6px 0;"><span style="color:#fbbf24;">●</span> Средне ({cfg.CONF_THRESHOLD_LOW}–{cfg.CONF_THRESHOLD_HIGH}): <b>{medium}</b></p>
        <p style="margin:6px 0;"><span style="color:#f87171;">●</span> Критично (&lt;{cfg.CONF_THRESHOLD_LOW}): <b>{critical}</b></p>
        """
        self.info_text.setHtml(html)

    def _show_seeding_info(self, parent_idx: int, seed_idx: int):
        obj = self.image_storage.class_object_image[parent_idx][seed_idx]

        html = f"""
        <p style="font-size:15px; font-weight:600; margin:0 0 12px 0;">Сеянец {seed_idx + 1}</p>
        <p style="margin:6px 0;"><b>Уверенность:</b> <span style="color:#22c55e;">{obj.confidence:.3f}</span></p>
        <p style="margin:6px 0; font-family:monospace;"><b>BBox:</b> {obj.bbox}</p>
        <p style="margin:6px 0;"><b>Поворот:</b> {obj.rotation_k * ROTATE_ANGLE_DEG}°</p>
        """

        if obj.image_all_class:
            html += '<p style="margin-top:12px;"><b>Классификация:</b></p>'
            for cls in obj.image_all_class:
                color = (
                    "#4ade80" if cls.confidence >= CONF_DISPLAY_HIGH
                    else "#fbbf24" if cls.confidence >= CONF_DISPLAY_MEDIUM
                    else "#f87171"
                )
                html += f'<p style="margin:4px 0;"><span style="color:{color}">●</span> {cls.class_name}: {cls.confidence:.3f}</p>'

        self.info_text.setHtml(html)


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
                mat = fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)  # 2x масштаб
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
                self.progress_bar.setValue(page_num + 1)
            doc.close()

            # Инициализация class_object_image для всех страниц
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
        self.zoom_factor *= ZOOM_FACTOR_INCREMENT
        self.update_image_zoom()

    def zoom_out(self) -> None:
        """Уменьшает изображение."""
        self.zoom_factor /= ZOOM_FACTOR_INCREMENT
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

    def _on_detection_start(self) -> None:
        """Показывает прогресс-бар при запуске worker."""
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

    def _on_detection_finished(self) -> None:
        """Скрывает прогресс-бар после завершения worker."""
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def _on_detection_result(self, index: int, results) -> None:
        """Обрабатывает результаты детекции и обновляет дерево."""
        image = self.image_storage.images[index]
        try:
            boxes: list[list[int]] = []
            scores: list[float] = []
            class_boxes_data: list[dict] = []
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = results[0].names[class_id]
                if str(class_name).lower() != DETECTION_CLASS_NAME:
                    continue

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
            indices = simple_nms(boxes, scores, iou_threshold=NMS_IOU_THRESHOLD)
            logger.info(
                "find_seedlings: после NMS осталось %s боксов", len(indices)
            )

            self.image_storage.class_object_image[index] = []
            parent_item = self.tree_widget.topLevelItem(index)
            if parent_item is not None:
                for i in reversed(range(parent_item.childCount())):
                    parent_item.takeChild(i)
            for i_out, i in enumerate(indices):
                data = class_boxes_data[i]
                x1, y1, x2, y2 = data["coords"]
                crop = image[y1:y2, x1:x2].copy()
                rotation_k = 0
                if crop.shape[1] > crop.shape[0]:
                    crop = np.rot90(crop, k=ROTATE_K)
                    rotation_k = ROTATE_K
                obj = ObjectImage(
                    class_name=data["class_name"],
                    confidence=data["score"],
                    image=[crop],
                    bbox=(x1, y1, x2, y2),
                    rotation_k=rotation_k,
                )
                self.image_storage.class_object_image[index].append(obj)
                self.tree_widget.add_child_item(
                    parent_item,
                    f"Seeding{i_out + 1}",
                    f"Уверенность: {data['score']:.2f}",
                    index,
                    i_out,
                    "seeding",
                    crop,
                )
            self.display_image_with_boxes(index)
            self.update_left_info({"type": "pdf", "index": index})  # ← добавь эту строку
            logger.info("find_seedlings: завершено")
        except Exception as e:  # pragma: no cover - логирование
            logger.error("Ошибка во время NMS или обработки результатов: %s", e)

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

        if self.model is None:
            QMessageBox.warning(
                self,
                "Модель не загружена",
                "Модель детекции не загружена. Проверьте путь к весам.",
            )
            return
        if not self.image_storage.images:
            logger.warning("find_seedlings: Нет изображений для обработки")
            return

        image = self.image_storage.images[current_index]
        if image is None:
            logger.warning("find_seedlings: Текущее изображение пустое")
            return

        self.worker = DetectionWorker(
            index=current_index,
            image=image,
            model=self.model,
            weights_path=self.weights_path,
        )
        self.worker.started.connect(self._on_detection_start)
        self.worker.result_ready.connect(self._on_detection_result)
        self.worker.finished.connect(self._on_detection_finished)
        self.worker.start()

    def find_all_seedlings(self) -> None:
        """Запускает поиск сеянцев на всех изображениях в фоновом потоке.

        Детекция выполняется в FindAllWorker; UI остаётся отзывчивым,
        прогресс отображается в статус-баре.
        """
        if self.model is None:
            QMessageBox.warning(
                self,
                "Модель не загружена",
                "Модель детекции не загружена. Дождитесь окончания загрузки.",
            )
            return
        if not self.image_storage.images:
            logger.warning("find_all_seedlings: Нет изображений")
            return

        if self._find_all_worker is not None and self._find_all_worker.isRunning():
            logger.warning("find_all_seedlings: уже выполняется")
            return

        if self.image_storage.class_object_image is None:
            self.image_storage.class_object_image = [
                [] for _ in range(len(self.image_storage.images))
            ]

        total = len(self.image_storage.images)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.action_find.setEnabled(False)
        self.action_find_all.setEnabled(False)

        self._find_all_worker = FindAllWorker(
            self.image_storage.images, self.model
        )
        self._find_all_worker.result_ready.connect(self._on_detection_result)
        self._find_all_worker.progress_updated.connect(self._on_find_all_progress)
        self._find_all_worker.finished.connect(self._on_find_all_finished)
        self._find_all_worker.start()

    def _on_find_all_progress(self, current: int, total: int) -> None:
        """Обновляет прогресс-бар при «Найти все»."""
        self.progress_bar.setValue(current)
        self._active_image_index = min(current, total - 1) if total > 0 else 0
        if current < total:
            self.update_left_info({"type": "pdf", "index": self._active_image_index})

    def _on_find_all_finished(self) -> None:
        """Завершение «Найти все»: скрыть прогресс, включить кнопки."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.action_find.setEnabled(True)
        self.action_find_all.setEnabled(True)
        self._find_all_worker = None
        logger.info("find_all_seedlings: завершено")

    def display_image_with_boxes(self, img_idx: int, seeding_idx: int = None):
        """
        Универсальный метод отрисовки.
        Если seeding_idx is None — рисуем оригинал и рамки сеянцев.
        Если seeding_idx задан — рисуем кроп сеянца и рамки его частей.
        """
        if img_idx >= len(self.image_storage.images):
            return

        # 1. Выбираем базовое изображение для отрисовки
        if seeding_idx is None:
            base_img = self.image_storage.images[img_idx]
            objects_to_draw = self.image_storage.class_object_image[img_idx]
            is_crop_view = False
        else:
            obj = self.image_storage.class_object_image[img_idx][seeding_idx]
            base_img = obj.image[0]
            objects_to_draw = obj.image_all_class or []
            is_crop_view = True

        self.display_image(base_img)

        # 2. Отрисовываем рамки (BBoxItem)
        for i, obj in enumerate(objects_to_draw):
            if obj.bbox:
                x1, y1, x2, y2 = obj.bbox
                rect = QRectF(x1, y1, x2 - x1, y2 - y1)

                # Создаем универсальный BBoxItem
                # Передаем сам объект 'obj', чтобы BBoxItem мог менять его координаты
                item = BBoxItem(rect, obj)
                item.setEditable(True)
                self.graphics_scene.addItem(item)
                self.rect_items[i] = item



    def display_seeding_with_boxes(self, parent_idx: int, seed_idx: int):
        """Перегрузка для отображения конкретного сеянца."""
        self.display_image_with_boxes(parent_idx, seeding_idx=seed_idx)

    def display_class_image(self, parent_idx, seed_idx, class_idx):
        """Отображение конкретной части (зум на часть на кропе сеянца)."""
        # Просто вызываем отрисовку кропа сеянца, рамки частей подгрузятся сами
        self.display_image_with_boxes(parent_idx, seeding_idx=seed_idx)

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
        """Классификация частей растения (цветок, корень, стебель)."""
        if not self.image_storage.class_object_image:
            logger.warning("classify: Сначала найдите сеянцы")
            return

        if self.classify_model is None:
            try:
                self.classify_model = YOLO(DEFAULT_CLASSIFY_WEIGHTS_PATH)
            except Exception as e:
                logger.exception("Не удалось загрузить модель классификации: %s", e)
                QMessageBox.critical(
                    self,
                    "Ошибка загрузки модели",
                    f"Не удалось загрузить модель классификации:\n{DEFAULT_CLASSIFY_WEIGHTS_PATH}\n\n{e}",
                )
                return

        logger.info("classify: старт вторичной классификации")

        # Проходим по всем изображениям и всем найденным сеянцам
        for img_idx, objects in enumerate(self.image_storage.class_object_image):
            for obj_idx, seeding_obj in enumerate(objects):
                if not seeding_obj.image:
                    continue

                # Запуск модели на кропе сеянца
                results = self.classify_model(seeding_obj.image[0])

                # Важно: инициализируем список, если он пуст, чтобы избежать вылета
                if seeding_obj.image_all_class is None:
                    seeding_obj.image_all_class = []
                else:
                    seeding_obj.image_all_class.clear()

                # Ищем соответствующий элемент в дереве для добавления детей
                parent_item = self.tree_widget.topLevelItem(img_idx).child(obj_idx)

                for result in results:
                    for box in result.boxes:
                        conf = float(box.conf)
                        cls_id = int(box.cls)
                        class_name = result.names[cls_id]

                        # Координаты части относительно кропа
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        local_bbox = (coords[0], coords[1], coords[2], coords[3])

                        # Вырезаем изображение части
                        part_img = seeding_obj.image[0][coords[1]:coords[3], coords[0]:coords[2]]

                        # Создаем объект части
                        new_part = AllClassImage(
                            class_name=class_name,
                            confidence=conf,
                            image=part_img,
                            bbox=local_bbox
                        )
                        seeding_obj.image_all_class.append(new_part)

                        # Добавляем в дерево
                        if parent_item:
                            self.tree_widget.add_class_item(
                                parent_item,
                                class_name,
                                f"Уверенность: {conf:.2f}",
                                img_idx,
                                obj_idx,
                                len(seeding_obj.image_all_class) - 1
                            )

        # Обновляем текущий вид
        self.display_image_with_boxes(self._active_image_index)
        logger.info("classify: успешно завершено")



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
            QMessageBox.information(
                self,
                "Отчёт создан",
                f"Отчёт сохранён:\n{output_path}",
            )
        except Exception as e:
            logger.exception("Ошибка при создании отчёта: %s", e)
            QMessageBox.critical(
                self,
                "Ошибка создания отчёта",
                f"Не удалось создать PDF-отчёт:\n{e}",
            )
