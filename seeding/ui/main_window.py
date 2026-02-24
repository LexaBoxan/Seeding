"""Главное окно приложения Seeding.

Реализует ImageEditor — окно с загрузкой изображений/PDF, детекцией сеянцев YOLOv8,
классификацией частей растения, деревом слоёв и генерацией PDF-отчётов.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import cv2
import fitz
import numpy as np
from PyQt5.QtCore import (
    QByteArray,
    QEvent,
    QPoint,
    QPointF,
    QRectF,
    QSettings,
    QSize,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)

from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGroupBox,
    QHBoxLayout,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QDoubleSpinBox,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QButtonGroup,
    QScrollArea,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QDialog,
    QTextEdit,
    QShortcut,
)
from ultralytics import YOLO

import seeding.config as cfg
from seeding.controllers import AppController
from seeding.config import (
    CALIBRATION_PIXELS_PER_MM_DEFAULT,
    CONF_THRESHOLD_HIGH_DEFAULT,
    CONF_THRESHOLD_LOW_DEFAULT,
    DETECTION_CLASS_NAME,
    DETECTION_CONFIDENCE_THRESHOLD,
    DETECTION_IOU_THRESHOLD,
    DEFAULT_CLASSIFY_WEIGHTS_PATH,
    PANEL_LAYERS_MAX_WIDTH,
    PANEL_LAYERS_MIN_WIDTH,
    PANEL_LAYOUT_MARGINS,
    PDF_RENDER_SCALE,
    QSETTINGS_APP,
    QSETTINGS_ORG,
    ROTATE_ANGLE_DEG,
    ROTATE_K,
    USE_CACHE_DEFAULT,
    VIEW_BACKGROUND_B,
    VIEW_BACKGROUND_G,
    VIEW_BACKGROUND_R,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    WINDOW_X,
    WINDOW_Y,
    ZOOM_FACTOR_INCREMENT,
    ZOOM_FACTOR_INITIAL,
)
from seeding.models import (
    AllClassImage,
    AppState,
    MeasurementRecord,
    ObjectImage,
    OriginalImage,
)
from seeding.services import ExportService, ImageService, ReportService
from seeding.storage import StorageService
from seeding.utils import clip_bbox_to_image, resolve_weights_path, rotate_bbox

from .bbox_item import BBoxItem
from .export_dialog import ExportDialog
from .icon_manager import IconManager
from .i18n import tr
from .layout_state import normalize_qbytearray
from .metrics import UiMetrics
from .preferences import load_ui_preferences
from .settings_dialog import SettingsDialog
from .statistics_panel import StatisticsPanel
from .theme_manager import apply_theme
from .thumbnails_panel import ThumbnailsPanel
from .tree_widget import LayerTreeWidget

logger = logging.getLogger(__name__)

INPUT_FILE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;"
    "PDF Files (*.pdf);;"
    "All Files (*)"
)

TOOLBAR_ACTION_SPECS = (
    (
        "action_mask",
        "action_mask.svg",
        "Маска",
        "Создать маску объектов на текущем изображении",
        "create_mask",
        None,
        QStyle.SP_FileDialogNewFolder,
    ),
    (
        "action_find",
        "action_detect.svg",
        "Найти на странице",
        "Поиск сеянцев на текущем изображении (Ctrl+F, D)",
        "find_seedlings",
        "Ctrl+F",
        QStyle.SP_MediaPlay,
    ),
    (
        "action_find_all",
        "action_detect_all.svg",
        "Найти на всех",
        "Поиск сеянцев на всех изображениях (Ctrl+Shift+F)",
        "find_all_seedlings",
        "Ctrl+Shift+F",
        QStyle.SP_BrowserReload,
    ),
    (
        "action_classify",
        "action_classify.svg",
        "Классифицировать",
        "Классификация частей растения (Ctrl+C)",
        "classify",
        "Ctrl+C",
        QStyle.SP_FileDialogDetailedView,
    ),
    (
        "action_rotate",
        "action_rotate.svg",
        "Повернуть",
        "Повернуть изображение (Ctrl+R, R)",
        "rotate_image",
        "Ctrl+R",
        QStyle.SP_BrowserReload,
    ),
    (
        "action_report",
        "action_report.svg",
        "Создать отчёт",
        "Создать PDF-отчёт (Ctrl+P)",
        "create_report",
        "Ctrl+P",
        QStyle.SP_FileDialogContentsView,
    ),
    (
        "action_zoom_in",
        "action_zoom_in.svg",
        "Приблизить",
        "Приблизить (Ctrl++)",
        "zoom_in",
        "Ctrl++",
        QStyle.SP_ArrowUp,
    ),
    (
        "action_zoom_out",
        "action_zoom_out.svg",
        "Отдалить",
        "Отдалить (Ctrl+-)",
        "zoom_out",
        "Ctrl+-",
        QStyle.SP_ArrowDown,
    ),
    (
        "action_fit",
        "action_fit.svg",
        "Вписать в окно",
        "Вписать в окно (Ctrl+0)",
        "fit_to_window",
        "Ctrl+0",
        QStyle.SP_DesktopIcon,
    ),
    (
        "action_open",
        "action_open.svg",
        "Открыть",
        "Открыть изображения или PDF (Ctrl+O)",
        "open_image",
        "Ctrl+O",
        QStyle.SP_DialogOpenButton,
    ),
    (
        "action_add",
        "action_add.svg",
        "Добавить файлы",
        "Добавить файлы к проекту (Ctrl+Shift+O)",
        "add_files",
        "Ctrl+Shift+O",
        QStyle.SP_FileIcon,
    ),
    (
        "action_export",
        "action_export.svg",
        "Экспорт",
        "Экспорт результатов (Ctrl+E)",
        "export_results",
        "Ctrl+E",
        QStyle.SP_DialogSaveButton,
    ),
    (
        "action_save",
        "action_save.svg",
        "Сохранить",
        "Сохранить изменения (Ctrl+S)",
        "save_changes",
        "Ctrl+S",
        QStyle.SP_DialogSaveButton,
    ),
    (
        "action_settings",
        "action_settings.svg",
        "Настройки",
        "Параметры приложения",
        "open_settings",
        None,
        QStyle.SP_ComputerIcon,
    ),
)


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
        """Сохраняет путь к весам модели для фоновой загрузки."""
        super().__init__()
        self.weights_path = weights_path

    def run(self) -> None:  # pragma: no cover - поток
        """Загружает YOLO-модель и отправляет результат через сигналы Qt."""
        try:
            model = YOLO(self.weights_path)
            self.model_loaded.emit(model)
        except Exception as e:
            logger.exception("Ошибка загрузки модели")
            self.model_error.emit(str(e))


class DetectionWorker(QThread):
    """Поток для детекции одного изображения."""

    result_ready = pyqtSignal(int, object)

    def __init__(
        self,
        index: int,
        image: np.ndarray,
        model: YOLO | None = None,
        weights_path: str | None = None,
        conf_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
    ):
        """нициализирует фоновую задачу детекции одной страницы.

        Можно передать уже загруженную модель или только путь к весам.
        """
        super().__init__()
        self.index = index
        self.image = image
        self.model = model
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold

    def run(self) -> None:  # pragma: no cover - поток
        """Выполняет детекцию и отправляет предсказания в основной поток."""
        model = self.model
        if model is None and self.weights_path:
            model = YOLO(self.weights_path)
        if model is None:
            return
        results = model(self.image, conf=self.conf_threshold)
        self.result_ready.emit(self.index, results)


class FindAllWorker(QThread):
    """Поток для поиска сеянцев на всех изображениях."""

    result_ready = pyqtSignal(int, object)
    progress_updated = pyqtSignal(int, int)

    def __init__(
        self,
        images: list,
        model: YOLO,
        conf_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
        *,
        indices: list[int] | None = None,
        progress_start: int = 0,
        progress_total: int | None = None,
    ):
        """нициализирует пакетную детекцию для набора изображений.

        Параметры ``progress_start/progress_total`` используются для
        согласованного отображения прогресса при частичной догрузке.
        """
        super().__init__()
        self.images = images
        self.model = model
        self._cancel = False
        self.conf_threshold = conf_threshold
        self.indices = indices or list(range(len(images)))
        self.progress_start = max(0, int(progress_start))
        if progress_total is None:
            progress_total = self.progress_start + len(images)
        self.progress_total = max(0, int(progress_total))

    def cancel(self) -> None:
        """Запросить отмену обработки."""
        self._cancel = True

    def run(self) -> None:  # pragma: no cover - поток
        """Последовательно обрабатывает все изображения до завершения/отмены."""
        processed = self.progress_start
        total = self.progress_total
        self.progress_updated.emit(processed, total)

        for local_idx, image in enumerate(self.images):
            if self._cancel:
                return
            page_index = (
                self.indices[local_idx]
                if local_idx < len(self.indices)
                else local_idx
            )
            results = self.model(image, conf=self.conf_threshold)
            self.result_ready.emit(page_index, results)
            processed += 1
            self.progress_updated.emit(processed, total)


class ImageEditor(QMainWindow):
    """
    Главное окно приложения для работы с изображениями и PDF.

    Позволяет загружать файлы, управлять слоями и искать сеянцы при помощи YOLOv8.
    """

    def __init__(self, weights_path: str):
        """Создаёт главное окно, сервисы и стартовое состояние приложения.

        Параметры:
            weights_path: путь к модели детекции, переданный из точки входа.
        """
        super().__init__()

        self.weights_path = weights_path
        self.classify_weights_path = str(DEFAULT_CLASSIFY_WEIGHTS_PATH)
        self.detection_confidence_threshold = DETECTION_CONFIDENCE_THRESHOLD
        self.detection_iou_threshold = DETECTION_IOU_THRESHOLD
        self.pixels_per_mm = CALIBRATION_PIXELS_PER_MM_DEFAULT
        self.use_cache = USE_CACHE_DEFAULT

        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        self._load_runtime_settings(settings)

        self.ui_preferences = load_ui_preferences()
        self.current_language = self.ui_preferences.language
        self.current_theme = self.ui_preferences.theme
        self.setWindowTitle(tr(self.current_language, "window_title", "Анализ сеянцев"))
        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.zoom_factor = ZOOM_FACTOR_INITIAL
        self.image_storage = OriginalImage()
        self.model = None
        self.classify_model = None
        self._find_all_worker = None

        self._active_image_index = 0
        self.app_state = AppState(
            image_storage=self.image_storage,
            active_image_index=self._active_image_index,
            zoom_factor=self.zoom_factor,
            report_dir=settings.value("report_dir", "", type=str),
            pixels_per_mm=self.pixels_per_mm,
            use_cache=self.use_cache,
        )
        self.image_service = ImageService()
        self.report_service = ReportService()
        self.export_service = ExportService()
        self.storage_service = StorageService()
        self.app_controller = AppController(
            image_service=self.image_service,
            report_service=self.report_service,
        )
        self.ui_metrics = UiMetrics()
        self.icon_manager = IconManager(self)
        self._active_tool = "select"
        self._space_hand_active = False
        self._tool_before_space = "select"
        self._measure_start_scene_pos: QPointF | None = None
        self._measure_line_item: QGraphicsLineItem | None = None
        self._measure_text_item: QGraphicsTextItem | None = None
        self._calibration_pending = False
        self._show_boxes = True
        self._box_class_visibility: dict[str, bool] = {
            "seeding": True,
            "inflorescence": True,
            "stem": True,
            "root": True,
            "other": True,
        }
        self._activity_log: list[str] = []
        self._auto_theme_fallback = "dark"
        self._last_detection_count = 0
        self._pending_classify_after_find_all = False

        self.setup_ui()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self._find_all_progress_dialog: QProgressDialog | None = None
        self._find_all_cancelled = False

        self._start_model_loading()

    def _tr(self, key: str, fallback: str) -> str:
        """Возвращает локализованную строку для текущего языка интерфейса."""
        return tr(self.current_language, key, fallback)

    def _load_runtime_settings(self, settings: QSettings) -> None:
        """Читает настраиваемые пороги из ``QSettings``."""
        cfg.CONF_THRESHOLD_HIGH = self._read_float_setting(
            settings,
            "conf_high",
            CONF_THRESHOLD_HIGH_DEFAULT,
        )
        cfg.CONF_THRESHOLD_LOW = self._read_float_setting(
            settings,
            "conf_low",
            CONF_THRESHOLD_LOW_DEFAULT,
        )
        self.detection_confidence_threshold = min(
            max(
                self._read_float_setting(
                    settings,
                    "detect_conf",
                    DETECTION_CONFIDENCE_THRESHOLD,
                ),
                0.0,
            ),
            1.0,
        )
        self.detection_iou_threshold = min(
            max(
                self._read_float_setting(
                    settings,
                    "detect_iou",
                    DETECTION_IOU_THRESHOLD,
                ),
                0.0,
            ),
            1.0,
        )
        self.pixels_per_mm = max(
            self._read_float_setting(
                settings,
                "pixels_per_mm",
                CALIBRATION_PIXELS_PER_MM_DEFAULT,
            ),
            0.0,
        )
        self.use_cache = bool(
            settings.value("use_cache", USE_CACHE_DEFAULT, type=bool)
        )
        detect_weights_path = settings.value(
            "detect_weights_path",
            "",
            type=str,
        ).strip()
        if detect_weights_path:
            self.weights_path = detect_weights_path

        classify_weights_path = settings.value(
            "classify_weights_path",
            "",
            type=str,
        ).strip()
        if classify_weights_path:
            self.classify_weights_path = classify_weights_path
        else:
            self.classify_weights_path = str(DEFAULT_CLASSIFY_WEIGHTS_PATH)
        logger.info(
            "Настройки загружены: High=%s, Low=%s, DetectConf=%s, DetectIoU=%s, PxPerMm=%s, Cache=%s",
            cfg.CONF_THRESHOLD_HIGH,
            cfg.CONF_THRESHOLD_LOW,
            self.detection_confidence_threshold,
            self.detection_iou_threshold,
            self.pixels_per_mm,
            self.use_cache,
        )
        if hasattr(self, "app_state"):
            self.app_state.pixels_per_mm = self.pixels_per_mm
            self.app_state.use_cache = self.use_cache
        if hasattr(self, "calibration_px_per_mm_label"):
            self._refresh_calibration_card()

    @staticmethod
    def _read_float_setting(
        settings: QSettings,
        key: str,
        default: float,
    ) -> float:
        """Безопасно читает число с fallback к значению по умолчанию."""
        try:
            return float(settings.value(key, default))
        except (TypeError, ValueError):
            return float(default)

    def setup_ui(self) -> None:
        """      ."""
        self.setup_actions()
        self.create_left_panel()
        self.create_central_widget()
        self.create_right_panel()
        self.create_panel_docks()
        self.create_toolbox_dock()
        self._build_dashboard_layout()
        self.connect_signals()
        self._apply_language(self.current_language)

        self._setup_shortcuts()
        self._set_detection_actions_enabled(False)
        self._restore_layout_settings()
        self._populate_model_selector()
        self._refresh_project_files_list()
        self._refresh_detection_result_card()
        self._refresh_calibration_card()
        self._refresh_activity_view()
        self._update_header_badges()

    def init_ui(self) -> None:
        """       UI."""
        self.setup_ui()

    def setup_actions(self) -> None:
        """     ."""
        self._register_toolbar_actions()
        self.create_menu()

    def connect_signals(self) -> None:
        """    ."""
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)
        self.tree_widget.measure_requested.connect(
            self._on_tree_measure_requested
        )
        self.tree_widget.classify_requested.connect(
            self._on_tree_classify_requested
        )
        self.tree_widget.add_part_requested.connect(
            self._on_tree_add_part_requested
        )
        self.tree_widget.add_seedling_requested.connect(
            self._on_tree_add_seedling_requested
        )
        self.tree_widget.delete_requested.connect(
            self._on_tree_delete_requested
        )
        self.statistics_panel.export_csv_requested.connect(
            self._export_statistics_csv
        )
        self.thumbnails_panel.image_selected.connect(
            self._on_thumbnail_selected
        )

    def _set_detection_actions_enabled(self, enabled: bool) -> None:
        """    ."""
        self.action_find.setEnabled(enabled)
        self.action_find_all.setEnabled(enabled)
        if hasattr(self, "process_current_button"):
            self.process_current_button.setEnabled(enabled)
        if hasattr(self, "process_all_button"):
            self.process_all_button.setEnabled(enabled)

    def _build_dashboard_layout(self) -> None:
        """ layout    ."""
        shell = QWidget(self)
        shell.setObjectName("appShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(12, 8, 12, 10)
        shell_layout.setSpacing(10)

        self.header_bar = self._create_header_bar()
        shell_layout.addWidget(self.header_bar)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(12)

        self.left_sidebar = self._create_left_sidebar()
        self.left_sidebar.setMinimumWidth(300)
        self.left_sidebar.setMaximumWidth(360)
        body_layout.addWidget(self.left_sidebar, 0)

        self.center_column = self._create_center_column()
        body_layout.addWidget(self.center_column, 1)

        self.right_sidebar = self._create_right_sidebar()
        self.right_sidebar.setMinimumWidth(PANEL_LAYERS_MIN_WIDTH + 32)
        self.right_sidebar.setMaximumWidth(PANEL_LAYERS_MAX_WIDTH + 120)
        body_layout.addWidget(self.right_sidebar, 0)

        shell_layout.addLayout(body_layout, 1)
        self.setCentralWidget(shell)

    def _create_header_bar(self) -> QFrame:
        """   ."""
        header = QFrame(self)
        header.setObjectName("headerBar")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        brand_widget = QWidget(header)
        brand_layout = QHBoxLayout(brand_widget)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(10)

        brand_icon = QLabel(brand_widget)
        brand_icon.setObjectName("brandIcon")
        detect_icon = self.icon_manager.get_icon(
            "action_detect.svg",
            fallback_standard_icon=QStyle.SP_TitleBarMenuButton,
        )
        if not detect_icon.isNull():
            brand_icon.setPixmap(detect_icon.pixmap(20, 20))
        brand_layout.addWidget(brand_icon, 0, Qt.AlignVCenter)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(0)
        self.brand_title_label = QLabel("Seeding Detection", brand_widget)
        self.brand_title_label.setObjectName("brandTitle")
        self.brand_subtitle_label = QLabel(
            self._tr(
                "header_subtitle",
                "Seedling detection on images",
            ),
            brand_widget,
        )
        self.brand_subtitle_label.setObjectName("brandSubtitle")
        title_col.addWidget(self.brand_title_label)
        title_col.addWidget(self.brand_subtitle_label)
        brand_layout.addLayout(title_col)

        self.brand_version_badge = QLabel("v1.20", brand_widget)
        self.brand_version_badge.setObjectName("versionBadge")
        brand_layout.addWidget(self.brand_version_badge, 0, Qt.AlignVCenter)

        layout.addWidget(brand_widget, 0)
        layout.addStretch(1)

        self.theme_switch = QFrame(header)
        self.theme_switch.setObjectName("themeSwitch")
        theme_layout = QHBoxLayout(self.theme_switch)
        theme_layout.setContentsMargins(4, 4, 4, 4)
        theme_layout.setSpacing(4)

        self.theme_button_group = QButtonGroup(self)
        self.theme_button_group.setExclusive(True)
        self.theme_light_button = QPushButton(
            self._tr("settings_theme_light", "Light"),
            self.theme_switch,
        )
        self.theme_light_button.setCheckable(True)
        self.theme_light_button.setProperty("segmented", True)
        self.theme_light_button.clicked.connect(
            lambda: self._on_theme_switch_requested("light")
        )
        self.theme_button_group.addButton(self.theme_light_button)
        theme_layout.addWidget(self.theme_light_button)

        self.theme_dark_button = QPushButton(
            self._tr("settings_theme_dark", "Dark"),
            self.theme_switch,
        )
        self.theme_dark_button.setCheckable(True)
        self.theme_dark_button.setProperty("segmented", True)
        self.theme_dark_button.clicked.connect(
            lambda: self._on_theme_switch_requested("dark")
        )
        self.theme_button_group.addButton(self.theme_dark_button)
        theme_layout.addWidget(self.theme_dark_button)

        self.theme_auto_button = QPushButton(
            self._tr("theme_auto", "Auto"),
            self.theme_switch,
        )
        self.theme_auto_button.setCheckable(True)
        self.theme_auto_button.setProperty("segmented", True)
        self.theme_auto_button.clicked.connect(
            lambda: self._on_theme_switch_requested("auto")
        )
        self.theme_button_group.addButton(self.theme_auto_button)
        theme_layout.addWidget(self.theme_auto_button)
        layout.addWidget(self.theme_switch, 0, Qt.AlignVCenter)

        self.mode_badge = QLabel("", header)
        self.mode_badge.setObjectName("modeBadge")
        layout.addWidget(self.mode_badge, 0, Qt.AlignVCenter)

        self.model_selector_combo = QComboBox(header)
        self.model_selector_combo.setObjectName("modelSelector")
        self.model_selector_combo.setMinimumWidth(230)
        self.model_selector_combo.currentIndexChanged.connect(
            self._on_model_selector_changed
        )
        layout.addWidget(self.model_selector_combo, 0, Qt.AlignVCenter)

        self.classify_model_selector_combo = QComboBox(header)
        self.classify_model_selector_combo.setObjectName("modelSelector")
        self.classify_model_selector_combo.setMinimumWidth(230)
        self.classify_model_selector_combo.currentIndexChanged.connect(
            self._on_classify_model_selector_changed
        )
        layout.addWidget(self.classify_model_selector_combo, 0, Qt.AlignVCenter)

        return header

    def _create_left_sidebar(self) -> QWidget:
        """       ."""
        sidebar = QWidget(self)
        sidebar.setObjectName("leftSidebar")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        files_card, files_layout = self._build_panel_card(
            self._tr("panel_files", "Files")
        )
        self.open_image_button = self._create_panel_button(
            text=self._tr("open_image", "Open image"),
            icon_name="action_open.svg",
            primary=True,
            handler=self.open_image,
        )
        files_layout.addWidget(self.open_image_button)

        self.open_pdf_button = self._create_panel_button(
            text=self._tr("open_pdf", "Open PDF"),
            icon_name="action_report.svg",
            primary=False,
            handler=self.open_pdf_file,
        )
        files_layout.addWidget(self.open_pdf_button)

        self.open_folder_button = self._create_panel_button(
            text=self._tr("open_folder", "Open folder"),
            icon_name="action_add.svg",
            primary=False,
            handler=self.open_folder,
        )
        files_layout.addWidget(self.open_folder_button)

        self.files_layers_label = QLabel("Layers (0)", files_card)
        self.files_layers_label.setObjectName("panelSubTitle")
        files_layout.addWidget(self.files_layers_label)

        self.project_files_list = QListWidget(files_card)
        self.project_files_list.setObjectName("projectFilesList")
        self.project_files_list.itemClicked.connect(
            self._on_project_file_clicked
        )
        files_layout.addWidget(self.project_files_list, 1)
        layout.addWidget(files_card, 3)

        calibration_card, calibration_layout = self._build_panel_card(
            self._tr("panel_calibration", "Calibration")
        )
        self.calibration_mode_button = self._create_panel_button(
            text=self._tr("calibration_mode", "Calibration mode"),
            icon_name="tool_measure.svg",
            primary=False,
            handler=self._start_calibration_from_settings,
        )
        calibration_layout.addWidget(self.calibration_mode_button)

        self.calibration_hint_label = QLabel(
            self._tr(
                "calibration_hint",
                "Run calibration to convert pixels to real units.",
            ),
            calibration_card,
        )
        self.calibration_hint_label.setObjectName("panelHint")
        self.calibration_hint_label.setWordWrap(True)
        calibration_layout.addWidget(self.calibration_hint_label)

        self.calibration_px_per_mm_label = QLabel("", calibration_card)
        self.calibration_px_per_mm_label.setObjectName("metricChip")
        calibration_layout.addWidget(self.calibration_px_per_mm_label)

        self.calibration_mm_per_px_label = QLabel("", calibration_card)
        self.calibration_mm_per_px_label.setObjectName("metricChip")
        calibration_layout.addWidget(self.calibration_mm_per_px_label)

        self.calibration_scale_label = QLabel("", calibration_card)
        self.calibration_scale_label.setObjectName("panelHint")
        calibration_layout.addWidget(self.calibration_scale_label)

        self.calibration_restart_button = self._create_panel_button(
            text=self._tr("calibration_restart", "Restart"),
            icon_name="action_rotate.svg",
            primary=False,
            handler=self._start_calibration_from_settings,
        )
        calibration_layout.addWidget(self.calibration_restart_button)

        self.calibration_reset_button = self._create_panel_button(
            text=self._tr("calibration_reset", "Reset coefficient"),
            icon_name="action_zoom_out.svg",
            primary=False,
            handler=self._reset_calibration,
        )
        calibration_layout.addWidget(self.calibration_reset_button)
        layout.addWidget(calibration_card, 2)

        process_card, process_layout = self._build_panel_card(
            self._tr("panel_processing", "Processing")
        )
        self.process_all_button = self._create_panel_button(
            text=self._tr("process_all", "Process all (YOLO)"),
            icon_name="action_detect_all.svg",
            primary=True,
            handler=self.find_all_seedlings,
        )
        process_layout.addWidget(self.process_all_button)

        self.process_classify_button = self._create_panel_button(
            text=self._tr(
                "process_classify",
                "Segmentation + classification (all photos)",
            ),
            icon_name="action_classify.svg",
            primary=False,
            handler=self.process_all_segmentation_classification,
        )
        process_layout.addWidget(self.process_classify_button)
        layout.addWidget(process_card, 1)

        layout.addStretch(1)
        return sidebar

    def _create_center_column(self) -> QWidget:
        """       canvas."""
        center = QWidget(self)
        center.setObjectName("centerColumn")
        layout = QVBoxLayout(center)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        toolbar_frame = QFrame(center)
        toolbar_frame.setObjectName("canvasToolbar")
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(12, 8, 12, 8)
        toolbar_layout.setSpacing(8)

        self.process_current_button = self._create_panel_button(
            text=self._tr("process_current", "Find seedlings"),
            icon_name="action_detect.svg",
            primary=True,
            handler=self.find_seedlings,
            parent=toolbar_frame,
        )
        toolbar_layout.addWidget(self.process_current_button)

        self.select_tool_button = self._create_tool_toggle_button(
            self._tr("tool_select_short", "Select"),
            "select",
            parent=toolbar_frame,
        )
        self.hand_tool_button = self._create_tool_toggle_button(
            self._tr("tool_hand_short", "Pan"),
            "hand",
            parent=toolbar_frame,
        )
        self.measure_tool_button = self._create_tool_toggle_button(
            self._tr("tool_measure_short", "Measure"),
            "measure",
            parent=toolbar_frame,
        )
        self.tool_toggle_button_group = QButtonGroup(self)
        self.tool_toggle_button_group.setExclusive(True)
        self.tool_toggle_button_group.addButton(self.select_tool_button)
        self.tool_toggle_button_group.addButton(self.hand_tool_button)
        self.tool_toggle_button_group.addButton(self.measure_tool_button)
        toolbar_layout.addWidget(self.select_tool_button)
        toolbar_layout.addWidget(self.hand_tool_button)
        toolbar_layout.addWidget(self.measure_tool_button)
        toolbar_layout.addStretch(1)

        self.show_boxes_button = self._create_panel_button(
            text=self._tr("toggle_boxes", "Show boxes"),
            icon_name="action_detect.svg",
            primary=False,
            handler=self._toggle_boxes_visibility,
            parent=toolbar_frame,
        )
        self.show_boxes_button.setCheckable(True)
        self.show_boxes_button.setChecked(True)
        toolbar_layout.addWidget(self.show_boxes_button)

        self.box_class_filter_bar = QFrame(toolbar_frame)
        self.box_class_filter_bar.setObjectName("boxClassFilterBar")
        class_filter_layout = QHBoxLayout(self.box_class_filter_bar)
        class_filter_layout.setContentsMargins(0, 0, 0, 0)
        class_filter_layout.setSpacing(6)
        self.seedling_filter_button = self._create_box_class_filter_button(
            self._tr("class_seedling", "Seedling"),
            "seeding",
            parent=self.box_class_filter_bar,
        )
        self.inflorescence_filter_button = self._create_box_class_filter_button(
            self._tr("class_inflorescence", "Inflorescence"),
            "inflorescence",
            parent=self.box_class_filter_bar,
        )
        self.stem_filter_button = self._create_box_class_filter_button(
            self._tr("class_stem", "Stem"),
            "stem",
            parent=self.box_class_filter_bar,
        )
        self.root_filter_button = self._create_box_class_filter_button(
            self._tr("class_root", "Root"),
            "root",
            parent=self.box_class_filter_bar,
        )
        for button in (
            self.seedling_filter_button,
            self.inflorescence_filter_button,
            self.stem_filter_button,
            self.root_filter_button,
        ):
            class_filter_layout.addWidget(button)
        toolbar_layout.addWidget(self.box_class_filter_bar)

        self.rotate_button = self._create_panel_button(
            text=self._tr("rotate", "Rotate"),
            icon_name="action_rotate.svg",
            primary=False,
            handler=self.rotate_image,
            parent=toolbar_frame,
        )
        toolbar_layout.addWidget(self.rotate_button)

        layout.addWidget(toolbar_frame, 0)
        layout.addWidget(self.canvas_host, 1)
        return center

    def _create_right_sidebar(self) -> QWidget:
        """       ."""
        sidebar_content = QWidget(self)
        sidebar_content.setObjectName("rightSidebarContent")
        layout = QVBoxLayout(sidebar_content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(12)

        results_card, results_layout = self._build_panel_card(
            self._tr("panel_detection_results", "Detection results")
        )
        self.results_found_chip = QLabel("Found: 0", results_card)
        self.results_found_chip.setObjectName("foundChip")
        results_layout.addWidget(self.results_found_chip, 0, Qt.AlignLeft)

        self.results_text = QTextEdit(results_card)
        self.results_text.setObjectName("resultsText")
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(120)
        results_layout.addWidget(self.results_text, 1)
        layout.addWidget(results_card, 0)

        history_card, history_layout = self._build_panel_card(
            self._tr("panel_activity", "Activity log")
        )
        self.history_refresh_button = self._create_panel_button(
            text=self._tr("refresh_history", "Refresh"),
            icon_name="action_detect_all.svg",
            primary=False,
            handler=self._refresh_activity_view,
            parent=history_card,
        )
        history_layout.addWidget(self.history_refresh_button, 0, Qt.AlignRight)

        self.history_text = QTextEdit(history_card)
        self.history_text.setObjectName("historyText")
        self.history_text.setReadOnly(True)
        self.history_text.setMinimumHeight(140)
        history_layout.addWidget(self.history_text, 1)
        layout.addWidget(history_card, 0)

        inspector_card, inspector_layout = self._build_panel_card(
            self._tr("panel_inspector", "Layers and details")
        )
        self.right_tabs.setMinimumHeight(460)
        inspector_layout.addWidget(self.right_tabs, 1)
        layout.addWidget(inspector_card, 1)

        scroll = QScrollArea(self)
        scroll.setObjectName("rightSidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setViewportMargins(0, 0, 6, 0)
        scroll.setWidget(sidebar_content)
        return scroll

    def _build_panel_card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        """ -  ."""
        card = QFrame(self)
        card.setObjectName("panelCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title_label = QLabel(title, card)
        title_label.setObjectName("panelCardTitle")
        layout.addWidget(title_label)

        divider = QFrame(card)
        divider.setObjectName("panelDivider")
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Plain)
        layout.addWidget(divider)
        return card, layout

    def _create_panel_button(
        self,
        *,
        text: str,
        icon_name: str,
        primary: bool,
        handler: Callable[[], None],
        parent: QWidget | None = None,
    ) -> QPushButton:
        """   ."""
        button = QPushButton(text, parent or self)
        button.setObjectName(
            "primaryActionButton" if primary else "secondaryActionButton"
        )
        button.setProperty("variant", "primary" if primary else "secondary")
        button.setCursor(Qt.PointingHandCursor)
        icon = self.icon_manager.get_icon(
            icon_name,
            fallback_standard_icon=QStyle.SP_DialogApplyButton,
        )
        if not icon.isNull():
            button.setIcon(icon)
        button.clicked.connect(handler)
        return button

    def _create_tool_toggle_button(
        self,
        text: str,
        tool_name: str,
        *,
        parent: QWidget,
    ) -> QPushButton:
        """    canvas."""
        button = QPushButton(text, parent)
        button.setObjectName("toolToggleButton")
        button.setCheckable(True)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(
            lambda checked: checked and self._set_checked_tool(tool_name)
        )
        return button

    def _create_box_class_filter_button(
        self,
        text: str,
        class_key: str,
        *,
        parent: QWidget,
    ) -> QPushButton:
        """Создаёт кнопку-фильтр для управления видимостью класса боксов."""
        button = QPushButton(self._short_box_filter_label(class_key), parent)
        button.setObjectName("classFilterButton")
        button.setToolTip(text)
        button.setCheckable(True)
        button.setChecked(self._box_class_visibility.get(class_key, True))
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(
            lambda checked, key=class_key: self._on_box_class_filter_toggled(
                key,
                checked,
            )
        )
        return button

    def _short_box_filter_label(self, class_key: str) -> str:
        """Возвращает короткую подпись кнопки фильтра класса боксов."""
        language = (self.current_language or "en").strip().lower()
        if language == "ru":
            return {
                "seeding": "Сеян.",
                "inflorescence": "Соцв.",
                "stem": "Стеб.",
                "root": "Кор.",
                "other": "Проч.",
            }.get(class_key, "Кл.")
        return {
            "seeding": "Seed",
            "inflorescence": "Infl",
            "stem": "Stem",
            "root": "Root",
            "other": "Other",
        }.get(class_key, "Class")

    def create_panel_docks(self) -> None:
        """     ( QDockWidget)."""
        self.right_tabs = QTabWidget(self)
        self.right_tabs.setObjectName("rightTabs")
        self.statistics_panel = StatisticsPanel(self)
        self.thumbnails_panel = ThumbnailsPanel(self)
        self.right_tabs.addTab(
            self.right_panel,
            tr(self.current_language, "tab_layers", "Слои"),
        )
        self.right_tabs.addTab(
            self.left_panel,
            tr(self.current_language, "tab_properties", "Свойства"),
        )
        self.right_tabs.addTab(
            self.statistics_panel,
            tr(self.current_language, "tab_statistics", "Статистика"),
        )
        self.right_tabs.addTab(
            self.thumbnails_panel,
            tr(self.current_language, "tab_thumbnails", "Миниатюры"),
        )
        self.right_tabs.setMinimumWidth(PANEL_LAYERS_MIN_WIDTH)

    def create_toolbox_dock(self) -> None:
        """      QAction ."""
        self.toolbox_toolbar = QToolBar("", self)
        self.toolbox_toolbar.setObjectName("toolboxToolbar")
        self.toolbox_toolbar.setOrientation(Qt.Horizontal)
        self.toolbox_toolbar.setMovable(False)
        self.toolbox_toolbar.setIconSize(
            QSize(
                self.ui_metrics.icon_size_toolbox,
                self.ui_metrics.icon_size_toolbox,
            )
        )
        self._set_uniform_button_style(self.toolbox_toolbar)

        self.tool_action_group = QActionGroup(self)
        self.tool_action_group.setExclusive(True)

        self.action_tool_select = self._create_tool_action(
            attr_name="action_tool_select",
            icon_name="tool_select.svg",
            tool_name="select",
            hint=tr(self.current_language, "tool_select", "Выбор объектов (V)"),
            fallback_icon=QStyle.SP_ArrowUp,
        )
        self.action_tool_hand = self._create_tool_action(
            attr_name="action_tool_hand",
            icon_name="tool_hand.svg",
            tool_name="hand",
            hint=tr(
                self.current_language,
                "tool_hand",
                "Панорамирование (H, Space)",
            ),
            fallback_icon=QStyle.SP_DialogOpenButton,
        )
        self.action_tool_zoom = self._create_tool_action(
            attr_name="action_tool_zoom",
            icon_name="tool_zoom.svg",
            tool_name="zoom",
            hint=tr(
                self.current_language,
                "tool_zoom",
                "Режим масштабирования (Z)",
            ),
            fallback_icon=QStyle.SP_FileDialogDetailedView,
        )
        self.action_tool_measure = self._create_tool_action(
            attr_name="action_tool_measure",
            icon_name="tool_measure.svg",
            tool_name="measure",
            hint=tr(
                self.current_language,
                "tool_measure",
                "Линейка (M): измерение двумя точками",
            ),
            fallback_icon=QStyle.SP_FileDialogDetailedView,
        )

        self._append_actions_to_toolbox()
        self.toolbox_toolbar.hide()
        self._activate_default_tool()

    def _append_actions_to_toolbox(self) -> None:
        """     -."""
        action_groups = [
            [self.action_open, self.action_add, self.action_save],
            [self.action_export, self.action_report],
            [self.action_find, self.action_find_all, self.action_classify],
            [self.action_rotate, self.action_mask],
            [self.action_zoom_in, self.action_zoom_out, self.action_fit],
            [self.action_settings],
        ]
        self.toolbox_toolbar.addSeparator()
        for group_index, group_actions in enumerate(action_groups):
            for action in group_actions:
                self.toolbox_toolbar.addAction(action)
            if group_index < len(action_groups) - 1:
                self.toolbox_toolbar.addSeparator()

    def _create_tool_action(
        self,
        *,
        attr_name: str,
        icon_name: str,
        tool_name: str,
        hint: str,
        fallback_icon: QStyle.StandardPixmap,
    ) -> QAction:
        """   ."""
        action = QAction(
            self.icon_manager.get_icon(
                icon_name,
                fallback_standard_icon=fallback_icon,
            ),
            "",
            self,
        )
        action.setCheckable(True)
        self._set_action_hints(action, hint)
        action.triggered.connect(
            lambda checked, name=tool_name: checked and self._set_active_tool(name)
        )
        self.tool_action_group.addAction(action)
        self.toolbox_toolbar.addAction(action)
        setattr(self, attr_name, action)
        return action

    def _activate_default_tool(self) -> None:
        """   ."""
        self._active_tool = "select"
        self._space_hand_active = False
        self._tool_before_space = "select"
        self.action_tool_select.setChecked(True)
        self._set_active_tool("select")

    def _restore_layout_settings(self) -> None:
        """     ."""
        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        geometry = settings.value("window_geometry", QByteArray())
        geometry_bytes = normalize_qbytearray(geometry)
        if not geometry_bytes.isEmpty():
            self.restoreGeometry(geometry_bytes)

        tool_name = settings.value("active_tool", "select", type=str)
        self._set_checked_tool(tool_name)

    def _save_layout_settings(self) -> None:
        """     ."""
        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        settings.setValue("window_geometry", self.saveGeometry())
        settings.setValue("active_tool", getattr(self, "_active_tool", "select"))
        settings.sync()

    def _on_theme_switch_requested(self, theme_name: str) -> None:
        """    ."""
        selected_theme = self._auto_theme_fallback if theme_name == "auto" else theme_name
        self._apply_theme(selected_theme)
        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        settings.setValue("ui_theme", selected_theme)
        settings.sync()
        self._update_header_badges()

    @staticmethod
    def _normalize_path(path: str) -> str:
        """    ."""
        return os.path.normcase(os.path.normpath(path))

    def _populate_model_selector(self) -> None:
        """Перезаполняет селекторы моделей детекции и классификации."""
        if hasattr(self, "model_selector_combo"):
            self._populate_single_model_combo(
                combo=self.model_selector_combo,
                current_path=self.weights_path,
                header_key="header_detect_model",
                header_fallback="Detection",
                preset_options=(
                    ("YOLOv8 Nano (fast)", "models/yolov8n.pt"),
                    ("YOLOv8 Small", "models/yolov8s.pt"),
                    ("YOLOv8 Medium", "models/yolov8m.pt"),
                    ("YOLOv8 Large (accurate)", "models/yolov8l.pt"),
                ),
            )
        if hasattr(self, "classify_model_selector_combo"):
            self._populate_single_model_combo(
                combo=self.classify_model_selector_combo,
                current_path=self.classify_weights_path,
                header_key="header_classify_model",
                header_fallback="Segmentation",
                preset_options=(
                    ("YOLOv8 Nano Seg", "models/yolov8n-seg.pt"),
                    ("YOLOv8 Small Seg", "models/yolov8s-seg.pt"),
                    ("YOLOv8 Medium Seg", "models/yolov8m-seg.pt"),
                    ("YOLOv8 Large Seg", "models/yolov8l-seg.pt"),
                ),
            )

    def _populate_single_model_combo(
        self,
        *,
        combo: QComboBox,
        current_path: str,
        header_key: str,
        header_fallback: str,
        preset_options: Sequence[tuple[str, str]],
    ) -> None:
        """Заполняет переданный комбобокс модельными путями без дубликатов."""
        current = str(current_path)
        current_normalized = self._normalize_path(current)
        seen: set[str] = set()

        options: list[tuple[str, str]] = [
            (
                f"{self._tr(header_key, header_fallback)}: {Path(current).name}",
                current,
            )
        ]
        for label, raw_path in preset_options:
            resolved = resolve_weights_path(
                raw_path,
                base_dirs=(cfg.PROJECT_ROOT, Path.cwd()),
            )
            if resolved is None:
                continue
            options.append((label, str(resolved)))

        models_dir = cfg.PROJECT_ROOT / "models"
        if models_dir.is_dir():
            for model_path in sorted(models_dir.glob("*.pt")):
                options.append((f"Custom: {model_path.name}", str(model_path)))

        combo.blockSignals(True)
        combo.clear()
        current_index = 0
        for label, option_path in options:
            normalized = self._normalize_path(option_path)
            if normalized in seen:
                continue
            seen.add(normalized)
            combo.addItem(label, option_path)
            if normalized == current_normalized:
                current_index = combo.count() - 1

        if combo.count() == 0:
            combo.addItem(Path(current).name, current)
            current_index = 0
        combo.setCurrentIndex(current_index)
        combo.blockSignals(False)

    def _on_model_selector_changed(self, index: int) -> None:
        """      ."""
        if index < 0 or not hasattr(self, "model_selector_combo"):
            return
        selected_path = str(self.model_selector_combo.itemData(index) or "").strip()
        if not selected_path:
            return
        resolved = resolve_weights_path(
            selected_path,
            base_dirs=(cfg.PROJECT_ROOT, Path.cwd()),
        )
        if resolved is None:
            self.statusBar().showMessage(
                self._tr("status_model_missing", "Model is not loaded"),
                3000,
            )
            return
        resolved_path = str(resolved)
        if self._normalize_path(resolved_path) == self._normalize_path(self.weights_path):
            return

        self.weights_path = resolved_path
        self.model = None
        self._set_detection_actions_enabled(False)
        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        settings.setValue("detect_weights_path", self.weights_path)
        settings.sync()
        self._populate_model_selector()
        self._update_header_badges()
        self._append_activity(
            self._tr("activity_model_selected", "model-selected"),
            self.weights_path,
        )
        self._start_model_loading()

    def _on_classify_model_selector_changed(self, index: int) -> None:
        """Обрабатывает переключение модели классификации/сегментации."""
        if index < 0 or not hasattr(self, "classify_model_selector_combo"):
            return
        selected_path = str(
            self.classify_model_selector_combo.itemData(index) or ""
        ).strip()
        if not selected_path:
            return
        resolved = resolve_weights_path(
            selected_path,
            base_dirs=(cfg.PROJECT_ROOT, Path.cwd()),
        )
        if resolved is None:
            self.statusBar().showMessage(
                self._tr("status_classify_model_missing", "Segmentation model not found"),
                3000,
            )
            return
        resolved_path = str(resolved)
        if self._normalize_path(resolved_path) == self._normalize_path(self.classify_weights_path):
            return

        self.classify_weights_path = resolved_path
        self.classify_model = None
        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        settings.setValue("classify_weights_path", self.classify_weights_path)
        settings.sync()
        self._populate_model_selector()
        self._append_activity(
            self._tr("activity_classify_model_selected", "classify-model-selected"),
            self.classify_weights_path,
        )
        self.statusBar().showMessage(
            self._tr("status_classify_model_updated", "Segmentation model updated"),
            2500,
        )

    def _update_header_badges(self) -> None:
        """      ."""
        if hasattr(self, "mode_badge"):
            model_name = Path(self.weights_path).name
            self.mode_badge.setText(
                self._tr("header_mode_badge", "Detection: {name}").format(
                    name=model_name
                )
            )
        if hasattr(self, "theme_dark_button") and hasattr(self, "theme_light_button"):
            current_theme = (self.current_theme or "dark").strip().lower()
            if current_theme == "light":
                self.theme_light_button.setChecked(True)
            elif current_theme == "dark":
                self.theme_dark_button.setChecked(True)
            else:
                self.theme_auto_button.setChecked(True)

    def _refresh_project_files_list(self) -> None:
        """     ."""
        if not hasattr(self, "project_files_list"):
            return
        self.project_files_list.clear()
        count = self.tree_widget.topLevelItemCount() if hasattr(self, "tree_widget") else 0
        if hasattr(self, "files_layers_label"):
            self.files_layers_label.setText(f"{self._tr('layers', 'Layers')} ({count})")

        for idx in range(count):
            root_item = self.tree_widget.topLevelItem(idx)
            text = root_item.text(0) if root_item is not None else f"{idx + 1}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            self.project_files_list.addItem(item)
        self._set_active_file_row(self._active_image_index)

    def _on_project_file_clicked(self, item: QListWidgetItem) -> None:
        """       ."""
        if item is None:
            return
        index = int(item.data(Qt.UserRole))
        if index < 0 or index >= len(self.image_storage.images):
            return
        self._active_image_index = index
        self.app_state.active_image_index = index
        self.display_image_with_boxes(index)
        tree_item = self.tree_widget.topLevelItem(index)
        if tree_item is not None:
            self.tree_widget.setCurrentItem(tree_item)
        self.update_left_info({"type": "pdf", "index": index})

    def _set_active_file_row(self, index: int) -> None:
        """    ."""
        if not hasattr(self, "project_files_list"):
            return
        if 0 <= index < self.project_files_list.count():
            self.project_files_list.blockSignals(True)
            self.project_files_list.setCurrentRow(index)
            self.project_files_list.blockSignals(False)

    def _refresh_detection_result_card(self) -> None:
        """    ."""
        if not hasattr(self, "results_text"):
            return
        count = 0
        lines: list[str] = []
        if (
            self.image_storage.class_object_image
            and 0 <= self._active_image_index < len(self.image_storage.class_object_image)
        ):
            objects = self.image_storage.class_object_image[self._active_image_index] or []
            count = len(objects)
            for idx, obj in enumerate(objects[:20]):
                lines.append(f"{self._tr('class_seedling', 'Seedling')} #{idx + 1}")
                lines.append(
                    f"{self._tr('confidence_label', 'Confidence')}: {obj.confidence * 100:.1f}%"
                )
                lines.append("")

        self._last_detection_count = count
        if hasattr(self, "results_found_chip"):
            self.results_found_chip.setText(
                self._tr("results_found", "Found: {count}").format(count=count)
            )
        self.results_text.setPlainText("\n".join(lines).strip())

    def _refresh_calibration_card(self) -> None:
        """      ."""
        if not hasattr(self, "calibration_px_per_mm_label"):
            return
        if self.pixels_per_mm > 0:
            mm_per_px = 1.0 / self.pixels_per_mm
            self.calibration_px_per_mm_label.setText(
                self._tr("calibration_coeff", "Coefficient: {value:.2f} px/mm").format(
                    value=self.pixels_per_mm
                )
            )
            self.calibration_mm_per_px_label.setText(
                self._tr("calibration_step", "Step: {value:.3f} mm/px").format(
                    value=mm_per_px
                )
            )
            self.calibration_scale_label.setText(
                self._tr(
                    "calibration_scale",
                    "Scale: 10 mm = {value:.1f} px",
                ).format(value=self.pixels_per_mm * 10.0)
            )
        else:
            self.calibration_px_per_mm_label.setText(
                self._tr("calibration_coeff", "Coefficient: not set")
            )
            self.calibration_mm_per_px_label.setText(
                self._tr("calibration_step", "Step: not set")
            )
            self.calibration_scale_label.setText(
                self._tr(
                    "calibration_scale",
                    "Run calibration to measure in mm.",
                )
            )

    def _reset_calibration(self) -> None:
        """      ."""
        self.pixels_per_mm = CALIBRATION_PIXELS_PER_MM_DEFAULT
        self.app_state.pixels_per_mm = self.pixels_per_mm
        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        settings.setValue("pixels_per_mm", self.pixels_per_mm)
        settings.sync()
        self._refresh_calibration_card()
        self._refresh_current_view()
        self.statusBar().showMessage(
            self._tr("calibration_reset_done", "Calibration reset"),
            3000,
        )

    def _append_activity(self, action: str, details: str = "") -> None:
        """     ."""
        ts = datetime.now().strftime("%d.%m.%Y, %H:%M:%S")
        payload = f"{action}\n{ts}"
        if details:
            payload = f"{payload}\n{details}"
        self._activity_log.insert(0, payload)
        self._activity_log = self._activity_log[:300]
        self._refresh_activity_view()

    def _refresh_activity_view(self) -> None:
        """   ."""
        if not hasattr(self, "history_text"):
            return
        self.history_text.setPlainText("\n\n".join(self._activity_log))

    def _toggle_boxes_visibility(self) -> None:
        """  bboxes   ."""
        self._show_boxes = bool(
            getattr(self, "show_boxes_button", None).isChecked()
            if hasattr(self, "show_boxes_button")
            else True
        )
        self._append_activity(
            self._tr("activity_toggle_boxes", "toggle-boxes"),
            self._tr("toggle_state", "Boxes: ON")
            if self._show_boxes
            else self._tr("toggle_state_off", "Boxes: OFF"),
        )
        self._refresh_current_view()

    def _on_box_class_filter_toggled(self, class_key: str, checked: bool) -> None:
        """Обновляет видимость боксов выбранного класса и перерисовывает canvas."""
        self._box_class_visibility[class_key] = bool(checked)
        self._refresh_current_view()

    def _is_box_class_visible(self, class_key: str) -> bool:
        """Возвращает, включён ли показ боксов указанного класса."""
        return bool(self._box_class_visibility.get(class_key, True))

    def _refresh_current_view(self) -> None:
        """    canvas."""
        item_data = getattr(self.app_state, "selected_item", None)
        if isinstance(item_data, dict):
            item_type = item_data.get("type")
            if item_type in ("original", "pdf"):
                index = int(item_data.get("index", self._active_image_index))
                self.display_image_with_boxes(index, preserve_view=True)
                return
            if item_type == "seeding":
                self.display_image_with_boxes(
                    int(item_data.get("parent_index", self._active_image_index)),
                    seeding_idx=int(item_data.get("index", 0)),
                    preserve_view=True,
                )
                return
            if item_type == "class":
                self.display_image_with_boxes(
                    int(item_data.get("parent_index", self._active_image_index)),
                    seeding_idx=int(item_data.get("seeding_index", 0)),
                    preserve_view=True,
                )
                return
        self.display_image_with_boxes(
            self._active_image_index,
            preserve_view=True,
        )

    def _apply_language(self, language: str) -> None:
        """Применяет язык интерфейса для ключевых элементов."""
        self.current_language = language
        self.setWindowTitle(tr(language, "window_title", "Анализ сеянцев"))
        if hasattr(self, "file_menu"):
            self.file_menu.setTitle(tr(language, "menu_file", "Файл"))
        if hasattr(self, "view_menu"):
            self.view_menu.setTitle(tr(language, "menu_view", "Вид"))
        if hasattr(self, "tools_menu"):
            self.tools_menu.setTitle(tr(language, "menu_tools", "Анализ"))
        if hasattr(self, "action_measurements_history"):
            text = tr(
                language,
                "menu_measurements_history",
                "стория измерений",
            )
            self.action_measurements_history.setText(text)
            self._set_action_hints(self.action_measurements_history, text)
        if hasattr(self, "action_measurements_export"):
            text = tr(
                language,
                "menu_measurements_export",
                "Экспорт измерений",
            )
            self.action_measurements_export.setText(text)
            self._set_action_hints(self.action_measurements_export, text)
        if hasattr(self, "action_clear_cache"):
            text = tr(language, "menu_clear_cache", "Очистить кэш")
            self.action_clear_cache.setText(text)
            self._set_action_hints(self.action_clear_cache, text)
        if hasattr(self, "left_panel"):
            self.left_panel.setTitle("")
        if hasattr(self, "right_panel"):
            self.right_panel.setTitle("")
        if hasattr(self, "right_tabs"):
            self.right_tabs.setTabText(0, tr(language, "tab_layers", "Слои"))
            self.right_tabs.setTabText(1, tr(language, "tab_properties", "Свойства"))
            self.right_tabs.setTabText(
                2,
                tr(language, "tab_statistics", "Статистика"),
            )
            self.right_tabs.setTabText(
                3,
                tr(language, "tab_thumbnails", "Миниатюры"),
            )
        if hasattr(self, "layers_dock"):
            self.layers_dock.setWindowTitle(tr(language, "dock_layers", "Слои"))
        if hasattr(self, "tree_widget"):
            self.tree_widget.set_language(language)
        if hasattr(self, "tree_search_edit"):
            self.tree_search_edit.setPlaceholderText(
                tr(language, "tree_search_placeholder", "Поиск...")
            )
        if hasattr(self, "tree_conf_filter"):
            self.tree_conf_filter.setPrefix(
                tr(language, "tree_conf_prefix", "Conf≥")
            )
        if hasattr(self, "tree_class_filter"):
            current_class = self.tree_class_filter.currentData() or "all"
            all_label = tr(language, "tree_filter_all", "Все")
            idx = self.tree_class_filter.findData("all")
            if idx >= 0:
                self.tree_class_filter.setItemText(idx, all_label)
            idx = self.tree_class_filter.findData(current_class)
            if idx >= 0:
                self.tree_class_filter.setCurrentIndex(idx)
        if hasattr(self, "statistics_panel"):
            self.statistics_panel.set_language(language)
        if hasattr(self, "thumbnails_panel"):
            self.thumbnails_panel.set_language(language)
        if hasattr(self, "action_tool_select"):
            self._set_action_hints(
                self.action_tool_select,
                tr(language, "tool_select", "Выбор объектов (V)")
            )
        if hasattr(self, "action_tool_hand"):
            self._set_action_hints(
                self.action_tool_hand,
                tr(language, "tool_hand", "Панорамирование (H, Space)")
            )
        if hasattr(self, "action_tool_zoom"):
            self._set_action_hints(
                self.action_tool_zoom,
                tr(language, "tool_zoom", "Режим масштабирования (Z)")
            )
        if hasattr(self, "action_tool_measure"):
            self._set_action_hints(
                self.action_tool_measure,
                tr(
                    language,
                    "tool_measure",
                    "Линейка (M): измерение двумя точками",
                ),
            )
        if hasattr(self, "brand_subtitle_label"):
            self.brand_subtitle_label.setText(
                self._tr(
                    "header_subtitle",
                    "Seedling detection on images",
                )
            )
        if hasattr(self, "theme_light_button"):
            self.theme_light_button.setText(
                self._tr("settings_theme_light", "Light")
            )
        if hasattr(self, "theme_dark_button"):
            self.theme_dark_button.setText(
                self._tr("settings_theme_dark", "Dark")
            )
        if hasattr(self, "theme_auto_button"):
            self.theme_auto_button.setText(
                self._tr("theme_auto", "Auto")
            )
        if hasattr(self, "open_image_button"):
            self.open_image_button.setText(
                self._tr("open_image", "Open image")
            )
        if hasattr(self, "open_pdf_button"):
            self.open_pdf_button.setText(
                self._tr("open_pdf", "Open PDF")
            )
        if hasattr(self, "open_folder_button"):
            self.open_folder_button.setText(
                self._tr("open_folder", "Open folder")
            )
        if hasattr(self, "calibration_mode_button"):
            self.calibration_mode_button.setText(
                self._tr("calibration_mode", "Calibration mode")
            )
        if hasattr(self, "calibration_restart_button"):
            self.calibration_restart_button.setText(
                self._tr("calibration_restart", "Restart")
            )
        if hasattr(self, "calibration_reset_button"):
            self.calibration_reset_button.setText(
                self._tr("calibration_reset", "Reset coefficient")
            )
        if hasattr(self, "process_current_button"):
            self.process_current_button.setText(
                self._tr("process_current", "Find seedlings")
            )
        if hasattr(self, "process_all_button"):
            self.process_all_button.setText(
                self._tr("process_all", "Process all (YOLO)")
            )
        if hasattr(self, "process_classify_button"):
            self.process_classify_button.setText(
                self._tr(
                    "process_classify",
                    "Segmentation + classification (all photos)",
                )
            )
        if hasattr(self, "seedling_filter_button"):
            full_text = self._tr("class_seedling", "Seedling")
            self.seedling_filter_button.setText(
                self._short_box_filter_label("seeding")
            )
            self.seedling_filter_button.setToolTip(full_text)
        if hasattr(self, "inflorescence_filter_button"):
            full_text = self._tr("class_inflorescence", "Inflorescence")
            self.inflorescence_filter_button.setText(
                self._short_box_filter_label("inflorescence")
            )
            self.inflorescence_filter_button.setToolTip(full_text)
        if hasattr(self, "stem_filter_button"):
            full_text = self._tr("class_stem", "Stem")
            self.stem_filter_button.setText(
                self._short_box_filter_label("stem")
            )
            self.stem_filter_button.setToolTip(full_text)
        if hasattr(self, "root_filter_button"):
            full_text = self._tr("class_root", "Root")
            self.root_filter_button.setText(
                self._short_box_filter_label("root")
            )
            self.root_filter_button.setToolTip(full_text)
        if hasattr(self, "show_boxes_button"):
            self.show_boxes_button.setText(
                self._tr("toggle_boxes", "Show boxes")
            )
        if hasattr(self, "rotate_button"):
            self.rotate_button.setText(
                self._tr("rotate", "Rotate")
            )
        if hasattr(self, "history_refresh_button"):
            self.history_refresh_button.setText(
                self._tr("refresh_history", "Refresh")
            )
        self._refresh_calibration_card()
        self._refresh_project_files_list()
        self._refresh_detection_result_card()
        self._populate_model_selector()
        self._update_header_badges()
        if hasattr(self, "empty_state_title"):
            self.empty_state_title.setText(
                tr(
                    language,
                    "empty_state_title",
                    "Перетащите файл сюда или откройте через кнопку",
                )
            )
        if hasattr(self, "empty_state_hint"):
            self.empty_state_hint.setText(
                tr(
                    language,
                    "empty_state_hint",
                    "Поддерживаются изображения и PDF-документы",
                )
            )
        if hasattr(self, "empty_open_button"):
            self.empty_open_button.setText(
                tr(language, "empty_state_open", "Открыть файл")
            )

    def _apply_theme(self, theme: str) -> None:
        """Применяет тему интерфейса в рантайме."""
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, theme)
            self.current_theme = theme
            self._update_header_badges()

    def _set_checked_tool(self, tool_name: str) -> None:
        """Активирует инструмент по имени через состояние QAction."""
        normalized_tool = (
            tool_name if tool_name in {"select", "hand", "measure"} else "select"
        )
        mapping = {
            "select": self.action_tool_select,
            "hand": self.action_tool_hand,
            "measure": self.action_tool_measure,
        }
        action = mapping.get(normalized_tool, self.action_tool_select)
        action.setChecked(True)
        self._set_active_tool(normalized_tool)

    def _set_active_tool(self, tool_name: str) -> None:
        """Применяет режим взаимодействия для выбранного инструмента."""
        if tool_name not in {"select", "hand", "measure"}:
            tool_name = "select"
        if tool_name != "measure":
            self._reset_measure_state(clear_items=True)
        self._active_tool = tool_name
        self._sync_tool_toggle_buttons(tool_name)
        self._set_scene_bboxes_editable(tool_name == "select")
        if tool_name == "hand":
            self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.graphics_view.setCursor(Qt.OpenHandCursor)
            return
        if tool_name == "measure":
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            self.graphics_view.setCursor(Qt.CrossCursor)
            self.statusBar().showMessage(
                self._tr(
                    "status_measure_pick_two_points",
                    "Линейка: выберите две точки на изображении",
                ),
                2500,
            )
            return
        self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        self.graphics_view.setCursor(Qt.ArrowCursor)

    def _set_scene_bboxes_editable(self, editable: bool) -> None:
        """Переключает редактируемость уже отрисованных bbox на сцене."""
        if not hasattr(self, "rect_items"):
            return
        for item in self.rect_items.values():
            if isinstance(item, BBoxItem):
                item.setEditable(editable)

    def _sync_tool_toggle_buttons(self, active_tool: str) -> None:
        """     ."""
        mapping = {
            "select": getattr(self, "select_tool_button", None),
            "hand": getattr(self, "hand_tool_button", None),
            "measure": getattr(self, "measure_tool_button", None),
        }
        for tool_name, button in mapping.items():
            if button is None:
                continue
            button.blockSignals(True)
            button.setChecked(tool_name == active_tool)
            button.blockSignals(False)

    def _image_scene_rect(self) -> QRectF:
        """Возвращает границы текущего изображения в координатах сцены."""
        if hasattr(self, "_original_pixmap"):
            return QRectF(
                0.0,
                0.0,
                float(self._original_pixmap.width()),
                float(self._original_pixmap.height()),
            )
        return QRectF()

    def _clamp_scene_pos_to_image(self, scene_pos: QPointF) -> QPointF | None:
        """Ограничивает точку границами изображения."""
        rect = self._image_scene_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return None
        return QPointF(
            min(max(scene_pos.x(), rect.left()), rect.right()),
            min(max(scene_pos.y(), rect.top()), rect.bottom()),
        )

    def _reset_measure_state(self, *, clear_items: bool) -> None:
        """Сбрасывает временное состояние инструмента линейки."""
        self._measure_start_scene_pos = None
        for attr_name in ("_measure_line_item", "_measure_text_item"):
            item = getattr(self, attr_name, None)
            if clear_items and item is not None:
                try:
                    scene = item.scene()
                    if scene is not None:
                        scene.removeItem(item)
                except RuntimeError:
                    pass
            setattr(self, attr_name, None)

    def _start_manual_measure(self, scene_pos: QPointF) -> None:
        """Начинает интерактивное измерение на canvas."""
        self._reset_measure_state(clear_items=True)
        self._measure_start_scene_pos = scene_pos
        pen = QPen(QColor(46, 226, 201))
        pen.setWidth(2)
        self._measure_line_item = self.graphics_scene.addLine(
            scene_pos.x(),
            scene_pos.y(),
            scene_pos.x(),
            scene_pos.y(),
            pen,
        )
        self._measure_line_item.setZValue(50)
        self._measure_text_item = self.graphics_scene.addText("")
        self._measure_text_item.setDefaultTextColor(QColor(46, 226, 201))
        self._measure_text_item.setFlag(
            QGraphicsItem.ItemIgnoresTransformations,
            True,
        )
        self._measure_text_item.setZValue(51)
        self._update_manual_measure(scene_pos)
        if self._calibration_pending:
            self.statusBar().showMessage(
                self._tr(
                    "status_calibration_pick_second_point",
                    "Калибровка: выберите вторую точку эталонного отрезка",
                ),
                3500,
            )
        else:
            self.statusBar().showMessage(
                self._tr(
                    "status_measure_pick_second_point",
                    "Линейка: выберите вторую точку",
                ),
                2500,
            )

    def _update_manual_measure(self, scene_pos: QPointF) -> None:
        """Обновляет визуализацию линии и подписи измерения."""
        start = self._measure_start_scene_pos
        if start is None:
            return
        if self._measure_line_item is not None:
            self._measure_line_item.setLine(
                start.x(),
                start.y(),
                scene_pos.x(),
                scene_pos.y(),
            )

        dx = scene_pos.x() - start.x()
        dy = scene_pos.y() - start.y()
        diagonal_px = float((dx ** 2 + dy ** 2) ** 0.5)
        label = f"{diagonal_px:.2f}px"
        if self.pixels_per_mm > 0:
            label += f" / {diagonal_px / self.pixels_per_mm:.2f} мм"

        if self._measure_text_item is not None:
            mid_x = (start.x() + scene_pos.x()) / 2.0
            mid_y = (start.y() + scene_pos.y()) / 2.0
            self._measure_text_item.setPlainText(label)
            self._measure_text_item.setPos(mid_x + 6.0, mid_y + 6.0)

    def _finish_manual_measure(self, scene_pos: QPointF) -> None:
        """Завершает измерение и сохраняет запись в историю."""
        start = self._measure_start_scene_pos
        if start is None:
            return

        self._update_manual_measure(scene_pos)

        width_px = int(round(abs(scene_pos.x() - start.x())))
        height_px = int(round(abs(scene_pos.y() - start.y())))
        diagonal_px = float((width_px ** 2 + height_px ** 2) ** 0.5)

        if self._calibration_pending:
            self._calibration_pending = False
            self._apply_calibration_from_measurement(diagonal_px)
            self._measure_start_scene_pos = None
            return

        width_mm = height_mm = diagonal_mm = None
        if self.pixels_per_mm > 0:
            width_mm = width_px / self.pixels_per_mm
            height_mm = height_px / self.pixels_per_mm
            diagonal_mm = diagonal_px / self.pixels_per_mm

        record = MeasurementRecord(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            source_file=self.image_storage.file_path,
            page_index=self._active_image_index,
            object_index=-1,
            width_px=width_px,
            height_px=height_px,
            diagonal_px=diagonal_px,
            pixels_per_mm=self.pixels_per_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            diagonal_mm=diagonal_mm,
        )
        self.storage_service.append_measurement(record)
        self._measure_start_scene_pos = None

        mm_suffix = ""
        if diagonal_mm is not None:
            mm_suffix = self._tr(
                "status_measure_mm_suffix",
                ", {value:.2f} мм",
            ).format(value=diagonal_mm)
        self.statusBar().showMessage(
            self._tr(
                "status_measure_result",
                "змерение: {value:.2f}px{suffix}",
            ).format(
                value=diagonal_px,
                suffix=mm_suffix,
            ),
            4000,
        )

    def _start_calibration_from_settings(self) -> None:
        """Включает режим калибровки через измерение отрезка на активной странице."""
        if not self.image_storage.images:
            self._show_warning_message(
                self._tr("calibration_title", "Калибровка"),
                self._tr(
                    "calibration_open_image_first",
                    "Сначала откройте изображение или PDF.",
                ),
            )
            return
        self._calibration_pending = True
        self._reset_measure_state(clear_items=True)
        self._set_checked_tool("measure")
        self.statusBar().showMessage(
            self._tr(
                "status_calibration_pick_segment",
                (
                    "Калибровка: отметьте две точки эталонного отрезка, "
                    "после второго клика введите длину в мм."
                ),
            ),
            6500,
        )

    def _apply_calibration_from_measurement(self, diagonal_px: float) -> None:
        """Запрашивает длину в мм и рассчитывает коэффициент калибровки px/mm."""
        if diagonal_px <= 0:
            self.statusBar().showMessage(
                self._tr(
                    "status_calibration_zero_segment",
                    "Калибровка: отрезок нулевой длины, повторите измерение.",
                ),
                3500,
            )
            return

        mm_value, ok = QInputDialog.getDouble(
            self,
            self._tr("calibration_title", "Калибровка"),
            self._tr(
                "calibration_input_prompt",
                "змерено: {pixels:.2f} px.\nВведите реальную длину отрезка в миллиметрах:",
            ).format(pixels=diagonal_px),
            10.0,
            0.0001,
            1_000_000.0,
            4,
        )
        if not ok:
            self.statusBar().showMessage(
                self._tr(
                    "status_calibration_cancelled",
                    "Калибровка отменена пользователем.",
                ),
                3000,
            )
            return
        if mm_value <= 0:
            self._show_warning_message(
                self._tr("calibration_title", "Калибровка"),
                self._tr(
                    "calibration_mm_positive",
                    "Длина в миллиметрах должна быть больше нуля.",
                ),
            )
            return

        self.pixels_per_mm = float(diagonal_px) / float(mm_value)
        self.app_state.pixels_per_mm = self.pixels_per_mm
        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        settings.setValue("pixels_per_mm", self.pixels_per_mm)
        settings.sync()
        self.statusBar().showMessage(
            self._tr(
                "status_calibration_applied",
                "Калибровка применена: {value:.4f} px/mm",
            ).format(value=self.pixels_per_mm),
            4500,
        )
        self._refresh_calibration_card()
        self._refresh_current_view()

    def eventFilter(self, watched, event):
        """Обрабатывает клики инструментов в viewport canvas."""
        if hasattr(self, "graphics_view") and watched is self.graphics_view.viewport():
            active_tool = getattr(self, "_active_tool", "select")
            if (
                event.type() == QEvent.Wheel
                and event.modifiers() & Qt.ControlModifier
            ):
                delta_y = event.angleDelta().y()
                if delta_y > 0:
                    self.zoom_in()
                    return True
                if delta_y < 0:
                    self.zoom_out()
                    return True
            if active_tool == "measure":
                if event.type() == QEvent.MouseButtonPress:
                    if event.button() == Qt.RightButton:
                        was_calibration = self._calibration_pending
                        self._calibration_pending = False
                        self._reset_measure_state(clear_items=True)
                        self.statusBar().showMessage(
                            self._tr(
                                "status_calibration_cancelled_short",
                                "Калибровка отменена.",
                            )
                            if was_calibration
                            else self._tr(
                                "status_measure_cancelled",
                                "Линейка: измерение отменено",
                            ),
                            2000,
                        )
                        return True
                    if event.button() == Qt.LeftButton:
                        raw_scene_pos = self.graphics_view.mapToScene(event.pos())
                        scene_pos = self._clamp_scene_pos_to_image(raw_scene_pos)
                        if scene_pos is None:
                            return True
                        if self._measure_start_scene_pos is None:
                            self._start_manual_measure(scene_pos)
                        else:
                            self._finish_manual_measure(scene_pos)
                        return True

                if (
                    event.type() == QEvent.MouseMove
                    and self._measure_start_scene_pos is not None
                ):
                    raw_scene_pos = self.graphics_view.mapToScene(event.pos())
                    scene_pos = self._clamp_scene_pos_to_image(raw_scene_pos)
                    if scene_pos is not None:
                        self._update_manual_measure(scene_pos)
                    return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        """Временно переключает на инструмент «Рука» при удержании Space."""
        active_tool = getattr(self, "_active_tool", "select")
        if event.key() == Qt.Key_Delete and self._delete_selected_bbox_items():
            event.accept()
            return
        if event.key() == Qt.Key_Escape and active_tool == "measure":
            was_calibration = self._calibration_pending
            self._calibration_pending = False
            self._reset_measure_state(clear_items=True)
            self.statusBar().showMessage(
                self._tr(
                    "status_calibration_cancelled_short",
                    "Калибровка отменена.",
                )
                if was_calibration
                else self._tr(
                    "status_measure_cancelled",
                    "Линейка: измерение отменено",
                ),
                2000,
            )
            event.accept()
            return
        if (
            event.key() == Qt.Key_Space
            and not event.isAutoRepeat()
            and not getattr(self, "_space_hand_active", False)
        ):
            self._space_hand_active = True
            self._tool_before_space = active_tool
            self._set_checked_tool("hand")
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Возвращает предыдущий инструмент после отпускания Space."""
        if (
            event.key() == Qt.Key_Space
            and not event.isAutoRepeat()
            and getattr(self, "_space_hand_active", False)
        ):
            self._space_hand_active = False
            self._set_checked_tool(getattr(self, "_tool_before_space", "select"))
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _delete_selected_bbox_items(self) -> bool:
        """Удаляет выбранные на сцене bbox-элементы и синхронизирует дерево/данные."""
        if not hasattr(self, "graphics_scene"):
            return False
        selected_bbox_items = [
            item
            for item in self.graphics_scene.selectedItems()
            if isinstance(item, BBoxItem)
        ]
        if not selected_bbox_items:
            return False
        removed = self._delete_objects_by_identity(
            [item.obj for item in selected_bbox_items]
        )
        if removed <= 0:
            return False
        self.statusBar().showMessage(
            self._tr("status_deleted_boxes", "Deleted boxes: {count}").format(
                count=removed
            ),
            2500,
        )
        return True

    def _delete_objects_by_identity(self, targets: Sequence[object]) -> int:
        """Удаляет объекты/части по identity и обновляет интерфейс."""
        if not targets or not self.image_storage.class_object_image:
            return 0
        target_ids = {id(target) for target in targets}
        removed_count = 0
        changed_pages: set[int] = set()

        for page_idx, objects in enumerate(self.image_storage.class_object_image):
            for obj_idx in reversed(range(len(objects))):
                obj = objects[obj_idx]
                if id(obj) in target_ids:
                    objects.pop(obj_idx)
                    removed_count += 1
                    changed_pages.add(page_idx)
                    continue
                parts = obj.image_all_class or []
                for part_idx in reversed(range(len(parts))):
                    if id(parts[part_idx]) in target_ids:
                        parts.pop(part_idx)
                        removed_count += 1
                        changed_pages.add(page_idx)

        if removed_count <= 0:
            return 0

        for page_idx in sorted(changed_pages):
            self._rebuild_tree_page(page_idx)
        self._refresh_tree_filter_classes()
        self._apply_tree_filters()
        self._refresh_statistics_panel()
        self._refresh_thumbnails_panel()
        self._refresh_detection_result_card()
        self._refresh_project_files_list()
        self.app_state.selected_item = {
            "type": "pdf",
            "index": self._active_image_index,
        }
        self.display_image_with_boxes(self._active_image_index)
        return removed_count

    def closeEvent(self, event):
        """Сохраняет layout и геометрию окна при закрытии."""
        if self._find_all_worker is not None and self._find_all_worker.isRunning():
            self._find_all_worker.cancel()
            self._find_all_worker.wait(1200)
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            self.worker.wait(800)
        if (
            getattr(self, "_model_load_worker", None) is not None
            and self._model_load_worker.isRunning()
        ):
            self._model_load_worker.wait(800)
        if self._find_all_progress_dialog is not None:
            try:
                self._find_all_progress_dialog.close()
            except RuntimeError:
                pass
            self._find_all_progress_dialog = None

        self._save_layout_settings()
        super().closeEvent(event)

    def _start_model_loading(self) -> None:
        """Запускает загрузку модели детекции в фоновом потоке."""
        self.statusBar().showMessage(
            tr(
                self.current_language,
                "status_model_loading",
                "Загрузка модели детекции...",
            )
        )
        self._update_header_badges()
        self._model_load_worker = ModelLoadWorker(self.weights_path)
        self._model_load_worker.model_loaded.connect(self._on_model_loaded)
        self._model_load_worker.model_error.connect(self._on_model_error)
        self._model_load_worker.finished.connect(self._on_model_load_finished)
        self._model_load_worker.start()

    def _on_model_loaded(self, model) -> None:
        """Обработка успешной загрузки модели."""
        self.model = model
        self._set_detection_actions_enabled(True)
        self.statusBar().showMessage(
            tr(self.current_language, "status_model_ready", "Модель загружена"),
            3000,
        )
        logger.info("Модель детекции успешно загружена")
        self._update_header_badges()
        self._append_activity(
            self._tr("activity_model_ready", "model-ready"),
            self.weights_path,
        )

    def _on_model_error(self, error_msg: str) -> None:
        """Обработка ошибки загрузки модели."""
        self._show_error_message(
            "Ошибка загрузки модели",
            (
                f"Не удалось загрузить модель:\n{self.weights_path}\n\n"
                f"{error_msg}"
            ),
        )
        self.statusBar().showMessage(
            tr(self.current_language, "status_model_error", "Ошибка загрузки модели"),
            5000,
        )
        self._update_header_badges()
        self._append_activity(
            self._tr("activity_model_error", "model-error"),
            error_msg,
        )

    def _on_model_load_finished(self) -> None:
        """Скрывает индикатор загрузки после завершения worker."""
        if self.model is None:
            self._set_detection_actions_enabled(False)
            self.statusBar().showMessage(
                tr(
                    self.current_language,
                    "status_model_missing",
                    "Модель не загружена",
                )
            )
        self._model_load_worker = None
        self._update_header_badges()

    def create_menu(self):
        """Создаёт верхнее меню и привязывает существующие QAction."""
        menu_bar = self.menuBar()
        self.file_menu = menu_bar.addMenu(
            tr(self.current_language, "menu_file", "Файл")
        )
        self.file_menu.addAction(self.action_open)
        self.file_menu.addAction(self.action_add)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.action_save)
        self.file_menu.addAction(self.action_export)
        self.file_menu.addAction(self.action_report)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.action_settings)

        self.view_menu = menu_bar.addMenu(
            tr(self.current_language, "menu_view", "Вид")
        )
        self.view_menu.addAction(self.action_zoom_in)
        self.view_menu.addAction(self.action_zoom_out)
        self.view_menu.addAction(self.action_fit)
        self.view_menu.addAction(self.action_rotate)

        self.tools_menu = menu_bar.addMenu(
            tr(self.current_language, "menu_tools", "Анализ")
        )
        self.tools_menu.addAction(self.action_find)
        self.tools_menu.addAction(self.action_find_all)
        self.tools_menu.addAction(self.action_classify)
        self.tools_menu.addSeparator()

        self.action_measurements_history = QAction(
            tr(
                self.current_language,
                "menu_measurements_history",
                "стория измерений",
            ),
            self,
        )
        self._set_action_hints(
            self.action_measurements_history,
            tr(
                self.current_language,
                "menu_measurements_history",
                "стория измерений",
            ),
        )
        self.action_measurements_history.triggered.connect(
            self.show_measurement_history
        )
        self.tools_menu.addAction(self.action_measurements_history)

        self.action_measurements_export = QAction(
            tr(
                self.current_language,
                "menu_measurements_export",
                "Экспорт измерений",
            ),
            self,
        )
        self._set_action_hints(
            self.action_measurements_export,
            tr(
                self.current_language,
                "menu_measurements_export",
                "Экспорт измерений",
            ),
        )
        self.action_measurements_export.triggered.connect(
            self.export_measurement_history
        )
        self.tools_menu.addAction(self.action_measurements_export)

        self.action_clear_cache = QAction(
            tr(self.current_language, "menu_clear_cache", "Очистить кэш"),
            self,
        )
        self._set_action_hints(
            self.action_clear_cache,
            tr(self.current_language, "menu_clear_cache", "Очистить кэш"),
        )
        self.action_clear_cache.triggered.connect(self.clear_local_cache)
        self.tools_menu.addAction(self.action_clear_cache)

    def _set_uniform_button_style(self, toolbar: QToolBar) -> None:
        """Применяет единый размер hit-area к кнопкам тулбара."""
        toolbar.setStyleSheet(
            "QToolButton {"
            f"min-width: {self.ui_metrics.tool_button_size}px;"
            f"min-height: {self.ui_metrics.tool_button_size}px;"
            f"max-width: {self.ui_metrics.tool_button_size}px;"
            f"max-height: {self.ui_metrics.tool_button_size}px;"
            "}"
        )

    def _create_toolbar_action(
        self,
        *,
        attr_name: str,
        icon_name: str,
        text: str,
        hint: str,
        handler: Callable[[], None],
        shortcut: str | None = None,
        fallback_icon: QStyle.StandardPixmap | None = None,
    ) -> QAction:
        """Создаёт ``QAction`` тулбара и сохраняет его в атрибут класса."""
        action = QAction(
            self.icon_manager.get_icon(
                icon_name,
                fallback_standard_icon=fallback_icon,
            ),
            text,
            self,
        )
        self._set_action_hints(action, hint)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(handler)
        setattr(self, attr_name, action)
        return action

    def _register_toolbar_actions(self) -> None:
        """Регистрирует все действия верхнего тулбара."""
        for (
            attr_name,
            icon_name,
            text,
            hint,
            handler_name,
            shortcut,
            fallback_icon,
        ) in TOOLBAR_ACTION_SPECS:
            self._create_toolbar_action(
                attr_name=attr_name,
                icon_name=icon_name,
                text=text,
                hint=hint,
                handler=getattr(self, handler_name),
                shortcut=shortcut,
                fallback_icon=fallback_icon,
            )

    def _set_action_hints(self, action: QAction, text: str) -> None:
        """Назначает tooltip и status tip для действия."""
        action.setToolTip(text)
        action.setStatusTip(text)

    def _show_info_message(self, title: str, text: str) -> None:
        """Показывает информационное сообщение пользователю."""
        QMessageBox.information(self, title, text)

    def _show_warning_message(self, title: str, text: str) -> None:
        """Показывает предупреждение пользователю."""
        QMessageBox.warning(self, title, text)

    def _show_error_message(self, title: str, text: str) -> None:
        """Показывает сообщение об ошибке пользователю."""
        QMessageBox.critical(self, title, text)

    def _build_detection_cache_key(self, page_index: int) -> str | None:
        """Формирует ключ кэша детекции для текущей страницы."""
        if page_index >= len(self.image_storage.images):
            return None
        image = self.image_storage.images[page_index]
        if not isinstance(image, np.ndarray):
            return None
        sample = image[::16, ::16]
        checksum = int(sample.astype(np.uint64).sum())
        return self.storage_service.build_detection_cache_key(
            source_file=self.image_storage.file_path,
            page_index=page_index,
            image_shape=image.shape,
            image_checksum=checksum,
            detect_weights_path=self.weights_path,
            conf_threshold=self.detection_confidence_threshold,
            iou_threshold=self.detection_iou_threshold,
        )

    def _build_classification_cache_key(
        self,
        page_index: int,
        object_index: int,
        obj: ObjectImage,
    ) -> str:
        """Формирует ключ кэша классификации выбранного сеянца."""
        crop_checksum = 0
        if obj.image and isinstance(obj.image[0], np.ndarray):
            crop_sample = obj.image[0][::8, ::8]
            crop_checksum = int(crop_sample.astype(np.uint64).sum())
        return self.storage_service.build_classification_cache_key(
            source_file=self.image_storage.file_path,
            page_index=page_index,
            object_index=object_index,
            object_bbox=obj.bbox,
            rotation_k=int(getattr(obj, "rotation_k", 0)),
            crop_checksum=crop_checksum,
            classify_weights_path=self.classify_weights_path,
        )

    def _apply_detection_objects(
        self,
        index: int,
        objects: list[ObjectImage],
        *,
        from_cache: bool = False,
    ) -> None:
        """Применяет список детекций к дереву и canvas."""
        if self.tree_widget.topLevelItem(index) is None:
            logger.warning(
                "apply_detection_objects: parent tree item not found for index %s",
                index,
            )
            return

        self.image_storage.class_object_image[index] = objects
        self._rebuild_tree_page(index)

        self.display_image_with_boxes(index)
        self.update_left_info({"type": "pdf", "index": index})
        self._refresh_tree_filter_classes()
        self._apply_tree_filters()
        self._refresh_statistics_panel()
        self._refresh_thumbnails_panel()
        self._refresh_project_files_list()
        self._set_active_file_row(index)
        self._refresh_detection_result_card()
        self._append_activity(
            self._tr("activity_process_image", "process-image"),
            f"{self._tr('results_found', 'Found: {count}').format(count=len(objects))}",
        )

        if from_cache:
            self.statusBar().showMessage(
                self._tr(
                    "status_detection_loaded_from_cache",
                    "Детекция загружена из кэша",
                ),
                3000,
            )

    def _classify_single_object(
        self,
        page_index: int,
        object_index: int,
        seeding_obj: ObjectImage,
    ) -> list:
        """Классифицирует один сеянец с поддержкой загрузки/сохранения кэша."""
        cache_key = self._build_classification_cache_key(
            page_index,
            object_index,
            seeding_obj,
        )
        if self.use_cache:
            cached_parts = self.storage_service.load_classification_parts(cache_key)
            if cached_parts is not None:
                seeding_obj.image_all_class = cached_parts
                self.image_service.sync_crops_and_parts(self.image_storage)
                return cached_parts

        results = self.classify_model(seeding_obj.image[0])
        parts = self.app_controller.run_classification_for_selection(
            self.app_state,
            page_index,
            object_index,
            results,
        )
        if self.use_cache:
            self.storage_service.save_classification_parts(cache_key, parts)
        return parts

    def export_results(self) -> None:
        """Экспортирует результаты в выбранные форматы."""
        if not self.image_storage.images:
            self._show_warning_message(
                "Нет данных",
                "Сначала откройте изображения и выполните анализ.",
            )
            return

        dialog = ExportDialog(
            self,
            default_dir=self._default_report_dir(),
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        options = dialog.options
        if not any(options.values()):
            self._show_warning_message(
                "Нет форматов",
                "Выберите хотя бы один формат экспорта.",
            )
            return

        output_dir = dialog.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        exported: list[str] = []
        try:
            if options["json"]:
                path = self.export_service.export_json(
                    self.image_storage,
                    output_dir,
                )
                exported.append(str(path))
            if options["csv"]:
                path = self.export_service.export_csv(
                    self.image_storage,
                    output_dir,
                )
                exported.append(str(path))
            if options["coco"]:
                path = self.export_service.export_coco(
                    self.image_storage,
                    output_dir,
                )
                exported.append(str(path))
            if options["yolo"]:
                path = self.export_service.export_yolo(
                    self.image_storage,
                    output_dir,
                )
                exported.append(str(path))
            if options["annotated"]:
                path = self.export_service.export_annotated_images(
                    self.image_storage,
                    output_dir,
                )
                exported.append(str(path))
        except Exception as error:
            logger.exception("Ошибка экспорта: %s", error)
            self._show_error_message(
                "Ошибка экспорта",
                f"Не удалось выполнить экспорт:\n{error}",
            )
            return

        self._show_info_message(
            "Экспорт завершён",
            "Сохранено:\n" + "\n".join(exported),
        )

    def _select_input_files(self, caption: str) -> list[str]:
        """Открывает диалог выбора изображений/PDF и возвращает список путей."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            caption,
            "",
            INPUT_FILE_FILTER,
        )
        return files

    def _reset_project_data(self) -> None:
        """Сбрасывает текущие данные проекта перед новым открытием файлов."""
        self._pending_classify_after_find_all = False
        self.image_storage = OriginalImage()
        self.app_state.image_storage = self.image_storage
        self.app_state.selected_item = None
        self._reset_measure_state(clear_items=True)
        self.tree_widget.clear()
        self._refresh_tree_filter_classes()
        self._apply_tree_filters()
        self._refresh_statistics_panel()
        self._refresh_thumbnails_panel()
        self._show_empty_state()
        self._refresh_project_files_list()
        self._refresh_detection_result_card()

    def _append_files_to_project(self, files: Sequence[str]) -> None:
        """Добавляет выбранные файлы в текущий проект."""
        for file_path in files:
            if file_path.lower().endswith(".pdf"):
                self._add_pdf(file_path)
            else:
                self._add_image(file_path)

    def _ensure_detection_storage(self) -> None:
        """Гарантирует наличие контейнера объектов для каждой страницы."""
        if self.image_storage.class_object_image is None:
            self.image_storage.class_object_image = [
                [] for _ in self.image_storage.images
            ]

    def _ensure_detection_model(self, message: str) -> bool:
        """Проверяет, что модель детекции загружена."""
        if self.model is not None:
            return True
        self._show_warning_message(
            self._tr("status_model_missing", "Модель не загружена"),
            message,
        )
        return False

    def _get_current_image_for_detection(self) -> np.ndarray | None:
        """Возвращает активное изображение для детекции или ``None``."""
        if not self.image_storage.images:
            logger.warning("Нет изображений для обработки")
            return None
        if self._active_image_index >= len(self.image_storage.images):
            self._active_image_index = 0
            self.app_state.active_image_index = 0
        image = self.image_storage.images[self._active_image_index]
        if image is None:
            logger.warning("Текущее изображение пустое")
            return None
        return image

    def _has_any_detected_seedlings(self) -> bool:
        """Проверяет, есть ли в проекте хотя бы одна детекция сеянца."""
        if not self.image_storage.class_object_image:
            return False
        return any(bool(page_objects) for page_objects in self.image_storage.class_object_image)

    def process_all_segmentation_classification(self) -> None:
        """Выполняет сегментацию/классификацию по всем фото.

        Если детекция не выполнена, сначала запускает пакетную YOLO-детекцию,
        после её завершения автоматически стартует классификация.
        """
        if not self.image_storage.images:
            self._show_warning_message(
                self._tr("common_no_data_title", "No data"),
                self._tr(
                    "stats_export_no_data",
                    "Open images first to build statistics.",
                ),
            )
            return

        if self._has_any_detected_seedlings():
            self.classify()
            return

        self._pending_classify_after_find_all = True
        self.find_all_seedlings()
        if (
            self._pending_classify_after_find_all
            and self._find_all_worker is None
            and not self.progress_bar.isVisible()
        ):
            # find_all_seedlings завершился раньше старта worker (ошибка/нет модели)
            self._pending_classify_after_find_all = False


    def create_left_panel(self):
        """Создаёт панель свойств выбранного элемента в левой колонке."""
        self.left_panel = QGroupBox("")
        self.left_panel.setMinimumWidth(self.ui_metrics.panel_min_width)
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
                '<p style="margin:0;">Выберите элемент в дереве</p>'
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
        self.app_state.active_image_index = new_idx
        self.display_image_with_boxes(new_idx)
        self.thumbnails_panel.set_active_index(new_idx)

        item = self.tree_widget.topLevelItem(new_idx)
        if item:
            self.tree_widget.setCurrentItem(item)

        self.update_left_info({"type": "pdf", "index": new_idx})
        self._set_active_file_row(new_idx)
        self._refresh_detection_result_card()

    def open_settings(self):
        """Открывает диалог настроек и применяет изменённые параметры."""
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            if dialog.calibration_requested:
                self._start_calibration_from_settings()
                return
            if not dialog.save_settings():
                return
            previous_detect_weights = self.weights_path
            previous_classify_weights = self.classify_weights_path
            settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
            self._load_runtime_settings(settings)
            self.app_state.report_dir = settings.value("report_dir", "", type=str)
            self._apply_theme(dialog.selected_theme)
            self._apply_language(dialog.selected_language)
            if self.weights_path != previous_detect_weights:
                self.model = None
                self._set_detection_actions_enabled(False)
                self._start_model_loading()
            if self.classify_weights_path != previous_classify_weights:
                self.classify_model = None
            self.display_image_with_boxes(self._active_image_index)
            self._populate_model_selector()
            self._update_header_badges()
            self._refresh_calibration_card()
            self._append_activity(
                self._tr("activity_settings_updated", "settings-updated"),
                "",
            )
            logger.info("Настройки обновлены и применены.")

    def _setup_shortcuts(self):
        """Настраивает все горячие клавиши."""
        QShortcut(QKeySequence(Qt.Key_Left),  self, lambda: self.switch_image(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.switch_image(1))

        QShortcut(QKeySequence("R"),            self, self.rotate_image)
        QShortcut(QKeySequence("D"),            self, self.find_seedlings)

        QShortcut(QKeySequence("V"), self, lambda: self._set_checked_tool("select"))
        QShortcut(QKeySequence("H"), self, lambda: self._set_checked_tool("hand"))
        QShortcut(
            QKeySequence("M"),
            self,
            lambda: self._set_checked_tool("measure"),
        )

        QShortcut(QKeySequence("Ctrl+="),       self, self.zoom_in)

    def create_central_widget(self):
        """Создаёт центральный canvas и пустое состояние проекта."""
        self.canvas_host = QWidget(self)
        self.canvas_host.setObjectName("canvasHost")
        host_layout = QVBoxLayout(self.canvas_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)

        self.canvas_stack = QStackedWidget(self.canvas_host)
        self.canvas_stack.setObjectName("canvasStack")
        host_layout.addWidget(self.canvas_stack)

        self.scroll_area = DraggableScrollArea()
        self.scroll_area.setObjectName("centralScroll")
        self.scroll_area.setWidgetResizable(True)

        self.graphics_scene = QGraphicsScene(self)
        self.graphics_scene.setBackgroundBrush(
            QColor(VIEW_BACKGROUND_R, VIEW_BACKGROUND_G, VIEW_BACKGROUND_B)
        )
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setObjectName("centralView")
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        self.graphics_view.viewport().installEventFilter(self)

        self.image_item = QGraphicsPixmapItem()
        self.graphics_scene.addItem(self.image_item)

        self.rect_items = {}

        self.scroll_area.setWidget(self.graphics_view)
        self.empty_state_widget = self._create_empty_state_widget()
        self.canvas_stack.addWidget(self.empty_state_widget)
        self.canvas_stack.addWidget(self.scroll_area)
        self.canvas_stack.setCurrentWidget(self.empty_state_widget)

    def _create_empty_state_widget(self) -> QWidget:
        """Создаёт экран пустого состояния для canvas."""
        widget = QWidget(self)
        widget.setObjectName("emptyState")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(
            self.ui_metrics.padding_l,
            self.ui_metrics.padding_l,
            self.ui_metrics.padding_l,
            self.ui_metrics.padding_l,
        )
        layout.addStretch()

        self.empty_state_title = QLabel(
            "Перетащите файл сюда или откройте через кнопку"
        )
        self.empty_state_title.setObjectName("emptyStateTitle")
        self.empty_state_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.empty_state_title)

        self.empty_state_hint = QLabel("Поддерживаются изображения и PDF-документы")
        self.empty_state_hint.setObjectName("emptyStateHint")
        self.empty_state_hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.empty_state_hint)

        self.empty_open_button = QPushButton("Открыть файл")
        self.empty_open_button.setObjectName("emptyOpenButton")
        self.empty_open_button.clicked.connect(self.open_image)
        self.empty_open_button.setFixedWidth(180)
        self.empty_open_button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.empty_open_button, alignment=Qt.AlignHCenter)
        layout.addStretch()
        return widget

    def _show_empty_state(self) -> None:
        """Показывает пустое состояние центральной области."""
        if hasattr(self, "canvas_stack") and hasattr(self, "empty_state_widget"):
            self.canvas_stack.setCurrentWidget(self.empty_state_widget)

    def _show_canvas(self) -> None:
        """Показывает canvas с изображением."""
        if hasattr(self, "canvas_stack") and hasattr(self, "scroll_area"):
            self.canvas_stack.setCurrentWidget(self.scroll_area)

    def create_right_panel(self):
        """Создаёт правую панель с деревом слоёв."""
        self.right_panel = QGroupBox("")
        self.right_panel.setObjectName("layersGroup")
        self.right_panel.setMinimumWidth(PANEL_LAYERS_MIN_WIDTH)
        self.right_panel.setMaximumWidth(PANEL_LAYERS_MAX_WIDTH)

        layout = QVBoxLayout()
        left, top, right, bottom = PANEL_LAYOUT_MARGINS
        layout.setContentsMargins(left, top, right + 10, bottom)

        filters_layout = QHBoxLayout()
        filters_layout.setSpacing(6)
        self.tree_search_edit = QLineEdit(self)
        self.tree_search_edit.setPlaceholderText(
            tr(self.current_language, "tree_search_placeholder", "Поиск...")
        )
        self.tree_search_edit.textChanged.connect(self._apply_tree_filters)
        filters_layout.addWidget(self.tree_search_edit)

        self.tree_class_filter = QComboBox(self)
        self.tree_class_filter.setMinimumWidth(84)
        self.tree_class_filter.addItem(
            tr(self.current_language, "tree_filter_all", "Все"),
            "all",
        )
        self.tree_class_filter.currentIndexChanged.connect(
            self._apply_tree_filters
        )
        filters_layout.addWidget(self.tree_class_filter)

        self.tree_conf_filter = QDoubleSpinBox(self)
        self.tree_conf_filter.setMinimumWidth(90)
        self.tree_conf_filter.setRange(0.0, 1.0)
        self.tree_conf_filter.setSingleStep(0.05)
        self.tree_conf_filter.setDecimals(2)
        self.tree_conf_filter.setPrefix(
            tr(self.current_language, "tree_conf_prefix", "Conf≥")
        )
        self.tree_conf_filter.valueChanged.connect(self._apply_tree_filters)
        filters_layout.addWidget(self.tree_conf_filter)
        layout.addLayout(filters_layout)

        self.tree_widget = LayerTreeWidget()
        self.tree_widget.setObjectName("layerTree")
        self.tree_widget.setMinimumHeight(360)
        layout.addWidget(self.tree_widget, 1)
        self.right_panel.setLayout(layout)

    def _refresh_tree_filter_classes(self) -> None:
        """Перестраивает список классов фильтра на основе текущего дерева."""
        if not hasattr(self, "tree_class_filter"):
            return
        classes = {
            "all": tr(self.current_language, "tree_filter_all", "Все"),
            "seeding": tr(self.current_language, "class_seedling", "Сеянец"),
        }
        for i in range(self.tree_widget.topLevelItemCount()):
            root = self.tree_widget.topLevelItem(i)
            for child_idx in range(root.childCount()):
                child = root.child(child_idx)
                classes.setdefault("seeding", "Seeding")
                for cls_idx in range(child.childCount()):
                    class_item = child.child(cls_idx)
                    key = class_item.text(0).strip().lower()
                    if key:
                        classes.setdefault(key, class_item.text(0).strip())

        current = self.tree_class_filter.currentData() or "all"
        self.tree_class_filter.blockSignals(True)
        self.tree_class_filter.clear()
        for key, label in classes.items():
            self.tree_class_filter.addItem(label, key)
        idx = self.tree_class_filter.findData(current)
        if idx >= 0:
            self.tree_class_filter.setCurrentIndex(idx)
        self.tree_class_filter.blockSignals(False)

    def _apply_tree_filters(self) -> None:
        """Применяет текущие фильтры поиска/класса/уверенности к дереву."""
        search_text = (
            self.tree_search_edit.text() if hasattr(self, "tree_search_edit") else ""
        )
        class_filter = (
            self.tree_class_filter.currentData()
            if hasattr(self, "tree_class_filter")
            else "all"
        )
        min_conf = (
            float(self.tree_conf_filter.value())
            if hasattr(self, "tree_conf_filter")
            else 0.0
        )
        self.tree_widget.apply_filter(
            search_text=search_text,
            class_filter=str(class_filter or "all"),
            min_confidence=min_conf,
        )

    @staticmethod
    def _normalize_part_key(name: str | None) -> str:
        """Нормализует имя части к каноническому ключу."""
        value = (name or "").strip().lower()
        if not value:
            return "other"
        if value in {"соцветие", "цветок", "flower", "inflorescence"}:
            return "inflorescence"
        if value in {"стебель", "stem"}:
            return "stem"
        if value in {"корень", "root"}:
            return "root"
        if value in {"сеянец", "seeding", "seedling"}:
            return "seeding"
        return value

    def _display_part_name(self, class_name: str | None) -> str:
        """Возвращает локализованное имя части для отображения в UI."""
        key = self._normalize_part_key(class_name)
        if key == "inflorescence":
            return tr(self.current_language, "class_inflorescence", "Соцветие")
        if key == "stem":
            return tr(self.current_language, "class_stem", "Стебель")
        if key == "root":
            return tr(self.current_language, "class_root", "Корень")
        if key == "seeding":
            return tr(self.current_language, "class_seedling", "Сеянец")
        value = (class_name or "").strip()
        if value:
            return value
        return tr(self.current_language, "class_other", "Другое")

    @staticmethod
    def _is_manual_part(part: AllClassImage) -> bool:
        """Определяет, добавлена ли часть вручную (без bbox и реального кропа)."""
        if part.bbox is not None:
            return False
        if isinstance(part.image, np.ndarray):
            return part.image.size == 0
        return False

    def _build_part_description(self, part: AllClassImage) -> str:
        """Строит описание части для дерева слоёв."""
        if self._is_manual_part(part):
            return tr(self.current_language, "manual_added", "Добавлено вручную")
        short_conf = self._tr("confidence_short", "Conf")
        return (
            f"{short_conf}: "
            f"{part.confidence * 100:.0f}%"
        )

    def _rebuild_tree_page(self, page_index: int) -> None:
        """Пересобирает узлы сеянцев/частей для одной страницы дерева."""
        if not self.image_storage.class_object_image:
            return
        if page_index >= len(self.image_storage.class_object_image):
            return

        root_item = self.tree_widget.topLevelItem(page_index)
        if root_item is None:
            return

        for item_idx in reversed(range(root_item.childCount())):
            root_item.takeChild(item_idx)

        objects = self.image_storage.class_object_image[page_index]
        short_conf = self._tr("confidence_short", "Conf")
        for obj_idx, obj in enumerate(objects):
            crop_preview = obj.image[0] if obj.image else np.empty((0, 0, 3))
            child_item = self.tree_widget.add_child_item(
                root_item,
                f"Seeding{obj_idx + 1}",
                (
                    f"{short_conf}: "
                    f"{obj.confidence * 100:.0f}%"
                ),
                page_index,
                obj_idx,
                "seeding",
                crop_preview,
                confidence=obj.confidence,
            )
            for class_idx, part in enumerate(obj.image_all_class or []):
                self.tree_widget.add_class_item(
                    child_item,
                    self._display_part_name(part.class_name),
                    self._build_part_description(part),
                    page_index,
                    obj_idx,
                    class_idx,
                    confidence=part.confidence,
                )

    def _on_tree_measure_requested(self, parent_idx: int, seed_idx: int) -> None:
        """Показывает базовое измерение выбранного сеянца в пикселях."""
        if not self.image_storage.class_object_image:
            return
        if parent_idx >= len(self.image_storage.class_object_image):
            return
        objects = self.image_storage.class_object_image[parent_idx]
        if seed_idx >= len(objects):
            return
        obj = objects[seed_idx]
        if not obj.bbox:
            return
        x1, y1, x2, y2 = obj.bbox
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        diagonal = float((width ** 2 + height ** 2) ** 0.5)

        width_mm = height_mm = diagonal_mm = None
        if self.pixels_per_mm > 0:
            width_mm = width / self.pixels_per_mm
            height_mm = height / self.pixels_per_mm
            diagonal_mm = diagonal / self.pixels_per_mm

        record = MeasurementRecord(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            source_file=self.image_storage.file_path,
            page_index=parent_idx,
            object_index=seed_idx,
            width_px=width,
            height_px=height,
            diagonal_px=diagonal,
            pixels_per_mm=self.pixels_per_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            diagonal_mm=diagonal_mm,
        )
        self.storage_service.append_measurement(record)

        mm_block = ""
        if diagonal_mm is not None:
            mm_block = (
                self._tr(
                    "measure_object_mm_block",
                    "\nШирина: {width_mm:.2f} мм\nВысота: {height_mm:.2f} мм\nДиагональ: {diagonal_mm:.2f} мм",
                ).format(
                    width_mm=width_mm,
                    height_mm=height_mm,
                    diagonal_mm=diagonal_mm,
                )
            )
        self._show_info_message(
            self._tr("measure_object_title", "змерение объекта"),
            self._tr(
                "measure_object_text",
                "Сеянец {index}\nШирина: {width}px\nВысота: {height}px\nДиагональ: {diagonal:.2f}px{mm_block}",
            ).format(
                index=seed_idx + 1,
                width=width,
                height=height,
                diagonal=diagonal,
                mm_block=mm_block,
            ),
        )

    def _on_tree_classify_requested(self, parent_idx: int, seed_idx: int) -> None:
        """Классифицирует один выбранный сеянец из контекстного меню."""
        if not self.image_storage.class_object_image:
            return
        if parent_idx >= len(self.image_storage.class_object_image):
            return
        page_objects = self.image_storage.class_object_image[parent_idx]
        if seed_idx >= len(page_objects):
            return
        seeding_obj = page_objects[seed_idx]
        if not seeding_obj.image:
            return

        if self.classify_model is None:
            try:
                self.classify_model = YOLO(self.classify_weights_path)
            except Exception as e:
                self._show_error_message(
                    self._tr("status_model_error", "Ошибка загрузки модели"),
                    self._tr(
                        "classify_model_load_failed",
                        "Не удалось загрузить модель классификации:\n{error}",
                    ).format(error=e),
                )
                logger.exception("Ошибка загрузки классификатора: %s", e)
                return

        parts = self._classify_single_object(
            parent_idx,
            seed_idx,
            seeding_obj,
        )
        _ = parts
        self._rebuild_tree_page(parent_idx)
        self._refresh_tree_filter_classes()
        self._apply_tree_filters()
        self.display_seeding_with_boxes(parent_idx, seed_idx)

    def _on_tree_add_part_requested(self, parent_idx: int, seed_idx: int) -> None:
        """Добавляет вручную часть к выбранному сеянцу."""
        if not self.image_storage.class_object_image:
            return
        if parent_idx >= len(self.image_storage.class_object_image):
            return
        page_objects = self.image_storage.class_object_image[parent_idx]
        if seed_idx >= len(page_objects):
            return

        options = [
            (
                tr(self.current_language, "class_inflorescence", "Соцветие"),
                "inflorescence",
            ),
            (tr(self.current_language, "class_stem", "Стебель"), "stem"),
            (tr(self.current_language, "class_root", "Корень"), "root"),
            (tr(self.current_language, "class_other", "Другое"), "other"),
        ]
        labels = [label for label, _ in options]
        selected, ok = QInputDialog.getItem(
            self,
            tr(self.current_language, "add_part_title", "Добавить часть"),
            tr(
                self.current_language,
                "add_part_prompt",
                "Выберите тип части для добавления:",
            ),
            labels,
            0,
            False,
        )
        if not ok or not selected:
            return

        selected_key = dict(options).get(selected, "other")
        obj = page_objects[seed_idx]
        if obj.image_all_class is None:
            obj.image_all_class = []
        crop_image = (
            obj.image[0]
            if obj.image and isinstance(obj.image[0], np.ndarray)
            else None
        )
        if crop_image is not None and crop_image.size > 0:
            crop_h, crop_w = crop_image.shape[:2]
        elif obj.bbox:
            crop_w = max(1, obj.bbox[2] - obj.bbox[0])
            crop_h = max(1, obj.bbox[3] - obj.bbox[1])
            crop_image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        else:
            crop_w = crop_h = 80
            crop_image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        bx1 = max(0, int(crop_w * 0.25))
        by1 = max(0, int(crop_h * 0.25))
        bx2 = min(crop_w, max(bx1 + 1, int(crop_w * 0.75)))
        by2 = min(crop_h, max(by1 + 1, int(crop_h * 0.75)))
        part_crop = crop_image[by1:by2, bx1:bx2].copy()
        obj.image_all_class.append(
            AllClassImage(
                class_name=selected_key,
                confidence=1.0,
                image=part_crop,
                bbox=(bx1, by1, bx2, by2),
            )
        )

        self._rebuild_tree_page(parent_idx)
        self._refresh_tree_filter_classes()
        self._apply_tree_filters()
        self.display_seeding_with_boxes(parent_idx, seed_idx)
        self._refresh_statistics_panel()
        self.statusBar().showMessage(
            tr(self.current_language, "manual_added", "Добавлено вручную"),
            2500,
        )

    def _on_tree_add_seedling_requested(self, page_idx: int) -> None:
        """Добавляет вручную сеянец на страницу с дефолтным bbox."""
        if page_idx >= len(self.image_storage.images):
            self._show_warning_message(
                tr(self.current_language, "add_part_title", "Добавить часть"),
                tr(
                    self.current_language,
                    "add_seedling_error",
                    "Невозможно добавить сеянец: нет активного изображения.",
                ),
            )
            return
        image = self.image_storage.images[page_idx]
        if not isinstance(image, np.ndarray):
            self._show_warning_message(
                tr(self.current_language, "add_part_title", "Добавить часть"),
                tr(
                    self.current_language,
                    "add_seedling_error",
                    "Невозможно добавить сеянец: нет активного изображения.",
                ),
            )
            return

        self._ensure_detection_storage()
        height, width = image.shape[:2]
        x1 = max(0, int(width * 0.3))
        y1 = max(0, int(height * 0.3))
        x2 = min(width, max(x1 + 1, int(width * 0.7)))
        y2 = min(height, max(y1 + 1, int(height * 0.7)))
        bbox = (x1, y1, x2, y2)
        crop = image[y1:y2, x1:x2].copy()
        self.image_storage.class_object_image[page_idx].append(
            ObjectImage(
                class_name="seeding",
                confidence=1.0,
                image=[crop],
                image_all_class=[],
                bbox=bbox,
                rotation_k=0,
            )
        )

        self._rebuild_tree_page(page_idx)
        self._refresh_tree_filter_classes()
        self._apply_tree_filters()
        self._active_image_index = page_idx
        self.app_state.active_image_index = page_idx
        self.display_image_with_boxes(page_idx)
        self._refresh_statistics_panel()
        self.statusBar().showMessage(
            tr(self.current_language, "manual_seedling_added", "Сеянец добавлен вручную"),
            3000,
        )

    def _on_tree_delete_requested(self, payload: dict) -> None:
        """Удаляет выбранный объект/класс из структуры проекта."""
        item_type = payload.get("type")
        if item_type == "seeding":
            parent_idx = int(payload["parent_index"])
            seed_idx = int(payload["index"])
            if (
                self.image_storage.class_object_image
                and parent_idx < len(self.image_storage.class_object_image)
            ):
                objects = self.image_storage.class_object_image[parent_idx]
                if seed_idx < len(objects):
                    objects.pop(seed_idx)
                    self._rebuild_tree_page(parent_idx)
                    self.display_image_with_boxes(parent_idx)
        elif item_type == "class":
            parent_idx = int(payload["parent_index"])
            seed_idx = int(payload["seeding_index"])
            class_idx = int(payload["class_index"])
            if (
                self.image_storage.class_object_image
                and parent_idx < len(self.image_storage.class_object_image)
            ):
                objects = self.image_storage.class_object_image[parent_idx]
                if seed_idx < len(objects):
                    obj = objects[seed_idx]
                    if obj.image_all_class and class_idx < len(obj.image_all_class):
                        obj.image_all_class.pop(class_idx)
                        self._rebuild_tree_page(parent_idx)
                        self.display_seeding_with_boxes(parent_idx, seed_idx)

        self._refresh_statistics_panel()
        self._refresh_tree_filter_classes()
        self._apply_tree_filters()



    def _has_classified_parts(self) -> bool:
        """Проверяет, есть ли в проекте хотя бы один классифицированный объект."""
        if not self.image_storage.class_object_image:
            return False
        for objects in self.image_storage.class_object_image:
            for obj in objects:
                if obj.image_all_class:
                    return True
        return False

    def on_tree_item_clicked(self, item, column):
        """Обрабатывает выбор узла дерева и синхронизирует активный просмотр."""
        _ = column
        item_data = item.data(0, Qt.UserRole)
        if item_data:
            if item_data["type"] in ("original", "pdf"):
                idx = item_data["index"]
                self._active_image_index = idx
                self.app_state.active_image_index = idx
                self.display_image_with_boxes(idx)
            elif item_data["type"] == "seeding":
                parent_idx = item_data["parent_index"]
                seed_idx = item_data["index"]
                self._active_image_index = parent_idx
                self.app_state.active_image_index = parent_idx
                self.display_seeding_with_boxes(parent_idx, seed_idx)
            elif item_data["type"] == "class":
                parent_idx = item_data["parent_index"]
                seed_idx = item_data["seeding_index"]
                class_idx = item_data["class_index"]
                self._active_image_index = parent_idx
                self.app_state.active_image_index = parent_idx
                self.display_class_image(parent_idx, seed_idx, class_idx)

            self.app_state.selected_item = item_data
            self.update_left_info(item_data)
            self.thumbnails_panel.set_active_index(self._active_image_index)
            self._set_active_file_row(self._active_image_index)
            self._refresh_detection_result_card()

    def open_image(self) -> None:
        """Открывает один или несколько файлов (изображения + PDF)."""
        files = self._select_input_files("Открыть изображения или PDF")
        if not files:
            return

        self._reset_project_data()
        self.app_controller.open_files(self.app_state, files)
        self._append_files_to_project(files)
        self._finalize_after_load()
        self._append_activity(
            self._tr("activity_file_opened", "file-opened"),
            "\n".join(files[:3]),
        )

    def open_pdf_file(self) -> None:
        """ PDF-   ."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("open_pdf", "Open PDF"),
            "",
            "PDF Files (*.pdf)",
        )
        if not file_path:
            return
        self._reset_project_data()
        self.app_controller.open_files(self.app_state, [file_path])
        self._append_files_to_project([file_path])
        self._finalize_after_load()
        self._append_activity(
            self._tr("activity_pdf_opened", "pdf-opened"),
            file_path,
        )

    def open_folder(self) -> None:
        """       ."""
        folder = QFileDialog.getExistingDirectory(
            self,
            self._tr("open_folder", "Open folder"),
            "",
        )
        if not folder:
            return

        directory = Path(folder)
        files: list[str] = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.pdf"):
            files.extend(str(path) for path in sorted(directory.glob(pattern)))
        if not files:
            self._show_warning_message(
                self._tr("common_no_data_title", "No data"),
                self._tr(
                    "open_folder_empty",
                    "No images or PDF files found in the selected folder.",
                ),
            )
            return

        self._reset_project_data()
        self.app_controller.open_files(self.app_state, files)
        self._append_files_to_project(files)
        self._finalize_after_load()
        self._append_activity(
            self._tr("activity_folder_opened", "folder-opened"),
            f"{folder}\n{self._tr('files_count', 'Files')}: {len(files)}",
        )


    def add_files(self) -> None:
        """Добавляет новые файлы к уже открытому проекту."""
        files = self._select_input_files("Добавить изображения или PDF")
        if not files:
            return

        self._append_files_to_project(files)
        self._finalize_after_load()
        self._append_activity(
            self._tr("activity_files_added", "files-added"),
            "\n".join(files[:3]),
        )

    def _add_image(self, file_path: str):
        """Добавляет одно изображение."""
        image = self.load_image(file_path)
        if image is None:
            self._show_warning_message(
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
            name, "зображение", idx, "original", image
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
            self._show_error_message(
                "Ошибка загрузки PDF",
                f"Не удалось загрузить PDF:\n{pdf_path}\n\n{e}",
            )

    def _finalize_after_load(self):
        """Общие действия после открытия/добавления файлов."""
        self._ensure_detection_storage()
        self._refresh_tree_filter_classes()
        self._apply_tree_filters()
        self._refresh_statistics_panel()
        self._refresh_thumbnails_panel()
        self._refresh_project_files_list()
        if self.image_storage.images:
            self._active_image_index = 0
            self.app_state.active_image_index = 0
            self.display_image_with_boxes(0)
            self.update_left_info({"type": "original", "index": 0})
            self.thumbnails_panel.set_active_index(0)
            self._set_active_file_row(0)
            self._refresh_detection_result_card()
            return

        self._show_empty_state()
        self._refresh_detection_result_card()

    def _refresh_statistics_panel(self) -> None:
        """Пересчитывает и отображает статистику проекта."""
        if not hasattr(self, "statistics_panel"):
            return
        summary = StatisticsPanel.build_summary(self.image_storage)
        self.statistics_panel.set_summary(summary)

    def _refresh_thumbnails_panel(self) -> None:
        """Обновляет список миниатюр в правой панели."""
        if not hasattr(self, "thumbnails_panel"):
            return
        images = [img for img in self.image_storage.images if isinstance(img, np.ndarray)]
        self.thumbnails_panel.set_images(images)
        self.thumbnails_panel.set_active_index(self._active_image_index)

    def _on_thumbnail_selected(self, index: int) -> None:
        """Переходит к выбранному изображению по клику миниатюры."""
        if index < 0 or index >= len(self.image_storage.images):
            return
        self._active_image_index = index
        self.app_state.active_image_index = index
        self.display_image_with_boxes(index)
        item = self.tree_widget.topLevelItem(index)
        if item:
            self.tree_widget.setCurrentItem(item)
        self.update_left_info({"type": "pdf", "index": index})
        self._set_active_file_row(index)
        self._refresh_detection_result_card()

    def _export_statistics_csv(self) -> None:
        """Экспортирует сводную статистику в CSV."""
        summary = StatisticsPanel.build_summary(self.image_storage)
        if summary.pages_count == 0:
            self._show_warning_message(
                self._tr("common_no_data_title", "Нет данных"),
                self._tr(
                    "stats_export_no_data",
                    "Сначала откройте изображения для формирования статистики.",
                ),
            )
            return

        default_dir = self._default_report_dir()
        default_name = (
            f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("stats_export_dialog_title", "Сохранить статистику"),
            os.path.join(default_dir, default_name),
            "CSV Files (*.csv)",
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".csv"):
            output_path += ".csv"
        StatisticsPanel.export_summary_csv(
            output_path,
            summary,
            language=self.current_language,
        )
        self.statusBar().showMessage(
            self._tr(
                "stats_export_done",
                "Статистика сохранена: {path}",
            ).format(path=output_path),
            3000,
        )

    def show_measurement_history(self) -> None:
        """Открывает диалог с последними записями измерений."""
        records = self.storage_service.load_measurements(limit=300)
        if not records:
            self._show_info_message(
                self._tr("measure_history_title", "стория измерений"),
                self._tr(
                    "measure_history_empty",
                    "Записи измерений пока отсутствуют.",
                ),
            )
            return

        rows: list[str] = []
        for record in records:
            if record.object_index >= 0:
                object_label = str(record.object_index + 1)
            else:
                object_label = self._tr("measure_history_manual", "ручное")
            mm_suffix = ""
            if record.diagonal_mm is not None:
                mm_suffix = (
                    f" | ширина={record.width_mm:.2f} мм"
                    f", высота={record.height_mm:.2f} мм"
                    f", диагональ={record.diagonal_mm:.2f} мм"
                )
            rows.append(
                (
                    f"{record.timestamp} | файл={record.source_file or '-'}"
                    f" | стр={record.page_index + 1}"
                    f" | объект={object_label}"
                    f" | ширина={record.width_px}px"
                    f", высота={record.height_px}px"
                    f", диагональ={record.diagonal_px:.2f}px"
                    f"{mm_suffix}"
                )
            )

        dialog = QDialog(self)
        dialog.setWindowTitle(self._tr("measure_history_title", "стория измерений"))
        dialog.resize(900, 460)
        layout = QVBoxLayout(dialog)

        text_view = QTextEdit(dialog)
        text_view.setReadOnly(True)
        text_view.setPlainText("\n".join(rows))
        layout.addWidget(text_view)

        buttons_layout = QHBoxLayout()
        export_button = QPushButton(
            self._tr("measure_history_export_csv", "Экспорт CSV"),
            dialog,
        )
        export_button.clicked.connect(self.export_measurement_history)
        close_button = QPushButton(self._tr("common_close", "Закрыть"), dialog)
        close_button.clicked.connect(dialog.accept)
        buttons_layout.addStretch()
        buttons_layout.addWidget(export_button)
        buttons_layout.addWidget(close_button)
        layout.addLayout(buttons_layout)
        dialog.exec_()

    def export_measurement_history(self) -> None:
        """Экспортирует историю измерений в CSV-файл."""
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("measure_history_export_title", "Экспорт истории измерений"),
            os.path.join(
                self._default_report_dir(),
                f"measurement_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            ),
            "CSV Files (*.csv)",
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".csv"):
            output_path += ".csv"
        saved_path = self.storage_service.export_measurements_csv(output_path)
        self._show_info_message(
            self._tr("measure_export_title", "Экспорт измерений"),
            self._tr("measure_export_done", "Файл сохранен:\n{path}").format(
                path=saved_path
            ),
        )

    def clear_local_cache(self) -> None:
        """Очищает кэш детекции/классификации и сообщает результат."""
        removed_count = self.storage_service.clear_cache()
        self._show_info_message(
            self._tr("cache_clear_title", "Очистка кэша"),
            self._tr("cache_clear_done", "Удалено файлов кэша: {count}").format(
                count=removed_count
            ),
        )

    def _show_page_stats(self, idx: int):
        """Формирует HTML-сводку по количеству и качеству сеянцев на странице."""
        if (not self.image_storage.class_object_image or
                idx >= len(self.image_storage.class_object_image)):
            self.info_text.setHtml(
                '<p style="font-size:14px; margin:0;">'
                f'<b>Страница {idx + 1}</b></p>'
                '<p style="margin-top:12px;">Нет сеянцев</p>'
            )
            return

        objs: list[ObjectImage] = self.image_storage.class_object_image[idx]
        total = len(objs)

        high = sum(1 for o in objs if o.confidence >= cfg.CONF_THRESHOLD_HIGH)
        medium = sum(1 for o in objs if cfg.CONF_THRESHOLD_LOW <= o.confidence < cfg.CONF_THRESHOLD_HIGH)
        critical = total - high - medium

        html = f"""
        <p style="font-size:15px; font-weight:600; margin:0 0 16px 0;">Страница {idx + 1}</p>
        <p style="margin:8px 0;"><b>Всего сеянцев:</b> {total}</p>
        <p style="margin:6px 0;">● Хорошо (≥{cfg.CONF_THRESHOLD_HIGH}): <b>{high}</b></p>
        <p style="margin:6px 0;">● Средне ({cfg.CONF_THRESHOLD_LOW}–{cfg.CONF_THRESHOLD_HIGH}): <b>{medium}</b></p>
        <p style="margin:6px 0;">● Критично (&lt;{cfg.CONF_THRESHOLD_LOW}): <b>{critical}</b></p>
        """
        self.info_text.setHtml(html)

    def _show_seeding_info(self, parent_idx: int, seed_idx: int):
        """Показывает в панели свойств подробности по выбранному сеянцу."""
        obj = self.image_storage.class_object_image[parent_idx][seed_idx]

        html = f"""
        <p style="font-size:15px; font-weight:600; margin:0 0 12px 0;">Сеянец {seed_idx + 1}</p>
        <p style="margin:6px 0;"><b>Уверенность:</b> {obj.confidence:.3f}</p>
        <p style="margin:6px 0; font-family:monospace;"><b>BBox:</b> {obj.bbox}</p>
        <p style="margin:6px 0;"><b>Поворот:</b> {obj.rotation_k * ROTATE_ANGLE_DEG}°</p>
        """

        if obj.image_all_class:
            html += '<p style="margin-top:12px;"><b>Классификация:</b></p>'
            for cls in obj.image_all_class:
                html += (
                    f'<p style="margin:4px 0;">● {cls.class_name}: '
                    f"{cls.confidence:.3f}</p>"
                )

        self.info_text.setHtml(html)


    def load_image(self, file_name: str) -> np.ndarray | None:
        """Загружает изображение с диска."""
        try:
            image = cv2.imread(file_name)
            return image
        except Exception as e:
            logger.error("Ошибка при загрузке изображения: %s", e)
            return None

    def display_image(
        self,
        image: np.ndarray,
        *,
        preserve_view: bool = False,
        previous_zoom: float | None = None,
        previous_center: QPointF | None = None,
    ) -> None:
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
        self._show_canvas()

        self._reset_measure_state(clear_items=True)
        self.graphics_scene.clear()
        self.image_item = self.graphics_scene.addPixmap(self._original_pixmap)
        self.rect_items = {}

        scroll_size = self.scroll_area.viewport().size()
        ratio_w = scroll_size.width() / self._original_pixmap.width()
        ratio_h = scroll_size.height() / self._original_pixmap.height()
        self.min_fit_zoom = min(ratio_w, ratio_h, 1.0)
        if preserve_view and previous_zoom is not None:
            self.zoom_factor = max(float(previous_zoom), self.min_fit_zoom)
        else:
            self.zoom_factor = self.min_fit_zoom
        self.update_image_zoom()
        if preserve_view and previous_center is not None:
            scene_rect = self.graphics_scene.sceneRect()
            clamped_center = QPointF(
                min(max(previous_center.x(), scene_rect.left()), scene_rect.right()),
                min(max(previous_center.y(), scene_rect.top()), scene_rect.bottom()),
            )
            self.graphics_view.centerOn(clamped_center)

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
        self.app_state.zoom_factor = self.zoom_factor
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

        self.app_state.selected_item = item_data
        angle = float(ROTATE_ANGLE_DEG * ROTATE_K)
        result = self.app_controller.rotate_selection(
            self.app_state,
            item_data,
            angle=angle,
            rotate_k=ROTATE_K,
        )
        if result is None:
            logger.warning("rotate_image: Неизвестный тип данных")
            return

        self._active_image_index = result.page_index
        self.app_state.active_image_index = result.page_index
        if result.target == "page":
            logger.info("rotate_image: зображение %s повернуто", result.page_index)
            self.display_image_with_boxes(result.page_index)
            self.update_left_info({"type": "pdf", "index": result.page_index})
            return

        logger.info(
            "rotate_image: Crop %s (от оригинала %s) повернут",
            result.crop_index,
            result.page_index,
        )
        self.display_image(result.image)

    def create_mask(self) -> None:
        """Создаёт бинарную маску найденных объектов на текущем изображении."""
        if not self.image_storage.images:
            self._show_warning_message(
                "Нет изображения",
                "Сначала откройте изображение или PDF.",
            )
            return

        idx = self._active_image_index
        if idx >= len(self.image_storage.images):
            idx = 0
            self._active_image_index = 0
            self.app_state.active_image_index = 0

        image = self.image_storage.images[idx]
        if image is None or not isinstance(image, np.ndarray):
            self._show_warning_message(
                "Пустое изображение",
                "Невозможно построить маску для пустого изображения.",
            )
            return

        objects = []
        if (
            self.image_storage.class_object_image
            and idx < len(self.image_storage.class_object_image)
        ):
            objects = self.image_storage.class_object_image[idx] or []

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for obj in objects:
            if not obj.bbox:
                continue
            x1, y1, x2, y2 = obj.bbox
            x1 = max(0, min(width, int(x1)))
            x2 = max(0, min(width, int(x2)))
            y1 = max(0, min(height, int(y1)))
            y2 = max(0, min(height, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

        if not objects or int(mask.sum()) == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

        while len(self.image_storage.masks) <= idx:
            self.image_storage.masks.append(np.zeros((1, 1), dtype=np.uint8))
        self.image_storage.masks[idx] = mask

        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)
        preview = cv2.addWeighted(image, 0.72, colored_mask, 0.28, 0.0)

        self.display_image(preview)
        if objects:
            for i, obj in enumerate(objects):
                if obj.bbox:
                    x1, y1, x2, y2 = obj.bbox
                    rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                    item = self._create_scene_bbox_item(rect, obj)
                    self.graphics_scene.addItem(item)
                    self.rect_items[i] = item

        non_zero = int(np.count_nonzero(mask))
        ratio = (non_zero / float(mask.size)) * 100.0
        self.statusBar().showMessage(
            f"Маска построена: покрытие {ratio:.1f}% ({non_zero} px)",
            4000,
        )
        logger.info("Маска создана для изображения %s: %s пикселей", idx, non_zero)

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
        try:
            objects = self.app_controller.run_detection(
                self.app_state,
                index,
                results,
                detection_class_name=DETECTION_CLASS_NAME,
                iou_threshold=self.detection_iou_threshold,
                rotate_k=ROTATE_K,
            )
            logger.info("find_seedlings: после NMS осталось %s боксов", len(objects))
            if self.use_cache:
                cache_key = self._build_detection_cache_key(index)
                if cache_key:
                    self.storage_service.save_detection_objects(cache_key, objects)
            self._apply_detection_objects(index, objects)
            logger.info("find_seedlings: завершено")
        except Exception as error:  # pragma: no cover - логирование
            logger.error(
                "Ошибка во время NMS или обработки результатов: %s",
                error,
            )

    def find_seedlings(self) -> None:
        """Запускает модель YOLOv8 для поиска сеянцев на текущем изображении.

        Результаты проходят через простую процедуру NMS. Каждая найденная
        область добавляется в хранилище `image_storage` и отображается в дереве
        слоёв. Если ширина вырезанного участка больше его высоты, изображение
        поворачивается на 90 градусов для вертикальной ориентации.
        """

        self._ensure_detection_storage()

        logger.info("find_seedlings: start")
        current_index = self._active_image_index
        logger.debug("find_seedlings: current_index = %s", current_index)

        if not self._ensure_detection_model(
            "Модель детекции не загружена. Проверьте путь к весам.",
        ):
            return

        image = self._get_current_image_for_detection()
        if image is None:
            return

        if self.use_cache:
            cache_key = self._build_detection_cache_key(current_index)
            if cache_key:
                cached_objects = self.storage_service.load_detection_objects(
                    cache_key
                )
                if cached_objects is not None:
                    self.image_storage.class_object_image[current_index] = (
                        cached_objects
                    )
                    self.image_service.refresh_page_crops(
                        self.image_storage,
                        current_index,
                        rotate_k=ROTATE_K,
                        clear_classification=True,
                    )
                    self._apply_detection_objects(
                        current_index,
                        cached_objects,
                        from_cache=True,
                    )
                    logger.info(
                        "find_seedlings: загружено из кэша (%s объектов)",
                        len(cached_objects),
                    )
                    return

        self.worker = DetectionWorker(
            index=current_index,
            image=image,
            model=self.model,
            weights_path=self.weights_path,
            conf_threshold=self.detection_confidence_threshold,
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
        if not self._ensure_detection_model(
            "Модель детекции не загружена. Дождитесь окончания загрузки.",
        ):
            return
        if not self.image_storage.images:
            logger.warning("find_all_seedlings: Нет изображений")
            return

        if self._find_all_worker is not None and self._find_all_worker.isRunning():
            logger.warning("find_all_seedlings: уже выполняется")
            return

        self._ensure_detection_storage()
        if self.image_storage.class_object_image is None:
            self.image_storage.class_object_image = [
                [] for _ in self.image_storage.images
            ]

        total = len(self.image_storage.images)
        pending_images: list[np.ndarray] = []
        pending_indices: list[int] = []
        cached_count = 0

        if self.use_cache:
            for page_index, image in enumerate(self.image_storage.images):
                cache_key = self._build_detection_cache_key(page_index)
                cached_objects = (
                    self.storage_service.load_detection_objects(cache_key)
                    if cache_key
                    else None
                )
                if cached_objects is None:
                    pending_images.append(image)
                    pending_indices.append(page_index)
                    continue

                self.image_storage.class_object_image[page_index] = cached_objects
                self.image_service.refresh_page_crops(
                    self.image_storage,
                    page_index,
                    rotate_k=ROTATE_K,
                    clear_classification=True,
                )
                self._apply_detection_objects(
                    page_index,
                    cached_objects,
                    from_cache=True,
                )
                cached_count += 1
        else:
            pending_images = list(self.image_storage.images)
            pending_indices = list(range(total))

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(cached_count)
        self._set_detection_actions_enabled(False)
        self._find_all_cancelled = False
        if self._find_all_progress_dialog is not None:
            try:
                self._find_all_progress_dialog.close()
            except RuntimeError:
                pass
            self._find_all_progress_dialog = None
        self._find_all_progress_dialog = QProgressDialog(
            "Пакетная детекция...",
            "Отмена",
            0,
            total,
            self,
        )
        self._find_all_progress_dialog.setWindowTitle("Пакетная обработка")
        self._find_all_progress_dialog.setWindowModality(Qt.WindowModal)
        self._find_all_progress_dialog.setMinimumDuration(0)
        self._find_all_progress_dialog.canceled.connect(
            self._cancel_find_all_seedlings
        )
        self._find_all_progress_dialog.setValue(cached_count)
        self._find_all_progress_dialog.setLabelText(
            f"Обработка изображений: {cached_count}/{total}"
        )
        self._find_all_progress_dialog.show()

        if not pending_images:
            self._on_find_all_finished()
            self.statusBar().showMessage(
                "Все изображения загружены из кэша",
                3000,
            )
            logger.info("find_all_seedlings: все страницы взяты из кэша")
            return

        self._find_all_worker = FindAllWorker(
            pending_images,
            self.model,
            conf_threshold=self.detection_confidence_threshold,
            indices=pending_indices,
            progress_start=cached_count,
            progress_total=total,
        )
        self._find_all_worker.result_ready.connect(self._on_detection_result)
        self._find_all_worker.progress_updated.connect(self._on_find_all_progress)
        self._find_all_worker.finished.connect(self._on_find_all_finished)
        self._find_all_worker.start()

    def _cancel_find_all_seedlings(self) -> None:
        """Отменяет пакетную детекцию по запросу пользователя."""
        if self._find_all_worker is None:
            return
        self._find_all_cancelled = True
        self._pending_classify_after_find_all = False
        self._find_all_worker.cancel()
        self.statusBar().showMessage("Отмена пакетной детекции...", 2000)

    def _on_find_all_progress(self, current: int, total: int) -> None:
        """Обновляет прогресс-бар при «Найти все»."""
        safe_total = max(int(total), 0)
        safe_current = max(0, min(int(current), safe_total)) if safe_total else 0
        self.progress_bar.setValue(safe_current)

        dialog = self._find_all_progress_dialog
        if dialog is not None:
            try:
                if dialog.maximum() != safe_total:
                    dialog.setMaximum(safe_total)
                dialog.setValue(safe_current)
                if self._find_all_progress_dialog is dialog:
                    dialog.setLabelText(
                        f"Обработка изображений: {safe_current}/{safe_total}"
                    )
            except RuntimeError:
                if self._find_all_progress_dialog is dialog:
                    self._find_all_progress_dialog = None

        if safe_total > 0:
            self.statusBar().showMessage(
                f"Пакетная детекция: {safe_current}/{safe_total}",
                800,
            )

    def _on_find_all_finished(self) -> None:
        """Завершение «Найти все»: скрыть прогресс, включить кнопки."""
        should_run_classify = (
            self._pending_classify_after_find_all and not self._find_all_cancelled
        )
        self._pending_classify_after_find_all = False
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self._set_detection_actions_enabled(True)
        dialog = self._find_all_progress_dialog
        self._find_all_progress_dialog = None
        if dialog is not None:
            try:
                dialog.close()
            except RuntimeError:
                pass
        if self._find_all_cancelled:
            self.statusBar().showMessage(
                "Пакетная детекция отменена пользователем",
                3000,
            )
        else:
            self.statusBar().showMessage(
                "Пакетная детекция завершена",
                3000,
            )
        self._find_all_worker = None
        self._find_all_cancelled = False
        logger.info("find_all_seedlings: завершено")
        if should_run_classify:
            self.classify()

    def _create_scene_bbox_item(
        self,
        rect: QRectF,
        obj,
        *,
        bbox_update_callback=None,
        class_label: str | None = None,
    ) -> BBoxItem:
        """Создаёт bbox-элемент сцены с подписью класса и размеров."""
        label = class_label or self._display_part_name(
            getattr(obj, "class_name", None)
        )
        item = BBoxItem(
            rect,
            obj,
            bbox_update_callback=bbox_update_callback,
            class_label=label,
            pixels_per_mm=self.pixels_per_mm,
        )
        item.setEditable(getattr(self, "_active_tool", "select") == "select")
        return item

    def _part_bbox_to_global(
        self,
        page_width: int,
        page_height: int,
        seed_obj: ObjectImage,
        part_obj: AllClassImage,
    ) -> tuple[int, int, int, int] | None:
        """Преобразует bbox части из локальных координат кропа в координаты страницы."""
        if not seed_obj.bbox or not part_obj.bbox:
            return None
        sx1, sy1, sx2, sy2 = seed_obj.bbox
        lx1, ly1, lx2, ly2 = part_obj.bbox

        rotation_k = int(getattr(seed_obj, "rotation_k", 0)) % 4
        ux1, uy1, ux2, uy2 = lx1, ly1, lx2, ly2
        if rotation_k:
            crop_h = max(1, sy2 - sy1)
            crop_w = max(1, sx2 - sx1)
            if seed_obj.image and isinstance(seed_obj.image[0], np.ndarray):
                crop_h, crop_w = seed_obj.image[0].shape[:2]
            ux1, uy1, ux2, uy2 = rotate_bbox(
                lx1,
                ly1,
                lx2,
                ly2,
                crop_w,
                crop_h,
                (-rotation_k) % 4,
            )

        global_bbox = (sx1 + ux1, sy1 + uy1, sx1 + ux2, sy1 + uy2)
        return clip_bbox_to_image(global_bbox, page_width, page_height)

    def _global_bbox_to_part_local(
        self,
        seed_obj: ObjectImage,
        part_obj: AllClassImage,
        global_bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Преобразует bbox части со страницы обратно в локальные координаты кропа."""
        if not seed_obj.bbox:
            return part_obj.bbox or global_bbox
        sx1, sy1, sx2, sy2 = seed_obj.bbox
        crop_w = max(1, sx2 - sx1)
        crop_h = max(1, sy2 - sy1)
        ux1, uy1, ux2, uy2 = global_bbox
        local_unrot = clip_bbox_to_image(
            (ux1 - sx1, uy1 - sy1, ux2 - sx1, uy2 - sy1),
            crop_w,
            crop_h,
        )
        if local_unrot is None:
            return part_obj.bbox or global_bbox

        rotation_k = int(getattr(seed_obj, "rotation_k", 0)) % 4
        if not rotation_k:
            return local_unrot

        rx1, ry1, rx2, ry2 = rotate_bbox(
            local_unrot[0],
            local_unrot[1],
            local_unrot[2],
            local_unrot[3],
            crop_w,
            crop_h,
            rotation_k,
        )
        rotated_w, rotated_h = (
            (crop_h, crop_w) if rotation_k % 2 else (crop_w, crop_h)
        )
        return (
            clip_bbox_to_image((rx1, ry1, rx2, ry2), rotated_w, rotated_h)
            or (part_obj.bbox or global_bbox)
        )

    def display_image_with_boxes(
        self,
        img_idx: int,
        seeding_idx: int = None,
        *,
        preserve_view: bool = False,
    ):
        """
        Универсальный метод отрисовки.
        Если seeding_idx is None — рисуем оригинал и рамки сеянцев.
        Если seeding_idx задан — рисуем кроп сеянца и рамки его частей.
        """
        if img_idx >= len(self.image_storage.images):
            return

        if seeding_idx is None:
            base_img = self.image_storage.images[img_idx]
            objects_to_draw = (
                self.image_storage.class_object_image[img_idx]
                if self.image_storage.class_object_image
                else []
            )
        else:
            obj = self.image_storage.class_object_image[img_idx][seeding_idx]
            base_img = obj.image[0]
            objects_to_draw = obj.image_all_class or []

        previous_zoom = None
        previous_center = None
        if (
            preserve_view
            and hasattr(self, "_original_pixmap")
            and self._original_pixmap is not None
        ):
            previous_zoom = float(getattr(self, "zoom_factor", 1.0))
            previous_center = self.graphics_view.mapToScene(
                self.graphics_view.viewport().rect().center()
            )

        self.display_image(
            base_img,
            preserve_view=preserve_view,
            previous_zoom=previous_zoom,
            previous_center=previous_center,
        )
        self._active_image_index = img_idx
        self.app_state.active_image_index = img_idx

        if self._show_boxes:
            item_idx = 0
            if seeding_idx is None:
                page_height = int(base_img.shape[0]) if isinstance(base_img, np.ndarray) else 0
                page_width = int(base_img.shape[1]) if isinstance(base_img, np.ndarray) else 0
                for seed_obj in objects_to_draw:
                    if seed_obj.bbox and self._is_box_class_visible("seeding"):
                        x1, y1, x2, y2 = seed_obj.bbox
                        rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                        item = self._create_scene_bbox_item(
                            rect,
                            seed_obj,
                            class_label=self._tr("class_seedling", "Seedling"),
                        )
                        self.graphics_scene.addItem(item)
                        self.rect_items[item_idx] = item
                        item_idx += 1

                    for part_obj in seed_obj.image_all_class or []:
                        part_key = self._normalize_part_key(part_obj.class_name)
                        if not self._is_box_class_visible(part_key):
                            continue
                        if page_width <= 0 or page_height <= 0:
                            continue
                        global_bbox = self._part_bbox_to_global(
                            page_width,
                            page_height,
                            seed_obj,
                            part_obj,
                        )
                        if global_bbox is None:
                            continue
                        px1, py1, px2, py2 = global_bbox
                        rect = QRectF(px1, py1, px2 - px1, py2 - py1)
                        item = self._create_scene_bbox_item(
                            rect,
                            part_obj,
                            bbox_update_callback=(
                                lambda new_bbox, parent_obj=seed_obj, part=part_obj: self._global_bbox_to_part_local(
                                    parent_obj,
                                    part,
                                    new_bbox,
                                )
                            ),
                            class_label=self._display_part_name(part_obj.class_name),
                        )
                        self.graphics_scene.addItem(item)
                        self.rect_items[item_idx] = item
                        item_idx += 1
            else:
                for part_obj in objects_to_draw:
                    part_key = self._normalize_part_key(part_obj.class_name)
                    if not self._is_box_class_visible(part_key):
                        continue
                    if not part_obj.bbox:
                        continue
                    x1, y1, x2, y2 = part_obj.bbox
                    rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                    item = self._create_scene_bbox_item(
                        rect,
                        part_obj,
                        class_label=self._display_part_name(part_obj.class_name),
                    )
                    self.graphics_scene.addItem(item)
                    self.rect_items[item_idx] = item
                    item_idx += 1
        self._set_active_file_row(img_idx)
        self._refresh_detection_result_card()



    def display_seeding_with_boxes(self, parent_idx: int, seed_idx: int):
        """Перегрузка для отображения конкретного сеянца."""
        self.display_image_with_boxes(parent_idx, seeding_idx=seed_idx)

    def display_class_image(self, parent_idx, seed_idx, class_idx):
        """Отображение конкретной части (зум на часть на кропе сеянца)."""
        self.display_image_with_boxes(parent_idx, seeding_idx=seed_idx)

    def save_changes(self) -> None:
        """Пересохраняет crop-изображения после изменения рамок."""
        self.app_controller.save_crops(self.app_state)
        logger.info("save_changes: обновлённые координаты сохранены")

    def classify(self) -> None:
        """Классификация частей растения (цветок, корень, стебель)."""
        if not self.image_storage.class_object_image:
            logger.warning("classify: Сначала найдите сеянцы")
            return

        if self.classify_model is None:
            try:
                self.classify_model = YOLO(self.classify_weights_path)
            except Exception as e:
                logger.exception("Не удалось загрузить модель классификации: %s", e)
                self._show_error_message(
                    "Ошибка загрузки модели",
                    (
                        "Не удалось загрузить модель классификации:\n"
                        f"{self.classify_weights_path}\n\n{e}"
                    ),
                )
                return

        logger.info("classify: старт вторичной классификации")

        for img_idx, objects in enumerate(self.image_storage.class_object_image):
            for obj_idx, seeding_obj in enumerate(objects):
                if not seeding_obj.image:
                    continue

                parts = self._classify_single_object(
                    img_idx,
                    obj_idx,
                    seeding_obj,
                )

                root_item = self.tree_widget.topLevelItem(img_idx)
                parent_item = root_item.child(obj_idx) if root_item else None
                if parent_item:
                    for item_idx in reversed(range(parent_item.childCount())):
                        parent_item.takeChild(item_idx)

                for class_idx, part in enumerate(parts):
                    if parent_item:
                        self.tree_widget.add_class_item(
                            parent_item,
                            part.class_name,
                            self._build_part_description(part),
                            img_idx,
                            obj_idx,
                            class_idx,
                            confidence=part.confidence,
                        )

        self.display_image_with_boxes(self._active_image_index)
        self._refresh_tree_filter_classes()
        self._apply_tree_filters()
        self._refresh_statistics_panel()
        self._refresh_detection_result_card()
        self._append_activity(
            self._tr("activity_classify_done", "classify"),
            self._tr("results_found", "Found: {count}").format(
                count=self._last_detection_count
            ),
        )
        logger.info("classify: успешно завершено")

    def _default_report_dir(self) -> str:
        """Возвращает приоритетную папку для сохранения отчётов."""
        if self.app_state.report_dir and os.path.isdir(self.app_state.report_dir):
            return self.app_state.report_dir

        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        configured_dir = settings.value("report_dir", "", type=str)
        if configured_dir and os.path.isdir(configured_dir):
            self.app_state.report_dir = configured_dir
            return configured_dir

        if self.image_storage.file_path:
            source_dir = os.path.dirname(self.image_storage.file_path)
            if source_dir and os.path.isdir(source_dir):
                return source_dir

        return os.getcwd()

    def create_report(self) -> None:
        """Создаёт PDF-отчёт по текущим результатам детекции."""
        if not self.image_storage.images:
            logger.warning("create_report: Нет данных для отчёта")
            return

        default_dir = self._default_report_dir()
        default_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        initial_path = os.path.join(default_dir, default_name)
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить PDF-отчёт",
            initial_path,
            "PDF Files (*.pdf)",
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".pdf"):
            output_path += ".pdf"

        try:
            saved_path = self.app_controller.generate_report(
                self.app_state,
                output_path,
            )
            self.app_state.report_dir = os.path.dirname(saved_path)
            logger.info("Отчёт сохранён: %s", saved_path)
            self._show_info_message(
                "Отчёт создан",
                f"Отчёт сохранён:\n{saved_path}",
            )
        except Exception as e:
            logger.exception("Ошибка при создании отчёта: %s", e)
            self._show_error_message(
                "Ошибка создания отчёта",
                f"Не удалось создать PDF-отчёт:\n{e}",
            )
