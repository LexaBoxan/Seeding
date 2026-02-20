"""Главное окно приложения Seeding.

Реализует ImageEditor — окно с загрузкой изображений/PDF, детекцией сеянцев YOLOv8,
классификацией частей растения, деревом слоёв и генерацией PDF-отчётов.
"""

import logging
import os
from datetime import datetime
from typing import Callable, Sequence

import cv2
import fitz
import numpy as np
from PyQt5.QtCore import (
    QByteArray,
    QEvent,
    QPoint,
    QRectF,
    QSettings,
    QSize,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPixmap, QTransform

from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QDockWidget,
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
    DETECTION_CLASS_NAME,
    DEFAULT_CLASSIFY_WEIGHTS_PATH,
    NMS_IOU_THRESHOLD,
    PANEL_LAYERS_MAX_WIDTH,
    PANEL_LAYERS_MIN_WIDTH,
    PANEL_LAYOUT_MARGINS,
    PDF_RENDER_SCALE,
    QSETTINGS_APP,
    QSETTINGS_ORG,
    ROTATE_ANGLE_DEG,
    ROTATE_K,
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
from seeding.models import AppState, ObjectImage, OriginalImage
from seeding.services import ImageService, ReportService

from .bbox_item import BBoxItem
from .icon_manager import IconManager
from .i18n import tr
from .layout_state import normalize_qbytearray
from .metrics import UiMetrics
from .preferences import load_ui_preferences
from .settings_dialog import SettingsDialog
from .theme_manager import apply_theme
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
        "Создать маску (в разработке)",
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
    """Поток для детекции одного изображения."""

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
    """Поток для поиска сеянцев на всех изображениях."""

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

        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        self._load_runtime_settings(settings)

        self.ui_preferences = load_ui_preferences()
        self.current_language = self.ui_preferences.language
        self.current_theme = self.ui_preferences.theme
        self.setWindowTitle(tr(self.current_language, "window_title", "Анализ сеянцев"))
        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.zoom_factor = ZOOM_FACTOR_INITIAL
        self.image_storage = OriginalImage()
        self.weights_path = weights_path
        self.model = None
        self.classify_model = None
        self._find_all_worker = None

        self._active_image_index = 0
        self.app_state = AppState(
            image_storage=self.image_storage,
            active_image_index=self._active_image_index,
            zoom_factor=self.zoom_factor,
            report_dir=settings.value("report_dir", "", type=str),
        )
        self.image_service = ImageService()
        self.report_service = ReportService()
        self.app_controller = AppController(
            image_service=self.image_service,
            report_service=self.report_service,
        )
        self.ui_metrics = UiMetrics()
        self.icon_manager = IconManager(self)

        self.setup_ui()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

        self._start_model_loading()

    def _load_runtime_settings(self, settings: QSettings) -> None:
        """Читает настраиваемые пороги из ``QSettings``."""
        cfg.CONF_THRESHOLD_HIGH = float(
            settings.value("conf_high", cfg.CONF_THRESHOLD_HIGH),
        )
        cfg.CONF_THRESHOLD_LOW = float(
            settings.value("conf_low", cfg.CONF_THRESHOLD_LOW),
        )
        logger.info(
            "Настройки загружены: High=%s, Low=%s",
            cfg.CONF_THRESHOLD_HIGH,
            cfg.CONF_THRESHOLD_LOW,
        )

    def setup_ui(self) -> None:
        """Собирает интерфейс и применяет стартовые параметры окна."""
        self.setup_actions()
        self.create_left_panel()
        self.create_central_widget()
        self.create_right_panel()
        self.create_panel_docks()
        self.create_toolbox_dock()
        self.connect_signals()
        self._apply_language(self.current_language)

        central_wrapper = QWidget(self)
        wrapper_layout = QVBoxLayout(central_wrapper)
        wrapper_layout.setContentsMargins(
            self.ui_metrics.padding_m,
            self.ui_metrics.padding_s,
            self.ui_metrics.padding_m,
            self.ui_metrics.padding_s,
        )
        wrapper_layout.setSpacing(0)
        wrapper_layout.addWidget(self.canvas_host)
        self.setCentralWidget(central_wrapper)

        self._setup_shortcuts()
        self._set_detection_actions_enabled(False)
        self._restore_layout_settings()

    def init_ui(self) -> None:
        """Совместимый алиас для старого имени метода инициализации UI."""
        self.setup_ui()

    def setup_actions(self) -> None:
        """Создаёт действия верхнего меню и тулбара."""
        self.create_top_toolbar()
        self.create_menu()

    def connect_signals(self) -> None:
        """Подключает сигналы виджетов к обработчикам."""
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)

    def _set_detection_actions_enabled(self, enabled: bool) -> None:
        """Включает или выключает действия детекции."""
        self.action_find.setEnabled(enabled)
        self.action_find_all.setEnabled(enabled)

    def create_panel_docks(self) -> None:
        """Создаёт правый док с вкладками «Слои/Свойства»."""
        self.right_tabs = QTabWidget(self)
        self.right_tabs.setObjectName("rightTabs")
        self.right_tabs.addTab(
            self.right_panel,
            tr(self.current_language, "tab_layers", "Слои"),
        )
        self.right_tabs.addTab(
            self.left_panel,
            tr(self.current_language, "tab_properties", "Свойства"),
        )
        self.layers_dock = QDockWidget(
            tr(self.current_language, "dock_layers", "Слои"),
            self,
        )
        self.layers_dock.setObjectName("layersDock")
        self.layers_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.layers_dock.setWidget(self.right_tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, self.layers_dock)
        self.layers_dock.setMinimumWidth(PANEL_LAYERS_MIN_WIDTH)

    def create_toolbox_dock(self) -> None:
        """Создаёт левую панель инструментов без текстовых подписей."""
        self.toolbox_toolbar = QToolBar("", self)
        self.toolbox_toolbar.setObjectName("toolboxToolbar")
        self.toolbox_toolbar.setOrientation(Qt.Vertical)
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

        self._attach_toolbox_dock()
        self._activate_default_tool()

    def _create_tool_action(
        self,
        *,
        attr_name: str,
        icon_name: str,
        tool_name: str,
        hint: str,
        fallback_icon: QStyle.StandardPixmap,
    ) -> QAction:
        """Создаёт переключаемое действие панели инструментов."""
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

    def _attach_toolbox_dock(self) -> None:
        """Размещает тулбар инструментов в левом доке."""
        self.toolbox_dock = QDockWidget("", self)
        self.toolbox_dock.setObjectName("toolboxDock")
        self.toolbox_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.toolbox_dock.setWidget(self.toolbox_toolbar)
        self.toolbox_dock.setMinimumWidth(self.ui_metrics.toolbox_width)
        self.toolbox_dock.setMaximumWidth(self.ui_metrics.toolbox_width + 8)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.toolbox_dock)

    def _activate_default_tool(self) -> None:
        """Инициализирует состояние активного инструмента."""
        self._active_tool = "select"
        self._space_hand_active = False
        self._tool_before_space = "select"
        self.action_tool_select.setChecked(True)
        self._set_active_tool("select")

    def _restore_layout_settings(self) -> None:
        """Восстанавливает геометрию окна, доки и активный инструмент."""
        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        geometry = settings.value("window_geometry", QByteArray())
        geometry_bytes = normalize_qbytearray(geometry)
        if not geometry_bytes.isEmpty():
            self.restoreGeometry(geometry_bytes)

        state = settings.value("window_state", QByteArray())
        state_bytes = normalize_qbytearray(state)
        if not state_bytes.isEmpty():
            self.restoreState(state_bytes)

        tool_name = settings.value("active_tool", "select", type=str)
        self._set_checked_tool(tool_name)

    def _save_layout_settings(self) -> None:
        """Сохраняет геометрию окна, состояние доков и активный инструмент."""
        settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        settings.setValue("window_geometry", self.saveGeometry())
        settings.setValue("window_state", self.saveState())
        settings.setValue("active_tool", self._active_tool)
        settings.sync()

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
        if hasattr(self, "left_panel"):
            self.left_panel.setTitle("")
        if hasattr(self, "right_panel"):
            self.right_panel.setTitle("")
        if hasattr(self, "right_tabs"):
            self.right_tabs.setTabText(0, tr(language, "tab_layers", "Слои"))
            self.right_tabs.setTabText(1, tr(language, "tab_properties", "Свойства"))
        if hasattr(self, "layers_dock"):
            self.layers_dock.setWindowTitle(tr(language, "dock_layers", "Слои"))
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

    def _set_checked_tool(self, tool_name: str) -> None:
        """Активирует инструмент по имени через состояние QAction."""
        mapping = {
            "select": self.action_tool_select,
            "hand": self.action_tool_hand,
            "zoom": self.action_tool_zoom,
        }
        action = mapping.get(tool_name, self.action_tool_select)
        action.setChecked(True)
        self._set_active_tool(tool_name)

    def _set_active_tool(self, tool_name: str) -> None:
        """Применяет режим взаимодействия для выбранного инструмента."""
        self._active_tool = tool_name
        if tool_name == "hand":
            self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.graphics_view.setCursor(Qt.OpenHandCursor)
            return
        if tool_name == "zoom":
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            self.graphics_view.setCursor(Qt.CrossCursor)
            return
        self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        self.graphics_view.setCursor(Qt.ArrowCursor)

    def eventFilter(self, watched, event):
        """Обрабатывает клики в режиме лупы."""
        if (
            hasattr(self, "graphics_view")
            and watched is self.graphics_view.viewport()
            and event.type() == QEvent.MouseButtonPress
            and self._active_tool == "zoom"
        ):
            if event.button() == Qt.LeftButton:
                self.zoom_in()
                return True
            if event.button() == Qt.RightButton:
                self.zoom_out()
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        """Временно переключает на инструмент «Рука» при удержании Space."""
        if (
            event.key() == Qt.Key_Space
            and not event.isAutoRepeat()
            and not self._space_hand_active
        ):
            self._space_hand_active = True
            self._tool_before_space = self._active_tool
            self._set_checked_tool("hand")
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Возвращает предыдущий инструмент после отпускания Space."""
        if (
            event.key() == Qt.Key_Space
            and not event.isAutoRepeat()
            and self._space_hand_active
        ):
            self._space_hand_active = False
            self._set_checked_tool(self._tool_before_space)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event):
        """Сохраняет layout и геометрию окна при закрытии."""
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

    def create_top_toolbar(self):
        """Создаёт верхний тулбар и регистрирует действия приложения."""
        self.toolbar = QToolBar("Инструменты", self)
        self.toolbar.setObjectName("mainToolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(
            QSize(
                self.ui_metrics.icon_size_toolbar,
                self.ui_metrics.icon_size_toolbar,
            )
        )
        self.toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.toolbar.setMinimumHeight(self.ui_metrics.toolbar_height)
        self.toolbar.setMaximumHeight(self.ui_metrics.toolbar_height)
        self._set_uniform_button_style(self.toolbar)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        self._register_toolbar_actions()
        toolbar_groups = [
            [
                self.action_mask,
                self.action_find,
                self.action_find_all,
                self.action_classify,
                self.action_rotate,
            ],
            [self.action_report],
            [self.action_zoom_in, self.action_zoom_out, self.action_fit],
            [self.action_open, self.action_add],
            [self.action_save, self.action_settings],
        ]
        for group_index, actions in enumerate(toolbar_groups):
            for action in actions:
                self.toolbar.addAction(action)
            if group_index < len(toolbar_groups) - 1:
                self.toolbar.addSeparator()

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
        self.image_storage = OriginalImage()
        self.app_state.image_storage = self.image_storage
        self.tree_widget.clear()
        self._show_empty_state()

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
        self._show_warning_message("Модель не загружена", message)
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


    def create_left_panel(self):
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

        item = self.tree_widget.topLevelItem(new_idx)
        if item:
            self.tree_widget.setCurrentItem(item)

        self.update_left_info({"type": "pdf", "index": new_idx})

    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            if not dialog.save_settings():
                return
            settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
            self.app_state.report_dir = settings.value("report_dir", "", type=str)
            self._apply_theme(dialog.selected_theme)
            self._apply_language(dialog.selected_language)
            self.display_image_with_boxes(self._active_image_index)
            logger.info("Настройки обновлены и применены.")

    def _setup_shortcuts(self):
        """Настраивает все горячие клавиши."""
        QShortcut(QKeySequence(Qt.Key_Left),  self, lambda: self.switch_image(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.switch_image(1))

        QShortcut(QKeySequence("R"),            self, self.rotate_image)
        QShortcut(QKeySequence("D"),            self, self.find_seedlings)

        QShortcut(QKeySequence("V"), self, lambda: self._set_checked_tool("select"))
        QShortcut(QKeySequence("H"), self, lambda: self._set_checked_tool("hand"))
        QShortcut(QKeySequence("Z"), self, lambda: self._set_checked_tool("zoom"))

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
        layout.setContentsMargins(*PANEL_LAYOUT_MARGINS)

        self.tree_widget = LayerTreeWidget()
        self.tree_widget.setObjectName("layerTree")

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

    def open_image(self) -> None:
        """Открывает один или несколько файлов (изображения + PDF)."""
        files = self._select_input_files("Открыть изображения или PDF")
        if not files:
            return

        self._reset_project_data()
        self.app_controller.open_files(self.app_state, files)
        self._append_files_to_project(files)
        self._finalize_after_load()


    def add_files(self) -> None:
        """Добавляет новые файлы к уже открытому проекту."""
        files = self._select_input_files("Добавить изображения или PDF")
        if not files:
            return

        self._append_files_to_project(files)
        self._finalize_after_load()

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
            self._show_error_message(
                "Ошибка загрузки PDF",
                f"Не удалось загрузить PDF:\n{pdf_path}\n\n{e}",
            )

    def _finalize_after_load(self):
        """Общие действия после открытия/добавления файлов."""
        self._ensure_detection_storage()
        if self.image_storage.images:
            self._active_image_index = 0
            self.app_state.active_image_index = 0
            self.display_image_with_boxes(0)
            self.update_left_info({"type": "original", "index": 0})
            return

        self._show_empty_state()

    def _show_page_stats(self, idx: int):
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
        self._show_canvas()

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
            logger.info("rotate_image: Изображение %s повернуто", result.page_index)
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
        try:
            objects = self.app_controller.run_detection(
                self.app_state,
                index,
                results,
                detection_class_name=DETECTION_CLASS_NAME,
                iou_threshold=NMS_IOU_THRESHOLD,
                rotate_k=ROTATE_K,
            )
            logger.info("find_seedlings: после NMS осталось %s боксов", len(objects))

            parent_item = self.tree_widget.topLevelItem(index)
            if parent_item is not None:
                for i in reversed(range(parent_item.childCount())):
                    parent_item.takeChild(i)
            else:
                logger.warning("find_seedlings: parent tree item not found for index %s", index)
                return

            for i_out, obj in enumerate(objects):
                crop_preview = obj.image[0] if obj.image else np.empty((0, 0, 3))
                self.tree_widget.add_child_item(
                    parent_item,
                    f"Seeding{i_out + 1}",
                    f"Уверенность: {obj.confidence:.2f}",
                    index,
                    i_out,
                    "seeding",
                    crop_preview,
                )
            self.display_image_with_boxes(index)
            self.update_left_info({"type": "pdf", "index": index})
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

        total = len(self.image_storage.images)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self._set_detection_actions_enabled(False)

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
        self.app_state.active_image_index = self._active_image_index
        if current < total:
            self.update_left_info({"type": "pdf", "index": self._active_image_index})

    def _on_find_all_finished(self) -> None:
        """Завершение «Найти все»: скрыть прогресс, включить кнопки."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self._set_detection_actions_enabled(True)
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

        if seeding_idx is None:
            base_img = self.image_storage.images[img_idx]
            objects_to_draw = self.image_storage.class_object_image[img_idx]
        else:
            obj = self.image_storage.class_object_image[img_idx][seeding_idx]
            base_img = obj.image[0]
            objects_to_draw = obj.image_all_class or []

        self.display_image(base_img)

        for i, obj in enumerate(objects_to_draw):
            if obj.bbox:
                x1, y1, x2, y2 = obj.bbox
                rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                item = BBoxItem(rect, obj)
                item.setEditable(True)
                self.graphics_scene.addItem(item)
                self.rect_items[i] = item



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
                self.classify_model = YOLO(DEFAULT_CLASSIFY_WEIGHTS_PATH)
            except Exception as e:
                logger.exception("Не удалось загрузить модель классификации: %s", e)
                self._show_error_message(
                    "Ошибка загрузки модели",
                    (
                        "Не удалось загрузить модель классификации:\n"
                        f"{DEFAULT_CLASSIFY_WEIGHTS_PATH}\n\n{e}"
                    ),
                )
                return

        logger.info("classify: старт вторичной классификации")

        for img_idx, objects in enumerate(self.image_storage.class_object_image):
            for obj_idx, seeding_obj in enumerate(objects):
                if not seeding_obj.image:
                    continue

                results = self.classify_model(seeding_obj.image[0])
                parts = self.app_controller.run_classification_for_selection(
                    self.app_state,
                    img_idx,
                    obj_idx,
                    results,
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
                            f"Уверенность: {part.confidence:.2f}",
                            img_idx,
                            obj_idx,
                            class_idx,
                        )

        self.display_image_with_boxes(self._active_image_index)
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
