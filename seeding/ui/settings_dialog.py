"""Диалог настроек порогов, темы и языка интерфейса."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

import seeding.config as cfg
from seeding.config import (
    CALIBRATION_PIXELS_PER_MM_DEFAULT,
    CONF_THRESHOLD_HIGH_DEFAULT,
    CONF_THRESHOLD_LOW_DEFAULT,
    DEFAULT_CLASSIFY_WEIGHTS_PATH,
    DEFAULT_WEIGHTS_PATH,
    DETECTION_CONFIDENCE_THRESHOLD,
    DETECTION_IOU_THRESHOLD,
    DIALOG_LABEL_MIN_WIDTH,
    DIALOG_LAYOUT_SPACING,
    DIALOG_SETTINGS_MIN_WIDTH,
    DIALOG_SPINBOX_MIN_WIDTH,
    DIALOG_SPINBOX_STEP,
    PROJECT_ROOT,
    QSETTINGS_APP,
    QSETTINGS_ORG,
    USE_CACHE_DEFAULT,
)
from seeding.utils import resolve_weights_path

from .preferences import (
    DEFAULT_UI_LANGUAGE,
    DEFAULT_UI_THEME,
    load_ui_preferences,
    save_ui_preferences,
)
from .i18n import tr
from .styles import build_dialog_stylesheet


class SettingsDialog(QDialog):
    """Диалог, сохраняющий пользовательские параметры в QSettings."""

    def __init__(self, parent=None):
        """Создаёт диалог настроек и загружает текущие значения из QSettings."""
        super().__init__(parent)
        self.settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        self.ui_preferences = load_ui_preferences()
        self._language = self.ui_preferences.language
        self.setWindowTitle(self._t("settings_title", "Настройки"))
        self.setMinimumWidth(DIALOG_SETTINGS_MIN_WIDTH)
        self._base_dirs = (PROJECT_ROOT, Path.cwd())
        self.calibration_requested = False
        self.setStyleSheet(build_dialog_stylesheet(self.ui_preferences.theme))
        self.init_ui()

    def _t(self, key: str, fallback: str) -> str:
        """Возвращает локализованную строку для текущего языка диалога."""
        return tr(self._language, key, fallback)

    def _read_probability_setting(self, key: str, default: float) -> float:
        """Читает и ограничивает значение вероятности из настроек."""
        try:
            value = float(self.settings.value(key, default))
        except (TypeError, ValueError):
            value = float(default)
        return min(max(value, 0.0), 1.0)

    def init_ui(self) -> None:
        """Создаёт элементы диалога настроек."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(DIALOG_LAYOUT_SPACING)

        frame = QFrame()
        frame.setObjectName("settingsCard")
        frame.setStyleSheet(
            """
            QFrame#settingsCard {
                border: 1px solid #6b7280;
                border-radius: 8px;
                padding: 16px;
            }
            """
        )
        frame_layout = QVBoxLayout(frame)
        frame_layout.setSpacing(12)

        high_layout = QHBoxLayout()
        lbl_high = QLabel(self._t("settings_high_quality", "Порог высокого качества:"))
        lbl_high.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        high_layout.addWidget(lbl_high)
        self.spin_high = QDoubleSpinBox()
        self.spin_high.setRange(0.0, 1.0)
        self.spin_high.setSingleStep(DIALOG_SPINBOX_STEP)
        self.spin_high.setValue(
            self._read_probability_setting(
                "conf_high",
                CONF_THRESHOLD_HIGH_DEFAULT,
            )
        )
        self.spin_high.setMinimumWidth(DIALOG_SPINBOX_MIN_WIDTH)
        high_layout.addWidget(self.spin_high)
        frame_layout.addLayout(high_layout)

        low_layout = QHBoxLayout()
        lbl_low = QLabel(
            self._t("settings_medium_quality", "Порог среднего качества:")
        )
        lbl_low.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        low_layout.addWidget(lbl_low)
        self.spin_low = QDoubleSpinBox()
        self.spin_low.setRange(0.0, 1.0)
        self.spin_low.setSingleStep(DIALOG_SPINBOX_STEP)
        self.spin_low.setValue(
            self._read_probability_setting(
                "conf_low",
                CONF_THRESHOLD_LOW_DEFAULT,
            )
        )
        self.spin_low.setMinimumWidth(DIALOG_SPINBOX_MIN_WIDTH)
        low_layout.addWidget(self.spin_low)
        frame_layout.addLayout(low_layout)

        detect_conf_layout = QHBoxLayout()
        lbl_detect_conf = QLabel(
            self._t("settings_detect_conf", "Минимальная уверенность детекции:")
        )
        lbl_detect_conf.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        detect_conf_layout.addWidget(lbl_detect_conf)
        self.spin_detect_conf = QDoubleSpinBox()
        self.spin_detect_conf.setRange(0.0, 1.0)
        self.spin_detect_conf.setSingleStep(DIALOG_SPINBOX_STEP)
        self.spin_detect_conf.setValue(
            self._read_probability_setting(
                "detect_conf",
                DETECTION_CONFIDENCE_THRESHOLD,
            )
        )
        self.spin_detect_conf.setMinimumWidth(DIALOG_SPINBOX_MIN_WIDTH)
        detect_conf_layout.addWidget(self.spin_detect_conf)
        frame_layout.addLayout(detect_conf_layout)

        detect_iou_layout = QHBoxLayout()
        lbl_detect_iou = QLabel(self._t("settings_detect_iou", "Порог IoU для NMS:"))
        lbl_detect_iou.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        detect_iou_layout.addWidget(lbl_detect_iou)
        self.spin_detect_iou = QDoubleSpinBox()
        self.spin_detect_iou.setRange(0.0, 1.0)
        self.spin_detect_iou.setSingleStep(DIALOG_SPINBOX_STEP)
        self.spin_detect_iou.setValue(
            self._read_probability_setting(
                "detect_iou",
                DETECTION_IOU_THRESHOLD,
            )
        )
        self.spin_detect_iou.setMinimumWidth(DIALOG_SPINBOX_MIN_WIDTH)
        detect_iou_layout.addWidget(self.spin_detect_iou)
        frame_layout.addLayout(detect_iou_layout)

        calibration_layout = QHBoxLayout()
        lbl_calibration = QLabel(
            self._t("settings_calibration", "Коэффициент калибровки (px/mm):")
        )
        lbl_calibration.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        calibration_layout.addWidget(lbl_calibration)
        self.spin_pixels_per_mm = QDoubleSpinBox()
        self.spin_pixels_per_mm.setRange(0.0, 10000.0)
        self.spin_pixels_per_mm.setSingleStep(0.1)
        self.spin_pixels_per_mm.setDecimals(4)
        self.spin_pixels_per_mm.setValue(
            max(
                0.0,
                float(
                    self.settings.value(
                        "pixels_per_mm",
                        CALIBRATION_PIXELS_PER_MM_DEFAULT,
                    )
                ),
            )
        )
        self.spin_pixels_per_mm.setMinimumWidth(DIALOG_SPINBOX_MIN_WIDTH)
        calibration_layout.addWidget(self.spin_pixels_per_mm)
        self.calibrate_button = QPushButton(
            self._t("settings_calibrate_button", "Калибровка по линейке")
        )
        self.calibrate_button.clicked.connect(self._request_calibration)
        calibration_layout.addWidget(self.calibrate_button)
        frame_layout.addLayout(calibration_layout)

        cache_layout = QHBoxLayout()
        lbl_cache = QLabel(self._t("settings_use_cache", "Использовать кэш результатов:"))
        lbl_cache.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        cache_layout.addWidget(lbl_cache)
        self.check_use_cache = QCheckBox()
        self.check_use_cache.setChecked(
            bool(self.settings.value("use_cache", USE_CACHE_DEFAULT, type=bool))
        )
        cache_layout.addWidget(self.check_use_cache)
        cache_layout.addStretch()
        frame_layout.addLayout(cache_layout)

        detect_weights_layout = QHBoxLayout()
        lbl_detect_weights = QLabel(self._t("settings_detect_model", "Модель детекции (.pt):"))
        lbl_detect_weights.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        detect_weights_layout.addWidget(lbl_detect_weights)
        self.detect_weights_edit = QLineEdit()
        self.detect_weights_edit.setText(
            self.settings.value(
                "detect_weights_path",
                str(DEFAULT_WEIGHTS_PATH),
                type=str,
            )
        )
        detect_weights_layout.addWidget(self.detect_weights_edit)
        self.detect_weights_button = QPushButton(self._t("settings_pick", "Выбрать"))
        self.detect_weights_button.clicked.connect(self._choose_detection_weights)
        detect_weights_layout.addWidget(self.detect_weights_button)
        frame_layout.addLayout(detect_weights_layout)

        classify_weights_layout = QHBoxLayout()
        lbl_classify_weights = QLabel(
            self._t("settings_classify_model", "Модель классификации (.pt):")
        )
        lbl_classify_weights.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        classify_weights_layout.addWidget(lbl_classify_weights)
        self.classify_weights_edit = QLineEdit()
        self.classify_weights_edit.setText(
            self.settings.value(
                "classify_weights_path",
                str(DEFAULT_CLASSIFY_WEIGHTS_PATH),
                type=str,
            )
        )
        classify_weights_layout.addWidget(self.classify_weights_edit)
        self.classify_weights_button = QPushButton(self._t("settings_pick", "Выбрать"))
        self.classify_weights_button.clicked.connect(
            self._choose_classification_weights
        )
        classify_weights_layout.addWidget(self.classify_weights_button)
        frame_layout.addLayout(classify_weights_layout)

        report_layout = QHBoxLayout()
        lbl_report = QLabel(self._t("settings_report_dir", "Папка отчётов по умолчанию:"))
        lbl_report.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        report_layout.addWidget(lbl_report)

        self.report_dir_edit = QLineEdit()
        self.report_dir_edit.setPlaceholderText(
            self._t("settings_report_dir_empty", "Не выбрана")
        )
        self.report_dir_edit.setText(
            self.settings.value("report_dir", "", type=str)
        )
        report_layout.addWidget(self.report_dir_edit)

        self.report_dir_button = QPushButton(self._t("settings_pick", "Выбрать"))
        self.report_dir_button.clicked.connect(self._choose_report_dir)
        report_layout.addWidget(self.report_dir_button)
        frame_layout.addLayout(report_layout)

        theme_layout = QHBoxLayout()
        lbl_theme = QLabel(self._t("settings_theme", "Тема интерфейса:"))
        lbl_theme.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        theme_layout.addWidget(lbl_theme)
        self.theme_combo = QComboBox()
        self.theme_combo.addItem(self._t("settings_theme_dark", "Тёмная"), "dark")
        self.theme_combo.addItem(self._t("settings_theme_light", "Светлая"), "light")
        current_theme_idx = self.theme_combo.findData(
            self.ui_preferences.theme,
        )
        if current_theme_idx >= 0:
            self.theme_combo.setCurrentIndex(current_theme_idx)
        theme_layout.addWidget(self.theme_combo)
        frame_layout.addLayout(theme_layout)

        language_layout = QHBoxLayout()
        lbl_language = QLabel(self._t("settings_language", "Язык интерфейса:"))
        lbl_language.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        language_layout.addWidget(lbl_language)
        self.language_combo = QComboBox()
        self.language_combo.addItem(self._t("settings_lang_ru", "Русский"), "ru")
        self.language_combo.addItem(self._t("settings_lang_en", "English"), "en")
        current_language_idx = self.language_combo.findData(
            self.ui_preferences.language,
        )
        if current_language_idx >= 0:
            self.language_combo.setCurrentIndex(current_language_idx)
        language_layout.addWidget(self.language_combo)
        frame_layout.addLayout(language_layout)

        layout.addWidget(frame)
        layout.addSpacing(8)

        self.reset_button = QPushButton(
            self._t("settings_reset", "Сбросить по умолчанию")
        )
        self.reset_button.clicked.connect(self._reset_defaults)
        layout.addWidget(self.reset_button)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_report_dir(self) -> None:
        """Открывает выбор папки для отчётов."""
        current = self.report_dir_edit.text().strip() or os.getcwd()
        directory = QFileDialog.getExistingDirectory(
            self,
            self._t("settings_choose_report_dir", "Выберите папку отчётов"),
            current,
        )
        if directory:
            self.report_dir_edit.setText(directory)

    def _request_calibration(self) -> None:
        """Закрывает диалог и запрашивает запуск калибровки в главном окне."""
        self.calibration_requested = True
        self.accept()

    def _choose_detection_weights(self) -> None:
        """Открывает выбор весов модели детекции."""
        current = self.detect_weights_edit.text().strip() or os.getcwd()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self._t("settings_choose_detect_model", "Выберите модель детекции"),
            current,
            "PyTorch weights (*.pt);;All files (*)",
        )
        if file_path:
            self.detect_weights_edit.setText(file_path)

    def _choose_classification_weights(self) -> None:
        """Открывает выбор весов модели классификации."""
        current = self.classify_weights_edit.text().strip() or os.getcwd()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self._t("settings_choose_classify_model", "Выберите модель классификации"),
            current,
            "PyTorch weights (*.pt);;All files (*)",
        )
        if file_path:
            self.classify_weights_edit.setText(file_path)

    def _reset_defaults(self) -> None:
        """Возвращает параметры интерфейса и моделей к значениям по умолчанию."""
        self.spin_high.setValue(CONF_THRESHOLD_HIGH_DEFAULT)
        self.spin_low.setValue(CONF_THRESHOLD_LOW_DEFAULT)
        self.spin_detect_conf.setValue(DETECTION_CONFIDENCE_THRESHOLD)
        self.spin_detect_iou.setValue(DETECTION_IOU_THRESHOLD)
        self.spin_pixels_per_mm.setValue(CALIBRATION_PIXELS_PER_MM_DEFAULT)
        self.check_use_cache.setChecked(USE_CACHE_DEFAULT)
        self.detect_weights_edit.setText(str(DEFAULT_WEIGHTS_PATH))
        self.classify_weights_edit.setText(str(DEFAULT_CLASSIFY_WEIGHTS_PATH))
        self.report_dir_edit.clear()
        theme_index = self.theme_combo.findData(DEFAULT_UI_THEME)
        if theme_index >= 0:
            self.theme_combo.setCurrentIndex(theme_index)
        language_index = self.language_combo.findData(DEFAULT_UI_LANGUAGE)
        if language_index >= 0:
            self.language_combo.setCurrentIndex(language_index)

    def save_settings(self) -> bool:
        """Сохраняет параметры и применяет их в рантайме."""
        new_high = self.spin_high.value()
        new_low = self.spin_low.value()
        if new_high < new_low:
            QMessageBox.information(
                self,
                self._t("settings_thresholds_swapped_title", "Пороги скорректированы"),
                self._t(
                    "settings_thresholds_swapped_text",
                    "Значения порогов автоматически поменяны местами.",
                ),
            )
            new_high, new_low = new_low, new_high

        cfg.CONF_THRESHOLD_HIGH = new_high
        cfg.CONF_THRESHOLD_LOW = new_low
        self.settings.setValue("conf_high", new_high)
        self.settings.setValue("conf_low", new_low)
        self.settings.setValue("detect_conf", self.spin_detect_conf.value())
        self.settings.setValue("detect_iou", self.spin_detect_iou.value())
        self.settings.setValue("pixels_per_mm", self.spin_pixels_per_mm.value())
        self.settings.setValue("use_cache", self.check_use_cache.isChecked())

        detect_input = self.detect_weights_edit.text().strip()
        detect_source = detect_input or str(DEFAULT_WEIGHTS_PATH)
        detect_weights = resolve_weights_path(
            detect_source,
            base_dirs=self._base_dirs,
        )
        if detect_weights is None:
            QMessageBox.critical(
                self,
                self._t("settings_model_path_error", "Ошибка пути к модели"),
                self._t(
                    "settings_detect_model_missing",
                    "Не удалось найти веса детекции.\nПуть: {path}",
                ).format(path=detect_source),
            )
            return False

        classify_input = self.classify_weights_edit.text().strip()
        classify_source = classify_input or str(DEFAULT_CLASSIFY_WEIGHTS_PATH)
        classify_weights = resolve_weights_path(
            classify_source,
            base_dirs=self._base_dirs,
        )
        if classify_weights is None:
            QMessageBox.critical(
                self,
                self._t("settings_model_path_error", "Ошибка пути к модели"),
                self._t(
                    "settings_classify_model_missing",
                    "Не удалось найти веса классификации.\nПуть: {path}",
                ).format(path=classify_source),
            )
            return False

        self.settings.setValue("detect_weights_path", str(detect_weights))
        self.settings.setValue("classify_weights_path", str(classify_weights))

        report_dir = self.report_dir_edit.text().strip()
        if report_dir and os.path.isdir(report_dir):
            self.settings.setValue("report_dir", report_dir)
        elif not report_dir:
            self.settings.setValue("report_dir", "")

        theme = self.theme_combo.currentData() or DEFAULT_UI_THEME
        language = self.language_combo.currentData() or DEFAULT_UI_LANGUAGE
        save_ui_preferences(theme=theme, language=language)
        self.settings.sync()
        return True

    @property
    def selected_theme(self) -> str:
        """Возвращает ключ выбранной темы."""
        return str(self.theme_combo.currentData() or DEFAULT_UI_THEME)

    @property
    def selected_language(self) -> str:
        """Возвращает ключ выбранного языка."""
        return str(self.language_combo.currentData() or DEFAULT_UI_LANGUAGE)
