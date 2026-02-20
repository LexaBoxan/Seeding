"""Диалог настроек порогов, темы и языка интерфейса."""

from __future__ import annotations

import os

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import (
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
    DIALOG_LABEL_MIN_WIDTH,
    DIALOG_LAYOUT_SPACING,
    DIALOG_SETTINGS_MIN_WIDTH,
    DIALOG_SPINBOX_MIN_WIDTH,
    DIALOG_SPINBOX_STEP,
    QSETTINGS_APP,
    QSETTINGS_ORG,
)

from .preferences import (
    DEFAULT_UI_LANGUAGE,
    DEFAULT_UI_THEME,
    load_ui_preferences,
    save_ui_preferences,
)
from .styles import build_dialog_stylesheet


class SettingsDialog(QDialog):
    """Диалог, сохраняющий пользовательские параметры в QSettings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(DIALOG_SETTINGS_MIN_WIDTH)
        self.settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        self.ui_preferences = load_ui_preferences()
        self.setStyleSheet(build_dialog_stylesheet(self.ui_preferences.theme))
        self.init_ui()

    def init_ui(self) -> None:
        """Создаёт элементы диалога настроек."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(DIALOG_LAYOUT_SPACING)

        frame = QFrame()
        frame.setStyleSheet(
            """
            QFrame {
                border: 1px solid #6b7280;
                border-radius: 8px;
                padding: 16px;
            }
            """
        )
        frame_layout = QVBoxLayout(frame)
        frame_layout.setSpacing(12)

        high_layout = QHBoxLayout()
        lbl_high = QLabel("Порог высокого качества:")
        lbl_high.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        high_layout.addWidget(lbl_high)
        self.spin_high = QDoubleSpinBox()
        self.spin_high.setRange(0.0, 1.0)
        self.spin_high.setSingleStep(DIALOG_SPINBOX_STEP)
        self.spin_high.setValue(cfg.CONF_THRESHOLD_HIGH)
        self.spin_high.setMinimumWidth(DIALOG_SPINBOX_MIN_WIDTH)
        high_layout.addWidget(self.spin_high)
        frame_layout.addLayout(high_layout)

        low_layout = QHBoxLayout()
        lbl_low = QLabel("Порог среднего качества:")
        lbl_low.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        low_layout.addWidget(lbl_low)
        self.spin_low = QDoubleSpinBox()
        self.spin_low.setRange(0.0, 1.0)
        self.spin_low.setSingleStep(DIALOG_SPINBOX_STEP)
        self.spin_low.setValue(cfg.CONF_THRESHOLD_LOW)
        self.spin_low.setMinimumWidth(DIALOG_SPINBOX_MIN_WIDTH)
        low_layout.addWidget(self.spin_low)
        frame_layout.addLayout(low_layout)

        report_layout = QHBoxLayout()
        lbl_report = QLabel("Папка отчётов по умолчанию:")
        lbl_report.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        report_layout.addWidget(lbl_report)

        self.report_dir_edit = QLineEdit()
        self.report_dir_edit.setPlaceholderText("Не выбрана")
        self.report_dir_edit.setText(
            self.settings.value("report_dir", "", type=str)
        )
        report_layout.addWidget(self.report_dir_edit)

        self.report_dir_button = QPushButton("Выбрать")
        self.report_dir_button.clicked.connect(self._choose_report_dir)
        report_layout.addWidget(self.report_dir_button)
        frame_layout.addLayout(report_layout)

        theme_layout = QHBoxLayout()
        lbl_theme = QLabel("Тема интерфейса:")
        lbl_theme.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        theme_layout.addWidget(lbl_theme)
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Тёмная", "dark")
        self.theme_combo.addItem("Светлая", "light")
        current_theme_idx = self.theme_combo.findData(
            self.ui_preferences.theme,
        )
        if current_theme_idx >= 0:
            self.theme_combo.setCurrentIndex(current_theme_idx)
        theme_layout.addWidget(self.theme_combo)
        frame_layout.addLayout(theme_layout)

        language_layout = QHBoxLayout()
        lbl_language = QLabel("Язык интерфейса:")
        lbl_language.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        language_layout.addWidget(lbl_language)
        self.language_combo = QComboBox()
        self.language_combo.addItem("Русский", "ru")
        self.language_combo.addItem("English", "en")
        current_language_idx = self.language_combo.findData(
            self.ui_preferences.language,
        )
        if current_language_idx >= 0:
            self.language_combo.setCurrentIndex(current_language_idx)
        language_layout.addWidget(self.language_combo)
        frame_layout.addLayout(language_layout)

        layout.addWidget(frame)
        layout.addSpacing(8)

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
            "Выберите папку отчётов",
            current,
        )
        if directory:
            self.report_dir_edit.setText(directory)

    def save_settings(self) -> bool:
        """Сохраняет параметры и применяет их в рантайме."""
        new_high = self.spin_high.value()
        new_low = self.spin_low.value()
        if new_high < new_low:
            QMessageBox.information(
                self,
                "Пороги скорректированы",
                "Значения порогов автоматически поменяны местами.",
            )
            new_high, new_low = new_low, new_high

        cfg.CONF_THRESHOLD_HIGH = new_high
        cfg.CONF_THRESHOLD_LOW = new_low
        self.settings.setValue("conf_high", new_high)
        self.settings.setValue("conf_low", new_low)

        report_dir = self.report_dir_edit.text().strip()
        if report_dir and os.path.isdir(report_dir):
            self.settings.setValue("report_dir", report_dir)
        elif not report_dir:
            self.settings.setValue("report_dir", "")

        theme = self.theme_combo.currentData() or DEFAULT_UI_THEME
        language = self.language_combo.currentData() or DEFAULT_UI_LANGUAGE
        save_ui_preferences(theme=theme, language=language)
        return True

    @property
    def selected_theme(self) -> str:
        """Возвращает ключ выбранной темы."""
        return str(self.theme_combo.currentData() or DEFAULT_UI_THEME)

    @property
    def selected_language(self) -> str:
        """Возвращает ключ выбранного языка."""
        return str(self.language_combo.currentData() or DEFAULT_UI_LANGUAGE)
