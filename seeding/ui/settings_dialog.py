"""Диалог настроек порогов уверенности.

Позволяет изменять CONF_THRESHOLD_HIGH и CONF_THRESHOLD_LOW
с сохранением в QSettings между сессиями.
"""

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QDialogButtonBox,
    QFrame,
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

from .styles import DIALOG_STYLESHEET


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(DIALOG_SETTINGS_MIN_WIDTH)
        self.setStyleSheet(DIALOG_STYLESHEET)

        self.settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(DIALOG_LAYOUT_SPACING)

        # Блок порогов уверенности
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #23262b;
                border: 1px solid #3b4048;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        frame_layout = QVBoxLayout(frame)

        # Порог высокий
        h_layout = QHBoxLayout()
        lbl_high = QLabel("Отличная уверенность (зелёный):")
        lbl_high.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        h_layout.addWidget(lbl_high)
        self.spin_high = QDoubleSpinBox()
        self.spin_high.setRange(0.0, 1.0)
        self.spin_high.setSingleStep(DIALOG_SPINBOX_STEP)
        self.spin_high.setValue(cfg.CONF_THRESHOLD_HIGH)
        self.spin_high.setMinimumWidth(DIALOG_SPINBOX_MIN_WIDTH)
        h_layout.addWidget(self.spin_high)
        frame_layout.addLayout(h_layout)

        # Порог низкий
        l_layout = QHBoxLayout()
        lbl_low = QLabel("Хорошая уверенность (оранжевый):")
        lbl_low.setMinimumWidth(DIALOG_LABEL_MIN_WIDTH)
        l_layout.addWidget(lbl_low)
        self.spin_low = QDoubleSpinBox()
        self.spin_low.setRange(0.0, 1.0)
        self.spin_low.setSingleStep(DIALOG_SPINBOX_STEP)
        self.spin_low.setValue(cfg.CONF_THRESHOLD_LOW)
        self.spin_low.setMinimumWidth(DIALOG_SPINBOX_MIN_WIDTH)
        l_layout.addWidget(self.spin_low)
        frame_layout.addLayout(l_layout)
        frame_layout.setSpacing(12)

        layout.addWidget(frame)
        layout.addSpacing(8)

        # Кнопки
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def save_settings(self):
        """Перезаписываем глобальные переменные и сохраняем в реестр."""
        new_high = self.spin_high.value()
        new_low = self.spin_low.value()

        # 1. МЕНЯЕМ НАПРЯМУЮ В ГЛОБАЛЬНОМ КОНФИГЕ
        cfg.CONF_THRESHOLD_HIGH = new_high
        cfg.CONF_THRESHOLD_LOW = new_low

        # 2. СОХРАНЯЕМ В ПАМЯТЬ (QSettings)
        self.settings.setValue("conf_high", new_high)
        self.settings.setValue("conf_low", new_low)