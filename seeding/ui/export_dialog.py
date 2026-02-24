"""Диалог выбора параметров экспорта результатов."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class ExportDialog(QDialog):
    """Диалог выбора форматов и папки экспорта."""

    def __init__(self, parent=None, default_dir: str = "") -> None:
        """Инициализирует диалог экспорта с базовыми параметрами.

        Параметры:
            parent: родительский виджет Qt.
            default_dir: стартовая папка, предлагаемая пользователю.
        """
        super().__init__(parent)
        self.setWindowTitle("Экспорт результатов")
        self.resize(420, 280)
        self._default_dir = default_dir or os.getcwd()
        self._build_ui()

    def _build_ui(self) -> None:
        """Создаёт элементы формы выбора папки и форматов экспорта."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Папка:"))
        self.path_edit = QLineEdit(self)
        self.path_edit.setText(self._default_dir)
        path_row.addWidget(self.path_edit)
        pick_btn = QPushButton("Выбрать", self)
        pick_btn.clicked.connect(self._pick_dir)
        path_row.addWidget(pick_btn)
        layout.addLayout(path_row)

        self.chk_json = QCheckBox("JSON", self)
        self.chk_json.setChecked(True)
        self.chk_csv = QCheckBox("CSV", self)
        self.chk_csv.setChecked(True)
        self.chk_coco = QCheckBox("COCO", self)
        self.chk_yolo = QCheckBox("YOLO", self)
        self.chk_annotated = QCheckBox("Изображения с аннотациями", self)
        self.chk_annotated.setChecked(True)

        layout.addWidget(self.chk_json)
        layout.addWidget(self.chk_csv)
        layout.addWidget(self.chk_coco)
        layout.addWidget(self.chk_yolo)
        layout.addWidget(self.chk_annotated)
        layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_dir(self) -> None:
        """Открывает диалог выбора папки."""
        current = self.path_edit.text().strip() or self._default_dir
        selected = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для экспорта",
            current,
        )
        if selected:
            self.path_edit.setText(selected)

    @property
    def output_dir(self) -> Path:
        """Возвращает выбранную папку экспорта."""
        path = self.path_edit.text().strip() or self._default_dir
        return Path(path).expanduser()

    @property
    def options(self) -> dict[str, bool]:
        """Возвращает выбранные форматы экспорта."""
        return {
            "json": self.chk_json.isChecked(),
            "csv": self.chk_csv.isChecked(),
            "coco": self.chk_coco.isChecked(),
            "yolo": self.chk_yolo.isChecked(),
            "annotated": self.chk_annotated.isChecked(),
        }
