"""Диалог выбора параметров экспорта результатов."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from .i18n import tr


class ExportDialog(QDialog):
    """Диалог выбора форматов и папки экспорта."""

    def __init__(
        self,
        parent=None,
        default_dir: str = "",
        *,
        language: str = "ru",
    ) -> None:
        """Инициализирует диалог экспорта с базовыми параметрами.

        Параметры:
            parent: родительский виджет Qt.
            default_dir: стартовая папка, предлагаемая пользователю.
        """
        super().__init__(parent)
        self._language = language
        self._applying_preset = False
        self.resize(460, 360)
        self._default_dir = default_dir or os.getcwd()
        self._build_ui()

    def _t(self, key: str, fallback: str) -> str:
        """Возвращает локализованную строку диалога."""
        return tr(self._language, key, fallback)

    def _build_ui(self) -> None:
        """Создаёт элементы формы выбора папки и форматов экспорта."""
        self.setWindowTitle(self._t("export_dialog_title", "Export results"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        preset_row = QHBoxLayout()
        preset_row.addWidget(
            QLabel(self._t("export_preset_label", "Preset:"), self)
        )
        self.preset_combo = QComboBox(self)
        self.preset_combo.addItem(
            self._t("export_preset_review", "Review (Recommended)"),
            "review",
        )
        self.preset_combo.addItem(
            self._t("export_preset_dataset", "Dataset"),
            "dataset",
        )
        self.preset_combo.addItem(
            self._t("export_preset_archive", "Archive"),
            "archive",
        )
        self.preset_combo.addItem(
            self._t("export_preset_custom", "Custom"),
            "custom",
        )
        self.preset_combo.currentIndexChanged.connect(
            self._apply_selected_preset
        )
        preset_row.addWidget(self.preset_combo)
        layout.addLayout(preset_row)

        path_row = QHBoxLayout()
        path_row.addWidget(
            QLabel(self._t("export_path_label", "Folder:"), self)
        )
        self.path_edit = QLineEdit(self)
        self.path_edit.setText(self._default_dir)
        path_row.addWidget(self.path_edit)
        pick_btn = QPushButton(self._t("export_browse", "Browse"), self)
        pick_btn.clicked.connect(self._pick_dir)
        path_row.addWidget(pick_btn)
        layout.addLayout(path_row)

        self.chk_json = QCheckBox("JSON", self)
        self.chk_json.setChecked(True)
        self.chk_csv = QCheckBox("CSV", self)
        self.chk_csv.setChecked(True)
        self.chk_coco = QCheckBox("COCO", self)
        self.chk_yolo = QCheckBox("YOLO", self)
        self.chk_annotated = QCheckBox(
            self._t("export_format_annotated", "Annotated images"),
            self,
        )
        self.chk_annotated.setChecked(True)
        self.chk_metadata = QCheckBox(
            self._t("export_format_metadata", "Metadata"),
            self,
        )
        self.chk_metadata.setChecked(True)

        self._option_checkboxes = {
            "json": self.chk_json,
            "csv": self.chk_csv,
            "coco": self.chk_coco,
            "yolo": self.chk_yolo,
            "annotated": self.chk_annotated,
            "metadata": self.chk_metadata,
        }

        layout.addWidget(self.chk_json)
        layout.addWidget(self.chk_csv)
        layout.addWidget(self.chk_coco)
        layout.addWidget(self.chk_yolo)
        layout.addWidget(self.chk_annotated)
        layout.addWidget(self.chk_metadata)
        for checkbox in self._option_checkboxes.values():
            checkbox.toggled.connect(self._on_option_toggled)
        self._apply_preset("review")
        layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _apply_selected_preset(self) -> None:
        """Применяет выбранный пресет экспорта."""
        self._apply_preset(str(self.preset_combo.currentData() or "review"))

    def _apply_preset(self, preset_name: str) -> None:
        """Переключает набор отмеченных форматов по пресету."""
        preset_map = {
            "review": {
                "json": True,
                "csv": True,
                "coco": False,
                "yolo": False,
                "annotated": True,
                "metadata": True,
            },
            "dataset": {
                "json": False,
                "csv": False,
                "coco": True,
                "yolo": True,
                "annotated": False,
                "metadata": True,
            },
            "archive": {
                "json": True,
                "csv": True,
                "coco": True,
                "yolo": True,
                "annotated": True,
                "metadata": True,
            },
        }
        if preset_name not in preset_map:
            return

        self._applying_preset = True
        try:
            for key, checkbox in self._option_checkboxes.items():
                checkbox.setChecked(
                    bool(preset_map[preset_name].get(key, False))
                )
        finally:
            self._applying_preset = False

    def _on_option_toggled(self) -> None:
        """Переводит пресет в custom после ручного изменения форматов."""
        if self._applying_preset:
            return
        custom_index = self.preset_combo.findData("custom")
        if (
            custom_index >= 0
            and self.preset_combo.currentIndex() != custom_index
        ):
            self.preset_combo.setCurrentIndex(custom_index)

    def _pick_dir(self) -> None:
        """Открывает диалог выбора папки."""
        current = self.path_edit.text().strip() or self._default_dir
        selected = QFileDialog.getExistingDirectory(
            self,
            self._t("export_pick_directory_title", "Select export folder"),
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
    def selected_preset(self) -> str:
        """Возвращает выбранный пресет экспорта."""
        return str(self.preset_combo.currentData() or "review")

    @property
    def options(self) -> dict[str, bool]:
        """Возвращает выбранные форматы экспорта."""
        return {
            "json": self.chk_json.isChecked(),
            "csv": self.chk_csv.isChecked(),
            "coco": self.chk_coco.isChecked(),
            "yolo": self.chk_yolo.isChecked(),
            "annotated": self.chk_annotated.isChecked(),
            "metadata": self.chk_metadata.isChecked(),
        }
