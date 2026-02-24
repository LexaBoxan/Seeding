"""Панель миниатюр для быстрой навигации по изображениям."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from .i18n import tr


class ThumbnailsPanel(QWidget):
    """Виджет списка миниатюр с выбором текущего изображения."""

    image_selected = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        """Инициализирует панель миниатюр и её внутренний список."""
        super().__init__(parent)
        self._language = "ru"
        self._build_ui()

    def set_language(self, language: str) -> None:
        """Устанавливает язык подсказок миниатюр."""
        self._language = language

    def _build_ui(self) -> None:
        """Создаёт список миниатюр и настраивает режим отображения иконок."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.list_widget = QListWidget(self)
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setMovement(QListWidget.Static)
        self.list_widget.setSpacing(8)
        self.list_widget.setIconSize(QSize(88, 88))
        self.list_widget.setWordWrap(True)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

    def set_images(self, images: list[np.ndarray]) -> None:
        """Заполняет список миниатюрами переданных изображений."""
        self.list_widget.clear()
        for idx, image in enumerate(images):
            icon = self._build_icon(image)
            item = QListWidgetItem(icon, str(idx + 1))
            item.setData(Qt.UserRole, idx)
            item.setToolTip(
                tr(self._language, "thumb_item", "Изображение {index}").format(
                    index=idx + 1
                )
            )
            self.list_widget.addItem(item)

    def set_active_index(self, index: int) -> None:
        """Выделяет текущую миниатюру по индексу."""
        if index < 0 or index >= self.list_widget.count():
            return
        self.list_widget.setCurrentRow(index)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Проксирует выбор миниатюры наружу."""
        idx = int(item.data(Qt.UserRole))
        self.image_selected.emit(idx)

    @staticmethod
    def _build_icon(image: np.ndarray) -> QIcon:
        """Создаёт QIcon миниатюры из numpy-изображения."""
        if image is None or not isinstance(image, np.ndarray):
            pix = QPixmap(88, 88)
            pix.fill(Qt.transparent)
            return QIcon(pix)

        if image.ndim == 2:
            rgb = image
            q_image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.shape[1],
                QImage.Format_Grayscale8,
            )
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            q_image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.shape[1] * 3,
                QImage.Format_RGB888,
            )

        pixmap = QPixmap.fromImage(q_image).scaled(
            88,
            88,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        return QIcon(pixmap)
