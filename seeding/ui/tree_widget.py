"""Виджет дерева слоёв: страницы/изображения → сеянцы → части."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTreeWidget,
    QTreeWidgetItem,
)


class LayerTreeWidget(QTreeWidget):
    """QTreeWidget для отображения структуры результатов анализа."""

    def __init__(self) -> None:
        """Инициализирует таблицу слоёв и базовые параметры колонок."""
        super().__init__()
        self.setHeaderLabels(["Название", "Описание"])
        header = self.header()
        header.setMinimumSectionSize(80)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def add_root_item(
        self,
        name: str,
        description: str,
        index: int,
        image_type: str,
        image,
    ) -> QTreeWidgetItem:
        """Добавляет корневой элемент дерева (страницу/изображение)."""
        _ = image  # Сохраняем сигнатуру метода для совместимости вызовов.
        root = QTreeWidgetItem(self)
        root.setText(0, name)
        root.setText(1, description)
        root.setData(0, Qt.UserRole, {"index": index, "type": image_type})
        root.setFlags(root.flags() & ~Qt.ItemIsEditable)
        self.addTopLevelItem(root)
        return root

    def add_child_item(
        self,
        parent: QTreeWidgetItem,
        name: str,
        description: str,
        parent_index: int,
        index: int,
        image_type: str,
        image,
    ) -> QTreeWidgetItem:
        """Добавляет дочерний элемент сеянца к корневому узлу."""
        _ = (image_type, image)  # Параметры оставлены для совместимости API.
        child = QTreeWidgetItem(parent)
        child.setText(0, name)
        child.setText(1, description)
        child.setData(
            0,
            Qt.UserRole,
            {"type": "seeding", "parent_index": parent_index, "index": index},
        )
        child.setFlags(child.flags() & ~Qt.ItemIsEditable)
        parent.addChild(child)
        return child

    def add_class_item(
        self,
        parent: QTreeWidgetItem,
        name: str,
        description: str,
        parent_index: int,
        seeding_index: int,
        class_index: int,
    ) -> QTreeWidgetItem:
        """Добавляет узел классификации под выбранным сеянцем."""
        child = QTreeWidgetItem(parent)
        child.setText(0, name)
        child.setText(1, description)
        child.setData(
            0,
            Qt.UserRole,
            {
                "type": "class",
                "parent_index": parent_index,
                "seeding_index": seeding_index,
                "class_index": class_index,
            },
        )
        child.setFlags(child.flags() & ~Qt.ItemIsEditable)
        parent.addChild(child)
        return child
