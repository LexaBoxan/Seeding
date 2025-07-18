"""Дерево слоёв для отображения оригиналов и найденных объектов."""

from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem
from PyQt5.QtCore import Qt


class LayerTreeWidget(QTreeWidget):
    """Простой QTreeWidget для отображения иерархии изображений."""

    def __init__(self) -> None:
        """Конструктор дерева слоёв."""
        super().__init__()
        self.setHeaderLabels(["Название", "Описание"])

    def add_root_item(self, name, description, index, image_type, image):
        """
        Добавляет корневой элемент (страницу, оригинал и т.д.) в дерево.
        """
        root = QTreeWidgetItem(self)
        root.setText(0, name)
        root.setText(1, description)
        # Сохраняем изображение и индекс внутри UserRole
        root.setData(0, Qt.UserRole, {"index": index, "type": image_type})
        self.addTopLevelItem(root)
        return root

    def add_child_item(
        self, parent, name, description, parent_index, index, image_type, image
    ):
        """
        Добавляет дочерний элемент к выбранному родителю.
        """
        child = QTreeWidgetItem(parent)
        child.setText(0, name)
        child.setText(1, description)
        # Тут parent_index — это индекс оригинального изображения, index — это индекс сеянца (crop-а)
        child.setData(
            0,
            Qt.UserRole,
            {"type": "seeding", "parent_index": parent_index, "index": index},
        )
        parent.addChild(child)
        return child
