"""Виджет дерева слоёв: страницы/изображения → сеянцы → части."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
)

from .i18n import tr


class LayerTreeWidget(QTreeWidget):
    """QTreeWidget для отображения структуры результатов анализа."""

    measure_requested = pyqtSignal(int, int)
    classify_requested = pyqtSignal(int, int)
    add_part_requested = pyqtSignal(int, int)
    add_seedling_requested = pyqtSignal(int)
    delete_requested = pyqtSignal(dict)
    CONFIDENCE_ROLE = Qt.UserRole + 100

    def __init__(self) -> None:
        """Инициализирует таблицу слоёв и базовые параметры колонок."""
        super().__init__()
        self._language = "ru"
        self.setHeaderLabels(["", ""])
        header = self.header()
        header.setMinimumSectionSize(80)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        self.setColumnWidth(1, 128)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.set_language(self._language)

    def resizeEvent(self, event) -> None:
        """Поддерживает читаемую ширину колонки описания при ресайзе."""
        super().resizeEvent(event)
        width = max(110, min(160, int(self.viewport().width() * 0.38)))
        self.setColumnWidth(1, width)

    def set_language(self, language: str) -> None:
        """Применяет язык для заголовков и контекстных действий дерева."""
        self._language = language
        self.setHeaderLabels(
            [
                tr(language, "tree_col_name", "Название"),
                tr(language, "tree_col_description", "Описание"),
            ]
        )

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
        confidence: float | None = None,
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
        if confidence is not None:
            child.setData(1, self.CONFIDENCE_ROLE, float(confidence))
        child.setFlags(child.flags() & ~Qt.ItemIsEditable)
        parent.addChild(child)
        return child

    def apply_filter(
        self,
        *,
        search_text: str = "",
        class_filter: str = "all",
        min_confidence: float = 0.0,
    ) -> None:
        """Применяет фильтрацию дерева по строке, классу и уверенности."""
        query = search_text.strip().lower()
        wanted = class_filter.strip().lower()
        for root_idx in range(self.topLevelItemCount()):
            root = self.topLevelItem(root_idx)
            root_visible = False
            root_text = f"{root.text(0)} {root.text(1)}".lower()
            root_match = not query or query in root_text

            for child_idx in range(root.childCount()):
                child = root.child(child_idx)
                child_visible = self._item_matches(
                    child,
                    query=query,
                    class_filter=wanted,
                    min_confidence=min_confidence,
                )
                has_visible_class = False

                for class_idx in range(child.childCount()):
                    class_item = child.child(class_idx)
                    class_visible = self._item_matches(
                        class_item,
                        query=query,
                        class_filter=wanted,
                        min_confidence=min_confidence,
                    )
                    class_item.setHidden(not class_visible)
                    has_visible_class = has_visible_class or class_visible

                if child.childCount() > 0:
                    child_visible = child_visible or has_visible_class

                child.setHidden(not child_visible)
                root_visible = root_visible or child_visible

            root.setHidden(not (root_match or root_visible))

    @staticmethod
    def _item_matches(
        item: QTreeWidgetItem,
        *,
        query: str,
        class_filter: str,
        min_confidence: float,
    ) -> bool:
        """Проверяет, удовлетворяет ли узел текущим фильтрам."""
        item_text = f"{item.text(0)} {item.text(1)}".lower()
        if query and query not in item_text:
            return False

        if class_filter and class_filter != "all":
            if class_filter not in item.text(0).lower():
                return False

        confidence = LayerTreeWidget._extract_confidence(item)
        if confidence is not None and confidence < min_confidence:
            return False
        return True

    @staticmethod
    def _extract_confidence(item: QTreeWidgetItem) -> float | None:
        """Извлекает уверенность из metadata узла или текстового fallback."""
        meta_value = item.data(1, LayerTreeWidget.CONFIDENCE_ROLE)
        if meta_value is not None:
            try:
                return float(meta_value)
            except (TypeError, ValueError):
                return None

        text = item.text(1)
        candidate = text.replace(",", ".")
        token = "".join(
            ch for ch in candidate if ch.isdigit() or ch in (".", "-", "+")
        )
        if not token:
            return None
        try:
            return float(token)
        except ValueError:
            return None

    def _show_context_menu(self, pos) -> None:
        """Показывает контекстное меню для текущего узла дерева."""
        item = self.itemAt(pos)
        if item is None:
            return

        payload = item.data(0, Qt.UserRole) or {}
        item_type = payload.get("type")
        menu = QMenu(self)

        if item_type == "seeding":
            action_measure = menu.addAction(
                tr(self._language, "tree_action_measure", "Измерить")
            )
            action_classify = menu.addAction(
                tr(self._language, "tree_action_classify", "Классифицировать")
            )
            action_add_part = menu.addAction(
                tr(self._language, "tree_action_add_part", "Добавить часть")
            )
            menu.addSeparator()
            action_delete = menu.addAction(
                tr(self._language, "tree_action_delete", "Удалить")
            )
            chosen = menu.exec_(self.viewport().mapToGlobal(pos))
            if chosen == action_measure:
                self.measure_requested.emit(
                    int(payload["parent_index"]),
                    int(payload["index"]),
                )
            elif chosen == action_classify:
                self.classify_requested.emit(
                    int(payload["parent_index"]),
                    int(payload["index"]),
                )
            elif chosen == action_add_part:
                self.add_part_requested.emit(
                    int(payload["parent_index"]),
                    int(payload["index"]),
                )
            elif chosen == action_delete:
                self.delete_requested.emit(payload)
            return

        if item_type == "class":
            action_add_part = menu.addAction(
                tr(self._language, "tree_action_add_part", "Добавить часть")
            )
            menu.addSeparator()
            action_delete = menu.addAction(
                tr(self._language, "tree_action_delete", "Удалить")
            )
            chosen = menu.exec_(self.viewport().mapToGlobal(pos))
            if chosen == action_add_part:
                self.add_part_requested.emit(
                    int(payload["parent_index"]),
                    int(payload["seeding_index"]),
                )
            elif chosen == action_delete:
                self.delete_requested.emit(payload)
            return

        if item_type in ("original", "pdf"):
            action_add_seedling = menu.addAction(
                tr(self._language, "tree_action_add_seedling", "Добавить сеянец")
            )
            chosen = menu.exec_(self.viewport().mapToGlobal(pos))
            if chosen == action_add_seedling:
                self.add_seedling_requested.emit(int(payload["index"]))

    def add_class_item(
        self,
        parent: QTreeWidgetItem,
        name: str,
        description: str,
        parent_index: int,
        seeding_index: int,
        class_index: int,
        confidence: float | None = None,
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
        if confidence is not None:
            child.setData(1, self.CONFIDENCE_ROLE, float(confidence))
        child.setFlags(child.flags() & ~Qt.ItemIsEditable)
        parent.addChild(child)
        return child
