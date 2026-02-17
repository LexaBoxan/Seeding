"""Интерактивный прямоугольник для отображения bounding box.

Поддерживает изменение размера, перемещение и цветовую индикацию
по уровню уверенности детекции (зелёный / оранжевый / красный).
"""

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsRectItem, QStyleOptionGraphicsItem, QWidget

import seeding.config as cfg

# Толщина пера рамки
BBOX_PEN_WIDTH = 2


def get_color_by_confidence(conf: float) -> QColor:
    """Возвращает цвет рамки по уровню уверенности."""
    if conf >= cfg.CONF_THRESHOLD_HIGH:
        return QColor(Qt.green)
    if conf >= cfg.CONF_THRESHOLD_LOW:
        return QColor(*cfg.BBOX_COLOR_ORANGE_RGB)
    return QColor(Qt.red)


class BBoxItem(QGraphicsRectItem):
    """
    Универсальный интерактивный прямоугольник.
    Связывается с объектами ObjectImage (сеянцы) или AllClassImage (части).
    """

    HANDLE_SIZE = 8.0  # Размер ручек изменения размера (px)

    def __init__(
        self,
        rect: QRectF,
        obj,
        parent: QGraphicsItem | None = None,
        offset=(0, 0),
    ):
        super().__init__(rect, parent)
        self.obj = obj  # Может быть ObjectImage или AllClassImage
        self.offset = offset

        # Цвет рамки по уверенности
        color = get_color_by_confidence(getattr(obj, "confidence", 0.0))
        self.setPen(QPen(color, BBOX_PEN_WIDTH))

        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self._editable = False
        self._handle = None
        self._handles = {}
        self._update_handles()

    def _update_handles(self) -> None:
        """Пересчет координат ручек изменения размера."""
        r = self.rect()
        s = self.HANDLE_SIZE
        self._handles = {
            "tl": QRectF(r.x() - s / 2, r.y() - s / 2, s, s),
            "tr": QRectF(r.right() - s / 2, r.y() - s / 2, s, s),
            "bl": QRectF(r.x() - s / 2, r.bottom() - s / 2, s, s),
            "br": QRectF(r.right() - s / 2, r.bottom() - s / 2, s, s),
        }

    def setEditable(self, state: bool) -> None:
        self._editable = state
        self.setFlag(QGraphicsItem.ItemIsMovable, state)
        self.update()

    def paint(self, painter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None):
        # Рисуем основную рамку
        super().paint(painter, option, widget)
        # Если включен режим редактирования, рисуем белые квадратики по углам
        if self._editable:
            painter.setBrush(Qt.white)
            painter.setPen(QPen(Qt.black, 1))
            for handle_rect in self._handles.values():
                painter.drawRect(handle_rect)

    def mousePressEvent(self, event):
        if self._editable:
            for name, rect in self._handles.items():
                if rect.contains(event.pos()):
                    self._handle = name
                    break
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._editable and self._handle:
            r = self.rect()
            pos = event.pos()
            if self._handle == "tl":
                r.setTopLeft(pos)
            elif self._handle == "tr":
                r.setTopRight(pos)
            elif self._handle == "bl":
                r.setBottomLeft(pos)
            elif self._handle == "br":
                r.setBottomRight(pos)
            self.setRect(r)
        else:
            super().mouseMoveEvent(event)

        if self._editable:
            self._update_handles()
            self.update_bbox()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._handle = None
        if self._editable:
            self._update_handles()
            self.update_bbox()

    def update_bbox(self) -> None:
        """Обновляет координаты bbox в связанном объекте данных."""
        r = self.rect().normalized()
        ox, oy = self.offset
        # Записываем новые координаты обратно в объект (сеянец или часть)
        self.obj.bbox = (
            int(r.left() + ox),
            int(r.top() + oy),
            int(r.right() + ox),
            int(r.bottom() + oy),
        )