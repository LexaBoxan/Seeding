"""BBoxItem — совместим с передачей QRectF или кортежа (x1,y1,x2,y2). Полностью защищён от крашей."""

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QPen, QBrush, QColor
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsItem


class BBoxItem(QGraphicsRectItem):
    MIN_SIZE = 30.0

    def __init__(self, arg, obj, parent=None, color=Qt.green, offset=(0, 0)):
        """
        arg может быть:
        - QRectF
        - или кортеж (x1, y1, x2, y2)
        """
        if isinstance(arg, QRectF):
            rect = arg.normalized()
        else:
            # предполагаем кортеж (x1, y1, x2, y2)
            try:
                x1, y1, x2, y2 = arg
                left = min(x1, x2)
                top = min(y1, y2)
                width = max(self.MIN_SIZE, max(x1, x2) - left)
                height = max(self.MIN_SIZE, max(y1, y2) - top)
                rect = QRectF(left, top, width, height)
            except (TypeError, ValueError):
                logger.error(f"Неподдерживаемый аргумент для BBoxItem: {type(arg)} {arg}")
                rect = QRectF(0, 0, self.MIN_SIZE, self.MIN_SIZE)

        # Финальная защита
        if rect.width() < self.MIN_SIZE:
            rect.setWidth(self.MIN_SIZE)
        if rect.height() < self.MIN_SIZE:
            rect.setHeight(self.MIN_SIZE)

        super().__init__(rect, parent)

        self.obj = obj
        self.offset = offset

        self.setPen(QPen(color, 3, Qt.SolidLine))
        self.setBrush(QBrush(QColor(0, 255, 0, 20)))

        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges
        )

        self._editable = False
        self._handle = None
        self._handles = {}

        self._update_handles()

    def _update_handles(self):
        r = self.rect()
        s = self.MIN_SIZE / 2
        if r.width() < self.MIN_SIZE * 1.5 or r.height() < self.MIN_SIZE * 1.5:
            self._handles = {}
            return
        self._handles = {
            "tl": QRectF(r.left(), r.top(), s, s),
            "tr": QRectF(r.right() - s, r.top(), s, s),
            "bl": QRectF(r.left(), r.bottom() - s, s, s),
            "br": QRectF(r.right() - s, r.bottom() - s, s, s),
        }

    def setEditable(self, state: bool):
        self._editable = state
        self.setFlag(QGraphicsItem.ItemIsMovable, state)
        self.update()

    def paint(self, painter, option, widget=None):
        r = self.rect()
        if r.width() < 1 or r.height() < 1:
            return
        super().paint(painter, option, widget)

        if self._editable:
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(QBrush(Qt.white))
            for h in self._handles.values():
                if not h.isEmpty():
                    painter.drawRect(h)

    def mousePressEvent(self, event):
        if self._editable:
            pos = event.pos()
            for name, rect in self._handles.items():
                if rect.contains(pos):
                    self._handle = name
                    return
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

            r = r.normalized()
            if r.width() < self.MIN_SIZE:
                r.setWidth(self.MIN_SIZE)
            if r.height() < self.MIN_SIZE:
                r.setHeight(self.MIN_SIZE)

            self.setRect(r)
            self._update_handles()
            self.update_bbox()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._handle = None
        self._update_handles()
        self.update_bbox()

    def update_bbox(self):
        r = self.rect().normalized()
        ox, oy = self.offset
        x1 = max(0, int(r.left() + ox))
        y1 = max(0, int(r.top() + oy))
        x2 = max(x1 + int(self.MIN_SIZE), int(r.right() + ox))
        y2 = max(y1 + int(self.MIN_SIZE), int(r.bottom() + oy))
        self.obj.bbox = (x1, y1, x2, y2)