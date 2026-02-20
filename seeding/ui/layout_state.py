"""Утилиты для сохранения и восстановления layout окна."""

from __future__ import annotations

from PyQt5.QtCore import QByteArray


def normalize_qbytearray(value) -> QByteArray:
    """Преобразует значение из QSettings в ``QByteArray``."""
    if isinstance(value, QByteArray):
        return value
    if isinstance(value, (bytes, bytearray)):
        return QByteArray(bytes(value))
    return QByteArray()
