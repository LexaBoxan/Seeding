from PyQt5.QtCore import QByteArray

from seeding.ui.layout_state import normalize_qbytearray


def test_normalize_qbytearray_accepts_qbytearray():
    raw = QByteArray(b"abc")
    normalized = normalize_qbytearray(raw)
    assert isinstance(normalized, QByteArray)
    assert bytes(normalized) == b"abc"


def test_normalize_qbytearray_accepts_bytes():
    normalized = normalize_qbytearray(b"xyz")
    assert isinstance(normalized, QByteArray)
    assert bytes(normalized) == b"xyz"


def test_normalize_qbytearray_returns_empty_for_invalid_value():
    normalized = normalize_qbytearray({"unexpected": True})
    assert isinstance(normalized, QByteArray)
    assert normalized.isEmpty()
