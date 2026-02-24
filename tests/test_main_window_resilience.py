import importlib
import os
import sys
import types

import numpy as np
from PyQt5.QtCore import QPointF
from PyQt5.QtCore import QEvent, QSettings
from PyQt5.QtWidgets import QGraphicsItem
from PyQt5.QtWidgets import QApplication

from seeding.config import QSETTINGS_APP, QSETTINGS_ORG


def _ensure_offscreen_qt() -> tuple[QApplication, bool]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created = app is None
    if app is None:
        app = QApplication([])
    return app, created


def _isolate_qsettings(tmp_path) -> None:
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
    settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    settings.clear()
    settings.sync()


def _import_main_window_module(monkeypatch):
    class FakeYOLO:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return []

    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        types.SimpleNamespace(YOLO=FakeYOLO),
    )
    sys.modules.pop("seeding.ui.main_window", None)
    return importlib.import_module("seeding.ui.main_window")


def test_event_filter_works_even_if_active_tool_missing(tmp_path, monkeypatch):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)
    module = _import_main_window_module(monkeypatch)
    monkeypatch.setattr(
        module.ImageEditor,
        "_start_model_loading",
        lambda self: None,
    )

    window = module.ImageEditor("dummy_weights.pt")
    delattr(window, "_active_tool")

    event = QEvent(QEvent.MouseButtonPress)
    handled = window.eventFilter(window.graphics_view.viewport(), event)
    assert handled is False

    window.close()
    if created:
        app.quit()


def test_find_all_progress_safe_without_dialog(tmp_path, monkeypatch):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)
    module = _import_main_window_module(monkeypatch)
    monkeypatch.setattr(
        module.ImageEditor,
        "_start_model_loading",
        lambda self: None,
    )

    window = module.ImageEditor("dummy_weights.pt")
    window._find_all_progress_dialog = None

    window.progress_bar.setRange(0, 10)
    window._on_find_all_progress(3, 10)
    assert window.progress_bar.value() == 3

    window._on_find_all_progress(-5, 0)
    assert window.progress_bar.value() == 0

    window.close()
    if created:
        app.quit()


def test_measure_label_ignores_view_transform(tmp_path, monkeypatch):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)
    module = _import_main_window_module(monkeypatch)
    monkeypatch.setattr(
        module.ImageEditor,
        "_start_model_loading",
        lambda self: None,
    )

    window = module.ImageEditor("dummy_weights.pt")
    window._start_manual_measure(QPointF(10.0, 10.0))

    assert window._measure_text_item is not None
    assert (
        window._measure_text_item.flags()
        & QGraphicsItem.ItemIgnoresTransformations
    )

    window.close()
    if created:
        app.quit()


def test_calibration_measurement_updates_pixels_per_mm(tmp_path, monkeypatch):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)
    module = _import_main_window_module(monkeypatch)
    monkeypatch.setattr(
        module.ImageEditor,
        "_start_model_loading",
        lambda self: None,
    )

    window = module.ImageEditor("dummy_weights.pt")
    window.image_storage.images = [np.zeros((120, 120, 3), dtype=np.uint8)]
    window.display_image(window.image_storage.images[0])
    window._calibration_pending = True

    monkeypatch.setattr(
        module.QInputDialog,
        "getDouble",
        lambda *args, **kwargs: (10.0, True),
    )
    monkeypatch.setattr(
        window.storage_service,
        "append_measurement",
        lambda record: None,
    )

    window._start_manual_measure(QPointF(0.0, 0.0))
    window._finish_manual_measure(QPointF(100.0, 0.0))

    assert abs(window.pixels_per_mm - 10.0) < 1e-9
    settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    assert abs(float(settings.value("pixels_per_mm")) - 10.0) < 1e-9

    window.close()
    if created:
        app.quit()
