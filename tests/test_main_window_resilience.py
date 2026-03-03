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
    source_file = tmp_path / "sample.png"
    source_file.write_text("stub", encoding="utf-8")
    window.storage_service = module.StorageService(tmp_path / "storage")
    window.image_storage.images = [np.zeros((120, 120, 3), dtype=np.uint8)]
    window.image_storage.source_files = [str(source_file)]
    window.image_storage.file_path = str(source_file)
    window.display_image_with_boxes(0)
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
    assert window.storage_service.load_calibration(source_file) == 10.0

    window.close()
    if created:
        app.quit()


def test_switching_images_restores_calibration_per_source_file(
    tmp_path,
    monkeypatch,
):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)
    module = _import_main_window_module(monkeypatch)
    monkeypatch.setattr(
        module.ImageEditor,
        "_start_model_loading",
        lambda self: None,
    )

    window = module.ImageEditor("dummy_weights.pt")
    first_file = tmp_path / "first.png"
    second_file = tmp_path / "second.pdf"
    first_file.write_text("first", encoding="utf-8")
    second_file.write_text("second", encoding="utf-8")

    window.storage_service = module.StorageService(tmp_path / "storage")
    window.storage_service.save_calibration(first_file, 6.5)
    window.image_storage.images = [
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.zeros((64, 64, 3), dtype=np.uint8),
    ]
    window.image_storage.class_object_image = [[], []]
    window.image_storage.source_files = [
        str(first_file),
        str(second_file),
    ]
    window.image_storage.file_path = str(first_file)

    window.display_image_with_boxes(0)
    assert abs(window.pixels_per_mm - 6.5) < 1e-9

    window.display_image_with_boxes(1)
    assert (
        abs(window.pixels_per_mm - module.CALIBRATION_PIXELS_PER_MM_DEFAULT)
        < 1e-9
    )

    window.display_image_with_boxes(0)
    assert abs(window.pixels_per_mm - 6.5) < 1e-9

    window.close()
    if created:
        app.quit()


def test_pdf_load_result_appends_pages_with_source_mapping(
    tmp_path,
    monkeypatch,
):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)
    module = _import_main_window_module(monkeypatch)
    monkeypatch.setattr(
        module.ImageEditor,
        "_start_model_loading",
        lambda self: None,
    )

    window = module.ImageEditor("dummy_weights.pt")
    base_file = tmp_path / "image.png"
    pdf_file = tmp_path / "batch.pdf"
    base_file.write_text("base", encoding="utf-8")
    pdf_file.write_text("pdf", encoding="utf-8")

    window.image_storage.images = [np.zeros((32, 32, 3), dtype=np.uint8)]
    window.image_storage.class_object_image = [[]]
    window.image_storage.source_files = [str(base_file)]
    window.image_storage.file_path = str(base_file)

    window._on_pdf_load_result(
        str(pdf_file),
        [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((24, 24, 3), dtype=np.uint8),
        ],
    )

    assert len(window.image_storage.images) == 3
    assert len(window.image_storage.class_object_image) == 3
    assert window.image_storage.source_files == [
        str(base_file),
        str(pdf_file),
        str(pdf_file),
    ]
    assert window.tree_widget.topLevelItemCount() == 2

    window.close()
    if created:
        app.quit()


def test_detection_results_list_focuses_bbox_on_click(tmp_path, monkeypatch):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)
    module = _import_main_window_module(monkeypatch)
    monkeypatch.setattr(
        module.ImageEditor,
        "_start_model_loading",
        lambda self: None,
    )

    window = module.ImageEditor("dummy_weights.pt")
    window.image_storage.images = [np.zeros((200, 200, 3), dtype=np.uint8)]
    window.image_storage.class_object_image = [[
        module.ObjectImage(
            class_name="seeding",
            confidence=0.91,
            image=[],
            bbox=(10, 10, 50, 70),
        ),
        module.ObjectImage(
            class_name="seeding",
            confidence=0.84,
            image=[],
            bbox=(120, 110, 170, 180),
        ),
    ]]
    window.display_image_with_boxes(0)

    assert window.results_list.count() == 2

    item = window.results_list.item(1)
    window._on_result_item_clicked(item)

    assert window._active_result_page_index == 0
    assert window._active_result_index == 1
    assert window.results_list.currentRow() == 1
    assert window._result_bbox_items[1]._highlighted is True
    assert window.zoom_factor >= window.min_fit_zoom

    window.close()
    if created:
        app.quit()
