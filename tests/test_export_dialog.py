import os

from PyQt5.QtWidgets import QApplication

from seeding.ui.export_dialog import ExportDialog


def _ensure_offscreen_qt() -> tuple[QApplication, bool]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created = app is None
    if app is None:
        app = QApplication([])
    return app, created


def test_export_dialog_review_preset_is_default():
    app, created = _ensure_offscreen_qt()

    dialog = ExportDialog(default_dir=".")

    assert dialog.selected_preset == "review"
    assert dialog.options == {
        "json": True,
        "csv": True,
        "coco": False,
        "yolo": False,
        "annotated": True,
        "metadata": True,
    }

    dialog.close()
    if created:
        app.quit()


def test_export_dialog_dataset_preset_and_manual_toggle_switch_to_custom():
    app, created = _ensure_offscreen_qt()

    dialog = ExportDialog(default_dir=".")
    dataset_index = dialog.preset_combo.findData("dataset")
    dialog.preset_combo.setCurrentIndex(dataset_index)

    assert dialog.selected_preset == "dataset"
    assert dialog.options == {
        "json": False,
        "csv": False,
        "coco": True,
        "yolo": True,
        "annotated": False,
        "metadata": True,
    }

    dialog.chk_json.setChecked(True)

    assert dialog.selected_preset == "custom"
    assert dialog.options["json"] is True

    dialog.close()
    if created:
        app.quit()


def test_export_dialog_localizes_custom_labels_for_english():
    app, created = _ensure_offscreen_qt()

    dialog = ExportDialog(default_dir=".", language="en")

    assert dialog.windowTitle() == "Export results"
    assert dialog.chk_annotated.text() == "Annotated images"
    assert dialog.chk_metadata.text() == "Metadata"
    assert dialog.preset_combo.itemText(0) == "Review (Recommended)"

    dialog.close()
    if created:
        app.quit()
