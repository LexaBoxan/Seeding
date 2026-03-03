import os

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication

from seeding.config import (
    CALIBRATION_PIXELS_PER_MM_DEFAULT,
    DEFAULT_CLASSIFY_WEIGHTS_PATH,
    DEFAULT_WEIGHTS_PATH,
    DETECTION_CONFIDENCE_THRESHOLD,
    DETECTION_IOU_THRESHOLD,
    PDF_RENDER_SCALE_DEFAULT,
    QSETTINGS_APP,
    QSETTINGS_ORG,
    USE_CACHE_DEFAULT,
)
from seeding.ui.preferences import DEFAULT_UI_LANGUAGE, DEFAULT_UI_THEME
from seeding.ui.settings_dialog import SettingsDialog


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


def test_settings_dialog_saves_detection_and_model_settings(tmp_path):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)

    detect_weights = tmp_path / "models" / "detect.pt"
    classify_weights = tmp_path / "models" / "classify.pt"
    detect_weights.parent.mkdir(parents=True, exist_ok=True)
    detect_weights.write_bytes(b"detect")
    classify_weights.write_bytes(b"classify")

    dialog = SettingsDialog()
    dialog.spin_high.setValue(0.91)
    dialog.spin_low.setValue(0.52)
    dialog.spin_detect_conf.setValue(0.37)
    dialog.spin_detect_iou.setValue(0.44)
    dialog.spin_pixels_per_mm.setValue(7.5)
    dialog.spin_pdf_render_scale.setValue(2.5)
    dialog.check_use_cache.setChecked(False)
    dialog.detect_weights_edit.setText(str(detect_weights))
    dialog.classify_weights_edit.setText(str(classify_weights))
    dialog.report_dir_edit.setText(str(tmp_path))

    assert dialog.save_settings() is True

    settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    assert float(settings.value("conf_high")) == 0.91
    assert float(settings.value("conf_low")) == 0.52
    assert float(settings.value("detect_conf")) == 0.37
    assert float(settings.value("detect_iou")) == 0.44
    assert float(settings.value("pixels_per_mm")) == 7.5
    assert float(settings.value("pdf_render_scale")) == 2.5
    assert settings.value("use_cache", type=bool) is False
    assert settings.value("detect_weights_path", type=str) == str(
        detect_weights.resolve()
    )
    assert settings.value("classify_weights_path", type=str) == str(
        classify_weights.resolve()
    )
    assert settings.value("report_dir", type=str) == str(tmp_path)

    dialog.close()
    if created:
        app.quit()


def test_settings_dialog_reset_defaults_restores_default_values(tmp_path):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)

    dialog = SettingsDialog()
    dialog.spin_high.setValue(0.99)
    dialog.spin_low.setValue(0.22)
    dialog.spin_detect_conf.setValue(0.4)
    dialog.spin_detect_iou.setValue(0.3)
    dialog.spin_pixels_per_mm.setValue(10.0)
    dialog.spin_pdf_render_scale.setValue(6.0)
    dialog.check_use_cache.setChecked(False)
    dialog.detect_weights_edit.setText("custom_detect.pt")
    dialog.classify_weights_edit.setText("custom_classify.pt")
    dialog.theme_combo.setCurrentIndex(
        dialog.theme_combo.findData("light")
    )
    dialog.language_combo.setCurrentIndex(
        dialog.language_combo.findData("en")
    )
    dialog.report_dir_edit.setText(str(tmp_path))

    dialog._reset_defaults()

    assert dialog.spin_detect_conf.value() == DETECTION_CONFIDENCE_THRESHOLD
    assert dialog.spin_detect_iou.value() == DETECTION_IOU_THRESHOLD
    assert (
        dialog.spin_pixels_per_mm.value()
        == CALIBRATION_PIXELS_PER_MM_DEFAULT
    )
    assert dialog.spin_pdf_render_scale.value() == PDF_RENDER_SCALE_DEFAULT
    assert dialog.check_use_cache.isChecked() is USE_CACHE_DEFAULT
    assert dialog.detect_weights_edit.text() == str(DEFAULT_WEIGHTS_PATH)
    assert dialog.classify_weights_edit.text() == str(
        DEFAULT_CLASSIFY_WEIGHTS_PATH
    )
    assert dialog.theme_combo.currentData() == DEFAULT_UI_THEME
    assert dialog.language_combo.currentData() == DEFAULT_UI_LANGUAGE
    assert dialog.report_dir_edit.text() == ""

    dialog.close()
    if created:
        app.quit()


def test_settings_dialog_accepts_onnx_model_paths(tmp_path):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)

    detect_weights = tmp_path / "models" / "detect.onnx"
    classify_weights = tmp_path / "models" / "classify.onnx"
    detect_weights.parent.mkdir(parents=True, exist_ok=True)
    detect_weights.write_bytes(b"detect-onnx")
    classify_weights.write_bytes(b"classify-onnx")

    dialog = SettingsDialog()
    dialog.detect_weights_edit.setText(str(detect_weights))
    dialog.classify_weights_edit.setText(str(classify_weights))

    assert dialog.save_settings() is True

    settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    assert settings.value("detect_weights_path", type=str) == str(
        detect_weights.resolve()
    )
    assert settings.value("classify_weights_path", type=str) == str(
        classify_weights.resolve()
    )

    dialog.close()
    if created:
        app.quit()


def test_settings_dialog_can_request_interactive_calibration(tmp_path):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)

    dialog = SettingsDialog()
    assert dialog.calibration_requested is False

    dialog._request_calibration()

    assert dialog.calibration_requested is True
    assert dialog.result() == dialog.Accepted

    dialog.close()
    if created:
        app.quit()


def test_settings_dialog_can_apply_recommended_model_paths(tmp_path):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)

    dialog = SettingsDialog()
    dialog._apply_recommended_detection_model()
    dialog._apply_recommended_classification_model()

    assert dialog.detect_weights_edit.text().endswith("models/bestCrop.pt")
    assert dialog.classify_weights_edit.text().endswith(
        "models/bestKlassSeg.pt"
    )
    assert "Рекомендуемая модель" in dialog.detect_model_status_label.text()
    assert "Рекомендуемая модель" in dialog.classify_model_status_label.text()

    dialog.close()
    if created:
        app.quit()
