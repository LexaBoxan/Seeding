from pathlib import Path

from seeding.path_utils import ensure_dir, resolve_weights_path


def test_resolve_weights_path_supports_unix_and_windows_separators(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    weights = models_dir / "bestCrop.pt"
    weights.write_bytes(b"weights")

    unix_style = "models/bestCrop.pt"
    windows_style = "models\\bestCrop.pt"

    resolved_unix = resolve_weights_path(unix_style, base_dirs=(tmp_path,))
    resolved_windows = resolve_weights_path(
        windows_style,
        base_dirs=(tmp_path,),
    )

    assert resolved_unix == weights.resolve()
    assert resolved_windows == weights.resolve()


def test_resolve_weights_path_finds_name_without_extension(tmp_path):
    weights = tmp_path / "bestCrop.pt"
    weights.write_bytes(b"weights")

    resolved = resolve_weights_path("bestCrop", base_dirs=(tmp_path,))
    assert resolved == weights.resolve()


def test_resolve_weights_path_rejects_missing_name_without_extension(tmp_path):
    assert resolve_weights_path("missing_model", base_dirs=(tmp_path,)) is None


def test_resolve_weights_path_keeps_pt_alias_when_no_local_file(tmp_path):
    resolved = resolve_weights_path("yolo11n.pt", base_dirs=(tmp_path,))
    assert isinstance(resolved, Path)
    assert resolved.name == "yolo11n.pt"


def test_ensure_dir_creates_nested_directory(tmp_path):
    directory = ensure_dir(tmp_path / "reports" / "daily")
    assert directory.is_dir()
