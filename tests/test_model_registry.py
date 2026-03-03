import json

from seeding.model_registry import (
    build_model_selector_options,
    find_model_spec,
    get_recommended_model_spec,
    inspect_model_reference,
    resolve_model_reference,
)


def test_resolve_model_reference_supports_registry_aliases(tmp_path):
    registry_path = tmp_path / "models.json"
    model_path = tmp_path / "models" / "detector.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"detector")
    registry_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "key": "seed-detector",
                        "role": "detect",
                        "label": "Seed detector",
                        "path": "models/detector.pt",
                        "recommended": True,
                        "aliases": ["bestCrop", "detect-default"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_reference(
        "detect-default",
        role="detect",
        base_dirs=(tmp_path,),
        registry_path=registry_path,
    )

    assert resolved == model_path.resolve()


def test_inspect_model_reference_reports_catalog_metadata(tmp_path):
    registry_path = tmp_path / "models.json"
    model_path = tmp_path / "models" / "segmenter.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"classify-data")
    registry_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "key": "seed-seg",
                        "role": "classify",
                        "label": "Seed segmenter",
                        "path": "models/segmenter.pt",
                        "recommended": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    status = inspect_model_reference(
        "seed-seg",
        role="classify",
        base_dirs=(tmp_path,),
        registry_path=registry_path,
    )

    assert status.exists is True
    assert status.recommended is True
    assert status.is_catalog_model is True
    assert status.label == "Seed segmenter"
    assert status.filename == "segmenter.pt"
    assert status.size_bytes == len(b"classify-data")


def test_build_model_selector_options_returns_existing_recommended_entries(tmp_path):
    registry_path = tmp_path / "models.json"
    recommended_path = tmp_path / "models" / "detector.pt"
    optional_path = tmp_path / "models" / "detector2.pt"
    recommended_path.parent.mkdir(parents=True)
    recommended_path.write_bytes(b"a")
    optional_path.write_bytes(b"b")
    registry_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "key": "seed-detector",
                        "role": "detect",
                        "label": "Seed detector",
                        "path": "models/detector.pt",
                        "recommended": True,
                    },
                    {
                        "key": "seed-detector-2",
                        "role": "detect",
                        "label": "Seed detector 2",
                        "path": "models/detector2.pt",
                        "recommended": False,
                    },
                    {
                        "key": "missing-detector",
                        "role": "detect",
                        "label": "Missing detector",
                        "path": "models/missing.pt",
                        "recommended": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    options = build_model_selector_options(
        "detect",
        base_dirs=(tmp_path,),
        registry_path=registry_path,
    )

    assert options == [
        ("Seed detector [recommended]", str(recommended_path.resolve())),
        ("Seed detector 2", str(optional_path.resolve())),
    ]

    spec = get_recommended_model_spec("detect", registry_path=registry_path)
    assert spec is not None
    assert spec.key == "seed-detector"
    assert find_model_spec(
        "detector.pt",
        role="detect",
        registry_path=registry_path,
    ) == spec
