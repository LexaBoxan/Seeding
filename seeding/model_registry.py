"""Helpers for working with known bundled and custom model files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from seeding.config import PROJECT_ROOT
from seeding.utils import resolve_weights_path

MODEL_REGISTRY_PATH = PROJECT_ROOT / "models" / "models.json"


@dataclass(frozen=True)
class ModelSpec:
    """Static model metadata from the local registry."""

    key: str
    role: str
    label: str
    path: str
    recommended: bool = False
    aliases: tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        """Returns the file name part of the model path."""
        return Path(self.path).name


@dataclass(frozen=True)
class ModelStatus:
    """Resolved model information for UI and validation."""

    reference: str
    resolved_path: Path | None
    exists: bool
    size_bytes: int | None
    spec: ModelSpec | None = None

    @property
    def filename(self) -> str:
        """Returns the resolved or referenced file name."""
        if self.resolved_path is not None:
            return self.resolved_path.name
        if self.spec is not None:
            return self.spec.filename
        return Path(self.reference).name or self.reference

    @property
    def label(self) -> str:
        """Returns the best available display label."""
        if self.spec is not None:
            return self.spec.label
        return self.filename

    @property
    def recommended(self) -> bool:
        """Returns whether the resolved model is recommended."""
        return bool(self.spec and self.spec.recommended)

    @property
    def is_catalog_model(self) -> bool:
        """Returns whether the model is defined in the registry."""
        return self.spec is not None


def _normalize_reference(value: str) -> str:
    """Normalizes a model key, alias or path for comparisons."""
    return value.strip().replace("\\", "/").lower()


def _registry_items_from_payload(payload: object) -> list[dict[str, object]]:
    """Extracts model item dictionaries from registry JSON."""
    if isinstance(payload, dict):
        items = payload.get("models", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def load_model_registry(
    registry_path: str | Path | None = None,
) -> list[ModelSpec]:
    """Loads known model metadata from the local JSON registry."""
    path = Path(registry_path) if registry_path is not None else MODEL_REGISTRY_PATH
    if not path.is_file():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    specs: list[ModelSpec] = []
    for item in _registry_items_from_payload(payload):
        key = str(item.get("key", "")).strip()
        role = str(item.get("role", "")).strip()
        label = str(item.get("label", "")).strip()
        model_path = str(item.get("path", "")).strip()
        if not key or not role or not label or not model_path:
            continue
        aliases = tuple(
            str(alias).strip()
            for alias in item.get("aliases", [])
            if str(alias).strip()
        )
        specs.append(
            ModelSpec(
                key=key,
                role=role,
                label=label,
                path=model_path,
                recommended=bool(item.get("recommended", False)),
                aliases=aliases,
            )
        )
    return specs


def iter_model_specs(
    *,
    role: str | None = None,
    registry_path: str | Path | None = None,
) -> list[ModelSpec]:
    """Returns all registry models or only models for a specific role."""
    specs = load_model_registry(registry_path)
    if role is None:
        return specs
    return [spec for spec in specs if spec.role == role]


def find_model_spec(
    reference: str,
    *,
    role: str | None = None,
    registry_path: str | Path | None = None,
) -> ModelSpec | None:
    """Finds a registry model by key, alias, relative path or file name."""
    normalized = _normalize_reference(reference)
    if not normalized:
        return None

    for spec in iter_model_specs(role=role, registry_path=registry_path):
        variants = {
            _normalize_reference(spec.key),
            _normalize_reference(spec.path),
            _normalize_reference(spec.filename),
        }
        variants.update(_normalize_reference(alias) for alias in spec.aliases)
        if normalized in variants:
            return spec
    return None


def get_recommended_model_spec(
    role: str,
    *,
    registry_path: str | Path | None = None,
) -> ModelSpec | None:
    """Returns the recommended registry model for the given role."""
    for spec in iter_model_specs(role=role, registry_path=registry_path):
        if spec.recommended:
            return spec
    return None


def resolve_model_reference(
    reference: str,
    *,
    role: str | None = None,
    base_dirs: Iterable[Path] | None = None,
    registry_path: str | Path | None = None,
) -> Path | None:
    """Resolves a model path, filename or registry alias to an actual file."""
    raw_reference = reference.strip()
    if not raw_reference and role is not None:
        spec = get_recommended_model_spec(role, registry_path=registry_path)
        raw_reference = spec.path if spec is not None else ""
    if not raw_reference:
        return None

    if base_dirs is None:
        base_dirs = (PROJECT_ROOT, Path.cwd())
    else:
        base_dirs = tuple(base_dirs)

    spec = find_model_spec(raw_reference, role=role, registry_path=registry_path)
    if spec is not None:
        resolved = resolve_weights_path(spec.path, base_dirs=base_dirs)
        if resolved is not None:
            return resolved
    return resolve_weights_path(raw_reference, base_dirs=base_dirs)


def inspect_model_reference(
    reference: str,
    *,
    role: str | None = None,
    base_dirs: Iterable[Path] | None = None,
    registry_path: str | Path | None = None,
) -> ModelStatus:
    """Builds an inspectable model status for the provided reference."""
    raw_reference = reference.strip()
    spec = find_model_spec(raw_reference, role=role, registry_path=registry_path)
    resolved = resolve_model_reference(
        raw_reference,
        role=role,
        base_dirs=base_dirs,
        registry_path=registry_path,
    )
    exists = resolved is not None and resolved.is_file()
    size_bytes = resolved.stat().st_size if exists and resolved is not None else None
    return ModelStatus(
        reference=raw_reference,
        resolved_path=resolved,
        exists=exists,
        size_bytes=size_bytes,
        spec=spec,
    )


def build_model_selector_options(
    role: str,
    *,
    base_dirs: Iterable[Path] | None = None,
    registry_path: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Returns existing registry-backed models as selector options."""
    options: list[tuple[str, str]] = []
    for spec in iter_model_specs(role=role, registry_path=registry_path):
        resolved = resolve_model_reference(
            spec.path,
            role=role,
            base_dirs=base_dirs,
            registry_path=registry_path,
        )
        if resolved is None:
            continue
        label = spec.label
        if spec.recommended:
            label = f"{label} [recommended]"
        options.append((label, str(resolved)))
    return options


def format_size_mb(size_bytes: int | None) -> str:
    """Formats a file size in megabytes for UI display."""
    if size_bytes is None:
        return "0.0 MB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


__all__ = [
    "MODEL_REGISTRY_PATH",
    "ModelSpec",
    "ModelStatus",
    "build_model_selector_options",
    "find_model_spec",
    "format_size_mb",
    "get_recommended_model_spec",
    "inspect_model_reference",
    "iter_model_specs",
    "load_model_registry",
    "resolve_model_reference",
]
