"""Пакет `seeding` с графическим приложением для работы с изображениями."""

from seeding.application import (  # noqa: F401  - упрощаем доступ к конвейеру
    RootAnalyzer,
    RootAnalysisResult,
    RootMorphology,
    RootViability,
    SeedlingPipeline,
)

__all__ = [
    "main",
    "RootAnalyzer",
    "RootAnalysisResult",
    "RootMorphology",
    "RootViability",
    "SeedlingPipeline",
]


def main() -> None:
    """Точка входа для запуска из командной строки."""
    from .main import main as _main

    _main()
