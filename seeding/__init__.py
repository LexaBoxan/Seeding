"""ImageEditor package."""

__all__ = ["main"]


def main() -> None:
    """Точка входа для запуска из командной строки."""
    from .main import main as _main

    _main()
