import ast
from pathlib import Path

from seeding.ui.i18n import TRANSLATIONS


def _collect_translation_keys() -> set[str]:
    keys: set[str] = set()
    root = Path("seeding")
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None
            )
            if (
                name == "tr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                keys.add(node.args[1].value)
            if (
                name == "_tr"
                and len(node.args) >= 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
            if (
                name == "_t"
                and len(node.args) >= 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
    return keys


def test_all_runtime_translation_keys_exist_for_ru_and_en():
    runtime_keys = _collect_translation_keys()
    for language in ("ru", "en"):
        locale_keys = set(TRANSLATIONS[language])
        missing = sorted(runtime_keys - locale_keys)
        assert missing == []
