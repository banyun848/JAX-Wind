"""Dependency checks run on the compute node with the verification suite."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src/jaxwind"


def test_installed_package_does_not_import_applications():
    for path in ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [item.name for item in node.names] if isinstance(node, ast.Import) else []
            assert not any(name == "applications" or name.startswith("applications.") for name in names), str(path)


def test_numerical_modules_do_not_depend_on_execution_layers():
    forbidden = ("jaxwind.config", "jaxwind.simulation", "jaxwind.runtime", "jaxwind.workflows", "jaxwind.cli")
    for directory in (ROOT / "numerics", ROOT / "formulations"):
        for path in directory.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [item.name for item in node.names] if isinstance(node, ast.Import) else []
                assert not any(name.startswith(forbidden) for name in names), str(path)
