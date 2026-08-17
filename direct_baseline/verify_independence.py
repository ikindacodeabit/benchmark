#!/usr/bin/env python3
"""Statically verify that direct inference cannot register KVPress hooks."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BANNED_CALLS = {"register_forward_hook", "register_forward_pre_hook"}


def main() -> None:
    errors: list[str] = []
    for path in sorted(ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "kvpress" or alias.name.startswith("kvpress."):
                        errors.append(f"{path.name}:{node.lineno}: imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "kvpress" or module.startswith("kvpress."):
                    errors.append(f"{path.name}:{node.lineno}: imports {module}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in BANNED_CALLS:
                    errors.append(f"{path.name}:{node.lineno}: calls {node.func.attr}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("PASS: no kvpress imports and no hook registration calls")


if __name__ == "__main__":
    main()
