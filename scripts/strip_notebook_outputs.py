#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from typing import Any


def _strip_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    notebook = dict(notebook)
    metadata = dict(notebook.get("metadata") or {})
    metadata.pop("widgets", None)
    notebook["metadata"] = metadata

    cells = []
    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict):
            cells.append(cell)
            continue
        cleaned = dict(cell)
        if cleaned.get("cell_type") == "code":
            cleaned["execution_count"] = None
            cleaned["outputs"] = []
        cell_metadata = dict(cleaned.get("metadata") or {})
        for key in ("execution", "ExecuteTime"):
            cell_metadata.pop(key, None)
        cleaned["metadata"] = cell_metadata
        cells.append(cleaned)
    notebook["cells"] = cells
    return notebook


def main() -> int:
    raw = sys.stdin.read()
    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError:
        sys.stdout.write(raw)
        return 0
    cleaned = _strip_notebook(notebook)
    sys.stdout.write(json.dumps(cleaned, ensure_ascii=False, indent=1, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
