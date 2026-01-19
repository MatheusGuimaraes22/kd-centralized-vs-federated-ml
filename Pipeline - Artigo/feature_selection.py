from __future__ import annotations

import json
from pathlib import Path
from typing import List


def load_drop_cols(
    default: List[str],
    k: int = 6,
    method: str = "chi2",
    path: Path | None = None,
) -> List[str]:
    root = Path(__file__).resolve().parents[1]
    if path is None:
        path = root / "results" / f"{method}_topk_drop.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        cols = data.get("drop_cols")
        if isinstance(cols, list) and cols:
            return cols
    return default


def build_scenarios(drop_cols: List[str], include_opt: bool = True) -> List[dict]:
    base = [
        {"label": "all_nosmote", "use_smote": False, "drop": []},
        {"label": "all_smote", "use_smote": True, "drop": []},
        {"label": "drop_nosmote", "use_smote": False, "drop": drop_cols},
        {"label": "drop_smote", "use_smote": True, "drop": drop_cols},
    ]
    if not include_opt:
        return base
    extra = [
        {"label": f"{b['label']}_opt", "use_smote": b["use_smote"], "drop": b["drop"], "base_label": b["label"]}
        for b in base
    ]
    for b in base:
        b["base_label"] = b["label"]
    return base + extra
