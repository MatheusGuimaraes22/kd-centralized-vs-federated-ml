from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.ckd_paper_kfold import load_data  # type: ignore


def infer_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols, cat_cols = [], []
    for col in X.columns:
        vc = pd.to_numeric(X[col], errors="coerce")
        if vc.notna().mean() >= 0.5:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return num_cols, cat_cols


def map_feature_to_column(name: str, cat_cols: list[str]) -> str:
    if name.startswith("num__"):
        return name.split("__", 1)[1]
    if name.startswith("cat__"):
        rest = name.split("__", 1)[1]
        for col in cat_cols:
            prefix = f"{col}_"
            if rest.startswith(prefix):
                return col
        return rest.split("_", 1)[0]
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description="Seleciona top-k features por chi-square para drop.")
    ap.add_argument("--k", type=int, default=6, help="Quantidade de features a remover.")
    ap.add_argument("--out", default=str(ROOT / "results" / "chi2_topk_drop.json"))
    args = ap.parse_args()

    X, y = load_data()
    num_cols, cat_cols = infer_columns(X)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    X_proc = pre.fit_transform(X)
    scores, _ = chi2(X_proc, y)
    feat_names = pre.get_feature_names_out()

    col_scores: dict[str, float] = {c: 0.0 for c in X.columns}
    for name, score in zip(feat_names, scores):
        col = map_feature_to_column(name, cat_cols)
        if col in col_scores:
            col_scores[col] += float(score)

    ranked = sorted(col_scores.items(), key=lambda x: x[1], reverse=True)
    topk = [c for c, _ in ranked[: args.k]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "method": "chi2",
        "k": args.k,
        "drop_cols": topk,
        "scores": ranked,
    }
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")

    txt_path = out_path.with_suffix(".txt")
    txt_path.write_text("\n".join(topk) + "\n", encoding="utf-8")
    print(f"Salvo: {out_path}")
    print(f"Salvo: {txt_path}")


if __name__ == "__main__":
    main()
