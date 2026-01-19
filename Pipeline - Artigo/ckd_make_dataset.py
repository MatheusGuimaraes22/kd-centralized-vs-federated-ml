#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepara o dataset CKD limpo, aplica split holdout e serializa o pipeline global.

Outputs gerados em --outdir (padrão: data/processed):
- ckd_clean.csv            -> dataset limpo completo (retrocompatibilidade)
- ckd_train.csv            -> dados usados para treino/FL (sem holdout)
- holdout_raw.csv          -> registros reservados para teste externo
- holdout_preprocessed.csv -> holdout transformado via pipeline global
- preprocess.joblib        -> ColumnTransformer ajustado nos dados de treino
- feature_schema.txt       -> resumo das colunas e das features derivadas
- schema.json / metadata.json com detalhes do processamento
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_arff(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cols, data_start = [], None
    for i, line in enumerate(text):
        ls = line.strip()
        if not ls or ls.startswith("%"):
            continue
        if ls.lower().startswith("@attribute"):
            parts = ls.split()
            if len(parts) >= 2:
                cols.append(parts[1].strip("'\""))
        if ls.lower().startswith("@data"):
            data_start = i + 1
            break
    if not cols or data_start is None:
        raise ValueError("ARFF inválido: não foi possível detectar atributos/@data.")

    data_lines = [l for l in text[data_start:] if l.strip() and not l.strip().startswith("%")]

    kept, skipped = [], 0
    for ln in data_lines:
        row = [c.strip() for c in ln.split(",")]
        if len(row) > len(cols) and row[-1] == "":
            row = row[:-1]
        if len(row) != len(cols):
            skipped += 1
            continue
        kept.append(",".join(row))
    if skipped:
        print(f"Aviso: linhas de dados puladas por formato inconsistente: {skipped}")
    return pd.read_csv(StringIO("\n".join(kept)), header=None, names=cols)


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"?": np.nan, "": np.nan, "None": np.nan, "nan": np.nan})
            )

    target_col = None
    for cand in ["class", "Class", "CLASS", "ckd_class", "target"]:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        target_col = df.columns[-1]

    y_raw = df[target_col].astype(str).str.lower().str.replace(r"[^a-z0-9_]+", "", regex=True)
    y = y_raw.map({"ckd": 1, "notckd": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Não foi possível mapear o alvo para {0,1}.")

    X = df.drop(columns=[target_col]).copy()
    num_cols, cat_cols = [], []
    for col in X.columns:
        vc = pd.to_numeric(X[col], errors="coerce")
        if vc.notna().mean() >= 0.5:
            num_cols.append(col)
            X[col] = vc
        else:
            cat_cols.append(col)
            X[col] = X[col].astype(str)

    schema = {
        "target": target_col,
        "mapping": {"ckd": 1, "notckd": 0, "1": 1, "0": 0, "yes": 1, "no": 0},
        "numeric": num_cols,
        "categorical": cat_cols,
        "n_rows": int(len(df)),
    }
    X["target"] = y.astype(int).values
    return X, schema


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def feature_names(pre: ColumnTransformer, sample_df: pd.DataFrame) -> list[str]:
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        transformed = pre.transform(sample_df.iloc[:1])
        return [f"f{i}" for i in range(transformed.shape[1])]


def write_schema_txt(path: Path, schema: dict, derived_features: Iterable[str]) -> None:
    lines = [
        "# CKD feature schema",
        f"target: {schema['target']}",
        f"numeric ({len(schema['numeric'])}): {', '.join(schema['numeric'])}",
        f"categorical ({len(schema['categorical'])}): {', '.join(schema['categorical'])}",
        "derived_features:",
    ]
    lines.extend(f"  - {name}" for name in derived_features)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Limpa CKD, cria holdout e serializa o pipeline global.")
    ap.add_argument("--in", dest="inp", required=True, help="ARFF/CSV bruto (UCI CKD)")
    ap.add_argument("--outdir", default="data/processed", help="Pasta de saída")
    ap.add_argument("--holdout-size", type=float, default=0.2, help="Proporção reservada para holdout externo")
    ap.add_argument("--seed", type=int, default=42, help="Seed para o split estratificado")
    args = ap.parse_args()

    inp = Path(args.inp)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if inp.suffix.lower() == ".arff":
        df_raw = read_arff(inp)
    else:
        df_raw = pd.read_csv(inp)

    df_clean, schema = normalize(df_raw)

    X = df_clean.drop(columns=["target"])
    y = df_clean["target"].values

    holdout_size = max(0.0, min(0.5, float(args.holdout_size)))
    if holdout_size > 0:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=holdout_size, random_state=args.seed)
        train_idx, holdout_idx = next(splitter.split(X, y))
    else:
        train_idx = np.arange(len(df_clean))
        holdout_idx = np.array([], dtype=int)

    train_df = df_clean.iloc[train_idx].reset_index(drop=True)
    holdout_df = df_clean.iloc[holdout_idx].reset_index(drop=True)

    pre = build_preprocessor(schema["numeric"], schema["categorical"])
    pre.fit(train_df.drop(columns=["target"]))
    joblib.dump(pre, outdir / "preprocess.joblib")

    derived = feature_names(pre, train_df.drop(columns=["target"]))
    write_schema_txt(outdir / "feature_schema.txt", schema, derived)

    df_clean.to_csv(outdir / "ckd_clean.csv", index=False, encoding="utf-8")
    train_df.to_csv(outdir / "ckd_train.csv", index=False, encoding="utf-8")
    (outdir / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "holdout_size": holdout_size,
        "seed": args.seed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "source": inp.as_posix(),
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if len(holdout_df):
        holdout_df.to_csv(outdir / "holdout_raw.csv", index=False, encoding="utf-8")
        holdout_trans = pre.transform(holdout_df.drop(columns=["target"]))
        holdout_proc = pd.DataFrame(holdout_trans, columns=derived)
        holdout_proc.insert(0, "target", holdout_df["target"].values)
        holdout_proc.to_csv(outdir / "holdout_preprocessed.csv", index=False, encoding="utf-8")
    else:
        for fname in ["holdout_raw.csv", "holdout_preprocessed.csv"]:
            path = outdir / fname
            if path.exists():
                path.unlink()

    print(f"OK -> {outdir.as_posix()}")
    print(" - ckd_clean.csv (dataset completo)")
    print(" - ckd_train.csv (sem holdout)")
    if len(holdout_df):
        print(" - holdout_raw.csv / holdout_preprocessed.csv")
    print(" - preprocess.joblib / feature_schema.txt")
    print(" - schema.json / metadata.json")


if __name__ == "__main__":
    main()
