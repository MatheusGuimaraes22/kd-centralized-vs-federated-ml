from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
ARFF_PATH = ROOT / "Chronic_Kidney_Disease" / "chronic_kidney_disease_full.arff"
RANDOM_STATE = 42

def _parse_arff(path: Path) -> pd.DataFrame:
    """Lightweight ARFF reader that strips whitespace and handles trailing commas."""
    attr_names: list[str] = []
    data_rows: list[list[str]] = []
    in_data = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not in_data:
                if line.lower().startswith("@attribute"):
                    parts = line.split()
                    if len(parts) >= 2:
                        attr_names.append(parts[1].strip("'\""))
                if line.lower().startswith("@data"):
                    in_data = True
                continue

            if not line or line.startswith("%"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) > len(attr_names) and parts[-1] == "":
                parts = parts[:-1]
            if len(parts) == len(attr_names):
                data_rows.append(parts)

    df = pd.DataFrame(data_rows, columns=attr_names)
    df = df.replace("?", np.nan)
    return df

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = _parse_arff(ARFF_PATH)

    numeric_cols = [
        "age",
        "bp",
        "sg",
        "al",
        "su",
        "bgr",
        "bu",
        "sc",
        "sod",
        "pot",
        "hemo",
        "pcv",
        "wbcc",
        "rbcc",
    ]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ["class"]]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[categorical_cols] = df[categorical_cols].astype("category")
    df["class"] = df["class"].str.lower()

    X = df.drop(columns=["class"])
    y = df["class"].map({"ckd": 1, "notckd": 0})
    return X, y

def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = ImbPipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

def evaluate_models(X: pd.DataFrame, y: pd.Series, use_smote: bool = False) -> None:
    numeric_cols = [
        "age",
        "bp",
        "sg",
        "al",
        "su",
        "bgr",
        "bu",
        "sc",
        "sod",
        "pot",
        "hemo",
        "pcv",
        "wbcc",
        "rbcc",
    ]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    models = {
        "logreg": LogisticRegression(
            max_iter=500, n_jobs=-1, random_state=RANDOM_STATE, class_weight=None
        ),
        "logreg_balanced": LogisticRegression(
            max_iter=500, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight=None,
        ),
    }

    sampler = SMOTE(random_state=RANDOM_STATE) if use_smote else None
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "recall": "recall",
        "balanced_accuracy": "balanced_accuracy",
    }

    print("\n--- Using SMOTE ---" if use_smote else "\n--- Baseline (no resampling) ---")
    for name, model in models.items():
        steps = [("prep", preprocessor)]
        if sampler:
            steps.append(("smote", sampler))
        steps.append(("model", model))
        pipe = ImbPipeline(steps=steps)

        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        print(f"\n{name}")
        for metric, values in scores.items():
            if metric.startswith("test_"):
                vals = np.array(values)
                print(f"  {metric[5:]}: {vals.mean():.3f} ± {vals.std():.3f}")

if __name__ == "__main__":
    X, y = load_data()
    evaluate_models(X, y, use_smote=False)
    evaluate_models(X, y, use_smote=True)
