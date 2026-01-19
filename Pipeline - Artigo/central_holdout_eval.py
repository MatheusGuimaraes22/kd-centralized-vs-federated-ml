from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    precision_score,
    accuracy_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.ckd_paper_kfold import load_data, build_preprocessor  # type: ignore
from scripts.feature_selection import build_scenarios, load_drop_cols  # type: ignore

try:
    from fedtree import FLClassifier  # type: ignore
    FEDTREE_AVAILABLE = True
except Exception:
    FEDTREE_AVAILABLE = False

try:
    from xgboost import XGBClassifier  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

RANDOM_STATE = 42
TEST_SIZE = 0.2
DROP_COLS = load_drop_cols(["sg", "pcc", "pcv", "hemo", "rbcc", "al"], k=6)
NUMERIC_ALL = [
    "age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"
]


def prepare_data(use_smote: bool, drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    X, y = load_data()
    if drop_cols:
        X = X.drop(columns=drop_cols)
    X = X.loc[:, X.notna().any()]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    if use_smote:
        X_train_enc = pd.get_dummies(X_train, dummy_na=True).fillna(0)
        X_test_enc = pd.get_dummies(X_test, dummy_na=True).fillna(0)
        X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
        X_train_enc, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_enc, y_train)
        X_train = pd.DataFrame(X_train_enc, columns=X_train_enc.columns)
        X_test = X_test_enc
        pre = ColumnTransformer([("id", "passthrough", X_train.columns.tolist())])
    else:
        numeric_cols = [c for c in NUMERIC_ALL if c in X_train.columns]
        categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
        pre = build_preprocessor(numeric_cols, categorical_cols)

    return X_train, X_test, y_train, y_test, pre


def _strip_model_prefix(params: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for k, v in params.items():
        if k.startswith("model__"):
            out[k.replace("model__", "", 1)] = v
    return out


def load_hparam_results(path: Path) -> Dict[str, Dict[str, Dict[str, object]]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    tuned: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in data:
        scenario = row.get("scenario")
        model = row.get("model")
        params = row.get("best_params") or {}
        if not scenario or not model:
            continue
        tuned.setdefault(scenario, {})[model] = _strip_model_prefix(params)
    return tuned


def build_models(tuned_params: Dict[str, Dict[str, object]] | None = None) -> Dict[str, object]:
    models: Dict[str, object] = {
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=RANDOM_STATE),
        "dt": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "gbm": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "lr": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "svm": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
    if XGB_AVAILABLE:
        models["xgb"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if tuned_params:
        for name, params in tuned_params.items():
            if name in models and params:
                models[name].set_params(**params)
    return models


def eval_metrics(y_true: pd.Series, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "f1": f1_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "brier": brier_score_loss(y_true, prob),
    }

def eval_fedtree(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    pre: ColumnTransformer,
) -> Dict[str, float] | None:
    if not FEDTREE_AVAILABLE:
        return None

    pre_local = clone(pre)
    X_tr = pre_local.fit_transform(X_train, y_train)
    X_te = pre_local.transform(X_test)
    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()
    if hasattr(X_te, "toarray"):
        X_te = X_te.toarray()
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)

    model = FLClassifier(
        n_parties=2,
        mode="horizontal",
        max_depth=6,
        n_trees=200,
        learning_rate=0.1,
        objective="binary:logistic",
        verbose=0,
        seed=RANDOM_STATE,
    )
    model.fit(X_tr, y_train.to_numpy())
    prob = model.predict_proba(X_te)[:, 1]
    return eval_metrics(y_test, prob)


def main() -> None:
    tuned = load_hparam_results(ROOT / "results" / "hparam_results.json")

    rows = []
    for scen in build_scenarios(DROP_COLS, include_opt=True):
        label = scen["label"]
        use_smote = scen["use_smote"]
        drops = scen["drop"]
        base_label = scen["base_label"]
        X_train, X_test, y_train, y_test, pre = prepare_data(use_smote, drops)
        tuned_params = tuned.get(base_label, {}) if label.endswith("_opt") else {}
        for name, model in build_models(tuned_params).items():
            pipe = ImbPipeline([("prep", pre), ("model", model)])
            pipe.fit(X_train, y_train)
            prob = pipe.predict_proba(X_test)[:, 1]
            metrics = eval_metrics(y_test, prob)
            for m, v in metrics.items():
                rows.append({"scenario": label, "model": name, "metric": m, "value": v})

        ft_metrics = eval_fedtree(X_train, X_test, y_train, y_test, pre)
        if ft_metrics is not None:
            for m, v in ft_metrics.items():
                rows.append({"scenario": label, "model": "fedtree", "metric": m, "value": v})

    out_csv = ROOT / "results" / "central_holdout_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Salvo: {out_csv}")


if __name__ == "__main__":
    main()
