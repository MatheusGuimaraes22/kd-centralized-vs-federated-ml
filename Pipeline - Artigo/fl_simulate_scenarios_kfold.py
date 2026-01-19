from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.ckd_paper_kfold import build_preprocessor, load_data  # type: ignore
from scripts.feature_selection import build_scenarios, load_drop_cols  # type: ignore

try:
    from fedtree import FLClassifier
    FEDTREE_AVAILABLE = True
except Exception:
    FEDTREE_AVAILABLE = False

RANDOM_STATE = 42
N_SPLITS = 10
NUM_CLIENTS = 5
ROUNDS = 10
NUM_TREES_PER_CLIENT = 40
DROP_COLS = load_drop_cols(["sg", "pcc", "pcv", "hemo", "rbcc", "al"], k=6)
NUMERIC_ALL = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"]

def make_clients(X: pd.DataFrame, y: pd.Series, n_clients: int = NUM_CLIENTS) -> List[Tuple[pd.DataFrame, pd.Series]]:
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=RANDOM_STATE)
    return [(X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)) for _, idx in skf.split(X, y)]

def prepare_fold(X: pd.DataFrame, y: pd.Series, train_idx, test_idx, use_smote: bool, drop_cols: List[str]):
    X = X.copy()
    if drop_cols:
        X = X.drop(columns=drop_cols)
    X = X.loc[:, X.notna().any()]
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

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

def eval_metrics(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, prob),
        "f1": f1_score(y, pred),
        "recall": recall_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
    }

def train_fedavg_lr(clients, pre):
    def make_pipe():
        return ImbPipeline([("prep", clone(pre)), ("model", SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", tol=None, random_state=RANDOM_STATE))])

    global_pipe = make_pipe()
    global_pipe.fit(clients[0][0], clients[0][1])
    classes = global_pipe.named_steps["model"].classes_

    for _ in range(ROUNDS):
        coefs, intercepts = [], []
        for Xc, yc in clients:
            pipe = make_pipe()
            pipe.fit(Xc, yc)
            coefs.append(pipe.named_steps["model"].coef_)
            intercepts.append(pipe.named_steps["model"].intercept_)
        global_pipe.named_steps["model"].coef_ = np.mean(coefs, axis=0)
        global_pipe.named_steps["model"].intercept_ = np.mean(intercepts, axis=0)
        global_pipe.named_steps["model"].classes_ = classes

    return global_pipe

class FedEnsemble:
    def __init__(self, models):
        self.models = list(models)
        self.classes_ = np.array([0, 1])
    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

def train_fed_rf_vote(clients, pre):
    models = []
    for idx, (Xc, yc) in enumerate(clients):
        rf = RandomForestClassifier(n_estimators=NUM_TREES_PER_CLIENT, random_state=RANDOM_STATE + idx, n_jobs=-1)
        pipe = ImbPipeline([("prep", clone(pre)), ("model", rf)])
        pipe.fit(Xc, yc)
        models.append(pipe)
    return FedEnsemble(models)

def train_fedtree(pre, X_train, y_train):
    if not FEDTREE_AVAILABLE:
        return None, None
    pre_local = clone(pre)
    X_tr = pre_local.fit_transform(X_train, y_train)
    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()
    X_tr = np.asarray(X_tr, dtype=np.float64)
    y_tr = y_train.to_numpy()
    model = FLClassifier(n_parties=2, mode="horizontal", max_depth=6, n_trees=200, learning_rate=0.1, objective="binary:logistic", verbose=0, seed=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    return model, pre_local

def predict_fedtree(model, pre, X_test):
    X_te = pre.transform(X_test)
    if hasattr(X_te, "toarray"):
        X_te = X_te.toarray()
    X_te = np.asarray(X_te, dtype=np.float64)
    return model.predict_proba(X_te)[:, 1]

def main():
    X, y = load_data()
    scenarios = build_scenarios(DROP_COLS, include_opt=True)
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for scen in scenarios:
        label = scen["label"]
        use_smote = scen["use_smote"]
        drops = scen["drop"]
        scores = {"fedavg_lr": {}, "fed_rf_vote": {}, "FedTree-GBDT": {}}
        for m in ["roc_auc", "f1", "recall", "balanced_accuracy"]:
            scores["fedavg_lr"][m] = []
            scores["fed_rf_vote"][m] = []
            scores["FedTree-GBDT"][m] = []

        for train_idx, test_idx in outer.split(X, y):
            X_train, X_test, y_train, y_test, pre = prepare_fold(X, y, train_idx, test_idx, use_smote, drops)
            clients = make_clients(X_train, y_train)

            fedavg = train_fedavg_lr(clients, pre)
            mets = eval_metrics(fedavg, X_test, y_test)
            for m, v in mets.items():
                scores["fedavg_lr"][m].append(v)

            fedrf = train_fed_rf_vote(clients, pre)
            mets = eval_metrics(fedrf, X_test, y_test)
            for m, v in mets.items():
                scores["fed_rf_vote"][m].append(v)

            if FEDTREE_AVAILABLE:
                ft_model, ft_pre = train_fedtree(pre, X_train, y_train)
                if ft_model is not None:
                    prob = predict_fedtree(ft_model, ft_pre, X_test)
                    pred = (prob >= 0.5).astype(int)
                    scores["FedTree-GBDT"]["roc_auc"].append(roc_auc_score(y_test, prob))
                    scores["FedTree-GBDT"]["f1"].append(f1_score(y_test, pred))
                    scores["FedTree-GBDT"]["recall"].append(recall_score(y_test, pred))
                    scores["FedTree-GBDT"]["balanced_accuracy"].append(balanced_accuracy_score(y_test, pred))

        for model_name, sm in scores.items():
            if model_name == "FedTree-GBDT" and not FEDTREE_AVAILABLE:
                continue
            for metric, vals in sm.items():
                if not vals:
                    continue
                rows.append({
                    "scenario": label,
                    "model": model_name,
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                })

    out_csv = ROOT / "results" / "fl_simulation_metrics_kfold.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
