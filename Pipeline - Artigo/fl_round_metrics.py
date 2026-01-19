from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

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
ROUNDS = 10
NUM_CLIENTS = 5
TEST_SIZE = 0.2
NUM_TREES_PER_CLIENT = 40
DROP_COLS = load_drop_cols(["sg", "pcc", "pcv", "hemo", "rbcc", "al"], k=6)
NUMERIC_ALL = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"]

def make_clients(X: pd.DataFrame, y: pd.Series, n_clients: int = NUM_CLIENTS) -> List[Tuple[pd.DataFrame, pd.Series]]:
    skf = StratifiedKFold(n_clients, shuffle=True, random_state=RANDOM_STATE)
    return [(X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)) for _, idx in skf.split(X, y)]

def prepare_data(use_smote: bool, drop_cols: List[str]):
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

def eval_metrics(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, prob),
        "f1": f1_score(y, pred),
        "recall": recall_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
    }

def train_fedavg_lr_rounds(clients, pre):
    global_pipe = ImbPipeline([("prep", pre), ("model", SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", tol=None, random_state=RANDOM_STATE))])
    global_pipe.fit(clients[0][0], clients[0][1])
    classes = global_pipe.named_steps["model"].classes_
    for _ in range(ROUNDS):
        coefs, intercepts = [], []
        for Xc, yc in clients:
            pipe = ImbPipeline([("prep", pre), ("model", SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", tol=None, random_state=RANDOM_STATE))])
            pipe.fit(Xc, yc)
            coefs.append(pipe.named_steps["model"].coef_)
            intercepts.append(pipe.named_steps["model"].intercept_)
        global_pipe.named_steps["model"].coef_ = np.mean(coefs, axis=0)
        global_pipe.named_steps["model"].intercept_ = np.mean(intercepts, axis=0)
        global_pipe.named_steps["model"].classes_ = classes
        yield global_pipe

class FedEnsemble:
    def __init__(self, models): self.models = list(models); self.classes_ = np.array([0,1])
    def predict_proba(self, X): return np.mean([m.predict_proba(X) for m in self.models], axis=0)

def train_fed_rf_vote(clients, pre):
    models = []
    for idx, (Xc, yc) in enumerate(clients):
        rf = RandomForestClassifier(n_estimators=NUM_TREES_PER_CLIENT, random_state=RANDOM_STATE+idx, n_jobs=-1)
        pipe = ImbPipeline([("prep", pre), ("model", rf)])
        pipe.fit(Xc, yc)
        models.append(pipe)
    return FedEnsemble(models)

def train_fedtree_once(pre, X_train, y_train):
    if not FEDTREE_AVAILABLE:
        return None
    X_arr = pre.fit_transform(X_train, y_train)
    if hasattr(X_arr, "toarray"):
        X_arr = X_arr.toarray()
    X_arr = np.asarray(X_arr, dtype=np.float64)
    y_arr = y_train.to_numpy()
    cv = StratifiedKFold(n_splits=ROUNDS, shuffle=True, random_state=RANDOM_STATE)
    scores = {"roc_auc": [], "f1": [], "recall": [], "balanced_accuracy": []}
    for train_idx, test_idx in cv.split(X_arr, y_arr):
        est = FLClassifier(n_parties=2, mode="horizontal", max_depth=6, n_trees=200, learning_rate=0.1, objective="binary:logistic", verbose=0, seed=RANDOM_STATE)
        est.fit(X_arr[train_idx], y_arr[train_idx])
        prob = est.predict_proba(X_arr[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        scores["roc_auc"].append(roc_auc_score(y_arr[test_idx], prob))
        scores["f1"].append(f1_score(y_arr[test_idx], pred))
        scores["recall"].append(recall_score(y_arr[test_idx], pred))
        scores["balanced_accuracy"].append(balanced_accuracy_score(y_arr[test_idx], pred))
    return {m: np.mean(v) for m, v in scores.items()}

def main():
    scenarios = build_scenarios(DROP_COLS, include_opt=True)
    round_logs = []
    for scen in scenarios:
        label = scen["label"]
        use_smote = scen["use_smote"]
        drops = scen["drop"]
        X_train, X_test, y_train, y_test, pre = prepare_data(use_smote, drops)
        clients = make_clients(X_train, y_train)

        # FedAvg LR por rodada
        for r, model in enumerate(train_fedavg_lr_rounds(clients, pre), start=1):
            mets = eval_metrics(model, X_test, y_test)
            for m, v in mets.items():
                round_logs.append({"scenario": label, "round": r, "model": "fedavg_lr", "metric": m, "value": v})

        # Fed RF (vote) ? valores repetidos em todas as rodadas s? para plot
        fed_rf = train_fed_rf_vote(clients, pre)
        mets_rf = eval_metrics(fed_rf, X_test, y_test)
        for r in range(1, ROUNDS+1):
            for m, v in mets_rf.items():
                round_logs.append({"scenario": label, "round": r, "model": "fed_rf_vote", "metric": m, "value": v})

        # FedTree ? m?dia 10-fold, replicada por rodada
        ft_scores = train_fedtree_once(pre, X_train, y_train)
        if ft_scores:
            for r in range(1, ROUNDS+1):
                for m, v in ft_scores.items():
                    round_logs.append({"scenario": label, "round": r, "model": "FedTree-GBDT", "metric": m, "value": v})

    out_csv = ROOT / "results" / "fl_round_metrics.csv"
    pd.DataFrame(round_logs).to_csv(out_csv, index=False)
    print(f"Salvo: {out_csv}")

    # Plota curvas
    df = pd.read_csv(out_csv)
    for m in df["metric"].unique():
        sub = df[df["metric"]==m]
        plt.figure(figsize=(8,4))
        sns.lineplot(data=sub, x="round", y="value", hue="model", marker="o")
        plt.title(f"Convergencia - {m}")
        plt.ylabel(m); plt.xlabel("Rodada"); plt.ylim(0,1.01); plt.tight_layout()
        out = ROOT / "results" / f"convergence_{m}.png"
        plt.savefig(out, dpi=200); plt.close()
        print(f"Salvo: {out}")

if __name__ == "__main__":
    main()
