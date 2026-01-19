from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ckd_paper_kfold import build_preprocessor, load_data  # noqa: E402

try:
    from fedtree import FLClassifier  # type: ignore

    FEDTREE_AVAILABLE = True
except Exception:
    FEDTREE_AVAILABLE = False

RANDOM_STATE = 42
NUM_CLIENTS = 5
ROUNDS = 10
NUM_TREES_PER_CLIENT = 40
NUMERIC_ALL = [
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


def dirichlet_split(
    X: pd.DataFrame, y: pd.Series, n_clients: int, alpha: float, rng: np.random.Generator
) -> List[Tuple[pd.DataFrame, pd.Series]]:
    Xy = X.assign(_y=y.values)
    splits = [[] for _ in range(n_clients)]
    for cls in y.unique():
        X_cls = Xy[Xy._y == cls]
        idxs = X_cls.index.to_numpy()
        if len(idxs) < n_clients:
            raise ValueError("Classe com menos amostras do que clientes.")
        rng.shuffle(idxs)
        for client in range(n_clients):
            splits[client].append(X_cls.loc[[idxs[client]]])
        remaining = idxs[n_clients:]
        proportions = rng.dirichlet([alpha] * n_clients)
        counts = (proportions * len(remaining)).astype(int)
        while counts.sum() < len(remaining):
            counts[rng.integers(0, n_clients)] += 1
        idx = 0
        for client, c in enumerate(counts):
            if c == 0:
                continue
            part_idx = remaining[idx : idx + c]
            splits[client].append(X_cls.loc[part_idx])
            idx += c
    client_splits = []
    for parts in splits:
        concat = pd.concat(parts, axis=0)
        Xc = concat.drop(columns=["_y"]).reset_index(drop=True)
        yc = concat["_y"].reset_index(drop=True)
        client_splits.append((Xc, yc))
    return client_splits


def stratified_split(
    X: pd.DataFrame, y: pd.Series, n_clients: int, seed: int
) -> List[Tuple[pd.DataFrame, pd.Series]]:
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=seed)
    out: List[Tuple[pd.DataFrame, pd.Series]] = []
    for train_idx, _ in skf.split(X, y):
        out.append(
            (X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True))
        )
    return out


def make_clients(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    split_type: str,
    alpha: float,
    n_clients: int,
    seed: int,
) -> List[Tuple[pd.DataFrame, pd.Series]]:
    if split_type == "iid":
        return stratified_split(X_train, y_train, n_clients, seed)
    rng = np.random.default_rng(seed)
    return dirichlet_split(X_train, y_train, n_clients, alpha=alpha, rng=rng)


class PreFittedModel:
    def __init__(self, pre, model):
        self.pre = pre
        self.model = model
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        Xt = self.pre.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        return self.model.predict_proba(Xt)


class FedEnsemble:
    def __init__(self, pre, models):
        self.pre = pre
        self.models = list(models)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        Xt = self.pre.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        probs = [m.predict_proba(Xt) for m in self.models]
        return np.mean(probs, axis=0)


def train_fedavg_lr(clients, pre):
    base_lr = SGDClassifier(
        loss="log_loss",
        max_iter=1,
        learning_rate="optimal",
        tol=None,
        random_state=RANDOM_STATE,
    )
    X0 = pre.transform(clients[0][0])
    if hasattr(X0, "toarray"):
        X0 = X0.toarray()
    base_lr.fit(X0, clients[0][1])
    classes = base_lr.classes_

    for _ in range(ROUNDS):
        coefs = []
        intercepts = []
        for Xc, yc in clients:
            Xc_t = pre.transform(Xc)
            if hasattr(Xc_t, "toarray"):
                Xc_t = Xc_t.toarray()
            lr = SGDClassifier(
                loss="log_loss",
                max_iter=1,
                learning_rate="optimal",
                tol=None,
                random_state=RANDOM_STATE,
            )
            lr.fit(Xc_t, yc)
            coefs.append(lr.coef_)
            intercepts.append(lr.intercept_)
        base_lr.coef_ = np.mean(coefs, axis=0)
        base_lr.intercept_ = np.mean(intercepts, axis=0)
        base_lr.classes_ = classes
    return PreFittedModel(pre, base_lr)


def train_fed_rf(clients, pre):
    models = []
    for idx, (Xc, yc) in enumerate(clients):
        rf = RandomForestClassifier(
            n_estimators=NUM_TREES_PER_CLIENT,
            random_state=RANDOM_STATE + idx,
            n_jobs=-1,
        )
        Xc_t = pre.transform(Xc)
        if hasattr(Xc_t, "toarray"):
            Xc_t = Xc_t.toarray()
        rf.fit(Xc_t, yc)
        models.append(rf)
    return FedEnsemble(pre, models)


class FedTreeEnsemble:
    def __init__(self, pre, models):
        self.pre = pre
        self.models = list(models)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        Xt = self.pre.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        Xt = np.asarray(Xt, dtype=np.float64)
        probs = [m.predict_proba(Xt) for m in self.models]
        return np.mean(probs, axis=0)


def train_fedtree_ensemble(clients, pre):
    if not FEDTREE_AVAILABLE:
        return None
    models = []
    for _, (Xc, yc) in enumerate(clients):
        Xc_t = pre.transform(Xc)
        if hasattr(Xc_t, "toarray"):
            Xc_t = Xc_t.toarray()
        Xc_t = np.asarray(Xc_t, dtype=np.float64)
        yc_arr = yc.to_numpy()
        est = FLClassifier(
            n_parties=2,
            mode="horizontal",
            max_depth=6,
            n_trees=200,
            learning_rate=0.1,
            objective="binary:logistic",
            verbose=0,
            seed=RANDOM_STATE,
        )
        est.fit(Xc_t, yc_arr)
        models.append(est)
    return FedTreeEnsemble(pre, models)


def build_pre(X_train: pd.DataFrame, y_train: pd.Series) -> ColumnTransformer:
    numeric_cols = [c for c in NUMERIC_ALL if c in X_train.columns]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    pre = build_preprocessor(numeric_cols, categorical_cols)
    pre.fit(X_train, y_train)
    return pre


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--split-type", choices=["iid", "non_iid"], default="iid")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--scenario", type=str, default="all_nosmote")
    parser.add_argument("--outdir", type=str, default=str(ROOT / "results"))
    args = parser.parse_args()

    X, y = load_data()
    outer = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    preds: Dict[str, List[Dict[str, object]]] = {
        "central_rf": [],
        "fedavg_lr": [],
        "fed_rf_vote": [],
        "fedtree_gbdt": [],
    }

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        row_ids = X.index[test_idx].to_numpy()

        pre = build_pre(X_train, y_train)

        seed = RANDOM_STATE + fold_idx * 100 + (0 if args.split_type == "iid" else 1)
        clients = make_clients(X_train, y_train, args.split_type, args.alpha, NUM_CLIENTS, seed)

        # Central RF
        central = ImbPipeline(
            [("prep", pre), ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))]
        )
        central.fit(X_train, y_train)
        prob = central.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        for rid, yt, pr, pdx in zip(row_ids, y_test, prob, pred):
            preds["central_rf"].append(
                {"row_id": int(rid), "target": int(yt), "probability": float(pr), "prediction": int(pdx), "fold": fold_idx}
            )

        # FedAvg-LR
        fedavg = train_fedavg_lr(clients, pre)
        prob = fedavg.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        for rid, yt, pr, pdx in zip(row_ids, y_test, prob, pred):
            preds["fedavg_lr"].append(
                {"row_id": int(rid), "target": int(yt), "probability": float(pr), "prediction": int(pdx), "fold": fold_idx}
            )

        # FedRF
        fedrf = train_fed_rf(clients, pre)
        prob = fedrf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        for rid, yt, pr, pdx in zip(row_ids, y_test, prob, pred):
            preds["fed_rf_vote"].append(
                {"row_id": int(rid), "target": int(yt), "probability": float(pr), "prediction": int(pdx), "fold": fold_idx}
            )

        # FedTree
        if FEDTREE_AVAILABLE:
            fedtree = train_fedtree_ensemble(clients, pre)
            if fedtree is not None:
                prob = fedtree.predict_proba(X_test)[:, 1]
                pred = (prob >= 0.5).astype(int)
                for rid, yt, pr, pdx in zip(row_ids, y_test, prob, pred):
                    preds["fedtree_gbdt"].append(
                        {"row_id": int(rid), "target": int(yt), "probability": float(pr), "prediction": int(pdx), "fold": fold_idx}
                    )

    for name, rows in preds.items():
        if not rows:
            continue
        out_path = outdir / f"oof_predictions_{args.scenario}_{args.split_type}_{name}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print("Salvo:", out_path)


if __name__ == "__main__":
    main()
