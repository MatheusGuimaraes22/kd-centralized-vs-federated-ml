from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.ckd_paper_kfold import build_preprocessor  # type: ignore

try:
    from fedtree import FLClassifier  # type: ignore

    FEDTREE_AVAILABLE = True
except Exception:
    FEDTREE_AVAILABLE = False

RANDOM_STATE = 42
NUM_CLIENTS = 5
ROUNDS = 10
NUM_TREES_PER_CLIENT = 40


def load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_clients(
    X: pd.DataFrame, y: pd.Series, n_clients: int
) -> List[Tuple[pd.DataFrame, pd.Series]]:
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=RANDOM_STATE)
    return [
        (X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True))
        for _, idx in skf.split(X, y)
    ]


def train_fedavg_lr(
    clients: List[Tuple[pd.DataFrame, pd.Series]], pre
) -> ImbPipeline:
    def make_pipe() -> ImbPipeline:
        base = SGDClassifier(
            loss="log_loss",
            max_iter=1,
            learning_rate="optimal",
            tol=None,
            random_state=RANDOM_STATE,
        )
        return ImbPipeline([("prep", pre), ("model", base)])

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
    def __init__(self, models: List[ImbPipeline]):
        self.models = list(models)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def train_fed_rf(clients, pre) -> FedEnsemble:
    models = []
    for idx, (Xc, yc) in enumerate(clients):
        rf = RandomForestClassifier(
            n_estimators=NUM_TREES_PER_CLIENT,
            random_state=RANDOM_STATE + idx,
            n_jobs=-1,
        )
        pipe = ImbPipeline([("prep", pre), ("model", rf)])
        pipe.fit(Xc, yc)
        models.append(pipe)
    return FedEnsemble(models)


def train_fedtree(pre, X_train, y_train):
    if not FEDTREE_AVAILABLE:
        return None, None
    pre_local = pre
    X_tr = pre_local.fit_transform(X_train, y_train)
    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()
    X_tr = np.asarray(X_tr, dtype=np.float64)
    y_tr = y_train.to_numpy()
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
    model.fit(X_tr, y_tr)
    return model, pre_local


def predict_fedtree(model, pre, X_test):
    X_te = pre.transform(X_test)
    if hasattr(X_te, "toarray"):
        X_te = X_te.toarray()
    X_te = np.asarray(X_te, dtype=np.float64)
    return model.predict_proba(X_te)[:, 1]


def save_predictions(outdir: Path, name: str, y: np.ndarray, prob: np.ndarray) -> Path:
    preds = (prob >= 0.5).astype(int)
    df = pd.DataFrame({"target": y, "probability": prob, "prediction": preds})
    out_path = outdir / f"holdout_predictions_{name}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def plot_calibration(y: np.ndarray, probs: Dict[str, np.ndarray], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, p in probs.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Fração de positivos")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Treina FL e gera predições/curva de calibração no holdout.")
    ap.add_argument("--scenario", default="all_nosmote")
    ap.add_argument("--train-csv", default=str(PROJECT_ROOT / "data" / "processed" / "ckd_train.csv"))
    ap.add_argument("--holdout-csv", default=str(PROJECT_ROOT / "data" / "processed" / "holdout_raw.csv"))
    ap.add_argument("--schema", default=str(PROJECT_ROOT / "data" / "processed" / "schema.json"))
    ap.add_argument("--outdir", default=str(PROJECT_ROOT / "results_holdout" / "fl"))
    args = ap.parse_args()

    train_df = pd.read_csv(args.train_csv)
    holdout_df = pd.read_csv(args.holdout_csv)
    if "target" not in train_df.columns or "target" not in holdout_df.columns:
        raise ValueError("CSV precisa conter a coluna 'target'.")

    schema = load_schema(Path(args.schema))
    numeric_cols = schema.get("numeric", [])
    categorical_cols = schema.get("categorical", [])
    pre = build_preprocessor(numeric_cols, categorical_cols)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"].astype(int)
    X_holdout = holdout_df.drop(columns=["target"])
    y_holdout = holdout_df["target"].astype(int).to_numpy()

    clients = split_clients(X_train, y_train, NUM_CLIENTS)

    outdir = Path(args.outdir) / args.scenario
    outdir.mkdir(parents=True, exist_ok=True)

    probs_map: Dict[str, np.ndarray] = {}
    brier_map: Dict[str, float] = {}

    fedavg = train_fedavg_lr(clients, pre)
    p_fedavg = fedavg.predict_proba(X_holdout)[:, 1]
    save_predictions(outdir, "fedavg_lr", y_holdout, p_fedavg)
    probs_map["fedavg_lr"] = p_fedavg
    brier_map["fedavg_lr"] = float(brier_score_loss(y_holdout, p_fedavg))

    fedrf = train_fed_rf(clients, pre)
    p_fedrf = fedrf.predict_proba(X_holdout)[:, 1]
    save_predictions(outdir, "fed_rf_vote", y_holdout, p_fedrf)
    probs_map["fed_rf_vote"] = p_fedrf
    brier_map["fed_rf_vote"] = float(brier_score_loss(y_holdout, p_fedrf))

    if FEDTREE_AVAILABLE:
        ft_model, ft_pre = train_fedtree(pre, X_train, y_train)
        if ft_model is not None:
            p_fedtree = predict_fedtree(ft_model, ft_pre, X_holdout)
            save_predictions(outdir, "fedtree_gbdt", y_holdout, p_fedtree)
            probs_map["fedtree_gbdt"] = p_fedtree
            brier_map["fedtree_gbdt"] = float(brier_score_loss(y_holdout, p_fedtree))

    # plot
    labels = {k: f"{k} (Brier={v:.3f})" for k, v in brier_map.items()}
    plot_probs = {labels[k]: probs_map[k] for k in probs_map}
    plot_calibration(y_holdout, plot_probs, outdir / "calibration_plot_fl.png")

    print(f"OK -> {outdir}")


if __name__ == "__main__":
    main()
