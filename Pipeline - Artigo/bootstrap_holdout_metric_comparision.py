from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1
        for k in range(i, j):
            ranks[order[k]] = mid
        i = j
    return ranks


def fast_delong(preds_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive = preds_sorted_transposed[:, :m]
    negative = preds_sorted_transposed[:, m:]

    k = preds_sorted_transposed.shape[0]
    tx = np.zeros((k, m))
    ty = np.zeros((k, n))
    tz = np.zeros((k, m + n))
    for r in range(k):
        tx[r, :] = compute_midrank(positive[r, :])
        ty[r, :] = compute_midrank(negative[r, :])
        tz[r, :] = compute_midrank(preds_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, s


def delong_roc_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> float:
    y_true = y_true.astype(int)
    order = np.argsort(-y_true)
    preds = np.vstack((pred1, pred2))[:, order]
    label_1_count = int(y_true.sum())
    aucs, cov = fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return 1.0
    z = diff / np.sqrt(var)
    # two-sided p-value
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(p)


def eval_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "f1": f1_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
    }


def bootstrap_ci(
    y_true: np.ndarray, prob: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator
) -> Dict[str, Tuple[float, float, float]]:
    metrics = {"roc_auc": [], "f1": [], "recall": [], "balanced_accuracy": []}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        pr = prob[idx]
        vals = eval_metrics(yt, pr)
        for k, v in vals.items():
            metrics[k].append(v)
    out = {}
    alpha = (1.0 - ci) / 2.0
    for k, vals in metrics.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = (float(arr.mean()), float(np.quantile(arr, alpha)), float(np.quantile(arr, 1 - alpha)))
    return out


def read_pred_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df["target"].to_numpy(dtype=int), df["probability"].to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--central",
        default=str(root / "results" / "holdout_predictions_all_nosmote_rf.csv"),
    )
    parser.add_argument(
        "--fedavg",
        default=str(root.parent / "results_holdout" / "fl" / "all_nosmote" / "holdout_predictions_fedavg_lr.csv"),
    )
    parser.add_argument(
        "--fedrf",
        default=str(root.parent / "results_holdout" / "fl" / "all_nosmote" / "holdout_predictions_fed_rf_vote.csv"),
    )
    parser.add_argument(
        "--fedtree",
        default=str(root.parent / "results_holdout" / "fl" / "all_nosmote" / "holdout_predictions_fedtree_gbdt.csv"),
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-csv",
        default=str(root / "results" / "bootstrap_holdout_metric_comparison.csv"),
    )
    parser.add_argument(
        "--out-plot",
        default=str(root / "results" / "bootstrap_holdout_metric_comparison_plot.png"),
    )
    parser.add_argument(
        "--out-delong",
        default=str(root / "results" / "delong_roc_comparison.csv"),
    )
    args = parser.parse_args()

    paths = {
        "central_rf": Path(args.central),
        "fedavg_lr": Path(args.fedavg),
        "fed_rf_vote": Path(args.fedrf),
        "fedtree_gbdt": Path(args.fedtree),
    }
    for name, p in paths.items():
        if not p.exists():
            raise SystemExit(f"Arquivo nao encontrado: {name} -> {p}")

    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, object]] = []
    preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for name, p in paths.items():
        y_true, prob = read_pred_csv(p)
        preds[name] = (y_true, prob)
        stats = bootstrap_ci(y_true, prob, args.n_boot, args.ci, rng)
        for metric, (mean, low, high) in stats.items():
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    # DeLong (central vs FL)
    delong_rows = []
    y_c, p_c = preds["central_rf"]
    for name in ["fedavg_lr", "fed_rf_vote", "fedtree_gbdt"]:
        y_f, p_f = preds[name]
        if not np.array_equal(y_c, y_f):
            raise SystemExit(f"targets diferentes entre central_rf e {name}")
        pval = delong_roc_test(y_c, p_c, p_f)
        delong_rows.append({"model_a": "central_rf", "model_b": name, "p_value": pval})
    pd.DataFrame(delong_rows).to_csv(args.out_delong, index=False)

    # Plot bootstrap CI
    metrics = ["roc_auc", "f1", "recall", "balanced_accuracy"]
    models = ["central_rf", "fedavg_lr", "fed_rf_vote", "fedtree_gbdt"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    axes = axes.flatten()
    x = np.arange(len(models))
    width = 0.6
    for ax, metric in zip(axes, metrics):
        sub = out_df[out_df["metric"] == metric].set_index("model").reindex(models)
        means = sub["mean"].to_numpy()
        lows = sub["ci_low"].to_numpy()
        highs = sub["ci_high"].to_numpy()
        errs = np.vstack([means - lows, highs - means])
        ax.bar(x, means, width=width, yerr=errs, capsize=3)
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Salvo:", args.out_csv)
    print("Salvo:", args.out_delong)
    print("Salvo:", out_plot)


if __name__ == "__main__":
    main()
