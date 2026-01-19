from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bootstrap_ci(values: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator):
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means.append(sample.mean())
    means = np.asarray(means)
    alpha = (1.0 - ci) / 2.0
    low = np.quantile(means, alpha)
    high = np.quantile(means, 1.0 - alpha)
    return means.mean(), values.std(ddof=1) if n > 1 else 0.0, low, high


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--csv",
        default=str(root / "results" / "perturbation_iid_vs_noniid_cv.csv"),
        help="CSV com resultados por fold",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument(
        "--out-csv",
        default=str(root / "results" / "perturbation_iid_vs_noniid_cv_bootstrap.csv"),
    )
    parser.add_argument(
        "--out-png",
        default=str(root / "results" / "perturbation_iid_vs_noniid_cv_bootstrap.png"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV nao encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(args.seed)

    rows = []
    for (split, model, metric), group in df.groupby(["split", "model", "metric"]):
        values = group["value"].to_numpy(dtype=float)
        mean, std, low, high = bootstrap_ci(values, args.n_boot, args.ci, rng)
        rows.append(
            {
                "split": split,
                "model": model,
                "metric": metric,
                "mean": mean,
                "std": std,
                "ci_low": low,
                "ci_high": high,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    # tabela para PNG
    table_df = out_df.copy()
    table_df["mean_ci"] = table_df.apply(
        lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
        axis=1,
    )
    table_df = table_df.sort_values(["split", "model", "metric"])

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(table_df))))
    ax.axis("off")
    col_labels = ["Split", "Model", "Metric", "Mean [CI]"]
    cell_text = table_df[["split", "model", "metric", "mean_ci"]].values
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    for i in range(len(table_df)):
        tbl[(i + 1, 0)].set_text_props(ha="left")
        tbl[(i + 1, 1)].set_text_props(ha="left")
        tbl[(i + 1, 2)].set_text_props(ha="left")

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("Salvo:", args.out_csv)
    print("Salvo:", args.out_png)


if __name__ == "__main__":
    main()
