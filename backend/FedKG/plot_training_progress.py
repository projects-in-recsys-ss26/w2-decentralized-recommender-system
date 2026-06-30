"""
Plot the training progress on the held-out TEST data.

The per-round test metrics are produced during training in main.py via
`test_server(server.model, global_data, config, 3)` and written to
`all_rounds_results.csv`. The test item is the *last* interaction of every
user (see get_global_data in src/util.py: train = User[u][:-2], test = User[u][-1]),
so it is never seen during training -> this is a genuine held-out test signal.

This script reads that CSV, draws one curve per metric over the communication
rounds, and saves BOTH the figure (PNG) and a copy of the underlying data (CSV)
into a single output folder.

Usage:
    # auto-detect the most recent run under output/
    python plot_training_progress.py

    # or point it at a specific results CSV / output folder
    python plot_training_progress.py --csv path/to/all_rounds_results.csv
    python plot_training_progress.py --out-dir plots/my_run
"""
import argparse
import glob
import os
import shutil

import matplotlib

matplotlib.use("Agg")  # headless backend: render to file without a display
import matplotlib.pyplot as plt
import pandas as pd

METRICS = ["NDCG5", "NDCG10", "NDCG20", "HR5", "HR10", "HR20"]


def find_latest_results_csv():
    """Return the most recently modified all_rounds_results.csv under output/."""
    pattern = os.path.join("output", "**", "all_rounds_results.csv")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(
            "No all_rounds_results.csv found under output/. "
            "Run a training (main.py) first or pass --csv explicitly."
        )
    return max(matches, key=os.path.getmtime)


def plot_progress(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    if "rounds" not in df.columns:
        raise ValueError(f"Expected a 'rounds' column in {csv_path}, got {list(df.columns)}")

    metrics = [m for m in METRICS if m in df.columns]
    if not metrics:
        raise ValueError(f"None of the expected metrics {METRICS} are in {csv_path}")

    os.makedirs(out_dir, exist_ok=True)

    # --- Figure: one curve per metric over the communication rounds ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metrics:
        ax.plot(df["rounds"], df[metric], marker="", linewidth=1.8, label=metric)

    # Mark the best round per metric so the peak test performance is visible.
    best_metric = "NDCG10" if "NDCG10" in metrics else metrics[0]
    best_idx = df[best_metric].idxmax()
    best_round = int(df["rounds"].iloc[best_idx])
    best_val = df[best_metric].iloc[best_idx]
    ax.axvline(best_round, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"best {best_metric}={best_val:.2f}\n@ round {best_round}",
        xy=(best_round, best_val),
        xytext=(8, -12),
        textcoords="offset points",
        fontsize=9,
        color="grey",
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Performance (%)")
    ax.set_title("Perfomance without GAN")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Metric", ncol=2)
    fig.tight_layout()

    png_path = os.path.join(out_dir, "training_progress.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # --- Data: copy the underlying test metrics next to the figure -----------
    csv_out = os.path.join(out_dir, "training_progress.csv")
    shutil.copyfile(csv_path, csv_out)

    return png_path, csv_out, best_round, best_metric, best_val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to all_rounds_results.csv. Defaults to the latest run under output/.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Folder to save the plot + CSV into. "
        "Defaults to a 'training_progress' folder next to the source CSV.",
    )
    args = parser.parse_args()

    csv_path = args.csv or find_latest_results_csv()
    out_dir = args.out_dir or os.path.join(os.path.dirname(csv_path), "training_progress")

    png_path, csv_out, best_round, best_metric, best_val = plot_progress(csv_path, out_dir)

    print(f"Source test metrics : {csv_path}")
    print(f"Saved plot          : {png_path}")
    print(f"Saved CSV           : {csv_out}")
    print(f"Best {best_metric:<7}     : {best_val:.2f} @ round {best_round}")


if __name__ == "__main__":
    main()
