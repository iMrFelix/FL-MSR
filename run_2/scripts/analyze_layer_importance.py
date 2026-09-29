"""Analyze and compare results from the layer-importance dissemination experiments.

Reads one report.json per experiment from <results_dir>/expN/results/report.json
and generates four comparison figures:

  1. Val accuracy vs round number  (convergence rate, topology-independent)
  2. Val accuracy vs wall-clock time  (learning speed in real seconds)
  3. Communication duration per round  (median across worker nodes)
  4. Bytes sent per traffic class per round  (per-layer experiments only)

Usage
-----
    python scripts/analyze_layer_importance.py --results-dir ./results/

    # or with explicit experiment list:
    python scripts/analyze_layer_importance.py \\
        --results-dir ./results/ \\
        --experiments exp1 exp2 exp3 exp4 exp5 exp6

Output
------
  ./results/figures/fig1_accuracy_vs_round.png
  ./results/figures/fig2_accuracy_vs_time.png
  ./results/figures/fig3_comm_duration.png
  ./results/figures/fig4_bytes_per_class.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Experiment metadata for readable labels
# ---------------------------------------------------------------------------

EXPERIMENT_META: dict[str, dict] = {
    "exp1": {
        "label": "E1: Mono 10 Mbps",
        "color": "#1f77b4",
        "linestyle": "-",
        "part": 1,
    },
    "exp2": {
        "label": "E2: Mono 5 Mbps",
        "color": "#ff7f0e",
        "linestyle": "-",
        "part": 1,
    },
    "exp3": {
        "label": "E3: Mono 1 Mbps",
        "color": "#2ca02c",
        "linestyle": "-",
        "part": 1,
    },
    "exp4": {
        "label": "E4: Mono 10 Mbps (ref)",
        "color": "#1f77b4",
        "linestyle": "--",
        "part": 2,
    },
    "exp5": {
        "label": "E5: Per-layer uniform 6+3+1 Mbps",
        "color": "#d62728",
        "linestyle": "-",
        "part": 2,
    },
    "exp6": {
        "label": "E6: Per-layer grad-norm 6+3+1 Mbps",
        "color": "#9467bd",
        "linestyle": "-",
        "part": 2,
    },
    # --- ε-deadline preliminary (extension 02; docs/experiments/01-*) ---
    "prelim_mono_10mbps": {
        "label": "Mono 10 Mbps (ref)",
        "color": "#1f77b4",
        "linestyle": "--",
        "part": "prelim",
    },
    "prelim_perlayer_eps0": {
        "label": "Per-layer ε=0 (sync baseline)",
        "color": "#9467bd",
        "linestyle": "-",
        "part": "prelim",
    },
    "prelim_perlayer_eps01_drop": {
        "label": "Per-layer ε=0.1 + drop",
        "color": "#2ca02c",
        "linestyle": "-",
        "part": "prelim",
    },
    "prelim_perlayer_eps02_drop": {
        "label": "Per-layer ε=0.2 + drop",
        "color": "#d62728",
        "linestyle": "-",
        "part": "prelim",
    },
    "prelim_perlayer_eps05_drop": {
        "label": "Per-layer ε=0.5 + drop",
        "color": "#ff7f0e",
        "linestyle": "-",
        "part": "prelim",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_report(report_path: Path) -> dict:
    with open(report_path) as f:
        return json.load(f)


def extract_worker_metrics(report: dict) -> dict[str, list[dict]]:
    """Return per-node metric lists, excluding the aggregator (train_loss == 0)."""
    node_metrics: dict[str, list[dict]] = {}
    for round_entry in report["per_round"]:
        for node_id, metrics in round_entry["nodes"].items():
            if node_id not in node_metrics:
                node_metrics[node_id] = []
            node_metrics[node_id].append({"round": round_entry["round"], **metrics})

    # Keep only worker nodes (aggregator has train_loss == 0 for every round)
    workers = {}
    for node_id, rounds in node_metrics.items():
        if any(r.get("train_loss", 0.0) > 0 for r in rounds):
            workers[node_id] = sorted(rounds, key=lambda r: r["round"])
    return workers


def compute_mean_over_workers(
    worker_metrics: dict[str, list[dict]], field: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rounds, mean_values) averaged over all worker nodes."""
    # Collect per-round values across workers
    by_round: dict[int, list[float]] = {}
    for rounds in worker_metrics.values():
        for r in rounds:
            rn = r["round"]
            val = r.get(field)
            if val is not None:
                by_round.setdefault(rn, []).append(float(val))
    sorted_rounds = sorted(by_round.keys())
    means = np.array([np.mean(by_round[rn]) for rn in sorted_rounds])
    return np.array(sorted_rounds), means


def compute_cumulative_time(worker_metrics: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    """Cumulative wall-clock time per round for each worker node."""
    cumulative: dict[str, np.ndarray] = {}
    for node_id, rounds in worker_metrics.items():
        durations = np.array([r.get("round_duration_s", 0.0) for r in rounds])
        cumulative[node_id] = np.cumsum(durations)
    return cumulative


def aggregate_layer_comm_metrics(
    worker_metrics: dict[str, list[dict]],
) -> dict[int, list[float]]:
    """Return mean bytes_sent per traffic class per round (workers only).

    Returns {class_index: [mean_bytes_round_0, mean_bytes_round_1, ...]}.
    Only meaningful for per-layer experiments where layer_comm_metrics is populated.
    """
    by_round_class: dict[int, dict[int, list[float]]] = {}
    for rounds in worker_metrics.values():
        for r in rounds:
            rn = r["round"]
            for lm in r.get("layer_comm_metrics", []):
                tc = int(lm.get("traffic_class", 0))
                bs = float(lm.get("bytes_sent", 0))
                by_round_class.setdefault(rn, {}).setdefault(tc, []).append(bs)

    if not by_round_class:
        return {}

    sorted_rounds = sorted(by_round_class.keys())
    all_classes = sorted({tc for per_class in by_round_class.values() for tc in per_class})

    result: dict[int, list[float]] = {tc: [] for tc in all_classes}
    for rn in sorted_rounds:
        for tc in all_classes:
            vals = by_round_class[rn].get(tc, [0.0])
            result[tc].append(sum(vals))   # total bytes this class this round

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

FIGSIZE = (9, 5)
DPI = 150


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Val accuracy vs round
# ---------------------------------------------------------------------------

def plot_accuracy_vs_round(
    experiments: dict[str, tuple[dict, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for exp_id, (meta, rounds, accuracy) in experiments.items():
        ax.plot(
            rounds, accuracy * 100,
            label=meta["label"],
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=1.8,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Val accuracy vs round\n(equal rounds = equal local compute)")
    ax.legend(fontsize=8, loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 2 — Val accuracy vs wall-clock time
# ---------------------------------------------------------------------------

def plot_accuracy_vs_time(
    experiments: dict[str, tuple[dict, np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for exp_id, (meta, cum_time, accuracy) in experiments.items():
        ax.plot(
            cum_time / 60.0, accuracy * 100,
            label=meta["label"],
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=1.8,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Wall-clock time (minutes)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Val accuracy vs wall-clock time\n(accounts for comm overhead)")
    ax.legend(fontsize=8, loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 3 — Communication duration per round
# ---------------------------------------------------------------------------

def plot_comm_duration(
    experiments: dict[str, tuple[dict, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for exp_id, (meta, rounds, comm_dur) in experiments.items():
        ax.plot(
            rounds, comm_dur,
            label=meta["label"],
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=1.8,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("comm_duration_s (s)")
    ax.set_title("Communication duration per round\n(median across worker nodes)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 4 — Bytes per traffic class per round (per-layer experiments only)
# ---------------------------------------------------------------------------

def plot_bytes_per_class(
    experiments: dict[str, tuple[dict, np.ndarray, dict[int, list[float]]]],
    output_path: Path,
) -> None:
    perlayer_exps = {k: v for k, v in experiments.items() if v[2]}
    if not perlayer_exps:
        print("  No per-layer experiments found; skipping fig4.")
        return

    ncols = len(perlayer_exps)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    class_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, (exp_id, (meta, rounds, by_class)) in zip(axes, perlayer_exps.items()):
        bottom = np.zeros(len(rounds))
        for tc, bytes_list in sorted(by_class.items()):
            vals = np.array(bytes_list) / 1024   # → KB
            ax.bar(
                rounds, vals, bottom=bottom,
                label=f"Class {tc}",
                color=class_colors[tc % len(class_colors)],
                alpha=0.85,
            )
            bottom += vals
        ax.set_title(meta["label"], fontsize=8)
        ax.set_xlabel("Round")
        if ax is axes[0]:
            ax.set_ylabel("Bytes sent (KB)")
        ax.legend(fontsize=7)

    fig.suptitle("Bytes sent per traffic class per round (worker nodes, per-layer mode)")
    fig.tight_layout()
    save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir", default="./results/",
        help="Root directory containing one sub-dir per experiment.",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        default=None,
        help=(
            "Experiment sub-directory names to include.  If omitted, auto-"
            "detect every immediate sub-directory of --results-dir that "
            "contains a results/report.json file."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write figures (default: <results_dir>/figures/).",
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_root / "figures"

    # Auto-detect experiments when --experiments is not given.
    if args.experiments is None:
        if not results_root.is_dir():
            print(f"--results-dir {results_root} is not a directory")
            sys.exit(1)
        experiments = sorted(
            p.name for p in results_root.iterdir()
            if p.is_dir() and (p / "results" / "report.json").exists()
        )
        if not experiments:
            print(f"No experiments with results/report.json under "
                  f"{results_root}; nothing to do.")
            sys.exit(1)
        print(f"Auto-detected experiments: {experiments}")
    else:
        experiments = args.experiments

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    loaded: dict[str, dict] = {}
    for exp_id in experiments:
        report_path = results_root / exp_id / "results" / "report.json"
        if not report_path.exists():
            print(f"[warn] {report_path} not found — skipping {exp_id}")
            continue
        loaded[exp_id] = load_report(report_path)
        print(f"[ok]   Loaded {report_path}  "
              f"({loaded[exp_id]['experiment'].get('total_rounds', '?')} rounds, "
              f"model={loaded[exp_id]['experiment'].get('model', '?')})")

    if not loaded:
        print("No reports found.  Did you run the experiments first?")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Derive per-experiment series
    # ------------------------------------------------------------------
    acc_round:  dict[str, tuple] = {}
    acc_time:   dict[str, tuple] = {}
    comm_round: dict[str, tuple] = {}
    bytes_class: dict[str, tuple] = {}

    for exp_id, report in loaded.items():
        meta = EXPERIMENT_META.get(exp_id, {
            "label": exp_id, "color": "gray", "linestyle": "-", "part": 0,
        })
        workers = extract_worker_metrics(report)
        if not workers:
            print(f"[warn] No worker nodes found in {exp_id}; skipping.")
            continue

        rounds, acc = compute_mean_over_workers(workers, "val_accuracy")
        _, comm = compute_mean_over_workers(workers, "comm_duration_s")
        cum_times = compute_cumulative_time(workers)

        # Use the worker with the most rounds as the time axis
        representative = max(cum_times, key=lambda n: len(cum_times[n]))
        cum_t = cum_times[representative]
        # Align accuracy to cumulative time (may differ in length if workers differ)
        min_len = min(len(acc), len(cum_t))
        acc_t = acc[:min_len]
        ct = cum_t[:min_len]

        by_class = aggregate_layer_comm_metrics(workers)

        acc_round[exp_id]  = (meta, rounds, acc)
        acc_time[exp_id]   = (meta, ct, acc_t)
        comm_round[exp_id] = (meta, rounds, comm)
        bytes_class[exp_id] = (meta, rounds, by_class)

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    print(f"\nWriting figures to {output_dir}/")

    plot_accuracy_vs_round(acc_round, output_dir / "fig1_accuracy_vs_round.png")
    plot_accuracy_vs_time(acc_time, output_dir / "fig2_accuracy_vs_time.png")
    plot_comm_duration(comm_round, output_dir / "fig3_comm_duration.png")
    plot_bytes_per_class(bytes_class, output_dir / "fig4_bytes_per_class.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
