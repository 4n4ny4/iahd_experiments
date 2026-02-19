#!/usr/bin/env python3
"""
Regenerate plots from an ablation run directory.
Reads summary.json and across_task_results.json, produces PNG plots.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_per_task_value_ablation(summary, out_dir):
    tasks = sorted(summary["per_task"].keys())
    if not tasks:
        return
    k_values = summary["k_values"]

    delta = np.zeros((len(k_values), len(tasks)))
    for i, k in enumerate(k_values):
        key = f"k_{k}"
        for j, task in enumerate(tasks):
            delta[i, j] = summary["per_task"][task]["conditions"][key][
                "delta_value_rate_vs_baseline"
            ]

    fig, ax = plt.subplots(figsize=(max(11, len(tasks) * 1.1), max(4.5, len(k_values) * 1.2)))
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([f"k={k}" for k in k_values])
    ax.set_title("Within-Task: Delta Accuracy vs Baseline (pp)")
    for i in range(len(k_values)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Delta Accuracy (pp)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_within_task_delta_heatmap.png"), dpi=150)
    plt.close(fig)

    x = np.arange(len(tasks))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(12, len(tasks) * 1.2), 6))
    baseline_acc = [
        100 * summary["per_task"][t]["baseline"]["value_rate"] for t in tasks
    ]
    ax.bar(x - 1.5 * width, baseline_acc, width, label="Baseline", color="#4C72B0")
    colors = ["#55A868", "#DD8452", "#8172B2"]
    for i, k in enumerate(k_values):
        acc = [
            100 * summary["per_task"][t]["conditions"][f"k_{k}"]["value_rate"]
            for t in tasks
        ]
        ax.bar(x + (i - 1) * width, acc, width, label=f"k={k}", color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Within-Task: Ablating Each Task's Top Heads")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_within_task_accuracy.png"), dpi=150)
    plt.close(fig)

    x = np.arange(len(k_values))
    baseline_rates = [
        100 * summary["overall_by_k"][f"k_{k}"]["baseline_value_rate"] for k in k_values
    ]
    ablated_rates = [
        100 * summary["overall_by_k"][f"k_{k}"]["ablated_value_rate"] for k in k_values
    ]
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baseline_rates, width, label="Baseline", color="#4C72B0")
    ax.bar(x + width / 2, ablated_rates, width, label="Ablated (within-task)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Within-Task: Aggregate Accuracy Across All Tasks")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_within_task_aggregate.png"), dpi=150)
    plt.close(fig)


def plot_across_task_accuracy(across_task_results, tasks, k_values, out_dir):
    x = np.arange(len(k_values))
    total_baseline_val = sum(across_task_results["baseline"][t]["value_match"] for t in tasks)
    total_baseline_att = sum(across_task_results["baseline"][t]["attempts"] for t in tasks)
    baseline_agg = total_baseline_val / max(1, total_baseline_att)
    ablated_agg = []
    for k in k_values:
        total_val = sum(across_task_results["by_k"][str(k)]["task_value_match"].get(t, 0) for t in tasks)
        total_att = sum(across_task_results["by_k"][str(k)]["task_attempts"].get(t, 0) for t in tasks)
        ablated_agg.append(total_val / max(1, total_att))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        [100 * baseline_agg] * len(k_values),
        width,
        label="Baseline",
        color="#4C72B0",
    )
    ax.bar(
        x + width / 2,
        [100 * r for r in ablated_agg],
        width,
        label="Ablated (across-task)",
        color="#C44E52",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Across-Task: Ablating Global Top Heads")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_across_task_accuracy.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Regenerate ablation run plots from JSON outputs")
    parser.add_argument(
        "run_dir",
        help="Path to ablation run directory (e.g. experiments/outputs/ablation/run_20260219_033343)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override output directory for plots (default: same as run_dir)",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else run_dir

    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    summary_path = os.path.join(run_dir, "summary.json")
    across_path = os.path.join(run_dir, "across_task_results.json")

    if not os.path.isfile(summary_path):
        raise SystemExit(f"Missing {summary_path}")
    if not os.path.isfile(across_path):
        raise SystemExit(f"Missing {across_path}")

    with open(summary_path) as f:
        summary = json.load(f)
    with open(across_path) as f:
        across_task_results = json.load(f)

    tasks = sorted(summary["per_task"].keys())
    k_values = summary["k_values"]

    os.makedirs(out_dir, exist_ok=True)

    plot_per_task_value_ablation(summary, out_dir)
    plot_across_task_accuracy(across_task_results, tasks, k_values, out_dir)

    print(f"Plots saved to: {out_dir}")
    for name in [
        "ablation_within_task_delta_heatmap.png",
        "ablation_within_task_accuracy.png",
        "ablation_within_task_aggregate.png",
        "ablation_across_task_accuracy.png",
    ]:
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            print(f"  - {name}")


if __name__ == "__main__":
    main()
