"""Post-training report generator.

Generates a structured JSON report from collected metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_report(
    metrics: list[dict[str, Any]],
    experiment_config: dict[str, Any],
    output_path: str,
) -> dict[str, Any]:
    """Generate a structured training report.

    Args:
        metrics: List of per-node, per-round metric dicts.
        experiment_config: The experiment configuration.
        output_path: Path to write the JSON report.

    Returns:
        The report dict.
    """
    # Organize by round
    rounds_data: dict[int, dict[str, dict]] = {}
    for m in metrics:
        r = m["round"]
        if r not in rounds_data:
            rounds_data[r] = {}
        rounds_data[r][m["node_id"]] = {
            k: v for k, v in m.items() if k not in ("round", "node_id")
        }

    # Build per-round list
    per_round = []
    for r in sorted(rounds_data.keys()):
        per_round.append({"round": r, "nodes": rounds_data[r]})

    # Compute summary statistics
    node_ids = list({m["node_id"] for m in metrics})
    node_ids.sort()

    best_val_accuracy = {}
    final_val_accuracy = {}
    convergence_round = {}  # Round where val_accuracy first exceeds 0.95

    for node_id in node_ids:
        node_metrics = sorted(
            [m for m in metrics if m["node_id"] == node_id],
            key=lambda x: x["round"],
        )
        if node_metrics:
            best_val_accuracy[node_id] = max(m["val_accuracy"] for m in node_metrics)
            final_val_accuracy[node_id] = node_metrics[-1]["val_accuracy"]
            # Find convergence round (first round with val_acc > 0.95)
            conv = None
            for m in node_metrics:
                if m["val_accuracy"] >= 0.95:
                    conv = m["round"]
                    break
            convergence_round[node_id] = conv

    total_time = 0.0
    if metrics:
        # Sum of round durations for any one node (they run in parallel)
        sample_node = node_ids[0]
        node_metrics = [m for m in metrics if m["node_id"] == sample_node]
        total_time = sum(m["round_duration_s"] for m in node_metrics)

    report = {
        "experiment": {
            "algorithm": experiment_config.get("algorithm", "unknown"),
            "dataset": experiment_config.get("dataset", {}).get("name", "unknown"),
            "model": experiment_config.get("model", "unknown"),
            "num_nodes": len(node_ids),
            "total_rounds": experiment_config.get("total_rounds", 0),
        },
        "per_round": per_round,
        "summary": {
            "total_training_time_s": round(total_time, 2),
            "best_val_accuracy": best_val_accuracy,
            "final_val_accuracy": final_val_accuracy,
            "convergence_round_095": convergence_round,
        },
    }

    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Training report written to {output_file}")
    return report
