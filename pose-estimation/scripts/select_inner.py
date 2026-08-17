"""Aggregate inner-selection runs into one configuration and step budget.

Consumes the JSON records written by ``train_moveport.py --inner-fold`` for one
outer fold and variant: groups them by configuration, requires every inner fold per
configuration, scores by the mean subject-averaged validation MPJPE, and
reports the winning configuration with its median step budget for retraining.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def aggregate_inner_runs(directory: str | Path, n_inner: int = 3) -> dict:
    """Select the configuration with the best mean inner validation."""
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "inner_fold" in record:
            records.append(record)
    if not records:
        raise ValueError("no inner-selection records found")
    outer_folds = {int(record["fold"]) for record in records}
    if len(outer_folds) != 1:
        raise ValueError(f"selection records span more than one outer fold: {sorted(outer_folds)}")
    by_config: dict[str, dict] = {}
    for record in records:
        key = json.dumps(record["config"], sort_keys=True)
        group = by_config.setdefault(key, {"config": record["config"], "runs": {}})
        inner = int(record["inner_fold"])
        if inner in group["runs"]:
            raise ValueError(f"duplicate inner fold {inner} for config {key}")
        group["runs"][inner] = record
    candidates = []
    rejected = []
    for group in by_config.values():
        if sorted(group["runs"]) != list(range(n_inner)):
            raise ValueError(
                f"config {group['config']} is missing inner folds "
                f"{sorted(set(range(n_inner)) - set(group['runs']))}"
            )
        diverged = [i for i in range(n_inner) if group["runs"][i].get("diverged")]
        if diverged:
            # A diverged run's best epoch predates its collapse; its score and step
            # budget describe an aborted trajectory, not the configuration.
            rejected.append({"lr": group["config"]["lr"], "inner_folds": diverged})
            continue
        scores = [group["runs"][i]["val_subject_averaged_mm"] for i in range(n_inner)]
        budgets = [group["runs"][i]["best_step"] for i in range(n_inner)]
        candidates.append({
            "config": group["config"],
            "mean_val_subject_averaged_mm": float(np.mean(scores)),
            "per_inner_val_mm": [float(s) for s in scores],
            "median_budget_steps": int(np.median(budgets)),
        })
    if not candidates:
        raise ValueError(f"every candidate configuration diverged: {rejected}")
    candidates.sort(key=lambda c: c["mean_val_subject_averaged_mm"])
    winner = candidates[0]
    return {
        **winner,
        "inner_folds": list(range(n_inner)),
        "candidates": candidates,
        "rejected_diverged": rejected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--n-inner", type=int, default=3)
    args = parser.parse_args()
    selection = aggregate_inner_runs(args.runs, args.n_inner)
    print(json.dumps(selection, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
