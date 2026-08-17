"""Recover inner-selection runs from their logs and re-apply the selection rule.

The inner run tag omits the learning rate, so two swept rates write one JSON
name and the second overwrites the first; the per-run log keeps the rate in its
own filename and survives. This reads the logs back, re-runs the documented
rule over the full sweep, and reports where the executed selection differs.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


LOG = re.compile(r"^log_(?P<lr>[^_]+)_(?P<aug>with|without)_i(?P<inner>\d+)$")
SUMMARY = re.compile(r"inner val (?P<val>[\d.]+) mm subject-averaged; budget (?P<budget>\d+) steps")


def read_log(path: Path) -> dict:
    """One inner run's outcome, or its divergence, as the log recorded it."""
    match = LOG.match(path.name)
    if match is None:
        raise ValueError(f"{path} is not an inner-selection log")
    text = path.read_text(encoding="utf-8", errors="replace")
    row = {"lr": float(match["lr"]), "augment": match["aug"] == "with",
           "inner_fold": int(match["inner"]), "diverged": "diverged; stopping" in text}
    summary = SUMMARY.search(text)
    if summary is None:
        if not row["diverged"]:
            raise ValueError(f"{path} has neither a summary line nor a divergence")
        return {**row, "val_mm": None, "budget_steps": None}
    return {**row, "val_mm": float(summary["val"]), "budget_steps": int(summary["budget"])}


def surviving_records(directory: Path) -> dict[tuple, dict]:
    """Full-precision runs whose JSON was not overwritten, keyed like the logs."""
    records = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "inner_fold" not in record:
            continue
        key = (float(record["config"]["lr"]), bool(record["config"]["augment"]),
               int(record["inner_fold"]))
        records[key] = record
    return records


def select_from_logs(directory: Path, n_inner: int = 3) -> dict:
    """Apply `select_inner.aggregate_inner_runs`'s rule to the recovered runs.

    A log prints one decimal, which is coarse enough to tie two configurations
    that the full-precision record separates, so a surviving JSON wins.
    """
    rows = [read_log(path) for path in sorted(directory.glob("log_*"))]
    if not rows:
        raise ValueError(f"no inner-selection logs under {directory}")
    exact = surviving_records(directory)
    for row in rows:
        record = exact.get((row["lr"], row["augment"], row["inner_fold"]))
        row["source"] = "log" if record is None else "json"
        if record is not None and not row["diverged"]:
            row["val_mm"] = float(record["val_subject_averaged_mm"])
            row["budget_steps"] = int(record["best_step"])
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row["lr"], row["augment"]), []).append(row)
    candidates, rejected = [], []
    for (lr, augment), runs in sorted(groups.items()):
        if sorted(r["inner_fold"] for r in runs) != list(range(n_inner)):
            raise ValueError(f"lr {lr} augment {augment} is missing inner folds")
        if any(r["diverged"] for r in runs):
            rejected.append({"lr": lr, "augment": augment})
            continue
        candidates.append({
            "lr": lr, "augment": augment,
            "mean_val_mm": float(np.mean([r["val_mm"] for r in runs])),
            "per_inner_val_mm": [r["val_mm"] for r in sorted(runs, key=lambda r: r["inner_fold"])],
            "median_budget_steps": int(np.median([r["budget_steps"] for r in runs])),
            "source": "json" if all(r.get("source") == "json" for r in runs) else "log",
        })
    if not candidates:
        raise ValueError(f"every configuration diverged under {directory}")
    candidates.sort(key=lambda c: c["mean_val_mm"])
    return {"winner": candidates[0], "candidates": candidates, "rejected_diverged": rejected}


def audit(root: Path) -> dict:
    rows = []
    for selection_path in sorted(root.glob("f*/*/selection.json")):
        variant_dir = selection_path.parent
        recovered = select_from_logs(variant_dir / "inner")
        executed = json.loads(selection_path.read_text(encoding="utf-8"))
        winner = recovered["winner"]
        # the log prints one decimal, so a tie below that is not a ranking
        runner_up = recovered["candidates"][1] if len(recovered["candidates"]) > 1 else None
        rows.append({
            "fold": int(variant_dir.parent.name[1:]),
            "variant": variant_dir.name,
            "executed_lr": executed["config"]["lr"],
            "executed_augment": executed["config"]["augment"],
            "executed_budget_steps": executed["median_budget_steps"],
            "full_sweep_lr": winner["lr"],
            "full_sweep_augment": winner["augment"],
            "full_sweep_budget_steps": winner["median_budget_steps"],
            "full_sweep_mean_val_mm": winner["mean_val_mm"],
            "margin_over_runner_up_mm": (None if runner_up is None else
                                         round(runner_up["mean_val_mm"] - winner["mean_val_mm"], 3)),
            "candidates": recovered["candidates"],
            "same_configuration": (winner["lr"] == executed["config"]["lr"]
                                   and winner["augment"] == executed["config"]["augment"]),
        })
    changed = [r for r in rows if not r["same_configuration"]]
    return {"schema": "inner-log-audit-v1", "variant_folds": len(rows),
            "configuration_changes": len(changed), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.matrix)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for row in report["rows"]:
        flag = "same" if row["same_configuration"] else "CHANGES"
        print(f"f{row['fold']} {row['variant']:16} executed lr {row['executed_lr']:g} "
              f"aug {int(row['executed_augment'])} budget {row['executed_budget_steps']:>6} | "
              f"full sweep lr {row['full_sweep_lr']:g} aug {int(row['full_sweep_augment'])} "
              f"budget {row['full_sweep_budget_steps']:>6} "
              f"margin {row['margin_over_runner_up_mm']} mm  {flag}")
    print(f"{report['configuration_changes']} of {report['variant_folds']} variant-folds change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
