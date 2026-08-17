"""Inner-fold validation curves for one selected configuration, beside the sweep it won.

The candidate panel reads `inner_log_audit.json`, so it carries the two learning rates whose
inner records overwrote each other; `viz_selection_stage.py` plots surviving records only.

    python scripts/viz_training_curves.py --formal results/formal --fold 0 \\
        --variant shank_imu_dense --audit results/inner_log_audit.json --out training_curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WARMUP_EPOCHS = 20


def inner_histories(formal: Path, fold: int, variant: str, lr: float, augment: bool) -> list[dict]:
    """Every inner run of one configuration, ordered by inner fold."""
    runs = []
    for path in sorted((formal / f"f{fold}" / variant / "inner").glob("*.json")):
        record = json.loads(path.read_text())
        config = record["config"]
        if config["lr"] == lr and config["augment"] == augment and "history" in record:
            runs.append(record)
    return sorted(runs, key=lambda record: record["inner_fold"])


def audit_row(audit: Path, fold: int, variant: str) -> dict:
    """The recovered four-configuration sweep for one fold and variant."""
    rows = json.loads(audit.read_text())["rows"]
    for row in rows:
        if row["fold"] == fold and row["variant"] == variant:
            return row
    raise ValueError(f"no audit row for fold {fold} variant {variant}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, default=Path("results/formal"))
    parser.add_argument("--audit", type=Path, default=Path("results/inner_log_audit.json"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--variant", default="shank_imu_dense")
    parser.add_argument("--out", default="training_curves.png")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    row = audit_row(args.audit, args.fold, args.variant)
    runs = inner_histories(args.formal, args.fold, args.variant,
                           row["executed_lr"], row["executed_augment"])
    if not runs:
        raise SystemExit("the selected configuration has no surviving per-epoch history")
    steps_per_epoch = runs[0]["steps_per_epoch"]

    fig, (ax_curve, ax_select) = plt.subplots(1, 2, figsize=(9.6, 3.5))

    colours = ["#8B111A", "#1F4E79", "#2E7D32"]
    for run, colour in zip(runs, colours):
        history = run["history"]
        epochs = [entry["epoch"] for entry in history]
        values = [entry["val_subject_averaged_mm"] for entry in history]
        ax_curve.plot(epochs, values, linewidth=1.4, color=colour,
                      label=f"inner fold {run['inner_fold']}")
        best = min(range(len(values)), key=values.__getitem__)
        ax_curve.plot([epochs[best]], [values[best]], "o", ms=4.5, color=colour)

    ax_curve.axvline(WARMUP_EPOCHS, color="#111827", linewidth=1.0, linestyle=":")
    ax_curve.annotate("MPJPE → beta-NLL", xy=(WARMUP_EPOCHS, 0.93), xycoords=("data", "axes fraction"),
                      xytext=(4, 0), textcoords="offset points", fontsize=8, color="#111827")
    budget_epoch = row["executed_budget_steps"] / steps_per_epoch
    ax_curve.axvline(budget_epoch, color="#9AA0A6", linewidth=1.0, linestyle="--")
    ax_curve.annotate(f"median budget {row['executed_budget_steps']} steps",
                      xy=(budget_epoch, 0.80), xycoords=("data", "axes fraction"),
                      xytext=(4, 0), textcoords="offset points", fontsize=8, color="#6B7280")
    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel("inner validation MPJPE (mm)")
    ax_curve.set_title(f"fold {args.fold}, {args.variant}, selected configuration", fontsize=9)
    ax_curve.legend(frameon=False, fontsize=8)

    candidates = sorted(row["candidates"], key=lambda c: c["mean_val_mm"])
    labels = [f"lr {c['lr']:g}\n{'mirror' if c['augment'] else 'no mirror'}" for c in candidates]
    positions = range(len(candidates))
    for x, candidate in zip(positions, candidates):
        ax_select.scatter([x] * len(candidate["per_inner_val_mm"]), candidate["per_inner_val_mm"],
                          s=16, color="#9AA0A6", zorder=2)
        winner = x == 0
        ax_select.scatter([x], [candidate["mean_val_mm"]], s=70, zorder=3,
                          color="#8B111A" if winner else "#111827",
                          marker="D" if winner else "P")
        if candidate["source"] == "log":
            ax_select.annotate("recovered from log", xy=(x, candidate["mean_val_mm"]),
                               xytext=(0, -14), textcoords="offset points", fontsize=7,
                               color="#6B7280", ha="center")
    ax_select.set_xlim(-0.55, len(candidates) - 0.35)
    ax_select.set_xticks(list(positions))
    ax_select.set_xticklabels(labels, fontsize=8)
    ax_select.set_ylabel("inner validation MPJPE (mm)")
    ax_select.set_title("configuration selection: mean of three inner folds", fontsize=9)
    ax_select.scatter([], [], s=16, color="#9AA0A6", label="one inner fold")
    ax_select.scatter([], [], s=70, color="#8B111A", marker="D", label="selected")
    ax_select.legend(frameon=False, fontsize=8, loc="upper left")

    for ax in (ax_curve, ax_select):
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
