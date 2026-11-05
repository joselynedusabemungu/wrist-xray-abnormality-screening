from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

from src import config
from src.eval_harness import compute_metrics, load_predictions, write_json


def choose_threshold_for_recall(
    labels: np.ndarray, probs: np.ndarray, recall_target: float
) -> float:

    if not 0.0 < recall_target <= 1.0:
        raise ValueError("recall_target must be in (0, 1].")
    _, recall, thresholds = precision_recall_curve(labels, probs)
    ok = recall[:-1] >= recall_target
    if not ok.any():
        raise ValueError("No threshold reaches the recall target on validation data.")
    return float(thresholds[ok].max())


def threshold_report(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> dict[str, Any]:

    block = compute_metrics(labels, probs, threshold)
    block["flagged_fraction"] = float((np.asarray(probs) >= threshold).mean())
    return block


def plot_pr_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    model_name: str,
    recall_target: float,
) -> Figure:

    precision, recall, _ = precision_recall_curve(labels, probs)
    at_op = compute_metrics(labels, probs, threshold)
    at_default = compute_metrics(labels, probs, config.DECISION_THRESHOLD)
    fig = Figure(figsize=(6, 4.6))
    ax = fig.subplots()
    ax.plot(recall, precision, color="tab:blue", label="PR curve (test)")
    ax.axhline(
        float(np.mean(labels)),
        color="gray",
        linestyle=":",
        label="no-skill (prevalence)",
    )
    ax.axvline(
        recall_target,
        color="tab:red",
        linestyle="--",
        alpha=0.5,
        label=f"recall target {recall_target:.2f}",
    )
    ax.scatter(
        [at_op["recall"]],
        [at_op["precision"]],
        color="tab:red",
        zorder=5,
        s=70,
        label=f"operating point t={threshold:.3f}",
    )
    ax.scatter(
        [at_default["recall"]],
        [at_default["precision"]],
        color="black",
        marker="s",
        zorder=5,
        s=50,
        label="default t=0.5",
    )
    ax.set_xlabel("Recall (Abnormal)")
    ax.set_ylabel("Precision (Abnormal)")
    ax.set_title(
        f"{model_name}\nPrecision-recall on test; threshold chosen on validation",
        fontsize=10,
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def run(
    model_key: str, recall_target: float = config.RECALL_TARGET
) -> tuple[dict[str, Any], Figure]:

    val = load_predictions(model_key, "val")
    test = load_predictions(model_key, "test")
    threshold = choose_threshold_for_recall(val["labels"], val["probs"], recall_target)

    at_op_test = threshold_report(test["labels"], test["probs"], threshold)
    at_default_test = threshold_report(
        test["labels"], test["probs"], config.DECISION_THRESHOLD
    )
    report = {
        "model_key": model_key,
        "recall_target": recall_target,
        "threshold": threshold,
        "threshold_chosen_on": "validation split (patients disjoint from train and test)",
        "validation_at_threshold": threshold_report(
            val["labels"], val["probs"], threshold
        ),
        "test_at_threshold": at_op_test,
        "test_at_default_0.5": at_default_test,
        "target_met_on_test": bool(at_op_test["recall"] >= recall_target),
        "precision_cost": float(at_default_test["precision"] - at_op_test["precision"]),
        "specificity_cost": float(
            at_default_test["specificity"] - at_op_test["specificity"]
        ),
        "extra_flagged_fraction": float(
            at_op_test["flagged_fraction"] - at_default_test["flagged_fraction"]
        ),
        "disclaimer": config.DISCLAIMER_SHORT,
    }
    fig = plot_pr_curve(
        test["labels"],
        test["probs"],
        threshold,
        config.MODEL_DISPLAY_NAMES[model_key],
        recall_target,
    )
    write_json(config.operating_point_json(model_key), report)
    config.operating_point_png(model_key).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.operating_point_png(model_key), dpi=150, bbox_inches="tight")
    return report, fig


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description="Recall-target operating point analysis."
    )
    parser.add_argument("--model", choices=config.MODEL_KEYS, default="resnet")
    parser.add_argument("--recall-target", type=float, default=config.RECALL_TARGET)
    args = parser.parse_args(argv)
    try:
        report, _ = run(args.model, args.recall_target)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    t, d = report["test_at_threshold"], report["test_at_default_0.5"]
    print(
        f"[{args.model}] threshold chosen on validation for recall >= {args.recall_target}: {report['threshold']:.4f}"
    )
    print(
        f"  test @ threshold : recall={t['recall']:.3f} precision={t['precision']:.3f} specificity={t['specificity']:.3f} flagged={t['flagged_fraction']:.1%}"
    )
    print(
        f"  test @ 0.5       : recall={d['recall']:.3f} precision={d['precision']:.3f} specificity={d['specificity']:.3f} flagged={d['flagged_fraction']:.1%}"
    )
    print(
        f"  target met on test: {report['target_met_on_test']}   precision cost: {report['precision_cost']:.3f}"
    )
    print(
        f"Wrote {config.operating_point_json(args.model)} and {config.operating_point_png(args.model)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
