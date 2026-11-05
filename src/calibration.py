from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import brier_score_loss

from src import config
from src.eval_harness import load_predictions, write_json


def reliability_bins(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = config.CALIBRATION_BINS
) -> list[dict[str, Any]]:

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bins.append(
                {
                    "lo": float(edges[b]),
                    "hi": float(edges[b + 1]),
                    "count": int(m.sum()),
                    "mean_confidence": float(probs[m].mean()),
                    "fraction_abnormal": float(labels[m].mean()),
                }
            )
    return bins


def expected_calibration_error(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = config.CALIBRATION_BINS
) -> float:

    bins = reliability_bins(labels, probs, n_bins)
    total = sum(b["count"] for b in bins)
    return float(
        sum(
            b["count"] / total * abs(b["mean_confidence"] - b["fraction_abnormal"])
            for b in bins
        )
    )


def evaluate_abstention(
    labels: np.ndarray, probs: np.ndarray, low: float, high: float
) -> dict[str, Any]:

    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    auto_norm = probs < low
    auto_abn = probs >= high
    abstain = ~(auto_norm | auto_abn)
    decided = auto_norm | auto_abn
    pred = auto_abn.astype(int)

    def _safe(num: float, den: float) -> float:
        return float(num / den) if den else float("nan")

    tp = int(((pred == 1) & (labels == 1) & decided).sum())
    fp = int(((pred == 1) & (labels == 0) & decided).sum())
    tn = int(((pred == 0) & (labels == 0) & decided).sum())
    fn = int(((pred == 0) & (labels == 1) & decided).sum())
    n_abn = int((labels == 1).sum())
    return {
        "low": float(low),
        "high": float(high),
        "n": len(labels),
        "fraction_abstained": float(abstain.mean()),
        "n_auto_clear": int(auto_norm.sum()),
        "n_auto_flag": int(auto_abn.sum()),
        "n_abstained": int(abstain.sum()),
        "auto_decided_metrics": {
            "accuracy": _safe(tp + tn, tp + tn + fp + fn),
            "precision_abnormal": _safe(tp, tp + fp),
            "recall_abnormal": _safe(tp, tp + fn),
            "recall_normal": _safe(tn, tn + fp),
            "confusion_matrix": [[tn, fp], [fn, tp]],
        },
        "auto_clear_miss_rate": _safe(fn, int(auto_norm.sum())),
        "abnormal_auto_cleared_fraction": _safe(fn, n_abn),
        "abnormal_sent_to_human_fraction": _safe(
            int((abstain & (labels == 1)).sum()), n_abn
        ),
    }


def plot_calibration(
    bins: list[dict[str, Any]], probs: np.ndarray, low: float, high: float, title: str
) -> Figure:

    fig = Figure(figsize=(10, 4.2))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    ax1.plot(
        [b["mean_confidence"] for b in bins],
        [b["fraction_abnormal"] for b in bins],
        marker="o",
        label="model",
    )
    ax1.set_xlabel("Mean predicted P(Abnormal)")
    ax1.set_ylabel("Observed fraction Abnormal")
    ax1.set_title("Reliability diagram (test)")
    ax1.legend()
    ax2.hist(probs, bins=20, range=(0, 1), color="tab:blue", alpha=0.8)
    ax2.axvspan(
        low,
        high,
        color="orange",
        alpha=0.25,
        label=f"abstain band [{low:.2f}, {high:.2f})",
    )
    ax2.set_xlabel("Predicted P(Abnormal)")
    ax2.set_ylabel("Number of images")
    ax2.set_title("Score distribution and abstention band")
    ax2.legend()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def run(model_key: str) -> tuple[dict[str, Any], Figure]:

    test = load_predictions(model_key, "test")
    labels, probs = test["labels"], test["probs"]
    low, high, note = config.resolve_abstention_band(
        config.load_operating_threshold(model_key)
    )
    if note.startswith("WARNING"):
        print(note, file=sys.stderr)
    bins = reliability_bins(labels, probs)
    report = {
        "model_key": model_key,
        "ece": expected_calibration_error(labels, probs),
        "brier": float(brier_score_loss(labels, probs)),
        "bins": bins,
        "band_note": note,
        "abstention": evaluate_abstention(labels, probs, low, high),
        "disclaimer": config.DISCLAIMER_SHORT,
    }
    fig = plot_calibration(
        bins, probs, low, high, config.MODEL_DISPLAY_NAMES[model_key]
    )
    write_json(config.calibration_json(model_key), report)
    config.calibration_png(model_key).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.calibration_png(model_key), dpi=150, bbox_inches="tight")
    return report, fig


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(description="Calibration and abstention analysis.")
    parser.add_argument("--model", choices=config.MODEL_KEYS, default="resnet")
    args = parser.parse_args(argv)
    try:
        report, _ = run(args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    a = report["abstention"]
    m = a["auto_decided_metrics"]
    print(f"[{args.model}] ECE={report['ece']:.3f} Brier={report['brier']:.3f}")
    print(
        f"  band [{a['low']:.3f}, {a['high']:.3f}): abstained {a['fraction_abstained']:.1%} of test images"
    )
    print(
        f"  auto-decided subset: accuracy={m['accuracy']:.3f} recall(abn)={m['recall_abnormal']:.3f} precision(abn)={m['precision_abnormal']:.3f}"
    )
    print(f"  abnormal rate among auto-cleared images: {a['auto_clear_miss_rate']:.3f}")
    print(
        f"Wrote {config.calibration_json(args.model)} and {config.calibration_png(args.model)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
