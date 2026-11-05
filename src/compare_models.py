from __future__ import annotations

import math
import sys

import pandas as pd

from src import config
from src.eval_harness import read_json

COLUMNS = [
    "model_key",
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "specificity",
    "train_time_s",
    "inference_ms",
    "interpretable",
]


def build_comparison() -> pd.DataFrame:

    classical = read_json(config.CLASSICAL_METRICS_JSON, "make train-classical")
    resnet = read_json(config.RESNET_METRICS_JSON, "make train-resnet")
    reports = {
        "grayscale": classical["grayscale"],
        "hog": classical["hog"],
        "resnet": resnet,
    }
    rows = []
    for key in config.MODEL_KEYS:
        rep = reports[key]
        test = rep["test"]
        rows.append(
            {
                "model_key": key,
                "model": config.MODEL_DISPLAY_NAMES[key],
                "accuracy": test["accuracy"],
                "precision": test["precision"],
                "recall": test["recall"],
                "f1": test["f1"],
                "auc": test["auc"],
                "specificity": test["specificity"],
                "train_time_s": rep["train_time_s"],
                "inference_ms": rep["inference_ms"],
                "interpretable": config.MODEL_INTERPRETABLE[key],
            }
        )
    df = pd.DataFrame(rows, columns=COLUMNS)
    numeric = df.drop(columns=["model_key", "model", "interpretable"])
    bad = numeric.isna() | numeric.map(lambda v: isinstance(v, float) and math.isnan(v))
    if bad.to_numpy().any():
        raise ValueError(
            f"NaN/missing values in comparison table:\n{df[bad.any(axis=1)]}"
        )
    if len(df) != 3:
        raise ValueError(f"Expected 3 model rows, got {len(df)}.")
    smoke = bool(classical.get("smoke_test")) or bool(resnet.get("smoke_test"))
    df.attrs["smoke_test"] = smoke
    return df


def to_markdown(df: pd.DataFrame) -> str:

    header = "| Model | Accuracy | Precision (Abn.) | Recall (Abn.) | F1 (Abn.) | AUC | Train time (s) | Inference (ms/img) | Interpretable |\n"
    header += "|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
    lines = []
    for r in df.itertuples(index=False):
        lines.append(
            f"| {r.model} | {r.accuracy:.3f} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {r.auc:.3f} | "
            f"{r.train_time_s:.0f} | {r.inference_ms:.1f} | {r.interpretable} |"
        )
    note = (
        "\nTest set = official MURA `valid` wrist studies (never used for fitting or model selection). "
        "Threshold 0.5; positive class = Abnormal. Train time includes feature extraction (classical) or all epochs (ResNet). "
        "Inference = mean single-image latency, file read to probability, on this machine.\n"
        "\n*Coursework / portfolio prototype. NOT a diagnostic tool.*\n"
    )
    if df.attrs.get("smoke_test"):
        note = (
            "\n> **SMOKE-TEST RUN: these numbers are NOT meaningful. Re-run the full training.**\n"
            + note
        )
    return "# Model comparison (test set)\n\n" + header + "\n".join(lines) + "\n" + note


def main(argv: list[str] | None = None) -> int:

    try:
        df = build_comparison()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.MODEL_COMPARISON_CSV, index=False)
    config.MODEL_COMPARISON_MD.write_text(to_markdown(df), encoding="utf-8")
    print(to_markdown(df))
    print(f"Wrote {config.MODEL_COMPARISON_CSV} and {config.MODEL_COMPARISON_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
