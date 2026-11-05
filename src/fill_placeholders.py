from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from src import config
from src.eval_harness import read_json

TOKEN_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
DOC_FILES = [
    "README.md",
    "docs/DECISION.md",
    "docs/EVALUATION.md",
    "docs/DATA_CARD.md",
    "docs/MODEL_CARD.md",
]
MODEL_TOKEN_PREFIX = {"GRAYSCALE": "grayscale", "HOG": "hog", "RESNET": "resnet"}
METRIC_FIELDS = {
    "ACCURACY": "accuracy",
    "PRECISION": "precision",
    "RECALL": "recall",
    "F1": "f1",
    "AUC": "auc",
}


def _fmt3(x: float) -> str:
    return f"{x:.3f}"


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def build_values() -> dict[str, str]:

    values: dict[str, str] = {}

    data = read_json(config.DATA_SUMMARY_JSON, "make data-check")["splits"]
    values["N_TRAIN_IMAGES"] = f"{data['train']['n_images']:,}"
    values["N_VAL_IMAGES"] = f"{data['val']['n_images']:,}"
    values["N_TEST_IMAGES"] = f"{data['test']['n_images']:,}"
    values["N_TRAIN_PATIENTS"] = f"{data['train']['n_patients']:,}"
    values["N_TEST_PATIENTS"] = f"{data['test']['n_patients']:,}"
    values["TRAIN_PCT_ABNORMAL"] = f"{data['train']['pct_abnormal']:.1f}%"
    values["TEST_PCT_ABNORMAL"] = f"{data['test']['pct_abnormal']:.1f}%"

    classical = read_json(config.CLASSICAL_METRICS_JSON, "make train-classical")
    resnet = read_json(config.RESNET_METRICS_JSON, "make train-resnet")
    reports = {
        "grayscale": classical["grayscale"],
        "hog": classical["hog"],
        "resnet": resnet,
    }
    for prefix, key in MODEL_TOKEN_PREFIX.items():
        rep = reports[key]
        for suffix, field in METRIC_FIELDS.items():
            values[f"{prefix}_{suffix}"] = _fmt3(rep["test"][field])
        values[f"{prefix}_TRAIN_TIME"] = f"{rep['train_time_s']:.0f}"
        values[f"{prefix}_INFERENCE_MS"] = f"{rep['inference_ms']:.1f}"

    op = read_json(config.operating_point_json("resnet"), "make operating-point")
    op_hog = read_json(config.operating_point_json("hog"), "make operating-point")
    values["OP_RECALL_TARGET"] = f"{op['recall_target']:.2f}"
    values["OP_THRESHOLD"] = f"{op['threshold']:.3f}"
    values["OP_TEST_RECALL"] = _fmt3(op["test_at_threshold"]["recall"])
    values["OP_TEST_PRECISION"] = _fmt3(op["test_at_threshold"]["precision"])
    values["OP_TEST_PRECISION_AT_05"] = _fmt3(op["test_at_default_0.5"]["precision"])
    values["OP_TEST_RECALL_AT_05"] = _fmt3(op["test_at_default_0.5"]["recall"])
    values["OP_TEST_FLAGGED_FRACTION"] = _pct(
        op["test_at_threshold"]["flagged_fraction"]
    )
    values["HOG_OP_TEST_PRECISION"] = _fmt3(op_hog["test_at_threshold"]["precision"])
    values["HOG_OP_TEST_RECALL"] = _fmt3(op_hog["test_at_threshold"]["recall"])

    cal = read_json(config.calibration_json("resnet"), "make calibration")
    ab = cal["abstention"]
    values["ABSTAIN_LOW"] = f"{ab['low']:.3f}"
    values["ABSTAIN_HIGH"] = f"{ab['high']:.2f}"
    values["ABSTAIN_FRACTION"] = _pct(ab["fraction_abstained"])
    values["AUTO_CLEAR_MISS_RATE"] = _pct(ab["auto_clear_miss_rate"])
    values["ECE"] = _fmt3(cal["ece"])
    return values


def _render(text: str, values: dict[str, str], name: str) -> str:
    unknown = sorted({t for t in TOKEN_RE.findall(text) if t not in values})
    if unknown:
        raise KeyError(
            f"{name}: no data source defined for token(s) {unknown}. Add them to src/fill_placeholders.py."
        )
    return TOKEN_RE.sub(lambda m: values[m.group(1)], text)


def fill(root: Path) -> int:
    values = build_values()
    count = 0
    for rel in DOC_FILES:
        path = root / rel
        tmpl = path.with_suffix(path.suffix + ".tmpl")
        if not tmpl.exists():
            if not TOKEN_RE.search(path.read_text(encoding="utf-8")):
                continue
            shutil.copyfile(path, tmpl)
        path.write_text(
            _render(tmpl.read_text(encoding="utf-8"), values, rel), encoding="utf-8"
        )
        count += 1
        print(f"filled {rel}")
    return count


def check(root: Path) -> int:

    leftovers: dict[str, list[str]] = {}
    for rel in DOC_FILES:
        found = sorted(set(TOKEN_RE.findall((root / rel).read_text(encoding="utf-8"))))
        if found:
            leftovers[rel] = found
    if leftovers:
        for rel, toks in leftovers.items():
            print(
                f"UNFILLED in {rel}: {', '.join('{{' + t + '}}' for t in toks)}",
                file=sys.stderr,
            )
        print("Run the pipeline, then `make fill-docs`.", file=sys.stderr)
        return 1
    print("OK: no unfilled placeholders in README.md or docs/*.md")
    return 0


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description="Fill / check README and docs placeholders."
    )
    parser.add_argument(
        "--check", action="store_true", help="Only verify that no placeholders remain."
    )
    args = parser.parse_args(argv)
    root = config.PROJECT_ROOT
    if args.check:
        return check(root)
    try:
        fill(root)
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return check(root)


if __name__ == "__main__":
    sys.exit(main())
