from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from src import config, mura_utils
from src.eval_harness import (
    benchmark_latency,
    metrics_summary,
    run_eval,
    save_predictions,
    write_json,
)

MODEL_PATHS = {"grayscale": config.GRAYSCALE_MODEL_PATH, "hog": config.HOG_MODEL_PATH}


def train_one(kind: str, splits: dict) -> dict[str, Any]:

    from sklearn.metrics import roc_auc_score

    print(f"\n=== {config.MODEL_DISPLAY_NAMES[kind]} ===")
    t0 = time.perf_counter()
    feats = {}
    for name in ("train", "val", "test"):
        print(
            f"[{kind}] extracting features for {name} ({len(splits[name])} images)..."
        )
        feats[name] = mura_utils.make_feature_matrix(splits[name].path, kind)
    feature_time = time.perf_counter() - t0
    y = {name: splits[name].label.to_numpy() for name in splits}
    print(f"[{kind}] feature matrix: {feats['train'].shape[1]} features per image")

    best: tuple[float, float, Any] | None = None
    fit_time = 0.0
    for C in config.CLASSICAL_C_GRID:
        t1 = time.perf_counter()
        model = mura_utils.build_classical_pipeline(C)
        model.fit(feats["train"], y["train"])
        fit_time += time.perf_counter() - t1
        val_auc = float(
            roc_auc_score(y["val"], model.predict_proba(feats["val"])[:, 1])
        )
        print(f"[{kind}] C={C:<7} val AUC={val_auc:.4f}")
        if best is None or val_auc > best[0]:
            best = (val_auc, C, model)
    assert best is not None
    _, best_C, model = best
    print(f"[{kind}] selected C={best_C} (by validation AUC)")

    report: dict[str, Any] = {
        "model_key": kind,
        "display_name": config.MODEL_DISPLAY_NAMES[kind],
        "n_features": int(feats["train"].shape[1]),
        "selected_C": best_C,
        "c_grid": list(config.CLASSICAL_C_GRID),
        "train_time_s": float(feature_time + fit_time),
        "seed": config.RANDOM_SEED,
    }
    for name in ("val", "test"):
        result = run_eval(model, mura_utils.iter_numpy_batches(feats[name], y[name]))
        report[name] = metrics_summary(result)
        save_predictions(
            config.predictions_path(kind, name),
            result["probs"],
            result["labels"],
            splits[name].path,
        )
        print(
            f"[{kind}] {name}: acc={result['accuracy']:.3f} precision={result['precision']:.3f} "
            f"recall={result['recall']:.3f} f1={result['f1']:.3f} auc={result['auc']:.3f}"
        )

    extractor = mura_utils.FEATURE_FUNCTIONS[kind]
    report["inference_ms"] = benchmark_latency(
        lambda p: float(model.predict_proba(extractor(p).reshape(1, -1))[0, 1]),
        splits["test"].path.tolist(),
    )
    print(
        f"[{kind}] single-image latency (read+features+predict): {report['inference_ms']:.1f} ms"
    )

    metadata = {
        "kind": kind,
        "classical_image_size": list(config.CLASSICAL_IMAGE_SIZE),
        "selected_C": best_C,
        "seed": config.RANDOM_SEED,
        "disclaimer": config.DISCLAIMER_SHORT,
    }
    mura_utils.save_bundle(model, MODEL_PATHS[kind], metadata)
    print(f"[{kind}] saved {MODEL_PATHS[kind]}")
    return report


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description="Train grayscale+LR and HOG+LR baselines."
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Subsample TRAIN for a quick plumbing test.",
    )
    args = parser.parse_args(argv)

    mura_utils.seed_everything()
    try:
        splits = mura_utils.load_splits(max_train_images=args.max_train_images)
    except (FileNotFoundError, AssertionError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Split sizes:", {k: len(v) for k, v in splits.items()})

    reports = {kind: train_one(kind, splits) for kind in ("grayscale", "hog")}
    reports["smoke_test"] = args.max_train_images is not None
    write_json(config.CLASSICAL_METRICS_JSON, reports)
    print(f"\nWrote {config.CLASSICAL_METRICS_JSON}")
    if reports["smoke_test"]:
        print(
            "NOTE: smoke-test run. Re-run without --max-train-images for real results."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
