from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src import config

PathLike = str | Path


def compute_metrics(
    labels: np.ndarray, probs: np.ndarray, threshold: float = config.DECISION_THRESHOLD
) -> dict[str, Any]:

    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    if labels.shape != probs.shape:
        raise ValueError(
            f"labels {labels.shape} and probs {probs.shape} must have the same shape."
        )
    if np.isnan(probs).any():
        raise ValueError(
            "Predicted probabilities contain NaN; refusing to compute metrics."
        )
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Only one class present in labels; AUC and per-class metrics are undefined."
        )

    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    p, r, f, s = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )
    n = len(labels)
    recall_abn, recall_norm = float(r[1]), float(r[0])
    return {
        "threshold": float(threshold),
        "n": n,
        "n_abnormal": int(labels.sum()),
        "accuracy": float((tp + tn) / n),
        "balanced_accuracy": float((recall_abn + recall_norm) / 2.0),
        "precision": float(p[1]),
        "recall": recall_abn,
        "f1": float(f[1]),
        "specificity": recall_norm,
        "auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "per_class": {
            config.CLASS_NAMES[0]: {
                "precision": float(p[0]),
                "recall": float(r[0]),
                "f1": float(f[0]),
                "support": int(s[0]),
            },
            config.CLASS_NAMES[1]: {
                "precision": float(p[1]),
                "recall": float(r[1]),
                "f1": float(f[1]),
                "support": int(s[1]),
            },
        },
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def run_eval(
    model: Any,
    loader: Iterable,
    device: Any | None = None,
    threshold: float = config.DECISION_THRESHOLD,
) -> dict[str, Any]:

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    is_torch = False
    try:
        import torch

        is_torch = isinstance(model, torch.nn.Module)
    except ImportError:
        torch = None

    if is_torch:
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                if device is not None:
                    images = images.to(device)
                logits = model(images).reshape(-1)
                all_probs.append(torch.sigmoid(logits).float().cpu().numpy())
                all_labels.append(
                    labels.cpu().numpy()
                    if hasattr(labels, "cpu")
                    else np.asarray(labels)
                )
    elif hasattr(model, "predict_proba"):
        for X, y in loader:
            all_probs.append(
                np.asarray(model.predict_proba(X))[:, config.POSITIVE_CLASS_INDEX]
            )
            all_labels.append(np.asarray(y))
    else:
        raise TypeError(
            "run_eval needs a torch.nn.Module or an object with predict_proba()."
        )

    if not all_probs:
        raise ValueError(
            "The loader produced no batches; cannot evaluate an empty split."
        )
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    result = compute_metrics(labels, probs, threshold)
    result["probs"] = probs
    result["labels"] = labels
    return result


def metrics_summary(result: dict[str, Any]) -> dict[str, Any]:

    return {k: v for k, v in result.items() if k not in ("probs", "labels")}


class _NumpyEncoder(json.JSONEncoder):

    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: PathLike, obj: Any) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, cls=_NumpyEncoder), encoding="utf-8")


def read_json(path: PathLike, hint: str) -> Any:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nCreate it with: {hint}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def save_predictions(
    path: PathLike, probs: np.ndarray, labels: np.ndarray, paths: Sequence[str]
) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        probs=np.asarray(probs, dtype=float),
        labels=np.asarray(labels, dtype=int),
        paths=np.asarray(list(paths)),
    )


def load_predictions(
    model_key: str, split: str, hint: str | None = None
) -> dict[str, np.ndarray]:

    path = config.predictions_path(model_key, split)
    if not path.exists():
        cmd = hint or (
            "make train-resnet" if model_key == "resnet" else "make train-classical"
        )
        raise FileNotFoundError(
            f"Predictions not found: {path}\nCreate them with: {cmd}"
        )
    data = np.load(path, allow_pickle=False)
    return {"probs": data["probs"], "labels": data["labels"], "paths": data["paths"]}


def benchmark_latency(
    predict_fn: Callable[[str], float],
    paths: Sequence[str],
    n: int = config.LATENCY_N_IMAGES,
    warmup: int = 3,
) -> float:

    paths = list(paths)[: max(n, 1)]
    for p in paths[:warmup]:
        predict_fn(p)
    start = time.perf_counter()
    for p in paths:
        predict_fn(p)
    return 1000.0 * (time.perf_counter() - start) / len(paths)


def select_error_examples(
    paths: Sequence[str],
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = config.DECISION_THRESHOLD,
    k_fn: int = 2,
    k_fp: int = 2,
) -> pd.DataFrame:

    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    frame = pd.DataFrame({"path": list(paths), "label": labels, "prob": probs})
    fn = (
        frame[(labels == 1) & (preds == 0)]
        .sort_values("prob", ascending=True)
        .head(k_fn)
        .assign(kind="FN")
    )
    fp = (
        frame[(labels == 0) & (preds == 1)]
        .sort_values("prob", ascending=False)
        .head(k_fp)
        .assign(kind="FP")
    )
    return pd.concat([fn, fp], ignore_index=True)


def bright_pixel_fraction(path: PathLike, level: int = 250) -> float:

    from src.mura_utils import load_pil_image

    arr = np.asarray(load_pil_image(path))
    return float((arr >= level).mean())


def error_rates_by_bright_quartile(
    paths: Sequence[str],
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = config.DECISION_THRESHOLD,
) -> pd.DataFrame:

    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(probs, dtype=float) >= threshold).astype(int)
    frac = np.array([bright_pixel_fraction(p) for p in paths])
    quart = pd.qcut(
        pd.Series(frac).rank(method="first"),
        4,
        labels=["Q1 (fewest bright px)", "Q2", "Q3", "Q4 (most bright px)"],
    )
    rows = []
    for q in quart.cat.categories:
        m = (quart == q).to_numpy()
        neg, pos = m & (labels == 0), m & (labels == 1)
        rows.append(
            {
                "quartile": q,
                "n_images": int(m.sum()),
                "median_bright_fraction": float(np.median(frac[m])),
                "false_positive_rate": (
                    float((preds[neg] == 1).mean()) if neg.any() else math.nan
                ),
                "false_negative_rate": (
                    float((preds[pos] == 0).mean()) if pos.any() else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]], title: str = "Confusion matrix"
) -> Figure:

    cm_arr = np.asarray(cm)
    fig = Figure(figsize=(4.2, 4.0))
    ax = fig.subplots()
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1], list(config.CLASS_NAMES))
    ax.set_yticks([0, 1], list(config.CLASS_NAMES))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(cm_arr[i, j]),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
                fontsize=13,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig


def plot_error_examples(examples: pd.DataFrame) -> Figure:

    from src.mura_utils import load_pil_image

    n = max(len(examples), 1)
    fig = Figure(figsize=(3.4 * n, 3.9))
    axes = np.atleast_1d(fig.subplots(1, n))
    for ax, rec in zip(axes, examples.itertuples(index=False)):
        ax.imshow(load_pil_image(rec.path), cmap="gray")
        kind = "missed abnormal (FN)" if rec.kind == "FN" else "false alarm (FP)"
        ax.set_title(
            f"{kind}\ntrue={config.CLASS_NAMES[int(rec.label)]}  P(abn)={rec.prob:.2f}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes[len(examples) :]:
        ax.axis("off")
    return fig


def plot_augmentations(path: PathLike, n: int = 6) -> Figure:

    from src.mura_utils import load_pil_image

    image = load_pil_image(path)
    train_tf = config.get_train_transform()
    mean = np.array(config.IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(config.IMAGENET_STD).reshape(3, 1, 1)
    fig = Figure(figsize=(2.6 * (n + 1), 3.0))
    axes = fig.subplots(1, n + 1)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("original", fontsize=9)
    for ax in axes[1:]:
        t = train_tf(image).numpy()
        ax.imshow(np.clip(t * std + mean, 0, 1).transpose(1, 2, 0))
        ax.set_title("augmented", fontsize=9)
    for ax in axes:
        ax.axis("off")
    return fig
