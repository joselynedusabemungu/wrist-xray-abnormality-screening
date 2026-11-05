from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np

from src import config, mura_utils
from src.eval_harness import (
    benchmark_latency,
    metrics_summary,
    run_eval,
    save_predictions,
    write_json,
)


def _run_epoch(
    model: Any, loader: Any, device: Any, criterion: Any, optimizer: Any | None
) -> tuple[float, float, float]:

    import torch
    from sklearn.metrics import roc_auc_score

    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            targets = labels.float().unsqueeze(1).to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            n += images.size(0)
            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
            labels_all.append(labels.numpy())
    probs, labels_np = np.concatenate(probs_all), np.concatenate(labels_all)
    acc = float(((probs >= 0.5).astype(int) == labels_np).mean())
    auc = (
        float(roc_auc_score(labels_np, probs))
        if len(np.unique(labels_np)) > 1
        else float("nan")
    )
    return total_loss / n, acc, auc


def plot_training_curve(history: dict[str, list[float]]) -> Any:

    from matplotlib.figure import Figure

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig = Figure(figsize=(10, 3.8))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(epochs, history["train_loss"], marker="o", label="train")
    ax1.plot(epochs, history["val_loss"], marker="o", label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")
    ax1.set_title("Loss (early stopping watches validation loss)")
    ax1.legend()
    ax2.plot(
        epochs,
        history["val_auc"],
        marker="o",
        color="tab:green",
        label="validation AUC",
    )
    ax2.plot(
        epochs,
        history["val_acc"],
        marker="o",
        color="tab:gray",
        label="validation accuracy @0.5",
    )
    ax2.set_xlabel("epoch")
    ax2.set_title("Validation AUC / accuracy")
    ax2.legend()
    fig.tight_layout()
    return fig


def train(
    epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    max_train_images: int | None = None,
    pretrained: bool = True,
    cpu_deterministic: bool = False,
    num_workers: int = config.NUM_WORKERS,
) -> dict[str, Any]:

    import torch
    from torch import nn

    mura_utils.require_torch()
    mura_utils.seed_everything(cpu_deterministic=cpu_deterministic)
    device = mura_utils.get_device(force_cpu=cpu_deterministic)
    splits = mura_utils.load_splits(max_train_images=max_train_images)
    print("Split sizes:", {k: len(v) for k, v in splits.items()})
    loaders = mura_utils.make_dataloaders(
        splits, batch_size=batch_size, num_workers=num_workers
    )

    model = mura_utils.build_resnet18(pretrained=pretrained).to(device)
    n_params = mura_utils.count_parameters(model)
    pos_weight = None
    if config.USE_POS_WEIGHT:
        n_pos = float((splits["train"].label == 1).sum())
        pos_weight = torch.tensor(
            [(len(splits["train"]) - n_pos) / max(n_pos, 1.0)], device=device
        )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_auc": [],
    }
    best_val, best_epoch, bad_epochs = float("inf"), 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, _ = _run_epoch(
            model, loaders["train"], device, criterion, optimizer
        )
        va_loss, va_acc, va_auc = _run_epoch(
            model, loaders["val"], device, criterion, None
        )
        for key, value in zip(history, (tr_loss, va_loss, tr_acc, va_acc, va_auc)):
            history[key].append(float(value))
        print(
            f"Epoch {epoch:>2}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} auc {va_auc:.3f}"
        )
        if va_loss < best_val:
            best_val, best_epoch, bad_epochs = va_loss, epoch, 0
            mura_utils.save_checkpoint(
                model,
                config.RESNET_CKPT_PATH,
                {
                    "best_epoch": epoch,
                    "val_loss": float(va_loss),
                    "seed": config.RANDOM_SEED,
                    "torch_version": str(torch.__version__),
                },
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config.EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping: no validation-loss improvement for {config.EARLY_STOP_PATIENCE} epochs."
                )
                break
    train_time = time.perf_counter() - t_start

    model, meta = mura_utils.load_resnet_checkpoint(config.RESNET_CKPT_PATH, device)
    print(
        f"Loaded best checkpoint from epoch {meta.get('best_epoch')} (val loss {meta.get('val_loss'):.4f})."
    )

    report: dict[str, Any] = {
        "model_key": "resnet",
        "display_name": config.MODEL_DISPLAY_NAMES["resnet"],
        "device": str(device),
        "n_parameters": n_params,
        "pretrained": pretrained,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "train_time_s": float(train_time),
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": config.WEIGHT_DECAY,
            "max_epochs": epochs,
            "early_stop_patience": config.EARLY_STOP_PATIENCE,
            "pos_weight": config.USE_POS_WEIGHT,
            "seed": config.RANDOM_SEED,
        },
        "history": history,
        "smoke_test": max_train_images is not None,
        "torch_version": str(torch.__version__),
    }
    for name in ("val", "test"):
        result = run_eval(model, loaders[name], device)
        report[name] = metrics_summary(result)
        save_predictions(
            config.predictions_path("resnet", name),
            result["probs"],
            result["labels"],
            splits[name].path,
        )
        print(
            f"[resnet] {name}: acc={result['accuracy']:.3f} precision={result['precision']:.3f} "
            f"recall={result['recall']:.3f} f1={result['f1']:.3f} auc={result['auc']:.3f}"
        )

    eval_tf = config.get_eval_transform()

    def _predict(path: str) -> float:
        with torch.no_grad():
            x = eval_tf(mura_utils.load_pil_image(path)).unsqueeze(0).to(device)
            return float(torch.sigmoid(model(x)).item())

    report["inference_ms"] = benchmark_latency(_predict, splits["test"].path.tolist())
    print(
        f"[resnet] single-image latency (read+preprocess+forward, {device}): {report['inference_ms']:.1f} ms"
    )

    write_json(config.RESNET_METRICS_JSON, report)
    fig = plot_training_curve(history)
    config.TRAINING_CURVE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.TRAINING_CURVE_PNG, dpi=150, bbox_inches="tight")
    print(
        f"Wrote {config.RESNET_METRICS_JSON}, {config.TRAINING_CURVE_PNG}, {config.RESNET_CKPT_PATH}"
    )
    if report["smoke_test"]:
        print(
            "NOTE: smoke-test run. Metrics are NOT meaningful; re-run without --max-train-images."
        )
    return report


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 on MURA wrist.")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Subsample TRAIN for a quick plumbing test.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Random init (offline smoke tests only).",
    )
    parser.add_argument(
        "--cpu-deterministic",
        action="store_true",
        help="Force CPU and deterministic algorithms.",
    )
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    args = parser.parse_args(argv)
    try:
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_train_images=args.max_train_images,
            pretrained=not args.no_pretrained,
            cpu_deterministic=args.cpu_deterministic,
            num_workers=args.num_workers,
        )
    except (
        FileNotFoundError,
        AssertionError,
        ValueError,
        ImportError,
        RuntimeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
