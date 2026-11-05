from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _resolve_mura_root() -> Path:

    raw = os.environ.get("MURA_ROOT")
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    return PROJECT_ROOT / "MURA-v1.1"


MURA_ROOT: Path = _resolve_mura_root()
BODY_PART: str = "XR_WRIST"
SPLIT_TRAIN: str = "train"
SPLIT_VALID: str = "valid"

MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

GRAYSCALE_MODEL_PATH: Path = MODELS_DIR / "mura_grayscale_logistic.joblib"
HOG_MODEL_PATH: Path = MODELS_DIR / "mura_hog_logistic.joblib"
RESNET_CKPT_PATH: Path = MODELS_DIR / "resnet18_wrist.pt"
ONNX_PATH: Path = MODELS_DIR / "resnet18_wrist.onnx"

CLASSICAL_METRICS_JSON: Path = OUTPUTS_DIR / "classical_metrics.json"
RESNET_METRICS_JSON: Path = OUTPUTS_DIR / "resnet_metrics.json"
MODEL_COMPARISON_MD: Path = OUTPUTS_DIR / "model_comparison.md"
MODEL_COMPARISON_CSV: Path = OUTPUTS_DIR / "model_comparison.csv"
DATA_SUMMARY_JSON: Path = OUTPUTS_DIR / "data_summary.json"
TRAINING_CURVE_PNG: Path = OUTPUTS_DIR / "resnet_training_curve.png"
PREPROCESSING_CONTRACT_JSON: Path = OUTPUTS_DIR / "preprocessing_contract.json"

MODEL_KEYS: tuple[str, ...] = ("grayscale", "hog", "resnet")
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "grayscale": "Grayscale pixels + logistic regression",
    "hog": "HOG features + logistic regression",
    "resnet": "ResNet18 (ImageNet transfer learning)",
}
MODEL_INTERPRETABLE: dict[str, str] = {
    "grayscale": "Yes (one coefficient per pixel)",
    "hog": "Partly (one coefficient per HOG cell/orientation)",
    "resnet": "No (post-hoc saliency only)",
}


def predictions_path(model_key: str, split: str) -> Path:

    return OUTPUTS_DIR / f"{model_key}_{split}_predictions.npz"


def operating_point_json(model_key: str) -> Path:
    return OUTPUTS_DIR / f"operating_point_{model_key}.json"


def operating_point_png(model_key: str) -> Path:
    return OUTPUTS_DIR / f"operating_point_{model_key}.png"


def calibration_json(model_key: str) -> Path:
    return OUTPUTS_DIR / f"calibration_{model_key}.json"


def calibration_png(model_key: str) -> Path:
    return OUTPUTS_DIR / f"calibration_{model_key}.png"


RANDOM_SEED: int = 42
MAX_UNREADABLE_FRACTION: float = 0.01
UNREADABLE_IMAGES_JSON: Path = OUTPUTS_DIR / "unreadable_images.json"
VAL_FRACTION: float = 0.15


CLASS_NAMES: tuple[str, str] = ("Normal", "Abnormal")
POSITIVE_CLASS_INDEX: int = 1


IMG_SIZE: int = 224
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


HFLIP_PROB: float = 0.5
ROTATION_DEGREES: float = 7.0
JITTER_BRIGHTNESS: float = 0.10
JITTER_CONTRAST: float = 0.10


CLASSICAL_IMAGE_SIZE: tuple[int, int] = (128, 128)
HOG_ORIENTATIONS: int = 9
HOG_PIXELS_PER_CELL: tuple[int, int] = (16, 16)
HOG_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)
CLASSICAL_C_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)


BATCH_SIZE: int = 32
NUM_EPOCHS: int = 15
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-4
EARLY_STOP_PATIENCE: int = 3
USE_POS_WEIGHT: bool = False
NUM_WORKERS: int = int(os.environ.get("MURA_NUM_WORKERS", "0"))


DECISION_THRESHOLD: float = 0.5
RECALL_TARGET: float = 0.95
ABSTAIN_HIGH: float = 0.65
ABSTAIN_LOW_FALLBACK: float = 0.35
ABSTAIN_MIN_GAP: float = 0.05
CALIBRATION_BINS: int = 10
LATENCY_N_IMAGES: int = 50


DISCLAIMER_SHORT: str = (
    "Coursework / portfolio prototype. NOT a diagnostic tool. "
    "Not clinically validated. Never use on real patient data."
)
DISCLAIMER_LONG: str = (
    "This software is a prototype trained on the public "
    "MURA research dataset. It is NOT a medical device, has NOT been clinically "
    "validated, and must NOT be used to diagnose, triage or treat any patient. "
    "Do not upload real patient images."
)


@lru_cache(maxsize=1)
def get_eval_transform() -> Any:

    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError() from exc
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


@lru_cache(maxsize=1)
def get_train_transform() -> Any:

    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError() from exc
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=HFLIP_PROB),
            transforms.RandomRotation(
                degrees=ROTATION_DEGREES,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=JITTER_BRIGHTNESS, contrast=JITTER_CONTRAST
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


def describe_transform(transform: Any) -> list[tuple]:

    spec: list[tuple] = []
    for step in transform.transforms:
        name = type(step).__name__
        if name == "Grayscale":
            spec.append((name, int(step.num_output_channels)))
        elif name == "Resize":
            size = step.size
            size = (
                tuple(int(s) for s in size)
                if isinstance(size, (list, tuple))
                else (int(size), int(size))
            )
            spec.append((name, size))
        elif name == "Normalize":
            spec.append(
                (
                    name,
                    tuple(round(float(x), 6) for x in step.mean),
                    tuple(round(float(x), 6) for x in step.std),
                )
            )
        else:
            spec.append((name,))
    return spec


def __getattr__(name: str) -> Any:
    if name == "eval_transform":
        return get_eval_transform()
    if name == "train_transform":
        return get_train_transform()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_operating_threshold(model_key: str = "resnet") -> float | None:
    path = operating_point_json(model_key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["threshold"])
    except (ValueError, KeyError, TypeError, OSError):
        return None


def resolve_abstention_band(
    recall_threshold: float | None,
) -> tuple[float, float, str]:

    high = ABSTAIN_HIGH
    if recall_threshold is None:
        return (
            ABSTAIN_LOW_FALLBACK,
            high,
            (
                f"bound {ABSTAIN_LOW_FALLBACK}. Run `python -m src.operating_point` first."
            ),
        )
    low = float(recall_threshold)
    note = "lower bound = recall-target threshold chosen on the validation split"
    if low > high - ABSTAIN_MIN_GAP:
        low = high - ABSTAIN_MIN_GAP
        note += f" (clamped so the band is at least {ABSTAIN_MIN_GAP} wide)"
    return low, high, note
