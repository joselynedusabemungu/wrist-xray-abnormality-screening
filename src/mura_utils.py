from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from PIL import Image

from src import config

try:
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as _DatasetBase
except ImportError:
    torch = None
    DataLoader = None
    _DatasetBase = object

ImageSource = str | Path | BinaryIO
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


def seed_everything(
    seed: int = config.RANDOM_SEED, cpu_deterministic: bool = False
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if cpu_deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def require_torch() -> None:

    if torch is None:
        raise ImportError(
            "PyTorch is not installed in this environment. Run `make install` "
            "(or `pip install -r requirements.txt`), then retry. "
            'Check with: python -c "import torch; print(torch.__version__)"'
        )


def get_device(force_cpu: bool = False, verbose: bool = True) -> torch.device:

    require_torch()
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")
        if verbose:
            print(f"[device] Using CUDA: {torch.cuda.get_device_name(0)}")
        return device
    if verbose:
        print(
            "[device] WARNING: running on CPU. ResNet18 fine-tuning works on CPU but is "
            "much slower than on a GPU. Use `--max-train-images 256 --epochs 1` for a quick "
            "plumbing check before committing to a full run."
        )
    return torch.device("cpu")


def validate_mura_root(
    root: str | Path = config.MURA_ROOT, body_part: str = config.BODY_PART
) -> Path:

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"MURA root not found: {root}\n"
            "Fix: download MURA-v1.1 (Stanford ML Group), extract it, then either place it at "
            "./MURA-v1.1 or set the MURA_ROOT environment variable, e.g.\n"
            "  export MURA_ROOT=/path/to/MURA-v1.1"
        )
    for split in (config.SPLIT_TRAIN, config.SPLIT_VALID):
        folder = root / split / body_part
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Expected folder missing: {folder}\n"
                f"MURA_ROOT must point at the directory that directly contains "
                f"'{config.SPLIT_TRAIN}/' and '{config.SPLIT_VALID}/' (currently {root}). "
                f"Inside each there must be a '{body_part}/' folder with patient*/study*_* subfolders."
            )
        if not any(folder.glob("patient*")):
            raise FileNotFoundError(
                f"No 'patient*' folders found in {folder}. Is the download complete?"
            )
    return root


def scan_studies(
    root: str | Path = config.MURA_ROOT,
    split: str = config.SPLIT_TRAIN,
    body_part: str = config.BODY_PART,
) -> pd.DataFrame:

    base = Path(root) / split / body_part
    if not base.is_dir():
        raise FileNotFoundError(f"Missing folder: {base}. Check MURA_ROOT.")
    rows: list[dict[str, Any]] = []
    skipped = 0
    for study_dir in sorted(base.glob("patient*/study*_*")):
        if not study_dir.is_dir():
            continue
        name = study_dir.name.lower()
        if name.endswith("positive"):
            label = 1
        elif name.endswith("negative"):
            label = 0
        else:
            skipped += 1
            continue
        rows.append(
            {
                "study_dir": str(study_dir),
                "patient_id": study_dir.parent.name,
                "study_id": f"{study_dir.parent.name}/{study_dir.name}",
                "label": label,
                "split": split,
            }
        )
    if skipped:
        warnings.warn(
            f"{skipped} study folders under {base} had no positive/negative suffix and were skipped."
        )
    if not rows:
        raise FileNotFoundError(f"No study folders found under {base}.")
    return pd.DataFrame(rows)


def expand_studies_to_images(study_df: pd.DataFrame) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []
    for rec in study_df.itertuples(index=False):
        for img in sorted(Path(rec.study_dir).iterdir()):
            if img.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "path": str(img),
                        "patient_id": rec.patient_id,
                        "study_id": rec.study_id,
                        "label": int(rec.label),
                        "split": rec.split,
                    }
                )
    if not rows:
        raise FileNotFoundError(
            "Study folders were found but they contain no image files (.png/.jpg)."
        )
    return pd.DataFrame(rows)


def build_index(
    root: str | Path = config.MURA_ROOT,
    split: str = config.SPLIT_TRAIN,
    body_part: str = config.BODY_PART,
) -> pd.DataFrame:

    return expand_studies_to_images(scan_studies(root, split, body_part))


def assert_no_patient_leakage(splits: dict[str, pd.DataFrame]) -> None:

    names = list(splits)
    for i, a in enumerate(names):
        if len(splits[a]) == 0:
            raise AssertionError(f"Split '{a}' is empty.")
        for b in names[i + 1 :]:
            shared_patients = set(splits[a].patient_id) & set(splits[b].patient_id)
            if shared_patients:
                example = sorted(shared_patients)[:5]
                raise AssertionError(
                    f"PATIENT LEAKAGE between '{a}' and '{b}': {len(shared_patients)} patients "
                    f"appear in both (e.g. {example}). Do not trust any metric until this is fixed."
                )
            shared_paths = set(splits[a].path) & set(splits[b].path)
            if shared_paths:
                raise AssertionError(
                    f"IMAGE LEAKAGE between '{a}' and '{b}': {len(shared_paths)} identical files."
                )


def make_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    val_fraction: float = config.VAL_FRACTION,
    seed: int = config.RANDOM_SEED,
) -> dict[str, pd.DataFrame]:

    from sklearn.model_selection import train_test_split

    patient_label = train_df.groupby("patient_id")["label"].max()
    train_p, val_p = train_test_split(
        patient_label.index.to_numpy(),
        test_size=val_fraction,
        random_state=seed,
        stratify=patient_label.to_numpy(),
    )
    splits = {
        "train": train_df[train_df.patient_id.isin(set(train_p))].reset_index(
            drop=True
        ),
        "val": train_df[train_df.patient_id.isin(set(val_p))].reset_index(drop=True),
        "test": valid_df.reset_index(drop=True),
    }
    assert_no_patient_leakage(splits)
    return splits


def _is_readable(path: str) -> bool:
    try:
        load_pil_image(path)
        return True
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def drop_unreadable_images(
    df: pd.DataFrame, name: str
) -> tuple[pd.DataFrame, list[str]]:
    from joblib import Parallel, delayed

    ok = Parallel(n_jobs=-1)(delayed(_is_readable)(p) for p in df.path)
    mask = np.array(ok, dtype=bool)
    dropped = df.path[~mask].tolist()
    if dropped:
        print(
            f"WARNING: {len(dropped)} unreadable/truncated image(s) in '{name}' will be EXCLUDED:",
            file=sys.stderr,
        )
        for p in dropped:
            print(f"   {p}", file=sys.stderr)
        if len(dropped) / len(df) > config.MAX_UNREADABLE_FRACTION:
            raise RuntimeError(
                f"{len(dropped)} of {len(df)} images in '{name}' are unreadable. "
                "The dataset copy is probably corrupted; re-copy MURA-v1.1 and retry."
            )
    return df[mask].reset_index(drop=True), dropped


def load_splits(
    root: str | Path = config.MURA_ROOT,
    body_part: str = config.BODY_PART,
    max_train_images: int | None = None,
) -> dict[str, pd.DataFrame]:

    root = validate_mura_root(root, body_part)
    train_df = build_index(root, config.SPLIT_TRAIN, body_part)
    valid_df = build_index(root, config.SPLIT_VALID, body_part)
    train_df, _bad_train = drop_unreadable_images(train_df, "train")
    valid_df, _bad_valid = drop_unreadable_images(valid_df, "valid")
    splits = make_splits(train_df, valid_df)
    if max_train_images is not None and max_train_images < len(splits["train"]):
        sub = splits["train"].sample(
            n=max_train_images, random_state=config.RANDOM_SEED
        )
        if sub.label.nunique() < 2:
            raise ValueError(
                "Smoke-test subsample contains one class only; raise --max-train-images."
            )
        splits["train"] = sub.reset_index(drop=True)
        print(
            f"[smoke] TRAIN subsampled to {len(sub)} images. Metrics are NOT meaningful in this mode."
        )
    assert_no_patient_leakage(splits)
    return splits


def class_balance(df: pd.DataFrame) -> dict[str, Any]:

    n = len(df)
    n_abn = int((df.label == 1).sum())
    return {
        "n_images": n,
        "n_normal": n - n_abn,
        "n_abnormal": n_abn,
        "pct_abnormal": round(100.0 * n_abn / n, 2) if n else float("nan"),
        "n_studies": int(df.study_id.nunique()),
        "n_patients": int(df.patient_id.nunique()),
    }


def summarize_splits(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:

    return {name: class_balance(df) for name, df in splits.items()}


def compare_with_csv(
    root: str | Path = config.MURA_ROOT, body_part: str = config.BODY_PART
) -> dict[str, dict[str, int]] | None:

    root = Path(root)
    out: dict[str, dict[str, int]] = {}
    for split in (config.SPLIT_TRAIN, config.SPLIT_VALID):
        csv_path = root / f"{split}_image_paths.csv"
        if not csv_path.exists():
            return None
        needle = f"/{body_part}/"
        in_csv = sum(
            1
            for line in csv_path.read_text(encoding="utf-8").splitlines()
            if needle in line
        )
        on_disk = sum(
            1
            for p in (root / split / body_part).glob("patient*/study*_*/*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        out[split] = {"on_disk": on_disk, "in_csv": in_csv}
    return out


def sample_image_properties(
    df: pd.DataFrame, n: int = 200, seed: int = config.RANDOM_SEED
) -> pd.DataFrame:

    sample = df.sample(n=min(n, len(df)), random_state=seed)
    rows = []
    for p in sample.path:
        with Image.open(p) as im:
            rows.append({"width": im.width, "height": im.height, "mode": im.mode})
    props = pd.DataFrame(rows)
    props["aspect_ratio"] = props.width / props.height
    return props


def load_pil_image(source: ImageSource) -> Image.Image:

    with Image.open(source) as im:
        im.load()
        if im.mode in ("I", "I;16", "I;16B", "I;16L"):
            arr = np.asarray(im, dtype=np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            arr = (arr - lo) / max(hi - lo, 1e-6) * 255.0
            return Image.fromarray(arr.astype(np.uint8), mode="L")
        return im.convert("L")


class MuraDataset(_DatasetBase):

    def __init__(self, frame: pd.DataFrame, transform: Any) -> None:
        require_torch()
        self.paths: list[str] = frame["path"].tolist()
        self.labels: list[int] = frame["label"].astype("int64").tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Any, int]:

        path = self.paths[idx]
        try:
            image = load_pil_image(path)
        except Exception as exc:
            raise RuntimeError(f"Could not read image '{path}': {exc}") from exc
        return self.transform(image), self.labels[idx]


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloaders(
    splits: dict[str, pd.DataFrame],
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> dict[str, DataLoader]:

    require_torch()
    pin = bool(torch.cuda.is_available())
    gen = torch.Generator()
    gen.manual_seed(config.RANDOM_SEED)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin,
        "worker_init_fn": _seed_worker if num_workers > 0 else None,
    }
    return {
        "train": DataLoader(
            MuraDataset(splits["train"], config.get_train_transform()),
            shuffle=True,
            generator=gen,
            **common,
        ),
        "val": DataLoader(
            MuraDataset(splits["val"], config.get_eval_transform()),
            shuffle=False,
            **common,
        ),
        "test": DataLoader(
            MuraDataset(splits["test"], config.get_eval_transform()),
            shuffle=False,
            **common,
        ),
    }


def build_resnet18(pretrained: bool = True) -> torch.nn.Module:

    require_torch()
    from torch import nn
    from torchvision import models

    if pretrained:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:
            raise RuntimeError(
                "Could not load ImageNet ResNet18 weights (a one-time download at TRAINING time). "
                "If you are offline: download resnet18-f37072fd.pth on another machine and place it in "
                "~/.cache/torch/hub/checkpoints/, or pass --no-pretrained (not recommended; much weaker). "
                f"Original error: {exc}"
            ) from exc
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def save_checkpoint(
    model: torch.nn.Module, path: str | Path, meta: dict[str, Any]
) -> None:

    require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "arch": "resnet18",
        "img_size": config.IMG_SIZE,
        "imagenet_mean": list(config.IMAGENET_MEAN),
        "imagenet_std": list(config.IMAGENET_STD),
        "class_names": list(config.CLASS_NAMES),
        "output": "single logit; probability of Abnormal = sigmoid(logit)",
    }
    torch.save({"state_dict": model.state_dict(), "meta": {**contract, **meta}}, path)


def load_resnet_checkpoint(
    path: str | Path = config.RESNET_CKPT_PATH,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:

    require_torch()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {path}\nCreate them with: make train-resnet "
            "(or python -m src.train_resnet)."
        )
    device = device or torch.device("cpu")
    blob = torch.load(path, map_location=device, weights_only=True)
    model = build_resnet18(pretrained=False)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, dict(blob.get("meta", {}))


def count_parameters(model: torch.nn.Module) -> int:

    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def extract_grayscale_features(source: ImageSource) -> np.ndarray:

    im = load_pil_image(source).resize(config.CLASSICAL_IMAGE_SIZE, Image.BILINEAR)
    return (np.asarray(im, dtype=np.float32) / 255.0).reshape(-1)


def extract_hog_features(source: ImageSource) -> np.ndarray:

    try:
        from skimage.feature import hog
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for HOG features. Run `make install`."
        ) from exc
    im = load_pil_image(source).resize(config.CLASSICAL_IMAGE_SIZE, Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return hog(
        arr,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


FEATURE_FUNCTIONS = {
    "grayscale": extract_grayscale_features,
    "hog": extract_hog_features,
}


def make_feature_matrix(
    paths: Iterable[str], kind: str, n_jobs: int = -1
) -> np.ndarray:

    from joblib import Parallel, delayed

    if kind not in FEATURE_FUNCTIONS:
        raise ValueError(
            f"Unknown feature kind '{kind}'. Choose from {sorted(FEATURE_FUNCTIONS)}."
        )
    fn = FEATURE_FUNCTIONS[kind]
    rows = Parallel(n_jobs=n_jobs)(delayed(fn)(p) for p in list(paths))
    return np.vstack(rows).astype(np.float32)


def build_classical_pipeline(C: float) -> Any:

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def save_bundle(model: Any, path: str | Path, metadata: dict[str, Any]) -> None:

    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)


def load_bundle(path: str | Path) -> dict[str, Any]:

    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\nCreate it with: make train-classical"
        )
    return joblib.load(path)


def iter_numpy_batches(
    X: np.ndarray, y: np.ndarray, batch_size: int = 256
) -> Iterator[tuple[np.ndarray, np.ndarray]]:

    for i in range(0, len(X), batch_size):
        yield X[i : i + batch_size], y[i : i + batch_size]


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description="Verify the MURA checkout and the patient-level split."
    )
    parser.parse_args(argv)

    print(f"MURA_ROOT  : {config.MURA_ROOT}")
    print(f"Body part  : {config.BODY_PART}")
    try:
        splits = load_splits()
    except (FileNotFoundError, AssertionError, ValueError, RuntimeError) as exc:
        print(f"\nDATA CHECK FAILED\n{exc}", file=sys.stderr)
        return 1

    summary = summarize_splits(splits)
    table = pd.DataFrame(summary).T
    print("\nSplit summary (image-level; labels are study-level):")
    print(table.to_string())
    print(
        "\nPatient-level leakage check: PASSED (0 patients shared between train / val / test)."
    )

    csv_check = compare_with_csv()
    if csv_check is None:
        print(
            "\n(Official *_image_paths.csv not found beside the data; skipped completeness check.)"
        )
    else:
        print("\nCompleteness vs official CSVs:")
        for split, counts in csv_check.items():
            flag = "OK" if counts["on_disk"] == counts["in_csv"] else "MISMATCH"
            print(
                f"  {split:5s}: on disk={counts['on_disk']:>6}  in csv={counts['in_csv']:>6}  [{flag}]"
            )
            if flag == "MISMATCH":
                print(
                    "    WARNING: your checkout differs from the official file list (partial download?). "
                    "Every reported number will describe YOUR subset, not full MURA wrist."
                )

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "mura_root": str(config.MURA_ROOT),
        "body_part": config.BODY_PART,
        "splits": summary,
        "csv_check": csv_check,
    }
    config.DATA_SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {config.DATA_SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
