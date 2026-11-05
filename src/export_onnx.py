from __future__ import annotations

import sys

import numpy as np

from src import config, mura_utils
from src.eval_harness import write_json

PARITY_TOLERANCE = 1e-4


def build_contract(extra: dict | None = None) -> dict:

    mura_utils.require_torch()
    import torch

    _, meta = mura_utils.load_resnet_checkpoint(
        config.RESNET_CKPT_PATH, torch.device("cpu")
    )
    contract = {
        "preprocessing_steps": [
            list(map(_jsonable, step))
            for step in config.describe_transform(config.get_eval_transform())
        ],
        "image_decoding": "src.mura_utils.load_pil_image (8-bit grayscale) before the transform",
        "input": {
            "shape": ["batch", 3, config.IMG_SIZE, config.IMG_SIZE],
            "dtype": "float32",
        },
        "output": {"meaning": "single logit; P(Abnormal) = sigmoid(logit)"},
        "checkpoint_meta": meta,
        "disclaimer": config.DISCLAIMER_SHORT,
    }
    contract.update(extra or {})
    write_json(config.PREPROCESSING_CONTRACT_JSON, contract)
    return contract


def export(opset: int = 17) -> dict:

    mura_utils.require_torch()
    import torch

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "ONNX export needs `onnx` and `onnxruntime`. Run: make install-onnx"
        ) from exc

    model, _ = mura_utils.load_resnet_checkpoint(
        config.RESNET_CKPT_PATH, torch.device("cpu")
    )
    dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    config.ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(config.ONNX_PATH),
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=opset,
    )

    batch = torch.randn(4, 3, config.IMG_SIZE, config.IMG_SIZE)
    with torch.no_grad():
        expected = model(batch).numpy()
    session = ort.InferenceSession(
        str(config.ONNX_PATH), providers=["CPUExecutionProvider"]
    )
    got = session.run(["logit"], {"image": batch.numpy()})[0]
    max_diff = float(np.max(np.abs(expected - got)))
    if max_diff > PARITY_TOLERANCE:
        raise RuntimeError(
            f"ONNX parity check FAILED: max |torch - onnx| = {max_diff:.2e} > {PARITY_TOLERANCE:.0e}."
        )

    contract = build_contract(
        {
            "onnx_file": str(config.ONNX_PATH),
            "onnx_input_name": "image",
            "onnx_output_name": "logit",
            "parity_max_abs_diff": max_diff,
        }
    )
    print(
        f"Exported {config.ONNX_PATH}; parity max diff {max_diff:.2e} (tolerance {PARITY_TOLERANCE:.0e})."
    )
    return contract


def _jsonable(x):
    return list(x) if isinstance(x, tuple) else x


def main(argv: list[str] | None = None) -> int:

    try:
        export()
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
