from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Wrist X-ray prototype (NOT a diagnostic tool)", layout="wide"
)

PALETTE = ["#B2EBF2", "#80DEEA", "#4DD0E1", "#26C6DA", "#00BCD4"]
st.markdown(
    f"""
    <style>
    .stApp {{ background: linear-gradient(180deg, rgba(178,235,242,0.35) 0%, #FFFFFF 260px); }}
    [data-testid="stHeader"] {{ background: linear-gradient(90deg, {", ".join(PALETTE)}); }}
    [data-testid="stSidebar"] {{ background: {PALETTE[0]}; }}
    [data-testid="stMetricValue"] {{ color: #006064; }}
    [data-testid="stFileUploader"] section {{ border: 2px dashed {PALETTE[3]}; background: rgba(255,255,255,0.7); }}
    .stProgress > div > div > div > div {{ background-color: {PALETTE[4]}; }}
    h1, h2, h3 {{ color: #006064; }}
    </style>
    """,
    unsafe_allow_html=True,
)


st.error(
    "## NOT A DIAGNOSTIC TOOL\nCoursework / portfolio prototype. Not clinically validated. Never use on real patient data."
)

try:
    from src import config
except (ImportError, ModuleNotFoundError, OSError) as exc:
    st.error(
        f"Could not load project configuration (`src/config.py`): {exc}\n\nRun the app from the repository root."
    )
    st.stop()


EXPECTED_PREPROCESSING = [
    ("Grayscale", 3),
    ("Resize", (224, 224)),
    ("ToTensor",),
    ("Normalize", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]


def assert_preprocessing_parity() -> None:

    derived = config.describe_transform(config.get_eval_transform())
    if derived != EXPECTED_PREPROCESSING:
        raise AssertionError(
            "Preprocessing mismatch between config.get_eval_transform() and the contract this app expects.\n"
            f"  derived : {derived}\n  expected: {EXPECTED_PREPROCESSING}\n"
            "Predictions would silently differ from the evaluated model. Refusing to run."
        )


try:
    import torch
    from PIL import Image, UnidentifiedImageError

    from src import mura_utils
except ImportError as exc:
    st.error(
        f"A required package is missing: {exc}\n\nFix: `make install` (or `pip install -r requirements.txt`)."
    )
    st.stop()

try:
    assert_preprocessing_parity()
except (AssertionError, ImportError) as exc:
    st.error(f"Startup check failed:\n\n{exc}")
    st.stop()


@st.cache_resource(show_spinner="Loading model weights...")
def load_model():
    return mura_utils.load_resnet_checkpoint(
        config.RESNET_CKPT_PATH, torch.device("cpu")
    )


try:
    model, meta = load_model()
except FileNotFoundError as exc:
    st.error(f"Model weights are missing.\n\n{exc}")
    st.stop()
except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
    st.error(
        f"Could not load the model weights ({type(exc).__name__}): {exc}\n\nRe-create them with `make train-resnet`."
    )
    st.stop()


recall_threshold = config.load_operating_threshold("resnet")
LOW, HIGH, BAND_NOTE = config.resolve_abstention_band(recall_threshold)

with st.sidebar:
    st.header("About this prototype")
    st.write(config.DISCLAIMER_LONG)
    st.markdown(
        "- Trained on **wrist** radiographs only (MURA, single hospital system).\n"
        "- Label is coarse: *abnormal vs normal*, not a diagnosis.\n"
        "- Images are processed in memory and not stored."
    )
    st.subheader("Decision bands")
    st.write(f"Leans normal: P(abnormal) < **{LOW:.3f}**")
    st.write(f"Needs human review: **{LOW:.3f} - {HIGH:.2f}**")
    st.write(f"Leans abnormal: P(abnormal) >= **{HIGH:.2f}**")
    st.caption(BAND_NOTE)
    if recall_threshold is None:
        st.warning(
            "No validation-derived threshold found; using a fallback band. Run `make operating-point`."
        )
    st.subheader("Model")
    st.json(
        {
            k: meta.get(k)
            for k in ("arch", "img_size", "best_epoch", "val_loss", "torch_version")
        }
    )

MAX_BYTES = 25 * 1024 * 1024
upload = st.file_uploader(
    "Upload one wrist X-ray (PNG or JPG)",
    type=["png", "jpg", "jpeg"],
    help="Public research images only. Never real patient data.",
)

if upload is None:
    st.info(
        "Upload an image to see the model's probability and the abstention verdict."
    )
    st.stop()

if upload.size > MAX_BYTES:
    st.error("File is larger than 25 MB. Please upload a smaller image.")
    st.stop()

try:
    from io import BytesIO

    raw = upload.getvalue()
    Image.open(BytesIO(raw)).verify()
    image = mura_utils.load_pil_image(BytesIO(raw))
except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
    st.error(
        f"This file could not be read as an image ({type(exc).__name__}). Please upload a valid PNG or JPG radiograph."
    )
    st.stop()

try:
    tensor = config.get_eval_transform()(image).unsqueeze(0)
    with torch.no_grad():
        p_abnormal = float(torch.sigmoid(model(tensor)).item())
except Exception as exc:  # noqa: BLE001
    st.error(f"Prediction failed ({type(exc).__name__}): {exc}")
    st.stop()

if p_abnormal < LOW:
    verdict, level = "Leans NORMAL (auto-clear candidate)", st.success
elif p_abnormal >= HIGH:
    verdict, level = "Leans ABNORMAL (flag for review)", st.warning
else:
    verdict, level = "INCONCLUSIVE - a human must review this image", st.info

left, right = st.columns(2)
with left:
    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)
with right:
    st.subheader("Model output")
    st.metric("P(abnormal)", f"{p_abnormal:.1%}")
    level(f"**{verdict}**")
    st.progress(min(max(p_abnormal, 0.0), 1.0))
    st.caption(
        f"Bands: normal < {LOW:.3f} <= review < {HIGH:.2f} <= abnormal. "
        "The probability is a model score, not a diagnosis, and it is not guaranteed to be well calibrated "
        "(see `docs/EVALUATION.md`)."
    )
    st.warning(
        "This model only knows wrist radiographs. For any other body part, or an unusual view, "
        "the number above is meaningless."
    )
st.error(
    "NOT A DIAGNOSTIC TOOL. Do not use this output for diagnosis, triage, or treatment decisions."
)
