set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
export MURA_ROOT="${MURA_ROOT:-$(pwd)/MURA-v1.1}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31mFAILED: %s\033[0m\n' "$1" >&2; exit 1; }

step "0/7 Environment check"
"$PYTHON" -c "import torch, torchvision, sklearn, skimage; print('torch', torch.__version__)" \
  || fail "dependencies missing. Run: make install"

step "1/7 Data check (real counts + patient-leakage assertion)"
"$PYTHON" -m src.mura_utils || fail "data check (is MURA_ROOT=$MURA_ROOT correct?)"

step "2/7 Train classical baselines"
"$PYTHON" -m src.train_classical || fail "classical training"

step "3/7 Train ResNet18"
"$PYTHON" -m src.train_resnet || fail "ResNet training"

step "4/7 Compare models"
"$PYTHON" -m src.compare_models || fail "model comparison"

step "5/7 Operating point + calibration/abstention (all models)"
for m in resnet hog grayscale; do
  "$PYTHON" -m src.operating_point --model "$m" || fail "operating point ($m)"
  "$PYTHON" -m src.calibration --model "$m" || fail "calibration ($m)"
done

step "6/7 Copy artifacts"
mkdir -p outputs/artifacts
for f in models/mura_grayscale_logistic.joblib models/mura_hog_logistic.joblib models/resnet18_wrist.pt; do
  [ -f "$f" ] || fail "expected artifact missing: $f"
  cp "$f" outputs/artifacts/
done
(cd outputs/artifacts && { sha256sum ./* 2>/dev/null || shasum -a 256 ./*; } > SHA256SUMS.txt)
for f in outputs/model_comparison.csv outputs/model_comparison.md outputs/resnet_metrics.json outputs/classical_metrics.json; do
  [ -f "$f" ] || fail "expected output missing: $f"
done

step "7/7 Fill README/docs placeholders from real outputs"
"$PYTHON" -m src.fill_placeholders || fail "placeholder fill (see message above)"

printf '\n\033[32mAll steps passed.\033[0m Next: make notebook   (optional: make install-onnx export-onnx)   then: make app\n'
