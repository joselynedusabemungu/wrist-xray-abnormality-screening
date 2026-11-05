# Model Card — Wrist X-ray Abnormality Classifier (ResNet18)

*A prototype. Not a diagnostic tool. Not clinically validated. Never use on real patient data.*

## The problem

Wrist X-rays wait in a queue for a clinician's read. Patients wait for answers, clinicians spend time on studies that turn out normal, and a practice carries the delay and the risk of a late-caught abnormality. A model that scores each study could help read likely-abnormal studies sooner.

## Why it is worth getting right (and what happens if it is wrong)

A missed abnormal study (false negative) delays care; false alarms waste clinician time and breed alert fatigue; an over-trusted score invites people to stop looking. So this card focuses on what the model was trained on, how it was measured, where it fails, and what it must **not** be used for.

## Model details

- **Task:** binary classification of a single wrist radiograph, *Normal* vs *Abnormal* (positive class = Abnormal).
- **Architecture:** ResNet18 pretrained on ImageNet, final layer replaced by one logit; **all** layers fine-tuned. Output probability = sigmoid(logit). Roughly 11.2 million parameters (exact count in `outputs/resnet_metrics.json`).
- **Why this model:** a small residual network is a standard, well-understood choice for limited medical data; ImageNet pretraining gives useful low-level features; ResNet50 adds capacity and cost without evidence that capacity is the bottleneck; training from scratch on a few thousand images is unreliable. It is compared with two simple, interpretable baselines (grayscale+logistic, HOG+logistic) so its complexity has to be justified by the numbers.
- **Input contract (must be reproduced exactly):** grayscale → 3 channels, resize to 224×224, `ToTensor`, ImageNet normalisation (mean 0.485/0.456/0.406, std 0.229/0.224/0.225). Defined once in `src/config.py` and asserted at app startup.
- **Training:** Adam, learning rate 1e-4, weight decay 1e-4, batch 32, up to 15 epochs, early stopping on validation loss (patience 3), best checkpoint kept. Augmentation (training only): horizontal flip, ±7° rotation, ±10% brightness/contrast.
- **Files:** `models/resnet18_wrist.pt` (weights plus the preprocessing contract), optional `models/resnet18_wrist.onnx`.

## Training data

MURA v1.1 `XR_WRIST`, patient-level split; train {{N_TRAIN_IMAGES}} images ({{TRAIN_PCT_ABNORMAL}} abnormal) from {{N_TRAIN_PATIENTS}} patients; validation {{N_VAL_IMAGES}} images. See [`DATA_CARD.md`](DATA_CARD.md) for provenance, licence and biases, notably: single hospital, study-level labels, heterogeneous "abnormal", no demographic subgroup data.

## Performance (test set: official MURA `valid` wrist, {{N_TEST_IMAGES}} images, {{TEST_PCT_ABNORMAL}} abnormal)

| Metric (threshold 0.5, positive = Abnormal) | ResNet18 | HOG + LR | Grayscale + LR |
|---|---:|---:|---:|
| Accuracy | {{RESNET_ACCURACY}} | {{HOG_ACCURACY}} | {{GRAYSCALE_ACCURACY}} |
| Precision | {{RESNET_PRECISION}} | {{HOG_PRECISION}} | {{GRAYSCALE_PRECISION}} |
| Recall | {{RESNET_RECALL}} | {{HOG_RECALL}} | {{GRAYSCALE_RECALL}} |
| F1 | {{RESNET_F1}} | {{HOG_F1}} | {{GRAYSCALE_F1}} |
| ROC-AUC | {{RESNET_AUC}} | {{HOG_AUC}} | {{GRAYSCALE_AUC}} |

Recall-target operating point (chosen on validation): threshold {{OP_THRESHOLD}} gives test recall {{OP_TEST_RECALL}} at precision {{OP_TEST_PRECISION}}. Calibration error (ECE): {{ECE}}. Abstention band [{{ABSTAIN_LOW}}, {{ABSTAIN_HIGH}}) abstains on {{ABSTAIN_FRACTION}} of test images; {{AUTO_CLEAR_MISS_RATE}} of auto-cleared images are abnormal. Methodology and caveats: [`EVALUATION.md`](EVALUATION.md).

## Intended use

- Learning and demonstration of a careful, reproducible evaluation workflow for a medical-imaging classifier.
- A reference for discussing operating points, calibration and abstention with non-ML stakeholders.

## Out-of-scope uses (do not)

- Any clinical use: diagnosis, triage, screening, second reads, or treatment decisions.
- Use on real patient data, or on non-public images.
- Any body part other than the wrist, any modality other than plain radiographs, paediatric or post-operative populations (not characterised here), or images from another institution without fresh validation.
- Automatic clearing of studies without a human read.

## Ethical considerations and risks

- **Automation bias:** a precise-looking probability can reduce the care a human takes. The app shows an "inconclusive" state and an unmissable not-a-diagnosis banner for this reason.
- **Asymmetric harm:** a false negative can delay care; threshold choice is therefore a clinical and organisational decision, documented in `DECISION.md`.
- **Hidden subgroup failures:** no demographic metadata means unequal performance across groups cannot be ruled out.
- **Shortcut learning:** the model may rely on acquisition artefacts, text markers or hardware rather than pathology; error analysis in the notebook tests one such hypothesis but cannot rule out others.
- **Distribution shift:** different scanners, positioning or populations can degrade performance without any visible warning.
- **Privacy:** the app runs offline and stores nothing, but it must still never be pointed at real patient images.
- **Misleading polish:** a working demo can look like a product. It is not one.

## Caveats and recommendations

Single-site, no external validation, no regulatory pathway, coarse label, image-level only, small test set with no confidence intervals. Before any further use: validate locally with clinicians, quantify uncertainty, study-level aggregation, subgroup analysis, and a prospective monitoring plan.
