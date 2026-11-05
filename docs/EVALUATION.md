# Evaluation Report — How We Decided Whether the Model Is Any Good

*A prototype. Not a diagnostic tool. Not clinically validated. Never use on real patient data.*

## The problem

Wrist X-rays queue for a clinician's read, and a model that scores studies for "looks abnormal" could reorder or pre-screen that queue. Whether such a model is *usable* is an evaluation question, not a modelling one: the model can be impressive on a benchmark and still be unsafe in a clinic.

## Why evaluation is the point

The cost of errors is lopsided. A missed abnormal study delays care; a false alarm costs reading time and, if frequent, teaches staff to ignore the tool; an over-trusted score invites people to stop looking. A single accuracy figure hides all three. So this evaluation is built to answer: *how many abnormal studies does it miss, what does catching more of them cost, can its probabilities be believed, and what does it get wrong?*

## 1. Data discipline (so the numbers mean something)

- **Patient-level split.** Images from one patient are near-duplicates. If a patient appears in both train and test, the test score measures memorisation. `mura_utils.assert_no_patient_leakage` raises an error if any patient is in more than one split. `make data-check` prints the real counts and the result of that check.
- **Three roles for three splits.** Train fits the model. Validation (386 images, held-out *patients* from the official train set) chooses regularisation (classical models), stops training (ResNet) and chooses thresholds. Test (the official MURA `valid` wrist studies: 659 images, 207 patients) is touched only for reporting.
- **One harness.** Every model goes through `eval_harness.run_eval`, so the comparison table is generated, not typed.
- **One preprocessing contract.** The evaluation transform in `config.py` (grayscale→3 channels, resize 224, tensor, ImageNet normalisation) is the one the app uses, and the app asserts it at startup.

## 2. Why each metric was chosen

Positive class = *Abnormal*. Class balance: 39.5% abnormal in train, 44.8% in test.

| Metric | Question it answers | Why it is here |
|---|---|---|
| **Recall (sensitivity), Abnormal** | Of the truly abnormal images, how many did we catch? | The headline safety number. 1 − recall is the miss rate. |
| **Precision, Abnormal** | Of the images we flagged, how many were truly abnormal? | The cost side: low precision means wasted clinician time and alert fatigue. |
| **Specificity / Normal recall** | Of normal images, how many did we leave alone? | The other half of the confusion matrix; shows what false alarms look like. |
| **F1** | Harmonic mean of precision and recall | A single summary of the pair, useful for ranking models, never for deciding safety. |
| **ROC-AUC** | Does the score rank abnormal above normal, regardless of threshold? | Threshold-free quality of the *ranking*, which is what a queue-reordering use needs. |
| **Average precision / PR curve** | Precision across all recall levels | Better than ROC when positives are the minority, and it shows the cost of raising recall. |
| **Confusion matrix** | The four outcomes in raw counts | Keeps the clinical meaning of each error visible. |
| **Brier score, ECE, reliability diagram** | Can "0.8" be read as roughly 80%? | Over-confident probabilities invite over-trust. |
| **Accuracy** | Fraction correct | Reported, never alone. |

## 3. Why accuracy alone is insufficient

Accuracy counts all errors equally and is dominated by the larger class. A model can score a respectable accuracy while missing a large share of abnormal studies, and a model at chance on a roughly balanced set shows accuracy near 0.5, which looks like "a bit of signal" until AUC (about 0.5) reveals there is none. Every metrics block here therefore carries per-class precision/recall/F1 and AUC, and accuracy is never presented alone. (An early baseline in this project that scored near chance is the reason this rule exists; see README §6.)

## 4. Operating point: choosing a threshold on purpose

The default 0.5 is arbitrary. `src/operating_point.py`:

1. chooses, **on validation only**, the highest threshold with recall ≥ **0.95**;
2. freezes it and reports test recall, precision, specificity and the fraction flagged, next to the 0.5 default;
3. reports whether the target was **actually met on test**. Selecting on validation does not guarantee it transfers, and with a small test set the recall estimate itself is noisy.

Result for the ResNet: threshold 0.064; test recall 0.963 at precision 0.536 (vs 0.536 / 0.898 at 0.5).

## 5. Calibration and abstention

`src/calibration.py` draws a reliability diagram, computes ECE (0.116 for the ResNet), and evaluates the abstention band [0.064, 0.65): **58.7%** of test images are handed to a human; among images auto-cleared as normal, **8.5%** are abnormal. Metrics on the auto-decided subset are computed with the same per-class definitions. Abstention trades coverage for reliability; it is not free, because every abstained image is human work.

## 6. Error analysis

The notebook (section A6) displays the most *confident* mistakes of each kind (false negatives and false positives) and then tests a specific hypothesis rather than telling a story: that images with a lot of near-saturated pixels (a rough proxy for metal hardware) have a different error rate than the rest. The proxy is crude (bright pixels also come from markers and exposure), so the result is reported as a hypothesis test with its limits, whichever way it comes out. A named failure mode that the data does **not** support should be reported as rejected, not quietly dropped.

## 7. What this evaluation does *not* show

- **No external validation:** same source, different patients; not different hospitals or scanners.
- **Image-level, not study-level:** a clinical read uses several views; aggregation across views was not evaluated.
- **Coarse, noisy label:** "abnormal" includes fractures, hardware and degenerative change, and each image inherits its study's label, so some "abnormal" images show nothing abnormal.
- **No reader study:** nothing here compares the model with clinicians, or measures how clinicians behave when shown a score.
- **Small-sample uncertainty:** the test set is small (659 images from 207 patients); no confidence intervals are computed, so small differences between models should not be over-read.
- **Does not measure harm:** it cannot tell us whether the tool would improve or worsen outcomes.
