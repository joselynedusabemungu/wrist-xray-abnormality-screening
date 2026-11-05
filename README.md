# Wrist X-ray Abnormality Screening — Portfolio Prototype

*A prototype. Not a diagnostic tool. Not clinically validated. Never use on real patient data.*

## 1. The Problem

A wrist radiograph is one of the most routine imaging orders in urgent care, orthopaedic and primary-care settings. Before the patient hears anything definitive, a trained clinician has to look at every study. In a busy clinic, or one without a radiologist on site, those studies wait in a first-in, first-out queue. A displaced fracture can sit behind a dozen normal studies, because nobody can tell which is which until someone opens it.

Three groups feel this. **Patients** wait, often in pain or in a splint, for an answer. **Clinicians** burn time on studies that turn out to be normal and have to interrupt treating patients to read images. **Practice managers** carry the throughput problem and the liability: slow turnaround hurts the schedule, and a missed or late-caught abnormality is the expensive failure.

The reason machine learning is even on the table is narrow. If a model could score each study for "looks abnormal", the queue could be *reordered* (likely-abnormal first), and in a stricter use, clearly-normal studies could be fast-tracked for a quick human sign-off. I do not have throughput or cost numbers for any real practice, and I have not invented any. This project asks a smaller, testable question: **can a model score wrist studies well enough to be worth that, and how would we know when it is not?**

## 2. Why This Is Worth Solving (and Why Solving It Badly Is Worse Than Not Solving It)

**If it works.** Faster time-to-read for the minority of studies that are truly abnormal, and less clinician time spent on the rest.

**If it works badly, the failure is quiet.** The two errors are not symmetric:

- A **false negative** (an abnormal study the model calls normal) is the dangerous one. If the system lets a clinician skip or deprioritise that study, a fracture is missed or caught late. The harm lands on a patient, not on a benchmark.
- A **false positive** (a normal study flagged) costs clinician minutes and, if frequent, trains staff to ignore the flag (alert fatigue), which quietly turns the tool into noise.
- An **over-trusted model** is worse than both. A confident-looking probability makes it easy for a tired human to stop looking. A model that is accurate on one hospital's X-ray machines can be wrong in a way nobody notices on another's.

That is why this repository reports recall on the abnormal class, precision cost, calibration, and an *abstention band* (cases the model must hand to a human) rather than a single accuracy number, and why the deployment memo in [`docs/DECISION.md`](docs/DECISION.md) lists conditions under which it should **not** ship.

*Why this is relevant to practice-management software:* the same shape (a human-review queue, lopsided error costs, sensitive health data, and a temptation to over-trust the score) appears wherever software ranks or pre-screens clinical work. This is an analogy, not a claim that this model applies to any other product.

## 3. What This Project Does

This is an end-to-end, reproducible prototype that classifies single **wrist** radiographs from the public MURA v1.1 dataset as *normal* or *abnormal*. It trains and compares three models (grayscale pixels + logistic regression, HOG features + logistic regression, and a fine-tuned ResNet18) using one shared evaluation harness, on a **patient-level** split with an assertion that no patient appears in more than one split. It reports precision, recall, F1 and AUC (never accuracy alone), selects a recall-target threshold on validation data, measures calibration, defines an abstention band, and ships an offline Streamlit demo that refuses to present itself as a diagnostic tool. It is **not** a clinical product: it makes no diagnosis-specific claim, has no external validation and no regulatory pathway, and must not be used on real patient data.

## 4. Results

> Numbers below are **filled from real outputs**, never typed by hand. Source: `outputs/model_comparison.csv`, produced by `make compare` and inserted by `make fill-docs`. Test set = official MURA `valid` wrist studies (659 images from 207 patients, 44.8% abnormal); threshold 0.5; positive class = Abnormal.

| Model | Accuracy | Precision (Abn.) | Recall (Abn.) | F1 (Abn.) | AUC | Train time (s) | Inference (ms/img) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Grayscale pixels + logistic regression | 0.548 | 0.495 | 0.498 | 0.497 | 0.570 | 122 | 6.6 |
| HOG features + logistic regression | 0.683 | 0.657 | 0.610 | 0.633 | 0.710 | 9 | 4.9 |
| ResNet18 (ImageNet transfer learning) | 0.765 | 0.898 | 0.536 | 0.671 | 0.847 | 536 | 22.3 |

Training set: 2,217 images from 740 patients (39.5% abnormal); validation: 386 images. Latency is end-to-end (file read to probability) on the machine that ran the pipeline.

## 5. The Tradeoff That Matters

The default 0.5 threshold is not a clinical decision, it is an arbitrary number. For the ResNet at 0.5, test recall on abnormal images is **0.536** at precision **0.898**.

If a missed abnormality is the expensive error, we instead choose, **on the validation split only**, the highest threshold that reaches recall ≥ **0.95**. That threshold is **0.064**. Frozen and applied to the test split it gives recall **0.963** and precision **0.536**, and it flags **80.4%** of all images. The drop in precision between those two operating points is the price of safety: it is extra clinician review time spent on normal studies. Whether that price is acceptable is a product and clinical decision, not an ML one.

The simple baseline gets the same treatment. At the same recall target, HOG + logistic regression reaches recall **0.868** at precision **0.507** on test. If that is close to the ResNet, a simpler, cheaper, more interpretable model is the better engineering choice; the deep network has to earn its complexity with numbers.

The model's uncertain middle is handed to a human instead of being decided. With the band [0.064, 0.65), **58.7%** of test images are abstained on, and **8.5%** of the images the system would auto-clear are in fact abnormal (calibration error: ECE 0.116). Details: [`docs/EVALUATION.md`](docs/EVALUATION.md), [`docs/DECISION.md`](docs/DECISION.md).

## 6. What I Got Wrong / What I'd Do Differently

- **My "HOG" model was not HOG.** In the first version of this repo, `extract_hog` was an alias for raw resized pixels, so the "HOG baseline" was a second copy of the grayscale baseline. I only found it by auditing the code against the model's name. It now uses `skimage.feature.hog`, and I treat a model's *name* as a claim that needs a test. Next time I write a unit test for every feature extractor before training on it.
- **A draft that looked finished but had never run.** My earlier notebook contained "fill in" placeholders and `ModuleNotFoundError` outputs, and the write-ups described results I had not produced. I now generate every table from code (`run_eval` → JSON → CSV → docs), the notebook scans its own source and outputs for leftover tokens, and a placeholder can never silently pass as a result.
- **A chance-level baseline is information, not a verdict.** An early `outputs/metrics.json` from my first grayscale model showed accuracy 0.50 and ROC-AUC about 0.51, i.e. chance. The tempting reading is "wrist X-rays are hard"; the right one is "that representation carries almost no signal", which motivated proper HOG features and transfer learning. What I would do differently: always report per-class metrics and AUC beside accuracy from the first run, and test a model on a deliberately easy synthetic task before blaming the data.

## 7. Reproduce

Requires Python 3.10–3.12. A GPU is optional (CPU works but ResNet training is slow; use the smoke test first).

```bash

make install          
make data-check       
make smoke           
make train-all        
make compare         
make operating-point  
make calibration      
make fill-docs        
make notebook        
make app              

bash scripts/run_all.sh
```

Optional: `make install-onnx && make export-onnx` for an ONNX export with a numerical parity check.

Determinism: seeds come from `config.RANDOM_SEED`. Bitwise-identical results are **not** guaranteed on CUDA. For CPU-deterministic runs: `CUDA_VISIBLE_DEVICES="" MURA_NUM_WORKERS=0 python -m src.train_resnet --cpu-deterministic`.

## 8. Limitations

- **Single-site data.** MURA comes from one hospital system. Scanners, protocols, patient mix and labelling habits will differ elsewhere; performance should be expected to drop.
- **No external validation.** The test split is from the same source as training (patient-disjoint, but not site-disjoint).
- **No regulatory pathway.** This is not a medical device and was never assessed as one.
- **Coarse label.** "Abnormal vs normal" lumps fractures, hardware, degenerative change and more together. It is not diagnosis-specific, and labels are per study (every image inherits its study's label).
- **Image-level, not study-level.** Predictions are per image; a real workflow reads a study of several views.
- **Not for patient data.** The model must not be used on real patient data, and the app is for public research images only.

## Repository layout

```
src/config.py          paths, hyperparameters, thresholds, the shared eval transform
src/mura_utils.py      scanning, patient-level split + leakage assertion, datasets, features, ResNet helpers
src/eval_harness.py    run_eval(): the single source of every metric
src/train_classical.py grayscale+LR and HOG+LR      src/train_resnet.py   ResNet18 fine-tuning
src/compare_models.py  comparison table             src/export_onnx.py    optional ONNX export
src/operating_point.py recall-target threshold      src/calibration.py    reliability + abstention
src/fill_placeholders.py  fills README/docs from real outputs
notebooks/MURA_Wrist_Classification.ipynb   narrative over the library (A1–A8)
app.py                 offline Streamlit demo
docs/                  DECISION.md, EVALUATION.md, DATA_CARD.md, MODEL_CARD.md
```
