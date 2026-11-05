# Data Card — MURA v1.1, Wrist Subset (`XR_WRIST`)

*A portfolio prototype. Not a diagnostic tool. Not clinically validated. Never use on real patient data.*

## The problem

Wrist X-rays queue for a clinician's read, and a model that scores them could help order or pre-screen that queue. A model like that is only as trustworthy as the data it learned from and was tested on, so we need to be explicit about exactly what this data is and is not.

## Why the data matters more than the model

The costs of error are lopsided: a missed abnormality delays a patient's care, and a model that looks good on one hospital's images can fail silently on another's. Most of that risk lives in the data (who the patients were, which machines took the images, how labels were assigned), not in the network. This card exists so a reader can judge how far the results can be trusted.

## What the data is

- **Source:** MURA v1.1 (*musculoskeletal radiographs*), released by the Stanford ML Group. Citation: Rajpurkar et al., "MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs" (arXiv:1712.06957, 2017). This project uses only the `XR_WRIST` folders.
- **Content:** multi-view radiographs grouped into *studies* (one patient visit/exam), stored as `train/XR_WRIST/patientNNNNN/studyK_positive|negative/imageM.png`.
- **Labels:** one **study-level** label, encoded in the folder name (`positive` = abnormal, `negative` = normal). According to the dataset authors, labels come from radiologists' interpretation at the time of clinical reading at Stanford; they are not a re-adjudicated gold standard for the training split. Every image inherits its study's label.
- **Splits used here:** the official `valid` wrist split is the **test** set. Validation (386 images) is carved out of the official train split by *patient*. Train: 2,217 images from 740 patients. Test: 659 images from 207 patients.
- **Verify your copy:** `make data-check` prints the real counts, asserts patient-disjointness, and compares on-disk image counts with the official `*_image_paths.csv` files (a mismatch means a partial download and changes every number downstream).

## Class balance (computed from your checkout)

| Split | Abnormal share |
|---|---|
| Train | 39.5% |
| Test | 44.8% |

The classes are imbalanced rather than extremely skewed, but that is enough for accuracy to mislead; hence per-class metrics and AUC everywhere. Note that image-level balance differs from study-level balance because studies have different numbers of views.

## Provenance, licensing and privacy

- **Licence/terms:** MURA is distributed by Stanford for **research use under a data-use agreement** that requires registration. Do not redistribute it; this repository's `.gitignore` excludes the data. Read the current terms at the Stanford ML Group MURA page before any use beyond coursework.
- **Privacy:** the dataset is published de-identified by its authors. Do not attempt re-identification, and do not mix it with any real patient data. The demo app is for public research images only and keeps nothing it is shown.

## Known limitations and biases

- **Single site.** One hospital system, so one set of scanners, positioning habits, patient demographics and reporting style. Performance elsewhere is unknown and should be expected to be worse.
- **Demographics not documented here.** The release does not give per-image demographic labels in the files used, so subgroup performance (age, sex, ethnicity) **cannot be assessed** with this data. Absence of a measured disparity is not evidence of none.
- **Heterogeneous "abnormal".** The label bundles fractures, hardware, degenerative change, lesions and more. A model can learn proxies (for example hardware or image acquisition style) rather than "abnormality" in a clinically meaningful sense, and acute fractures and old surgical hardware are not the same clinical problem.
- **Label noise from study-level labelling.** A view in an abnormal study may show nothing abnormal; it is still labelled abnormal.
- **Label provenance.** Training labels reflect individual clinical reads, which vary between radiologists; nothing here measures that variability.
- **Age and sampling effects.** Images come from patients who were X-rayed (a selected population), not from a screening population, so prevalence in this dataset says nothing about prevalence in any other clinic.
- **Image format.** Images are PNG exports; real clinical workflows use DICOM with different bit depths and metadata. Preprocessing here would not transfer unchanged.

## Intended and out-of-scope use

Intended: coursework and a portfolio demonstration of careful evaluation. Out of scope: clinical use of any kind, use on non-wrist body parts (MURA's other body parts are deliberately excluded), and any claim about real-world prevalence or performance.
