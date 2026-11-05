# Model comparison (test set)

| Model | Accuracy | Precision (Abn.) | Recall (Abn.) | F1 (Abn.) | AUC | Train time (s) | Inference (ms/img) | Interpretable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Grayscale pixels + logistic regression | 0.548 | 0.495 | 0.498 | 0.497 | 0.570 | 122 | 6.6 | Yes (one coefficient per pixel) |
| HOG features + logistic regression | 0.683 | 0.657 | 0.610 | 0.633 | 0.710 | 9 | 4.9 | Partly (one coefficient per HOG cell/orientation) |
| ResNet18 (ImageNet transfer learning) | 0.765 | 0.898 | 0.536 | 0.671 | 0.847 | 536 | 22.3 | No (post-hoc saliency only) |

Test set = official MURA `valid` wrist studies (never used for fitting or model selection). Threshold 0.5; positive class = Abnormal. Train time includes feature extraction (classical) or all epochs (ResNet). Inference = mean single-image latency, file read to probability, on this machine.

*Coursework / portfolio prototype. NOT a diagnostic tool.*
