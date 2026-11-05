# Deployment Decision Memo — Wrist X-ray Abnormality Prototype

*For: a non-ML product manager. Status: A prototype. Not a diagnostic tool. Not clinically validated. Never use on real patient data.*

## The problem

Every wrist X-ray has to be read by a clinician before the patient gets an answer. When volume is high or no radiologist is on site, studies wait in arrival order, so a fracture can sit behind many normal studies. Patients wait, clinicians lose time reading images that turn out to be normal, and the practice carries both the throughput cost and the risk of a late-caught abnormality. A model that scores each study could let the practice read the likely-abnormal ones first.

## Why it is worth solving, and why solving it badly is worse than not solving it

Done well, abnormal studies are read sooner and clinicians spend less time on routine ones. Done badly, the damage is quiet: a **missed abnormal study** (false negative) delays care for a patient; **too many false alarms** train staff to ignore the flag; and a confident-looking percentage makes it easy for a busy person to **stop looking**. The cost of these errors is lopsided, so one accuracy number cannot tell you whether this is safe to use. The questions below are the ones that can.

## Recommendation

**Do not deploy this on real patients.** It is a prototype trained on one hospital's public research images with no outside validation. What follows is how it *would* have to be run, and what would have to be true first. If you want a simpler first step, a rule-based queue (oldest first, plus clinician-flagged urgent) needs no model and no validation, and is the right baseline to beat.

## The operating point (the one decision that is yours, not the data scientist's)

The model produces a score from 0 to 1. Where you draw the line decides which mistake you accept.

| | Default line (0.5) | Safety-first line (recall ≥ {{OP_RECALL_TARGET}}) |
|---|---|---|
| Line position | 0.500 | {{OP_THRESHOLD}} (chosen on validation data, then frozen) |
| Abnormal studies caught (recall, test) | {{OP_TEST_RECALL_AT_05}} | {{OP_TEST_RECALL}} |
| Of flagged studies, truly abnormal (precision, test) | {{OP_TEST_PRECISION_AT_05}} | {{OP_TEST_PRECISION}} |
| Share of all studies flagged (test) | see `outputs/operating_point_resnet.json` | {{OP_TEST_FLAGGED_FRACTION}} |

Reading it: the safety-first line catches more abnormal studies and pays for it in false alarms (lower precision). **Your question:** how many extra normal studies is the practice willing to review to catch one more abnormal one? Clinical leadership, not the model, answers that.

## The abstention band (when the system must hand off to a human)

| Score | System says |
|---|---|
| below {{ABSTAIN_LOW}} | "Leans normal": a candidate for fast sign-off, never an automatic discharge |
| {{ABSTAIN_LOW}} to {{ABSTAIN_HIGH}} | "Inconclusive": a person must read it |
| {{ABSTAIN_HIGH}} or above | "Leans abnormal": read first |

On the test data the band sent **{{ABSTAIN_FRACTION}}** of images to a human. Of the images it would have cleared as normal, **{{AUTO_CLEAR_MISS_RATE}}** were actually abnormal. That last number is the one to bring to your clinical lead: it is the miss rate if anyone trusts "leans normal". The probability is also not guaranteed to be well calibrated (calibration error {{ECE}}); a "90%" is a score, not a 90% chance.

## Monitoring plan (if it were ever piloted, with consent and oversight)

1. **Weekly:** share of studies in each band. A shift in how many land in "inconclusive" is an early sign that the images have changed (new machine, new protocol).
2. **Every study:** the clinician's final read is logged next to the model score, so recall and precision can be recomputed on real local data.
3. **Monthly:** every missed abnormal that the model scored "leans normal" is reviewed by a clinician; the pattern (for example, post-surgical hardware, unusual positioning) feeds back into the "do not use" list.
4. **Clinician override rate:** if staff override the flag most of the time, or never, the tool is not helping or is being over-trusted. Both are stop signals.
5. **Fixed review of thresholds** after any change to imaging equipment.

## Do not ship if

- It has not been validated on **your own site's images**, by a clinician, with the results written down.
- No regulatory or compliance review has covered it as software that influences clinical care.
- The recall at the agreed operating point, measured **on your local data**, is below what clinical leadership signed off.
- Any workflow lets "leans normal" **remove a human read** instead of only reordering the queue.
- Staff would see only a percentage and not the "inconclusive" state or the not-a-diagnosis banner.
- Real patient images would be uploaded to anything other than a system your privacy officer has approved. (The demo app is for public research images only.)
- There is no owner for the monitoring plan above.
- The images are not wrist radiographs. The model cannot tell, and will still output a confident number.

## When a simpler tool is the better call

Compare the table in the README. If a HOG + logistic-regression baseline gets close to the ResNet at the same recall target (HOG: recall {{HOG_OP_TEST_RECALL}}, precision {{HOG_OP_TEST_PRECISION}}), prefer it: it is faster, cheaper, and its behaviour can be inspected. If neither beats "oldest first, plus a clinician's urgent flag" by enough to justify the validation and monitoring burden, the right answer is not to use ML at all.
