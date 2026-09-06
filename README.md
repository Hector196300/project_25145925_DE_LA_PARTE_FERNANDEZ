# COMP0248 CW1 — Multitask Hand Gesture Detection, Segmentation and Classification

Student number 25145925 (de la Parte Fernández).

A single convolutional network that, from one RGB frame, predicts:

- a **binary hand mask** (U-Net style decoder),
- a **bounding box** (regressed, and also derived from the mask),
- a **gesture class** out of 10, with a confidence.

4.98 M parameters, trained from scratch. No pretrained weights, no detection
framework, no smartphone images in training.

---

## 1. Requirements

```bash
python -m pip install -r requirements.txt
```

Python 3.10+. A CUDA GPU is strongly recommended; every script falls back to CPU
with `--cpu`.

`torchvision` is not listed as a dependency because nothing in `src/` imports
it; it is harmless if already present, since it ships alongside CUDA PyTorch. No
pretrained backbone and no high-level detection or segmentation framework is
used anywhere in the model, training loop or evaluation scripts.

**What is written here, and what comes from PyTorch.** The architecture
(`ConvBlock`, `UpBlock`, `HandGestureNet`) subclasses `torch.nn.Module`;
`GestureDataset` subclasses `torch.utils.data.Dataset` and implements its own
indexing, contributor-level splitting and augmentation; the training loop, the
loss composition and every metric are written out in full. Used as provided:
`nn.Conv2d`, `nn.BatchNorm2d`, `nn.ConvTranspose2d`, `nn.BCEWithLogitsLoss`,
`nn.CrossEntropyLoss`, `torch.optim.AdamW` and `CosineAnnealingLR`. Implemented
directly because they are not standard `nn` losses or metrics: soft Dice, IoU,
GIoU, the confusion matrix and macro-averaged F1.

---

## 2. Expected directory layout

Run every command from the project root. The scripts resolve paths relative to
it.

```
.
├── src/
│   ├── dataloader.py        dataset indexing, splitting, augmentation
│   ├── model.py             architecture + multitask loss
│   ├── train.py             training loop
│   ├── evaluate.py          all metrics, writes results/*_metrics.json
│   ├── visualise.py         figures for the report
│   └── utils.py             shared pure helpers (boxes, metrics, IO)
├── rgb_only/                pooled training corpus, one folder per contributor
│   └── <studentno>_<surname>/G01_call/clip01/{rgb,annotation}/frame_XXX.png
├── COMP0248_Test_data_23/   held-out RealSense test set, same inner layout
├── smartphone_dataset/
│   └── 25145925_DE_LA_PARTE_FERNANDEZ/G01_call/clip01/{rgb,annotation}/...
├── weights/                 created by train.py
└── results/                 created by evaluate.py and visualise.py
    └── figs/
```

Only frames that have a mask in `annotation/` are indexed — a frame without one
can supply neither a segmentation target nor a box. `find_gesture_root()`
tolerates contributors who nested their `Gxx_name/` folders one level deeper,
and gesture folder names are matched case-insensitively.

**Excluded folders** (`EXCLUDE_STUDENTS` in `dataloader.py`):

| Folder | Reason |
|---|---|
| `COMP0248_Manuel de Castro Ribeiro Jardim` | no masks at all, nothing to learn from |
| `25150455_Guan 2` | byte-duplicate of `25150455_Guan`; keeping both double-weights one subject |

**Splitting is by contributor, not by image.** Frames inside a 5 s / 3 fps clip
are near-duplicates, so an image-level split would leak near-copies into
validation. `--val-fraction 0.2` with `--seed 0` yields 23 training folders
(2300 annotated frames) and 6 validation folders (599 annotated frames) out of
the 29 usable.

Sanity-check the layout before training:

```bash
python src/dataloader.py rgb_only
```

This prints the usable folders, the train/val split, per-class counts and the
shapes of one sample.

---

## 3. Reproducing the experiments

Three settings, two trained models.

### Experiment 1 — baseline, generic augmentation

```bash
python src/train.py --name baseline --epochs 40 --num-workers 4
```

Writes `weights/baseline_best.pt`, `weights/baseline_last.pt` and
`results/baseline_history.json`. Best checkpoint was epoch 35/40.

Evaluate in domain:

```bash
python src/evaluate.py --weights weights/baseline_best.pt \
    --val-split --name exp1_val

python src/evaluate.py --weights weights/baseline_best.pt \
    --dataset COMP0248_Test_data_23 --name exp1_realsense_test
```

### Experiment 2 — zero-shot transfer to the smartphone set

Same weights, no fine-tuning, no retraining:

```bash
python src/evaluate.py --weights weights/baseline_best.pt \
    --dataset smartphone_dataset/25145925_DE_LA_PARTE_FERNANDEZ \
    --name exp2_phone_zeroshot
```

### Experiment 3 — domain-targeted augmentation

```bash
python src/train.py --name domain_aug --epochs 40 --domain-aug --num-workers 4
```

Best checkpoint was epoch 31/40. Evaluate on all three sets:

```bash
python src/evaluate.py --weights weights/domain_aug_best.pt \
    --val-split --name exp3_val

python src/evaluate.py --weights weights/domain_aug_best.pt \
    --dataset COMP0248_Test_data_23 --name exp3_realsense_test

python src/evaluate.py --weights weights/domain_aug_best.pt \
    --dataset smartphone_dataset/25145925_DE_LA_PARTE_FERNANDEZ \
    --name exp3_phone
```

`--domain-aug` does **not** mean "more augmentation". It applies the same
families of transform with ranges chosen to span the measured
RealSense-to-smartphone gap: hand area 3.19× larger, saturation 1.70× higher,
red channel 1.33× higher. Scale becomes log-uniform 0.7–2.2 about the *hand
centroid* (then re-centred, so large zooms don't push the hand out of frame),
saturation shifts to 0.8–2.0 and red gain to 1.00–1.35. No smartphone image is
used in training — only the summary statistics of the gap.

Note that `--augment-strength` (which scales the *generic* policy) is a separate
knob and was left at 1.0 for every run reported. The docstring at the top of
`train.py` still describes an earlier plan that used `--augment-strength 1.8`
for Experiment 3; the results in the report come from `--domain-aug` instead.

### Comparing two runs

```bash
python src/evaluate.py --compare results/exp2_phone_zeroshot_metrics.json \
                                 results/exp3_phone_metrics.json
```

---

## 4. Figures

All figures are saved, never shown, so this works headless over SSH. Pass
`--compact` for anything going into the two-column paper.

```bash
mkdir -p results/figs

# training curves
python src/visualise.py curves --history results/baseline_history.json \
    --out results/figs/baseline_curves.png --compact
python src/visualise.py curves --history results/domain_aug_history.json \
    --out results/figs/domain_aug_curves.png --compact

# confusion matrices
python src/visualise.py confusion --metrics results/exp1_realsense_test_metrics.json \
    --out results/figs/exp1_cm.png
python src/visualise.py confusion --metrics results/exp2_phone_zeroshot_metrics.json \
    --out results/figs/exp2_cm.png --compact
python src/visualise.py confusion --metrics results/exp3_phone_metrics.json \
    --out results/figs/exp3_cm.png --compact

# qualitative overlays
python src/visualise.py overlays --weights weights/baseline_best.pt \
    --dataset smartphone_dataset/25145925_DE_LA_PARTE_FERNANDEZ --n 4 \
    --out results/figs/exp2_overlays.png --compact
python src/visualise.py overlays --weights weights/domain_aug_best.pt \
    --dataset smartphone_dataset/25145925_DE_LA_PARTE_FERNANDEZ --n 4 \
    --out results/figs/exp3_overlays.png --compact

# side-by-side bar chart across experiments
python src/visualise.py compare --metrics \
    results/exp1_realsense_test_metrics.json \
    results/exp2_phone_zeroshot_metrics.json \
    results/exp3_phone_metrics.json \
    --out results/figs/comparison.png --compact
```

In the overlay figures the **solid red** box is the regressed prediction, the
**dashed orange** box is derived from the predicted mask, and the two IoU values
in each title are quoted in that order. Green is ground truth.

---

## 5. Outputs written by `evaluate.py`

For each `--name N`:

- `results/N_metrics.json` — every metric, per-class precision/recall/F1, the
  confusion matrix, and a `meta` block recording which weights and dataset
  produced it.
- `results/N_confusion.csv` — the same matrix, rows true, columns predicted.
- `results/N_samples.json` — per-image records (both boxes, both IoUs,
  predicted label, confidence, source paths) for the first `--save-samples`
  images, used to build qualitative figures and to inspect failures.

### Reading the `detection` block

`detection` is **not** a single fixed quantity. `evaluate.py` computes detection
metrics twice — once from the regressed box and once from the tight box around
the predicted mask — and copies whichever scores higher into `detection`,
recording which one in `detection["source"]`. Both are always preserved in full
under `detection_regressed` and `detection_from_mask`.

In all six evaluations reported, `source` was `mask_derived`. The report states
this explicitly and tabulates both sources rather than quoting only the better
number. If you want a fixed source instead of the automatic choice, read
`detection_regressed` / `detection_from_mask` directly.

---

## 6. Headline results

| | E1 val | E1 RS test | E2 phone | E3 val | E3 RS test | E3 phone |
|---|---|---|---|---|---|---|
| frames | 599 | 3450 | 60 | 599 | 3450 | 60 |
| det acc @0.5 (mask) | 0.755 | 0.875 | 0.900 | 0.810 | 0.786 | 0.900 |
| det acc @0.5 (regressed) | 0.486 | 0.454 | 0.567 | 0.167 | 0.125 | 0.533 |
| mask mean IoU | 0.796 | 0.847 | 0.651 | 0.786 | 0.784 | 0.822 |
| Dice | 0.853 | 0.889 | 0.760 | 0.849 | 0.841 | 0.897 |
| top-1 accuracy | 0.815 | 0.875 | 0.617 | 0.821 | 0.843 | 0.783 |
| macro F1 | 0.818 | 0.878 | 0.648 | 0.822 | 0.845 | 0.781 |

E1/E2 use `baseline_best.pt`; E3 uses `domain_aug_best.pt`.

---

## 7. Known issues and caveats

- **The smartphone test set is 60 frames, one subject, one session.** The 95%
  Wilson interval on E2 accuracy is [0.490, 0.729] and on E3 [0.664, 0.869].
  These overlap; a paired test over the same 60 frames would be the correct
  analysis.
- **The RealSense test set appears densely annotated** (every frame of each
  clip, not two keyframes), so its 3450 frames are far fewer independent
  observations than the count suggests.
- **Single seed per configuration.** Run-to-run variance is unmeasured; small
  differences carry no weight.
- **Label noise in validation.** At least one held-out contributor's masks
  include the forearm, not just the hand. That contributor's frames score mask
  IoU around 0.03–0.05 despite visually correct predictions, which depresses
  validation segmentation and mask-derived detection numbers.
- **`domain_gap.py` is referenced but not included** in this submission. It
  produced the 3.19× / 1.70× / 1.33× measurements cited in
  `_augment_domain`'s docstring and in the report.
- **Depth is unused.** `depth/`, `depth_raw/` and `depth_metadata.json` are
  present in the dataset format but the model consumes RGB only.
- Duplicate helpers: `box_from_mask`, `confusion_matrix` and `per_class_prf`
  exist in both `utils.py` and their point of use, deliberately, so
  `evaluate.py` shows its own metric computation. They agree.
- In `visualise.py`'s `overlays`, the `set_ylabel` calls after `axis("off")`
  have no effect; the row identity is carried by the caption instead.

---

## 8. Report

`coursework1LSA.tex` builds the 6-page IEEE conference report. It expects these
files next to it (copy them out of `results/figs/`):

```
baseline_curves.png
exp2_cm.png          exp3_cm.png
exp2_overlays.png    exp3_overlays.png
```

Compile with pdfLaTeX. Export the PDF as
`coursework1LSA_Hector_de_la_Parte_Fernandez.pdf` for submission.

`domain_aug_curves.png`, `exp1_cm.png` and `comparison.png` are also produced by
Section 4 but are not used in the final six-page report.
