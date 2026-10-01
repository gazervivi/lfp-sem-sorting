# SEM-based sorting of retired LiFePO4 cathodes — dataset, splits, code and results

Companion repository for the manuscript:

> **How evaluation protocols distort SEM-based classification of retired LiFePO4
> cathodes for direct recycling: leakage quantification, spatial-independence
> verification, and corrected performance estimates** (under review)

This repository releases everything needed to reproduce and audit the study:
the full SEM image set, all split manifests, every per-tile prediction file,
all training/evaluation/analysis scripts, and the complete run metrics.
Model weights are distributed as a GitHub Release asset (see below).

---

## 1. What is in the dataset

**176 SEM images** (1280 × 960 px) of LiFePO4 cathode surfaces from **four
cells**, one degradation state per cell:

| Class | Folder | State | Images | Cell |
|---|---|---|---|---|
| `A_new` | `new` | Pristine, uncycled | 50 | Cell 1 |
| `B_2V` | `2v` | Over-discharged to 2 V, gassing | 47 | Cell 2 |
| `C_0V` | `0v` | Over-discharged to 0 V, copper dissolution | 39 | Cell 3 |
| `D_0Vcu` | `0V-析铜` | 0 V hold with visible copper deposition | 40 | Cell 4 |

Counts are taken from `data/manifest_clean.csv` (the de-duplicated corpus
that all experiments used). After md5 de-duplication the corpus is
177 → 176 images (the `0v` folder contains 40 files, one of which is an
exact duplicate, leaving class C with 39). **One class = one cell**: class
identity and cell identity are fully confounded by design of the source
campaign (see Limitations).

Each image in `data/raw/<batch>/<class-folder>/` is accompanied by a `.txt`
metadata file with SEM acquisition parameters (accelerating voltage, stage
coordinates STGX/STGY, working distance, etc.). The stage coordinates are
what make the spatial-independence verification (Protocol P3) possible.

```
data/
├── raw/                         # 177 original images + 177 metadata .txt files
├── manifest_clean.csv           # per-image manifest: class, MAG, STGX/Y, md5, dup flag
└── splits/                      # all split manifests used in the paper
    ├── protocol_tilelevel_seed4{2,3,4}.csv    # P1: random tile-level split (leaky)
    ├── protocol_imagelevel_seed4{2,3,4}.csv   # P2: random image-level split
    ├── blocked_imagelevel.csv                 # P3: spatial-block split
    ├── kfold5_f{1..5}_imagelevel.csv          # KF: 5-fold image-level CV
    └── summary.json, blocked_summary.json, kfold5_summary.json
```

Split CSV columns: `tile, cls, source_image, src_group, x, y, split`.
Each source image yields 16 tiles of 256 × 256 cropped at **random positions**
(fixed seed); `x, y` is the crop origin in the source image (`-1` in the
k-fold manifests means "not recorded" — regenerate with `prepare_dataset.py`
if exact origins are needed). 176 images × 16 tiles = **2816 tiles**.

## 2. Evaluation protocols (the point of the paper)

| Protocol | Split unit | What it estimates |
|---|---|---|
| **P1** tile-level | random tiles | Leaky: tiles of one image fall into train *and* test. All 176 images are affected. |
| **P2** image-level | whole images | No tile leakage; ~20% of test images still have a spatially overlapping training neighbour (at the nominal 10 µm FOV). |
| **P3** spatial-block | contiguous stage-coordinate regions | Min test↔train stage distance 90.1 µm — no overlap under any plausible FOV. Single hold-out, not spatial CV. |
| **KF** 5-fold | whole images, stratified | Every image is tested exactly once; the protocol-honest coverage baseline. Trains on ~70% per fold (not matched to P1/P2's 60%). |

Headline results (tile-level accuracy): P1 98.30% vs P2 96.59% (ResNet-50,
seed 42); the gap is small in aggregate but **concentrated in class C recall**
(image-level C recall: 0.98–1.00 under P1 → 0.925 under KF), i.e. leakage
hides exactly the errors that matter for safety-critical sorting.
Image-level cluster bootstrap CIs include zero for part of the backbones —
leakage inflates scores directionally but the magnitude is not precisely
resolvable at n = 176. See the manuscript and `results/analysis_v3/`.

## 3. Repository layout

```
code/        all training / evaluation / analysis scripts (Python, PyTorch)
data/        raw SEM images + metadata, manifest, all split manifests
results/
├── predictions/   per-run per-tile prediction CSVs (46 files; test+val per run)
├── metrics/       per-run metrics.json + merged k-fold metrics and error lists
├── figures/       paper figures (morphology grid, protocols, error cases, …)
├── analysis_v3/   image-level cluster bootstrap, validation-only threshold
│                  selection, silhouette robustness, P2 seed variance
└── fov_overlap_report.json   FOV 5/10/15/20 µm overlap sensitivity
docs/        server environment README, backup/restore notes (Chinese)
```

Model weights (22 × `best.pt`, 1.4 GB total; two Swin-T files exceed the
100 MB git limit) are **not in the git tree**. Download
`lfp_sem_model_weights.zip` from the
[Releases](../../releases) page. Checkpoints were saved with full pickle —
load with `torch.load(path, weights_only=False)`.

## 4. Reproduce

```bash
# 1. environment (Python 3.10+, CUDA optional but recommended)
pip install torch torchvision timm scikit-learn pandas pillow matplotlib

# 2. tiles + splits (deterministic, fixed seeds)
python code/prepare_dataset.py --raw data/raw --out data/v2
python code/make_kfold.py         # 5-fold image-level manifests
python code/make_blocked_split.py # spatial-block split from stage coordinates

# 3. train one protocol/backbone, e.g. P2 + ResNet-50 seed 42
python code/train_e1.py --protocol P2 --backbone resnet50 --seed 42 \
    --splits data/v2/splits --tiles data/v2/tiles

# 4. evaluation & analyses
python code/eval_e2_aggregate.py  # image-level aggregation + rejection curves
python code/eval_e3_cam.py        # HiResCAM saliency + AOPC (exploratory)
python code/eval_e4_tsne.py       # feature-space t-SNE + silhouette
python code/analysis_v3.py        # bootstrap CIs, validation-only thresholds
```

Expected wall time: ~10 min/run for ResNet-50 / EfficientNet-B0 on one
RTX-class GPU (early stopping, ~30–50 epochs); Swin-T and DINOv2 longer.

## 5. Limitations (read before citing the numbers)

- **class ≡ cell**: each class is one physical cell. All accuracies —
  including the KF baseline — are optimistic upper bounds for unseen cells
  of unknown history. Leave-one-battery-out validation on more cells is
  required before any deployment claim.
- Protocol comparisons are **not strictly paired**: P1, P2 and KF evaluate on
  different test sets; comparability rests on identical split ratios/seeds
  and on the consistent direction across four backbones.
- KF folds train on ~70% of images vs ~60% for P1/P2 (not matched).
- P3 is a single spatial hold-out (42 test images), not spatial cross-validation.
- Rejection-threshold results are proof-of-concept: the validation set
  (35 images, 0 errors) cannot support threshold calibration — see
  `results/analysis_v3/analysis_v3_summary.txt`.
- HiResCAM/AOPC analysis is exploratory; no validated causal evidence for or
  against saliency faithfulness.

## 6. License & citation

- **Code**: MIT License (see `LICENSE`).
- **Images, split manifests and prediction files**: CC BY 4.0
  (see `LICENSE-DATA`). Please cite the manuscript when using the dataset.

If you use this repository, please cite:

```bibtex
@article{zhang2026lfpsem,
  title   = {How evaluation protocols distort SEM-based classification of
             retired LiFePO4 cathodes for direct recycling: leakage
             quantification, spatial-independence verification, and
             corrected performance estimates},
  author  = {Zhang, Wentao and {[co-authors]}},
  journal = {under review},
  year    = {2026}
}
```

## 7. Contact

Wentao Zhang — mapery@163.com
Issues and questions: please use GitHub Issues.
