# Learning when to scan: OCT-privileged selective acquisition for resource-efficient silicon carbide wafer inspection

Code for

> Qi Lan, Zheng Li, Weiyi Zhang, Xipeng Shen, Jianlong Yang.
> **Learning when to scan: OCT-privileged selective acquisition for resource-efficient silicon carbide wafer inspection.**
> School of Biomedical Engineering, Shanghai Jiao Tong University, 2026.

A robot-guided microscopy–OCT inspection framework in which OCT volumes serve as
*privileged information* during training of a microscopy-only Student. The Student
predicts a three-class defect label and an ordinal subsurface-likelihood score; a
label-free, two-threshold (multi-level Otsu) rule then decides per candidate whether
to accept the microscopy decision directly (low / high score) or refer it for OCT
(intermediate band). Across five leave-one-wafer-out folds the method retains 100%
subsurface-defect recall at 27.4% of the candidate-guided OCT acquisition budget.

---

## Repository layout

```
SiC-OCT-privileged-inspection/
├── code/                     # training + evaluation pipeline
│   ├── common.py             # config, dataset, q*/U* targets, samplers, losses
│   ├── models.py             # OCTEncoder (3D ResNet-18), Teacher, Student, classifier
│   ├── train_mae.py          # Stage A: tube-masked autoencoder pretraining on OCT
│   ├── train_lowo.py         # outer leave-one-wafer-out training (Teacher + 3 Students)
│   ├── targets.py            # risk cross-fitting, q*/U* construction (Eqs. 2-4)
│   ├── losses.py             # teacher / student / direct / direct-reg objectives
│   ├── referral.py           # multi-level Otsu, deployment rule, footprint budget
│   ├── evaluate.py           # per-wafer AUROC/AUPRC, operating point, macro-F1
│   └── run_all.py            # pytest gate + train + evaluate entry point
├── tests/test_methods.py     # unit tests (fold disjointness, targets, Otsu, footprints)
├── data/                     # local input directory; data files are not included
├── requirements.txt
└── LICENSE                   # MIT
```

## Data

Data files are not currently included in this repository. To run training or evaluation,
place `data/manifest.csv`, the precomputed DINOv2 features in `data/patches/`, and the
raw OCT volumes in `data/volumes/` locally, as required by the corresponding pipeline
stage. Training requires `data/volumes/<roi_id>.npy`; otherwise it will raise
`FileNotFoundError`. The unit tests use synthetic in-memory inputs and do not require
these data files.

## Usage

```bash
conda create -n sic-oct python=3.12
pip install -r requirements.txt

# unit tests
python -m pytest -q

# train all five leave-one-wafer-out folds (requires data/volumes/, GPU recommended)
python code/run_all.py --train --output outputs/lowo
# or a single fold:
python code/train_lowo.py --test-wafer W1 \
  --mae-epochs 40 --teacher-epochs 150 --student-epochs 100

# evaluate held-out predictions: per-wafer AUROC/AUPRC, label-free Otsu operating
# point (thresholds, referral share, subsurface recall, acquisition budget), macro-F1
python code/evaluate.py \
  --predictions outputs/lowo/predictions_all_folds.csv \
  --manifest data/manifest.csv \
  --output outputs/metrics_per_wafer.csv
```

Verified environment: PyTorch 2.7.1+cu118, torchvision 0.22.1, pandas 2.3.3,
numpy 1.26.3 on Windows with an NVIDIA RTX 4090 (training smoke-tested end to end).

## License

MIT License, Copyright (c) 2026 Jianlong Yang and co-authors. See `LICENSE`.

---
