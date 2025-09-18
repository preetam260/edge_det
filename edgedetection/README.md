# Edge Detection Full Project (BSDS500) — Tasks 1–4

This repository contains a complete scaffold for on Edge Detection (Tasks 1-4):
- Task 1: Canny edge detection (scripts/run_canny.py)
- Task 2: Simple 3-layer CNN (scripts/train_cnn.py)
- Task 3: VGG16 + decoder (scripts/train_vgg.py)
- Task 4: HED-like model with side outputs (scripts/train_hed.py)
- Common utilities, dataset loader, losses, and inference/visualization scripts

## Quick setup

1. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Download BSDS500 dataset and place images/groundTruth folders under `data/BSDS500/` (preserve filenames).
   - Images should be under `data/BSDS500/images/{train,val,test}` or you can put them under `images/test` for quick runs.
   - Ground-truth `.mat` files under `data/BSDS500/groundTruth/{train,val,test}`

## Main scripts

- Task 1 (Canny): `python scripts/run_canny.py --images_dir data/BSDS500/images/test --gt_dir data/BSDS500/groundTruth/test --out_dir outputs/canny`
- Task 2 (CNN): `python scripts/train_cnn.py --data_dir data/BSDS500 --out_dir outputs/cnn --epochs 100`
- Task 3 (VGG+decoder): `python scripts/train_vgg.py --data_dir data/BSDS500 --out_dir outputs/vgg --epochs 100 --use_bilinear False`
- Task 4 (HED-like): `python scripts/train_hed.py --data_dir data/BSDS500 --out_dir outputs/hed --epochs 50`

## Files of interest
- `src/utils_bsds.py` — BSDS loaders + utilities (used by all scripts)
- `src/dataset.py` — PyTorch Dataset producing image tensors and averaged GT maps
- `src/losses.py` — class-balanced loss (HED-style)
- `src/models/cnn_simple.py` — Task 2 model
- `src/models/vgg_decoder.py` — Task 3 model (VGG backbone + transpose conv decoder)
- `src/models/hed_like.py` — Task 4 HED-like model with side outputs + fusion
- `scripts/infer_and_plot.py` — Run a checkpoint and visualize outputs vs GT

## Notes
- Training large networks benefits from a GPU. Scripts auto-detect CUDA if available.
- Use `--max_images` flags in training scripts for quick debugging.
