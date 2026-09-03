# Brain-Tumor-CNN-PyTorch

A Convolutional Neural Network (CNN) based project that uses PyTorch to classify brain MRI images into binary categories: **Tumor** vs. **Non-Tumor**. This project leverages transfer learning and data augmentation to build a predictive model capable of assisting in medical imaging and diagnostics, along with Grad-CAM visualizations so a prediction can be visually explained rather than trusted blindly.

---

## Overview

This repository provides an end-to-end pipeline to:
- Prepare and split the dataset, **deduplicating by image content first** so the same MRI scan can never leak across train/val/test under two different filenames
- Train a deep learning model, with class-weighted loss, an LR scheduler, and early stopping
- Validate performance during training
- Perform final evaluation on unseen test data, and separately, **5-fold cross-validation** for a more robust accuracy estimate on this small dataset
- Explain any single prediction with a Grad-CAM heatmap overlay

By utilizing a pre-trained EfficientNet-B0 model and customizing its classifier, the project achieves a good balance between parameter efficiency, speed, and accuracy on limited medical datasets.

---

## Key Technologies

- **PyTorch & Torchvision:** Core deep learning framework for model building and training
- **Scikit-Learn:** Evaluation metrics, confusion matrix, stratified k-fold cross-validation
- **OpenCV:** Grad-CAM heatmap generation
- **NumPy:** Numerical operations and tensor transformations

---

## Project Architecture

### 1. Dataset Preparation (`data/split_dataset.py`)

The raw Kaggle dataset (`archive (1)/brain_tumor_dataset/`) turned out to contain **25 duplicate images saved under different filenames** (confirmed by MD5 content hashing — 253 files, only 228 unique images). The original split script split by filename alone, so the *same* MRI scan could land in both the training set and the test set under two different names — a real form of data leakage that quietly inflates reported accuracy, since the model can effectively "see" a test image during training.

`split_dataset.py` now deduplicates by file content (MD5 hash) before doing anything else, then does a seeded, reproducible 80/10/10 split. A cross-split hash check after every run confirms zero overlap between train/val/test.

```bash
python data/split_dataset.py --seed 42
```

### 2. Data Pipeline & Augmentation (`src/data_loader.py`)

Training data undergoes augmentation:
- Resize to `224x224`
- Random horizontal flips
- Random rotations (±10°)
- Normalization using ImageNet statistics

Validation and test datasets are only resized and normalized (no augmentation) to ensure consistent evaluation.

### 3. Model Structure (`src/model.py`)

- Uses **EfficientNet-B0** pre-trained on ImageNet (`EfficientNet_B0_Weights.DEFAULT` — the current, non-deprecated torchvision weights API)
- Final classifier layer is replaced to match binary classification (`no` vs `yes`)
- Leverages transfer learning for faster convergence and better performance on a small dataset

### 4. Training & Validation (`run.py`, `src/train.py`, `src/evaluate.py`)

- Optimizer: `Adam`, with `ReduceLROnPlateau` scheduling on validation loss
- Loss function: `CrossEntropyLoss`, **class-weighted** — the dataset is imbalanced (roughly 61% `yes` / 39% `no` after deduplication), so an unweighted loss quietly biases the model toward the majority class
- Early stopping (5 epochs without val-loss improvement) so training doesn't run past the point where it's helping
- Random seeds (`random`, `numpy`, `torch`) are fixed for reproducibility
- The best-performing model (by validation accuracy) is saved automatically to `outputs/checkpoints/best_model.pth`
- Every run's metrics — per-epoch history, best val accuracy, final test accuracy/loss, dataset sizes — are saved to `outputs/metrics.json`, so the numbers below can always be regenerated and checked rather than taken on faith

### 5. Testing & Final Evaluation

After training completes, the pipeline evaluates the **unseen test dataset**, loading the best checkpoint with `map_location` set so it works whether or not the machine running it has a GPU. It generates a confusion matrix (`outputs/test_confusion_matrix.png`).

### 6. Cross-Validation (`src/kfold_eval.py`)

The test set is small (a few dozen images), so a single train/test split accuracy number carries real sampling variance. This script pools every deduplicated image and runs stratified 5-fold cross-validation, training a fresh model per fold, and reports mean accuracy ± standard deviation instead of one point estimate:

```bash
python -m src.kfold_eval --folds 5 --epochs 8
```

### 7. Explainability (`src/gradcam.py`, `src/predict.py`)

`src/gradcam.py` previously existed but was never called anywhere in the codebase — genuinely dead code — and had a real bug in it: it unnormalized images using only the red channel's ImageNet mean/std (`0.485`, `0.229`) as scalars for *all three* channels, which silently distorted the green and blue channels by up to ~0.08 (out of a 0-1 range) in every rendered overlay. It also mixed up OpenCV's BGR heatmap with an RGB base image.

Both are fixed, and `src/predict.py` wires Grad-CAM into an actual, runnable feature — single-image inference with a saved heatmap overlay showing what the model looked at:

```bash
python -m src.predict --image path/to/scan.jpg
```

This prints the prediction and confidence, and saves a Grad-CAM overlay to `outputs/gradcam_explanations/`.

---

## Results

**A note on how these numbers were produced:** the environment used to fix this project could reach GitHub and PyPI but not `download.pytorch.org` (where torchvision fetches ImageNet-pretrained weights from), so the full pretrained-transfer-learning training run could not be executed there. What *was* verified end-to-end there, using randomly-initialized weights as a stand-in, is that the entire pipeline runs correctly: the deduplicated/leak-free split, class-weighted loss, checkpointing with `map_location`, `run.py`'s metrics.json output, `src/predict.py`'s Grad-CAM overlay generation, and `src/kfold_eval.py`'s cross-validation all execute without errors and produce sane output shapes.

**Run `python run.py` yourself to get the real, pretrained-transfer-learning numbers** — it will print them to the console and save them to `outputs/metrics.json`. The previously-reported numbers below (83.33% val accuracy, 95.05% train accuracy, 0.79 val F1) came from the *old*, leaky split (duplicate images crossing train/val/test), so they were optimistic and are not trustworthy as-is; expect the honest, leak-free numbers to be somewhat lower, and check `outputs/metrics.json`'s `dataset_sizes` field, since val/test are small enough (a few dozen images each) that single-run accuracy has real variance — cross-reference against `python -m src.kfold_eval` for a more stable estimate.

---

## Run the Project Yourself

### 1. Download the dataset

This repo does not commit the dataset (see `.gitignore`) — download the "Brain MRI Images for Brain Tumor Detection" dataset from Kaggle and extract it so you have `archive (1)/brain_tumor_dataset/{yes,no}/` at the repo root.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Split the dataset (deduplicated, leak-free)

```bash
python data/split_dataset.py --seed 42
```

### 4. Train

```bash
python run.py
```

### 5. Try inference + Grad-CAM on one image

```bash
python -m src.predict --image data/test/yes/<some_file>.jpg
```

### 6. (Optional) Cross-validate

```bash
python -m src.kfold_eval --folds 5 --epochs 8
```

---

## Output Summary

After execution, the following outputs are generated (all gitignored — regenerate them by running the pipeline):

* Trained model checkpoint → `outputs/checkpoints/best_model.pth`
* Metrics (val/test accuracy, per-epoch history, dataset sizes) → `outputs/metrics.json`
* Confusion matrix visualization → `outputs/test_confusion_matrix.png`
* Grad-CAM explanation for a single image → `outputs/gradcam_explanations/<name>_gradcam.png`

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

7 tests covering: the content-hash deduplication logic itself, a regression test that rebuilds a dataset with intentional duplicate-content files and asserts zero content overlap across the resulting train/val/test splits (the actual leakage bug, caught directly), split reproducibility under a fixed seed, and the Grad-CAM unnormalization math (including a test that would have failed against the original scalar-based bug).

---

## Known Limitations

* **The dataset is small** (228 unique images after deduplication) by deep learning standards — this is why cross-validation is reported alongside a single train/test split, and why results (however honestly measured) should be read as a proof-of-concept rather than a clinically validated result.
* **This is not a diagnostic tool.** It's a portfolio/learning project demonstrating a transfer-learning pipeline, explainability, and honest evaluation practice — not something that should inform real medical decisions.
* Grad-CAM explanations show *where* the model focused, not *why* it's medically meaningful — a highlighted region isn't automatically a tumor boundary.

---

## Key Highlights

* Content-hash deduplication before splitting, closing a real data leakage bug where duplicate images crossed train/val/test — with a regression test pinning it
* Class-weighted loss, LR scheduling, and early stopping instead of a fixed 5-epoch loop
* Honest, reproducible metrics saved to `outputs/metrics.json` on every run, plus 5-fold cross-validation for a more robust estimate on a small dataset
* A working Grad-CAM explainability feature (`src/predict.py`) — the old `gradcam.py` was dead code with a real unnormalization bug; both are fixed and wired up
* Pinned `requirements.txt`, `.gitignore` (no more committed `venv/`, `.pyc` files, or dataset binaries), and a small `pytest` suite

---

## Author

**Sonali**

---

## If you like this project

Give it a star on GitHub ⭐
