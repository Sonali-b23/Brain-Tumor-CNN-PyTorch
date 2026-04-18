# Brain-Tumor-CNN-PyTorch

A Convolutional Neural Network (CNN) based project that uses PyTorch to classify brain MRI images into binary categories: **Tumor** vs. **Non-Tumor**. This project leverages transfer learning and data augmentation to build an accurate predictive model capable of assisting in medical imaging and diagnostics.

## Overview
This repository provides an end-to-end pipeline to train, evaluate, and save a functional PyTorch classification model. By utilizing a pre-trained EfficientNet-B0 model and customizing its classifier, the project achieves an optimal balance between parameter efficiency, speed, and accuracy on limited medical datasets. 

## Key Technologies
* **PyTorch & Torchvision:** Core deep learning framework for defining, training, and running the neural network.
* **Scikit-Learn:** Emphasized during evaluation for building confusion matrices and classification reports (Metrics: Precision, Recall, F1-Score).
* **NumPy:** Underlying data structure math and matrix transformations.

## Project Architecture

### 1. Data Pipeline & Augmentation (`src/data_loader.py`)
Properly organizing dataset arrays is paramount. Training data is subjected to the following transforms to enforce generalization:
* Aspect ratio resizing to `224x224`.
* Random horizontal flips and up to 10-degree rotations.
* Data Scaling & Normalization utilizing standard ImageNet means and standard deviations.

Validation and Testing datasets sidestep the randomness of augmentations but run the default resizing and normalization to properly structure input tensors. 

### 2. Model Structure (`src/model.py`)
This project utilizes **EfficientNet-B0** pre-trained on ImageNet. Its dense classifier is cleanly swapped out. The pre-trained features (which can identify edges and foundational shapes) funnel into a brand new customizable Linear classifier initialized strictly for predicting our `len(classes)` (two dimensions: `yes` and `no`).

### 3. Training & Evaluation (`src/train.py`, `src/evaluate.py`)
The pipeline runs on an `Adam` optimizer alongside a `CrossEntropyLoss` function.
During the loop, the architecture tracks metrics across multiple epochs. Every time the evaluation scripts notice a boost in general Validation Accuracy, a dynamic checkpoint runs, isolating and saving the `best_model.pth` state dictionary.

---

## Results

After executing an automated run of 5 epochs (Batch Size = 8, LR = 1e-4) over the structured MRI dataset:

- **Best Validation Accuracy:** 83.33% *(Captured on Epoch 4)*
- **Final Training Accuracy:** 95.05%
- **Final Validation F1-Score:** 0.79

The robust checkpointing ensures that despite instances of late-epoch overfitting, the optimal inference weights are preserved in `outputs/checkpoints/best_model.pth`.

## Run the Project Yourself
1. Ensure your dataset is split and placed cleanly into `data/train` and `data/val` folders. 
2. Setup and install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Boot the pipeline simply with:
   ```bash
   python run.py
   ```
