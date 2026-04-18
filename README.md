# Brain-Tumor-CNN-PyTorch

A Convolutional Neural Network (CNN) based project that uses PyTorch to classify brain MRI images into binary categories: **Tumor** vs. **Non-Tumor**. This project leverages transfer learning and data augmentation to build an accurate predictive model capable of assisting in medical imaging and diagnostics.

---

## Overview

This repository provides an end-to-end pipeline to:
- Prepare and split the dataset
- Train a deep learning model
- Validate performance during training
- Perform final evaluation on unseen test data

By utilizing a pre-trained EfficientNet-B0 model and customizing its classifier, the project achieves an optimal balance between parameter efficiency, speed, and accuracy on limited medical datasets.

---

## Key Technologies

- **PyTorch & Torchvision:** Core deep learning framework for model building and training  
- **Scikit-Learn:** Used for evaluation metrics and confusion matrix generation  
- **NumPy:** Numerical operations and tensor transformations  

---

## Project Architecture

### 1. Data Pipeline & Augmentation (`src/data_loader.py`)

The dataset is structured into **train, validation, and test sets**.

Training data undergoes augmentation:
- Resize to `224x224`
- Random horizontal flips
- Random rotations (±10°)
- Normalization using ImageNet statistics

Validation and test datasets are only resized and normalized (no augmentation) to ensure consistent evaluation.

---

### 2. Model Structure (`src/model.py`)

- Uses **EfficientNet-B0** pre-trained on ImageNet  
- Final classifier layer is replaced to match binary classification (`tumor` vs `no tumor`)  
- Leverages transfer learning for faster convergence and better performance  

---

### 3. Training & Validation (`src/train.py`, `src/evaluate.py`)

- Optimizer: `Adam`  
- Loss Function: `CrossEntropyLoss`  

During training:
- Model is evaluated on validation data after each epoch  
- The best-performing model (based on validation accuracy) is saved automatically  

Saved checkpoint:
```

outputs/checkpoints/best_model.pth

```

---

### 4. Testing & Final Evaluation

After training completes, the pipeline performs evaluation on the **unseen test dataset**:

- Loads the best saved model from `outputs/checkpoints/best_model.pth`  
- Runs inference on test data  
- Generates predictions and labels  
- Creates a **Confusion Matrix visualization**  

Output:
```

outputs/test_confusion_matrix.png

```

This ensures the model is evaluated on completely unseen data, providing a more realistic measure of performance.

---

## Results

After training for 5 epochs (Batch Size = 8, Learning Rate = 1e-4):

- **Best Validation Accuracy:** 83.33% *(Epoch 4)*  
- **Final Training Accuracy:** 95.05%  
- **Final Validation F1-Score:** 0.79  
- **Test Evaluation:** Confusion matrix generated for final model  

The checkpointing mechanism ensures that the best model is preserved even if overfitting occurs in later epochs.

---

## Run the Project Yourself

### 1. Prepare Dataset
Ensure dataset is properly structured:
```

data/
train/
val/
test/

```

You can use the provided script:
```

data/split_dataset.py

````

---

### 2. Install Dependencies
```bash
pip install -r requirements.txt
````

---

### 3. Run the Pipeline

```bash
python run.py
```

---

## Output Summary

After execution, the following outputs are generated:

* Trained model checkpoint → `outputs/checkpoints/best_model.pth`
* Confusion matrix visualization → `outputs/test_confusion_matrix.png`

---

## Future Improvements

* Add Grad-CAM visualizations for model interpretability
* Experiment with deeper architectures (EfficientNet variants, ResNet)
* Hyperparameter tuning for improved accuracy
* Deploy model using a web interface (Flask/Streamlit)

---