# src/utils.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Unnormalize and show image
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.pause(0.001)

# Plot a batch of images with predictions
def visualize_batch(inputs, classes, preds=None, title="Predictions"):
    inputs = inputs.cpu()
    fig = plt.figure(figsize=(12, 6))
    for i in range(min(8, inputs.size(0))):
        ax = fig.add_subplot(2, 4, i + 1)
        imshow(inputs[i])
        if preds is not None:
            ax.set_title(f"Pred: {preds[i]}\nTrue: {classes[i]}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(true, preds, class_names, save_path=None):
    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.show()
