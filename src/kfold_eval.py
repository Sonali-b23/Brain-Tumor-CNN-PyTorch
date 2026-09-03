"""
Stratified k-fold cross-validation for the brain tumor classifier.

The train/val/test split in run.py is a single 80/10/10 split, and the
val/test sets are small (a few dozen images each) -- so a single accuracy
number from that split carries real sampling variance and can look
noisier (or luckier) than the model actually is. This script instead pools
every deduplicated image (across the existing train/val/test folders, which
between them already contain every unique image) and runs stratified k-fold
cross-validation, training a fresh model on each fold, to give a more
robust accuracy estimate with a spread (mean +/- std) instead of one point
estimate.

Usage:
    python -m src.kfold_eval --folds 5 --epochs 8
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.model import create_model

IMG_SIZE = 224
CLASS_NAMES = ['no', 'yes']


class ImagePathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]


def collect_all_images(data_dir):
    """Every image currently under data/{train,val,test}/{yes,no} is
    already deduplicated by data/split_dataset.py, so pooling them back
    together for cross-validation is safe -- no image appears twice."""
    paths, labels = [], []
    for split in ['train', 'val', 'test']:
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, split, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath):
                    paths.append(fpath)
                    labels.append(class_idx)
    return paths, np.array(labels)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def run_kfold(data_dir='data', n_folds=5, n_epochs=8, batch_size=8, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths, labels = collect_all_images(data_dir)
    train_tf, eval_tf = get_transforms()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        train_paths = [paths[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_paths = [paths[i] for i in val_idx]
        val_labels = labels[val_idx]

        train_ds = ImagePathDataset(train_paths, train_labels, train_tf)
        val_ds = ImagePathDataset(val_paths, val_labels, eval_tf)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = create_model(len(CLASS_NAMES), device)
        counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
        class_weights = torch.tensor(counts.sum() / (len(counts) * counts),
                                      dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(n_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        fold_acc = correct / total
        fold_accuracies.append(fold_acc)
        print(f"Fold {fold + 1}/{n_folds}: val accuracy = {fold_acc:.4f} (n={total})")

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    print(f"\n{n_folds}-fold CV accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    return {'fold_accuracies': fold_accuracies, 'mean_accuracy': mean_acc, 'std_accuracy': std_acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_kfold(n_folds=args.folds, n_epochs=args.epochs,
              batch_size=args.batch_size, seed=args.seed)
