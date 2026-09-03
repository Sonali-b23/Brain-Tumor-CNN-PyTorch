import json
import os
import random

import numpy as np
import torch
import torch.optim as optim

from src.data_loader import get_data_loaders
from src.evaluate import evaluate
from src.model import create_model
from src.train import train_one_epoch
from src.utils import plot_confusion_matrix

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(image_folder_dataset, device):
    """Inverse-frequency class weights, so the loss doesn't let the
    majority class dominate. This dataset is imbalanced (~61% 'yes' /
    39% 'no' after deduplication), so an unweighted loss quietly biases
    the model toward predicting the majority class."""
    targets = [label for _, label in image_folder_dataset.samples]
    counts = np.bincount(targets)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    set_seed(SEED)

    DATA_DIR = 'data'
    BATCH_SIZE = 8
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    EARLY_STOP_PATIENCE = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloaders, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    model = create_model(len(class_names), DEVICE)

    class_weights = compute_class_weights(dataloaders['train'].dataset, DEVICE)
    print(f"Class names: {class_names} | class weights: {class_weights.tolist()}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)

    best_acc = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    os.makedirs('outputs/checkpoints', exist_ok=True)

    history = []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, dataloaders['val'], criterion, DEVICE)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'outputs/checkpoints/best_model.pth')
            print("Saved Best Model")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping: no val-loss improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load('outputs/checkpoints/best_model.pth', map_location=DEVICE))
    test_loss, test_acc, all_labels, all_preds = evaluate(model, dataloaders['test'], criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    plot_confusion_matrix(all_labels, all_preds, class_names, save_path='outputs/test_confusion_matrix.png')
    print("Test confusion matrix saved to 'outputs/test_confusion_matrix.png'")

    metrics = {
        'seed': SEED,
        'best_val_accuracy': best_acc,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'epochs_trained': len(history),
        'history': history,
        'dataset_sizes': {
            split: len(dataloaders[split].dataset) for split in ['train', 'val', 'test']
        },
        'note': (
            "train/val/test images are content-hash-deduplicated by data/split_dataset.py "
            "before splitting, so no image can appear in more than one split. val/test sets "
            "are small (see dataset_sizes), so single-run accuracy has real variance -- see "
            "src/kfold_eval.py for a cross-validated estimate."
        ),
    }
    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to 'outputs/metrics.json'")


if __name__ == '__main__':
    main()
