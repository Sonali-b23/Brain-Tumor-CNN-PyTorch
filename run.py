import torch
import torch.optim as optim
from src.data_loader import get_data_loaders
from src.model import create_model
from src.train import train_one_epoch
from src.evaluate import evaluate
import os

def main():
    DATA_DIR = 'data'
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloaders, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    model = create_model(len(class_names), DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    os.makedirs('outputs/checkpoints', exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, dataloaders['val'], criterion, DEVICE)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'outputs/checkpoints/best_model.pth')
            print("Saved Best Model")

if __name__ == '__main__':
    main()
