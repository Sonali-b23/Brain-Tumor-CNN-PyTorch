import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return epoch_loss, epoch_acc.item()
