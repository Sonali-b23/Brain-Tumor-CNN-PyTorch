import torch.nn as nn
import torchvision.models as models
import torch

def create_model(num_classes, device):
    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)
    return model
