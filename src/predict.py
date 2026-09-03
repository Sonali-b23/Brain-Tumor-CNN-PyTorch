"""
Single-image inference with a Grad-CAM explanation.

This is what actually turns src/gradcam.py from dead code into something you
can run and see the result of. It loads the trained model, classifies one
MRI image, and saves a Grad-CAM overlay showing which regions of the image
the model focused on to make its prediction.

Usage:
    python -m src.predict --image path/to/scan.jpg
    python -m src.predict --image path/to/scan.jpg --output outputs/explained.png
"""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from src.gradcam import GradCAM, overlay_cam_on_image
from src.model import create_model

IMG_SIZE = 224
CLASS_NAMES = ['no', 'yes']  # matches ImageFolder's alphabetical class order


def load_model(checkpoint_path, num_classes, device):
    model = create_model(num_classes, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    return tensor


def predict(image_path, checkpoint_path='outputs/checkpoints/best_model.pth',
            output_path=None, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(checkpoint_path, len(CLASS_NAMES), device)
    input_tensor = preprocess_image(image_path).to(device)

    # EfficientNet-B0's final conv block -- a reasonable, commonly-used
    # Grad-CAM target layer for this architecture.
    target_layer = model.features[-1]
    cam_engine = GradCAM(model, target_layer)

    with torch.enable_grad():
        mask, class_idx = cam_engine.generate(input_tensor)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1).squeeze()

    predicted_class = CLASS_NAMES[class_idx]
    confidence = probs[class_idx].item()

    overlay = overlay_cam_on_image(input_tensor, mask)

    if output_path is None:
        output_path = os.path.join('outputs', 'gradcam_explanations',
                                    os.path.splitext(os.path.basename(image_path))[0] + '_gradcam.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    import matplotlib.pyplot as plt
    plt.imsave(output_path, overlay)

    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': {name: probs[i].item() for i, name in enumerate(CLASS_NAMES)},
        'gradcam_path': output_path,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Path to an MRI image')
    parser.add_argument('--checkpoint', default='outputs/checkpoints/best_model.pth')
    parser.add_argument('--output', default=None, help='Where to save the Grad-CAM overlay')
    args = parser.parse_args()

    result = predict(args.image, checkpoint_path=args.checkpoint, output_path=args.output)
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.1%})")
    print(f"Probabilities: {result['probabilities']}")
    print(f"Grad-CAM overlay saved to: {result['gradcam_path']}")
