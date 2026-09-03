import cv2
import numpy as np
import torch

# ImageNet normalization stats used everywhere else in this project
# (src/data_loader.py, src/utils.py). Grad-CAM needs these to
# unnormalize a tensor back into a displayable image.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # register_full_backward_hook is the non-deprecated replacement for
        # register_backward_hook and has well-defined semantics for modules
        # with multiple inputs.
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam - np.min(cam)
            cam = cam / cam_max
        else:
            # Degenerate case: every activation was zero for this class
            # (can happen for an untrained/near-random model). Avoid a
            # divide-by-zero and just return a blank map instead of NaNs.
            cam = np.zeros_like(cam)

        return cam, class_idx


def unnormalize(img_tensor):
    """Convert a normalized CHW tensor back to an HWC image in [0, 1].

    Uses the same per-channel ImageNet mean/std as the rest of the project
    (src/utils.py). The original version of this function used the red
    channel's mean/std as scalars for all three channels, which silently
    distorted the green and blue channels (up to ~0.08 absolute error) --
    every Grad-CAM overlay was rendered on a slightly wrong-colored image.
    """
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img


def overlay_cam_on_image(img_tensor, mask):
    """Return an RGB (0-1 float) image with the Grad-CAM heatmap overlaid."""
    img = unnormalize(img_tensor)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    overlay = heatmap + img
    overlay = overlay / np.max(overlay)
    return overlay


def show_cam_on_image(img_tensor, mask):
    """Kept for backwards compatibility: displays the overlay with matplotlib."""
    import matplotlib.pyplot as plt

    overlay = overlay_cam_on_image(img_tensor, mask)
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()
