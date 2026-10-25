"""Classify a clothing image and save its Grad-CAM overlay.

python -m fashion_cnn.predict photo.png [--checkpoint reports/best.pt] [--out cam.png]
"""
import argparse

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageOps

from .gradcam import grad_cam
from .model import CLASSES, MEAN, STD, SmallResNet

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_image(path: str) -> torch.Tensor:
    """Any image -> 1x1x28x28 tensor in the training distribution (light garment on a dark background)."""
    img = ImageOps.grayscale(Image.open(path))
    arr = np.asarray(img, dtype=np.float32)
    if arr.mean() > 127:  # product photos are usually dark-on-light; Fashion-MNIST is light-on-dark
        img = ImageOps.invert(img)
    img = ImageOps.pad(img, (28, 28), color=0)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255)
    return ((x - MEAN) / STD)[None, None]


def load_model(checkpoint: str) -> SmallResNet:
    state = torch.load(checkpoint, map_location="cpu")
    width = state["stem.0.weight"].shape[0]
    model = SmallResNet(width=width)
    model.load_state_dict(state)
    return model.eval()


def classify(model: SmallResNet, x: torch.Tensor, top: int = 3) -> list[tuple[str, float]]:
    with torch.no_grad():
        p = ((model(x).softmax(1) + model(x.flip(-1)).softmax(1)) / 2)[0]
    vals, idx = p.topk(top)
    return [(CLASSES[i], round(float(v), 4)) for v, i in zip(vals, idx)]


def main(args):
    model = load_model(args.checkpoint)
    x = load_image(args.image)
    ranked = classify(model, x)
    for name, prob in ranked:
        print(f"{name:12s} {prob:.3f}")
    cam = grad_cam(model, x)[0].numpy()
    img = x[0, 0].numpy() * STD + MEAN
    fig, ax = plt.subplots(1, 2, figsize=(5, 2.6))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("input (28x28)", fontsize=9)
    ax[1].imshow(img, cmap="gray")
    ax[1].imshow(cam, cmap="jet", alpha=0.45)
    ax[1].set_title(f"{ranked[0][0]} ({ranked[0][1]:.0%})", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Grad-CAM overlay written to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--checkpoint", default="reports/best.pt")
    ap.add_argument("--out", default="gradcam_prediction.png")
    main(ap.parse_args())
