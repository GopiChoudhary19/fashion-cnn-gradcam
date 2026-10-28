import numpy as np
import torch
from PIL import Image

from fashion_cnn.model import SmallResNet
from fashion_cnn.predict import classify, load_image, load_model
from fashion_cnn.model import MEAN, STD


def test_load_image_matches_training_format(tmp_path):
    # a dark shirt on a white background, the usual product-photo look
    arr = np.full((90, 60), 255, np.uint8)
    arr[20:70, 15:45] = 30
    path = tmp_path / "shirt.png"
    Image.fromarray(arr).save(path)
    x = load_image(str(path))
    assert x.shape == (1, 1, 28, 28)
    raw = x[0, 0] * STD + MEAN
    assert raw[14, 14] > 0.5 and raw[0, 0] < 0.1  # inverted: garment bright, background dark


def test_checkpoint_roundtrip_and_ranking(tmp_path):
    torch.manual_seed(0)
    m = SmallResNet(width=8)
    torch.save(m.state_dict(), tmp_path / "best.pt")
    loaded = load_model(str(tmp_path / "best.pt"))
    ranked = classify(loaded, torch.randn(1, 1, 28, 28))
    probs = [p for _, p in ranked]
    assert len(ranked) == 3 and probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1


def test_transparent_png_is_treated_like_a_white_background(tmp_path):
    # a dark garment on a transparent canvas whose hidden RGB is black
    arr = np.zeros((100, 100, 4), np.uint8)
    arr[25:75, 30:70, :3] = 40
    arr[25:75, 30:70, 3] = 255
    path = tmp_path / "shirt.png"
    Image.fromarray(arr, "RGBA").save(path)
    raw = load_image(str(path))[0, 0] * STD + MEAN
    assert raw[14, 14] > 0.5 and raw[1, 1] < 0.1
