import torch

from fashion_cnn.gradcam import grad_cam
from fashion_cnn.model import SmallResNet, count_params


def test_forward_shape_and_size():
    m = SmallResNet()
    assert m(torch.randn(4, 1, 28, 28)).shape == (4, 10)
    assert count_params(m) < 1_000_000


def test_gradcam_range_and_shape():
    torch.manual_seed(0)
    cam = grad_cam(SmallResNet(), torch.randn(3, 1, 28, 28))
    assert cam.shape == (3, 28, 28)
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    m = SmallResNet(width=16)
    x, y = torch.randn(16, 1, 28, 28), torch.arange(16) % 10
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
