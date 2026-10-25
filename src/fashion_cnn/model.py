import torch
from torch import nn

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MEAN, STD = 0.2860, 0.3530  # Fashion-MNIST training-set pixel statistics


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
        )
        self.skip = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class SmallResNet(nn.Module):
    """~0.7M-parameter residual CNN for 28x28 grayscale images."""

    def __init__(self, num_classes: int = 10, width: int = 32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(1, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(ResBlock(w, w), ResBlock(w, w))
        self.stage2 = nn.Sequential(ResBlock(w, 2 * w, 2), ResBlock(2 * w, 2 * w))
        self.stage3 = nn.Sequential(ResBlock(2 * w, 4 * w, 2), ResBlock(4 * w, 4 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(4 * w, num_classes))

    def features(self, x):
        return self.stage3(self.stage2(self.stage1(self.stem(x))))

    def forward(self, x):
        return self.head(self.features(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = SmallResNet()
    print(count_params(m), m(torch.zeros(2, 1, 28, 28)).shape)
