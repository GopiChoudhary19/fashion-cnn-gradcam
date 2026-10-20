import torch
import torch.nn.functional as F


def grad_cam(model, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
    """Grad-CAM on the last convolutional stage. Returns heatmaps in [0, 1] with the input's H x W."""
    model.eval()
    feats = model.features(x)
    feats.retain_grad()
    logits = model.head(feats)
    if target is None:
        target = logits.argmax(1)
    logits.gather(1, target[:, None]).sum().backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    lo = cam.amin(dim=(1, 2), keepdim=True)
    hi = cam.amax(dim=(1, 2), keepdim=True)
    return ((cam - lo) / (hi - lo + 1e-8)).detach()
