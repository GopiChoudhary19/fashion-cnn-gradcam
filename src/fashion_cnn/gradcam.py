import torch
import torch.nn.functional as F


def grad_cam(model, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
    """Grad-CAM on the last convolutional stage. Returns heatmaps in [0, 1] with the input's H x W.

    The model's train/eval mode and its parameter gradients are left as they were, so this is safe to call
    in the middle of training.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            feats = model.features(x)
            logits = model.head(feats)
            if target is None:
                target = logits.argmax(1)
            # gradient w.r.t. the feature map only; backward() would also accumulate into every parameter's .grad
            (grads,) = torch.autograd.grad(logits.gather(1, target[:, None]).sum(), feats)
    finally:
        model.train(was_training)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    lo = cam.amin(dim=(1, 2), keepdim=True)
    hi = cam.amax(dim=(1, 2), keepdim=True)
    return ((cam - lo) / (hi - lo + 1e-8)).detach()
