"""Train a residual CNN on Fashion-MNIST, evaluate on the official test set and render Grad-CAM maps."""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from .gradcam import grad_cam
from .model import CLASSES, MEAN, STD, SmallResNet, count_params

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Batched augmentation on normalized tensors: random +-2px shift, horizontal flip, random erasing."""
    n = x.shape[0]
    pad = F.pad(x, (2, 2, 2, 2), value=-MEAN / STD)
    dx, dy = torch.randint(0, 5, (2, n))
    rows = (dy[:, None] + torch.arange(28))[:, :, None].expand(n, 28, 28)
    cols = (dx[:, None] + torch.arange(28))[:, None, :].expand(n, 28, 28)
    x = pad[torch.arange(n)[:, None, None], 0, rows, cols][:, None]
    flip = torch.rand(n) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    erase = torch.rand(n) < 0.25
    h, w = torch.randint(4, 10, (2, n))
    y0, x0 = torch.randint(0, 20, (2, n))
    ar = torch.arange(28)
    box = ((ar[None, :, None] >= y0[:, None, None]) & (ar[None, :, None] < (y0 + h)[:, None, None])
           & (ar[None, None, :] >= x0[:, None, None]) & (ar[None, None, :] < (x0 + w)[:, None, None]))
    return x.masked_fill((box & erase[:, None, None])[:, None], 0.0)


def loaders(root: str, batch: int, seed: int):
    to_tensor = lambda ds: ((ds.data.float() / 255 - MEAN) / STD)[:, None]
    full = datasets.FashionMNIST(root, train=True, download=True)
    test_raw = datasets.FashionMNIST(root, train=False, download=True)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full), generator=g)
    X, y = to_tensor(full), full.targets
    tr = TensorDataset(X[perm[:55000]], y[perm[:55000]])
    va = TensorDataset(X[perm[55000:]], y[perm[55000:]])
    test = TensorDataset(to_tensor(test_raw), test_raw.targets)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch, shuffle=sh, generator=g if sh else None)
    return mk(tr, True), mk(va, False), mk(test, False), test


@torch.no_grad()
def predict(model, loader, device, tta: bool = False):
    """Class probabilities; with tta=True, averaged over the image and its mirror (training uses flips too)."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        p = model(x).softmax(1)
        if tta:
            p = (p + model(x.flip(-1)).softmax(1)) / 2
        ps.append(p.cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    tr, va, te, test_ds = loaders(args.data_dir, args.batch, args.seed)

    model = SmallResNet(width=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    history, best_acc = [], 0.0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in tr:
            x, y = augment(x).to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(y)
        yv, pv = predict(model, va, device)
        acc = float((pv.argmax(1) == yv).mean())
        history.append({"epoch": ep + 1, "train_loss": run / len(tr.dataset), "val_acc": acc})
        print(f"epoch {ep + 1:2d}  loss {run / len(tr.dataset):.4f}  val_acc {acc:.4f}  {time.time() - t0:.0f}s", flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    # whether flip-averaging helps is checked on the validation split, not the test set
    yv, pv_tta = predict(model, va, device, tta=True)
    yt, pt_plain = predict(model, te, device)
    _, pt = predict(model, te, device, tta=True)
    pred = pt.argmax(1)
    rep = classification_report(yt, pred, target_names=CLASSES, output_dict=True, digits=4)
    cm = confusion_matrix(yt, pred)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    metrics = {
        "params": count_params(model),
        "epochs": args.epochs,
        "best_val_acc": round(best_acc, 4),
        "best_val_acc_tta": round(float((pv_tta.argmax(1) == yv).mean()), 4),
        "test_acc_no_tta": round(float((pt_plain.argmax(1) == yt).mean()), 4),
        "test_acc": round(float((pred == yt).mean()), 4),
        "test_macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "per_class_f1": {c: round(rep[c]["f1-score"], 4) for c in CLASSES},
        "most_confused_pair": f"{CLASSES[i]} -> {CLASSES[j]} ({int(off[i, j])} images)",
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    for a in range(10):
        for b in range(10):
            ax.text(b, a, cm[a, b], ha="center", va="center", fontsize=7, color="white" if cm[a, b] > 500 else "black")
    ax.set(xlabel="predicted", ylabel="true", title=f"Fashion-MNIST test confusion matrix (acc {metrics['test_acc']:.4f})")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)

    rng = np.random.default_rng(args.seed)
    wrong = np.where(pred != yt)[0]
    picks = list(rng.choice(np.where(pred == yt)[0], 6, replace=False)) + list(rng.choice(wrong, 4, replace=False))
    x = torch.stack([test_ds[k][0] for k in picks]).to(device)
    cams = grad_cam(model, x).cpu().numpy()
    fig, ax = plt.subplots(2, 10, figsize=(17, 4))
    for c, k in enumerate(picks):
        img = x[c, 0].cpu().numpy() * STD + MEAN
        ax[0, c].imshow(img, cmap="gray")
        ax[0, c].set_title(f"true: {CLASSES[yt[k]]}", fontsize=8)
        ax[1, c].imshow(img, cmap="gray")
        ax[1, c].imshow(cams[c], cmap="jet", alpha=0.45)
        ax[1, c].set_title(f"pred: {CLASSES[pred[k]]}", fontsize=8, color="green" if pred[k] == yt[k] else "red")
        ax[0, c].axis("off")
        ax[1, c].axis("off")
    fig.suptitle("Grad-CAM: where the network looks (last 4 columns are mistakes)")
    fig.tight_layout()
    fig.savefig(out / "gradcam.png", dpi=120)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda", action="store_true")
    main(ap.parse_args())
