"""Shared pieces: paths, model, data loading, training loop, evaluation."""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
for d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MnistCNN(nn.Module):
    """Two conv blocks + two FC layers. Inputs are [0,1] images of shape (N,1,28,28)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.drop1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(self.drop1(x), 1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.drop2(self.features(x)))


def load_mnist():
    """Returns (x_train, y_train, x_test, y_test) as float32 [0,1] NCHW / int64 numpy arrays."""
    tf = transforms.ToTensor()
    train = datasets.MNIST(os.path.join(DATA_DIR, "raw"), train=True, download=True, transform=tf)
    test = datasets.MNIST(os.path.join(DATA_DIR, "raw"), train=False, download=True, transform=tf)

    def to_np(ds):
        x = ds.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
        return x, ds.targets.numpy().astype(np.int64)

    return (*to_np(train), *to_np(test))


def make_loader(x, y, batch_size=128, shuffle=False):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, x, y, epochs=5, lr=1e-3, batch_size=128, adv_fn=None, log=print):
    """Standard Adam training. If adv_fn is given, each batch is half clean / half adversarial
    (adv_fn(model, xb, yb) -> perturbed xb), i.e. adversarial training."""
    device = get_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2), gamma=0.3)
    loader = make_loader(x, y, batch_size, shuffle=True)
    history = []
    for epoch in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        t0 = time.time()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if adv_fn is not None:
                half = xb.size(0) // 2
                model.eval()
                x_adv = adv_fn(model, xb[half:], yb[half:])
                model.train()
                xb = torch.cat([xb[:half], x_adv])
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        sched.step()
        rec = {"epoch": epoch + 1, "loss": loss_sum / total, "acc": correct / total,
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        log(f"epoch {rec['epoch']}: loss={rec['loss']:.4f} acc={rec['acc']:.4f} ({rec['seconds']}s)")
    return history


def pgd_torch(model, x, y, eps=0.3, alpha=0.05, steps=10, random_start=True):
    """L-inf PGD used inside the adversarial-training loop (fast, on-device)."""
    x_adv = x.clone().detach()
    if random_start:
        x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    return x_adv.detach()


@torch.no_grad()
def predict(model, x, batch_size=1000):
    device = get_device()
    model.to(device).eval()
    logits = []
    for i in range(0, len(x), batch_size):
        logits.append(model(torch.from_numpy(x[i:i + batch_size]).to(device)).cpu())
    return torch.cat(logits)


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def inference_time(model, x, n_single=200):
    """Batched throughput on the default device and single-image latency on CPU."""
    device = get_device()
    model.to(device).eval()
    xb = torch.from_numpy(x[:1000]).to(device)
    model(xb)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(5):
        model(xb)
    _sync(device)
    batch_ms = (time.perf_counter() - t0) / 5 * 1000

    cpu_model = model.to("cpu")
    times = []
    for i in range(n_single):
        xi = torch.from_numpy(x[i:i + 1])
        t0 = time.perf_counter()
        cpu_model(xi)
        times.append((time.perf_counter() - t0) * 1000)
    model.to(device)
    return {"device": device.type,
            "batch1000_ms": round(batch_ms, 2),
            "per_image_batched_ms": round(batch_ms / 1000, 4),
            "single_image_cpu_ms_median": round(float(np.median(times)), 3)}


def evaluate(model, x, y, name, with_timing=False):
    """Accuracy, mean CE loss, confusion matrix, per-class report (and optional timing)."""
    logits = predict(model, x)
    yt = torch.from_numpy(y)
    preds = logits.argmax(1).numpy()
    res = {
        "name": name,
        "n": int(len(y)),
        "accuracy": float((preds == y).mean()),
        "loss": float(F.cross_entropy(logits, yt).item()),
        "confusion_matrix": confusion_matrix(y, preds, labels=list(range(10))).tolist(),
        "report": classification_report(y, preds, labels=list(range(10)), output_dict=True, zero_division=0),
    }
    if with_timing:
        res["timing"] = inference_time(model, x)
    return res


def save_json(obj, name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def plot_confusion(cm, title, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=120)
    plt.close(fig)


def save_model(model, name, meta=None):
    path = os.path.join(MODEL_DIR, name)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)
    return path


def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    ckpt = torch.load(path)
    model = MnistCNN()
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
