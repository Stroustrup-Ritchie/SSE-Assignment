"""Blue team defense #2: Feature Squeezing (Xu et al., NDSS 2018) as an input-validation layer.

Two squeezers: 1-bit depth reduction and 2x2 median smoothing. A sample is flagged as
adversarial when the L1 distance between the model's softmax on the raw input and on any squeezed
input exceeds a threshold calibrated for a 5% false-positive rate on held-out clean training data.
We also report accuracy when the 1-bit squeezed input is fed to the model (squeeze-then-classify).
"""
import os
import sys

import numpy as np
import torch
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_DIR, load_mnist, load_model, predict, save_json  # noqa: E402

FPR_TARGET = 0.05
ADV_SETS = [f"{a}_eps{e}" for a in ("fgsm", "pgd") for e in (0.1, 0.2, 0.3)]


def bit_depth(x, bits=1):
    levels = 2 ** bits - 1
    return (np.round(x * levels) / levels).astype(np.float32)


def median_smooth(x, k=2):
    return median_filter(x, size=(1, 1, k, k)).astype(np.float32)


def squeeze_score(model, x):
    p = torch.softmax(predict(model, x), 1)
    scores = [(p - torch.softmax(predict(model, s(x)), 1)).abs().sum(1) for s in (bit_depth, median_smooth)]
    return torch.stack(scores).max(0).values.numpy()


def load_npz(name):
    with np.load(os.path.join(DATA_DIR, "adversarial", name + ".npz")) as d:
        return d["x"], d["y"]


def run(model_name, tag, x_val, x_te, y_te):
    model = load_model(model_name)
    thr = float(np.quantile(squeeze_score(model, x_val), 1 - FPR_TARGET))
    clean_scores = squeeze_score(model, x_te)
    out = {"model": model_name, "threshold": thr,
           "clean_test_fpr": float((clean_scores > thr).mean()),
           "clean_acc_squeezed_input": float((predict(model, bit_depth(x_te)).argmax(1).numpy() == y_te).mean()),
           "attacks": []}
    for name in ADV_SETS:
        # white-box samples crafted against this same model (adaptive to the model, not to the squeezer)
        x_adv, y_adv = load_npz(f"{name}_{tag}")
        s = squeeze_score(model, x_adv)
        preds = predict(model, x_adv).argmax(1).numpy()
        sq_preds = predict(model, bit_depth(x_adv)).argmax(1).numpy()
        fooled = preds != y_adv
        out["attacks"].append({
            "set": name,
            "detection_rate_all": float((s > thr).mean()),
            "detection_rate_successful_adv": float((s[fooled] > thr).mean()) if fooled.any() else None,
            "acc_no_defense": float((preds == y_adv).mean()),
            "acc_squeezed_input": float((sq_preds == y_adv).mean()),
            # system-level: a sample counts as handled if it is rejected or classified correctly
            "robust_rate_detect_or_correct": float(((s > thr) | ~fooled).mean()),
        })
        a = out["attacks"][-1]
        print(f"[{tag}] {name}: det={a['detection_rate_all']:.3f} acc={a['acc_no_defense']:.3f} "
              f"acc_sq={a['acc_squeezed_input']:.3f} handled={a['robust_rate_detect_or_correct']:.3f}")
    print(f"[{tag}] threshold={thr:.4f} clean FPR={out['clean_test_fpr']:.3f}")
    return out


def main():
    x_tr, _, x_te, y_te = load_mnist()
    x_val = x_tr[-5000:]
    results = {m: run(f"{m}.pt", m, x_val, x_te, y_te) for m in ("baseline", "adv_trained")}
    save_json(results, "feature_squeeze.json")


if __name__ == "__main__":
    main()
