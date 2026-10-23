"""Blue team defense #3: Activation Clustering (Chen et al., 2018) against the backdoor.

For each label, take the poisoned model's penultimate-layer activations on the (untrusted)
training data, reduce to 10 dims with ICA, and split into 2 clusters with k-means. Backdoored
samples share the trigger feature, so in the target class they form a small, well-separated
cluster. A class is judged poisoned if its smaller cluster holds < 35% of the class AND the
silhouette score is high; that cluster is removed and a fresh model is retrained on the rest.
"""
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.poison import MAIN_RATE, POISON_DIR, backdoor_metrics  # noqa: E402
from common import (MnistCNN, get_device, load_mnist, load_model, save_json, save_model,  # noqa: E402
                    set_seed, train_model)

SIZE_THRESHOLD = 0.35
SILHOUETTE_THRESHOLD = 0.10


@torch.no_grad()
def penultimate(model, x, batch_size=1000):
    device = get_device()
    model.to(device).eval()
    feats = [model.features(torch.from_numpy(x[i:i + batch_size]).to(device)).cpu()
             for i in range(0, len(x), batch_size)]
    return torch.cat(feats).numpy()


def activation_clustering(feats, y):
    flagged = np.zeros(len(y), dtype=bool)
    per_class = []
    for c in range(10):
        idx = np.where(y == c)[0]
        reduced = FastICA(n_components=10, random_state=0, max_iter=500).fit_transform(feats[idx])
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(reduced)
        sizes = np.bincount(labels, minlength=2)
        small = int(np.argmin(sizes))
        rel = float(sizes[small] / len(idx))
        sil = float(silhouette_score(reduced, labels, sample_size=min(3000, len(idx)), random_state=0))
        poisoned = rel < SIZE_THRESHOLD and sil > SILHOUETTE_THRESHOLD
        if poisoned:
            flagged[idx[labels == small]] = True
        per_class.append({"class": c, "smaller_cluster_rel_size": rel, "silhouette": sil, "flagged": poisoned})
        print(f"class {c}: rel_size={rel:.3f} silhouette={sil:.3f} poisoned={poisoned}")
    return flagged, per_class


def main():
    set_seed()
    _, _, x_te, y_te = load_mnist()
    with np.load(os.path.join(POISON_DIR, f"backdoor_train_p{MAIN_RATE}.npz")) as d:
        x_p, y_p, poison_idx = d["x"], d["y"], d["poison_idx"]
    is_poison = np.zeros(len(y_p), dtype=bool)
    is_poison[poison_idx] = True

    poisoned_model = load_model("poisoned.pt")
    flagged, per_class = activation_clustering(penultimate(poisoned_model, x_p), y_p)
    tp = int((flagged & is_poison).sum())
    detection = {"n_poison": int(is_poison.sum()), "n_flagged": int(flagged.sum()), "true_positives": tp,
                 "precision": tp / max(1, int(flagged.sum())), "recall": tp / int(is_poison.sum()),
                 "clean_samples_removed": int((flagged & ~is_poison).sum())}
    print(detection)

    model = MnistCNN()
    train_model(model, x_p[~flagged], y_p[~flagged], epochs=6)
    save_model(model, "poison_filtered.pt", {"defense": "activation clustering", **detection})
    results = {"detection": detection, "per_class": per_class,
               "before": backdoor_metrics(poisoned_model, x_te, y_te, "poisoned (no defense)"),
               "after": backdoor_metrics(model, x_te, y_te, "retrained after filtering")}
    for k in ("before", "after"):
        print(f"{k}: clean acc={results[k]['clean']['accuracy']:.4f} ASR={results[k]['attack_success_rate']:.4f}")
    save_json(results, "poison_filter.json")


if __name__ == "__main__":
    main()
