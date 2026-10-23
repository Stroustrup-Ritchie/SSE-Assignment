"""Red team attack #1 and #2 (OWASP ML01 - Input Manipulation): white-box FGSM and PGD
evasion samples generated with IBM's Adversarial Robustness Toolbox (ART).

Writes the adversarial test sets to data/adversarial/ and their metrics to results/.
"""
import argparse
import os
import sys

import numpy as np
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (DATA_DIR, RESULTS_DIR, evaluate, load_mnist, load_model, plot_confusion,  # noqa: E402
                    predict, save_json, set_seed)

ADV_DIR = os.path.join(DATA_DIR, "adversarial")
EPSILONS = [0.1, 0.2, 0.3]


def art_classifier(model):
    model.to("cpu").eval()
    return PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(), input_shape=(1, 28, 28),
                             nb_classes=10, clip_values=(0.0, 1.0), device_type="cpu")


def make_attack(name, clf, eps):
    if name == "fgsm":
        return FastGradientMethod(clf, eps=eps, batch_size=500)
    return ProjectedGradientDescent(clf, norm=np.inf, eps=eps, eps_step=eps / 8, max_iter=20,
                                    num_random_init=1, batch_size=500, verbose=False)


def save_examples(x_clean, advs, y, preds, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("clean", x_clean)] + list(advs.items())
    fig, axes = plt.subplots(len(rows), 8, figsize=(10, 1.4 * len(rows)))
    for r, (label, xs) in enumerate(rows):
        for c in range(8):
            ax = axes[r, c]
            ax.imshow(xs[c, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"pred {preds[label][c]}", fontsize=7)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle(f"true labels: {list(y[:8])}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline.pt", help="target model in models/")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    set_seed()
    os.makedirs(ADV_DIR, exist_ok=True)
    _, _, x_te, y_te = load_mnist()
    model = load_model(args.model)
    clf = art_classifier(model)

    results = {"target_model": args.model, "attacks": []}
    examples, example_preds = {}, {"clean": evaluate_preds(model, x_te[:8])}
    for name in ("fgsm", "pgd"):
        for eps in EPSILONS:
            x_adv = make_attack(name, clf, eps).generate(x=x_te).astype(np.float32)
            res = evaluate(model, x_adv, y_te, f"{args.tag} / {name} eps={eps}")
            res.update({"attack": name, "eps": eps,
                        "linf": float(np.abs(x_adv - x_te).max()),
                        "mean_l2": float(np.linalg.norm((x_adv - x_te).reshape(len(x_te), -1), axis=1).mean())})
            results["attacks"].append(res)
            print(f"{name} eps={eps}: acc={res['accuracy']:.4f} loss={res['loss']:.4f}")
            np.savez_compressed(os.path.join(ADV_DIR, f"{name}_eps{eps}_{args.tag}.npz"), x=x_adv, y=y_te)
            if eps == 0.3:
                plot_confusion(res["confusion_matrix"], f"{args.tag} - {name.upper()} eps=0.3",
                               f"cm_{args.tag}_{name}_eps0.3.png")
                examples[f"{name} 0.3"] = x_adv[:8]
                example_preds[f"{name} 0.3"] = evaluate_preds(model, x_adv[:8])
            # keep the model on CPU for ART after evaluate() moved it
            model.to("cpu")
    save_json(results, f"evasion_{args.tag}.json")
    save_examples(x_te[:8], examples, y_te, example_preds, f"examples_evasion_{args.tag}.png")


def evaluate_preds(model, x):
    out = predict(model, x).argmax(1).tolist()
    model.to("cpu")
    return out


if __name__ == "__main__":
    main()
