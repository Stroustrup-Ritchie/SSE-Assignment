"""Red team attack #3 (OWASP ML02 - Data Poisoning): backdoor / trojan via training-set poisoning.

A small fraction of non-target training images get a 3x3 corner pattern (ART add_pattern_bd)
and are relabelled to TARGET. The model learns "pattern => TARGET" while staying accurate on
clean data, so the attack is invisible to ordinary accuracy monitoring.
"""
import argparse
import os
import sys

import numpy as np
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (DATA_DIR, RESULTS_DIR, MnistCNN, evaluate, load_mnist, plot_confusion, predict,  # noqa: E402
                    save_json, save_model, set_seed, train_model)

POISON_DIR = os.path.join(DATA_DIR, "poisoned")
TARGET = 7
RATES = [0.001, 0.005, 0.01, 0.05]
MAIN_RATE = 0.01


def add_trigger(x):
    """x: (N,1,28,28) in [0,1] -> same shape with the white 3x3 pattern in the bottom-right corner."""
    return add_pattern_bd(x[:, 0].copy(), pixel_value=1.0)[:, None].astype(np.float32)


def poison_training_set(x, y, rate, rng):
    backdoor = PoisoningAttackBackdoor(add_trigger)
    candidates = np.where(y != TARGET)[0]
    idx = rng.choice(candidates, size=int(rate * len(x)), replace=False)
    x_p, y_p = x.copy(), y.copy()
    x_p[idx], _ = backdoor.poison(x[idx], y=np.full(len(idx), TARGET))
    y_p[idx] = TARGET
    return x_p, y_p, idx


def triggered_test_set(x, y):
    keep = y != TARGET
    return add_trigger(x[keep]), y[keep]


def save_trigger_examples(x, model, fname="examples_backdoor.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    trig = add_trigger(x)
    rows = (("clean", x, predict(model, x).argmax(1).tolist()),
            ("triggered", trig, predict(model, trig).argmax(1).tolist()))
    fig, axes = plt.subplots(2, len(x), figsize=(1.25 * len(x), 3))
    for r, (label, xs, preds) in enumerate(rows):
        for c in range(len(x)):
            axes[r, c].imshow(xs[c, 0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].set_title(f"pred {preds[c]}", fontsize=7)
        axes[r, 0].set_ylabel(label, fontsize=8)
    fig.suptitle("Backdoored model: trigger (3x3 bottom-right) flips every digit to 7", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=110)
    plt.close(fig)


def backdoor_metrics(model, x_te, y_te, tag):
    clean = evaluate(model, x_te, y_te, f"{tag} / clean test")
    x_trig, y_trig = triggered_test_set(x_te, y_te)
    trig = evaluate(model, x_trig, y_trig, f"{tag} / triggered test (true labels)")
    asr = float((predict(model, x_trig).argmax(1).numpy() == TARGET).mean())
    return {"clean": clean, "triggered": trig, "attack_success_rate": asr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(POISON_DIR, exist_ok=True)
    x_tr, y_tr, x_te, y_te = load_mnist()
    x_trig, y_trig = triggered_test_set(x_te, y_te)
    np.savez_compressed(os.path.join(POISON_DIR, "triggered_test.npz"), x=x_trig, y=y_trig)

    results = {"target_label": TARGET, "runs": []}
    for rate in RATES:
        set_seed()
        rng = np.random.default_rng(0)
        x_p, y_p, idx = poison_training_set(x_tr, y_tr, rate, rng)
        np.savez_compressed(os.path.join(POISON_DIR, f"backdoor_train_p{rate}.npz"), x=x_p, y=y_p, poison_idx=idx)

        model = MnistCNN()
        train_model(model, x_p, y_p, epochs=args.epochs)
        tag = f"poisoned p={rate}"
        m = backdoor_metrics(model, x_te, y_te, tag)
        m.update({"poison_rate": rate, "n_poison": int(len(idx))})
        results["runs"].append(m)
        print(f"rate={rate}: clean acc={m['clean']['accuracy']:.4f} "
              f"triggered acc={m['triggered']['accuracy']:.4f} ASR={m['attack_success_rate']:.4f}")
        if rate == MAIN_RATE:
            save_model(model, "poisoned", {"poison_rate": rate, "target": TARGET})
            plot_confusion(m["triggered"]["confusion_matrix"], f"Backdoored model (p={rate}) - triggered test",
                           "cm_poisoned_triggered.png")
            save_trigger_examples(x_te[y_te != TARGET][:8], model)
    save_json(results, "poison_backdoor.json")


if __name__ == "__main__":
    main()
