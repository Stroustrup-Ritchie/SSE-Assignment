"""Blue team defense #1: PGD adversarial training (Madry et al.).

Every mini-batch is half clean, half L-inf PGD adversarial examples crafted against the current
weights, so the model is trained on adversarial samples together with clean data.
"""
import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (MnistCNN, evaluate, load_mnist, pgd_torch, plot_confusion, save_json,  # noqa: E402
                    save_model, set_seed, train_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    set_seed()
    x_tr, y_tr, x_te, y_te = load_mnist()
    model = MnistCNN()
    adv_fn = functools.partial(pgd_torch, eps=args.eps, alpha=2.5 * args.eps / args.steps, steps=args.steps)
    history = train_model(model, x_tr, y_tr, epochs=args.epochs, adv_fn=adv_fn)
    save_model(model, "adv_trained", {"epochs": args.epochs, "pgd_eps": args.eps, "pgd_steps": args.steps})

    res = evaluate(model, x_te, y_te, "adv_trained / clean test", with_timing=True)
    res["train_history"] = history
    res["config"] = vars(args)
    save_json(res, "adv_trained_metrics.json")
    plot_confusion(res["confusion_matrix"], "Adversarially trained CNN - clean test", "cm_adv_trained_clean.png")
    print(f"clean test acc={res['accuracy']:.4f} loss={res['loss']:.4f}")


if __name__ == "__main__":
    main()
