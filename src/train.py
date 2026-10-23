"""Task 1-2: train the baseline CNN on clean MNIST and report clean-test metrics."""
import argparse

from common import (MnistCNN, evaluate, load_mnist, plot_confusion, save_json, save_model,
                    set_seed, train_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    set_seed()
    x_tr, y_tr, x_te, y_te = load_mnist()
    model = MnistCNN()
    history = train_model(model, x_tr, y_tr, epochs=args.epochs)
    save_model(model, "baseline", {"epochs": args.epochs, "data": "clean MNIST"})

    res = evaluate(model, x_te, y_te, "baseline / clean test", with_timing=True)
    res["train_history"] = history
    save_json(res, "baseline_metrics.json")
    plot_confusion(res["confusion_matrix"], "Baseline CNN - clean test", "cm_baseline_clean.png")
    print(f"clean test acc={res['accuracy']:.4f} loss={res['loss']:.4f} timing={res['timing']}")


if __name__ == "__main__":
    main()
