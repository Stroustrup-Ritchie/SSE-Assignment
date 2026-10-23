"""Collects all results/*.json into results/summary.md (tables used in REPORT.md),
and evaluates transfer attacks (samples crafted on the baseline, replayed on the defended model)."""
import json
import os

import numpy as np

from common import DATA_DIR, RESULTS_DIR, evaluate, load_model, save_json


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def pct(v):
    return f"{100 * v:.2f}%"


def main():
    base, adv = load("baseline_metrics.json"), load("adv_trained_metrics.json")
    ev_b, ev_a = load("evasion_baseline.json"), load("evasion_adv_trained.json")
    fs, pb = load("feature_squeeze.json"), load("poison_backdoor.json")
    pf, pf_ica = load("poison_filter.json"), load("poison_filter_ica.json")

    # reference: how often the clean baseline already predicts the target on triggered images
    with np.load(os.path.join(DATA_DIR, "poisoned", "triggered_test.npz")) as d:
        base_trig = evaluate(load_model("baseline"), d["x"], d["y"], "baseline / triggered test")
    base_asr = float(np.array(base_trig["confusion_matrix"])[:, pb["target_label"]].sum() / base_trig["n"])

    adv_model = load_model("adv_trained")
    transfer = []
    for a in ev_b["attacks"]:
        with np.load(os.path.join(DATA_DIR, "adversarial", f"{a['attack']}_eps{a['eps']}_baseline.npz")) as d:
            r = evaluate(adv_model, d["x"], d["y"], f"adv_trained / transfer {a['attack']} eps={a['eps']}")
        transfer.append({"attack": a["attack"], "eps": a["eps"], "accuracy": r["accuracy"], "loss": r["loss"]})
    save_json(transfer, "transfer_to_adv_trained.json")

    L = ["## Clean test set", "", "| Model | Accuracy | Loss | Macro F1 | batch-1000 (ms, %s) | single image CPU (ms) |"
         % base["timing"]["device"], "|---|---|---|---|---|---|"]
    for name, r in (("Baseline", base), ("Adversarially trained", adv)):
        L.append(f"| {name} | {pct(r['accuracy'])} | {r['loss']:.4f} | {r['report']['macro avg']['f1-score']:.4f} | "
                 f"{r['timing']['batch1000_ms']} | {r['timing']['single_image_cpu_ms_median']} |")

    L += ["", "## Evasion attacks (accuracy / loss)", "",
          "| Attack | eps | Baseline (white-box) | Adv-trained (white-box, adaptive) | Adv-trained (transfer from baseline) |",
          "|---|---|---|---|---|"]
    for b, a, t in zip(ev_b["attacks"], ev_a["attacks"], transfer):
        L.append(f"| {b['attack'].upper()} | {b['eps']} | {pct(b['accuracy'])} / {b['loss']:.3f} | "
                 f"{pct(a['accuracy'])} / {a['loss']:.3f} | {pct(t['accuracy'])} / {t['loss']:.3f} |")

    L += ["", "## Feature squeezing detector (threshold at 5% FPR on held-out clean data)", "",
          "| Model | Attack set | Detection rate | Acc. no defense | Acc. on 1-bit squeezed input | Rejected-or-correct |",
          "|---|---|---|---|---|---|"]
    for m, r in fs.items():
        for a in r["attacks"]:
            L.append(f"| {m} | {a['set']} | {pct(a['detection_rate_all'])} | {pct(a['acc_no_defense'])} | "
                     f"{pct(a['acc_squeezed_input'])} | {pct(a['robust_rate_detect_or_correct'])} |")
        L.append(f"| {m} | clean test (FPR) | {pct(r['clean_test_fpr'])} | - | {pct(r['clean_acc_squeezed_input'])} | - |")
    eps_keys = list(next(iter(fs.values()))["adaptive_bpda_squeezed_acc"])
    L += ["", "### Adaptive attack on squeeze-then-classify (BPDA-PGD, 40 steps)", "",
          "| Model | " + " | ".join(f"eps {e}" for e in eps_keys) + " |", "|---" * (len(eps_keys) + 1) + "|"]
    for m, r in fs.items():
        L.append(f"| {m} + 1-bit squeeze | " + " | ".join(pct(r["adaptive_bpda_squeezed_acc"][e]) for e in eps_keys) + " |")

    L += ["", "## Backdoor poisoning (target label %d)" % pb["target_label"], "",
          "| Poison rate | # poisoned | Clean acc | Triggered acc (true label) | Attack success rate |", "|---|---|---|---|---|"]
    L.append(f"| 0 (clean baseline) | 0 | {pct(base['accuracy'])} | {pct(base_trig['accuracy'])} | {pct(base_asr)} |")
    for r in pb["runs"]:
        L.append(f"| {pct(r['poison_rate'])} | {r['n_poison']} | {pct(r['clean']['accuracy'])} | "
                 f"{pct(r['triggered']['accuracy'])} | {pct(r['attack_success_rate'])} |")

    L += ["", "## Activation-clustering defense (1% poisoning)", "",
          "| Variant | Classes flagged | Flagged | Poison found | Precision | Recall | Clean removed |",
          "|---|---|---|---|---|---|---|"]
    for name, r in (("ICA (paper default)", pf_ica), ("PCA", pf)):
        d = r["detection"]
        flagged_cls = ", ".join(str(c["class"]) for c in r["per_class"] if c["flagged"]) or "none"
        L.append(f"| {name} | {flagged_cls} | {d['n_flagged']} | {d['true_positives']}/{d['n_poison']} | "
                 f"{d['precision']:.3f} | {d['recall']:.3f} | {d['clean_samples_removed']} |")
    L += ["", "| Model | Clean acc | Triggered acc | Attack success rate |", "|---|---|---|---|",
          f"| Clean baseline (reference) | {pct(base['accuracy'])} | {pct(base_trig['accuracy'])} | {pct(base_asr)} |"]
    for r, k, label in ((pf, "before", "Backdoored (no defense)"),
                        (pf_ica, "after", "Retrained after ICA filtering"),
                        (pf, "after", "Retrained after PCA filtering")):
        L.append(f"| {label} | {pct(r[k]['clean']['accuracy'])} | {pct(r[k]['triggered']['accuracy'])} | "
                 f"{pct(r[k]['attack_success_rate'])} |")

    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
