# Secure AI Systems: Red and Blue Teaming an MNIST CNN

A PyTorch CNN for MNIST, attacked (FGSM/PGD evasion and backdoor data poisoning) and defended
(PGD adversarial training, feature squeezing, activation-clustering data sanitisation).
See **[REPORT.md](REPORT.md)** for the threat model, SAST results, metrics and conclusions.

## Layout

| Path | What it does |
|---|---|
| `src/common.py` | CNN, MNIST loading, training loop (optionally adversarial), evaluation (accuracy, loss, confusion matrix, per-class report, inference time), integrity-checked model save/load |
| `src/train.py` | Trains the baseline model on clean MNIST |
| `src/attacks/evasion.py` | **Attack (OWASP ML01)**: FGSM and PGD adversarial test sets via IBM ART, eps 0.1/0.2/0.3 |
| `src/attacks/poison.py` | **Attack (OWASP ML02)**: backdoor poisoning (3x3 corner trigger, target label 7) at 0.1-5% poison rates |
| `src/defense/adv_train.py` | **Defense**: PGD adversarial training (50% clean / 50% adversarial per batch) |
| `src/defense/feature_squeeze.py` | **Defense**: feature-squeezing adversarial input detector |
| `src/defense/poison_filter.py` | **Defense**: activation clustering to find and drop poisoned training samples, then retrain |
| `src/summarize.py` | Builds `results/summary.md` and evaluates transfer attacks |
| `results/` | All metrics (JSON), confusion matrices, example images, SAST reports (`results/sast/`) |
| `models/` | Trained weights (safetensors) + `manifest.json` with SHA-256 hashes |

## Run

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash run_all.sh            # ~30-40 min on an Apple M2 (uses MPS / CUDA if available)
```

The generated adversarial and poisoned datasets used in the report are committed in
`data/adversarial/` and `data/poisoned/` (`.npz`: `x` images, `y` true labels; poisoned training
sets also hold `poison_idx`). Running the scripts again overwrites them.

## SAST

```bash
pip install bandit semgrep pip-audit
bandit -r src
semgrep scan --config p/python --config p/bandit --config p/trailofbits --config p/owasp-top-ten --config p/secrets --config p/default src
pip-audit -r requirements.txt
```
