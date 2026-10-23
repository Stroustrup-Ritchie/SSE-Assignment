#!/usr/bin/env bash
# Reproduces every number in REPORT.md. ~30-40 min on an Apple M2.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}

$PY src/train.py                                              # baseline + clean metrics
$PY src/attacks/evasion.py --model baseline --tag baseline # ML01: FGSM / PGD
$PY src/attacks/poison.py                                     # ML02: backdoor poisoning
$PY src/defense/adv_train.py                                  # defense 1: adversarial training
$PY src/attacks/evasion.py --model adv_trained --tag adv_trained  # adaptive re-attack
$PY src/defense/feature_squeeze.py                            # defense 2: input squeezing detector
$PY src/defense/poison_filter.py --reducer ica                # defense 3a: activation clustering, ICA (paper default; fails here)
$PY src/defense/poison_filter.py                              # defense 3b: activation clustering, PCA
$PY src/summarize.py                                          # tables -> results/summary.md
