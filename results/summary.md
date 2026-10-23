## Clean test set

| Model | Accuracy | Loss | Macro F1 | batch-1000 (ms, mps) | single image CPU (ms) |
|---|---|---|---|---|---|
| Baseline | 99.03% | 0.0295 | 0.9902 | 23.24 | 0.196 |
| Adversarially trained | 98.45% | 0.0501 | 0.9844 | 28.5 | 0.209 |

## Evasion attacks (accuracy / loss)

| Attack | eps | Baseline (white-box) | Adv-trained (white-box, adaptive) | Adv-trained (transfer from baseline) |
|---|---|---|---|---|
| FGSM | 0.1 | 90.53% / 0.277 | 97.61% / 0.075 | 97.86% / 0.071 |
| FGSM | 0.2 | 45.52% / 1.862 | 96.10% / 0.111 | 96.71% / 0.103 |
| FGSM | 0.3 | 14.82% / 4.350 | 94.06% / 0.165 | 94.33% / 0.170 |
| PGD | 0.1 | 80.28% / 0.621 | 97.34% / 0.080 | 97.86% / 0.069 |
| PGD | 0.2 | 2.06% / 8.096 | 95.04% / 0.135 | 96.89% / 0.097 |
| PGD | 0.3 | 0.64% / 20.376 | 90.36% / 0.255 | 95.00% / 0.147 |

## Feature squeezing detector (threshold at 5% FPR on held-out clean data)

| Model | Attack set | Detection rate | Acc. no defense | Acc. on 1-bit squeezed input | Rejected-or-correct |
|---|---|---|---|---|---|
| baseline | fgsm_eps0.1 | 33.29% | 90.53% | 97.29% | 99.90% |
| baseline | fgsm_eps0.2 | 84.81% | 45.52% | 93.79% | 99.16% |
| baseline | fgsm_eps0.3 | 93.58% | 14.82% | 86.00% | 97.99% |
| baseline | pgd_eps0.1 | 50.11% | 80.28% | 97.27% | 99.88% |
| baseline | pgd_eps0.2 | 98.41% | 2.06% | 93.73% | 98.60% |
| baseline | pgd_eps0.3 | 95.84% | 0.64% | 86.43% | 96.22% |
| baseline | clean test (FPR) | 5.88% | - | 98.72% | - |
| adv_trained | fgsm_eps0.1 | 10.89% | 97.61% | 97.72% | 99.73% |
| adv_trained | fgsm_eps0.2 | 14.68% | 96.10% | 96.33% | 99.38% |
| adv_trained | fgsm_eps0.3 | 20.18% | 94.06% | 94.73% | 98.77% |
| adv_trained | pgd_eps0.1 | 11.64% | 97.34% | 97.76% | 99.75% |
| adv_trained | pgd_eps0.2 | 17.04% | 95.04% | 96.26% | 99.36% |
| adv_trained | pgd_eps0.3 | 27.20% | 90.36% | 93.73% | 98.87% |
| adv_trained | clean test (FPR) | 5.81% | - | 98.41% | - |

### Adaptive attack on squeeze-then-classify (BPDA-PGD, 40 steps)

| Model | eps 0.1 | eps 0.2 | eps 0.3 | eps 0.4 |
|---|---|---|---|---|
| baseline + 1-bit squeeze | 96.99% | 92.84% | 81.48% | 51.87% |
| adv_trained + 1-bit squeeze | 97.72% | 96.65% | 95.06% | 90.95% |

## Backdoor poisoning (target label 7)

| Poison rate | # poisoned | Clean acc | Triggered acc (true label) | Attack success rate |
|---|---|---|---|---|
| 0 (clean baseline) | 0 | 99.03% | 99.02% | 0.13% |
| 0.10% | 60 | 98.99% | 0.72% | 99.28% |
| 0.50% | 300 | 98.90% | 0.02% | 99.98% |
| 1.00% | 600 | 99.06% | 0.01% | 99.99% |
| 5.00% | 3000 | 98.85% | 0.00% | 100.00% |

## Activation-clustering defense (1% poisoning)

| Variant | Classes flagged | Flagged | Poison found | Precision | Recall | Clean removed |
|---|---|---|---|---|---|---|
| ICA (paper default) | 1 | 2160 | 0/600 | 0.000 | 0.000 | 2160 |
| PCA | 7 | 598 | 598/600 | 1.000 | 0.997 | 0 |

| Model | Clean acc | Triggered acc | Attack success rate |
|---|---|---|---|
| Clean baseline (reference) | 99.03% | 99.02% | 0.13% |
| Backdoored (no defense) | 99.06% | 0.01% | 99.99% |
| Retrained after ICA filtering | 98.95% | 0.00% | 100.00% |
| Retrained after PCA filtering | 98.97% | 98.94% | 0.19% |
