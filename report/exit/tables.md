# Early exit: results (test; operating points chosen on val, cls final, fitted T)

## mobilenet: cheapest setting within δ = 1% of the full network

| split | full acc | exit acc | MAC frac | speedup | policy |
|---|---|---|---|---|---|
| 2,1×12 | 72.2 | 70.6 | 0.10 | 9.96 | margin (single, skip2) |
| 4,2,6,2 | 82.8 | 82.3 | 0.10 | 9.93 | margin (single, skip0) |
| 2,2,2,6,2 | 77.4 | 75.9 | 0.10 | 9.73 | margin (single, skip1) |
| 1×12,2 | 67.6 | 66.2 | 0.10 | 9.58 | sync_margin (ens2, skip3) |
| 1×10,4 | 67.9 | 66.4 | 0.11 | 9.14 | sync (ens, skip3) |
| LLS 1×14 | 67.6 | 67.2 | 0.12 | 8.40 | sync_margin (single, skip3) |
| 2,2,2,2,2,2,2 | 77.4 | 76.7 | 0.15 | 6.88 | sync (single, skip0) |
| single block | 82.6 | 82.6 | 1.00 | 1.00 | static (single, skip0) |

## mobilenet: overthinking

| split | exits | final | oracle | overthink (= oracle gap) | settle depth | flips |
|---|---|---|---|---|---|---|
| single block | 2 | 82.6 | 82.9 | 0.3 | 0.01 | 0.01 |
| 4,2,6,2 | 5 | 82.8 | 89.4 | 6.6 | 0.09 | 0.26 |
| 2,2,2,2,2,2,2 | 8 | 77.4 | 87.7 | 10.3 | 0.15 | 0.78 |
| 2,2,2,6,2 | 6 | 77.4 | 88.3 | 10.9 | 0.20 | 0.76 |
| 2,1×12 | 14 | 72.2 | 85.8 | 13.7 | 0.17 | 1.22 |
| 1×10,4 | 12 | 67.9 | 84.8 | 16.9 | 0.28 | 1.98 |
| LLS 1×14 | 15 | 67.6 | 84.9 | 17.2 | 0.23 | 2.05 |
| 1×12,2 | 14 | 67.6 | 85.2 | 17.6 | 0.25 | 2.02 |

## vgg8: cheapest setting within δ = 1% of the full network

| split | full acc | exit acc | MAC frac | speedup | policy |
|---|---|---|---|---|---|
| 2,2,1×3 | 89.0 | 88.0 | 0.57 | 1.74 | margin (single, skip0) |
| 2,2,2,1 | 89.2 | 88.4 | 0.58 | 1.73 | margin (ens2, skip0) |
| 2,2,3 | 89.7 | 88.7 | 0.59 | 1.69 | confidence (single, skip0) |
| LLS 1×7 | 85.9 | 85.2 | 0.60 | 1.67 | margin (single, skip0) |
| 3,3,1 | 90.0 | 89.2 | 0.68 | 1.48 | margin (ens2, skip0) |
| single block | 93.2 | 93.2 | 1.00 | 1.00 | static (single, skip0) |

## vgg8: overthinking

| split | exits | final | oracle | overthink (= oracle gap) | settle depth | flips |
|---|---|---|---|---|---|---|
| single block | 2 | 93.2 | 93.6 | 0.5 | 0.01 | 0.01 |
| 3,3,1 | 4 | 90.0 | 92.0 | 2.0 | 0.04 | 0.10 |
| 2,2,3 | 4 | 89.7 | 92.9 | 3.2 | 0.08 | 0.21 |
| 2,2,2,1 | 5 | 89.2 | 92.9 | 3.6 | 0.07 | 0.21 |
| 2,2,1×3 | 6 | 89.0 | 92.9 | 3.9 | 0.07 | 0.24 |
| LLS 1×7 | 8 | 85.9 | 92.9 | 7.0 | 0.17 | 0.87 |

## vgg8_cifar100: cheapest setting within δ = 1% of the full network

| split | full acc | exit acc | MAC frac | speedup | policy |
|---|---|---|---|---|---|
| 2,2,1×3 | 58.8 | 57.4 | 0.76 | 1.32 | confidence (ens, skip0) |
| LLS 1×7 | 54.5 | 53.4 | 0.80 | 1.25 | margin (ens2, skip0) |
| 2,5 | 67.3 | 65.6 | 0.84 | 1.19 | confidence (ens, skip0) |
| 1,1,5 | 63.7 | 62.3 | 0.91 | 1.10 | confidence (ens2, skip1) |

## vgg8_cifar100: overthinking

| split | exits | final | oracle | overthink (= oracle gap) | settle depth | flips |
|---|---|---|---|---|---|---|
| 2,5 | 3 | 67.3 | 73.6 | 6.4 | 0.34 | 0.66 |
| 1,1,5 | 4 | 63.7 | 70.4 | 6.7 | 0.59 | 1.69 |
| 2,2,1×3 | 6 | 58.8 | 72.8 | 14.0 | 0.37 | 1.37 |
| LLS 1×7 | 8 | 54.5 | 71.2 | 16.7 | 0.55 | 2.95 |

## Exit ensembles at full depth

| net | split | final | best ensemble | gain | window |
|---|---|---|---|---|---|
| mobilenet | LLS 1×14 | 67.6 | 68.4 | 0.7 | ens_w3 |
| mobilenet | 1×12,2 | 67.6 | 68.3 | 0.7 | ens_all |
| mobilenet | 1×10,4 | 67.9 | 68.4 | 0.5 | ens_w3 |
| mobilenet | 2,1×12 | 72.2 | 73.0 | 0.8 | ens_all |
| mobilenet | 2,2,2,2,2,2,2 | 77.4 | 78.1 | 0.7 | ens_all |
| mobilenet | 2,2,2,6,2 | 77.4 | 79.2 | 1.8 | ens_w2 |
| mobilenet | 4,2,6,2 | 82.8 | 83.9 | 1.1 | ens_all |
| vgg8 | LLS 1×7 | 85.9 | 86.1 | 0.1 | ens_w2 |
| vgg8 | 2,2,1×3 | 89.0 | 89.1 | 0.2 | ens_w3 |
| vgg8 | 2,2,2,1 | 89.2 | 89.2 | 0.0 | ens_w2 |
| vgg8 | 2,2,3 | 89.7 | 89.9 | 0.2 | ens_w2 |
| vgg8 | 3,3,1 | 90.0 | 90.2 | 0.1 | ens_w2 |

## VGG8 64 splits by first block size (δ = 1%)

| first block | splits | full acc | exit acc | MAC frac | train MB |
|---|---|---|---|---|---|
| 1.00 | 32.00 | 89.0 | 87.9 | 0.70 | 579.20 |
| 2.00 | 16.00 | 89.7 | 88.7 | 0.60 | 578.05 |
| 3.00 | 8.00 | 89.8 | 88.7 | 0.67 | 770.09 |
| 4.00 | 4.00 | 91.0 | 90.1 | 0.84 | 866.12 |
| 5.00 | 2.00 | 91.8 | 91.4 | 0.94 | 938.68 |
| 6.00 | 1.00 | 92.2 | 92.3 | 1.00 | 962.74 |
| 7.00 | 1.00 | 92.7 | 92.7 | 1.00 | 964.87 |


Cheapest 8:

| split | full acc | exit acc | MAC frac | train MB | policy |
|---|---|---|---|---|---|
| 2,1×3,2 | 88.9 | 87.4 | 0.55 | 578.05 | margin (single, skip0) |
| 2,1×5 | 88.0 | 86.9 | 0.55 | 578.05 | margin (single, skip0) |
| 2,1,1,2,1 | 88.7 | 88.0 | 0.57 | 578.05 | margin (single, skip0) |
| 2,1,1,3 | 88.9 | 87.9 | 0.57 | 578.05 | margin (single, skip0) |
| 2,1,2,2 | 89.3 | 88.4 | 0.58 | 578.05 | margin (single, skip0) |
| 2,1,2,1,1 | 89.0 | 88.1 | 0.58 | 578.05 | margin (single, skip0) |
| LLS 1×7 | 86.6 | 85.2 | 0.59 | 448.54 | confidence (single, skip1) |
| 2,2,1,2 | 88.9 | 87.9 | 0.59 | 578.05 | margin (ens, skip0) |

## Orthogonality of the fixed-basis heads (test)

Basis coherence = max |cos| between two class rows of a head's fixed basis. Class cosine = cosine between the two closest classes' centred mean head outputs (a perfect simplex gives -1/(C-1)).

| set | split | heads | basis coherence | class cos, first head | class cos, last head | last head acc |
|---|---|---|---|---|---|---|
| mobilenet | LLS 1×14 | 14 | 0.05 | 0.99 | 0.95 | 67.6 |
| mobilenet | 2,2,2,2,2,2,2 | 7 | 0.05 | 0.95 | 0.92 | 77.7 |
| mobilenet | 2,2,2,6,2 | 5 | 0.05 | 0.96 | 0.91 | 77.5 |
| mobilenet | single block | 1 | 0.05 | 0.75 | 0.75 | 82.6 |
| mobilenet | 1×12,2 | 13 | 0.05 | 0.99 | 0.95 | 67.5 |
| mobilenet | 1×10,4 | 11 | 0.05 | 0.99 | 0.95 | 68.1 |
| mobilenet | 2,1×12 | 13 | 0.05 | 0.96 | 0.94 | 72.3 |
| mobilenet | 4,2,6,2 | 4 | 0.05 | 0.89 | 0.85 | 82.7 |
| vgg8 | LLS 1×7 | 7 | 0.05 | 0.93 | 0.68 | 86.1 |
| vgg8 | 2,2,2,1 | 4 | 0.05 | 0.83 | 0.55 | 89.1 |
| vgg8 | 3,3,1 | 3 | 0.05 | 0.72 | 0.55 | 90.1 |
| vgg8 | 2,2,1×3 | 5 | 0.05 | 0.83 | 0.59 | 89.1 |
| vgg8 | 2,2,3 | 3 | 0.05 | 0.83 | 0.58 | 89.8 |
| vgg8 | single block | 1 | 0.05 | 0.55 | 0.55 | 93.2 |

## Learned vs fixed exits (δ = 1%)

| net | split | exits | full acc | exit acc | MAC frac | policy |
|---|---|---|---|---|---|---|
| mobilenet | 1×10,4 | fixed LLS heads | 67.9 | 66.4 | 0.11 | sync (ens, skip3) |
| mobilenet | 1×10,4 | linear probes (LLS backbone) | 67.9 | 66.4 | 0.05 | margin (single, skip2) |
| mobilenet | 1×12,2 | fixed LLS heads | 67.6 | 66.2 | 0.10 | sync_margin (ens2, skip3) |
| mobilenet | 1×12,2 | linear probes (LLS backbone) | 67.6 | 67.8 | 0.06 | sync (ens, skip0) |
| mobilenet | 2,1×12 | fixed LLS heads | 72.2 | 70.6 | 0.10 | margin (single, skip2) |
| mobilenet | 2,1×12 | linear probes (LLS backbone) | 72.2 | 70.5 | 0.04 | margin (ens2, skip1) |
| mobilenet | 2,2,2,2,2,2,2 | fixed LLS heads | 77.4 | 76.7 | 0.15 | sync (single, skip0) |
| mobilenet | 2,2,2,2,2,2,2 | linear probes (LLS backbone) | 77.4 | 76.1 | 0.08 | margin (single, skip1) |
| mobilenet | 2,2,2,6,2 | fixed LLS heads | 77.4 | 75.9 | 0.10 | margin (single, skip1) |
| mobilenet | 2,2,2,6,2 | linear probes (LLS backbone) | 77.4 | 75.4 | 0.07 | static (single, skip0) |
| mobilenet | 4,2,6,2 | fixed LLS heads | 82.8 | 82.3 | 0.10 | margin (single, skip0) |
| mobilenet | 4,2,6,2 | linear probes (LLS backbone) | 82.8 | 82.2 | 0.08 | confidence (single, skip0) |
| mobilenet | BP (exit per layer) | BP + frozen exits | 84.2 | 82.8 | 0.10 | margin (ens2, skip3) |
| mobilenet | BP (exit per layer) | BP + joint exits | 86.5 | 85.6 | 0.09 | margin (ens2, skip3) |
| mobilenet | LLS 1×14 | fixed LLS heads | 67.6 | 67.2 | 0.12 | sync_margin (single, skip3) |
| mobilenet | LLS 1×14 | linear probes (LLS backbone) | 67.6 | 67.5 | 0.07 | margin (single, skip2) |
| vgg8 | 2,2,1×3 | fixed LLS heads | 89.0 | 88.0 | 0.57 | margin (single, skip0) |
| vgg8 | 2,2,1×3 | linear probes (LLS backbone) | 89.0 | 87.8 | 0.56 | margin (single, skip0) |
| vgg8 | 2,2,2,1 | fixed LLS heads | 89.2 | 88.4 | 0.58 | margin (ens2, skip0) |
| vgg8 | 2,2,2,1 | linear probes (LLS backbone) | 89.2 | 87.8 | 0.56 | confidence (ens, skip0) |
| vgg8 | 2,2,3 | fixed LLS heads | 89.7 | 88.7 | 0.59 | confidence (single, skip0) |
| vgg8 | 2,2,3 | linear probes (LLS backbone) | 89.7 | 88.7 | 0.59 | margin (single, skip0) |
| vgg8 | 3,3,1 | fixed LLS heads | 90.0 | 89.2 | 0.68 | margin (ens2, skip0) |
| vgg8 | 3,3,1 | linear probes (LLS backbone) | 90.0 | 89.3 | 0.67 | confidence (ens, skip0) |
| vgg8 | BP (exit per layer) | BP + frozen exits | 93.3 | 92.1 | 0.70 | margin (ens2, skip0) |
| vgg8 | BP (exit per layer) | BP + joint exits | 93.0 | 91.8 | 0.52 | margin (single, skip0) |
| vgg8 | LLS 1×7 | fixed LLS heads | 85.9 | 85.2 | 0.60 | margin (single, skip0) |
| vgg8 | LLS 1×7 | linear probes (LLS backbone) | 85.9 | 85.2 | 0.55 | entropy (single, skip0) |

## VGG8 seeds (60 ep; s0 from the 64-split sweep), δ = 1%: mean (sd)

| split | seeds | full acc | exit acc | MAC frac |
|---|---|---|---|---|
| LLS 1×7 | 3 | 86.5 (0.1) | 85.3 (0.2) | 0.602 (0.011) |
| 2,5 | 3 | 91.4 (0.2) | 90.4 (0.1) | 0.651 (0.004) |
| 1,1,5 | 3 | 90.1 (0.1) | 88.9 (0.3) | 0.695 (0.014) |
| single block | 3 | 92.5 (0.3) | 92.5 (0.3) | 1.000 (0.000) |

## Spearman(k-NN prediction depth, exit depth), test; policy exits at their δ = 1% picks

| net | split | settle | oracle | any | margin | confidence | entropy | patience | sync | sync_margin |
|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet | LLS 1×14 | 0.62 | 0.33 | 0.36 | 0.34 | – | – | 0.45 | 0.46 | 0.36 |
| mobilenet | 2,2,2,2,2,2,2 | 0.60 | 0.42 | 0.39 | – | – | – | 0.40 | 0.39 | 0.39 |
| mobilenet | 2,2,2,6,2 | 0.59 | 0.39 | 0.30 | 0.30 | 0.37 | – | 0.36 | 0.28 | 0.28 |
| mobilenet | single block | 0.27 | 0.23 | – | -0.00 | – | 0.05 | – | – | – |
| mobilenet | 1×12,2 | 0.62 | 0.33 | 0.34 | 0.39 | 0.37 | – | 0.34 | 0.35 | 0.34 |
| mobilenet | 1×10,4 | 0.62 | 0.33 | 0.29 | 0.29 | 0.34 | 0.02 | 0.33 | 0.33 | 0.33 |
| mobilenet | 2,1×12 | 0.64 | 0.38 | 0.39 | 0.39 | 0.37 | – | 0.38 | 0.45 | 0.46 |
| mobilenet | 4,2,6,2 | 0.56 | 0.40 | 0.35 | 0.35 | 0.39 | – | 0.40 | 0.02 | 0.02 |
| vgg8 | LLS 1×7 | 0.60 | 0.47 | 0.59 | 0.59 | 0.56 | 0.59 | 0.55 | 0.58 | 0.62 |
| vgg8 | 2,2,2,1 | 0.48 | 0.42 | 0.48 | 0.48 | 0.49 | 0.49 | 0.39 | 0.40 | 0.43 |
| vgg8 | 3,3,1 | 0.42 | 0.41 | 0.44 | 0.44 | 0.45 | 0.50 | 0.37 | 0.01 | 0.02 |
| vgg8 | 2,2,1×3 | 0.51 | 0.43 | 0.50 | 0.50 | 0.51 | 0.51 | 0.41 | 0.02 | 0.02 |
| vgg8 | 2,2,3 | 0.48 | 0.42 | 0.50 | 0.51 | 0.50 | 0.52 | 0.38 | 0.35 | 0.41 |
| vgg8 | single block | 0.31 | 0.29 | – | -0.00 | – | – | – | – | – |
