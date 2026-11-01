# ResNet Scalability with LLS

## Implementation

This module applies LLS to CIFAR ResNets. A complete residual block is treated
as one training unit, including both convolutions, normalization, activation,
shortcut projection, and residual addition. This keeps each skip connection
inside a section and places gradient boundaries only between complete blocks.

The implementation supports ResNet-20, ResNet-32, ResNet-56, and a wider
ResNet-20 with 32/64/128 channels. It supports end-to-end backpropagation,
fixed LLS, LLS-M, and LLS-MxM. Units can be trained independently or grouped
into larger local sections. A pure LLS readout is also supported, allowing the
last LLS head to make the final prediction without a detached linear
classifier.

The experiments used CIFAR-10 augmentation, a 45k/5k training-validation
split, batch size 128, seed 0, AdamW Schedule-Free optimization, and model
selection based on validation accuracy.

## Reproducing the results

Check out the repository and install the dependencies:

```bash
git clone https://github.com/Yukash09/CS6886_Project.git
cd CS6886_Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically into the directory passed through
`--data-dir`. The saved scalability results were produced on Apple MPS with
the following commands:

```bash
python -m scalability.run --arch resnet20 --dataset cifar10aug --data-dir scalability --partition bp --epochs 100 --patience 10 --device mps --workers 4 --seed 0 --no-wandb

python -m scalability.run --arch resnet20 --dataset cifar10aug --data-dir scalability --partition uniform:1 --epochs 100 --patience 10 --device mps --workers 4 --seed 0 --no-wandb

python -m scalability.run --arch resnet20 --dataset cifar10aug --data-dir scalability --partition uniform:1 --lls-mode lls-mxm --epochs 100 --patience 10 --device mps --workers 4 --seed 0 --no-wandb

python -m scalability.run --arch resnet20 --dataset cifar10aug --data-dir scalability --partition aligned --lls-mode lls-mxm --epochs 200 --device mps --workers 4 --seed 0 --no-wandb

python -m scalability.run --arch wide_resnet20 --dataset cifar10aug --data-dir scalability --partition bp --epochs 300 --device mps --workers 4 --seed 0 --no-wandb

python -m scalability.run --arch wide_resnet20 --dataset cifar10aug --data-dir scalability --partition 7,3 --lls-mode lls-mxm --readout lls --epochs 300 --device mps --workers 4 --seed 0 --no-wandb
```

Each experiment writes `config.json`, `metrics.csv`, `summary.json`, and its
checkpoint to a separate directory under `runs/`. Use `--device cuda` on a
CUDA system or `--device auto` to select an available backend automatically.

## Results

The following values are test accuracy at the validation-selected checkpoint.
Peak memory is the analytic peak activation memory for training.

| Network | Method | Completed epochs | Test accuracy | Peak memory |
|---|---|---:|---:|---:|
| ResNet-20 | Backpropagation | 48 | 88.31% | 281.5 MB |
| ResNet-20 | Per-unit LLS | 57 | 51.42% | 56.5 MB |
| ResNet-20 | Per-unit LLS-MxM | 57 | 56.05% | 56.5 MB |
| ResNet-20 | Aligned 4-3-3 LLS-MxM | 200 | 71.88% | 170.0 MB |
| Wide-ResNet-20 | Backpropagation | 300 | 92.36% | 561.6 MB |
| Wide-ResNet-20 | 7-3 LLS-MxM with LLS readout | 300 | 89.83% | 490.0 MB |

Grouping ResNet-20 into resolution-aligned sections improved LLS-MxM by 15.83
percentage points compared with per-unit training. The matched Wide-ResNet-20
experiment reduced the gap to backpropagation to 2.53 points while reducing
peak activation memory by 12.7%. Its analytic training cost was 471.6 MMAC,
compared with 486.2 MMAC for backpropagation.
