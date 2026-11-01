# Vision Transformer with LLS

## Implementation

This module applies blockwise LLS to a Vision Transformer for CIFAR images.
ViT-Tiny contains 12 pre-normalized encoder blocks, 192-dimensional tokens,
three attention heads, a 4x MLP ratio, and 4x4 image patches.

Each complete encoder block is one training unit. Layer normalization,
multi-head self-attention, the attention residual connection, the MLP, and the
second residual connection remain together. Gradient detachment happens only
between complete encoder sections.

For a local prediction, the LLS head concatenates the CLS representation with
the mean patch representation before applying the fixed basis and LLS-MxM
projection. The evaluated local model groups the encoder into six sections of
two blocks and uses a detached final linear classifier. It is compared with an
end-to-end backpropagation model of the same architecture.

Both models used CIFAR-10 augmentation, a 45k/5k training-validation split,
batch size 128, seed 0, AdamW Schedule-Free optimization, and the same Tesla
V100 GPU. Both completed 100 epochs.

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
`--data-dir`. The matched ViT-Tiny runs were produced on a Tesla V100 with:

```bash
python -m vit.run --arch vit_tiny --dataset cifar10aug --data-dir scalability --partition bp --lls-mode lls --readout linear --epochs 100 --device cuda --workers 4 --seed 0 --no-wandb

python -m vit.run --arch vit_tiny --dataset cifar10aug --data-dir scalability --partition uniform:2 --lls-mode lls-mxm --readout linear --epochs 100 --device cuda --workers 4 --seed 0 --no-wandb
```

Each experiment writes `config.json`, `metrics.csv`, `summary.json`, and its
checkpoint to a separate directory under `runs/`. The two runs should use the
same device and software environment for a matched comparison.

## Results

The following values are from the checkpoint selected by validation accuracy,
which was epoch 96 for both models.

| Method | Test accuracy | Peak GPU memory | Mean training time |
|---|---:|---:|---:|
| Backpropagation | 81.00% | 1368.2 MB | 19.56 s/epoch |
| Six-section LLS-MxM | 77.87% | 353.3 MB | 23.29 s/epoch |

The LLS-MxM model is 3.13 percentage points below backpropagation while using
74.2% less peak GPU memory. The current implementation takes 19.1% longer per
epoch, so the main observed benefit is training-memory reduction rather than
wall-clock speed.
