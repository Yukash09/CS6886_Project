# Exploring LLS: Granularity, Early Exit and Scaling LLS

CS6886 (Systems Engineering for Deep Learning), IIT Madras. Yukash Ram (CS23B069), Hayagriv Venkatesh (ED23B024).

LLS (Apolinario et al., WACV 2025) is a local learning rule inspired by neural activity synchronization: each layer is
trained with its own loss to make its activity follow a fixed periodic pattern whose frequency identifies the class.
Since no error crosses layers, training needs no end-to-end backprop and much less activation memory, but per-layer
LLS trails BP by several points. This project studies how far that gap can be closed and what else the local heads
are good for, through four questions:

1. Granularity:\
   If a block of layers shares one LLS head, with BP inside the block and LLS between blocks, which grouping gives the
   best accuracy for a given training memory?
2. Early exit:\
   Since every LLS head already predicts the class, can these heads serve as free exits at inference, and are they
   interpretable?
3. Scalability:\
   Does LLS scale beyond VGG8 and MobileNetV1 to deeper residual networks (ResNet-20, Wide-ResNet-20)?
4. Attention:\
   Does LLS work on self-attention, i.e. on a Vision Transformer (ViT-Tiny)?

## Main results

Merging the first layers closes 89% of the LLS–BP accuracy gap on VGG8 at 1.29× the per-layer LLS memory, and 94% on
MobileNetV1 at 2.56×, where BP needs 8.21×. Used as exits, the same heads cut inference MACs by 85–90% on MobileNetV1
and 32–43% on VGG8 within one point of full accuracy. The lesson carries over to larger models: a two-block
Wide-ResNet-20 is 2.5 points below BP with 12.7% less memory, and a six-block ViT-Tiny is 3.1 points below BP with 74%
less memory.

## Reports

[Report](docs/short_report.pdf) · [Detailed report](docs/report.pdf)

The upstream LLS code is included unmodified in `third_party/LLS-DNN` (MIT).

## Setup
```bash
pip install -r requirements.txt
export WANDB_API_KEY=...                # optional; every run also logs to local files
export LLS_DATA=./data LLS_OUT=./runs   # optional; these are the defaults
python -m pytest -q                     # CPU unit tests
```
CIFAR10/100 are downloaded to `$LLS_DATA` on first use, and every command below skips finished runs and resumes
interrupted ones, so it is safe to re-run.

## Reproducing the results

### 1. Granularity (`lls_granular`)

#### A single run

`lls_granular.run` trains one network with one partition; the arguments below select the network, the data and how the
layers are grouped into blocks.
```bash
python -m lls_granular.run --arch vgg8 --dataset cifar10aug --partition 2,5 --epochs 60
```

| Argument | Values | Meaning |
|---|---|---|
| `--arch` | `vgg8` (7 layers), `mobilenet` (MobileNetV1, 14 layers) | network |
| `--dataset` | `cifar10aug` (default), `cifar10` (no augmentation), `cifar100aug` | dataset |
| `--partition` | `uniform:1` (default) | per-layer LLS: one head per layer, as in the paper |
| | `uniform:k` | blocks of `k` consecutive layers; the remainder forms the last block |
| | `aligned` | cut wherever the spatial resolution changes (at the pooling layers) |
| | sizes, e.g. `2,5` or `4,2,1,1,1,1,1,1,1,1` | explicit block sizes; they must sum to the number of layers |
| | `uniform:7` (VGG8) / `uniform:14` (MobileNet) | one block over the whole network (one LLS head at the end) |
| | `bp` | plain backprop baseline |
| `--epochs` | default 100 | training epochs |
| `--seed` | default 0 | random seed |
| `--lr`, `--wd`, `--optim` | `5e-3`, `0`, `AdamWSF` | schedule-free AdamW (as in the paper) |
| `--wave`, `--temp` | `square` / `cosine`, `0.5` | LLS basis shape and softmax temperature |
| `--bs` | default 128 | batch size |
| `--limit N` | | train on N images only (smoke test) |
| `--no-wandb`, `--fresh` | | disable wandb; restart the run from scratch |

Each run writes its config, per-epoch metrics, summary, log and checkpoint to `runs/<run>/` and logs to the wandb
project `lls-granularity` under the name `{arch}_{dataset}_{p<blocks>|bp}_e{epochs}_s{seed}`.

#### All experiments in the report

`lls_granular.launch` runs every run of a named experiment on all visible GPUs, one run per GPU, and prints a results
table at the end; several experiments can share one queue, e.g. `python -m lls_granular.launch cifar100 seeds`.

| Command | Runs |
|---|---|
| `python -m lls_granular.launch smoke` | 2-epoch sanity check on 1k images |
| `python -m lls_granular.launch sweep` | all 64 VGG8 partitions, 60 epochs (epoch 10 is the cheap proxy) |
| `python -m lls_granular.launch main` | VGG8, 300 epochs: LLS, BP, `uniform:2/3/7`, `aligned`, `2,2,3` |
| `python -m lls_granular.launch seeds` | VGG8, 60 epochs: LLS, `1,1,5`, `2,5`, one block, BP × seeds 1–2 |
| `python -m lls_granular.launch cifar100` | VGG8 / CIFAR100: LLS, `aligned`, BP, `1,1,5`, `2,5` |
| `python -m lls_granular.launch mobilenet300` | MobileNetV1, 300 epochs: LLS, BP, `uniform:2`, `aligned`, one block, early/late merges, `4,2,6,2` |
| `python -m lls_granular.launch mobilenet_gap` | MobileNetV1, 300 epochs: `3,1×11`, `4,1×10`, `4,2,1×8`, `4,2,2,2,2,2`, `4,2,6,1,1` |
| `python -m lls_granular.measure --arch vgg8` (or `mobilenet`) | training memory and MACs of every partition |

The experiments are defined in `lls_granular/exps.py`. `launch` also accepts `--seeds 0 1 2` to repeat every run,
`--gpus N`, `--dry-run` to list the runs without starting them, `--restore DIR` to resume from an earlier `runs/`
folder (e.g. a previous Kaggle output), and `-- <flags>` to pass flags to every run (e.g. `-- --no-wandb`).

#### Figures and tables

`python -m lls_granular.analyze --out report/granularity` pulls the finished runs from wandb and writes the
accuracy–memory fronts, per-layer head accuracy, proxy ranking, memory-budget picks, CSV tables and `results.md`;
experiments that have not finished are skipped, so it can be re-run as results come in.

### 2. Early exit (`lls_exit`)

Every LLS head already scores the classes, so a trained granular network is a multi-exit network: the early-exit study
reuses its checkpoints and trains only the learned exits used as baselines. The parts depend on each other as follows:

```
trained checkpoints ──> exit dumps ──> stopping-rule sweep ─┬─> figures and tables
        │                   │      └─> diagnostics ─────────┤
        │                   └────────> prediction depth ────┘   (also needs the sweep's picks.csv)
        └─> learned exits (frozen) ──> sweep / diagnostics with --runs
learned exits (joint): no dependency
```

#### Exit dumps

Needs the trained checkpoints (GPU). The dump saves every head's logits on the validation and test sets once, together with the MACs of each exit, so that
all stopping rules can be compared offline without running the network again.
```bash
python -m lls_exit.dump --exp mobilenet300      # 8 MobileNet partitions
python -m lls_exit.dump --exp main              # 6 VGG8 partitions
python -m lls_exit.dump --exp sweep             # all 64 VGG8 partitions (optional)
python -m lls_exit.dump --run-dir runs/<run>    # a single run
```

| Argument | Meaning |
|---|---|
| `--exp E ...` / `--run-dir DIR` | every non-BP run of these experiments / one run folder |
| `--seeds S ...` | with `--exp`: the seeds the runs used |
| `--n-train N` | class-balanced train subset kept for prediction depth (default 10k, 0 = none) |
| `--no-feats` | skip saving head inputs (`feats.pt`, needed only for prediction depth) |
| `--fresh`, `--partial`, `--upload-only` | redo; dump unfinished runs; re-upload existing dumps |
| `--limit N`, `--no-wandb`, `--workers 0` | smoke test; local only; if `/dev/shm` is full |

It writes `exits.pt`, `feats.pt` and `exit_summary.json` next to each `ckpt.pt` and uploads `exits.pt` as the
artifact `exits-<run>` to the training run on wandb.

#### Stopping-rule sweep

Needs the dumps (CPU). The sweep evaluates confidence, entropy, margin, patience, our sync rule and sync + margin at 41 thresholds each,
together with static truncation and the oracle. For each δ ∈ {0, 0.5, 1, 2}% it picks the cheapest setting whose
validation accuracy is within δ of the full network and reports that setting on test.
```bash
python -m lls_exit.sweep --exp mobilenet300 --out report/exit/mobilenet
python -m lls_exit.sweep --exp main         --out report/exit/vgg8
python -m lls_exit.sweep --exp sweep        --out report/exit/vgg8_sweep
```

| Argument | Meaning |
|---|---|
| `--exp E ...` / `--runs NAME ...` | experiments, or run names (e.g. the learned-exit runs) |
| `--dumps DIR` | read `DIR/<run>/exits.pt` instead of downloading the wandb artifacts |
| `--n` | thresholds per rule (default 41) |
| `--out` | output folder: `sweep.csv.gz`, `pareto.csv`, `picks.csv`, `temps.json` |

#### Diagnostics

Needs the dumps (CPU). The diagnostics compute each exit's accuracy and calibration (ECE), overthinking, the depth at which each class
settles, and the orthogonality of the basis. They take the same `--exp` / `--runs` / `--dumps` / `--out` arguments as
the sweep, plus `--split val|test`.
```bash
python -m lls_exit.diag --exp mobilenet300 --out report/exit/mobilenet
```

#### Learned exits

The `frozen` plans need the trained checkpoints and the `joint` plans need nothing (GPU). To test whether fixed heads are good exits, we train linear probes on the frozen LLS features and exits on BP
networks, either after the backbone (frozen) or together with it (joint).
```bash
python -m lls_exit.train_exits --plan mobilenet_frozen   # linear probes on the LLS runs + frozen exits on BP
python -m lls_exit.train_exits --plan mobilenet_joint    # BP trained jointly with an exit after every layer
python -m lls_exit.train_exits --plan vgg8_frozen
python -m lls_exit.train_exits --plan vgg8_joint
```
A single run is `--variant frozen --source runs/<run>` or `--variant joint --arch vgg8 --epochs 100 [--partition 2,2,3]`.
Its `exits.pt` has the same format as a dump, so the sweep and diagnostics run on it with `--runs <names>`.

#### Prediction depth

`knn` needs the dumps with `feats.pt` (GPU), and `analyze` also needs the sweep's `picks.csv` (CPU). Prediction depth (Baldock et al. 2021) measures how hard a sample is with a k-NN classifier on each layer's features;
we compare it with the exit each sample takes.
```bash
python -m lls_exit.depth knn --exp mobilenet300                                    # GPU, needs feats.pt
python -m lls_exit.depth analyze --exp mobilenet300 --out report/exit/mobilenet    # CPU, needs picks.csv
```

#### Figures and tables

Needs the sweep and diagnostics outputs, and the learned-exit and depth outputs for those figures (CPU). `python -m lls_exit.analyze` reads the outputs in `report/exit/<set>/` and writes all figures to `report/exit/figs/`
and all tables to `report/exit/tables.md`.

### 3. Scalability and attention: ResNet and ViT (`scalability`, `vit`)

Both modules reuse the granular LLS code and change only what a unit is: a complete residual block for ResNets and a
complete encoder block for the ViT, so block boundaries never cut through a skip connection. For the ViT, the LLS head
projects the concatenation of the CLS token and the mean patch token. The config, metrics and summary of the eight runs
below are stored in `runs/`.

#### ResNet and Wide-ResNet

```bash
python -m scalability.run --arch resnet20 --partition bp --epochs 100 --patience 10 --data-dir scalability --device cuda --no-wandb
python -m scalability.run --arch resnet20 --partition uniform:1 --epochs 100 --patience 10 --data-dir scalability --device cuda --no-wandb
python -m scalability.run --arch resnet20 --partition uniform:1 --lls-mode lls-mxm --epochs 100 --patience 10 --data-dir scalability --device cuda --no-wandb
python -m scalability.run --arch resnet20 --partition aligned --lls-mode lls-mxm --epochs 200 --data-dir scalability --device cuda --no-wandb
python -m scalability.run --arch wide_resnet20 --partition bp --epochs 300 --data-dir scalability --device cuda --no-wandb
python -m scalability.run --arch wide_resnet20 --partition 7,3 --lls-mode lls-mxm --readout lls --epochs 300 --data-dir scalability --device cuda --no-wandb
```

#### ViT-Tiny

```bash
python -m vit.run --arch vit_tiny --partition bp --lls-mode lls --epochs 100 --data-dir scalability --device cuda --no-wandb
python -m vit.run --arch vit_tiny --partition uniform:2 --lls-mode lls-mxm --epochs 100 --data-dir scalability --device cuda --no-wandb
```

| Argument | Values | Meaning |
|---|---|---|
| `--arch` | `scalability.run`: `resnet20`, `resnet32`, `resnet56`, `wide_resnet20`; `vit.run`: `vit_tiny`, `vit_micro` | network |
| `--dataset` | `cifar10aug` (default), `cifar10`, `cifar100aug` | dataset |
| `--partition` | `bp`, `uniform:k`, `aligned`, explicit sizes (e.g. `4,3,3`, `7,3`) | as in granularity; `aligned` cuts at the ResNet stages. Defaults: `uniform:1` (ResNet), `uniform:2` (ViT) |
| `--lls-mode` | `lls`, `lls-m`, `lls-mxm` | fixed LLS head, a learned scale per class, or a learned C×C class-mixing matrix (default `lls` for ResNet, `lls-mxm` for ViT); use `lls` with `--partition bp` |
| `--readout` | `linear` (default), `lls` | predict with a detached linear classifier, or with the last LLS head |
| `--epochs`, `--patience` | default 300, off | training epochs; stop after this many epochs without validation improvement |
| `--device` | `auto` (default), `cuda`, `mps`, `cpu` | `auto` prefers CUDA, then Apple MPS, then CPU |
| `--lr`, `--wd` | ResNet `5e-3`, `0`; ViT `5e-4`, `0.05` | schedule-free AdamW |
| `--data-dir`, `--out`, `--no-wandb`, `--fresh` | | data folder, output folder, local logging only, restart |

Further experiment presets are defined in `scalability/exps.py` and `vit/exps.py` and run with
`python -m scalability.launch <exp>` or `python -m vit.launch <exp>`.

## Key results

### Granularity

Where the cuts go matters more than how many there are. Partitions with equal memory differ by 2–3 points on VGG8,
against a seed std of 0.1–0.3, and the best partition on each network merges the first layers, while merging late
layers does not help. Because the best cut differs per model, simple rules such as uniform blocks or cuts at pooling
layers are 2–2.5 points below the best partitions.

#### CIFAR10

Mem. is the peak training activation memory relative to per-layer LLS, and gap closed is the share of the LLS–BP
accuracy gap that a partition recovers.

| VGG8 (60 epochs) | Acc. (%) | Gap closed | Mem. | | MobileNetV1 (300 epochs) | Acc. (%) | Gap closed | Mem. |
|---|---|---|---|---|---|---|---|---|
| Per-layer LLS | 86.2 | 0% | 1.00 | | Per-layer LLS | 67.8 | 0% | 1.00 |
| Uniform `2,2,2,1` | 89.4 | 53% | 1.29 | | Merge last 4 | 66.7 | −7% | 2.37 |
| `1,1,5` | 89.9 | 60% | 1.00 | | Uniform `2×7` | 77.3 | 58% | 1.54 |
| **`2,5`** | **91.6** | **89%** | **1.29** | | **`4,2,1×8`** | **83.0** | **94%** | **2.56** |
| BP | 92.3 | 100% | 2.15 | | BP | 84.0 | 100% | 8.21 |

#### CIFAR100 (VGG8, 100 epochs)

The choice of partition matters more on the harder task, where `2,5` matches BP.

| Partition | CIFAR10 (%) | CIFAR100 (%) |
|---|---|---|
| Per-layer LLS | 86.6 | 54.8 |
| `aligned` (`2,2,1,1,1`) | 89.6 | 59.4 |
| `1,1,5` | 89.9 | 63.8 |
| **`2,5`** | **91.6** | **67.1** |
| BP | 92.8 | 67.2 |

| Accuracy vs training memory, all 64 VGG8 partitions | MobileNetV1 partitions |
|---|---|
| ![](report/granularity/vgg8_front.png) | ![](report/granularity/mobilenet_front.png) |

| Accuracy of each layer's own LLS head | Ranking partitions after 10 epochs vs after 60 |
|---|---|
| ![](report/granularity/lls_layers.png) | ![](report/granularity/proxy_rank.png) |

The per-layer heads explain why early merges win: a per-layer head forces the early layers to separate the classes
before they have built useful features, and accuracy stops improving after the first few layers (after layer 7 on
MobileNetV1), so merging late layers cannot recover what is lost. Since the partition must be searched per model, we
also checked a cheap search: ranking partitions after 10 epochs picks the best one in 4 of 5 memory budgets.

### Early exit

Within one point of full accuracy, the LLS heads cut inference MACs by 85–90% on MobileNetV1 and by 32–43% on VGG8,
which saves less because 42% of its MACs are in full-resolution early convolutions before its first useful exit. The
partition matters here too: on MobileNetV1, `4,2,6,2` reaches 82.3% at 10% of the MACs, against 67.2% at 12% for
per-layer LLS. The stopping rule matters less, because the heads are already calibrated (ECE ≤ 0.08), so a simple
margin threshold usually matches our sync rule.

#### Comparison with learned exits at δ = 1% (CIFAR10)

MACs are given as a fraction of the full network, and the LLS partitions are `4,2,6,2` for MobileNetV1 and `3,3,1`
for VGG8.

| Exits | MobileNetV1 Acc. (%) | MACs | VGG8 Acc. (%) | MACs |
|---|---|---|---|---|
| Fixed LLS heads (ours) | 82.3 | 0.10 | 89.2 | 0.68 |
| Linear probes on LLS | 82.2 | 0.08 | 89.3 | 0.67 |
| BP, frozen exits | 82.8 | 0.10 | 92.1 | 0.70 |
| BP, jointly trained exits | 85.6 | 0.09 | 91.8 | 0.52 |

The fixed LLS heads match trained linear probes without any exit parameters, while BP with jointly trained exits is
better at every cost, mainly because the BP networks are more accurate to begin with.

| Exit accuracy vs inference MACs | Inference saving vs accuracy, all 64 VGG8 partitions |
|---|---|
| ![](report/exit/figs/exit_profile.png) | ![](report/exit/figs/sweep64.png) |

| Fixed heads vs learned exits | Exit depth vs sample difficulty (k-NN prediction depth) |
|---|---|
| ![](report/exit/figs/learned.png) | ![](report/exit/figs/depth.png) |

#### Interpretability

The heads are calibrated, so their confidence is a reliable exit signal, and their per-layer accuracy shows where the
network stops improving, which tells us where to merge layers. The exit a sample takes also follows its difficulty
(Spearman 0.44–0.59 with k-NN prediction depth on VGG8), and weak exits come from confusable classes such as cat–dog
rather than from the basis, whose rows are nearly orthogonal.

### Scalability

#### ResNets on CIFAR10, seed 0

Memory is the analytic peak activation memory; the BP and per-block runs on ResNet-20 stopped early after 10 epochs
without validation improvement.

| Network | Method | Epochs | Acc. (%) | Memory (MB) |
|---|---|---|---|---|
| ResNet-20 | BP | 48 | 88.3 | 281.5 |
| | Per-block LLS | 57 | 51.4 | 56.5 |
| | Per-block LLS-MxM | 57 | 56.1 | 56.5 |
| | `4,3,3` LLS-MxM | 200 | 71.9 | 170.0 |
| Wide-ResNet-20 | BP | 300 | 92.4 | 561.6 |
| | `7,3` LLS-MxM, LLS readout | 300 | 89.8 | 490.0 |

As on VGG8 and MobileNetV1, per-block LLS loses most of the accuracy on ResNet-20, and grouping the blocks at the
stage boundaries recovers much of it: `4,3,3` improves on per-block LLS-MxM by 15.8 points with 40% less memory than
BP, though it is still 16.4 points below BP. The wider network, whose blocks have more capacity, closes most of the
gap: `7,3` is 2.5 points below BP with 12.7% less memory and 3% fewer training MACs, but it is not faster per epoch.

### Attention

#### ViT-Tiny on CIFAR10, 100 epochs, seed 0

Memory is the measured peak GPU memory, and both models are trained on the same V100.

| Method | Acc. (%) | Memory (MB) | Time (s/epoch) |
|---|---|---|---|
| BP | 81.0 | 1368.2 | 19.6 |
| `2,2,2,2,2,2` LLS-MxM | 77.9 | 353.3 | 23.3 |

The six-block LLS-MxM model is 3.1 points below BP with 74% less memory, the largest saving in the project, since
attention stores large intermediate tensors in every block and LLS keeps only two blocks' activations at a time. The
extra heads and optimizers make each epoch 19% longer, so, as for the CNNs, the benefit is memory rather than speed.
