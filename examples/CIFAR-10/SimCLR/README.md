## CIFAR-10 Experiment Notes (KTG)

This directory contains the complete set of experiments for KTG (Knowledge Transfer Graph) on CIFAR-10 with SimCLR. Use `dcl-train.py` for the search, `test/dcl-test.py` and `test/dml-test.py` for retraining and evaluation, and `pre-train.py` for pre-training single models.

### Highlights (TL;DR) — Focus on Node0 `resnet32`
- **pre-train (val)**: 73.60%
- **DCL (val)**: 74.16% (node0, Trial 0000)
- **DCL (test)**: 74.28% (node0), +0.68pt vs pre-train

### Table of Contents
- Dataset and preprocessing (SimCLR)
- Common training settings
- Available models
- DCL search summary and configuration
- Retraining + test results (DCL / DML)
- Single model test (pre-train)
- Quick Start
- Notes

---

### Dataset and preprocessing (SimCLR)
- **Split**: fixed `train=40,000 / val=10,000`
- **Test operation**: with `use_test_mode=True`, train on `train+val` and evaluate on `test`. For SimCLR training, we do not use normalization statistics; only SSL transforms are applied.
- **SimCLRTransforms**:
  - RandomResizedCrop(32)
  - RandomHorizontalFlip(p=0.5)
  - ColorJitter with p=0.8
  - RandomGrayscale(p=0.2)
  - ToTensor
  - Generate two views per sample and return `[q, k]`.
- For KNN validation, use simple `ToTensor()` without Normalize (overridden in scripts).

### Common training settings
- Batch size: 512
- Epochs: 400 (Warmup 10 epochs + CosineAnnealing)
- Optimizer: LARS (lr=1.0, weight_decay=1e-6, momentum=0.9, eta=0.001)
- Scheduler: `get_cosine_schedule_with_warmup`
- Num classes: 10 (used for the encoder's final layer size)
- Metric: KNN (K=20) for representation quality (train→val or train+val→test)

### Available models
- `resnet32`, `resnet110`, `wideresnet28_2`

### DCL search summary
- Study: `dcl_3` (number of nodes = 3)
- Log: `examples/CIFAR-10/SimCLR/optuna/dcl_3/optuna.log`
- Best trial: **Trial 0000**
- Best score: **KNN@20 74.16%** (estimated on `val`, node0)
- Trial outputs: `examples/CIFAR-10/SimCLR/runs/dcl_3/0000/`
- Number of trials: 51 (0000–0050)

#### Best trial configuration (Trial 0000)
- Models (node0→2): `[resnet32, wideresnet28_2, resnet110]`
- Gate matrix (row = i node = receiver, column = j node = sender. Each cell controls transfer j→i)

| i\j | 0 | 1 | 2 |
|---|---|---|---|
| 0 | ThroughGate | NegativeLinearGate | PositiveLinearGate |
| 1 | ThroughGate | ThroughGate | CutoffGate |
| 2 | CutoffGate | PositiveLinearGate | PositiveLinearGate |

> Note: In the code, the outer loop is `i` (receiver) and the inner loop is `j` (sender) (`{i}_{j}_gate`).

#### Node0 score (train→val, KNN@20)
| Trial | Best KNN@20(%) | Best Epoch |
|---:|---:|---:|
| 0000 | 74.16 | 359 |

### Single model results (pre-train.py, train→val)
- Log: `examples/CIFAR-10/SimCLR/runs/pre-train/`
- Checkpoints: `examples/CIFAR-10/SimCLR/checkpoint/pre-train/{model}/`

| Model | Best KNN@20(%) | Best Epoch |
|---|---:|---:|
| resnet32 | 73.60 | 363 |
| resnet110 | 79.03 | 282 |
| wideresnet28_2 | 79.36 | 334 |

> Note: Scores are logged to TensorBoard as `train_score` / `test_score`.

### Single model test results (pre-train.py, train+val → test)
- Log: `examples/CIFAR-10/SimCLR/test/runs/pre-train/`
- Checkpoints: `examples/CIFAR-10/SimCLR/test/checkpoint/pre-train/{model}/`

| Model | Best KNN@20(%) | Best Epoch |
|---|---:|---:|
| resnet32 | 73.74 | 369 |
| resnet110 | 79.13 | 358 |
| wideresnet28_2 | 79.39 | 374 |

### Retraining + test results (train+val → test)

#### dcl-test.py (retraining with the best trial)
- Log: `examples/CIFAR-10/SimCLR/test/runs/dcl_3/0000/`
- Checkpoints: `examples/CIFAR-10/SimCLR/test/checkpoint/dcl_3/0000/`

| Node | Model | Best KNN@20(%) | Best Epoch |
|---|---|---:|---:|
| 0 | resnet32 | 74.28 | 347 |
| 1 | wideresnet28_2 | 79.11 | 384 |
| 2 | resnet110 | 78.60 | 393 |

> Note: `dcl-test.py` reconstructs the graph with the best-trial configuration and evaluates on the `test` set.

#### dml-test.py (all edges ThroughGate)
- Log: `examples/CIFAR-10/SimCLR/test/runs/dml_3/0000/`
- Checkpoints: `examples/CIFAR-10/SimCLR/test/checkpoint/dml_3/0000/`

| Node | Model | Best KNN@20(%) | Best Epoch |
|---|---|---:|---:|
| 0 | resnet32 | 74.02 | 363 |
| 1 | wideresnet28_2 | 78.71 | 338 |
| 2 | resnet110 | 79.07 | 395 |

### Quick Start
1) Pre-train (optional)
```bash
cd examples/CIFAR-10/SimCLR
python pre-train.py --model resnet32
python pre-train.py --model resnet110
python pre-train.py --model wideresnet28_2
```

2) Search with Optuna (DCL)
```bash
cd examples/CIFAR-10/SimCLR
python dcl-train.py --num-nodes 3 --n_trials 100 \
  --gates ThroughGate CutoffGate PositiveLinearGate NegativeLinearGate
```

3) Retraining + evaluation with the best trial (test-operation mode)
```bash
cd examples/CIFAR-10/SimCLR/test
python dcl-test.py --num-nodes 3 --trial 0
# If omitted, the study's best_trial is used by default
```

### Notes
- `runs/` is the output directory for TensorBoard logs.
- `checkpoint/` stores the best models (including `pre-train`).
- For a node where all incoming edges are `CutoffGate` and `i!=0`, the corresponding model's pre-trained checkpoint is automatically loaded.

