## CIFAR-10 SimCLR 実験ノート（KTG）

このディレクトリは CIFAR-10 を用いた KTG（Knowledge Transfer Graph）上での SimCLR 実験一式です。単体の自己教師あり事前学習は `pre-train.py`、共同学習（DCL）の探索は `dcl-train.py` を使用します。

### ハイライト（TL;DR） — KNN@20 指標（`runs` 由来）
- **pre-train（train→val）/ resnet32**: 73.60%
- **DCL（val）/ resnet32（node0, best）**: 74.16%（Trial 0000, Epoch 359）

### 目次
- データセットと前処理（SimCLR）
- 学習設定（共通）
- 利用可能モデル
- DCL 探索サマリと設定
- 単体モデル（train→val, pre-train）
- Quick Start
- 補足

---

### データセットと前処理（SimCLR）
- 分割: `train=40,000 / val=10,000` の固定分割（`ktg.dataset.cifar_datasets.cifar10.get_datasets`）。
- テスト運用: `use_test_mode=True` で `train+val` を学習、`test` を評価に使用。SimCLR 学習時は正規化統計は使用せず、SSL 用の変換（`SimCLRTransforms`）のみを適用します。
- SimCLR 前処理（`ktg.transforms.ssl_transforms.SimCLRTransforms`）:
  - RandomResizedCrop(32)
  - RandomHorizontalFlip(p=0.5)
  - ColorJitter を p=0.8 で適用
  - RandomGrayscale(p=0.2)
  - ToTensor
  - 各サンプルから 2 view を生成し、`[q, k]` を返します。
- KNN 検証時は Normalize を行わない単純な `ToTensor()` を使用（スクリプト内で上書き）。

### 学習設定（共通）
- バッチサイズ: 512
- エポック数: 400（Warmup 10 epoch + CosineAnnealing）
- 最適化: LARS（lr=1.0, weight_decay=1e-6, momentum=0.9, eta=0.001）
- スケジューラ: `get_cosine_schedule_with_warmup`
- クラス数: 10（エンコーダの最終層サイズに使用）
- 評価指標: KNN（K=20）による表現品質評価（train→val もしくは train+val→test）

### 利用可能モデル
- `resnet32`, `resnet110`, `wideresnet28_2`

### DCL 探索サマリと設定
- スタディ: `dcl_3`（ノード数 = 3）
- ログ: `examples/CIFAR-10/SimCLR/optuna/dcl_3/optuna.log`
- ゲート候補（探索空間）: `ThroughGate`, `CutoffGate`, `PositiveLinearGate`, `NegativeLinearGate`
- トライアル数: 51（0000〜0050）
- ベストトライアル: 0000（node0 resnet32 KNN@20 = 74.16%, Epoch 359 on val）
- Trial 出力: `examples/CIFAR-10/SimCLR/runs/dcl_3/0000/`
- すべてのエッジが `CutoffGate` かつ `i!=0` のノードでは、対応モデルの SimCLR 事前学習重みを自動ロード（`checkpoint/pre-train/{model}`）。

#### ノード0（resnet32）スコア（train→val, KNN@20）
| トライアル | ベストKNN@20(%) | ベストEpoch |
|---:|---:|---:|
| 0000 | 74.16 | 359 |

### 単体モデル結果（pre-train.py, train→val）
- ログ: `examples/CIFAR-10/SimCLR/runs/pre-train/`
- チェックポイント: `examples/CIFAR-10/SimCLR/checkpoint/pre-train/{model}/`

| モデル | ベストKNN@20(%) | ベストEpoch |
|---|---:|---:|
| resnet32 | 73.60 | 363 |
| resnet110 | 79.03 | 282 |
| wideresnet28_2 | 79.36 | 334 |

> 備考: スコアは TensorBoard の `train_score` / `test_score` に記録。

### 単体モデル結果（test/pre-train.py, train+val→test）
- ログ: `examples/CIFAR-10/SimCLR/test/runs/pre-train/`
- チェックポイント: `examples/CIFAR-10/SimCLR/test/checkpoint/pre-train/{model}/`

| モデル | ベストKNN@20(%) | ベストEpoch |
|---|---:|---:|
| resnet32 | 73.74 | 369 |
| resnet110 | 79.13 | 358 |
| wideresnet28_2 | 79.39 | 374 |

### Quick Start
1) 事前学習（任意: 各モデルで実行可）
```bash
cd examples/CIFAR-10/SimCLR
python pre-train.py --model resnet32
python pre-train.py --model resnet110
python pre-train.py --model wideresnet28_2
```

2) Optuna による探索（DCL × SimCLR）
```bash
cd examples/CIFAR-10/SimCLR
python dcl-train.py --num-nodes 3 --n_trials 100 \
  --gates ThroughGate CutoffGate PositiveLinearGate NegativeLinearGate
```

3) テスト運用モードでの単体 SimCLR（train+val → test）
```bash
cd examples/CIFAR-10/SimCLR/test
python pre-train.py --model resnet32
```

### 補足
- `runs/` は TensorBoard のログ出力先です。
- `checkpoint/` にはベストモデルが保存されます（`pre-train` ほか）。
- SimCLR は自己教師あり学習のため、スクリプト内では `SimCLRLoss` を使用し、KNN で表現品質を定期評価します。
- DCL 探索中の各ノードのスコアは `runs/dcl_3/{trial}/{i}_{model}/` に出力されます。
- データはスクリプト実行時に `torchvision` から自動ダウンロードされます（既存の `data/` があればそれを使用）。

