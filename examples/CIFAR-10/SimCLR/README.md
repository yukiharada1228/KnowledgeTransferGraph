## CIFAR-10 SimCLR 実験ノート（KTG）

このディレクトリは CIFAR-10 を用いた KTG（Knowledge Transfer Graph）上での SimCLR 実験一式です。単体の自己教師あり事前学習は `pre-train.py`、共同学習（DCL）の探索は `dcl-train.py` を使用します。

### 目次
- データセットと前処理（SimCLR）
- 学習設定（共通）
- 利用可能モデル
- 探索（DCL）メモ
- KNN 検証結果
- 実行手順（Quick Start）
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

### 探索（DCL）メモ
- スタディ名: `dcl_3`（ノード数=3 をデフォルト想定）
- ログ: `examples/CIFAR-10/SimCLR/optuna/dcl_3/optuna.log`
- ゲート候補（探索空間）: `ThroughGate`, `CutoffGate`, `PositiveLinearGate`, `NegativeLinearGate`
- すべてのエッジが `CutoffGate` かつ `i!=0` のノードでは、対応モデルの SimCLR 事前学習重みを自動ロード（`checkpoint/pre-train/{model}`）。

#### 探索（DCL）スコア例（train→val, KNN@20）
| Trial | ノード | モデル | ベストKNN@20(%) | ベストEpoch |
|---:|---:|---|---:|---:|
| 0 | 0 | resnet32 | 45.76 | 6 |
| 0 | 1 | resnet110 | 46.18 | 6 |
| 0 | 2 | resnet110 | 46.22 | 6 |

### KNN 検証結果

#### 単体モデル（pre-train, train→val）
| モデル | ベストKNN@20(%) | ベストEpoch |
|---|---:|---:|
| resnet32 | 73.60 | 363 |
| resnet110 | 79.35 | 370 |
| wideresnet28_2 | 79.36 | 334 |

#### テスト運用（train+val → test, `test/pre-train.py`）
| モデル | ベストKNN@20(%) | ベストEpoch |
|---|---:|---:|
| resnet32 | 73.74 | 369 |

備考:
- スコアは TensorBoard の `train_score` / `test_score` に記録されています（上記ログディレクトリを参照）。
- DCL 探索中の各ノードのスコアは `runs/dcl_3/{trial}/{i}_{model}/` に出力されます。

### 実行手順（Quick Start）
1) 事前学習（任意: 各モデルで実行可）
```bash
cd examples/CIFAR-10/SimCLR
python pre-train.py --model resnet32
python pre-train.py --model resnet110
python pre-train.py --model wideresnet28_2
```

2) Optuna による探索（DCL, SimCLR 連携）
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
- データはスクリプト実行時に `torchvision` から自動ダウンロードされます（既存の `data/` があればそれを使用）。


