## CIFAR-100 実験ノート（KTG）

このディレクトリは CIFAR-100 を用いた KTG（Knowledge Transfer Graph）の実験一式です。探索は `dcl-train.py`、再学習と評価は `test/dcl-test.py`・`test/dml-test.py`、単体モデルの事前学習は `pre-train.py` を使用します。

### ハイライト（TL;DR）— Node0 の `resnet32` に注目
- **pre-train（test）**: 71.55%
- **DML（test）**: 72.10%（+0.55pt vs pre-train）
- **DCL（test）**: 73.54%（+1.99pt vs pre-train）

### 目次
- データセットと前処理
- 学習設定（共通）
- 利用可能モデル
- 探索（DCL）サマリと構成
- 再学習＋テスト結果（DCL / DML）
- 単体モデルテスト（pre-train）
- 実行手順（Quick Start）
- 補足

---

### データセットと前処理
- **分割**: `train=40,000 / val=10,000` の固定分割
- **テスト運用**: `use_test_mode=True` で `train+val` を学習、`test` を評価に使用（正規化統計も `train+val` 由来）
- **前処理**:
  - 学習時: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
  - 評価時: ToTensor + Normalize

### 学習設定（共通）
- バッチサイズ: 64
- エポック数: 200（CosineAnnealingLR, `eta_min=0`）
- 最適化: SGD（lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True）
- クラス数: 100

### 利用可能モデル
- `resnet32`, `resnet110`, `wideresnet28_2`

### 探索（DCL）サマリ
- スタディ: `dcl_3`（ノード数=3）
- ログ: `examples/CIFAR-100/optuna/dcl_3/optuna.log`
- ベスト試行: **Trial 11**
- ベストスコア: **Top-1 72.12%**（`val` 上の推定値）
- 試行出力: `examples/CIFAR-100/runs/dcl_3/0011/`
- 試行数: 100

#### ベスト試行の構成（Trial 11）
- モデル（ノード0→2）: `[resnet32, resnet32, resnet110]`
- ゲート行列（行=iノード=受け手、列=jノード=送り手。セルは j→i の転移を制御）

| i\j | 0 | 1 | 2 |
|---|---|---|---|
| 0 | PositiveLinearGate | PositiveLinearGate | PositiveLinearGate |
| 1 | CutoffGate | NegativeLinearGate | ThroughGate |
| 2 | ThroughGate | CutoffGate | ThroughGate |

> 注: コード上も外側ループが `i`（受け手）、内側が `j`（送り手）です（`{i}_{j}_gate`）。

### 再学習＋テスト結果（train+val → test）

#### dcl-test.py（ベスト試行で再学習）
- 実行ログ: `examples/CIFAR-100/test/runs/dcl_3/0011/`
- チェックポイント: `examples/CIFAR-100/test/checkpoint/dcl_3/0011/`

| ノード | モデル | ベストTop-1(%) | ベストEpoch |
|---|---|---:|---:|
| 0 | resnet32  | 73.54 | 198 |
| 1 | resnet32  | 72.92 | 200 |
| 2 | resnet110 | 75.60 | 200 |

> 備考: `dcl-test.py` はベスト試行の構成でグラフを再構築し、`test` セットで評価します。

#### dml-test.py（全エッジ ThroughGate）
- 実行ログ: `examples/CIFAR-100/test/runs/dml_3/0011/`
- チェックポイント: `examples/CIFAR-100/test/checkpoint/dml_3/0011/`

| ノード | モデル | ベストTop-1(%) | ベストEpoch |
|---|---|---:|---:|
| 0 | resnet32  | 72.10 | 200 |
| 1 | resnet32  | 72.64 | 199 |
| 2 | resnet110 | 74.96 | 200 |

> 備考: すべてのエッジを `ThroughGate` に固定した DML 構成で学習・評価します。

### 単体モデルのテスト結果（pre-train.py, train+val → test）
- 実行ログ: `examples/CIFAR-100/test/runs/pre-train/`
- チェックポイント: `examples/CIFAR-100/test/checkpoint/pre-train/{model}/`

| モデル | ベストTop-1(%) | ベストEpoch |
|---|---:|---:|
| resnet32 | 71.55 | 198 |
| resnet110 | 73.67 | 187 |
| wideresnet28_2 | 75.56 | 198 |

### 実行手順（Quick Start）
1) 事前学習（任意）
```bash
cd examples/CIFAR-100
python pre-train.py --model resnet32
python pre-train.py --model resnet110
python pre-train.py --model wideresnet28_2
```

2) Optuna による探索（DCL）
```bash
cd examples/CIFAR-100
python dcl-train.py --num-nodes 3 --n_trials 100 \
  --models resnet32 resnet110 wideresnet28_2 \
  --gates ThroughGate CutoffGate PositiveLinearGate NegativeLinearGate
```

3) ベスト試行での再学習＋評価（テスト運用モード）
```bash
cd examples/CIFAR-100/test
python dcl-test.py --num-nodes 3 --trial 11
# 省略時は study の best_trial を使用
```

### 補足
- `runs/` は TensorBoard のログ出力先です。
- `checkpoint/` にはベストモデルが保存されます（`pre-train` ほか）。
- ゲートがすべて `CutoffGate` かつ `i!=0` のノードは、該当モデルの事前学習チェックポイントを自動で読み込みます。


