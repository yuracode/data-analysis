# 環境セットアップガイド

このコースで使用する環境の準備手順です。コマ1（ガイダンス）の前に各自で完了させてください。

---

## 1. Python環境

Python 3.10 以上推奨。

```bash
# 仮想環境の作成（任意）
python -m venv venv
source venv/bin/activate     # macOS/Linux
# venv\Scripts\activate      # Windows

# 必要パッケージのインストール
pip install -r requirements.txt
```

## 2. Jupyter Lab の起動

```bash
jupyter lab
```

## 3. 日本語フォントの設定（matplotlib）

matplotlib のグラフで日本語が文字化けする場合は、以下のいずれかでフォントを設定してください。

### Linux
```bash
sudo apt install fonts-ipafont
```
コード内：
```python
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
```

### macOS
```python
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'
```

### Windows
```python
import matplotlib
matplotlib.rcParams['font.family'] = 'MS Gothic'
```

### 共通（推奨：japanize-matplotlib）
```bash
pip install japanize-matplotlib
```
```python
import japanize_matplotlib   # import するだけで日本語化
```

## 4. データセットの準備

### Titanic（Phase 1・4）
- Kaggle にアカウント登録（無料）
- https://www.kaggle.com/competitions/titanic/data から `train.csv` `test.csv` を取得
- プロジェクトルートの `data/titanic/` に配置

### House Prices（Phase 4）
- https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
- `data/house_prices/` に配置

### Phase 3 の演習データ
- 多くは `seaborn.load_dataset('titanic')` または各レッスン内で生成するサンプルデータを使用
- 別途ダウンロード不要

## 5. Kaggle CLI（任意）

コンペデータをコマンドラインで取得する場合：

```bash
pip install kaggle
# ~/.kaggle/kaggle.json に API キーを配置
kaggle competitions download -c titanic
```

## 6. 動作確認

以下のコードがエラーなく実行できれば準備完了：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import shap
import statsmodels.api as sm

print("セットアップ完了！")
df = sns.load_dataset('titanic')
print(df.shape)
```

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| `ModuleNotFoundError: No module named 'category_encoders'` | `pip install category_encoders` |
| `lightgbm` のインストールエラー（macOS） | `brew install libomp` の後に再インストール |
| グラフの日本語が「□」になる | 上記のフォント設定 or `japanize-matplotlib` |
| `shap` のインストールが遅い | `pip install --upgrade pip` の後に再試行 |
