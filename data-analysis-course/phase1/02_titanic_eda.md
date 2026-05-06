# [コマ2] Titanic EDA（探索的データ分析）

## 本日の目標
- KaggleのTitanicデータセットをダウンロードしてPandasで読み込める
- 基本統計量と欠損値の確認ができる
- 変数の分布と生存率の関係を可視化できる

## 本編

### セクション1：Kaggle環境のセットアップ

Kaggle APIでデータを取得するか、手動でダウンロードして `data/` ディレクトリに配置する。

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

print(train.shape)
train.head()
```

### セクション2：基本統計量・欠損値確認

```python
# 基本統計量
train.describe()

# 欠損値の確認
train.isnull().sum().sort_values(ascending=False)
```

| カラム | 説明 | 欠損 |
|--------|------|------|
| Age | 年齢 | あり |
| Cabin | 客室番号 | 多数 |
| Embarked | 乗船港 | 少数 |

### セクション3：生存率の可視化

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 性別
train.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[0], title='性別×生存率')

# 客室クラス
train.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axes[1], title='Pclass×生存率')

# 年齢分布
train[train['Survived'] == 1]['Age'].hist(ax=axes[2], alpha=0.6, label='生存')
train[train['Survived'] == 0]['Age'].hist(ax=axes[2], alpha=0.6, label='死亡')
axes[2].legend()
axes[2].set_title('年齢×生存')

plt.tight_layout()
plt.show()
```

## ハンズオン / 演習

1. `train.csv` を読み込み、`Survived` の割合を確認する
2. `Pclass` ・`Sex` ・`Age` の3変数について生存率との関係をグラフで確認する
3. 欠損値が多いカラムはどれか。どう対処するか仮説を立てる

```python
# ヒント：クロス集計
pd.crosstab(train['Pclass'], train['Sex'], values=train['Survived'], aggfunc='mean')
```

## ？リスト（疑問を記録しよう）

- 今日気になったこと・わからなかったことをここにメモする
- 例：「Cabinの欠損が多いのはなぜ？」

## 今日のまとめ
- EDAの最初のステップは「形を見る」（行数・列数・型・欠損）
- 可視化で仮説（女性・1等客室＝生存率高い）を直感的に確認できる
- 次のステップは欠損補完と特徴量エンジニアリング

## 次回予告
- Titanicデータの前処理（欠損補完・カテゴリ変数エンコーディング）を実施する
