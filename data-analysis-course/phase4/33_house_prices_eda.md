# [コマ33] House Prices EDA

## 本日の目標
- House Pricesデータセットの構造と変数の意味を把握できる
- 目的変数（SalePrice）の分布と変換の必要性を理解できる
- 住宅価格に影響する変数の仮説を立てて可視化で確認できる

## 前回の振り返り
- 回帰係数の意思決定への活用と、相関・因果の違いを学んだ

## 本編

### セクション1：House Prices のビジネス文脈

> **なぜ住宅価格を予測するのか？**

不動産分野でのユースケース：
- 購入者：「この物件は適正価格か？」
- 不動産会社：「リフォームすると価格がどれだけ上がるか？」
- 金融機関：「担保価値を客観的に評価したい」
- 自治体：「固定資産税の算定根拠を自動化したい」

→ 価格予測は「意思決定支援ツール」である

### セクション2：データの構造確認

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RANDOM_STATE = 42

train = pd.read_csv('data/house_prices/train.csv')
test  = pd.read_csv('data/house_prices/test.csv')

print(f"train: {train.shape}, test: {test.shape}")
print(f"カラム数: {len(train.columns)}")
print(f"\n欠損率上位10:\n{(train.isnull().mean() * 100).sort_values(ascending=False).head(10)}")
```

House Prices の主要変数カテゴリ：

| カテゴリ | 変数例 |
|---------|--------|
| 立地 | Neighborhood, MSZoning |
| 建物 | YearBuilt, GrLivArea, TotalBsmtSF |
| 品質 | OverallQual, OverallCond |
| 設備 | GarageArea, PoolArea, Fireplaces |
| 取引 | YrSold, MoSold, SaleType |

### セクション3：目的変数の分布と対数変換

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 元の分布
train['SalePrice'].hist(ax=axes[0], bins=50)
axes[0].set_title(f'SalePrice (歪度: {train["SalePrice"].skew():.2f})')

# 対数変換後
np.log1p(train['SalePrice']).hist(ax=axes[1], bins=50)
axes[1].set_title(f'log(SalePrice+1) (歪度: {np.log1p(train["SalePrice"]).skew():.2f})')

plt.tight_layout()
plt.show()

# 対数変換を適用
train['LogSalePrice'] = np.log1p(train['SalePrice'])
```

対数変換する理由：
- RMSEを最小化する際、価格が高い物件の誤差が大きく効きすぎる
- KaggleのHouse Pricesの評価指標はRMSLE（対数スケール）

## 今日のKaggle作業

### 問いの確認
住宅価格を精度よく予測するために、どの変数が重要か？ビジネス文脈ではどう解釈できるか？

### 作業手順
1. 欠損値の多いカラムを確認し、欠損の理由（「該当なし」か「不明」か）を考える
2. `OverallQual` vs `SalePrice` の散布図で品質と価格の関係を確認する
3. 相関ヒートマップで SalePrice と相関が高い数値変数を特定する

```python
# 相関上位の確認
corr = train.select_dtypes(include='number').corr()
top_corr = corr['SalePrice'].abs().sort_values(ascending=False).head(11)
print(top_corr)
```

### 詰まったときの対処
- Discussion で検索するキーワード例：「house prices EDA」「feature engineering house prices」

### 振り返り
- 今日の「?」リスト：

## 今日のまとめ
- House Prices は80変数以上あり、変数の意味を理解することが前処理の出発点
- 右裾が長い目的変数は対数変換でモデルの精度が改善することが多い
- 欠損値の「NA」は「データなし」ではなく「該当する設備なし」の意味であることが多い

## 次回予告
- House Prices 前処理：大量の欠損値と多様なカテゴリ変数の対処
