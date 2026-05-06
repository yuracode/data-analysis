# [コマ22] 統計的仮説検定②：相関・回帰の統計的側面（Udemy補足）

## 本日の目標
- ピアソン相関とスピアマン相関の使い分けを説明できる
- 単回帰分析の係数の統計的有意性を解釈できる
- 多重共線性の問題とVIFによる診断を実装できる

## 前回の振り返り
- t検定・カイ二乗検定・ANOVAの実装と、p値の正しい解釈（効果量との組み合わせ）を学んだ

## 本編

### セクション1：相関係数の使い分け

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

df = sns.load_dataset('titanic').dropna(subset=['age', 'fare']).copy()

# ピアソン相関（線形関係・正規分布を仮定）
r, p = stats.pearsonr(df['age'], df['fare'])
print(f"ピアソン age-fare r={r:.3f}, p={p:.4f}")

# スピアマン相関（順位相関・非線形でも有効・外れ値に頑健）
rho, p = stats.spearmanr(df['age'], df['fare'])
print(f"スピアマン age-fare ρ={rho:.3f}, p={p:.4f}")
```

| 手法 | 前提 | 向いているケース |
|------|------|----------------|
| ピアソン | 正規分布・線形 | 連続量の線形関係 |
| スピアマン | 順序尺度 | 非線形・外れ値あり |

### セクション2：回帰分析の統計的側面

```python
import statsmodels.api as sm

# titanicで「fare を pclass・age・sibsp で説明する」回帰
X = df[['pclass', 'age', 'sibsp']]
y = df['fare']

X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print(model.summary())
```

summaryの読み方：

| 項目 | 意味 |
|------|------|
| coef | 係数（他変数固定時のxが1増えたときのyの変化量） |
| P>\|t\| | 係数が0であるという帰無仮説のp値 |
| R-squared | 目的変数のばらつきを説明できている割合 |
| F-statistic | モデル全体の有意性 |

### セクション3：多重共線性の診断

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF（分散膨張因子）
vif_data = pd.DataFrame({
    'feature': X.columns,
    'VIF': [variance_inflation_factor(X_with_const.values, i+1)
            for i in range(len(X.columns))]
})
print(vif_data.sort_values('VIF', ascending=False))
# VIF > 10 は多重共線性の可能性大
```

多重共線性の対処：
- 相関が高い変数を1つに絞る
- PCA で次元削減する
- Ridge回帰などの正則化を使う

## ハンズオン / 演習

1. 自分のデータで主要変数の相関行列を計算し（ピアソン・スピアマン両方）、差異を確認する
2. `statsmodels` で単回帰を実行し、summary を読む
3. 特徴量にVIFを計算してVIF > 10のカラムがないか確認する

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- 外れ値があるデータはスピアマン相関を使うことを検討する
- 回帰のsummaryは係数・p値・R²を合わせて読む
- VIF > 10 は多重共線性のサインで、モデルの安定性に影響する

## 次回予告
- 機械学習①：分類モデルの選択と評価（Udemy補足）
