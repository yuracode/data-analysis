# [コマ15] 基本統計量と確率分布（Udemy補足）

## 本日の目標
- 記述統計（平均・分散・歪度・尖度）の意味とデータへの示唆を説明できる
- 正規分布・ポアソン分布・二項分布の使いどころを判断できる
- 外れ値の検出手法（IQR・z-score）を実装できる

## 前回の振り返り
- pandas の `groupby` ・`merge` ・`transform` でデータ操作の基礎を習得した

## 本編

### セクション1：記述統計の読み方

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RANDOM_STATE = 42

# describe() の読み方
df.describe()
# count: 欠損がないか確認
# std:   ばらつきの大きさ
# 50%:   中央値（平均と離れていれば歪んでいる）
# max:   外れ値の可能性

# 歪度・尖度
print(f"歪度: {df['col'].skew():.3f}")   # 0に近いほど対称
print(f"尖度: {df['col'].kurt():.3f}")   # 3が正規分布の基準
```

### セクション2：確率分布の使いどころ

| 分布 | 使いどころ | パラメータ |
|------|-----------|-----------|
| 正規分布 | 連続量・誤差 | μ（平均）、σ（標準偏差） |
| ポアソン分布 | 単位時間の発生回数 | λ（平均発生率） |
| 二項分布 | 試行の成功回数 | n（試行数）、p（成功確率） |
| 対数正規分布 | 価格・収入（右裾が長い） | μ、σ |

```python
from scipy.stats import norm, poisson, binom

x = np.linspace(-4, 4, 200)
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

axes[0].plot(x, norm.pdf(x), label='N(0,1)')
axes[0].set_title('正規分布')

k = np.arange(0, 15)
axes[1].bar(k, poisson.pmf(k, mu=3), label='λ=3')
axes[1].set_title('ポアソン分布')

axes[2].bar(k, binom.pmf(k, n=10, p=0.4), label='n=10,p=0.4')
axes[2].set_title('二項分布')

plt.tight_layout()
plt.show()
```

### セクション3：外れ値の検出

```python
# IQR法
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers_iqr = df[(df['col'] < lower) | (df['col'] > upper)]

# z-score法
z_scores = np.abs(stats.zscore(df['col'].dropna()))
outliers_z = df[z_scores > 3]

print(f"IQR法 外れ値数: {len(outliers_iqr)}")
print(f"z-score法 外れ値数: {len(outliers_z)}")
```

## ハンズオン / 演習

1. 自分のデータの主要カラムで `describe()` を実行し、気になる点をメモする
2. 歪度が大きいカラムを特定し、ヒストグラムで確認する
3. IQR法で外れ値を検出し、その行を除外した場合と含めた場合で平均がどう変わるか確認する

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- 歪度・尖度は「データが正規分布に近いか」を定量的に表す
- 確率分布は「このデータが何分布に従うか」を仮定することでモデル選択の根拠になる
- 外れ値は「除去すべきか」ではなく「なぜ発生したか」を先に考える

## 次回予告
- データの可視化：目的別のグラフ選択と seaborn・matplotlib の使い方
