# [コマ11] ビジネス知識②演習：需要予測・A/Bテストを模倣する

## 本日の目標
- 時系列データを可視化して季節性やトレンドを読み取れる
- A/Bテストの検定をPythonで実装できる
- 分析結果を「意思決定の提言」として表現できる

## 前回の振り返り
- 需要予測・価格弾力性・A/Bテストの概念と実装例を学んだ

## 本編

### セクション1：時系列データの生成と可視化

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 2年分の日次売上データを生成
dates = pd.date_range('2023-01-01', periods=730, freq='D')
trend = np.linspace(100, 130, 730)
seasonal = 20 * np.sin(2 * np.pi * np.arange(730) / 365)
noise = np.random.normal(0, 5, 730)
sales = pd.Series(trend + seasonal + noise, index=dates, name='sales')

# 可視化
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
sales.plot(ax=axes[0], title='日次売上')
sales.rolling(30).mean().plot(ax=axes[0], label='30日MA', color='red')
axes[0].legend()

# 月次集計
sales.resample('ME').mean().plot(ax=axes[1], kind='bar', title='月次平均売上')
plt.tight_layout()
plt.show()
```

### セクション2：A/Bテストのシミュレーション

学校での施策例：「補講プログラムへの参加が就職率に影響するか」

```python
np.random.seed(RANDOM_STATE)

n_each = 150
# コントロール群（通常授業のみ）
control_jobs = np.random.binomial(1, 0.68, n_each)
# テスト群（補講プログラムあり）
treatment_jobs = np.random.binomial(1, 0.78, n_each)

cvr_control   = control_jobs.mean()
cvr_treatment = treatment_jobs.mean()

print(f"コントロール群の就職率: {cvr_control:.3f}")
print(f"テスト群の就職率:       {cvr_treatment:.3f}")
print(f"差分:                   {cvr_treatment - cvr_control:+.3f}")

# 二群比率の検定
count = [treatment_jobs.sum(), control_jobs.sum()]
nobs  = [n_each, n_each]
z, p  = proportions_ztest(count, nobs)
print(f"p値: {p:.4f} → {'有意差あり' if p < 0.05 else '有意差なし'}")
```

### セクション3：結果の解釈と提言

```
[分析結果]
補講プログラム参加群の就職率は X%、非参加群は Y%。
p値 = Z（< 0.05）のため統計的に有意な差がある。

[提言]
補講プログラムは就職率向上に効果があると示唆される。
ただし、参加意欲の高い学生が選択的に参加している可能性（選択バイアス）
に注意が必要。ランダム割り当てによる再検証を推奨する。
```

## 演習

### お題
前回設計した「就職支援が必要なグループ」に向けた施策を考え、A/Bテストを設計する。

### やること
1. A群・B群の定義と評価指標を決める
2. 必要なサンプルサイズを考える（「何人いれば差を検出できるか」）
3. 上記のシミュレーションを自分の仮説に合わせて数値を変えて実行する

### 提出物
- A/Bテスト設計書（A群・B群・評価指標・期間・必要サンプルサイズ）

## ？リスト（疑問を記録しよう）
- p値が0.05以下でも実務で使えない結果はある？
- 選択バイアスをデータで補正する方法は？

## 今日のまとめ
- 時系列データはトレンドと季節性を分けて考える
- A/Bテストは検定だけでなく「選択バイアスの有無」まで考察する
- 提言は「何をすべきか」と「注意点（限界）」をセットで書く

## 次回予告
- Phase 2 後半：自分の問いを設計する①（教科書Ch5 まとめ + 自分の問い）
