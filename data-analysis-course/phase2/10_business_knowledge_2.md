# [コマ10] ビジネス知識②：需要予測・在庫・価格戦略（発展：教科書Ch5の応用）

> **教科書との対応**：需要予測・価格弾力性・A/Bテストは教科書本文に直接の節はない**発展トピック**。教科書Ch5の「商品の育成」「マーケティング・ミックス」を実務データ分析に応用したもの。仮説思考（教科書5.7〜5.8）の考え方を土台にしている。

## 本日の目標
- 需要予測の基本的なアプローチと使うデータを説明できる
- 価格弾力性の概念とデータ分析における意味を理解できる
- A/Bテストの設計の考え方を説明できる

## 前回の振り返り
- RFM分析・コホート分析を学生データに応用し、「支援が必要なグループ」をデータで定義した

## 本編

### セクション1：需要予測

**需要予測**：将来の売上・需要量を過去データから推定する分析

用途：在庫管理・人員配置・仕入れ計画

| 手法 | 用途 | 特徴 |
|------|------|------|
| 移動平均 | 短期・安定系列 | 実装が簡単 |
| 指数平滑法 | 直近を重視 | パラメータ1つ |
| SARIMA | 季節性あり | 統計的に扱いやすい |
| LightGBM等ML | 複雑なパターン | 特徴量エンジニアリングが鍵 |

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# サンプルデータ生成：1年分の日次売上
dates  = pd.date_range('2024-01-01', periods=365, freq='D')
trend  = np.linspace(100, 130, 365)
season = 20 * np.sin(2 * np.pi * np.arange(365) / 365)
noise  = np.random.normal(0, 5, 365)
sales  = pd.Series(trend + season + noise, index=dates, name='sales')

sales_ma7  = sales.rolling(window=7).mean()
sales_ma30 = sales.rolling(window=30).mean()

plt.figure(figsize=(12, 4))
sales.plot(label='実績', alpha=0.5)
sales_ma7.plot(label='7日移動平均')
sales_ma30.plot(label='30日移動平均')
plt.legend()
plt.title('売上と移動平均')
plt.show()
```

### セクション2：価格弾力性

**価格弾力性（Price Elasticity of Demand）**：価格が1%変化したとき、需要が何%変化するか

```
弾力性 = (需要の変化率) / (価格の変化率)
```

- 弾力性 < -1：弾力的（価格に敏感）→ 値下げが有効
- -1 < 弾力性 < 0：非弾力的（価格に鈍感）→ 値上げが有効

### セクション3：A/Bテストの設計

**A/Bテスト**：施策の効果をランダム化比較試験で検証する

```
A群（コントロール）：現行の施策
B群（テスト）：新しい施策

→ 統計的検定でB > A を確認
```

設計時の注意点：

| 項目 | 内容 |
|------|------|
| 割り当て | ランダムに分ける |
| サンプルサイズ | 検出力を事前に計算する |
| 評価指標 | 1つのKPIに絞る |
| 期間 | 十分な時間を確保する |

```python
from statsmodels.stats.proportion import proportions_ztest

# 二群の比率の検定（例：CVR比較）
cvr_a = 0.05  # コントロール群のCVR
cvr_b = 0.06  # テスト群のCVR
n = 1000      # 各群のサンプルサイズ

count_a = int(cvr_a * n)
count_b = int(cvr_b * n)

z_stat, p_value = proportions_ztest([count_b, count_a], [n, n])
print(f"p値: {p_value:.4f}")
print("有意差あり" if p_value < 0.05 else "有意差なし")
```

## ハンズオン / 演習

1. 移動平均のコードを使って自分でサンプルデータを作り可視化する
2. 自分のPhase 2の問いで「A/Bテストが有効かどうか」を考える
   - 何をA群・B群にするか？
   - 評価指標は何か？

## ？リスト（疑問を記録しよう）
- A/Bテストができないとき（過去データだけの場合）どうする？
- 需要予測のモデルはどう評価する？

## 今日のまとめ
- 需要予測は「どの手法を使うか」より「問いに合った手法か」を先に考える
- 価格弾力性はビジネス上の意思決定に直接使える指標
- A/Bテストは「本当に効果があるか」を検証する標準的な方法

## 次回予告
- コマ10の内容を使った演習：需要予測・A/Bテストをサンプルデータで模倣する
