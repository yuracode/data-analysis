# [コマ32] 教科書Ch8：統計的な意思決定支援

## 本日の目標
- 回帰分析の係数を「意思決定の根拠」として解釈できる
- 信頼区間と予測区間の違いを説明できる
- 「相関関係」と「因果関係」の違いをデータで議論できる

## 前回の振り返り
- 「データの比較」の3軸（時系列・属性・基準値）を確認し、SHAP・PDPで予測の根拠を分解した

## データ準備（自習用：このまま実行できます）

学生データを生成して回帰の意思決定支援を実演する。

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

n = 300
df = pd.DataFrame({
    'attendance_rate':      np.clip(np.random.normal(0.85, 0.10, n), 0, 1),
    'cert_count':           np.random.poisson(1.5, n),
    'study_hours_per_week': np.clip(np.random.normal(10, 4, n), 0, 30),
})
# 真の関係を仮定して final_score を生成（学習用）
df['final_score'] = (
    40
    + 50 * df['attendance_rate']
    + 3  * df['cert_count']
    + 1.2 * df['study_hours_per_week']
    + np.random.normal(0, 5, n)
).round(1)
```

## 本編

### セクション1：回帰係数の意思決定への活用

```python
X = df[['attendance_rate', 'cert_count', 'study_hours_per_week']]
y = df['final_score']

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())
```

係数の読み方（例）：
```
attendance_rate の係数 = 15.3 （p < 0.001）
→ 出席率が1（100%）増えると最終成績が15.3点上がる傾向
→ 「出席率向上施策を最優先に」という意思決定の根拠になる
```

### セクション2：信頼区間と予測区間

```python
# 予測値と信頼区間
predictions = model.get_prediction(X_const)
summary_frame = predictions.summary_frame(alpha=0.05)

# mean: 予測値
# mean_ci_lower/upper: 平均の信頼区間（モデルの不確実性）
# obs_ci_lower/upper: 予測区間（個別値の不確実性）
print(summary_frame[['mean', 'mean_ci_lower', 'mean_ci_upper',
                       'obs_ci_lower', 'obs_ci_upper']].head())
```

| 区間 | 意味 | 幅の比較 |
|------|------|---------|
| 信頼区間 | 「平均的な値」の不確実性 | 狭い |
| 予測区間 | 「個別の値」の不確実性 | 広い |

### セクション3：相関と因果の違い

```
相関：AとBが同時に変化する傾向がある
因果：AがBの原因である

例：
「アイスクリームの売上と溺死者数は正の相関がある」
→ これは因果ではなく「夏」という交絡変数が原因

データ分析で言えること：
○「X と Y の間に統計的に有意な正の相関がある」
×「X が原因で Y が増加する」（実験・RCTなしでは言えない）
```

因果推論のアプローチ（簡潔に）：
- ランダム化比較試験（RCT）
- 操作変数法
- 差分の差分法（DiD）
- 傾向スコアマッチング

## ハンズオン / 演習

1. 自分のデータで `statsmodels` を使って回帰分析を実施し、係数を意思決定の言葉で解釈する
2. 信頼区間を見て「不確実性が高い変数」を特定する
3. 「この分析結果から因果を主張できるか？できない理由は何か？」を1段落で書く

## 今日のまとめ
- 回帰係数は「この変数を変えると結果がどう変わるか」の定量的な根拠になる
- 予測区間は信頼区間より広く、個別の値を予測するときの不確実性を表す
- 相関関係から因果関係は言えない。因果を言いたければ設計から変える必要がある

## 次回予告
- House Prices EDA：Phase 4の実践コンペ用データの探索
