# [コマ24] 機械学習②：回帰モデルと評価指標（Udemy補足）

## 本日の目標
- 主要な回帰モデルの特徴と使いどころを説明できる
- RMSE・MAE・R²の違いとビジネス上の意味を説明できる
- 正則化（Ridge・Lasso）の効果を説明できる

## 前回の振り返り
- 分類モデルの比較・混同行列・ROC-AUCを学び、クラス不均衡への対処も確認した

## 本編

### セクション1：回帰モデルの比較

| モデル | 特徴 | 向いているケース |
|--------|------|----------------|
| 線形回帰 | 解釈しやすい | 線形関係が仮定できる場合 |
| Ridge | L2正則化で係数を小さく | 多重共線性あり |
| Lasso | L1正則化で係数をゼロに | 特徴量選択を同時にしたい |
| ElasticNet | Ridge + Lasso | バランスを取りたい |
| RandomForest回帰 | 非線形・頑健 | 複雑な関係 |
| GBM回帰 | 高精度 | Kaggle等スコア重視 |

ビジネス文脈：住宅価格予測・売上予測・需要予測では解釈性が求められることが多く、線形モデルが選ばれやすい。

### セクション2：評価指標

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.8, 5.1, 2.2, 7.5])

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R²:   {r2:.3f}")
```

| 指標 | 式 | 特徴 |
|------|-----|------|
| RMSE | √(Σ(y-ŷ)²/n) | 外れ値に敏感、単位が目的変数と同じ |
| MAE | Σ\|y-ŷ\|/n | 外れ値に頑健、解釈しやすい |
| R² | 1 - SS_res/SS_tot | 0〜1、モデルの説明力 |
| RMSLE | log変換したRMSE | 価格等の対数スケールで有効 |

### セクション3：Ridge・Lassoの正則化

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt

RANDOM_STATE = 42

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_scores = [cross_val_score(Ridge(alpha=a), X_train, y_train,
                cv=5, scoring='neg_root_mean_squared_error').mean()
                for a in alphas]

plt.figure(figsize=(7, 4))
plt.semilogx(alphas, [-s for s in ridge_scores], marker='o')
plt.xlabel('alpha（正則化の強さ）')
plt.ylabel('CV RMSE')
plt.title('Ridge の alpha チューニング')
plt.grid()
plt.show()
```

## ハンズオン / 演習

1. 自分のデータ（回帰問題の場合）で線形回帰・Ridge・GBM回帰を比較する
2. RMSE・MAE・R²の3指標を表にまとめる
3. Ridge の alpha を変化させてCVスコアを比較する

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- 評価指標はビジネスの文脈（外れ値の重み付け等）に合わせて選ぶ
- RidgeはVIF高い特徴量を持つデータに有効で、Lassoは特徴量選択を兼ねる
- Kaggle等のコンペでは評価指標が決まっているので、それに最適化する

## 次回予告
- 機械学習③：モデル評価・過学習対策・ハイパーパラメータチューニング（Udemy補足）
