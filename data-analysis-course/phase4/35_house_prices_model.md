# [コマ35] House Prices モデル学習・最適化

## 本日の目標
- 複数の回帰モデルを比較し、RMSLEで評価できる
- アンサンブルでスコアを改善できる
- 特徴量重要度からビジネス上の解釈を導ける

## 前回の振り返り
- House Pricesの欠損補完・特徴量エンジニアリング・エンコーディングを完了した

## 本編

### セクション1：モデル比較

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb

RANDOM_STATE = 42

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    'Ridge':  Ridge(alpha=10),
    'Lasso':  Lasso(alpha=0.001),
    'RF':     RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    'GBM':    GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE),
    'LGBM':   lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=kf,
                             scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    results[name] = rmse
    print(f"{name}: RMSE={rmse:.5f}")
```

### セクション2：アンサンブル

```python
# Ridge + GBM + LGBM の単純平均
from sklearn.base import BaseEstimator, RegressorMixin

class AverageEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1)

ensemble = AverageEnsemble([
    Ridge(alpha=10),
    GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE),
    lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1),
])

scores = cross_val_score(ensemble, X_train, y_train, cv=kf,
                         scoring='neg_root_mean_squared_error')
print(f"Ensemble RMSE: {-scores.mean():.5f}")
```

### セクション3：特徴量重要度とビジネス解釈

```python
gbm = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE)
gbm.fit(X_train, y_train)

feat_imp = pd.Series(gbm.feature_importances_, index=X_train.columns)
top15 = feat_imp.nlargest(15).sort_values()

top15.plot(kind='barh', figsize=(8, 6), title='特徴量重要度 Top 15')
plt.tight_layout()
plt.show()
```

ビジネス解釈の例：
```
1位：OverallQual → 総合品質が最も価格に影響
2位：TotalSF     → 延床面積が広いほど高価格
3位：GrLivArea   → 居住面積（地上）が重要

→ 「リフォームして総合品質を上げることが最も価格向上に効果的」
```

## 今日のKaggle作業

### 問いの確認
どのモデル・アンサンブルが最もRMSLEを最小化できるか？

### 作業手順
1. 5モデルのCVスコアを表にまとめる
2. アンサンブルを作成してCVスコアを比較する
3. 特徴量重要度の上位5変数についてビジネス解釈を書く

### 詰まったときの対処
- Discussion で検索するキーワード例：「house prices ensemble stacking」「lightgbm house prices」

### 振り返り
- 今日の「?」リスト：

## 今日のまとめ
- 単体モデルよりアンサンブルがスコアを改善することが多い
- 特徴量重要度は「どの変数がビジネス上重要か」を確認するツールでもある
- Kaggleのスコアと社内CVスコアが大きく乖離するときはリークを疑う

## 次回予告
- House Prices 提出：最終調整・提出・振り返り
