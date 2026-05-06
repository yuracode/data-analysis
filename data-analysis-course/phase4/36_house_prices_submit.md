# [コマ36] House Prices 提出・Phase 4 振り返り

## 本日の目標
- 最終モデルで予測ファイルを作成し、Kaggleに提出できる
- Titanicと比較してPhase 4での成長を言語化できる
- Phase 5（伝える・まとめる）に向けた準備ができている

## 前回の振り返り
- 複数モデルの比較・アンサンブル・特徴量重要度のビジネス解釈を実施した

## 本編

### セクション1：最終提出ファイルの作成

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

RANDOM_STATE = 42

# 最良アンサンブルを選定（前回のCVスコアをもとに）
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Out-Of-Fold 予測でアンサンブルの重みを調整することも可能
models_final = [
    Ridge(alpha=10),
    GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                               max_depth=4, random_state=RANDOM_STATE),
    lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03,
                      num_leaves=31, random_state=RANDOM_STATE, verbose=-1),
]

test_preds = np.zeros(len(X_test))
for model in models_final:
    model.fit(X_train, y_train)
    pred_log = model.predict(X_test)
    test_preds += np.expm1(pred_log) / len(models_final)

test_df = pd.read_csv('data/house_prices/test.csv')
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_preds
})
submission.to_csv('submission_house_prices.csv', index=False)
print(submission.describe())
```

### セクション2：Phase 4 の振り返り

| コマ | 学んだこと | 自分の作業 |
|------|-----------|-----------|
| 29 | 分析設計の精緻化・データリーク | |
| 30 | Titanic再分析・Pipeline化 | |
| 31 | モデル解釈（SHAP・PDP） | |
| 32 | 回帰の統計・相関と因果 | |
| 33 | House Prices EDA | |
| 34 | House Prices 前処理 | |
| 35 | モデル学習・アンサンブル | |
| 36 | 提出・振り返り | ← 今日 |

### セクション3：Phase 5 に向けた準備

Phase 5では「伝える」「まとめる」にフォーカスする。

準備チェックリスト：

```
□ 自分の分析テーマのデータが用意できている
□ EDA・前処理・モデリングを一通り実施した
□ モデルのCVスコアが記録されている
□ 主要な特徴量重要度を確認した
□ Phase 2の問いに対する暫定的な答えが出ている
□ Notebookが実行可能な状態になっている
```

## 今日のKaggle作業

### 問いの確認
House Prices の予測から「住宅価格に最も影響する要因」についてどんな示唆が得られたか？

### 作業手順
1. 最終提出ファイルを作成してKaggleにアップロードする
2. Public LBスコアを記録する
3. Phase 4 振り返り表を埋める

### 詰まったときの対処
- Discussion で検索するキーワード例：「house prices final submission」

### 振り返り
- 今日の「?」リスト：
  （Phase 5に持ち越したい疑問をメモ）

## 今日のまとめ
- 対数変換した目的変数を予測した場合は `expm1` で元のスケールに戻す
- Phase 1（Titanic）から Phase 4（House Prices）でデータ分析の深さが格段に上がった
- 次は「分析結果を人に伝える」スキルを磨く

## 次回予告
- Phase 5 開始：分析レポートの書き方（教科書Ch9）
