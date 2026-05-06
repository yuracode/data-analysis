# [コマ4] Titanic モデル学習・提出

## 本日の目標
- 複数の分類モデルを学習・比較できる
- 交差検証でモデルの汎化性能を評価できる
- Kaggleに予測結果を提出してスコアを確認できる

## 前回の振り返り
- 欠損補完・エンコーディング・特徴量エンジニアリングを実施し、`X_train` / `X_test` を作成した

## 本編

### セクション1：モデル学習と比較

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42

# ※前処理済みの X_train, y_train, X_test を読み込んでいる前提

models = {
    'LogReg':  LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    'RF':      RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'GBM':     GradientBoostingClassifier(random_state=RANDOM_STATE),
    'SVC':     SVC(random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### セクション2：特徴量重要度の確認

```python
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feat_imp.plot(kind='bar', figsize=(8, 4), title='Feature Importance')
plt.tight_layout()
plt.show()
```

### セクション3：予測と提出ファイル作成

```python
# 最良モデルで予測（例：RF）
best_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)

test = pd.read_csv('data/test.csv')
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
submission.to_csv('submission.csv', index=False)
print(submission.head())
```

## ハンズオン / 演習

1. 上記の4モデルを実行し、CVスコアを比較する
2. 最もスコアが高いモデルを選んで `submission.csv` を作成する
3. Kaggleにアップロードしてスコアを記録する（Public LBスコアをメモ）
4. ハイパーパラメータを1つ変えてスコアが変わるか試す

## ？リスト（疑問を記録しよう）
- CVスコアとKaggleのPublic LBスコアが違うのはなぜ？
- RandomForestのn_estimatorsを増やすとどうなる？

## 今日のまとめ
- 交差検証（CV）でモデルを比較し、汎化性能を見積もる
- 特徴量重要度でモデルが何を見ているか確認できる
- Kaggleへの提出によって「外部の正解データ」で検証できる

## 次回予告
- 教科書Ch1-2の内容をTitanicの作業と対応させて整理する
