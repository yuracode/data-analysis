# [コマ23] 機械学習①：分類モデルの選択と評価（Udemy補足）

## 本日の目標
- 主要な分類モデルの特徴と向いているケースを説明できる
- 混同行列・精度・再現率・F1スコアを使い分けて評価できる
- ROC-AUCの意味とビジネスでの使い方を説明できる

## 前回の振り返り
- 相関・回帰分析の統計的側面とVIFによる多重共線性診断を学んだ

## データ準備（自習用：このまま実行できます）

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE = 42

df = sns.load_dataset('titanic').copy()
df['family_size'] = df['sibsp'] + df['parch'] + 1

feature_cols = ['age', 'fare', 'family_size', 'sex', 'embarked', 'pclass']
X = df[feature_cols]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

num_cols = ['age', 'fare', 'family_size']
cat_cols = ['sex', 'embarked', 'pclass']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('scl', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols),
])
```

## 本編

### セクション1：分類モデルの比較

| モデル | 解釈性 | 精度 | 向いているケース |
|--------|--------|------|----------------|
| ロジスティック回帰 | 高 | 中 | 線形分離可能・係数解釈が必要 |
| 決定木 | 高 | 低〜中 | ルールベース説明が必要 |
| ランダムフォレスト | 中 | 高 | 汎用的 |
| GradientBoosting | 低 | 高 | スコアを最大化したい |
| SVM | 低 | 中〜高 | 中〜高次元 |

ビジネス文脈：「なぜそう予測したか」を説明する必要がある場合はロジスティック回帰や決定木を選ぶ。

### セクション2：評価指標の選択

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt

# モデル学習
pipe = Pipeline([
    ('prep',  preprocessor),
    ('model', GradientBoostingClassifier(random_state=RANDOM_STATE)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)

# 混同行列
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 分類レポート
print(classification_report(y_test, y_pred, target_names=['死亡', '生存']))

# ROC-AUC
auc = roc_auc_score(y_test, y_prob[:, 1])
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('偽陽性率 (FPR)')
plt.ylabel('真陽性率 (TPR)')
plt.title('ROC曲線')
plt.legend()
plt.show()
```

| 指標 | 意味 | 使う場面 |
|------|------|---------|
| 精度（Accuracy） | 全体の正解率 | クラス均衡のとき |
| 適合率（Precision） | 陽性予測の正確さ | 偽陽性コストが高いとき |
| 再現率（Recall） | 実陽性の検出率 | 偽陰性コストが高いとき（病気見逃し等） |
| F1スコア | 精度と再現率の調和平均 | クラス不均衡のとき |
| AUC | 閾値に依存しない総合評価 | モデル比較 |

### セクション3：クラス不均衡への対処

```python
from sklearn.ensemble import RandomForestClassifier

# 'balanced' を渡すだけで、クラス頻度の逆数で自動的に重み付けされる
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE
)
```

## ハンズオン / 演習

1. 自分のデータ（分類問題の場合）で3種類のモデルを学習し、F1スコアで比較する
2. ROC曲線を描いてAUCを確認する
3. 混同行列を見て「偽陰性と偽陽性のどちらがビジネス上コストが高いか」を考える

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- モデル選択は「精度」だけでなく「解釈性の必要度」も考慮する
- 評価指標は「何のミスが許されないか」というビジネス要件で選ぶ
- クラス不均衡は `class_weight='balanced'` で手軽に対処できる

## 次回予告
- 機械学習②：回帰モデルと評価指標（Udemy補足）
