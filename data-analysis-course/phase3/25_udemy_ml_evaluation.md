# [コマ25] 機械学習③：モデル評価・過学習対策・チューニング（Udemy補足）

## 本日の目標
- 学習曲線で過学習・アンダーフィットを診断できる
- 交差検証の種類（K-Fold・Stratified等）を使い分けられる
- GridSearchCV・Optunaでハイパーパラメータを最適化できる

## 前回の振り返り
- 回帰モデルの評価指標（RMSE・MAE・R²）と正則化（Ridge・Lasso）を学んだ

## データ準備（自習用：このまま実行できます）

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

df = sns.load_dataset('titanic').copy()
df['family_size'] = df['sibsp'] + df['parch'] + 1
df = df.dropna(subset=['age', 'fare', 'embarked'])
df['sex_enc']     = (df['sex'] == 'female').astype(int)
df['embarked_enc']= df['embarked'].astype('category').cat.codes

feature_cols = ['age', 'fare', 'family_size', 'sex_enc', 'embarked_enc', 'pclass']
X = df[feature_cols]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
```

## 本編

### セクション1：過学習の診断

```python
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(8, 4))
plt.plot(train_sizes, train_scores.mean(axis=1), label='訓練スコア')
plt.plot(train_sizes, val_scores.mean(axis=1), label='検証スコア')
plt.fill_between(train_sizes,
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
plt.fill_between(train_sizes,
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
plt.legend()
plt.xlabel('訓練サンプル数')
plt.ylabel('スコア')
plt.title('学習曲線')
plt.show()
```

| 状態 | 訓練スコア | 検証スコア | 対処 |
|------|-----------|-----------|------|
| 過学習 | 高 | 低 | 正則化・特徴量削減・データ増量 |
| 未学習 | 低 | 低 | モデルの複雑化・特徴量追加 |
| 良好 | 高 | 高 | 維持 |

### セクション2：交差検証の種類

```python
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score

# 通常の K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 層化K-Fold（クラス不均衡時に有効）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print('StratifiedKFold:', cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy').mean())

# 時系列データ用（未来データで検証）
tscv = TimeSeriesSplit(n_splits=5)
```

### セクション3：ハイパーパラメータチューニング

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}

gs = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    scoring='f1',
    n_jobs=-1,
)
gs.fit(X_train, y_train)

print(f"最良パラメータ: {gs.best_params_}")
print(f"CVスコア:       {gs.best_score_:.4f}")
```

## ハンズオン / 演習

1. 自分のモデルで学習曲線を描き、過学習・未学習のどちらか確認する
2. GridSearchCV で主要なハイパーパラメータを2〜3個チューニングする
3. チューニング前後のCVスコアを比較する

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- 学習曲線は「モデルの何が問題か」を診断する最初のツール
- 時系列データには必ずTimeSeriesSplitを使う（未来データで学習しない）
- GridSearchは計算コストが高いので、まず重要なパラメータに絞る

## 次回予告
- 機械学習④：特徴量エンジニアリングの実践（Udemy補足）
