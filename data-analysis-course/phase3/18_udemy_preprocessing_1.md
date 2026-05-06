# [コマ18] 前処理①：欠損値・外れ値・スケーリング（Udemy補足）

## 本日の目標
- 欠損値の補完戦略（削除・統計量・モデルベース）を使い分けられる
- スケーリング手法の違いと選択基準を説明できる
- Pipeline を使った前処理の一貫適用ができる

## 前回の振り返り
- EDAダッシュボードを作成し、問いに関連する変数の気づきとメモをまとめた

## データ準備（自習用：このまま実行できます）

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

df = sns.load_dataset('titanic')

# 数値カラムだけに絞ったDataFrame（このコマの実演用）
num_cols = ['age', 'fare', 'pclass', 'sibsp', 'parch']
data = df[num_cols + ['survived']].copy()

X = data[num_cols]
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
```

## 本編

### セクション1：欠損値補完の戦略

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# 欠損の種類を確認
print(X_train.isnull().mean().sort_values(ascending=False))

# 1. 単純補完（mean / median / most_frequent）
imp_median = SimpleImputer(strategy='median')
age_filled = imp_median.fit_transform(X_train[['age']])

# 2. KNN補完（近傍の値を使う）
imp_knn = KNNImputer(n_neighbors=5)
filled_knn = imp_knn.fit_transform(X_train[['age', 'fare']])

# 3. MICE（反復補完）：最も精度が高いが計算コスト大
imp_iter = IterativeImputer(random_state=RANDOM_STATE)
filled_iter = imp_iter.fit_transform(X_train)
print(f"MICE後の欠損数: {np.isnan(filled_iter).sum()}")
```

| 手法 | 向いているケース | 注意点 |
|------|----------------|--------|
| 中央値・平均値 | 欠損が少ない・単変量 | 分布を歪める可能性 |
| KNN | 欠損が少ない・多変量 | サンプルサイズに依存 |
| MICE | 欠損が多い | 計算コストが高い |

### セクション2：スケーリング

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler：平均0・標準偏差1（外れ値に敏感）
scaler_std = StandardScaler()

# MinMaxScaler：[0,1]に変換（外れ値に敏感）
scaler_mm = MinMaxScaler()

# RobustScaler：中央値・IQRベース（外れ値に頑健）
scaler_rb = RobustScaler()
```

| スケーラー | 向いているケース |
|-----------|----------------|
| StandardScaler | 外れ値が少ない・線形モデル |
| MinMaxScaler | 外れ値が少ない・NN系 |
| RobustScaler | 外れ値がある |
| スケーリング不要 | ツリー系（RF・GBM等） |

### セクション3：Pipeline で一貫適用

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(max_iter=500)),
])

pipe.fit(X_train, y_train)
print(f"テスト精度: {pipe.score(X_test, y_test):.3f}")
```

Pipeline を使うとtrainとtestに同じ変換が保証される（データリークを防ぐ）。

## ハンズオン / 演習

1. 自分のデータの欠損状況を確認し、補完戦略を決める
2. `SimpleImputer` と `KNNImputer` の両方で補完し、統計量がどう変わるか比較する
3. 数値変数を `StandardScaler` と `RobustScaler` でスケーリングし、分布の変化を確認する

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？（前処理・モデリング前）
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- 欠損値補完の選択は「欠損の量・パターン・後続モデル」で決める
- スケーリングはモデルによっては不要（ツリー系）
- Pipeline を使うとtrain・testで同じ変換が保証されデータリークを防げる

## 次回予告
- 前処理②：カテゴリ変数・特徴量エンジニアリング（Udemy補足）
