# [コマ19] 前処理②：カテゴリ変数・特徴量エンジニアリング（Udemy補足）

## 本日の目標
- カテゴリ変数のエンコーディング手法を使い分けられる
- 日付・テキストから特徴量を生成できる
- 特徴量選択の基本手法を実装できる

## 前回の振り返り
- 欠損値補完・スケーリング・Pipeline による一貫適用を学んだ

## 本編

### セクション1：カテゴリ変数のエンコーディング

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce  # pip install category-encoders

RANDOM_STATE = 42

# 1. One-Hot Encoding（カーディナリティが低い場合）
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit_transform(df[['color']])

# pandas でも可
pd.get_dummies(df['color'], prefix='color')

# 2. Ordinal Encoding（順序に意味がある場合）
order = ['low', 'mid', 'high']
df['rank_enc'] = df['rank'].map({v: i for i, v in enumerate(order)})

# 3. Target Encoding（カーディナリティが高い場合）
te = ce.TargetEncoder(cols=['city'])
te.fit(X_train, y_train)
X_train_enc = te.transform(X_train)
X_test_enc  = te.transform(X_test)
# ※ trainだけでfitすること（データリーク防止）
```

| 手法 | カーディナリティ | 注意点 |
|------|----------------|--------|
| One-Hot | 低（〜10） | 次元が増える |
| Ordinal | 順序あり | 順序の設定が必要 |
| Target Encoding | 高（10〜） | testへの適用時にデータリーク注意 |

### セクション2：特徴量エンジニアリング

```python
# 日付から特徴量を生成
df['date'] = pd.to_datetime(df['date'])
df['year']      = df['date'].dt.year
df['month']     = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# 数値変数の交互作用
df['age_x_fare'] = df['age'] * df['fare']
df['age_per_pclass'] = df['age'] / df['pclass']

# ビニング（連続値をカテゴリ化）
df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100],
                       labels=['child', 'young', 'adult', 'senior'])
```

### セクション3：特徴量選択

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# 統計的フィルタ
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)
selected_cols = X_train.columns[selector.get_support()]

# 特徴量重要度（モデルベース）
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns)
top_features = feat_imp.nlargest(10).index
```

## ハンズオン / 演習

1. 自分のデータのカテゴリ変数を確認し、エンコーディング手法を選んで実装する
2. 日付カラムがある場合は `dt` アクセサで特徴量を生成する
3. RandomForestの特徴量重要度で上位5変数を確認する

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- カテゴリ変数のエンコーディングは「カーディナリティ」と「後続モデル」で選ぶ
- 特徴量エンジニアリングはドメイン知識×データの掛け算で価値が出る
- 特徴量選択は「多すぎる特徴量」の過学習リスクを下げる

## 次回予告
- 前処理演習：自分のデータで前処理パイプラインを完成させる
