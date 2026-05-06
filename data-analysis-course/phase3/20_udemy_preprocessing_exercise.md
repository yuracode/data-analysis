# [コマ20] 前処理演習：前処理パイプラインを完成させる

## 本日の目標
- コマ18・19の内容を統合し、自分のデータの前処理パイプラインを完成できる
- `ColumnTransformer` で数値・カテゴリ変数を一括処理できる
- 前処理後のデータの品質を定量的に確認できる

## 前回の振り返り
- カテゴリ変数のエンコーディング・特徴量エンジニアリング・特徴量選択を学んだ

## データ準備（自習用：このまま実行できます）

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

df = sns.load_dataset('titanic').copy()
df['family_size'] = df['sibsp'] + df['parch'] + 1

feature_cols = ['age', 'fare', 'family_size', 'sex', 'embarked', 'pclass']
X = df[feature_cols]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
```

## 本編

### セクション1：ColumnTransformer で一括前処理

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_cols = ['age', 'fare', 'family_size']
cat_cols = ['sex', 'embarked', 'pclass']

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols),
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)  # transformのみ（fitしない）

print(f"前処理後の特徴量数: {X_train_proc.shape[1]}")
```

### セクション2：前処理後の品質チェック

```python
# 変換後のDataFrame化
ohe_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
all_cols = num_cols + list(ohe_cols)

X_train_df = pd.DataFrame(X_train_proc, columns=all_cols)

# チェック項目
print("欠損値:", X_train_df.isnull().sum().sum())
print("無限値:", np.isinf(X_train_df.values).sum())
print("統計量:\n", X_train_df.describe().round(2))
```

### セクション3：前処理の設計記録

前処理の決定はノートブックのMarkdownセルに記録する習慣をつける：

```markdown
## 前処理の設計記録

| カラム | 処理 | 理由 |
|--------|------|------|
| age | 中央値補完 → StandardScaler | 欠損約20%、外れ値少 |
| fare | 中央値補完 → StandardScaler | 外れ値あるが線形モデル用 |
| family_size | スケーリングのみ | 派生変数 |
| sex | One-Hot | カーディナリティ2 |
| embarked | 最頻値補完 → One-Hot | 欠損少、カーディナリティ3 |
| pclass | One-Hot | 順序ありだが今回はOne-Hotで扱う |
```

## 演習

### お題
自分のデータに `ColumnTransformer` を使った前処理パイプラインを実装する。

### やること
1. 数値変数・カテゴリ変数のリストを整理する
2. 各カラムの補完方法を決めてコメントに理由を書く
3. `fit_transform` 後に品質チェックを実行し、問題がないか確認する
4. 前処理の設計記録表を作成する

### 提出物
- 前処理パイプラインのコード（Notebook）
- 前処理設計記録表

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- `ColumnTransformer` + `Pipeline` でtrain・testへの一貫適用が保証される
- 前処理の決定は「なぜその手法を選んだか」を記録することが重要
- 品質チェック（欠損・無限値・統計量）は前処理完了の確認ステップとして必須

## 次回予告
- 統計的仮説検定①：t検定・カイ二乗検定・分散分析（Udemy補足）
