# [コマ26] 機械学習④：特徴量エンジニアリングの実践（Udemy補足）

## 本日の目標
- ドメイン知識を活かした特徴量生成の考え方を説明できる
- SHAP値を使ってモデルの予測根拠を可視化できる
- 特徴量の重要度を複数の手法で確認・比較できる

## 前回の振り返り
- 学習曲線・交差検証・GridSearchCVでモデル評価と過学習対策を実践した

## データ準備（自習用：このまま実行できます）

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

df = sns.load_dataset('titanic').copy()
df = df.dropna(subset=['age', 'fare', 'embarked']).reset_index(drop=True)
```

## 本編

### セクション1：ドメイン知識からの特徴量生成

特徴量エンジニアリングは「データの変換」より「ビジネスの理解」が本質：

```python
# Titanic でのドメイン知識ベースの特徴量
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone']    = (df['family_size'] == 1).astype(int)
df['fare_per_person'] = df['fare'] / df['family_size']
df['has_cabin']   = df['deck'].notna().astype(int)

# 名前タイトルの抽出はこの簡易データには無いので
# 代わりにビン化を実装
df['age_bin'] = pd.cut(df['age'], bins=[0,12,18,35,60,100],
                       labels=['child','teen','young','adult','senior'])

# 数値→Ordinal変換
for col in ['sex', 'embarked', 'age_bin']:
    df[f'{col}_enc'] = df[col].astype('category').cat.codes

feature_cols = ['pclass', 'age', 'fare', 'family_size', 'is_alone',
                'fare_per_person', 'has_cabin', 'sex_enc', 'embarked_enc', 'age_bin_enc']
X = df[feature_cols]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
```

### セクション2：SHAP値による解釈

```python
import shap
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 全体の特徴量重要度
shap.plots.beeswarm(shap_values)

# 1サンプルの予測根拠
shap.plots.waterfall(shap_values[0])
```

ビジネス活用：「この顧客の離脱リスクが高い理由は○○が要因」のように説明できる。

### セクション3：特徴量重要度の比較

```python
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
import seaborn as sns

# 1. モデルの特徴量重要度
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# 2. Permutation Importance（モデルに依存しない）
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE)
perm_imp = pd.Series(result.importances_mean, index=X_train.columns).sort_values(ascending=False)

# 比較
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
feat_imp.nlargest(10).sort_values().plot(kind='barh', ax=axes[0], title='モデル重要度 (Top10)')
perm_imp.nlargest(10).sort_values().plot(kind='barh', ax=axes[1], title='Permutation重要度 (Top10)')
plt.tight_layout()
plt.show()
```

## ハンズオン / 演習

1. 自分のデータで「ドメイン知識からこの特徴量を追加すると精度が上がるはず」という仮説を立て、実装する
2. SHAP値を計算し、予測に最も影響する特徴量を確認する
3. モデル重要度とPermutation重要度の上位が一致しているか比較する

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- 特徴量エンジニアリングはドメイン知識の言語化がスコア向上の鍵
- SHAP値は「モデルが何を見て予測しているか」を説明できる
- モデル重要度とPermutation重要度の不一致は多重共線性のサイン

## 次回予告
- 機械学習⑤：アンサンブル学習（Udemy補足）
