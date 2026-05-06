# [コマ31] 教科書Ch7：データの比較を通じた解釈・考察

## 本日の目標
- データを「比較」によって解釈する考え方（時系列・属性・基準値）を説明できる
- 比較の結果から仮説を立てて検証できる
- モデル予測の根拠を確認する補助的手段として SHAP・PDP を使える

## 前回の振り返り
- Titanicを教科書フレームワークで再分析し、Pipeline化とStratified KFoldで精度を改善した

## 本編

### セクション1：教科書Ch7「データの比較」3つの軸

教科書Ch7は「データは比較してはじめて意味になる」ことを示す。比較には3軸ある：

| 比較軸 | 例 |
|--------|-----|
| 時系列の比較 | 前年同月比・移動平均・前期比 |
| 属性の比較 | 性別・年齢層・地域・チャネルでの差 |
| 基準値との比較 | 目標値・業界平均・ベンチマーク |

### セクション2：Titanicでの「比較による解釈」演習

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt

RANDOM_STATE = 42

df = sns.load_dataset('titanic').dropna(subset=['age', 'fare', 'embarked']).copy()

# 全体の生存率を「基準値」として算出
overall_rate = df['survived'].mean()
print(f"全体の生存率: {overall_rate:.3f}")

# 属性ごとの比較
by_sex    = df.groupby('sex')['survived'].mean()
by_pclass = df.groupby('pclass')['survived'].mean()
by_age    = df.groupby(pd.cut(df['age'], bins=[0,18,35,60,100]))['survived'].mean()

# 「全体平均との差」で表現すると意思決定に直結する
diff_table = pd.DataFrame({
    '生存率':        pd.concat([by_sex, by_pclass, by_age]),
    '全体との差':    pd.concat([by_sex, by_pclass, by_age]) - overall_rate,
})
print(diff_table.round(3))
```

### セクション3：モデル予測を「比較」で解釈する

教科書Ch7の延長として、機械学習モデルの予測も「全体平均との差を説明する」ためにSHAPやPDPが使える：

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

df['family_size'] = df['sibsp'] + df['parch'] + 1
df['sex_enc']     = (df['sex'] == 'female').astype(int)
df['embarked_enc']= df['embarked'].astype('category').cat.codes

feature_cols = ['pclass', 'age', 'fare', 'family_size', 'sex_enc', 'embarked_enc']
X = df[feature_cols]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

model = GradientBoostingClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Partial Dependence Plot：「ある特徴量を変えたとき、平均的に予測がどう変わるか」
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=['age', 'fare', 'family_size'],
    ax=axes,
)
plt.suptitle('Partial Dependence Plot（特徴量の効果を「比較」で見る）')
plt.tight_layout()
plt.show()
```

```python
# SHAP：1サンプルが「全体平均からなぜズレたか」を分解する
import shap

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

shap.plots.beeswarm(shap_values)              # 全体傾向
shap.plots.waterfall(shap_values[0])          # 個別サンプル：基準値からの差を分解
```

## ハンズオン / 演習

1. Titanicデータで「全体生存率との差が最も大きい属性カテゴリ」を3つ列挙する
2. その3つに対して、教科書Ch7の3軸（時系列・属性・基準値）のどれに当てはまるかを書く
3. SHAP値（または特徴量重要度）の上位3変数が、上記の属性比較と一致するか確認する
4. 自分の分析テーマでも「比較で意味を出せる軸」を3つ書き出す

## 今日のまとめ
- データは比較してはじめて意味になる（教科書Ch7の核心）
- 比較は「時系列・属性・基準値」の3軸を意識する
- モデルの予測解釈（SHAP・PDP）も「基準値（平均予測）との差」を分解する道具

## 次回予告
- 教科書Ch8：統計解析でデータ分析の幅を広げる
