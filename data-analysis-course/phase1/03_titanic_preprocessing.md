# [コマ3] Titanic 前処理

## 本日の目標
- 欠損値補完の基本戦略を選択・実装できる
- カテゴリ変数をモデルに渡せる形に変換できる
- train/testで同じ処理を一貫して適用できる

## 前回の振り返り
- EDAで `Age`・`Cabin` に欠損、女性・1等客が生存率高いことを確認した

## 本編

### セクション1：欠損値補完

```python
import pandas as pd
import numpy as np

RANDOM_STATE = 42

train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

# Age：中央値補完
age_median = train['Age'].median()
train['Age'] = train['Age'].fillna(age_median)
test['Age']  = test['Age'].fillna(age_median)

# Embarked：最頻値補完
embarked_mode = train['Embarked'].mode()[0]
train['Embarked'] = train['Embarked'].fillna(embarked_mode)
test['Embarked']  = test['Embarked'].fillna(embarked_mode)

# Fare（testのみ欠損）：中央値補完
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# Cabin：欠損フラグ化（欠損が多すぎるため削除or有無のみ）
train['HasCabin'] = train['Cabin'].notna().astype(int)
test['HasCabin']  = test['Cabin'].notna().astype(int)
```

### セクション2：特徴量エンジニアリング

```python
# 家族人数
for df in [train, test]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# タイトル抽出
for df in [train, test]:
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare'
    )
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
```

### セクション3：エンコーディング

```python
from sklearn.preprocessing import LabelEncoder

cat_cols = ['Sex', 'Embarked', 'Title']
le = LabelEncoder()

for col in cat_cols:
    combined = pd.concat([train[col], test[col]])
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col]  = le.transform(test[col])

feature_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
                'FamilySize', 'IsAlone', 'HasCabin', 'Title']

X_train = train[feature_cols]
y_train = train['Survived']
X_test  = test[feature_cols]
```

## ハンズオン / 演習

1. 上記コードを実行し、`X_train.isnull().sum()` がすべて0になることを確認する
2. `FamilySize` と生存率の関係をグラフで確認する
3. `Title` ごとの生存率を集計して仮説を立てる

```python
train['Title_raw'] = pd.read_csv('data/train.csv')['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
train.groupby('Title_raw')['Survived'].agg(['mean','count'])
```

## ？リスト（疑問を記録しよう）
- 中央値補完と平均値補完はどう使い分ける？
- train の統計量を test に適用する理由は？

## 今日のまとめ
- 欠損補完は「trainの統計量でtestも埋める」が基本
- カテゴリ変数はLabelEncoderまたはone-hotで数値化する
- 特徴量エンジニアリングで新しい情報を生み出せる（Family、Title等）

## 次回予告
- 前処理済みデータでモデルを学習し、Kaggleに提出する
