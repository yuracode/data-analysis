# [コマ30] 教科書Ch6 続き：Titanicを教科書フレームワークで再分析

## 本日の目標
- 教科書Ch6のフレームワークを使ってTitanic分析を振り返り・改善できる
- Phase 1のコードと比較して「改善点」を言語化できる
- 分析設計の改善サイクル（Plan-Do-Check-Act）を回す経験を積む

## 前回の振り返り
- 分析設計の3軸（問いの明確さ・データの適切さ・手法の妥当性）とデータリーク防止を学んだ

## 本編

### セクション1：Phase 1 の分析を振り返る

Phase 1（コマ2〜4）でのTitanic分析の課題を整理する：

| 項目 | Phase 1 での対応 | 改善の余地 |
|------|----------------|-----------|
| 前処理 | 中央値補完・LabelEncoder | Pipelineで一括管理 |
| 特徴量 | 基本的なエンジニアリング | CabinのデコードやTicketの分析 |
| モデル評価 | CV（5-fold） | Stratified KFold に変更 |
| ハイパーパラメータ | デフォルト | GridSearchCVでチューニング |

### セクション2：改善版 Titanic パイプライン

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

RANDOM_STATE = 42

train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

# 特徴量エンジニアリング
for df in [train, test]:
    df['Title']      = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
    df['HasCabin']   = df['Cabin'].notna().astype(int)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})

num_cols = ['Age', 'Fare', 'FamilySize']
cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'HasCabin']

X_train = train[num_cols + cat_cols]
y_train = train['Survived']
X_test  = test[num_cols + cat_cols]

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
    ]), num_cols),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols),
])

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', GradientBoostingClassifier(random_state=RANDOM_STATE)),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

### セクション3：改善前後の比較

```python
# 提出ファイル作成
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission_v2.csv', index=False)
```

Phase 1 と今回のKaggle スコアを比較し、改善点と結果を記録する。

## 今日のKaggle作業

### 問いの確認
Titanicの「誰が生き残ったか」に対して、より精度の高い予測ができるか？

### 作業手順
1. 改善版パイプラインのコードを実装する
2. CVスコアを Phase 1 と比較する
3. Kaggleに提出してスコアを比較する

### 詰まったときの対処
- Discussion で検索するキーワード例：「feature engineering titanic」「best score titanic」

### 振り返り
- 今日の「?」リスト：
  （例）なぜ GradientBoosting が RF より良い場合があるのか？

## 今日のまとめ
- 同じデータでも設計を見直すことで精度が改善できる
- Pipelineでコードの保守性と安全性（リーク防止）が向上する
- 「改善した理由」を言語化することが次の問いを生む

## 次回予告
- 教科書Ch7：モデルの解釈と「なぜそう予測したか」の説明
