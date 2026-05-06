# [コマ34] House Prices 前処理

## 本日の目標
- House Prices の欠損パターンを分類し、適切に補完できる
- 多数のカテゴリ変数を効率的にエンコーディングできる
- 新しい特徴量を生成してモデルの入力として整備できる

## 前回の振り返り
- House Prices のEDAで構造を把握し、SalePriceの対数変換の理由を確認した

## 本編

### セクション1：欠損値の分類と補完

```python
import pandas as pd
import numpy as np

RANDOM_STATE = 42

train = pd.read_csv('data/house_prices/train.csv')
test  = pd.read_csv('data/house_prices/test.csv')

# SalePrice の対数変換
train['LogSalePrice'] = np.log1p(train['SalePrice'])

# House Prices の欠損は「該当なし」が多い
# NA = "None"（設備なし）として補完すべきカラム
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']

zero_cols  = ['GarageYrBlt', 'GarageArea', 'GarageCars',
              'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
              'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

mode_cols  = ['MSZoning', 'Electrical', 'Exterior1st', 'Exterior2nd',
              'KitchenQual', 'Functional', 'SaleType', 'Utilities']

for df in [train, test]:
    for c in none_cols:
        df[c] = df[c].fillna('None')
    for c in zero_cols:
        df[c] = df[c].fillna(0)
    for c in mode_cols:
        df[c] = df[c].fillna(df[c].mode()[0])
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
```

### セクション2：特徴量エンジニアリング

```python
for df in [train, test]:
    df['TotalSF']     = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath']   = (df['FullBath'] + 0.5 * df['HalfBath'] +
                         df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    df['HouseAge']    = df['YrSold'] - df['YearBuilt']
    df['RemodelAge']  = df['YrSold'] - df['YearRemodAdd']
    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['TotalPorch']  = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                         df['3SsnPorch'] + df['ScreenPorch'])
    df['HasPool']     = (df['PoolArea'] > 0).astype(int)
    df['HasGarage']   = (df['GarageArea'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)

    # 品質スコア統合
    quality_map = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0}
    for col in ['ExterQual','ExterCond','BsmtQual','KitchenQual','HeatingQC']:
        df[f'{col}_num'] = df[col].map(quality_map).fillna(0)
```

### セクション3：エンコーディングと最終整備

```python
from sklearn.preprocessing import LabelEncoder

# カテゴリ変数の確認
cat_cols = train.select_dtypes(include='object').columns.tolist()

# 全データを合わせてLabelEncoder
all_data = pd.concat([train.drop(['SalePrice','LogSalePrice'], axis=1), test])

for col in cat_cols:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str))

# 分割
n_train = len(train)
train_enc = all_data[:n_train]
test_enc  = all_data[n_train:]

feature_cols = [c for c in train_enc.columns if c not in ['Id']]
X_train = train_enc[feature_cols]
y_train = train['LogSalePrice']
X_test  = test_enc[feature_cols]
```

## 今日のKaggle作業

### 問いの確認
前処理後のデータで品質の高い特徴量セットを作れているか？

### 作業手順
1. 欠損補完後に `isnull().sum().sum()` が0になることを確認する
2. 新特徴量（TotalSF・HouseAge等）とSalePriceの相関を確認する
3. エンコーディング後のXとyの形状を確認する

### 詰まったときの対処
- Discussion で検索するキーワード例：「house prices missing values」「house prices feature engineering」

### 振り返り
- 今日の「?」リスト：

## 今日のまとめ
- House Pricesの欠損「NA」は「設備なし」を意味することが多く、"None"や0で補完する
- ドメイン知識（不動産の価値要因）から特徴量を設計すると精度が上がる
- 補完・エンジニアリング後は必ず品質チェック（欠損・型・統計量）を実施する

## 次回予告
- House Prices モデル学習：前処理済みデータでモデルを比較・最適化する
