# [コマ14] pandas によるデータ操作（Udemy補足）

## 本日の目標
- DataFrameの基本操作（選択・フィルタ・集計・結合）を自在に使える
- `groupby` で集計・変換を効率的に実施できる
- `merge` / `concat` でデータを適切に結合できる

## 前回の振り返り
- Phase 2 完了。分析計画書を作成し、自分の問いと手法の対応を整理した

## データ準備（自習用：このまま実行できます）

```python
import pandas as pd
import numpy as np
import seaborn as sns

RANDOM_STATE = 42

# seaborn 同梱のtitanicデータセットを使用（ダウンロード不要）
df = sns.load_dataset('titanic')
print(df.shape)
df.head()
```

## 本編

### セクション1：選択・フィルタ・型変換

```python
# 列選択
df[['age', 'fare', 'class']].head()
df.loc[:, 'survived':'age'].head()    # スライス
df.iloc[:, 0:3].head()                # 位置指定

# 条件フィルタ
df[df['age'] >= 20].head()
df.query('age >= 20 and fare > 50').head()

# 型変換
df['class'] = df['class'].astype('category')
df.dtypes
```

### セクション2：groupby 集計・変換

```python
# 集計
df.groupby('class')['fare'].agg(['mean', 'std', 'count'])

# 複数列・複数集計
df.groupby(['class', 'sex']).agg(
    avg_fare=('fare', 'mean'),
    survived_rate=('survived', 'mean'),
    n=('survived', 'count'),
)

# transform（元のDataFrameのサイズを維持）
df['fare_normalized'] = df.groupby('class')['fare'].transform(
    lambda x: (x - x.mean()) / x.std()
)
df[['class', 'fare', 'fare_normalized']].head()

# apply（柔軟な処理）
def top_n(group, n=3):
    return group.nlargest(n, 'fare')

df.groupby('class', group_keys=False).apply(top_n, n=3)[['class', 'sex', 'fare']]
```

### セクション3：merge / concat

```python
# サンプルテーブルを作成して結合の動作を確認
left = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
right = pd.DataFrame({'id': [1, 2, 4], 'score': [80, 90, 70]})

merged = pd.merge(left, right, on='id', how='left')
print(merged)
# how: 'inner' / 'left' / 'right' / 'outer'

# 縦方向に結合
df1 = df.head(5)
df2 = df.tail(5)
combined = pd.concat([df1, df2], ignore_index=True)

# ピボット
pivot = df.pivot_table(
    values='fare',
    index='class',
    columns='sex',
    aggfunc='mean',
)
print(pivot)
```

## ハンズオン / 演習

1. 上記のtitanicデータで `class × sex` の生存率クロス集計を作成する
2. 自分の分析計画書で使うデータを読み込み、`groupby` で基本集計を実施する
3. `merge` で2つのテーブルを結合する（Kaggleデータの場合は train + test の情報結合等）
4. `transform` を使ってグループ内の正規化を試す

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？（EDA・前処理・集計）
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- `query` は条件フィルタを読みやすく書ける
- `groupby` + `transform` でグループ内の正規化や比較が簡単にできる
- `merge` の `how` の選択はデータのスコープを決める重要な設計判断

## 次回予告
- 基本統計量と確率分布の復習（Udemy補足）
