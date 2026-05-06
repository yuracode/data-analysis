# [コマ14] pandas によるデータ操作（Udemy補足）

## 本日の目標
- DataFrameの基本操作（選択・フィルタ・集計・結合）を自在に使える
- `groupby` で集計・変換を効率的に実施できる
- `merge` / `concat` でデータを適切に結合できる

## 前回の振り返り
- Phase 2 完了。分析計画書を作成し、自分の問いと手法の対応を整理した

## 本編

### セクション1：選択・フィルタ・型変換

```python
import pandas as pd
import numpy as np

RANDOM_STATE = 42

# 列選択
df[['col_a', 'col_b']]
df.loc[:, 'col_a':'col_c']    # スライス
df.iloc[:, 0:3]               # 位置指定

# 条件フィルタ
df[df['age'] >= 20]
df.query('age >= 20 and score > 80')

# 型変換
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')
```

### セクション2：groupby 集計・変換

```python
# 集計
df.groupby('group')['value'].agg(['mean', 'std', 'count'])

# 複数列・複数集計
df.groupby(['group', 'sub'])['value'].agg(
    avg=('value', 'mean'),
    total=('value', 'sum'),
)

# transform（元のDataFrameのサイズを維持）
df['value_normalized'] = df.groupby('group')['value'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# apply（柔軟な処理）
def top_n(group, n=3):
    return group.nlargest(n, 'score')

df.groupby('group').apply(top_n, n=3)
```

### セクション3：merge / concat

```python
# 結合
merged = pd.merge(df_left, df_right, on='key', how='left')
# how: 'inner' / 'left' / 'right' / 'outer'

# 縦方向に結合
combined = pd.concat([df1, df2], ignore_index=True)

# ピボット
pivot = df.pivot_table(
    values='sales',
    index='month',
    columns='product',
    aggfunc='sum',
    fill_value=0
)
```

## ハンズオン / 演習

1. 自分の分析計画書で使うデータを読み込み、`groupby` で基本集計を実施する
2. `merge` で2つのテーブルを結合する（Kaggleデータの場合は train + test の情報結合等）
3. `transform` を使ってグループ内の正規化を試す

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
