# [コマ17] 可視化演習：EDAダッシュボードを作る

## 本日の目標
- 自分の分析テーマのデータを使ってEDAを一通り実施できる
- 問いに関連する変数の関係を複数のグラフで確認できる
- 「気づいたこと・仮説」をグラフに紐付けてメモできる

## 前回の振り返り
- 目的別のグラフ選択と seaborn・matplotlib の整え方を学んだ

## データ準備（自習用：このまま実行できます）

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

# titanic を target=survived として扱う
df = sns.load_dataset('titanic')
df = df.rename(columns={'survived': 'target'})
print(df.shape, df['target'].value_counts(normalize=True).round(3).to_dict())
```

## 本編

### セクション1：EDAの進め方チェックリスト

```
□ データの形（行数・列数・型）を確認した
□ 欠損値を確認した
□ 目的変数の分布を確認した
□ 各特徴量の分布を確認した
□ 目的変数と特徴量の関係を確認した
□ 特徴量間の相関を確認した
□ 外れ値を確認した
```

### セクション2：EDAテンプレートコード

```python
# --- 1. 基本情報 ---
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

# --- 2. 目的変数の分布 ---
fig, ax = plt.subplots()
df['target'].hist(ax=ax, bins=30)
ax.set_title('目的変数の分布')
plt.show()

# --- 3. 数値変数の分布（一括） ---
df.select_dtypes(include='number').hist(figsize=(12, 8), bins=20, layout=(-1, 4))
plt.suptitle('数値変数のヒストグラム', y=1.02)
plt.tight_layout()
plt.show()

# --- 4. 相関ヒートマップ ---
corr = df.select_dtypes(include='number').corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('相関ヒートマップ')
plt.tight_layout()
plt.show()

# --- 5. 目的変数との関係（上位相関変数） ---
top_corr = corr['target'].abs().sort_values(ascending=False).head(6)
print(top_corr)
```

### セクション3：気づきメモの書き方

グラフを見たら必ず「So What?」を書く：

```markdown
## EDA 気づきメモ

### グラフ：年齢×目的変数の散布図
- 観察：30〜45歳の層で目的変数が高い傾向がある
- 仮説：この年齢層に特有の行動パターンがある可能性
- 次のアクション：年齢ビンを作成して集計する

### グラフ：相関ヒートマップ
- 観察：特徴量AとBの相関が0.92と高い
- 仮説：多重共線性の可能性あり
- 次のアクション：VIF（分散膨張因子）を確認する
```

## 演習

### お題
自分の分析テーマのデータを使い、EDAを実施する。

### やること
1. EDAチェックリストを順番に実行する
2. 気になったグラフを3枚以上選び、気づきメモを書く
3. 「この先の分析で確認すべきこと」を箇条書きにまとめる

### 提出物
- EDAのJupyter Notebook（or コードと気づきメモ）

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- EDAは「何かを証明する」のではなく「データを理解する」プロセス
- グラフを見るたびに「So What?」を問いかけてメモに残す習慣をつける
- 相関が高い特徴量は多重共線性のリスクになることを頭に入れておく

## 次回予告
- データ前処理①：欠損値・外れ値・スケーリング（Udemy補足）
