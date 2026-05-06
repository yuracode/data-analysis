# [コマ16] データ可視化（Udemy補足）

## 本日の目標
- 問いの種類（比較・分布・関係・推移）に応じたグラフを選択できる
- seaborn と matplotlib を組み合わせて図を整えられる
- 「伝わるグラフ」と「探索的なグラフ」の違いを意識できる

## 前回の振り返り
- 記述統計・確率分布・外れ値検出の実装を学んだ

## 本編

### セクション1：グラフの選択指針

| 目的 | グラフ | 関数 |
|------|--------|------|
| 分布を見る | ヒストグラム・KDE | `histplot`, `kdeplot` |
| カテゴリ比較 | 棒グラフ・箱ひげ図 | `barplot`, `boxplot` |
| 2変数の関係 | 散布図 | `scatterplot`, `regplot` |
| 時系列推移 | 折れ線グラフ | `lineplot` |
| 相関の全体像 | ヒートマップ | `heatmap` |
| 多変数の関係 | ペアプロット | `pairplot` |

### セクション2：seaborn の基本パターン

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')

# 例：Titanicデータ（or 自分のデータに読み替える）
df = sns.load_dataset('titanic')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 分布
sns.histplot(df['age'].dropna(), ax=axes[0,0], kde=True)
axes[0,0].set_title('年齢の分布')

# カテゴリ比較
sns.barplot(data=df, x='pclass', y='survived', ax=axes[0,1])
axes[0,1].set_title('クラス別生存率')

# 関係
sns.boxplot(data=df, x='pclass', y='age', ax=axes[1,0])
axes[1,0].set_title('クラス別年齢分布')

# 相関
num_cols = df.select_dtypes(include='number')
sns.heatmap(num_cols.corr(), annot=True, fmt='.2f', ax=axes[1,1])
axes[1,1].set_title('相関ヒートマップ')

plt.tight_layout()
plt.show()
```

### セクション3：「伝わるグラフ」の作り方

探索目的（自分向け）と発表目的（他者向け）でグラフのゴールが違う：

```python
# 発表向け：タイトル・ラベル・色を丁寧に設定
fig, ax = plt.subplots(figsize=(8, 4))

survival_by_sex = df.groupby('sex')['survived'].mean()
colors = ['#e74c3c', '#3498db']

survival_by_sex.plot(kind='bar', ax=ax, color=colors, edgecolor='white', width=0.5)
ax.set_title('性別による生存率の違い', fontsize=14, pad=12)
ax.set_xlabel('')
ax.set_ylabel('生存率')
ax.set_xticklabels(['女性', '男性'], rotation=0)
ax.set_ylim(0, 1)
ax.axhline(y=df['survived'].mean(), color='gray', linestyle='--', label='全体平均')
ax.legend()

for i, v in enumerate(survival_by_sex):
    ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=11)

plt.tight_layout()
plt.show()
```

## ハンズオン / 演習

1. 自分のデータで4種類（分布・比較・関係・推移）のグラフを1枚ずつ作る
2. そのうち1枚を「発表向け」に仕上げる（タイトル・ラベル・強調色）

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：EDA段階か発表段階か？
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- グラフの選択は「何を伝えたいか」から逆算する
- EDA段階では多くのグラフを素早く作り、発表では1枚を丁寧に仕上げる
- seaborn は探索向き、matplotlib の細かい設定は発表向きに使い分ける

## 次回予告
- 可視化演習：自分のデータを使ってEDAダッシュボードを作る
