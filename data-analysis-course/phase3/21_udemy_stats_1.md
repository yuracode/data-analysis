# [コマ21] 統計的仮説検定①：t検定・カイ二乗検定・ANOVA（Udemy補足）

## 本日の目標
- 仮説検定の基本的な考え方（帰無仮説・p値・有意水準）を説明できる
- t検定・カイ二乗検定・ANOVAを使い分けられる
- p値の解釈の落とし穴（p値崇拝）を説明できる

## 前回の振り返り
- `ColumnTransformer` + `Pipeline` で前処理を一括実装し、品質チェックを実施した

## 本編

### セクション1：仮説検定の基本

```
帰無仮説（H₀）：差がない・関係がない
対立仮説（H₁）：差がある・関係がある

p値：帰無仮説が正しい場合に、観測データ以上に極端な結果が
     起きる確率

有意水準（α）：通常0.05（5%）
→ p < 0.05 のとき「帰無仮説を棄却する」
```

### セクション2：手法の選択

| 問い | 変数の型 | 手法 |
|------|---------|------|
| 2群の平均に差があるか | 連続 vs 2群 | t検定 |
| 3群以上の平均に差があるか | 連続 vs 多群 | 一元配置ANOVA |
| 2変数の独立性を検定する | カテゴリ vs カテゴリ | カイ二乗検定 |
| 2変数の相関を検定する | 連続 vs 連続 | ピアソン相関係数のt検定 |

```python
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

RANDOM_STATE = 42

df = sns.load_dataset('titanic').dropna(subset=['age', 'fare', 'embarked'])

# 1. 独立2標本t検定（性別ごとの運賃に差があるか）
fare_male   = df[df['sex'] == 'male']['fare']
fare_female = df[df['sex'] == 'female']['fare']
t_stat, p_value = stats.ttest_ind(fare_male, fare_female, equal_var=False)
print(f"[t検定 fare] t={t_stat:.3f}, p={p_value:.4f}")

# 2. 一元配置ANOVA（クラスごとの年齢に差があるか）
groups = [df[df['pclass'] == c]['age'] for c in df['pclass'].unique()]
f_stat, p_value = stats.f_oneway(*groups)
print(f"[ANOVA age] F={f_stat:.3f}, p={p_value:.4f}")

# 3. カイ二乗検定（性別と生存の独立性）
contingency = pd.crosstab(df['sex'], df['survived'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"[カイ二乗 sex×survived] χ²={chi2:.3f}, p={p:.4f}")
```

### セクション3：p値の解釈の注意点

- **p < 0.05 = 実務的に重要、ではない**
  - サンプルサイズが大きければ小さな差でも有意になる
  - 効果量（Cohen's d 等）を一緒に確認する
- **p > 0.05 = 差がない、ではない**
  - 「差を検出できなかった」にすぎない
- **多重検定**：検定を繰り返すと偽陽性が増える → Bonferroni補正等

```python
# 効果量の計算（Cohen's d）
def cohens_d(group1, group2):
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    return (group1.mean() - group2.mean()) / pooled_std

d = cohens_d(fare_female, fare_male)
print(f"Cohen's d = {d:.3f}")
# |d| < 0.2: 小, 0.2–0.5: 中, > 0.8: 大
```

## ハンズオン / 演習

1. 自分のデータで「2つのグループ間に差があるか」という問いを立て、t検定を実施する
2. p値と効果量を両方報告する
3. カテゴリ変数間の独立性をカイ二乗検定で確認する

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？（EDA・仮説検証）
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- p値は「差がある確率」ではなく「帰無仮説のもとでこの差が偶然起きる確率」
- 効果量とサンプルサイズを合わせて解釈することが重要
- 多重検定には注意し、Bonferroni補正等を検討する

## 次回予告
- 統計的仮説検定②：相関・回帰分析の統計的側面（Udemy補足）
