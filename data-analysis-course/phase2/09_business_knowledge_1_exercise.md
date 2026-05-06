# [コマ9] ビジネス知識①演習：学生データで顧客分析を模倣する

## 本日の目標
- RFM分析の考え方を学生データに応用できる
- コホート的な分析（入学年度別の追跡）を実装できる
- 分析結果を「意思決定に使えるメッセージ」に変換できる

## 前回の振り返り
- マーケティングファネル・RFM・コホート分析の概念と用途を学んだ

## 本編

### セクション1：学生データの設計

演習用の学生データを生成する：

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

n = 300
df = pd.DataFrame({
    'student_id': range(1, n + 1),
    'entry_year': np.random.choice([2021, 2022, 2023], n),
    'gpa': np.round(np.random.normal(2.8, 0.6, n).clip(0, 4), 1),
    'attendance_rate': np.round(np.random.beta(8, 2, n), 2),
    'cert_count': np.random.poisson(1.5, n),
    'got_job': np.random.binomial(1, 0.72, n),
})
```

### セクション2：RFM的な3軸の設計

学生データへの読み替え：

| RFM軸 | 学生版 | カラム |
|-------|--------|--------|
| Recency | 直近のアクティビティ（出席率） | attendance_rate |
| Frequency | 学習への関与度（GPA） | gpa |
| Monetary | 成果（資格取得数） | cert_count |

```python
# スコアリング
df['R_score'] = pd.qcut(df['attendance_rate'], q=3, labels=[1,2,3])
df['F_score'] = pd.qcut(df['gpa'], q=3, labels=[1,2,3], duplicates='drop')
df['M_score'] = pd.cut(df['cert_count'], bins=[-1,0,2,100], labels=[1,2,3])

df['RFM_total'] = df['R_score'].astype(int) + df['F_score'].astype(int) + df['M_score'].astype(int)
```

### セクション3：コホート分析（入学年度別追跡）

```python
cohort = df.groupby('entry_year').agg(
    n_students = ('student_id', 'count'),
    avg_gpa    = ('gpa', 'mean'),
    avg_cert   = ('cert_count', 'mean'),
    job_rate   = ('got_job', 'mean'),
).reset_index()

print(cohort)
```

## 演習

### お題
上記の学生データを使い、「どんな学生が就職できていないか」を分析してください。

### やること
1. RFMスコアと `got_job` の関係をヒートマップや集計表で確認する
2. 就職できなかった学生の属性（GPA・出席率・資格数）を就職できた学生と比較する
3. 「就職支援が最も必要な学生グループ」をデータで定義する

```python
# ヒント：ピボットテーブル
pd.pivot_table(df, values='got_job', index='R_score', columns='M_score', aggfunc='mean')
```

### 提出物
- 分析結果をまとめた短いレポート（Markdownセル or テキスト）
- 「支援が必要な学生グループ」の定義と根拠

## ？リスト（疑問を記録しよう）
- サンプルデータで得た結論を実際のデータに使えるか？
- スコアの区切り方（分位数 vs 固定値）で結論が変わる？

## 今日のまとめ
- ビジネス分析の手法（RFM・コホート）は他のドメインにも転用できる
- 分析結果は「このグループに何をするか」まで落とさないと価値にならない
- データの設計（どう集計するか）が分析の質を決める

## 次回予告
- ビジネス知識②：需要予測・在庫管理・価格戦略とデータ分析（教科書Ch5後半）
