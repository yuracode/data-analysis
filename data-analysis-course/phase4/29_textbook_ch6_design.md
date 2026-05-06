# [コマ29] 教科書Ch6：分析設計の精緻化

## 本日の目標
- Udemyで学んだ前処理・モデリングの操作を教科書の「なぜ」で回収できる
- 分析設計書（目的・手法・データ・評価基準）の品質を高められる
- 前処理の選択判断に根拠を持てる

## 前回の振り返り
- Phase 3 総復習。分析計画書を更新し、Phase 4 の準備チェックを実施した

## 本編

### セクション1：Udemyの前処理を教科書で「回収する」

Udemy では「どうやるか」を中心に学んだ。教科書 Ch6 ではその**「なぜ」**を整理する。

| Udemyでやったこと | 教科書が与える「なぜ」 |
|------------------|----------------------|
| 中央値で欠損補完 | 平均は外れ値に引っ張られるため中央値が頑健 |
| StandardScaler | 線形モデルは特徴量のスケール差に敏感 |
| One-Hot Encoding | ツリー系以外は「大小関係の誤った学習」を防ぐ |
| 交差検証 | テストデータへの汎化性能を不偏推定する |
| early_stopping | 過学習の防止と計算コストの削減 |

### セクション2：分析設計の3つの軸

教科書 Ch6 は分析設計を以下の3軸で評価する：

```
1. 問いの明確さ
   └── KPIが測定可能か、スコープが絞られているか

2. データの適切さ
   └── バイアスがないか、代表性があるか、リークがないか

3. 手法の妥当性
   └── 問いに対して手法が適切か
       モデルの仮定がデータに合っているか
```

### セクション3：データリークを防ぐ

分析設計で最も致命的なミスの一つ：

```python
# NG：testデータの情報がtrainに混入
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(pd.concat([X_train, X_test]))  # ← リーク！

# OK：trainだけでfit
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)         # ← transformのみ

# さらに良い：Pipelineで自動保証
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('model', ...)])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

リークが起きやすい操作：
- 全データでのスケーリング・補完
- Target Encoding を全データでfit
- 目的変数と相関する変数の漏れ込み

## ハンズオン / 演習

1. 自分の前処理コードを見直し、データリークが起きていないか確認する
2. 前処理の各選択に「なぜその手法を選んだか」のコメントを追加する
3. 分析設計を3軸（問いの明確さ・データの適切さ・手法の妥当性）で自己評価する

## 今日のまとめ
- 前処理の「正しさ」は操作の正確さではなく、問いとデータへの適切さで判断する
- データリークは「testデータの情報がtrainに混入する」すべてのケースが対象
- Pipeline は設計上の安全装置として積極的に使う

## 次回予告
- 教科書Ch6 続き：Titanic を教科書のフレームワークで再分析する
