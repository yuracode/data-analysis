# CLAUDE.md - データ分析入門 授業資料生成指示書

## プロジェクト概要

専門学校のICT学科向け「データ分析入門」授業資料を生成する。
全45コマ・90分/コマ。対象は scikit-learn・TensorFlow の経験がある学生。

---

## ディレクトリ構成

```
data-analysis-course/
├── CLAUDE.md
├── phase1/
│   ├── 01_guidance.md
│   ├── 02_titanic_eda.md
│   ├── 03_titanic_preprocessing.md
│   ├── 04_titanic_model.md
│   └── 05_textbook_ch1_ch2.md
├── phase2/
│   ├── 06_question_clarification.md
│   ├── 07_analysis_story.md
│   ├── 08_business_knowledge_1.md
│   ├── 09_business_knowledge_1_exercise.md
│   ├── 10_business_knowledge_2.md
│   ├── 11_business_knowledge_2_exercise.md
│   ├── 12_your_question_1.md
│   └── 13_your_question_2.md
├── phase3/
│   ├── 14_udemy_pandas.md
│   ├── 15_udemy_basic_stats.md
│   ├── 16_udemy_visualization.md
│   ├── 17_udemy_visualization_exercise.md
│   ├── 18_udemy_preprocessing_1.md
│   ├── 19_udemy_preprocessing_2.md
│   ├── 20_udemy_preprocessing_exercise.md
│   ├── 21_udemy_stats_1.md
│   ├── 22_udemy_stats_2.md
│   ├── 23_udemy_ml_classification.md
│   ├── 24_udemy_ml_regression.md
│   ├── 25_udemy_ml_evaluation.md
│   ├── 26_udemy_feature_engineering.md
│   ├── 27_udemy_ensemble.md
│   └── 28_udemy_review.md
├── phase4/
│   ├── 29_textbook_ch6_design.md
│   ├── 30_textbook_ch6_titanic.md
│   ├── 31_textbook_ch7_interpretation.md
│   ├── 32_textbook_ch8_stats.md
│   ├── 33_house_prices_eda.md
│   ├── 34_house_prices_preprocessing.md
│   ├── 35_house_prices_model.md
│   └── 36_house_prices_submit.md
├── phase5/
│   ├── 37_textbook_ch9_reporting.md
│   ├── 38_textbook_ch9_notebook.md
│   ├── 39_textbook_ch10_ai.md
│   ├── 40_textbook_ch10_business.md
│   ├── 41_final_theme.md
│   ├── 42_final_analysis.md
│   ├── 43_final_notebook.md
│   ├── 44_presentation_1.md
│   └── 45_presentation_2.md
└── assets/
    └── (図・サンプルコード等)
```

---

## 各ファイルの生成ルール

### ファイル形式

- **出力形式：Markdown（.md）固定**
- PDF指定がある場合のみPDFに変換する
- コードブロックは必ず言語指定する（` ```python ` など）

### 必須セクション構成（全コマ共通）

各 `.md` ファイルは以下の構成で生成すること：

```markdown
# [コマ番号] [タイトル]

## 本日の目標
- 箇条書きで2〜3項目

## 前回の振り返り（1コマ目を除く）
- 前回のポイントを1〜2行で

## 本編
### セクション1
### セクション2
### セクション3

## ハンズオン / 演習
- 具体的な作業手順
- コード例（必要な場合）

## 今日のまとめ
- 箇条書きで2〜3項目

## 次回予告
- 1行
```

---

## フェーズ別 生成指示

### Phase 1（コマ1〜5）：導入・Kaggle体験

**目的**
- Kaggleに触れて「なぜこの分析をするのか？」という疑問を育てる
- 教科書の全体ステップと接続する

**生成時の注意**
- コマ1はガイダンス資料なので既存資料を使用・生成不要
- コマ2〜4はKaggle Titanicの作業手順をステップごとに丁寧に記述する
- コマ5は教科書Ch1-2の要約＋Titanicでやったことの対応表を入れる
- **「？リスト」（疑問を記録するセクション）を各コマに入れる**

**サンプルコード方針**
- pandas, matplotlib, seaborn, scikit-learn を使用
- 過度な説明は不要（経験者向け）
- コードは動作する最小限のものを提示

---

### Phase 2（コマ6〜13）：問いを立てる

**目的**
- 教科書Ch3〜5の内容を講義形式でまとめる
- 「与えられた問いを解く」から「自分で問いを設計する」へ

**生成時の注意**
- ビジネス用語（KPI・ロジックツリー・マーケティングファネル等）は必ず定義を入れる
- コマ12〜13の演習は「学校の学生データを使った場合の問い設計」を例題にする
- ロジックツリー・分析ストーリーはテキストベースの図で表現する（Mermaid可）

**演習のフォーマット**

```markdown
## 演習

### お題
（具体的な設定を記載）

### やること
1.
2.
3.

### 提出物
- （何を提出するか）
```

---

### Phase 3（コマ14〜28）：Udemyで手法を学ぶ

**目的**
- Udemyの内容に対応した補足・まとめ資料を生成する
- 「手法の意味」を押さえる講義メモとして機能させる

**生成時の注意**
- Udemyの動画と並走する資料なのでコードの網羅的な説明は不要
- 「この手法はなぜ使うか」「ビジネスでどう使うか」の視点を必ず入れる
- 各コマ末尾に「Phase 2で立てた問いと接続できるか？」チェック欄を入れる

**チェック欄テンプレート**

```markdown
## Phase 2との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？
- [ ] 使えない場合：なぜか？
```

---

### Phase 4（コマ29〜36）：前処理・分析を深める

**目的**
- 教科書Ch6〜8の内容をUdemyの実践と接続する
- House PricesでTitanic以上の「問いの設計」を体験させる

**生成時の注意**
- コマ29〜30は「Udemyでやった前処理の意味を教科書で回収する」構成にする
- コマ33〜36はHouse Pricesの分析手順を段階的に記述する
- House Pricesでは「なぜ住宅価格を予測するのか」というビジネス文脈を必ず入れる

**Kaggle演習コマの構成**

```markdown
## 今日のKaggle作業

### 問いの確認
（今日の分析で答えようとしている問いを再確認）

### 作業手順
1.
2.
3.

### 詰まったときの対処
- Discussion で検索するキーワード例：

### 振り返り
- 今日の「?」リスト：
```

---

### Phase 5（コマ37〜45）：伝える・まとめる

**目的**
- 分析結果を「人を動かすアウトプット」に変える
- 最終課題でPhase 1〜4の学びを統合する

**生成時の注意**
- コマ37〜38はNotebookをレポートとして整える具体的な手順を入れる
- コマ39〜40はAI・MLを「使う側の視点」で解説する（実装詳細は不要）
- コマ41〜43は最終課題のガイドシートとして機能させる
- コマ44〜45は発表・評価シートを含める

**最終課題ガイドシートの必須項目**

```markdown
## 最終課題 要件

### テーマ
自由（ただし以下を満たすこと）
- 実データを使用すること
- 問いが明確であること
- ビジネス・社会的な意味があること

### 提出物
1. Kaggle Notebook（またはJupyter Notebook）
2. 発表スライド（形式自由）

### 評価軸
| 項目 | 観点 |
|------|------|
| 問いの質 | ビジネス視点で意味のある問いか |
| 分析の適切さ | 問いに対して手法が合っているか |
| 解釈・考察の深さ | 数字を意味として語れているか |
| 伝わるか | Notebookを見た人が理解できるか |
```

---

## コーディングスタイル規約

```python
# インポートは以下を基本とする
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# 日本語フォント設定（matplotlib）
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'  # Linux環境

# 乱数シード固定
RANDOM_STATE = 42
```

---

## 禁止事項

- 過度な初心者向け説明（対象は scikit-learn・TensorFlow 経験者）
- ツールや言語の環境構築手順の詳細説明（別途資料あり）
- 特定のUdemy講座名・URL・講師名の記載
- PDFの無断生成（明示的に指定された場合のみ）

---

## 生成コマンド例

```bash
# Phase 1 を全コマ一括生成
generate phase1

# 特定コマのみ生成
generate phase2/06_question_clarification.md

# 全フェーズ一括生成
generate all
```

---

## 備考

- 教科書：『データ分析のインプットとアウトプットが1冊で学べる』（渋谷智之著）
- Kaggle コンペ：Titanic（Phase 1・4）、House Prices（Phase 4）
- Udemy：別途受講（Phase 3 はその補足資料として機能）
- 対象学生：Python・scikit-learn・TensorFlow 経験あり
- 1コマ：90分

