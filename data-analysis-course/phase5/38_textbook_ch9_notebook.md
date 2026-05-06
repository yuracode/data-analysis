# [コマ38] Notebookをレポートとして整える

## 本日の目標
- Jupyter Notebookを「他者が読める」状態に整備できる
- Markdownセルを効果的に使ってストーリーを構成できる
- HTML・PDF出力ができる

## 前回の振り返り
- エグゼクティブサマリーの構造とストーリーテリングの型を学んだ

## 本編

### セクション1：Notebookの構成原則

「動くコード」から「読めるレポート」への変換：

```
分析Notebook（自分向け）    発表Notebook（他者向け）
─────────────────────────────────────────────
試行錯誤のコード            整理されたコード
コメントなし                目的・解釈のMarkdownセル
グラフはとりあえず表示      グラフはタイトル・軸ラベル付き
セルが順不同                上から順に実行可能
```

### セクション2：Notebookの整備手順

**Step 1：セルの整理**
```
□ 上から順に実行して全セルが通ることを確認（Kernel > Restart & Run All）
□ 試行錯誤の途中コードは削除 or 別ファイルに移動
□ 最終的に使わなかった変数・関数を削除
```

**Step 2：Markdownセルの追加**

各コードブロックの前に目的を書く：

```markdown
## 2. 目的変数の分布確認

SalePriceは右裾が長い分布（歪度=1.88）のため、対数変換（log1p）を適用します。
変換後の歪度は0.12となり、ほぼ正規分布に近づきます。
```

**Step 3：グラフの整備**

```python
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt

# タイトル・軸ラベル・単位・注記を必ず入れる
fig, ax = plt.subplots(figsize=(8, 4))
# ... グラフのコード ...
ax.set_title('住宅価格と総面積の関係', fontsize=13)
ax.set_xlabel('総面積 (平方フィート)')
ax.set_ylabel('販売価格 (ドル)')
ax.text(0.02, 0.95, 'n=1,460件', transform=ax.transAxes, va='top', color='gray')
plt.tight_layout()
```

### セクション3：HTML・PDF出力

```bash
# HTML出力（Notebookの実行結果込み）
jupyter nbconvert --to html my_analysis.ipynb

# PDF出力（LaTeX が必要）
jupyter nbconvert --to pdf my_analysis.ipynb

# テンプレートを使ったきれいなHTMLレポート
pip install nbconvert[webpdf]
jupyter nbconvert --to webpdf my_analysis.ipynb
```

**Kaggle Notebook の場合**：
- 右上の「Share」から公開URLを取得
- Output タブで生成したCSVファイルを確認

## ハンズオン / 演習

1. 自分の分析Notebookを「Restart & Run All」で通す
2. 各主要セクションの前にMarkdownセルで目的と解釈を書く（5箇所以上）
3. HTMLに出力して、ブラウザで「第三者として読んだとき」の問題点を3つ挙げる

## 今日のまとめ
- 良いNotebookは「上から読んで分析の意図と結論が伝わる」もの
- コードだけでなく「なぜこの分析をしたか」「結果から何が言えるか」を書く
- Restart & Run All が通らないNotebookは提出物として不完全

## 次回予告
- 教科書Ch10：AIとビジネスの接点（使う側の視点で理解する）
