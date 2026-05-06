# [コマ31] 教科書Ch7：モデルの解釈と説明可能なAI

## 本日の目標
- ブラックボックスモデルを説明する手法（SHAP・LIME・Partial Dependence）を使い分けられる
- 「局所的な解釈」と「大域的な解釈」の違いを説明できる
- モデルの予測結果をビジネス関係者に伝えられる言葉に変換できる

## 前回の振り返り
- Titanicを教科書フレームワークで再分析し、Pipeline化とStratified KFoldで精度を改善した

## 本編

### セクション1：解釈可能性の2つのレベル

```
大域的（Global）解釈：モデル全体の傾向を理解する
└── 特徴量重要度、Partial Dependence Plot, SHAP summary

局所的（Local）解釈：1つの予測を説明する
└── SHAP waterfall, LIME
```

ビジネスでは局所的解釈が重要な場面が多い：
- 「なぜこの顧客に離脱リスクが出たのか」
- 「この審査で融資が否決された理由は何か」

### セクション2：Partial Dependence Plot（PDP）

```python
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import GradientBoostingClassifier

RANDOM_STATE = 42

model = GradientBoostingClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=['Age', 'Fare', 'FamilySize'],
    ax=axes,
)
plt.suptitle('Partial Dependence Plot')
plt.tight_layout()
plt.show()
```

PDPの読み方：「他の変数を平均的な値に固定したとき、この変数がどう影響するか」

### セクション3：SHAP による局所的解釈

```python
import shap

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# 個別予測の説明
idx = 0  # 説明したいサンプルのインデックス
shap.plots.waterfall(shap_values[idx])

# 全体の特徴量重要度
shap.plots.beeswarm(shap_values)

# 2変数の交互作用
shap.plots.scatter(shap_values[:, 'Age'], color=shap_values[:, 'Pclass'])
```

## ハンズオン / 演習

1. 自分のモデルにPDPを適用し、主要変数の影響を確認する
2. SHAP値を使って「最もリスクが高いサンプル」と「最も安全なサンプル」の予測根拠を比較する
3. 非技術者向けに「なぜこの予測になったか」を1〜2文で説明する文章を書く

## 今日のまとめ
- PDPは「ある変数の全体的な効果」、SHAP waterfallは「1サンプルの理由」を示す
- XAI（説明可能なAI）はモデルの「信頼性の確認」と「ビジネス説明」の両方に使える
- 解釈の目的に合わせてグローバル・ローカルを使い分ける

## 次回予告
- 教科書Ch8：統計的な意思決定支援（回帰分析の深掘りと推論）
