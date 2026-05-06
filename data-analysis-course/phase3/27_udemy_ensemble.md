# [コマ27] 機械学習⑤：アンサンブル学習（Udemy補足）

## 本日の目標
- Bagging・Boosting・Stackingの違いを説明できる
- LightGBMを使った実践的なモデリングができる
- アンサンブルがなぜ有効かを直感的に説明できる

## 前回の振り返り
- SHAP値によるモデル解釈とドメイン知識を活かした特徴量生成を実践した

## 本編

### セクション1：アンサンブル学習の種類

```
Bagging（独立した複数モデルの平均）
例：Random Forest
→ 分散を減らす（過学習対策）

Boosting（前のモデルの誤りを次のモデルが修正）
例：XGBoost, LightGBM, CatBoost
→ バイアスを減らす（精度向上）

Stacking（複数モデルの出力をメタモデルで統合）
例：Level-0モデル + Level-1 LogReg
→ 多様なモデルを組み合わせて精度最大化
```

### セクション2：LightGBMの実践

```python
import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAGothic'
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42

# データ準備
df = sns.load_dataset('titanic').copy()
df = df.dropna(subset=['age', 'fare', 'embarked']).reset_index(drop=True)
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['sex_enc']     = (df['sex'] == 'female').astype(int)
df['embarked_enc']= df['embarked'].astype('category').cat.codes

feature_cols = ['pclass', 'age', 'fare', 'family_size', 'sex_enc', 'embarked_enc']
X = df[feature_cols]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': RANDOM_STATE,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

# X_train を reset_index しておくと .iloc で安定して扱える
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    model = lgb.LGBMClassifier(**params, n_estimators=1000)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / 5

print(f"OOF AUC: {roc_auc_score(y_train, oof_preds):.4f}")
```

### セクション3：なぜアンサンブルが有効か

```
直感的な説明：
「3人の専門家がそれぞれ独立して予測し、多数決を取る」
→ 1人が外れても他の2人が正解することが多い

数学的根拠：
各モデルの誤差が独立している場合、平均を取ることで
誤差の分散が 1/n に減少する
```

## ハンズオン / 演習

1. LightGBMを自分のデータに適用し、CVスコアを確認する
2. `learning_rate` と `num_leaves` を変えてスコアの変化を確認する
3. RandomForestとLightGBMの予測確率の平均を取り（シンプルアンサンブル）、単体モデルより精度が上がるか確認する

## Phase 2 との接続チェック

- [ ] 今日学んだ手法は自分の「問い」に使えるか？
- [ ] 使える場合：どのステップで使うか？
- [ ] 使えない場合：なぜか？

## 今日のまとめ
- Baggingは分散削減、Boostingはバイアス削減という役割がある
- LightGBMはearly_stoppingで過学習を自動制御できる
- シンプルな予測値の平均でもアンサンブル効果が得られる

## 次回予告
- Phase 3 総復習：Udemyで学んだ手法の整理と自分の問いとの対応確認
