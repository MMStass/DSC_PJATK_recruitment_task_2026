import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score


def train_stratified_xgb_with_te(
        data: pd.DataFrame,
        target_col: str,
        features: list,
        te_features: list,
        xgb_params: dict,
        n_splits: int = 5
):
    X = data[features]
    y = data[target_col]

    skkf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(data))
    fold_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skkf.split(X, y)):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx].copy()

        encoder = TargetEncoder(smooth="auto")
        X_train[te_features] = encoder.fit_transform(X_train[te_features], y_train)
        X_val[te_features] = encoder.transform(X_val[te_features])

        model = xgb.XGBClassifier(
            **xgb_params,
            early_stopping_rounds=50,
            eval_metric="auc"
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds_trial = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds_trial

        fold_auc = roc_auc_score(y_val, preds_trial)
        fold_scores.append(fold_auc)
        models.append(model)

        print(f"Fold {fold + 1} | AUC: {fold_auc:.4f}")

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"---|\nOOF AUC: {overall_auc:.4f}|---")

    return models, fold_scores, oof_preds