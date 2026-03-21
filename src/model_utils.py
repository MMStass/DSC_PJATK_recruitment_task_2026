import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import itertools


class ModelWrapper:
    """Base class for model wrappers to standardize API."""
    def fit(self, X_train, y_train, X_val, y_val):
        pass
    def predict_proba(self, X):
        pass
    def get_feature_importances(self, feature_names):
        # Domyślna implementacja dla modeli, które tego nie wspierają
        return None


class LightGBMWrapper(ModelWrapper):
    def __init__(self, params):
        self.params = params
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(1000)
            ]
        )

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names):
        # Gain - specyficzny dla lgbm
        return pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.booster_.feature_importance(importance_type='gain')
        })

def run_universal_cv(X_train, y_train, X_test, model_wrapper, preprocessor_func = None, preprocessor_kwargs=None, n_splits = 5, random_state =42):
    """
    Universal CV Loop for Stacking generation.
    """
    print("Starting Universal CV Loop...")
    skf = StratifiedKFold(n_splits = n_splits, random_state= random_state, shuffle = True)

    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))

    if preprocessor_kwargs is None:
        preprocessor_kwargs = {}

    fold_importances = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Fold: {fold}")

        X_tr = X_train.iloc[train_idx].copy()
        y_tr = y_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx].copy()
        y_va = y_train.iloc[val_idx]
        X_te = X_test.copy()

        if preprocessor_func is not None:
            X_tr, X_va, X_te = preprocessor_func(X_tr, y_tr, X_va, X_te, **preprocessor_kwargs)

        model_wrapper.fit(X_tr, y_tr, X_va, y_va)

        oof_predictions[val_idx] = model_wrapper.predict_proba(X_va)
        test_predictions += model_wrapper.predict_proba(X_te) / skf.n_splits

        imp_df = model_wrapper.get_feature_importances(X_tr.columns)
        if imp_df is not None:
            fold_importances.append(imp_df)

        del X_tr, y_tr, X_va, y_va, X_te
        gc.collect()

    cv_auc = roc_auc_score(y_train, oof_predictions)
    print(f"\nCV AUC: {cv_auc:.5f}")

    mean_importances_df = None
    if fold_importances:
        mean_importances_df = pd.concat(fold_importances).groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()

    return oof_predictions, test_predictions, cv_auc, mean_importances_df