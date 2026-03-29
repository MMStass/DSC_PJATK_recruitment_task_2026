import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from src.utils.model_utils import ModelWrapper

def run_cross_validation_loop(
        X: pd.DataFrame,
        test: pd.DataFrame,
        y: pd.Series,
        model: ModelWrapper,
        n_splits,
        target_encoding: bool,
        pseudo_target_encoding: bool
):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y))
    test_predictions = np.zeros(len(test))
    fold_importances = []



    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        print(f"Fold: {fold}")


        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()

        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx].copy()

        X_test = test.copy()

        if pseudo_target_encoding:

            pseudo_target_columns = [col for col in X.columns if col.startswith('PT_')]
            columns_to_target_encode = [col for col in X.columns if col.startswith('TE_') or '_TE_' in col]

            for pseudo_target in pseudo_target_columns:

                pt_train_target = X_train[pseudo_target]

                pseudo_target_encoder = TargetEncoder(cv=n_splits, shuffle=True, random_state=42, smooth=10).set_output(
                    transform='pandas')

                encoded_train = pseudo_target_encoder.fit_transform(X_train[columns_to_target_encode], pt_train_target)
                encoded_val = pseudo_target_encoder.transform(X_val[columns_to_target_encode])
                encoded_test = pseudo_target_encoder.transform(X_test[columns_to_target_encode])

                encoded_train.columns = [f"{pseudo_target}_{col}" for col in encoded_train.columns]
                encoded_val.columns = [f"{pseudo_target}_{col}" for col in encoded_val.columns]
                encoded_test.columns = [f"{pseudo_target}_{col}" for col in encoded_test.columns]

                X_train = pd.concat([X_train, encoded_train], axis=1)
                X_val = pd.concat([X_val, encoded_val], axis=1)
                X_test = pd.concat([X_test, encoded_test], axis=1)

        if target_encoding:

            columns_to_target_encode = [col for col in X.columns if col.startswith('TE_') or '_TE_' in col]

            if columns_to_target_encode:
                target_encoder = TargetEncoder(cv=n_splits, shuffle=True, random_state=42, smooth=10).set_output(transform="pandas")

                X_train[columns_to_target_encode] = target_encoder.fit_transform(X_train[columns_to_target_encode], y_train)
                X_val[columns_to_target_encode] = target_encoder.transform(X_val[columns_to_target_encode])
                X_test[columns_to_target_encode] = target_encoder.transform(X_test[columns_to_target_encode])



        model.fit(X_train, y_train, X_val, y_val)

        oof_predictions[val_idx] = model.predict_proba(X_val)
        test_predictions += model.predict_proba(X_test) / skf.n_splits

        imp_df = model.get_feature_importances(X_train.columns)
        if imp_df is not None:
            fold_importances.append(imp_df)

        del X_train, y_train, X_val, y_val, X_test
        gc.collect()

    cv_auc = roc_auc_score(y, oof_predictions)

    print(f"\nCV AUC: {cv_auc:.5f}")

    mean_importances_df = None
    if fold_importances:
        mean_importances_df = pd.concat(fold_importances).groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()

    return oof_predictions, test_predictions, cv_auc, mean_importances_df

