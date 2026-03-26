import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder
from src import features as fe
import itertools


def build_all_lgbm_features(train_df, test_df, orig_df):
    TARGET = 'loan_paid_back'
    print("Building LightGBM feature space with Pseudo-TE...")

    combined = pd.concat([
        train_df.drop(columns=[TARGET], errors='ignore'),
        test_df
    ])

    # 1.Domain Knowledge
    combined['default_risk'] = (combined['debt_to_income_ratio'] * 0.40 +
                                (850 - combined['credit_score']) / 850 * 0.35 +
                                combined['interest_rate'] / 100 * 0.25)

    # 2. Triple Binning & Round Hack
    qcut_cols = ['loan_amount', 'annual_income']
    qcut_df = fe.make_quantile_binned_features(combined, qcut_cols, n_bins=10000)
    cut_df = fe.make_uniform_binned_features(combined, qcut_cols, n_bins=10000)
    log_cut_df = fe.make_log_binned_features(combined, qcut_cols, n_bins=10000)
    round_half_df = fe.make_rounded_halved_features(combined, qcut_cols)

    combined = pd.concat([combined, qcut_df, cut_df, log_cut_df, round_half_df], axis=1)

    orig_round_half_df = fe.make_rounded_halved_features(orig_df, qcut_cols)
    orig_df = pd.concat([orig_df, orig_round_half_df], axis=1)

    # 3. Count Features
    high_card_cols = ['employment_status', 'loan_purpose', 'grade_subgrade']
    count_df = fe.make_count_features(combined, high_card_cols)
    combined = pd.concat([combined, count_df], axis=1)

    # 4. Deep Digits & Combinations
    print("Extracting digits...")
    float_cols = ['annual_income', 'debt_to_income_ratio', 'loan_amount', 'interest_rate']
    digits_df = fe.make_deep_digits_features(combined, float_cols)

    ir_cols = [c for c in digits_df.columns if "interest_rate" in c]
    ir_pairs = fe.get_feature_pairs(ir_cols)
    digits_comb_df = fe.make_categorical_interaction_features(digits_df, ir_pairs)

    combined = pd.concat([combined, digits_df, digits_comb_df], axis=1)

    base_cat_cols = high_card_cols + list(round_half_df.columns)

    # 5. Split back to Train/Test
    X_train_full = combined.iloc[:len(train_df)].copy()
    X_test_full = combined.iloc[len(train_df):].copy()

    # 6. Map Original TE & Pseudo-TE
    print("Mapping Original TE & Pseudo-TE...")
    orig_te_mapped_train = pd.DataFrame(index=X_train_full.index)
    orig_te_mapped_test = pd.DataFrame(index=X_test_full.index)

    pseudo_targets = ['debt_to_income_ratio', 'interest_rate', 'annual_income', 'loan_amount']

    for col in base_cat_cols:
        orig_mean = orig_df.groupby(col)[TARGET].mean()
        orig_te_mapped_train[f'TE_orig_{col}'] = X_train_full[col].map(orig_mean).astype('float32')
        orig_te_mapped_test[f'TE_orig_{col}'] = X_test_full[col].map(orig_mean).astype('float32')

        for p_target in pseudo_targets:
            pseudo_mean = orig_df.groupby(col)[p_target].mean()
            col_name = f'PseudoTE_{p_target}_by_{col}'
            orig_te_mapped_train[col_name] = X_train_full[col].map(pseudo_mean).astype('float32')
            orig_te_mapped_test[col_name] = X_test_full[col].map(pseudo_mean).astype('float32')

    X_train_full = pd.concat([X_train_full, orig_te_mapped_train], axis=1)
    X_test_full = pd.concat([X_test_full, orig_te_mapped_test], axis=1)

    columns_to_te = (high_card_cols +
                     list(qcut_df.columns) +
                     list(cut_df.columns) +
                     list(log_cut_df.columns) +
                     list(round_half_df.columns) +
                     list(digits_comb_df.columns))

    cols_to_drop = ['gender', 'marital_status', 'education_level']
    X_train_full = X_train_full.drop(columns=cols_to_drop, errors='ignore')
    X_test_full = X_test_full.drop(columns=cols_to_drop, errors='ignore')

    # Rzutowanie na category przed pętlą (wymagane dla lgbm_te_preprocessor)
    for c in columns_to_te:
        if c in X_train_full.columns:
            X_train_full[c] = X_train_full[c].astype(str).astype('category')
            X_test_full[c] = X_test_full[c].astype(str).astype('category')

    return X_train_full, X_test_full, train_df[TARGET], columns_to_te


def lgbm_te_preprocessor(X_tr, y_tr, X_va, X_te, columns_to_te):
    """Specific inside-CV preprocessing for LightGBM with required parameter."""
    if columns_to_te is None or len(columns_to_te) == 0:
        return X_tr, X_va, X_te

    encoder = TargetEncoder(cv=5, shuffle=True, random_state=42)

    # 1. Fit and transform
    X_tr_te = encoder.fit_transform(X_tr[columns_to_te], y_tr)
    X_va_te = encoder.transform(X_va[columns_to_te])
    X_te_te = encoder.transform(X_te[columns_to_te])

    # 2. Create optimized DataFrames from numpy arrays
    df_tr_te = pd.DataFrame(X_tr_te, columns=columns_to_te, index=X_tr.index)
    df_va_te = pd.DataFrame(X_va_te, columns=columns_to_te, index=X_va.index)
    df_te_te = pd.DataFrame(X_te_te, columns=columns_to_te, index=X_te.index)

    # 3. Drop original category columns to avoid duplication
    X_tr_clean = X_tr.drop(columns=columns_to_te)
    X_va_clean = X_va.drop(columns=columns_to_te)
    X_te_clean = X_te.drop(columns=columns_to_te)

    # 4. Concat efficiently
    X_tr_final = pd.concat([X_tr_clean, df_tr_te], axis=1)
    X_va_final = pd.concat([X_va_clean, df_va_te], axis=1)
    X_te_final = pd.concat([X_te_clean, df_te_te], axis=1)

    # 5. Row-level aggregation
    agg_list = ['mean', 'std', 'min', 'max']
    for agg in agg_list:
        X_tr_final[f'TE_row_{agg}'] = X_tr_final[columns_to_te].agg(agg, axis=1)
        X_va_final[f'TE_row_{agg}'] = X_va_final[columns_to_te].agg(agg, axis=1)
        X_te_final[f'TE_row_{agg}'] = X_te_final[columns_to_te].agg(agg, axis=1)

    return X_tr_final, X_va_final, X_te_final