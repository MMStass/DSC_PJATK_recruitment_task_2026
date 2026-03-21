import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder
from src import features as fe
import itertools


def build_all_features(train_df, test_df, orig_df):
    TARGET = 'loan_paid_back'
    print("Building unified feature space with deep digit extraction...")

    combined = pd.concat([
        train_df.drop(columns=[TARGET], errors='ignore'),
        test_df
    ])

    # 1.Domain Knowledge
    combined['default_risk'] = (combined['debt_to_income_ratio'] * 0.40 +
                                (850 - combined['credit_score']) / 850 * 0.35 +
                                combined['interest_rate'] / 100 * 0.25)

    # 2.Binning
    qcut_cols = ['loan_amount', 'annual_income']
    qcut_df = fe.make_quantile_binned_features(combined, qcut_cols, n_bins=10000)
    cut_df = fe.make_uniform_binned_features(combined, qcut_cols, n_bins=10000)

    combined = pd.concat([combined, qcut_df, cut_df], axis=1)

    # 3.Count Features
    high_card_cols = ['employment_status', 'loan_purpose', 'grade_subgrade']
    count_df = fe.make_count_features(combined, high_card_cols)
    combined = pd.concat([combined, count_df], axis=1)

    # 4.Aggregate Features
    agg_df_1 = fe.make_aggregate_features(combined, 'grade_subgrade', 'loan_amount', ['mean', 'std'])
    agg_df_2 = fe.make_aggregate_features(combined, 'employment_status', 'annual_income', ['mean', 'median'])
    combined = pd.concat([combined, agg_df_1, agg_df_2], axis=1)

    # 5.Deep Digit Extraction & Combinations
    print("Extracting digits...")
    float_cols = ['annual_income', 'debt_to_income_ratio', 'loan_amount', 'interest_rate']
    digits_df = fe.make_deep_digits_features(combined, float_cols)

    # Combinations for interest rate digits
    ir_cols = [c for c in digits_df.columns if "interest_rate" in c]
    ir_pairs = fe.get_feature_pairs(ir_cols)
    digits_comb_df = fe.make_categorical_interaction_features(digits_df, ir_pairs)

    combined = pd.concat([combined, digits_df, digits_comb_df], axis=1)

    # 6. Modular Density Ratios
    print("Calculating count ratios...")
    ratios_df = fe.make_density_ratio_features(combined, orig_df, high_card_cols)
    combined = pd.concat([combined, ratios_df], axis=1)

    # --- Split back to Train/Test ---
    X_train_full = combined.iloc[:len(train_df)].copy()
    X_test_full = combined.iloc[len(train_df):].copy()

    # F. Map Original TE
    te_columns = (high_card_cols +
                  list(qcut_df.columns) +
                  list(cut_df.columns) +
                  list(digits_df.columns) +
                  list(digits_comb_df.columns))

    print("Mapping original TE...")
    orig_te_mapped_train = pd.DataFrame(index=X_train_full.index)
    orig_te_mapped_test = pd.DataFrame(index=X_test_full.index)

    for col in high_card_cols:
        orig_mean = orig_df.groupby(col)[TARGET].mean()
        orig_te_mapped_train[f'TE_orig_{col}'] = X_train_full[col].map(orig_mean).astype('float32')
        orig_te_mapped_test[f'TE_orig_{col}'] = X_test_full[col].map(orig_mean).astype('float32')

    X_train_full = pd.concat([X_train_full, orig_te_mapped_train], axis=1)
    X_test_full = pd.concat([X_test_full, orig_te_mapped_test], axis=1)

    # Clean up standard text columns
    cols_to_drop = ['gender', 'marital_status', 'education_level']
    X_train_full = X_train_full.drop(columns=cols_to_drop)
    X_test_full = X_test_full.drop(columns=cols_to_drop)

    # Convert to native LGBM category types
    for c in te_columns:
        X_train_full[c] = X_train_full[c].astype('category')
        X_test_full[c] = X_test_full[c].astype('category')

    return X_train_full, X_test_full, train_df[TARGET], te_columns


def lgbm_te_preprocessor(X_tr, y_tr, X_va, X_te, columns_to_te):
    """Specific inside-CV preprocessing for LightGBM with required parameter."""
    encoder = TargetEncoder(cv=5, random_state=42, smooth=10.0)

    X_tr_te = encoder.fit_transform(X_tr[columns_to_te], y_tr)
    X_va_te = encoder.transform(X_va[columns_to_te])
    X_te_te = encoder.transform(X_te[columns_to_te])

    X_tr['TE_row_mean'] = np.mean(X_tr_te, axis=1)
    X_tr['TE_row_std'] = np.std(X_tr_te, axis=1)
    X_tr['TE_row_max'] = np.max(X_tr_te, axis=1)
    X_tr['TE_row_min'] = np.min(X_tr_te, axis=1)

    X_va['TE_row_mean'] = np.mean(X_va_te, axis=1)
    X_va['TE_row_std'] = np.std(X_va_te, axis=1)
    X_va['TE_row_max'] = np.max(X_va_te, axis=1)
    X_va['TE_row_min'] = np.min(X_va_te, axis=1)

    X_te['TE_row_mean'] = np.mean(X_te_te, axis=1)
    X_te['TE_row_std'] = np.std(X_te_te, axis=1)
    X_te['TE_row_max'] = np.max(X_te_te, axis=1)
    X_te['TE_row_min'] = np.min(X_te_te, axis=1)

    # In-place substitution
    X_tr = X_tr.drop(columns=columns_to_te).assign(**{col: X_tr_te[:, i] for i, col in enumerate(columns_to_te)})
    X_va = X_va.drop(columns=columns_to_te).assign(**{col: X_va_te[:, i] for i, col in enumerate(columns_to_te)})
    X_te = X_te.drop(columns=columns_to_te).assign(**{col: X_te_te[:, i] for i, col in enumerate(columns_to_te)})

    return X_tr, X_va, X_te