import pandas as pd
import numpy as np
import src.features as fe
import itertools


def build_all_cb_features(train_df, test_df, orig_df):
    TARGET = 'loan_paid_back'
    print("Building CatBoost feature space via modular features.py...")

    combined = pd.concat([
        train_df.drop(columns=[TARGET], errors='ignore'),
        test_df
    ])

    # 1. Domain Knowledge
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

    # 3. Modular Count Features
    high_card_cols = ['employment_status', 'loan_purpose', 'grade_subgrade']
    count_df = fe.make_count_features(combined, high_card_cols)
    combined = pd.concat([combined, count_df], axis=1)

    # 4. Modular Aggregate Features
    agg_df_1 = fe.make_aggregate_features(combined, 'grade_subgrade', 'loan_amount', ['mean', 'std'])
    agg_df_2 = fe.make_aggregate_features(combined, 'employment_status', 'annual_income', ['mean', 'median'])
    combined = pd.concat([combined, agg_df_1, agg_df_2], axis=1)

    # 5. Modular Deep Digit Extraction & Combinations
    print("Extracting digits...")
    float_cols = ['annual_income', 'debt_to_income_ratio', 'loan_amount', 'interest_rate']
    digits_df = fe.make_deep_digits_features(combined, float_cols)

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

    # 7. Map Original Target Encoding
    cat_columns = (high_card_cols +
                   list(qcut_df.columns) +
                   list(cut_df.columns) +
                   list(log_cut_df.columns) +
                   list(round_half_df.columns) +
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

    # Explicit conversion to string then category for CatBoost
    for c in cat_columns:
        if c in X_train_full.columns:
            X_train_full[c] = X_train_full[c].astype(str).astype('category')
            X_test_full[c] = X_test_full[c].astype(str).astype('category')

    return X_train_full, X_test_full, train_df[TARGET], cat_columns