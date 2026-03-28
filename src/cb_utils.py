import features as fe
import numpy as np
import pandas as pd


def build_all_cb_features(train_df, test_df, orig_df):
    TARGET = 'loan_paid_back'
    print("Building CatBoost feature space with KNN and Domain Knowledge...")

    # 1. KNN Feature Borrowing (Data Leakage Exploitation)
    print("Borrowing original features via KNN...")
    shared_cols = ['loan_amount', 'interest_rate', 'debt_to_income_ratio', 'credit_score', 'annual_income']
    borrow_cols = ['age', 'current_balance', 'installment', 'total_credit_limit', 'loan_term', 'num_of_delinquencies']

    train_df, test_df = fe.borrow_features_with_knn(
        train_df, test_df, orig_df, shared_cols, borrow_cols
    )

    combined = pd.concat([
        train_df.drop(columns=[TARGET], errors='ignore'),
        test_df
    ])

    # 2. Financial Risk & Demographic Features (Domain Knowledge)
    fin_risk_df = fe.make_financial_risk_features(combined)
    demographic_df = fe.make_demographic_features(combined)
    combined = pd.concat([combined, fin_risk_df, demographic_df], axis=1)

    # 3. Triple Binning & Round Hack
    qcut_cols = ['loan_amount', 'annual_income']
    qcut_df = fe.make_quantile_binned_features(combined, qcut_cols, n_bins=10000)
    cut_df = fe.make_uniform_binned_features(combined, qcut_cols, n_bins=10000)
    log_cut_df = fe.make_log_binned_features(combined, qcut_cols, n_bins=10000)
    round_half_df = fe.make_rounded_halved_features(combined, qcut_cols)

    combined = pd.concat([combined, qcut_df, cut_df, log_cut_df, round_half_df], axis=1)

    orig_round_half_df = fe.make_rounded_halved_features(orig_df, qcut_cols)
    orig_df = pd.concat([orig_df, orig_round_half_df], axis=1)

    high_card_cols = ['employment_status', 'loan_purpose', 'grade_subgrade']
    count_df = fe.make_count_features(combined, high_card_cols)
    combined = pd.concat([combined, count_df], axis=1)

    # 5. Deep Digits
    print("Extracting digits...")
    float_cols = ['annual_income', 'debt_to_income_ratio', 'loan_amount', 'interest_rate']
    digits_df = fe.make_deep_digits_features(combined, float_cols)
    combined = pd.concat([combined, digits_df], axis=1)

    # 6. Custom Scorecard
    scorecard_df = fe.make_custom_scorecard(combined)
    combined = pd.concat([combined, scorecard_df], axis=1)

    base_cat_cols = high_card_cols + list(round_half_df.columns)

    # 7. Split back to Train/Test
    X_train_full = combined.iloc[:len(train_df)].copy()
    X_test_full = combined.iloc[len(train_df):].copy()

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

    cat_columns = (high_card_cols +
                   list(qcut_df.columns) +
                   list(cut_df.columns) +
                   list(log_cut_df.columns) +
                   list(round_half_df.columns) +
                   list(digits_df.columns) +
                   ['character_proxy', 'gender', 'marital_status', 'education_level'])

    for c in cat_columns:
        if c in X_train_full.columns:
            X_train_full[c] = X_train_full[c].astype(str)
            X_test_full[c] = X_test_full[c].astype(str)

    return X_train_full, X_test_full, train_df[TARGET], cat_columns