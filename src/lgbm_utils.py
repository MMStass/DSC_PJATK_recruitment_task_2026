import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder
import itertools


def build_all_features(train_df, test_df, orig_df):
    TARGET = 'loan_paid_back'
    print("Building unified feature space with deep digit extraction...")

    combined = pd.concat([
        train_df.drop(columns=[TARGET], errors='ignore'),
        test_df
    ])

    # A. Domain Knowledge
    combined['default_risk'] = (combined['debt_to_income_ratio'] * 0.40 +
                                (850 - combined['credit_score']) / 850 * 0.35 +
                                combined['interest_rate'] / 100 * 0.25)

    # B. Modulo
    for c in ['annual_income', 'loan_amount']:
        for k in range(-2, 2):
            n = f'{c}_d{k}'
            combined[n] = ((combined[c] * 10 ** k) % 10).fillna(-1).astype("int8")

    # C. Binning
    print("Applying binning...")
    combined['loan_amount_qcut'] = pd.qcut(combined['loan_amount'], q=10000, duplicates='drop').astype(str)
    combined['annual_income_qcut'] = pd.qcut(combined['annual_income'], q=10000, duplicates='drop').astype(str)
    combined['loan_amount_cut'] = pd.cut(combined['loan_amount'], bins=10000).astype(str)
    combined['annual_income_cut'] = pd.cut(combined['annual_income'], bins=10000).astype(str)

    # D. Deep Digit Extraction
    print("Extracting digits and creating combinations...")
    digits_df = pd.DataFrame(index=combined.index)
    float_cols = ['annual_income', 'debt_to_income_ratio', 'loan_amount', 'interest_rate']

    for col in float_cols:
        splitted = combined[col].astype(str).str.split(".", expand=True)
        non_decimal = splitted[0]
        decimal = splitted[1].fillna('0')

        max_len_non_dec = non_decimal.str.len().max()
        max_len_dec = decimal.str.len().max()

        non_dec_padded = non_decimal.str.rjust(max_len_non_dec, '0')
        dec_padded = decimal.str.ljust(max_len_dec, '0')

        for i in range(max_len_non_dec):
            digits_df[f"{col}_int_digit_{i}"] = non_dec_padded.str[i]
        for i in range(max_len_dec):
            digits_df[f"{col}_dec_digit_{i}"] = dec_padded.str[i]

    # Add digit combinations for interest_rate
    ir_cols = [c for c in digits_df.columns if "interest_rate" in c]
    for cols in itertools.combinations(ir_cols, 2):
        comb_name = f"{cols[0]}_{cols[1]}_comb"
        digits_df[comb_name] = digits_df[cols[0]].astype(str) + "_" + digits_df[cols[1]].astype(str)

    combined = pd.concat([combined, digits_df], axis=1)

    # E. Density Ratios
    print("Calculating count ratios...")
    high_card_cols = ['employment_status', 'loan_purpose', 'grade_subgrade']
    count_ratios = {}

    for col in high_card_cols:
        synth_counts = combined.groupby(col, dropna=False).transform('size')
        orig_counts_map = orig_df.groupby(col, dropna=False).size()
        orig_counts = combined[col].map(orig_counts_map).fillna(1)
        count_ratios[f'{col}_count_ratio'] = synth_counts / orig_counts

    ratios_df = pd.DataFrame(count_ratios, index=combined.index)
    combined = pd.concat([combined, ratios_df], axis=1)

    # Split back
    X_train_full = combined.iloc[:len(train_df)].copy()
    X_test_full = combined.iloc[len(train_df):].copy()

    # F. Map Original TE
    te_columns = high_card_cols + ['loan_amount_qcut', 'annual_income_qcut', 'loan_amount_cut',
                                   'annual_income_cut'] + list(digits_df.columns)

    print("Mapping original TE...")
    orig_te_mapped_train = pd.DataFrame(index=X_train_full.index)
    orig_te_mapped_test = pd.DataFrame(index=X_test_full.index)

    for col in high_card_cols:
        orig_mean = orig_df.groupby(col)[TARGET].mean()
        orig_te_mapped_train[f'TE_orig_{col}'] = X_train_full[col].map(orig_mean).astype('float32')
        orig_te_mapped_test[f'TE_orig_{col}'] = X_test_full[col].map(orig_mean).astype('float32')

    X_train_full = pd.concat([X_train_full, orig_te_mapped_train], axis=1)
    X_test_full = pd.concat([X_test_full, orig_te_mapped_test], axis=1)

    cols_to_drop = ['gender', 'marital_status', 'education_level']
    X_train_full = X_train_full.drop(columns=cols_to_drop)
    X_test_full = X_test_full.drop(columns=cols_to_drop)

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