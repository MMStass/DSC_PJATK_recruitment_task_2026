from typing import Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
import itertools
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def load_data(path_to_data_folder: str, file_type='csv') -> tuple[pd.DataFrame, pd.DataFrame]:
    if file_type == 'csv':
        train = pd.read_csv(f'{path_to_data_folder}/FE_train.csv', index_col='id')
        test = pd.read_csv(f'{path_to_data_folder}/FE_test.csv', index_col='id')
        return train, test
    elif file_type == 'parquet':
        train = pd.read_parquet(f'{path_to_data_folder}/FE_train.parquet')
        test = pd.read_parquet(f'{path_to_data_folder}/FE_test.parquet')
        return train, test
    else:
        raise Exception("Invalid file type. Must be csv or parquet.")

def get_feature_pairs(columns: list) -> list[tuple[str, str]]:
    return list(itertools.combinations(columns, 2))

def make_categorical_interaction_features(data: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    categorical_interaction_features = {}
    for pair in pairs:
        name = f"{pair[0]}_{pair[1]}"
        categorical_interaction_features[name] = data[[pair[0],pair[1]]].astype(str).agg('_'.join, axis=1)
    return pd.DataFrame(categorical_interaction_features)

def make_numerical_interaction_features(data: pd.DataFrame, pairs: list[tuple[str, str]], operator: str) -> pd.DataFrame:
    numerical_interaction_features = {}
    if operator not in ['+','-','*','/']:
        raise Exception("Invalid operator. Must be '+' or '-' or '*' or '/'")
    for pair in pairs:
        name = f"{pair[0]}{operator}{pair[1]}"
        match operator:
            case '+':
                numerical_interaction_features[name] = data[pair[0]] + data[pair[1]]
            case '-':
                numerical_interaction_features[name] = data[pair[0]] - data[pair[1]]
            case '*':
                numerical_interaction_features[name] = data[pair[0]] * data[pair[1]]
            case '/':
                numerical_interaction_features[name] = ((data[pair[0]] / data[pair[1]])
                                                        .replace([np.inf, -np.inf], np.nan)) #potencjalnie sus np.nan

    return pd.DataFrame(numerical_interaction_features)

def determine_high_cardinality_features(data: pd.DataFrame, columns: list[str], threshold: int) -> list[str]:
    result = []
    for column in columns:
        if data[column].nunique() > threshold:
            result.append(column)
    return result

def make_count_features(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    count_features = {}
    for column in columns:
        name = f"CE_{column}"
        count_features[name] = data.groupby(column, dropna=False).transform('size')
    return pd.DataFrame(count_features)

def make_aggregate_features(
        data: pd.DataFrame,
        categorical_column: str,
        numerical_column: str,
        functions: list[str]
        ) -> pd.DataFrame:
    aggregate_features = {}
    for function in functions:
        name = f"{categorical_column}_{numerical_column}_{function}"
        aggregate_features[name] = data.groupby(categorical_column, dropna=False)[numerical_column].transform(function)
    return pd.DataFrame(aggregate_features)

def make_quantile_binned_features(data: pd.DataFrame, columns: list[str], n_bins: int) -> pd.DataFrame:
    quantile_binned_features = {}
    for column in columns:
        name = f"{column}_quantile_binned"
        quantile_binned_features[name] = pd.qcut(data[column], q=n_bins, duplicates='drop')
    return pd.DataFrame(quantile_binned_features)

def make_uniform_binned_features(data: pd.DataFrame, columns: list[str], n_bins: int) -> pd.DataFrame:
    quantile_binned_features = {}
    for column in columns:
        name = f"{column}_uniform_binned"
        quantile_binned_features[name] = pd.cut(data[column], bins=n_bins, duplicates='drop')
    return pd.DataFrame(quantile_binned_features)

def make_kmeans_binned_features(data: pd.DataFrame, columns: list[str], n_bins: int) -> pd.DataFrame:
    kmeans_binned_features = {}
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
    for column in columns:
        name = f"{column}_kmeans_binned"
        kmeans_binned_features[name] = discretizer.fit_transform(data[[column]]).ravel()
    return pd.DataFrame(kmeans_binned_features, index=data.index)


def make_log_binned_features(data: pd.DataFrame, columns: list[str], n_bins: int) -> pd.DataFrame:
    log_binned_features = {}
    for column in columns:
        name = f"{column}_log_binned"
        log_binned_features[name] = pd.cut(np.log1p(data[column]), bins = n_bins, duplicates = 'drop')
    return pd.DataFrame(log_binned_features)

def mark_for_target_encoding(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = data.copy()
    mapping = {col: f"TE_{col}" for col in columns}
    df = df.rename(columns=mapping)
    return df

# method for noise reduction by cutting decimal values and further binning
def make_rounded_halved_features(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rounded_features = {}
    for column in columns:
        name = f"{column}_round_half"
        rounded_features[name] = (data[column].astype(float).round() // 2).astype(str)
    return pd.DataFrame(rounded_features)

# generate features consisting of single digits to investigate relationships
# between them and the target value
def make_deep_digits_features(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    digits_features = {}

    for col in columns:
        split = data[col].astype(str).str.split('.', expand = True)
        non_decimal = split[0]
        decimal = split[1].fillna('0')

        max_len_non_dec = non_decimal.str.len().max()
        max_len_dec = decimal.str.len().max()

        non_dec_padded = non_decimal.str.rjust(max_len_non_dec, '0')
        dec_padded = decimal.str.ljust(max_len_dec, '0')

        for i in range(max_len_non_dec):
            digits_features[f"{col}_int_digit_{i}"] = non_dec_padded.str[i].astype(int)
        for i in range(max_len_dec):
            digits_features[f"{col}_dec_digit_{i}"] = dec_padded.str[i].astype(int)

    return pd.DataFrame(digits_features, index=data.index)


def make_density_ratio_features(data: pd.DataFrame, original_data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    density_ratio_features = {}

    for col in columns:
        synth_counts = data.groupby(col, dropna = False).transform('size')
        orig_counts_map = original_data.groupby(col, dropna = False).size()
        orig_counts = data[col].map(orig_counts_map).fillna(1)

        name = f"{col}_count_ratio"
        density_ratio_features[name] = synth_counts / orig_counts

    return pd.DataFrame(density_ratio_features, index=data.index)

def mark_as_pseudo_targets(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = data.copy()
    mapping = {col: f"PT_{col}" for col in columns}
    df = df.rename(columns=mapping)
    return df

def make_custom_binned_feature(data: pd.DataFrame, column: str, bins: list[Union[int,float]]) -> pd.DataFrame:
    name = f"{column}_custom_binned"
    df = pd.DataFrame()
    df[name] = pd.cut(data[column], bins=bins, duplicates='drop')
    return df

def make_financial_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out = pd.DataFrame(index = df.index)

    # 1. Default risk
    df_out['default_risk'] = (df['debt_to_income_ratio'] * 0.40 +
                              (850 - df['credit_score']) / 850 * 0.35 +
                              df['interest_rate'] / 100 * 0.25)

    # 2. Expected Loss (EL)
    df_out['expected_loss'] = df['loan_amount'] * df_out['default_risk']

    # 3. Expected Return & Risk-Adjusted Return
    df_out['expected_return'] = df['loan_amount'] * (df['interest_rate'] / 100)
    df_out['risk_adjusted_return'] = df_out['expected_return'] - df_out['expected_loss']

    return df_out

def make_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out = pd.DataFrame(index = df.index)

    # From 5C framework
    df_out['character_proxy'] = df['marital_status'].astype(str) + "_" + df['education_level'].astype(str)

    return df_out

def borrow_features_with_knn(train_df, test_df, orig_df, shared_cols, borrow_cols):
    scaler = StandardScaler()

    orig_scaled = scaler.fit_transform(orig_df[shared_cols].fillna(0))
    train_scaled = scaler.transform(train_df[shared_cols].fillna(0))
    test_scaled = scaler.transform(test_df[shared_cols].fillna(0))

    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(orig_scaled)

    train_distances, train_indices = nn.kneighbors(train_scaled)
    test_distances, test_indices = nn.kneighbors(test_scaled)

    train_out = train_df.copy()
    test_out = test_df.copy()

    for col in borrow_cols:
        if col in orig_df.columns:
            train_out[f'knn_orig_{col}'] = orig_df[col].iloc[train_indices.flatten()].values
            test_out[f'knn_orig_{col}'] = orig_df[col].iloc[test_indices.flatten()].values
        else:
            print(f"Warning: Column {col} not found in orig_df")

    train_out['knn_orig_distance'] = train_distances.flatten()
    test_out['knn_orig_distance'] = test_distances.flatten()

    return train_out, test_out

def make_custom_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    df_out = pd.DataFrame(index = df.index)
    scorecard_points = pd.Series(0, index = df.index, dtype = float)

    # 1. Debt to Income Ratio
    if 'debt_to_income_ratio' in df.columns:
        scorecard_points += np.where(df['debt_to_income_ratio'] < 20, 35,
                            np.where(df['debt_to_income_ratio'] <= 30, 25,
                            np.where(df['debt_to_income_ratio'] <= 40, 15,
                            np.where(df['debt_to_income_ratio'] <= 50, 5, 0))))

    # 2. Marital Status
    if 'marital_status' in df.columns:
        marital_mapping = {
            'Single': 10,
            'Divorced': 15,
            'Widowed': 15,
            'Married': 25
        }
        mapped_marital = df['marital_status'].map(marital_mapping).fillna(0)
        scorecard_points += mapped_marital

    if 'knn_orig_age' in df.columns:
        scorecard_points += np.where(df['knn_orig_age'] < 25, 5,
                            np.where(df['knn_orig_age'] <= 35, 15,
                            np.where(df['knn_orig_age'] <= 50, 25, 35)))

    df_out['custom_scorecard_points'] = scorecard_points
    return df_out