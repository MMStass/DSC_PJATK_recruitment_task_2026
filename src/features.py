import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import itertools

def load_data(path_to_raw_data: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(f'{path_to_raw_data}/train.csv', index_col='id')
    test = pd.read_csv(f'{path_to_raw_data}/test.csv', index_col='id')
    return train, test

def get_feature_pairs(columns: list) -> list[tuple[str, str]]:
    return list(itertools.combinations(columns, 2))

def make_categorical_interaction_features(data: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    categorical_interaction_features = {}
    for pair in pairs:
        name = f"{pair[0]}_{pair[1]}"
        categorical_interaction_features[name] = data[pair[0],pair[1]].astype(str).agg('_'.join, axis=1)
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

def determine_high_cardinality_features(data: pd.DataFrame, columns: list, threshold: int) -> list[str]:
    result = []
    for column in columns:
        if data[column].nunique() > threshold:
            result.append(column)
    return result

def make_count_features(data: pd.DataFrame, columns: list) -> pd.DataFrame:
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

def make_quantile_binned_features(data: pd.DataFrame, columns: list, n_bins: int) -> pd.DataFrame:
    quantile_binned_features = {}
    for column in columns:
        name = f"{column}_quantile_binned"
        quantile_binned_features[name] = pd.qcut(data[column], q=n_bins, duplicates='drop')
    return pd.DataFrame(quantile_binned_features)

def make_uniform_binned_features(data: pd.DataFrame, columns: list, n_bins: int) -> pd.DataFrame:
    quantile_binned_features = {}
    for column in columns:
        name = f"{column}_uniform_binned"
        quantile_binned_features[name] = pd.cut(data[column], bins=n_bins, duplicates='drop')
    return pd.DataFrame(quantile_binned_features)

def make_kmeans_binned_features(data: pd.DataFrame, columns: list, n_bins: int) -> pd.DataFrame:
    kmeans_binned_features = {}
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
    for column in columns:
        name = f"{column}_kmeans_binned"
        kmeans_binned_features[name] = discretizer.fit_transform(data[[column]]).ravel()
    return pd.DataFrame(kmeans_binned_features, index=data.index)

def mark_for_target_encoding(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    return data[columns].add_prefix('TE_')



