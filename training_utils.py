import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_sequences(data, target_col='n_flows', sequence_length=20, additional_features=None):
    target_data = data[target_col].values
    if additional_features and len(additional_features) > 0:
        feature_data = data[additional_features + [target_col]].values
    else:
        feature_data = target_data.reshape(-1, 1)

    X, y = [], []
    for i in range(sequence_length, len(feature_data)):
        X.append(feature_data[i-sequence_length:i])
        y.append(target_data[i])

    return np.array(X), np.array(y)

def create_traditional_features(df, target_col='n_flows', max_lags=10, selected_features=None):
    features_df = df.copy()

    if selected_features:
        for feature in selected_features:
            if feature in df.columns:
                features_df[feature] = df[feature]

    for lag in range(1, max_lags + 1):
        features_df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    for window in [3, 7, 14]:
        features_df[f'{target_col}_avg_{window}'] = df[target_col].rolling(window=window).mean()
        features_df[f'{target_col}_volatility_{window}'] = df[target_col].rolling(window=window).std()

    features_df[f'{target_col}_change_1'] = df[target_col].diff()
    features_df[f'{target_col}_growth_rate'] = df[target_col].pct_change()

    features_df = features_df.dropna()

    feature_cols = [col for col in features_df.columns if col != target_col]
    X = features_df[feature_cols]
    y = features_df[target_col]

    return X, y

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')

    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1:
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        direction_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
    else:
        direction_accuracy = 0

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'direction_accuracy': direction_accuracy
    }, y_true, y_pred