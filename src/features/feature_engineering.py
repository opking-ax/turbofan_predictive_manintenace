from config import DROPPED_SENSORS, ALL_SENSORS_COLUMNS
from sklearn.cluster import KMeans
import pandas as pd

def active_sensor_columns(all_sensor_columns, dropped_list=DROPPED_SENSORS):
    return [column for column in all_sensor_columns if column not in dropped_list]


def add_rolling_features(df, sensor_columns, windows_size=[5, 10, 20]):
    df = df.sort_values(by=["engine_pk", "cycle_number"])

    for sensor in sensor_columns:
        for window in windows_size:
            df[f"{sensor}_rollmean_{window}"] = df.groupby("engine_pk")[sensor].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"{sensor}_rollstd_{window}"] = df.groupby("engine_pk")[sensor].rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    return df

def add_rate_of_change_features(df, sensor_columns, lag=1):
    df = df.sort_values(by=['engine_pk', "cycle_number"])

    for sensor in sensor_columns:
        df[f"{sensor}_diff_{lag}"] = df.groupby("engine_pk")[sensor].diff(lag).fillna(0)

    return df


def identify_operating_regimes(df, op_settings_cols, n_regimes=6, kmeans=None):
    if kmeans is None:
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10).fit(df[op_settings_cols])
    df = df.copy()
    df['regime_id'] = kmeans.predict(df[op_settings_cols])
    return df, kmeans


def fit_regime_normalizer(train_df, sensor_columns):
    stats = {}
    for regime in train_df['regime_id'].unique():
        for sensor in sensor_columns:
            stats[(regime, sensor)] = {
                'mean': train_df.loc[train_df["regime_id"] == regime, sensor].mean(),
                'std': (train_df.loc[train_df['regime_id'] == regime, sensor].std())
            }

    return stats


def apply_regime_normalizer(df: pd.DataFrame, fitted_stats, sensor_columns):
    for regime in df['regime_id'].unique():
        regime_mask = df['regime_id'] == regime

        for sensor in sensor_columns:
            regime_mean = fitted_stats[(regime, sensor)]['mean']
            regime_std = fitted_stats[(regime, sensor)]['std']

            df.loc[regime_mask, sensor + '_normalized'] = (df.loc[regime_mask, sensor] - regime_mean) / regime_std

    return df


def build_features_tables(raw_cycles_df: pd.DataFrame, dataset_subset: str, fitted_artifacts=None, is_train=None):
    sensors_cols = active_sensor_columns(ALL_SENSORS_COLUMNS, DROPPED_SENSORS)
    base_cols = ["engine_pk", "cycle_number", "rul_label", "is_last_cycle"]
    op_settings_cols = ["op_settings_1", "op_settings_2", "op_settings_3"]
    fitted_artifacts = None

    df = raw_cycles_df[base_cols + sensors_cols + op_settings_cols].copy()

    if dataset_subset in ('FD002', 'FD004'):
        if is_train:
            df, kmeans = identify_operating_regimes(df, op_settings_cols, n_regimes=6)
            stats = fit_regime_normalizer(df, sensors_cols)
            fitted_artifacts = {"kmeans": kmeans, "stats": stats}
        else:
            if fitted_artifacts is None:
                raise ValueError("fitted_artifacts (from train) must be passed in when processing test data.")
            df, _ = identify_operating_regimes(df, op_settings_cols, kmeans=fitted_artifacts["kmeans"])

        df = apply_regime_normalizer(df, fitted_artifacts["stats"], sensors_cols)
        working_sensor_cols = [s + "_normalized" for s in sensors_cols]
    else:
        working_sensor_cols = sensors_cols

    df = add_rolling_features(df, working_sensor_cols, windows_size=[5, 10])
    df = add_rate_of_change_features(df, working_sensor_cols)

    return df, fitted_artifacts
