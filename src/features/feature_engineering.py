from config import DROPPED_SENSORS, ALL_SENSORS_COLUMNS

def active_sensor_columns(all_sensor_columns, dropped_list=DROPPED_SENSORS):
    return [column for column in all_sensor_columns if column not in dropped_list]


def add_rolling_features(df, sensor_columns, windows_size=[5, 10, 20]):
    df = df.sort_values(by=["engine_pk", "cycle_number"])

    for sensor in sensor_columns:
        for window in windows_size:
            df[f"{sensor}_rollmean_{window}"] = df.groupby("engine_pk")[sensor].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"{sensor}_rollstd_{window}"] = df.groupby("engine_pk")[sensor].rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)

    return df

def add_rate_of_change_features(df, sensor_columns, lag=1):
    df = df.sort_values(by=['engine_pk', "cycle_number"])

    for sensor in sensor_columns:
        df[f"{sensor}_diff_{lag}"] = df.groupby("engine_pk")[sensor].diff(lag).fillna(0)

    return df


def build_features_tables(raw_cycles_df):
    sensors_cols = active_sensor_columns(ALL_SENSORS_COLUMNS, DROPPED_SENSORS)
    base_cols = ["engine_pk", "cycle_number", "rul_label", "is_last_cycle"]

    df = raw_cycles_df[base_cols + sensors_cols]
    df = add_rolling_features(df, sensors_cols, windows_size=[5, 10, 20])
    df = add_rate_of_change_features(df, sensors_cols, lag=1)

    return df
