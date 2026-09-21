from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb

from models.train_regression import train_log_regression_model
from models.train_classification import (
    train_threshold_derived_classifier,
    train_classifier_model,
)
from features.feature_engineering import build_features_tables
from db.load_data import load_subset
from db.db_utils import get_connection, fetch_dataframe, write_clean_cycles_rows, get_best_regression_run

if __name__ == "__main__":
    train_file = Path("./data/train_FD001.txt")
    test_file = Path("./data/test_FD001.txt")
    rul_file = Path("./data/rul_FD001.txt")

    with get_connection() as conn:
        cursor = conn.cursor()
        loaded = load_subset("FD001", train_file, test_file, rul_file, cursor)
        print(f"loaded rows: {loaded}")

        raw_df = fetch_dataframe(cursor, "SELECT * FROM cycles;")

    features_df = build_features_tables(raw_df)

    with get_connection() as conn:
        cursor = conn.cursor()
        n_written = write_clean_cycles_rows(cursor, features_df, "v1")
        print(f"wrote {n_written} rows to clean_cycles")

    with get_connection() as conn:
        cursor = conn.cursor()
        EXISTING_REGRESSION_RUN_PK = get_best_regression_run(cursor)

    regression_groups = [
        ("linear_regression", LinearRegression()),
        ("random_forest", RandomForestRegressor()),
        ("xgboost", xgb.XGBRegressor()),
    ]

    classification_groups = [
        ("logreg_classifier", LogisticRegression()),
        ("rf_classifier", RandomForestClassifier()),
    ]

    regression_run_pk = {}
    for model_name, model_object in regression_groups:
        run_pk = train_log_regression_model(
            model_name, model_object, features_df, feature_set_verison="v1"
        )
        regression_run_pk[model_name] = run_pk
        print(f"{model_name}: logged as model_run_pk={run_pk}")

    dedicated_run_pks = {}
    for model_name, clf in classification_groups:
        run_pk = train_classifier_model(
            model_name, clf, features_df, feature_set_version="v1"
        )
        dedicated_run_pks[model_name] = run_pk
        print(f"{model_name}: logged as model_run_pk={run_pk}")

    if EXISTING_REGRESSION_RUN_PK is not None:
        td_run_pk = train_threshold_derived_classifier(
            EXISTING_REGRESSION_RUN_PK, feature_set_version="v1"
        )
        print(f"threshold_derived: logged as model_run_pk={td_run_pk}")
