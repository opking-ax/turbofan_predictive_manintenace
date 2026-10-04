from db.db_utils import get_connection, log_metrics, write_predictions, save_model_artifact
from psycopg2.extras import Json
from sklearn.model_selection import GroupKFold
from .metrics import phm08_score, compute_regression_metrics
import numpy as np
import pandas as pd
import argparse


def loads_features_and_labels(df: pd.DataFrame, label_col='rul_label'):
    NON_FEATURES_COLS = ['engine_pk', "cycle_pk", 'cycle_number', 'dataset_subset', 'split', 'is_last_cycle', label_col]

    feature_cols = [c for c in df.columns if c not in NON_FEATURES_COLS]

    X = df[feature_cols]
    y = df[label_col]
    groups = df['engine_pk']
    cycle_pks = df['cycle_pk']

    assert label_col not in X.columns, (
        f"Check that {label_col} is in the Columns list "
        f"Columns List: {X.columns}"
    )

    return X, y, groups, feature_cols, cycle_pks


def average_cv_scores(cv_scores):
    keys = cv_scores[0].keys
    return {k: float(np.mean([fold[k] for fold in cv_scores])) for k in keys}


def train_log_regression_model(model_name: str, model_object, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_set_version: argparse.Namespace):
    X_train, y_train, group_train, _, _ = loads_features_and_labels(train_df)
    X_test, y_test, _, _, cycle_test = loads_features_and_labels(test_df)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_runs (model_name, task_type, feature_set_version, hyperparams)
            VALUES (%s, %s, %s, %s) RETURNING model_run_pk;
            """, (model_name, "regression", feature_set_version, model_object.get_params()))

        model_run_pk = cursor.fetchone()[0]

        cv_scores = []
        kfold = GroupKFold(n_splits=5)

        for (train_idx, val_idx) in kfold.split(X_train, groups=group_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fitted_model = model_object.fit(X_fold_train, y_fold_train)
            preds = fitted_model(X_fold_val)

            cv_scores.append(compute_regression_metrics(y_fold_val, preds))

        log_metrics(cursor, model_run_pk, average_cv_scores(cv_scores), eval_split='cv')

        final_model = model_object.fit(X_train, y_train)
        save_model_artifact(final_model, model_run_pk)

        test_preds = final_model.predict(X_test)
        write_predictions(cycle_test, model_run_pk, test_preds)

        log_metrics(cursor, model_run_pk, compute_regression_metrics(y_test, test_preds), eval_split='holdout_test')

    return model_run_pk
