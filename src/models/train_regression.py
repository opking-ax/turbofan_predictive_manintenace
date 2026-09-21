from db.db_utils import get_connection, log_metrics, write_predictions, save_model_artifact
from psycopg2.extras import Json
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from metrics import phm08_score
import numpy as np


FEATURE_COLS = []

def spilt_xy(df, idx, feature_cols, label_cols):
    subset = df.iloc[idx]
    return subset[feature_cols], subset[label_cols]

def loads_features_and_labels(feature_cols):
    pass


def average_cv_scores(cv_scores):
    keys = cv_scores[0].keys
    return {k: float(np.mean([fold[k] for fold in cv_scores])) for k in keys}

def train_log_regression_model(model_name, model_object, feature_df, feature_set_verison, label_col='rul_label', group_col='engine_pk', feature_cols=None):

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""INSERT INTO model_runs (model_name, task_type, feature_set_verison, hyperparams)
            VALUES (%s, %s, %s, %s) RETURNING model_run_pk;
            """, (model_name, "regression", feature_set_verison, Json(model_object.get_params())))

        model_run_pk = cursor.fetchone()[0]

        cv_scores = []
        kfold = GroupKFold(n_splits=5)

        for (train_idx, val_idx) in kfold.split(feature_df, groups=feature_df[group_col]):
            X_train, y_train = spilt_xy(feature_df, train_idx, feature_cols, label_col)
            X_val, y_val = spilt_xy(feature_df, val_idx, feature_cols, label_col)

            fitted_model = model_object.fit(X_train, y_train)
            preds = fitted_model(X_val)

            cv_scores.append({
                'rmse': root_mean_squared_error(y_val, preds),
                'mae': mean_absolute_error(y_val, preds),
                'r2': r2_score(y_val, preds),
                'phm08': phm08_score(y_val, preds)
            })

        log_metrics(cursor, model_run_pk, cv_scores, eval_split='cv')

        X_full, y_full = feature_df[feature_cols], feature_df[label_col]
        final_model = model_object.fit(X_full, y_full)
        artifact_path = save_model_artifact(final_model, model_run_pk)
        cursor.execute("""UPDATE model_run SET notes = %s WHERE model_run_pk = %s;""", (f"artifact_path={artifact_path}", model_run_pk))

        test_cycle_pks, X_test, y_test = loads_features_and_labels(feature_cols)
        test_preds = final_model.predict(X_test)

        write_predictions(cursor, model_run_pk, test_cycle_pks, test_preds)

        holdout_scores = {
            'rmse': root_mean_squared_error(y_test, test_preds),
            'mae': mean_absolute_error(y_test, test_preds),
            'r2': r2_score(y_test, test_preds),
            'phm08': phm08_score(y_test, test_preds)
        }

        log_metrics(cursor, model_run_pk, holdout_scores, eval_split='holdout_test')

    return model_run_pk

