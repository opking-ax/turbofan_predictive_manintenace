from psycopg2.extras import Json
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from db.db_utils import get_connection, log_metrics, write_predictions, save_model_artifact
from train_regression import FEATURE_COLS, loads_features_and_labels, average_cv_scores

def classification_scores(true_labels, predicted_labels):
    holdout_scores = {
        'precision': precision_score(true_labels, predicted_labels),
        'recall': recall_score(true_labels, predicted_labels),
        'f1': f1_score(true_labels, predicted_labels)
    }
    return holdout_scores


def fetch_predictions(cursor, regression_model_run_pk):
    cursor.execute("""SELECT cycle_pk, predicted_rul FROM predictions WHERE model_run_pk = %s""", (regression_model_run_pk))
    rows = cursor.fetchall()
    if not rows:
        return [], []
    cycle_pks, predicted_rul = zip(*rows)
    return list(cycle_pks), list(predicted_rul)

def fetch_true_rul(cursor, cycle_pks):
    if not cycle_pks:
        return []

    cursor.execute("""SELECT cycle_pk, rul_label FROM cycles WHERE cycle_pk = ANY(%s)""", list(cycle_pks))
    lookup = dict(cursor.fetchall())
    return [lookup[pk] for pk in cycle_pks]

def train_threshold_derived_classifier(regression_model_run_pk, feature_set_version, threshold=30):
    with get_connection() as conn:
        cursor = conn.cursor()

        cycle_pks, reg_predictions = fetch_predictions(cursor, regression_model_run_pk)
        ture_rul = fetch_true_rul(cursor, cycle_pks)

        cursor.execute("""INSERT INTO model_runs (model_name, task_type, feature_set_version, hyperparams)
            VALUES (%s, %s, %, %s) RETURNING model_run_pk;
            """, ("threshold_derived", "classification", feature_set_version, {'threshold': threshold, 'source_model_run': regression_model_run_pk}))

        model_run_pk = cursor.fetchone()[0]

        predicted_labels = [pred_rul <= threshold for pred_rul in reg_predictions]
        true_labels = [rul <= threshold for rul in ture_rul]

        write_predictions(cursor, model_run_pk, cycle_pks, predicted_label=predicted_labels)

        scores = classification_scores(true_labels, predicted_labels)
        log_metrics(cursor, model_run_pk, scores, eval_split='holdout_test')
        return model_run_pk


def train_classifier_model(model_name, classifier_object, feature_df, feature_set_version, label_col="rul_label", group_col="engine_pk", feature_cols=None, threshold=30, decision_threshold=0.5):
    feature_cols = feature_cols or FEATURE_COLS

    binary_label = (feature_df[label_col] <= threshold).astype(int)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""INSERT INTO model_runs (model_name, task_type, feature_set_version, hyperparams)
            VALUES (%s, %s, %s, %s) RETURNING model_run_pk;
            """, (model_name, "classification", feature_set_version, Json({**classifier_object.get_params(), "threshold": threshold})))

        model_run_pk = cursor.fetchone()[0]

        cv_scores = []
        kfold = GroupKFold(n_splits=5)

        for (train_idx, val_idx) in kfold.split(feature_df, groups=feature_df[group_col]):
            X_train = feature_df.iloc[train_idx][feature_cols]
            y_train = binary_label.iloc[train_idx]
            X_val = feature_df[val_idx][feature_cols]
            y_val = binary_label.iloc[val_idx]

            fitted_model = classifier_object.fit(X_train, y_train)
            val_probs = fitted_model.predict_proba(X_val)[:, 1]
            val_preds = val_probs >= decision_threshold

            cv_scores.append(classification_scores(y_val, val_preds))

        log_metrics(cursor, model_run_pk, average_cv_scores(cv_scores), eval_split='holdout_test')

        X_full, y_full = feature_df[feature_cols], binary_label
        final_model = classifier_object.fit(X_full, y_full)
        artifact_path = save_model_artifact(final_model, model_run_pk)
        cursor.execute(
            """UPDATE model_runs SET notes = %s WHERE model_run_pk = %s""",
            (f"artifact_path={artifact_path}", model_run_pk)
        )

        test_cycle_pks, X_test, y_test_rul = loads_features_and_labels(feature_cols)
        y_test = (np.asarray(y_test_rul) <= threshold).astype(int)

        test_probs = final_model.predict_proba(X_test)[:, 1]
        test_preds = test_probs >= decision_threshold

        write_predictions(cursor, model_run_pk, test_cycle_pks, test_preds, test_probs)

        holdout_scores = classification_scores(y_test, test_probs)
        log_metrics(cursor, model_run_pk, holdout_scores, eval_split="holdout_test")

    return model_run_pk
