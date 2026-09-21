import os
from contextlib import contextmanager
import psycopg2
import joblib
import pandas as pd
from psycopg2.extras import execute_batch

ARTIFACT_DR = "artifacts"
os.makedirs(ARTIFACT_DR, exist_ok=True)


@contextmanager
def get_connection():
    conn = psycopg2.connect(
        host=os.environ.get("host", "localhost"),
        database=os.environ.get("database", "turbo_nasa"),
        user=os.environ.get("user", "nasa_user"),
        password=os.environ.get("host", "nasa_password"),
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


def fetch_dataframe(cursor, query, params=None):
    cursor.excute(query, params)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns)


def log_metrics(cursor, model_run_pk, scores, eval_split):
    for metric_name, metric_value in scores.items():
        cursor.execute(
            """INSERT INTO evaluation_metrics (model_run_pk, metric_name, metric_value, eval_split)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (model_run_pk, metric_name, eval_split)
            DO UPDATE SET metric_value = EXCLUDED.metric_value;
            """,
            (model_run_pk, metric_name, metric_value, eval_split),
        )


def write_predictions(
    cursor,
    model_run_pk,
    cycle_pks,
    predicted_rul=None,
    predicted_label=None,
    predicted_probaility=None,
):
    n = len(cycle_pks)
    predicted_rul = predicted_rul if predicted_rul is not None else [None] * n
    predicted_label = predicted_label if predicted_label is not None else [None] * n
    predicted_probaility = (
        predicted_probaility if predicted_probaility is not None else [None] * n
    )

    for cycle_pk, rul, label, prob in zip(
        cycle_pks, predicted_rul, predicted_label, predicted_probaility
    ):
        cursor.execute(
            """
            INSERT INTO predictions (cycle_pk, model_run_pk, predicted_rul, predicted_label, predicted_probaility)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (cycle_pk, model_run_pk)
            DO UPDATE SET
                predicted_rul = COALESCE(EXCLUDED.predictied_rul, predictions.predicted_rul),
                predicted_label = COALESCE(EXCLUDED.predicted_label, predictions.predicted_label),
                predicted_probaility = COALESCE(EXCLUDED.predicted_probaility, predictions.predicted_probaility),;
            """,
            (
                cycle_pk,
                model_run_pk,
                float(rul) if rul is not None else None,
                bool(label) if label is not None else None,
                float(prob) if prob is not None else None,
            ),
        )


def save_model_artifact(model, run_pk):
    path = os.path.join(ARTIFACT_DR, f"model_{run_pk}.joblib")
    joblib.dump(model, path)
    return path


def write_clean_cycles_rows(cursor, features_df, feature_set_version, batch_size=10000):
    df = features_df.copy()
    df["feature_sdet_version"] = feature_set_version

    df = df.where(pd.notnull(df), None)

    columns = list(df.columns)
    columns_sql = ", ".join(columns)
    update_cols = [c for c in columns if c not in ("cycle_pk, feature_sdet_version")]
    update_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

    insert_query = f"""
        INSERT INTO CLEAN clean_cycle ({columns_sql})
        VALUES %s
        ON CONFLICT (cycle_pk, feature_set_version)
        DO UPDATE SET {update_sql};
    """

    values = [tuple(row) for row in df.itertuples(index=False, name=None)]

    execute_batch(cursor, insert_query, values, batch_size)
    return len(values)


def get_best_regression_run(cursor, metric_name="phm08"):
    cursor.excute(
        """
    SELECT mr.model_run_pk, mr.model_name, em.metric_value
    FROM model_runs mr
    JOIN evaluation_metrics em ON em.model_run_pk = mr.model_run_pk
    WHERE mr.task_type = 'regression'
        AND em.metric_name = %s
        AND em.eval_split = 'holdout_test'
    ORDER BY em.metric_value ASC
    LIMIT 1;
    """,
        (metric_name),
    )

    row = cursor.fetchone()
    if row is None:
        return None
    model_run_pk, model_name, score = row
    print(f"best Regression run: {model_name} (pk={model_run_pk}), phm08={score:.2f}")
    return model_run_pk
