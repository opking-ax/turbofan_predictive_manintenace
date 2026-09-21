CREATE TABLE engines (
    engine_pk   SERIAL PRIMARY KEY,
    dataset_subset  TEXT NOT NULL CHECK (dataset_subset IN ('FD001','FD002','FD003','FD004')),
    split   TEXT NOT NULL CHECK (split IN ('train', 'test')),
    unit_number INT NOT NULL,
    true_val_last_cycle INT,
    UNIQUE (dataset_subset, split, unit_number)
);

CREATE TABLE cycles (
    cycle_pk    SERIAL PRIMARY KEY,
    engine_pk   INT NOT NULL REFERENCES engines(engine_pk) ON DELETE CASCADE,
    cycle_number    INT NOT NULL,
    op_setting_1    DOUBLE PRECISION,
    op_setting_2    DOUBLE PRECISION,
    op_setting_3    DOUBLE PRECISION,
    sensor_1        DOUBLE PRECISION, sensor_2      DOUBLE PRECISION, sensor_3      DOUBLE PRECISION,
    sensor_4        DOUBLE PRECISION, sensor_5      DOUBLE PRECISION, sensor_6      DOUBLE PRECISION,
    sensor_7        DOUBLE PRECISION, sensor_8      DOUBLE PRECISION, sensor_9      DOUBLE PRECISION,
    sensor_10       DOUBLE PRECISION, sensor_11     DOUBLE PRECISION, sensor_12     DOUBLE PRECISION,
    sensor_13       DOUBLE PRECISION, sensor_14     DOUBLE PRECISION, sensor_15     DOUBLE PRECISION,
    sensor_16       DOUBLE PRECISION, sensor_17     DOUBLE PRECISION, sensor_18     DOUBLE PRECISION,
    sensor_19       DOUBLE PRECISION, sensor_20     DOUBLE PRECISION, sensor_21     DOUBLE PRECISION,
    rul_label   INT,
    is_last_cycle   BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(engine_pk, cycle_number)
);

CREATE INDEX idx_cycles_engine_cycle ON cycles (engine_pk, cycle_number);
CREATE INDEX idx_cycles_last_cycle ON cycles (engine_pk) WHERE is_last_cycle = TRUE;

CREATE TABLE model_runs(
    model_run_pk    SERIAL PRIMARY KEY,
    model_name  TEXT NOT NULL,
    task_type   TEXT NOT NULL CHECK (task_type IN ('regression','classification')),
    feature_set_version TEXT NOT NULL,
    hyperparams JSONB,
    trained_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes TEXT
);

CREATE TABLE predictions(
    prediction_pk    SERIAL PRIMARY KEY,
    cycle_pk    INT NOT NULL REFERENCES cycles(cycle_pk) ON DELETE CASCADE,
    model_run_pk INT NOT NULL REFERENCES model_runs(model_run_pk) ON DELETE CASCADE,
    predicted_rul  DOUBLE PRECISION,
    predicted_label   BOOLEAN,
    predicted_probability DOUBLE PRECISION,
    UNIQUE(cycle_pk, model_run_pk)
);

CREATE INDEX idx_predictions_model_run ON predictions (model_run_pk);

CREATE TABLE evaluation_metrics(
    metrics_pk    SERIAL PRIMARY KEY,
    model_run_pk INT NOT NULL REFERENCES model_runs(model_run_pk) ON DELETE CASCADE,
    metric_name  TEXT NOT NULL,
    metric_value   DOUBLE PRECISION NOT NULL,
    eval_split TEXT NOT NULL CHECK (eval_split IN ('cv', 'holdout_test')),
    UNIQUE(model_run_pk, metric_name, eval_split)
);

CREATE TABLE clean_cycle (
    clean_cycle_pk SERIAL PRIMARY KEY,
    cycle_pk INT NOT NULL REFERENCES cycles(cycle_pk) ON DELETE CASCADE,
    engine_pk INT NOT NULL REFERENCES engines(engine_pk) ON DELETE CASCADE,
    feature_set_version TEXT NOT NULL,
    cycle_number INT NOT NULL,
    rul_label INT,
    is_last_cycle BOOLEAN NOT NULL DEFAULT FALSE,

    sensor_2 DOUBlE PRECISION,
    sensor_2_rollmean_5 DOUBlE PRECISION,
    sensor_2_rollstd_5 DOUBlE PRECISION,
    sensor_2_rollmean_10 DOUBlE PRECISION,
    sensor_2_rollstd_10 DOUBlE PRECISION,
    sensor_2_diff_1 DOUBlE PRECISION,

    sensor_3 DOUBlE PRECISION,
    sensor_3_rollmean_5 DOUBlE PRECISION,
    sensor_3_rollstd_5 DOUBlE PRECISION,
    sensor_3_rollmean_10 DOUBlE PRECISION,
    sensor_3_rollstd_10 DOUBlE PRECISION,
    sensor_3_diff_1 DOUBlE PRECISION,

    sensor_4 DOUBlE PRECISION,
    sensor_4_rollmean_5 DOUBlE PRECISION,
    sensor_4_rollstd_5 DOUBlE PRECISION,
    sensor_4_rollmean_10 DOUBlE PRECISION,
    sensor_4_rollstd_10 DOUBlE PRECISION,
    sensor_4_diff_1 DOUBlE PRECISION,

    sensor_7 DOUBlE PRECISION,
    sensor_7_rollmean_5 DOUBlE PRECISION,
    sensor_7_rollstd_5 DOUBlE PRECISION,
    sensor_7_rollmean_10 DOUBlE PRECISION,
    sensor_7_rollstd_10 DOUBlE PRECISION,
    sensor_7_diff_1 DOUBlE PRECISION,

    sensor_8 DOUBlE PRECISION,
    sensor_8_rollmean_5 DOUBlE PRECISION,
    sensor_8_rollstd_5 DOUBlE PRECISION,
    sensor_8_rollmean_10 DOUBlE PRECISION,
    sensor_8_rollstd_10 DOUBlE PRECISION,
    sensor_8_diff_1 DOUBlE PRECISION,

    sensor_9 DOUBlE PRECISION,
    sensor_9_rollmean_5 DOUBlE PRECISION,
    sensor_9_rollstd_5 DOUBlE PRECISION,
    sensor_9_rollmean_10 DOUBlE PRECISION,
    sensor_9_rollstd_10 DOUBlE PRECISION,
    sensor_9_diff_1 DOUBlE PRECISION,

    sensor_11 DOUBlE PRECISION,
    sensor_11_rollmean_5 DOUBlE PRECISION,
    sensor_11_rollstd_5 DOUBlE PRECISION,
    sensor_11_rollmean_10 DOUBlE PRECISION,
    sensor_11_rollstd_10 DOUBlE PRECISION,
    sensor_11_diff_1 DOUBlE PRECISION,

    sensor_12 DOUBlE PRECISION,
    sensor_12_rollmean_5 DOUBlE PRECISION,
    sensor_12_rollstd_5 DOUBlE PRECISION,
    sensor_12_rollmean_10 DOUBlE PRECISION,
    sensor_12_rollstd_10 DOUBlE PRECISION,
    sensor_12_diff_1 DOUBlE PRECISION,

    sensor_13 DOUBlE PRECISION,
    sensor_13_rollmean_5 DOUBlE PRECISION,
    sensor_13_rollstd_5 DOUBlE PRECISION,
    sensor_13_rollmean_10 DOUBlE PRECISION,
    sensor_13_rollstd_10 DOUBlE PRECISION,
    sensor_13_diff_1 DOUBlE PRECISION,

    sensor_14 DOUBlE PRECISION,
    sensor_14_rollmean_5 DOUBlE PRECISION,
    sensor_14_rollstd_5 DOUBlE PRECISION,
    sensor_14_rollmean_10 DOUBlE PRECISION,
    sensor_14_rollstd_10 DOUBlE PRECISION,
    sensor_14_diff_1 DOUBlE PRECISION,

    sensor_15 DOUBlE PRECISION,
    sensor_15_rollmean_5 DOUBlE PRECISION,
    sensor_15_rollstd_5 DOUBlE PRECISION,
    sensor_15_rollmean_10 DOUBlE PRECISION,
    sensor_15_rollstd_10 DOUBlE PRECISION,
    sensor_15_diff_1 DOUBlE PRECISION,

    sensor_17 DOUBlE PRECISION,
    sensor_17_rollmean_5 DOUBlE PRECISION,
    sensor_17_rollstd_5 DOUBlE PRECISION,
    sensor_17_rollmean_10 DOUBlE PRECISION,
    sensor_17_rollstd_10 DOUBlE PRECISION,
    sensor_17_diff_1 DOUBlE PRECISION,

    sensor_20 DOUBlE PRECISION,
    sensor_20_rollmean_5 DOUBlE PRECISION,
    sensor_20_rollstd_5 DOUBlE PRECISION,
    sensor_20_rollmean_10 DOUBlE PRECISION,
    sensor_20_rollstd_10 DOUBlE PRECISION,
    sensor_20_diff_1 DOUBlE PRECISION,

    sensor_21 DOUBlE PRECISION,
    sensor_21_rollmean_5 DOUBlE PRECISION,
    sensor_21_rollstd_5 DOUBlE PRECISION,
    sensor_21_rollmean_10 DOUBlE PRECISION,
    sensor_21_rollstd_10 DOUBlE PRECISION,
    sensor_21_diff_1 DOUBlE PRECISION,

    UNIQUE(cycle_pk, feature_set_version)
);

CREATE INDEX idx_clean_cycle_engine ON clean_cycle (engine_pk. feature_set_version);
CREATE INDEX idx_clean_cycle_last_cycle ON clean_cycle (engine_pk) WHERE is_last_cycle = TRUE;

CREATE VIEW fleet_status AS
SELECT
    e.engine_pk,
    e.dataset_subset,
    e.unit_number,
    c.cycle_number,
    p.predicted_rul,
    p.predicted_label,
    p.predicted_probaility,
    mr.model_name,
    CASE
        WHEN p.predicted_rul IS NULL THEN 'unknown'
        WHEN p.predicted_rul <= 15 THEN 'critical'
        WHEN p.predicted_rul <= 30 THEN 'elevated'
        ELSE 'nominal'
    END AS risk_tier
FROM engines e
JOIN cycles c ON c.engine_pk = e.engine_pk AND c.is_last_cycle = TRUE
JOIN predictions p ON p.cycle_pk = c.cycle_pk
JOIN model_runs mr ON mr.model_run_pk = p.model_run_pk;
