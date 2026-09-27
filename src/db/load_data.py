import pandas as pd
from pathlib import Path

from db.db_utils import get_connection


def load_subset(dataset_subset: str, train_file: Path, test_file: Path, rul_file: Path):
    """Loads one C-MAPSS subset into `engine`/`cycles`"""
    train_raw = read_whitesapce_delimited(train_file)
    test_raw = read_whitesapce_delimited(test_file)

    rul_truth = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["true_remaining_life"])

    inserted_row = []

    with get_connection() as conn:
        try:
            cursor = conn.cursor()
            _load_split(cursor, dataset_subset, "train", train_raw, inserted_row, rul_by_unit=None)
            assert_positional_order(dataset_subset, test_raw, rul_truth)

        except Exception as e:
            print(e)

    return pd.DataFrame(inserted_row)


def _create_engine_pk(cursor, dataset_subset, split, unit_number, true_rul_cycle=None):
    cursor.execute("""INSERT INTO engines (dataset_subset, split, unit_number, true_val_last_cycle)
        VALUES (%s, %s, %s, %s) ON CONFLICT (dataset_subset, split, unit_number) DO NOTHING
        RETURNING engine_pk;
        """, (dataset_subset, split, int(unit_number), true_rul_cycle),)

    row = cursor.fetchone()
    if row is not None:
        return row[0]

    cursor.execute(
        """SELECT engine_pk FROM engines
           WHERE dataset_subset = %s AND split = %s AND unit_number = %s;""",
           (dataset_subset, split, int(unit_number)),
        )
    return cursor.fetchone()[0]


def _load_split(cursor, dataset_subset, split, raw_df, inserted_rows, rul_by_unit):
    op_settings_col = ", ".join(f"op_setting_{i}" for i in range(1, 4))
    sensors_col = ", ".join(f"sensor_{i}" for i in range(1, 22))

    for unit_number in raw_df["unit_number"].unique():
        true_rul = rul_by_unit.get(unit_number) if rul_by_unit else None
        engine_pk = _create_engine_pk(cursor, dataset_subset, split, unit_number, true_rul)

        engine_cycle = raw_df[raw_df["unit_number"] == unit_number].sort_values("cycle_number")
        max_cycle = engine_cycle["cycle_number"].max()

        for _, row in engine_cycle.iterrows():
            is_last = bool(row["cycle_number"] == max_cycle)

            if split == "train":
                raw_rul = max_cycle - row["cycle_number"]
                rul_label = float(min(raw_rul, 125))
            else:
                rul_label = true_rul if is_last else None

            op_settings = [float(row[f"op_settings_{i}"]) for i in range(1, 4)]
            sensors = [float(row[f"sensor_{i}"]) for i in range(1, 22)]

            values = [engine_pk, int(row["cycle_number"]), op_settings[0],op_settings[1],op_settings[2]] + sensors + [rul_label,is_last]

            cursor.execute(f"""INSERT INTO cycles (engine_pk,cycle_number,{op_settings_col},{sensors_col},rul_label,is_last_cycle)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (engine_pk, cycle_number) DO NOTHING;""",
                (values)
            )

            record = {
                "engine_pk": engine_pk,
                "cycle_number": int(row["cycle_number"]),
                "rul_label": rul_label,
                "is_last_cycle": is_last
            }
            for i, sensors_value in enumerate(sensors, start=1):
                record[f'senosr_{i}'] = sensors_value
            inserted_rows.append(record)



def read_whitesapce_delimited(filepath):
    columns = (
    ["unit_number", "cycle_number"]
    + [f"op_settings_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)])

    df = pd.read_csv(filepath, sep=r"\s+", header=None, skipinitialspace=True, names=columns)
    df = df.dropna(axis=1, how="all")
    return df


def assert_positional_order(subset, test_raw, rul_truth):
    test_units = test_raw["unit_number"].unique()
    assert len(rul_truth) == len(test_units), (
        f"RUL file has {len(rul_truth)} rows but test has "
        f"{len(test_units)} engine. These must match 1:1, "
        "in order, or labels will be silently mismatched."
    )

    expected_units = list(test_units)
    assert expected_units == sorted(expected_units), (
        f"{subset}: test units numbers are not in positional order"
        f"{expected_units}"
    )
