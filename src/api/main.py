from fastapi import FastAPI, HTTPException
from db.db_utils import get_connection
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel


app = FastAPI()

class EngineDetail(BaseModel):
    engine: dict
    cycles: list[dict]
    predictions: list[dict]


@app.get("/fleet/status")
def get_fleet_status(risk_tier: str | None = None):
    with get_connection() as conn:
        with conn.cursor() as cursor:

            if risk_tier:
                sql = """
                    SELECT * FROM fleet_status WHERE risk_tier = %s;
                    """
                params = (risk_tier,)
            else:
                sql = """
                    SELECT * FROM fleet_status;
                    """
                params = None

            cursor.execute(sql, params)
            rows = cursor.fetchall()
    return rows


@app.get("/engine/{engine_pk}", response_model=EngineDetail)
def get_engine_details(engine_pk: int):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:

            cursor.execute("""
                SELECT * FROM engines WHERE engine_pk = %s;
                """, (engine_pk,)
            )
            engine = cursor.fetchone()

            if engine is None:
                raise HTTPException(status_code=404, detail="Engine not found",)

            cursor.execute("""
                SELECT * FROM cycles WHERE engine_pk = %s ORDER BY cycle_number;
                """, (engine_pk,)
            )
            cycles = cursor.fetchall()

            cursor.execute("""
                    SELECT p.* FROM predictions p
                    JOIN cycles c ON c.cycles_pk = p.cycle_pk
                    WHERE c.engine_pk = %s;
                """, (engine_pk,)
            )
            predictions = cursor.fetchall()

        return EngineDetail(engine, cycles, predictions)


@app.get("/models")
def get_all_model_run():
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT
                    mr.model_run_pk,
                    mr.model_name,
                    mr.task_type,
                    mr.feature_set_version,
                    mr.trained_at,
                    em.metric_name,
                    em.metric_value,
                    em.eval_split
                FROM model_runs mr
                LEFT JOIN evaluation_metrics em ON em.model_run_pk = mr.model_run_pk
                WHERE em.model_run_pk = %s;
                """
            )
            model_run = cursor.fetchall()

        return model_run
