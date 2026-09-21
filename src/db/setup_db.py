from pathlib import Path
from db_utils import get_connection

SCHEMA_FILE = Path("./db/schema.sql")

if __name__ == "__main__":
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(SCHEMA_FILE.read_text())
