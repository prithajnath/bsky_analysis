import duckdb
import os
from bsky import DB_FILENAME

HEADER = "sample_users"

sample_sql = os.path.join(os.path.dirname(__file__), "sql", f"{HEADER}.sql")
output_filename = f"{HEADER}.csv"


with duckdb.connect(DB_FILENAME) as conn:
    with open(sample_sql) as f:
        sample_users_sql = f.read()

    df = conn.execute(sample_users_sql).df()


df.to_csv(output_filename, index=False)
