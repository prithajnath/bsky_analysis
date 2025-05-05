import pandas as pd
import numpy as np
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from multiprocessing import Pool, cpu_count
from itertools import product
from matplotlib import pyplot as plt
import seaborn as sns
import duckdb


def _worker(args):
    X_scaled, eps, min_samples = args

    print(f"Running DBSCAN for {eps};{min_samples}")
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    if len(set(labels)) > 1 and len(set(labels)) < len(X_scaled):
        try:
            score = silhouette_score(X_scaled, labels)
        except:
            score = 0

    return eps, min_samples, score


def hyperparam_dbscan(X_scaled):

    eps_values = np.linspace(1, 10, 20)
    min_samples_values = range(3, 10)

    tasks = [
        (X_scaled, eps, min_samples)
        for eps, min_samples in product(eps_values, min_samples_values)
    ]
    total = len(tasks)
    chunksize = max(1, len(tasks) // (cpu_count() * 4))

    with Pool() as pool:
        results = pool.imap_unordered(_worker, tasks, chunksize=chunksize)
        max_score = max(i[2] for i in results)
        best_params = [i for i in results if i[2] == max_score][0]

    return best_params, max_score


if __name__ == "__main__":
    with duckdb.connect("file.db") as conn:

        conn.execute(
            """
        create view if not exists
        all_posts as select * from read_parquet('nborland_posts') union all select * from read_parquet('shawn_posts') union all select * from read_parquet('prithaj_posts');
        """
        )

        with open("sql/rolling_6_week_avg.sql") as f:
            rolling_avg_sql = f.read()

        rolling_avg_df = conn.execute(rolling_avg_sql).df()

    X_data = rolling_avg_df.pivot_table(
        columns="post_date", index="author_did", values="rolling_6week_avg"
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    result = hyperparam_dbscan(X_scaled)
    print(result)
