import pandas as pd
import numpy as np
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns

import re
import duckdb

HAS_MENTIONS_REGEX = r".*(@\w+)"

with open("url_regex.txt", "r") as f:
    URL_REGEX = f.read()


user_labels_df = pd.read_csv("kmeans_cluster_labels.csv")
users_df = pd.read_csv("sample_users.csv").rename(columns={"did": "author_did"})


with duckdb.connect("file.db") as conn:
    first_post_sql = """
    with ranked_posts as (
        select
            author_did,
            body,
            created_at,
            row_number() over (partition by author_did order by created_at) as r
        from
            all_posts
    ) select * from ranked_posts where r = 1;

    """
    first_post_df = conn.execute(first_post_sql).df()


# Has mentions in first post Boolean
def has_mentions_in_bio(bio):
    return True if re.match(HAS_MENTIONS_REGEX, bio) else False


# Has mentions in first post Boolean
def has_mentions_in_first_post(first_post):
    return True if re.match(HAS_MENTIONS_REGEX, first_post) else False


# Has URL in bio
def has_url_in_bio(bio):
    return True if re.match(URL_REGEX, bio) else False


# Has URL in first post
def has_url_in_first_post(first_post):
    return True if re.match(URL_REGEX, first_post) else False


transformations = [
    has_mentions_in_bio,
    has_mentions_in_first_post,
    has_url_in_bio,
    has_url_in_first_post,
]

df = pd.merge(first_post_df, users_df, how="left", on="author_did")


if __name__ == "__main__":
    for transformation in transformations:
        name = transformation.__name__
        if name.endswith("bio"):
            df[name] = df["bio"].apply(
                lambda x: transformation(x) if isinstance(x, str) else ""
            )
        if name.endswith("post"):
            df[name] = df["body"].apply(
                lambda x: transformation(x) if isinstance(x, str) else ""
            )

    final_df = df[["author_did"] + [i.__name__ for i in transformations]].sort_values(
        by="author_did"
    )

    final_df.to_csv("prithaj_features.csv", index=False)
