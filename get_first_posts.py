import duckdb
import pandas as pd

# connect to DuckDB database file
con = duckdb.connect("file.db")

# export first post date and body per user, deduplicated
con.execute("""
    COPY (
        SELECT
            author_did,
            created_at AS first_post_created_at,
            body AS first_post_body,
            reply_uri != 'None' AS is_reply,
            quote_uri != 'None' AS is_quote
        FROM (
            SELECT
                author_did,
                created_at,
                body,
                reply_uri,
                quote_uri,
                ROW_NUMBER() OVER (PARTITION BY author_did ORDER BY created_at ASC) AS rn
            FROM all_posts
        )
        WHERE rn = 1
    ) TO 'first_posts_per_user.csv' (HEADER, DELIMITER ',');
""")

con.close()

# load csvs
users_df = pd.read_csv("sample_users.csv")
first_posts_df = pd.read_csv("first_posts_per_user.csv")

# merge on did
merged_df = users_df.merge(
    first_posts_df,
    how="inner",
    left_on="did",
    right_on="author_did"
)

#  cleanup
merged_df = merged_df.drop(columns=["author_did"])

# Save final result
merged_df.to_csv("sample_users_with_first_posts.csv", index=False)
print("CSV written: sample_users_with_first_posts.csv")