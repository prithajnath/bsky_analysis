import duckdb
import pandas as pd
import matplotlib.pyplot as plt

# connecting to the DuckDB database
conn = duckdb.connect("file.db")

# load the users into a Pandas DataFrame
df = conn.execute("SELECT * FROM users").df()
#Export to csv Only uncomment when exporting to csv, and use different csv name.
#df.to_csv("users_export.csv", index=False)

# Set Pandas display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

print(df.head(20))

# filter out invalid `created_at` values before conversion
# Remove NaN values first
df = df[df["created_at"].notna()]
# Exclude rows where `created_at` is "0"
df = df[df["created_at"].astype(str) != "0"]

# convert `created_at` to datetime format, coercing errors (invalid values become NaT)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# drop any remaining NaT values
df = df.dropna(subset=["created_at"])

# ensure all dates are valid (year >= 1900 to avoid system issues)
df = df[df["created_at"].dt.year >= 1900]

# group by date and count the number of accounts created per day
df_grouped = df.groupby(df["created_at"].dt.date).size().reset_index(name="count")

# plot
plt.figure(figsize=(12, 6))
plt.plot(df_grouped["created_at"], df_grouped["count"], marker="o", linestyle="-", color="blue")

# Improve formatting
plt.xlabel("Account Creation Date")
plt.ylabel("Number of Accounts Created")
plt.title("Number of Accounts Created Per Day")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)

# Show plot
plt.show()
