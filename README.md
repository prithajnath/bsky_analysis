# Bluesky analysis

Script for Bluesky ETL. Will need Airflow or some kind of orchestrator to run scripts hourly/nightly

## Scripts

### fetch_user_profiles.py

Usage

```py
python fetch_user_profiles.py
```

Grabs user profiles and writes them to a CSV. You can run the script for as long as you want and press Ctrl-C and it will save the users to a CSV file

## Useful endpoints

- Grab profiles based on a term

  https://docs.bsky.app/docs/api/app-bsky-actor-get-profiles

- Grab followers of a user
  https://docs.bsky.app/docs/api/app-bsky-graph-get-followers
