# Bluesky analysis

Script for Bluesky ETL. Will need Airflow or some kind of orchestrator to run scripts hourly/nightly

## Scripts

You can run these scripts for as long as you want and press Ctrl-C and it will save the users to a CSV file

### fetch_user_profiles.py

Usage

```py
python fetch_user_profiles.py
```

Grabs user profiles and writes them to a CSV.

### fetch_user_posts_and_reposts.py

Usage

```py
python fetch_user_posts_and_reposts.py -hdl defnull.bsky.social
```

Grabs posts and re-posts for a given users.

## Useful endpoints

- Grab profiles based on a term https://docs.bsky.app/docs/api/app-bsky-actor-search-actors
- Grab followers of a user https://docs.bsky.app/docs/api/app-bsky-graph-get-followers
