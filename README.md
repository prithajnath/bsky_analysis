# Bluesky analysis

Script for Bluesky ETL. Will need Airflow or some kind of orchestrator to run scripts hourly/nightly

## Scripts

You can run these scripts for as long as you want and press Ctrl-C and it will save the users to a CSV file

### fetch_random_users.py

Grabs users for every letter of the alphabet and saves them to a CSV

```py
python -m fetch_random_users.py -l 100 -b 1

```

## Useful endpoints

- Grab profiles based on a term https://docs.bsky.app/docs/api/app-bsky-actor-search-actors
- Grab followers of a user https://docs.bsky.app/docs/api/app-bsky-graph-get-followers
