"""
python move_to_s3.py -t posts -f prithaj_posts
"""

import duckdb
import argparse
import boto3
import os
from bsky import DB_FILENAME
from tempfile import NamedTemporaryFile
from create_logger import logger
from tqdm import tqdm


S3_BUCKET = "bsky-analysis-files"


def dump_table_to_s3(tablename, filename):
    with NamedTemporaryFile("w") as f:
        tmpfile_name = f.name
        logger.debug(f"Writing parquet to {tmpfile_name}")

        # Export table to temporary parquet file
        with duckdb.connect(DB_FILENAME) as conn:
            conn.execute(
                f"""
            copy {tablename} to '{tmpfile_name}' (format parquet);
            """
            )

        # Move temp parquet file to S3
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

        logger.info(f"Uploading {tmpfile_name} to s3://{S3_BUCKET}/{filename}.parquet")
        progress_bar = tqdm(
            total=os.path.getsize(tmpfile_name),
            unit="B",
            unit_scale=True,
            desc="Uploading",
        )

        def upload_progress(chunk):
            progress_bar.update(chunk)

        s3.upload_file(
            Filename=tmpfile_name,
            Bucket=S3_BUCKET,
            Key=filename,
            Callback=upload_progress,
        )
        logger.info(f"Successfully uploaded parquet to s3://{S3_BUCKET}/{filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tablename", required=True)
    parser.add_argument("-f", "--filename", required=True)
    args = parser.parse_args()

    tablename = args.tablename
    filename = args.filename

    dump_table_to_s3(tablename=tablename, filename=filename)
