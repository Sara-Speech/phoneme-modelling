import os
from pathlib import Path
from urllib.parse import urlparse

import boto3



def download_s3_path(s3_path: str, local_dir: Path):
    if os.path.exists(local_dir):
        os.rmdir(local_dir)
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    bucket, folder = split_s3_path(s3_path)
    for page in paginator.paginate(Bucket=bucket, Prefix=folder):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('/'):
                continue
            relative_path = os.path.relpath(key, folder)
            local_file_path = local_dir / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_file_path))

def list_s3_path(s3_path: str) -> list[str]:
    s3 = boto3.client("s3")
    bucket, folder = split_s3_path(s3_path)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=folder)
    result = []
    for obj in response.get("Contents", []):
        result.append(obj["Key"])
    return result

def split_s3_path(s3_path: str) -> tuple[str, str]:
  parsed = urlparse(s3_path, allow_fragments=False)
  bucket = parsed.netloc
  key = parsed.path.lstrip("/")
  return bucket, key