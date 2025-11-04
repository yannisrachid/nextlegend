"""Helpers for accessing NextLegend data stored on AWS S3."""

from __future__ import annotations

import os
from functools import lru_cache
from io import BytesIO, StringIO
from typing import Any, Optional
from urllib.parse import urlparse

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from pathlib import Path

# Hard-coded defaults (project Tier: bucket is fixed)
DEFAULT_BUCKET = "nextlegend"

try:  # pragma: no cover - optional dependency during runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - best effort, fallback to environment only
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv:  # pragma: no branch - only executed when package available
    project_root = Path(__file__).resolve().parents[1]
    dotenv_file = project_root / ".env"
    if dotenv_file.exists():
        load_dotenv(dotenv_file, override=False)
    else:
        load_dotenv()

if not os.getenv("NEXTLEGEND_S3_BUCKET"):
    os.environ["NEXTLEGEND_S3_BUCKET"] = DEFAULT_BUCKET


class S3ConfigurationError(RuntimeError):
    """Raised when the S3 connection cannot be configured."""


def _ensure_bucket(bucket: Optional[str]) -> str:
    value = bucket or os.getenv("NEXTLEGEND_S3_BUCKET")
    if not value:
        raise S3ConfigurationError("Missing NEXTLEGEND_S3_BUCKET environment variable.")
    return value


@lru_cache(maxsize=1)
def get_s3_client() -> Any:
    """Return a cached boto3 S3 client."""

    try:
        return boto3.client("s3")
    except NoCredentialsError as exc:  # pragma: no cover - runtime feedback
        raise S3ConfigurationError(
            "AWS credentials are not configured. Fill the .env file or export the "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY variables."
        ) from exc
    except BotoCoreError as exc:  # pragma: no cover - runtime feedback
        raise S3ConfigurationError(f"Unable to initialise S3 client: {exc}") from exc


def resolve_s3_path(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Return (bucket, key) for an S3 path or key.

    Accepts either:
        * absolute URI: ``s3://bucket/key``;
        * raw key relative to the default bucket (NEXTLEGEND_S3_BUCKET).
    """

    if isinstance(path, os.PathLike):
        path = os.fspath(path)

    text = str(path)
    if text.startswith("s3://"):
        parsed = urlparse(text)
        bucket = _ensure_bucket(parsed.netloc or None)
        key = parsed.path.lstrip("/")
        if not key:
            raise S3ConfigurationError(f"Invalid S3 path without key: {text}")
        return bucket, key
    return _ensure_bucket(None), text.lstrip("/")


def _read_object(bucket: str, key: str) -> bytes:
    client = get_s3_client()
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"NoSuchKey", "404"}:
            raise FileNotFoundError(f"S3 object not found: s3://{bucket}/{key}") from exc
        raise
    return response["Body"].read()


def _put_object(bucket: str, key: str, body: bytes, *, content_type: Optional[str] = None) -> None:
    client = get_s3_client()
    extra_args = {"ContentType": content_type} if content_type else {}
    client.put_object(Bucket=bucket, Key=key, Body=body, **extra_args)


def read_csv_from_s3(path: str | os.PathLike[str], **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV file stored on S3 into a DataFrame."""

    bucket, key = resolve_s3_path(path)
    payload = _read_object(bucket, key)
    buffer = BytesIO(payload)
    return pd.read_csv(buffer, **read_csv_kwargs)


def write_csv_to_s3(
    df: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    index: bool = False,
    **to_csv_kwargs: Any,
) -> None:
    """Serialise a DataFrame to CSV and upload it to S3."""

    bucket, key = resolve_s3_path(path)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=index, **to_csv_kwargs)
    _put_object(bucket, key, csv_buffer.getvalue().encode("utf-8"), content_type="text/csv")


def read_json_from_s3(path: str | os.PathLike[str], **json_kwargs: Any) -> Any:
    bucket, key = resolve_s3_path(path)
    payload = _read_object(bucket, key)
    buffer = BytesIO(payload)
    return pd.read_json(buffer, **json_kwargs)


def download_bytes(path: str | os.PathLike[str]) -> bytes:
    bucket, key = resolve_s3_path(path)
    return _read_object(bucket, key)


def upload_bytes(data: bytes, path: str | os.PathLike[str], *, content_type: Optional[str] = None) -> None:
    bucket, key = resolve_s3_path(path)
    _put_object(bucket, key, data, content_type=content_type)


def object_exists(path: str | os.PathLike[str]) -> bool:
    bucket, key = resolve_s3_path(path)
    client = get_s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def list_prefix(prefix: str, *, bucket: Optional[str] = None) -> list[str]:
    resolved_bucket = _ensure_bucket(bucket)
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=resolved_bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            keys.append(item["Key"])
    return keys
