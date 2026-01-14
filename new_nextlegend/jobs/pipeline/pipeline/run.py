"""
Batch pipeline skeleton: download raw CSV (S3 or local), process, archive, upsert DB.
Placeholder for the v1 logic (clean + scores + percentiles + similarités).
"""

from __future__ import annotations

import argparse
import json
import datetime as dt
import os
import sys
import uuid
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
from botocore.client import Config
from sqlalchemy import create_engine
from io import BytesIO

from . import processing, db


@dataclass
class PipelineConfig:
    input_uri: str
    bucket: str
    prefix: str
    run_id: str
    input_kind: str
    similarity_prefix: str
    dry_run: bool
    limit: Optional[int]
    retain_runs: Optional[int]
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    db_url: str


@dataclass
class RunLog:
    run_id: str
    started_at: str
    ended_at: Optional[str] = None
    status: str = "running"
    source_uri: str = ""
    rows_processed: int = 0
    message: str = ""


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="NextLegend v2 pipeline runner (skeleton).")
    parser.add_argument(
        "--input-uri",
        default=os.getenv("PIPELINE_INPUT_URI", "/data/wyscout_players_final.csv"),
        help="S3 URI ou chemin local vers le CSV brut (default: /data/wyscout_players_final.csv ou PIPELINE_INPUT_URI).",
    )
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", ""), help="Target bucket for archives.")
    parser.add_argument("--prefix", default=os.getenv("S3_PREFIX", "new_nextlegend"), help="Prefix inside the bucket (e.g., new_nextlegend).")
    parser.add_argument("--run-id", default=str(uuid.uuid4()), help="Run identifier (default: uuid4).")
    parser.add_argument(
        "--input-kind",
        choices=["raw", "enriched"],
        default=os.getenv("PIPELINE_INPUT_KIND", "raw"),
        help="Input dataset type: raw (compute scores) or enriched (precomputed scores).",
    )
    parser.add_argument(
        "--similarity-prefix",
        default=os.getenv("SIMILARITY_PREFIX", "data/similarity/"),
        help="S3 key prefix or local folder for similarity files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="If set, skip writes to S3/DB.")
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("PIPELINE_LIMIT", "0") or "0") or None,
        help="Optional row limit for testing (PIPELINE_LIMIT).",
    )
    parser.add_argument(
        "--retain-runs",
        type=int,
        default=int(os.getenv("ARCHIVE_RETAIN_RUNS", "0") or "0"),
        help="If > 0, keep only the latest N runs under prefix/enriched and prefix/raw.",
    )

    args = parser.parse_args()

    return PipelineConfig(
        input_uri=args.input_uri,
        bucket=args.bucket,
        prefix=args.prefix.strip("/"),
        run_id=args.run_id,
        input_kind=args.input_kind,
        similarity_prefix=args.similarity_prefix,
        dry_run=args.dry_run,
        limit=args.limit,
        retain_runs=args.retain_runs if args.retain_runs and args.retain_runs > 0 else None,
        s3_endpoint=os.getenv("S3_ENDPOINT", ""),
        s3_access_key=os.getenv("S3_ACCESS_KEY", ""),
        s3_secret_key=os.getenv("S3_SECRET_KEY", ""),
        db_url=os.getenv("DATABASE_URL", ""),
    )


def s3_client(cfg: PipelineConfig):
    return boto3.client(
        "s3",
        endpoint_url=cfg.s3_endpoint or None,
        aws_access_key_id=cfg.s3_access_key or None,
        aws_secret_access_key=cfg.s3_secret_key or None,
        config=Config(signature_version="s3v4"),
    )


def download_input(cfg: PipelineConfig, tmp_dir: Path) -> Path:
    """
    Download the input CSV to a temp path. Supports local file passthrough.
    """
    input_uri = cfg.input_uri
    local_path = tmp_dir / "input.csv"
    if input_uri.startswith("s3://"):
        _, _, bucket_and_key = input_uri.partition("s3://")
        bucket, _, key = bucket_and_key.partition("/")
        print(f"[LOAD] downloading {input_uri} -> {local_path}")
        s3_client(cfg).download_file(bucket, key, str(local_path))
    else:
        src = Path(input_uri).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Input file not found: {src}")
        local_path.write_bytes(src.read_bytes())
    return local_path


def _list_s3_keys(client, bucket: str, prefix: str) -> list[str]:
    keys = []
    continuation = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix)
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])
        if resp.get("IsTruncated"):
            continuation = resp.get("NextContinuationToken")
        else:
            break
    return keys


def load_similarity(cfg: PipelineConfig) -> pd.DataFrame:
    """
    Load similarity files from S3 or local folder into a single DataFrame.
    Expected columns: player_a/team_a/competition_name_a, player_b/team_b/competition_name_b, profile, similarity.
    """
    prefix = (cfg.similarity_prefix or "").strip()
    if not prefix:
        return pd.DataFrame()

    if prefix.startswith("s3://"):
        _, _, bucket_and_key = prefix.partition("s3://")
        bucket, _, key_prefix = bucket_and_key.partition("/")
        bucket = bucket or cfg.bucket
        key_prefix = key_prefix.strip("/")
    else:
        bucket = cfg.bucket
        key_prefix = prefix.strip("/")

    if prefix.startswith("/") or Path(prefix).exists():
        folder = Path(prefix)
        if not folder.exists():
            return pd.DataFrame()
        frames = []
        for path in folder.glob("*.csv"):
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                print(f"[WARN] unable to read similarity file: {path}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not bucket or not key_prefix:
        return pd.DataFrame()

    client = s3_client(cfg)
    keys = _list_s3_keys(client, bucket, key_prefix + "/")
    frames = []
    for key in keys:
        if key.endswith("/"):
            continue
        if key.endswith(".csv"):
            obj = client.get_object(Bucket=bucket, Key=key)
            frames.append(pd.read_csv(obj["Body"]))
        elif key.endswith(".parquet"):
            obj = client.get_object(Bucket=bucket, Key=key)
            buf = BytesIO(obj["Body"].read())
            frames.append(pd.read_parquet(buf))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def upload_raw_if_local(cfg: PipelineConfig, local_path: Path):
    """
    Si l'entrée est locale et qu'un bucket est configuré, uploader le brut sur S3 (archivage).
    """
    if cfg.dry_run:
        print("[DRY-RUN] skip raw upload.")
        return
    if cfg.input_uri.startswith("s3://"):
        return
    if not cfg.bucket:
        print("[RAW] no bucket provided; skip raw upload.")
        return
    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    prefix = cfg.prefix or "new_nextlegend"
    key = f"{prefix}/raw/{cfg.run_id}_{ts}.csv"
    print(f"[RAW] upload local input -> s3://{cfg.bucket}/{key}")
    s3_client(cfg).upload_file(str(local_path), cfg.bucket, key)


def load_dataframe(path: Path, limit: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    return df


def process_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return processing.build_artifacts(df)


def archive_to_s3(cfg: PipelineConfig, artifacts: dict[str, pd.DataFrame]):
    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    prefix = f"{cfg.prefix or 'new_nextlegend'}/enriched/{cfg.run_id}_{ts}"
    client = s3_client(cfg)

    for name, df in artifacts.items():
        if df.empty:
            continue
        key = f"{prefix}/{name}.parquet"
        print(f"[ARCHIVE] {name} -> s3://{cfg.bucket}/{key}")
        buf = BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        client.put_object(Bucket=cfg.bucket, Key=key, Body=buf.getvalue())


def _parse_run_timestamp(label: str) -> tuple[int, str]:
    parts = label.rsplit("_", 1)
    if len(parts) == 2:
        ts = parts[1]
        try:
            dt_obj = dt.datetime.strptime(ts, "%Y%m%d-%H%M%S")
            return (int(dt_obj.timestamp()), label)
        except Exception:
            pass
    return (0, label)


def _delete_keys(client, bucket: str, keys: list[str]):
    for idx in range(0, len(keys), 1000):
        batch = keys[idx : idx + 1000]
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
        )


def prune_archives(cfg: PipelineConfig):
    if not cfg.bucket or not cfg.retain_runs or cfg.retain_runs <= 0:
        return
    client = s3_client(cfg)
    base_prefix = cfg.prefix or "new_nextlegend"
    enriched_prefix = f"{base_prefix}/enriched/"
    raw_prefix = f"{base_prefix}/raw/"

    enriched_keys = _list_s3_keys(client, cfg.bucket, enriched_prefix)
    run_prefixes = sorted(
        {key[len(enriched_prefix) :].split("/", 1)[0] for key in enriched_keys if key.startswith(enriched_prefix)},
        key=_parse_run_timestamp,
    )
    to_delete = run_prefixes[:-cfg.retain_runs] if len(run_prefixes) > cfg.retain_runs else []
    if to_delete:
        print(f"[ARCHIVE] pruning {len(to_delete)} enriched runs (retain {cfg.retain_runs})")
        for run_prefix in to_delete:
            keys = [key for key in enriched_keys if key.startswith(f"{enriched_prefix}{run_prefix}/")]
            _delete_keys(client, cfg.bucket, keys)

    raw_keys = _list_s3_keys(client, cfg.bucket, raw_prefix)
    raw_files = sorted(
        {key[len(raw_prefix) :] for key in raw_keys if key.startswith(raw_prefix)},
        key=_parse_run_timestamp,
    )
    raw_delete = raw_files[:-cfg.retain_runs] if len(raw_files) > cfg.retain_runs else []
    if raw_delete:
        print(f"[ARCHIVE] pruning {len(raw_delete)} raw files (retain {cfg.retain_runs})")
        keys = [f"{raw_prefix}{name}" for name in raw_delete]
        _delete_keys(client, cfg.bucket, keys)


def write_run_log(cfg: PipelineConfig, log: RunLog):
    if not cfg.bucket:
        return
    log.ended_at = log.ended_at or dt.datetime.utcnow().isoformat()
    key = f"{cfg.prefix or 'new_nextlegend'}/logs/{log.run_id}.json"
    client = s3_client(cfg)
    body = pd.Series(log.__dict__).to_json()
    print(f"[LOG] s3://{cfg.bucket}/{key}")
    client.put_object(Bucket=cfg.bucket, Key=key, Body=body.encode("utf-8"))


def upsert_db(cfg: PipelineConfig, artifacts: dict[str, pd.DataFrame]):
    """
    Placeholder: in the next steps, map artifacts to tables from DATA_MODEL.md.
    """
    if not cfg.db_url:
        print("[DB] DATABASE_URL not set; skipping.")
        return
    engine = create_engine(cfg.db_url)
    db.ensure_schema(engine)

    comps = artifacts.get("competitions", pd.DataFrame())
    seasons = artifacts.get("seasons", pd.DataFrame())
    players = artifacts.get("players", pd.DataFrame())
    clubs = artifacts.get("clubs", pd.DataFrame())
    fact = artifacts.get("player_seasons", pd.DataFrame())
    metrics = artifacts.get("player_metrics", pd.DataFrame())
    role_scores = artifacts.get("role_scores", pd.DataFrame())
    similarity = artifacts.get("player_similarity", pd.DataFrame())

    db.upsert_dimensions(engine, comps, seasons, players, clubs)
    ids = db.resolve_ids(engine)
    season_index = db.upsert_player_seasons(engine, fact, ids)
    replace_tables = cfg.input_kind == "enriched"
    db.upsert_player_metrics(engine, metrics, season_index, ids, replace=replace_tables, use_copy=replace_tables)
    db.upsert_role_scores(engine, role_scores, season_index, ids)
    db.upsert_similarity(engine, similarity, ids, season_index, replace=replace_tables, use_copy=replace_tables)
    db.insert_pipeline_run(engine, cfg.run_id, status="success", source_uri=cfg.input_uri, rows_processed=len(fact))


def main():
    cfg = parse_args()
    run_log = RunLog(run_id=cfg.run_id, started_at=dt.datetime.utcnow().isoformat(), status="running", source_uri=cfg.input_uri)
    if cfg.dry_run:
        print(f"[RUN] dry-run mode | run_id={cfg.run_id}")
    else:
        print(f"[RUN] run_id={cfg.run_id}")

    tmp_dir = Path("/tmp/pipeline")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_path = download_input(cfg, tmp_dir)
        upload_raw_if_local(cfg, input_path)
        df_raw = load_dataframe(input_path, cfg.limit)
        if cfg.input_kind == "enriched":
            print("[PIPELINE] ingesting enriched dataset")
            similarity_df = load_similarity(cfg)
            artifacts = processing.build_artifacts_from_enriched(df_raw, similarity_df=similarity_df)
        else:
            artifacts = process_dataframe(df_raw)
        run_log.rows_processed = len(df_raw)
        # Enrich log with artifact sizes
        sizes = {k: len(v) if isinstance(v, pd.DataFrame) else 0 for k, v in artifacts.items()}
        print(f"[PIPELINE] artifact sizes: {sizes}")
        run_log.message = json.dumps({"artifacts": sizes})

        if not cfg.dry_run:
            archive_to_s3(cfg, artifacts)
            prune_archives(cfg)
            upsert_db(cfg, artifacts)
            run_log.status = "success"
            run_log.ended_at = dt.datetime.utcnow().isoformat()
            write_run_log(cfg, run_log)
        else:
            print("[DRY-RUN] skipping archive and DB upsert.")
            run_log.status = "dry-run"
            run_log.ended_at = dt.datetime.utcnow().isoformat()
            write_run_log(cfg, run_log)
    except Exception as exc:  # noqa: BLE001
        run_log.status = "failed"
        run_log.message = str(exc)
        run_log.ended_at = dt.datetime.utcnow().isoformat()
        traceback.print_exc()
        print(f"[ERROR] {exc}", file=sys.stderr)
        try:
            write_run_log(cfg, run_log)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
