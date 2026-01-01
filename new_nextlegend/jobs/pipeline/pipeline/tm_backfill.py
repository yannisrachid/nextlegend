"""
Backfill Transfermarkt columns from the enriched CSV into player tables.
This avoids rewriting metrics/similarity when only TM data changes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
from botocore.client import Config
from sqlalchemy import create_engine

from . import db, processing


@dataclass
class BackfillConfig:
    input_uri: str
    bucket: str
    run_id: str
    limit: Optional[int]
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    db_url: str


BASE_COLS = [
    "player_id",
    "player",
    "competition_name",
    "calendar",
    "team_in_selected_period",
    "team",
    "minutes_played",
    "matches_played",
    "position",
    "second_position",
    "assigned_role",
    "assigned_role_pct_league",
    "assigned_role_pct_global",
    "global_score_adjusted",
    "league_strength_factor",
]


def parse_args() -> BackfillConfig:
    parser = argparse.ArgumentParser(description="Backfill Transfermarkt fields from enriched CSV.")
    parser.add_argument(
        "--input-uri",
        default=os.getenv("PIPELINE_INPUT_URI", "s3://nextlegend/data/wyscout_players_cleaned.csv"),
        help="S3 URI or local path to the enriched CSV.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for testing.")
    parser.add_argument("--run-id", default=str(uuid.uuid4()), help="Run identifier (default: uuid4).")

    args = parser.parse_args()

    return BackfillConfig(
        input_uri=args.input_uri,
        bucket=os.getenv("S3_BUCKET", ""),
        run_id=args.run_id,
        limit=args.limit,
        s3_endpoint=os.getenv("S3_ENDPOINT", ""),
        s3_access_key=os.getenv("S3_ACCESS_KEY", ""),
        s3_secret_key=os.getenv("S3_SECRET_KEY", ""),
        db_url=os.getenv("DATABASE_URL", ""),
    )


def s3_client(cfg: BackfillConfig):
    return boto3.client(
        "s3",
        endpoint_url=cfg.s3_endpoint or None,
        aws_access_key_id=cfg.s3_access_key or None,
        aws_secret_access_key=cfg.s3_secret_key or None,
        config=Config(signature_version="s3v4"),
    )


def download_input(cfg: BackfillConfig, tmp_dir: Path) -> Path:
    input_uri = cfg.input_uri
    local_path = tmp_dir / "tm_backfill_input.csv"
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


def load_dataframe(path: Path, limit: Optional[int]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    tm_cols = [c for c in header.columns if c.startswith("tm_")]
    usecols = [c for c in BASE_COLS if c in header.columns] + tm_cols
    df = pd.read_csv(path, usecols=usecols)
    if limit:
        df = df.head(limit)
    return df


def main() -> None:
    cfg = parse_args()
    if not cfg.db_url:
        raise RuntimeError("DATABASE_URL not set; cannot backfill.")

    tmp_dir = Path("/tmp/pipeline")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    input_path = download_input(cfg, tmp_dir)
    df_raw = load_dataframe(input_path, cfg.limit)
    artifacts = processing.build_artifacts_from_enriched(df_raw, similarity_df=None)

    engine = create_engine(cfg.db_url)
    db.ensure_schema(engine)

    comps = artifacts.get("competitions", pd.DataFrame())
    seasons = artifacts.get("seasons", pd.DataFrame())
    players = artifacts.get("players", pd.DataFrame())
    clubs = artifacts.get("clubs", pd.DataFrame())
    fact = artifacts.get("player_seasons", pd.DataFrame())

    db.upsert_dimensions(engine, comps, seasons, players, clubs)
    ids = db.resolve_ids(engine)
    db.upsert_player_seasons(engine, fact, ids)
    db.insert_pipeline_run(
        engine,
        cfg.run_id,
        status="success",
        source_uri=cfg.input_uri,
        rows_processed=len(fact),
        message=f"tm_backfill {dt.datetime.utcnow().isoformat()}",
    )


if __name__ == "__main__":
    main()
