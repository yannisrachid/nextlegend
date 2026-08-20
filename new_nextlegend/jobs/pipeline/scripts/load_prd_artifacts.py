"""Load precomputed pipeline artifacts into a low-memory production database."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import psycopg2
import pyarrow.parquet as pq
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import db
from pipeline.data_quality import analyze_club_player_counts, log_data_quality_warnings


def copy_batches(cursor, artifact: Path, table: str, columns: list[str], batch_size: int) -> int:
    parquet = pq.ParquetFile(artifact)
    total = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False, header=False, na_rep="\\N", quoting=csv.QUOTE_MINIMAL)
        buffer.seek(0)
        cursor.copy_expert(
            f"COPY {table} ({','.join(columns)}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')",
            buffer,
        )
        total += len(frame)
        print(f"[PRD-STAGE] {artifact.stem} {total}", flush=True)
    return total


def copy_player_metrics_stream(
    engine,
    artifact: Path,
    season_index: dict,
    ids: dict,
    batch_size: int,
) -> int:
    parquet = pq.ParquetFile(artifact)
    total = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        db.upsert_player_metrics(engine, frame, season_index, ids)
        total += len(frame)
        print(f"[PRD-LOAD] player_metrics {total}", flush=True)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="/data/prd_upsert_artifacts_current")
    parser.add_argument("--rows-processed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--metrics-batch-size", type=int, default=1_500)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    database_url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)

    engine = create_engine(database_url)
    db.ensure_schema(engine)

    print("[PRD-LOAD] reading base artifacts", flush=True)
    competitions = pd.read_parquet(artifact_dir / "competitions.parquet")
    seasons = pd.read_parquet(artifact_dir / "seasons.parquet")
    players = pd.read_parquet(artifact_dir / "players.parquet")
    clubs = pd.read_parquet(artifact_dir / "clubs.parquet")
    fact = pd.read_parquet(artifact_dir / "player_seasons.parquet")

    data_quality_warnings = analyze_club_player_counts(fact)
    log_data_quality_warnings(data_quality_warnings)

    print("[PRD-LOAD] upserting dimensions", flush=True)
    db.upsert_dimensions(engine, competitions, seasons, players, clubs)
    ids = db.resolve_ids(engine)

    print("[PRD-LOAD] replacing incoming fact slices", flush=True)
    db.purge_fact_slice(engine, fact, ids)
    season_index = db.upsert_player_seasons(engine, fact, ids)
    metrics_total = copy_player_metrics_stream(
        engine,
        artifact_dir / "player_metrics.parquet",
        season_index,
        ids,
        args.metrics_batch_size,
    )
    print(f"[PRD-LOAD] player_seasons loaded {len(fact)}", flush=True)
    print(f"[PRD-LOAD] player_metrics loaded {metrics_total}", flush=True)

    connection = psycopg2.connect(database_url)
    connection.autocommit = True
    cursor = connection.cursor()

    cursor.execute("DROP TABLE IF EXISTS role_scores_stage")
    cursor.execute(
        """CREATE UNLOGGED TABLE role_scores_stage (
        wyscout_id text, competition_name text, calendar text, team_in_selected_period text,
        profile text, raw_score double precision, pct_league double precision,
        pct_global double precision, pct_global_adjusted double precision)"""
    )
    role_columns = [
        "wyscout_id",
        "competition_name",
        "calendar",
        "team_in_selected_period",
        "profile",
        "raw_score",
        "pct_league",
        "pct_global",
        "pct_global_adjusted",
    ]
    role_total = copy_batches(
        cursor,
        artifact_dir / "role_scores.parquet",
        "role_scores_stage",
        role_columns,
        args.batch_size,
    )
    cursor.execute(
        "CREATE INDEX role_scores_stage_lookup ON "
        "role_scores_stage(wyscout_id,competition_name,calendar,team_in_selected_period)"
    )
    cursor.execute("ANALYZE role_scores_stage")
    cursor.execute(
        """INSERT INTO role_scores(
        player_season_id, profile, raw_score, pct_league, pct_global, pct_global_adjusted)
        SELECT ps.id, st.profile, st.raw_score, st.pct_league, st.pct_global, st.pct_global_adjusted
        FROM role_scores_stage st
        JOIN players p ON p.wyscout_id=st.wyscout_id
        JOIN competitions c ON c.name=st.competition_name
        JOIN seasons s ON s.label=st.calendar
        JOIN player_seasons ps ON ps.player_id=p.id
          AND ps.competition_id=c.id AND ps.season_id=s.id
          AND ps.team_in_selected_period IS NOT DISTINCT FROM st.team_in_selected_period
        ON CONFLICT(player_season_id,profile) DO UPDATE SET
          raw_score=EXCLUDED.raw_score, pct_league=EXCLUDED.pct_league,
          pct_global=EXCLUDED.pct_global, pct_global_adjusted=EXCLUDED.pct_global_adjusted,
          updated_at=NOW()"""
    )
    print(f"[PRD-LOAD] role_scores affected {cursor.rowcount}", flush=True)
    cursor.execute("DROP TABLE role_scores_stage")

    cursor.execute("DROP TABLE IF EXISTS player_similarity_stage")
    cursor.execute(
        """CREATE UNLOGGED TABLE player_similarity_stage (
        player_a_id text, player_a text, team_a text, competition_a text,
        player_b_id text, player_b text, team_b text, competition_b text,
        calendar_a text, calendar_b text, profile text, similarity double precision)"""
    )
    similarity_columns = [
        "player_a_id",
        "player_a",
        "team_a",
        "competition_a",
        "player_b_id",
        "player_b",
        "team_b",
        "competition_b",
        "calendar_a",
        "calendar_b",
        "profile",
        "similarity",
    ]
    similarity_total = copy_batches(
        cursor,
        artifact_dir / "player_similarity.parquet",
        "player_similarity_stage",
        similarity_columns,
        args.batch_size,
    )
    cursor.execute(
        "CREATE INDEX player_similarity_stage_a ON "
        "player_similarity_stage(player_a_id,competition_a,calendar_a,team_a)"
    )
    cursor.execute(
        "CREATE INDEX player_similarity_stage_b ON "
        "player_similarity_stage(player_b_id,competition_b,calendar_b,team_b)"
    )
    cursor.execute("ANALYZE player_similarity_stage")
    cursor.execute(
        """INSERT INTO player_similarity(
        profile, player_a_id, player_b_id, player_a_season_id, player_b_season_id, similarity)
        SELECT st.profile, pa.id, pb.id, psa.id, psb.id, st.similarity
        FROM player_similarity_stage st
        JOIN players pa ON pa.wyscout_id=st.player_a_id
        JOIN players pb ON pb.wyscout_id=st.player_b_id
        JOIN competitions ca ON ca.name=st.competition_a
        JOIN competitions cb ON cb.name=st.competition_b
        JOIN seasons sa ON sa.label=st.calendar_a
        JOIN seasons sb ON sb.label=st.calendar_b
        LEFT JOIN player_seasons psa ON psa.player_id=pa.id
          AND psa.competition_id=ca.id AND psa.season_id=sa.id
          AND psa.team_in_selected_period IS NOT DISTINCT FROM st.team_a
        LEFT JOIN player_seasons psb ON psb.player_id=pb.id
          AND psb.competition_id=cb.id AND psb.season_id=sb.id
          AND psb.team_in_selected_period IS NOT DISTINCT FROM st.team_b"""
    )
    print(f"[PRD-LOAD] similarity inserted {cursor.rowcount}", flush=True)
    cursor.execute("DROP TABLE player_similarity_stage")

    run_id = f"prd-{uuid.uuid4()}"
    cursor.execute(
        """INSERT INTO pipeline_runs(
        run_id, started_at, ended_at, status, source_uri, rows_processed, message)
        VALUES(%s, NOW(), NOW(), 'success', '/data/wyscout_players_final.csv', %s, %s)""",
        (
            run_id,
            args.rows_processed,
            json.dumps(
                {
                    "source": "production artifact load",
                    "role_scores": role_total,
                    "similarity": similarity_total,
                    "data_quality": {
                        "low_club_player_counts": data_quality_warnings,
                        "warning_count": len(data_quality_warnings),
                    },
                },
                ensure_ascii=False,
            ),
        ),
    )
    print(f"[PRD-LOAD] run {run_id}", flush=True)
    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()
