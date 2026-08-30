"""
DB utilities: ensure schema and perform bulk upserts aligned with DATA_MODEL.md.
This is a minimal implementation to unblock ingestion; it handles core tables and
uses psycopg2 execute_values for ON CONFLICT upserts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from io import StringIO

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from . import scoring_v2


RESERVED_PLAYER_METRIC_COLUMNS = {"player_season_id", "created_at", "updated_at"}


def _quote_ident(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS competitions (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    country TEXT,
    type TEXT,
    tier INT,
    strength_factor DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS seasons (
    id SERIAL PRIMARY KEY,
    label TEXT UNIQUE NOT NULL,
    start_year INT,
    end_year INT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS clubs (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    competition_id INT REFERENCES competitions(id),
    country TEXT,
    external_wyscout_id TEXT,
    external_tm_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, competition_id)
);

CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    wyscout_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    country TEXT,
    birth_date DATE,
    age INT,
    height_cm DOUBLE PRECISION,
    weight_kg DOUBLE PRECISION,
    foot TEXT,
    tm_id TEXT,
    tm_profile_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS player_seasons (
    id SERIAL PRIMARY KEY,
    player_id INT REFERENCES players(id),
    competition_id INT REFERENCES competitions(id),
    season_id INT REFERENCES seasons(id),
    club_id INT REFERENCES clubs(id),
    calendar TEXT,
    team_in_selected_period TEXT,
    position TEXT,
    second_position TEXT,
    minutes_played DOUBLE PRECISION,
    matches_played DOUBLE PRECISION,
    assigned_role TEXT,
    assigned_role_pct_league DOUBLE PRECISION,
    assigned_role_pct_global DOUBLE PRECISION,
    global_score_adjusted DOUBLE PRECISION,
    league_strength_factor DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(player_id, competition_id, season_id, club_id)
);

CREATE TABLE IF NOT EXISTS player_metrics (
    player_season_id INT PRIMARY KEY REFERENCES player_seasons(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS role_scores (
    id SERIAL PRIMARY KEY,
    player_season_id INT REFERENCES player_seasons(id),
    profile TEXT,
    raw_score DOUBLE PRECISION,
    pct_league DOUBLE PRECISION,
    pct_global DOUBLE PRECISION,
    pct_global_adjusted DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS role_scores_unique ON role_scores (player_season_id, profile);
CREATE INDEX IF NOT EXISTS player_seasons_competition_season_idx ON player_seasons (competition_id, season_id);
CREATE INDEX IF NOT EXISTS role_scores_player_season_idx ON role_scores (player_season_id);

CREATE TABLE IF NOT EXISTS player_metric_percentiles_global (
    id SERIAL PRIMARY KEY,
    player_season_id INT NOT NULL REFERENCES player_seasons(id) ON DELETE CASCADE,
    season_id INT NOT NULL REFERENCES seasons(id),
    metric_key TEXT NOT NULL,
    raw_value DOUBLE PRECISION,
    percentile DOUBLE PRECISION,
    position_group TEXT NOT NULL,
    sample_size INT NOT NULL,
    lower_is_better BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(player_season_id, metric_key)
);
CREATE INDEX IF NOT EXISTS player_metric_percentiles_global_ps_idx
    ON player_metric_percentiles_global(player_season_id);
CREATE INDEX IF NOT EXISTS player_metric_percentiles_global_scope_idx
    ON player_metric_percentiles_global(season_id, position_group, metric_key);

CREATE TABLE IF NOT EXISTS player_metric_percentiles_league (
    id SERIAL PRIMARY KEY,
    player_season_id INT NOT NULL REFERENCES player_seasons(id) ON DELETE CASCADE,
    season_id INT NOT NULL REFERENCES seasons(id),
    competition_id INT NOT NULL REFERENCES competitions(id),
    metric_key TEXT NOT NULL,
    raw_value DOUBLE PRECISION,
    percentile DOUBLE PRECISION,
    position_group TEXT NOT NULL,
    sample_size INT NOT NULL,
    lower_is_better BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(player_season_id, metric_key)
);
CREATE INDEX IF NOT EXISTS player_metric_percentiles_league_ps_idx
    ON player_metric_percentiles_league(player_season_id);
CREATE INDEX IF NOT EXISTS player_metric_percentiles_league_scope_idx
    ON player_metric_percentiles_league(season_id, competition_id, position_group, metric_key);

CREATE TABLE IF NOT EXISTS player_similarity (
    id SERIAL PRIMARY KEY,
    profile TEXT,
    player_a_id INT REFERENCES players(id),
    player_b_id INT REFERENCES players(id),
    player_a_season_id INT REFERENCES player_seasons(id),
    player_b_season_id INT REFERENCES player_seasons(id),
    similarity DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS player_similarity_a_season_idx ON player_similarity (player_a_season_id);
CREATE INDEX IF NOT EXISTS player_similarity_b_season_idx ON player_similarity (player_b_season_id);
DELETE FROM player_similarity
WHERE player_a_season_id IS NULL
   OR player_b_season_id IS NULL
   OR profile IS NULL;
DELETE FROM player_similarity newer
USING player_similarity older
WHERE newer.id > older.id
  AND newer.player_a_season_id = older.player_a_season_id
  AND newer.player_b_season_id = older.player_b_season_id
  AND newer.profile = older.profile;
CREATE UNIQUE INDEX IF NOT EXISTS player_similarity_unique_edge
    ON player_similarity (player_a_season_id, player_b_season_id, profile);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id TEXT UNIQUE,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    status TEXT,
    source_uri TEXT,
    enriched_uri TEXT,
    rows_processed INT,
    message TEXT,
    git_sha TEXT
);

CREATE TABLE IF NOT EXISTS scoring_snapshot_runs (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    snapshot_key TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season_id INT NOT NULL REFERENCES seasons(id),
    season_label TEXT NOT NULL,
    cadence TEXT NOT NULL DEFAULT 'biweekly',
    source_uri TEXT,
    scoring_model_version TEXT,
    scoring_model_hash TEXT,
    rows_snapshotted INT NOT NULL DEFAULT 0,
    metric_rows_snapshotted INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(season_id, snapshot_key, scoring_model_hash)
);
CREATE INDEX IF NOT EXISTS scoring_snapshot_runs_season_date_idx
    ON scoring_snapshot_runs(season_id, snapshot_date);
CREATE INDEX IF NOT EXISTS scoring_snapshot_runs_run_idx
    ON scoring_snapshot_runs(run_id);

CREATE TABLE IF NOT EXISTS player_score_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_run_id INT NOT NULL REFERENCES scoring_snapshot_runs(id) ON DELETE CASCADE,
    player_season_id INT NOT NULL REFERENCES player_seasons(id) ON DELETE CASCADE,
    player_id INT NOT NULL REFERENCES players(id),
    competition_id INT REFERENCES competitions(id),
    club_id INT REFERENCES clubs(id),
    position TEXT,
    position_group TEXT,
    minutes_played DOUBLE PRECISION,
    matches_played DOUBLE PRECISION,
    minutes_possible DOUBLE PRECISION,
    minutes_ratio DOUBLE PRECISION,
    global_score_adjusted DOUBLE PRECISION,
    assigned_role_pct_league DOUBLE PRECISION,
    assigned_role_pct_global DOUBLE PRECISION,
    league_strength_factor DOUBLE PRECISION,
    team_strength_z DOUBLE PRECISION,
    club_strength_modifier DOUBLE PRECISION,
    minutes_regularity_modifier DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(snapshot_run_id, player_season_id)
);
CREATE INDEX IF NOT EXISTS player_score_snapshots_player_season_idx
    ON player_score_snapshots(player_season_id);
CREATE INDEX IF NOT EXISTS player_score_snapshots_player_idx
    ON player_score_snapshots(player_id, snapshot_run_id);

CREATE TABLE IF NOT EXISTS player_metric_snapshots (
    id SERIAL PRIMARY KEY,
    score_snapshot_id INT NOT NULL REFERENCES player_score_snapshots(id) ON DELETE CASCADE,
    metric_key TEXT NOT NULL,
    raw_value DOUBLE PRECISION,
    percentile_global DOUBLE PRECISION,
    percentile_league DOUBLE PRECISION,
    metric_weight DOUBLE PRECISION,
    metric_family TEXT,
    lower_is_better BOOLEAN NOT NULL DEFAULT FALSE,
    scoring_model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(score_snapshot_id, metric_key)
);
CREATE INDEX IF NOT EXISTS player_metric_snapshots_metric_idx
    ON player_metric_snapshots(metric_key);

CREATE TABLE IF NOT EXISTS prospects (
    id SERIAL PRIMARY KEY,
    player_id INT UNIQUE REFERENCES players(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS club_needs (
    id SERIAL PRIMARY KEY,
    club_id INT REFERENCES clubs(id),
    need_label TEXT NOT NULL,
    contact_name TEXT,
    contact_phone TEXT,
    assigned_user TEXT DEFAULT 'admin',
    priority_stage TEXT NOT NULL,
    sort_order INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS club_need_players (
    id SERIAL PRIMARY KEY,
    club_need_id INT REFERENCES club_needs(id) ON DELETE CASCADE,
    player_id INT REFERENCES players(id),
    sort_order INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(club_need_id, player_id)
);

CREATE INDEX IF NOT EXISTS prospects_player_id_idx ON prospects(player_id);
CREATE INDEX IF NOT EXISTS club_needs_stage_order_idx ON club_needs(priority_stage, sort_order);
CREATE INDEX IF NOT EXISTS club_need_players_order_idx ON club_need_players(club_need_id, sort_order);
"""


def ensure_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(SCHEMA_SQL))


def truncate_fact_tables(engine: Engine) -> None:
    """
    Full refresh mode: rebuild fact and downstream tables from scratch.
    Dimensions and app tables are kept intact.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE player_metric_snapshots, player_score_snapshots, scoring_snapshot_runs, player_similarity, role_scores, player_metric_percentiles_global, player_metric_percentiles_league, player_metrics, player_seasons"
            )
        )


def purge_fact_slice(engine: Engine, fact: pd.DataFrame, ids: dict) -> None:
    """
    Delete existing fact rows for the same (competition, season) slices as the incoming dataset.
    This keeps historical seasons untouched while ensuring refreshed slices do not keep stale rows.
    """
    if fact.empty or "competition_name" not in fact.columns or "calendar" not in fact.columns:
        return

    scope = (
        fact[["competition_name", "calendar"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    if scope.empty:
        return

    scope["competition_id"] = scope["competition_name"].map(ids.get("competitions", {}))
    scope["season_id"] = scope["calendar"].map(ids.get("seasons", {}))
    scope = scope.dropna(subset=["competition_id", "season_id"])
    if scope.empty:
        return

    pairs = [
        {"competition_id": int(row.competition_id), "season_id": int(row.season_id)}
        for row in scope.itertuples(index=False)
    ]
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TEMP TABLE IF NOT EXISTS _nl_slice (
                    competition_id INT NOT NULL,
                    season_id INT NOT NULL
                ) ON COMMIT DROP
                """
            )
        )
        conn.execute(text("TRUNCATE _nl_slice"))
        conn.execute(
            text("INSERT INTO _nl_slice (competition_id, season_id) VALUES (:competition_id, :season_id)"),
            pairs,
        )

        targeted = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM player_seasons ps
                JOIN _nl_slice s
                  ON s.competition_id = ps.competition_id
                 AND s.season_id = ps.season_id
                """
            )
        ).scalar()
        if not targeted:
            print(f"[DB] fact slice purge: no existing rows for pairs={len(pairs)}")
            return

        similarity_deleted = conn.execute(
            text(
                """
                DELETE FROM player_similarity
                USING player_seasons ps, _nl_slice s
                WHERE s.competition_id = ps.competition_id
                  AND s.season_id = ps.season_id
                  AND (
                        player_similarity.player_a_season_id = ps.id
                     OR player_similarity.player_b_season_id = ps.id
                  )
                """
            ),
        ).rowcount
        role_scores_deleted = conn.execute(
            text(
                """
                DELETE FROM role_scores
                USING player_seasons ps, _nl_slice s
                WHERE s.competition_id = ps.competition_id
                  AND s.season_id = ps.season_id
                  AND role_scores.player_season_id = ps.id
                """
            )
        ).rowcount
        metric_pct_global_deleted = conn.execute(
            text(
                """
                DELETE FROM player_metric_percentiles_global
                USING player_seasons ps, _nl_slice s
                WHERE s.competition_id = ps.competition_id
                  AND s.season_id = ps.season_id
                  AND player_metric_percentiles_global.player_season_id = ps.id
                """
            )
        ).rowcount
        metric_pct_league_deleted = conn.execute(
            text(
                """
                DELETE FROM player_metric_percentiles_league
                USING player_seasons ps, _nl_slice s
                WHERE s.competition_id = ps.competition_id
                  AND s.season_id = ps.season_id
                  AND player_metric_percentiles_league.player_season_id = ps.id
                """
            )
        ).rowcount
        metrics_deleted = conn.execute(
            text(
                """
                DELETE FROM player_metrics
                USING player_seasons ps, _nl_slice s
                WHERE s.competition_id = ps.competition_id
                  AND s.season_id = ps.season_id
                  AND player_metrics.player_season_id = ps.id
                """
            )
        ).rowcount
        player_seasons_deleted = conn.execute(
            text(
                """
                DELETE FROM player_seasons
                USING _nl_slice s
                WHERE s.competition_id = player_seasons.competition_id
                  AND s.season_id = player_seasons.season_id
                """
            )
        ).rowcount
        print(
            "[DB] fact slice purge:"
            f" pairs={len(pairs)} targeted={int(targeted)}"
            f" deleted player_seasons={player_seasons_deleted}"
            f" metrics={metrics_deleted} role_scores={role_scores_deleted}"
            f" metric_pct_global={metric_pct_global_deleted} metric_pct_league={metric_pct_league_deleted}"
            f" similarity={similarity_deleted}"
        )


def _execute_upsert(
    conn,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence],
    conflict_cols: Sequence[str],
    update_cols: Sequence[str],
    page_size: int = 1000,
):
    quoted_table = _quote_ident(table)
    col_list = ", ".join(_quote_ident(c) for c in columns)
    conflict = ", ".join(_quote_ident(c) for c in conflict_cols)
    update_parts = [f"{_quote_ident(c)}=EXCLUDED.{_quote_ident(c)}" for c in update_cols]
    if update_cols and table != "player_similarity":
        update_parts.append("updated_at=NOW()")
    update = ", ".join(update_parts)
    placeholders = ", ".join([f":{c}" for c in columns])
    sql = f"INSERT INTO {quoted_table} ({col_list}) VALUES ({placeholders})"
    if conflict_cols:
        sql += f" ON CONFLICT ({conflict})"
        if update:
            sql += f" DO UPDATE SET {update}"
            distinct_checks = [
                f"{quoted_table}.{_quote_ident(c)} IS DISTINCT FROM EXCLUDED.{_quote_ident(c)}"
                for c in update_cols
            ]
            if distinct_checks:
                sql += " WHERE " + " OR ".join(distinct_checks)
        else:
            sql += " DO NOTHING"

    def _clean_value(value):
        if value is None:
            return None
        if isinstance(value, np.generic):
            value = value.item()
        try:
            return None if pd.isna(value) else value
        except Exception:
            return value

    batch = []
    for row in rows:
        cleaned = {col: _clean_value(val) for col, val in zip(columns, row)}
        batch.append(cleaned)
        if len(batch) >= page_size:
            conn.execute(text(sql), batch)
            batch = []
    if batch:
        conn.execute(text(sql), batch)


def upsert_dimensions(engine: Engine, competitions: pd.DataFrame, seasons: pd.DataFrame, players: pd.DataFrame, clubs: pd.DataFrame):
    with engine.begin() as conn:
        _execute_upsert(
            conn,
            "competitions",
            ["name"],
            competitions[["name"]].itertuples(index=False, name=None),
            ["name"],
            [],
        )
        _execute_upsert(
            conn,
            "seasons",
            ["label"],
            seasons[["label"]].itertuples(index=False, name=None),
            ["label"],
            [],
        )
        player_cols = ["wyscout_id", "name"]
        update_cols = ["name"]
        if "tm_id" in players.columns:
            player_cols.append("tm_id")
            update_cols.append("tm_id")
        if "tm_profile_url" in players.columns:
            player_cols.append("tm_profile_url")
            update_cols.append("tm_profile_url")
        _execute_upsert(
            conn,
            "players",
            player_cols,
            players[player_cols].itertuples(index=False, name=None),
            ["wyscout_id"],
            update_cols,
        )
    # clubs dépend des IDs competitions -> map après insertion
    if not clubs.empty:
        comp_map = _id_map(engine, "competitions", "name")
        clubs = clubs.copy()
        clubs["competition_id"] = clubs["competition_name"].map(comp_map)
        clubs = clubs.dropna(subset=["competition_id"])
        with engine.begin() as conn:
            _execute_upsert(
                conn,
                "clubs",
                ["name", "competition_id"],
                clubs[["name", "competition_id"]].itertuples(index=False, name=None),
                ["name", "competition_id"],
                [],
            )


def _id_map(engine: Engine, table: str, key_col: str) -> dict[str, int]:
    with engine.begin() as conn:
        rows = conn.execute(text(f"SELECT id, {key_col} FROM {table}")).fetchall()
    return {str(r[1]): r[0] for r in rows}


def _club_id_maps(engine: Engine) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT c.id, c.name, comp.name AS competition_name
                FROM clubs c
                LEFT JOIN competitions comp ON comp.id = c.competition_id
                """
            )
        ).fetchall()

    names_to_ids: dict[str, set[int]] = {}
    by_competition: dict[tuple[str, str], int] = {}
    for row in rows:
        club_id = int(row.id)
        club_name = str(row.name) if row.name is not None else ""
        competition_name = str(row.competition_name) if row.competition_name is not None else ""
        if club_name:
            names_to_ids.setdefault(club_name, set()).add(club_id)
        if club_name and competition_name:
            by_competition[(competition_name, club_name)] = club_id

    # Fallback by club name is only safe when name is unique across competitions.
    by_name_unique = {
        name: next(iter(ids_set))
        for name, ids_set in names_to_ids.items()
        if len(ids_set) == 1
    }
    return by_name_unique, by_competition


def _map_club_ids(
    df: pd.DataFrame,
    *,
    team_col: str,
    competition_col: str,
    clubs_by_competition: dict[tuple[str, str], int],
    clubs_by_name: dict[str, int],
    default_value=None,
) -> pd.Series:
    if team_col not in df.columns:
        return pd.Series(default_value, index=df.index, dtype="object")

    teams = df[team_col].astype("string").str.strip().replace({"": pd.NA})
    if competition_col in df.columns:
        competitions = df[competition_col].astype("string").str.strip().replace({"": pd.NA})
        mapped = pd.Series(
            [
                clubs_by_competition.get((str(comp), str(team)))
                if pd.notna(comp) and pd.notna(team)
                else None
                for comp, team in zip(competitions, teams)
            ],
            index=df.index,
            dtype="object",
        )
    else:
        mapped = pd.Series(None, index=df.index, dtype="object")

    fallback = teams.map(clubs_by_name).astype("object")
    mapped = mapped.where(mapped.notna(), fallback)
    if default_value is None:
        return mapped.where(mapped.notna(), None)
    return mapped.where(mapped.notna(), default_value)


def resolve_ids(engine: Engine):
    clubs_by_name, clubs_by_competition = _club_id_maps(engine)
    return {
        "competitions": _id_map(engine, "competitions", "name"),
        "seasons": _id_map(engine, "seasons", "label"),
        "players": _id_map(engine, "players", "wyscout_id"),
        "clubs": clubs_by_name,
        "clubs_by_competition": clubs_by_competition,
    }


def upsert_player_seasons(engine: Engine, fact: pd.DataFrame, ids: dict):
    if fact.empty:
        return pd.DataFrame()
    fact = fact.copy()
    fact["competition_id"] = fact["competition_name"].map(ids["competitions"])
    fact["season_id"] = fact["calendar"].map(ids["seasons"])
    fact["player_id"] = fact["wyscout_id"].astype(str).map(ids["players"])
    if "team_in_selected_period" in fact.columns:
        fact["club_id"] = _map_club_ids(
            fact,
            team_col="team_in_selected_period",
            competition_col="competition_name",
            clubs_by_competition=ids.get("clubs_by_competition", {}),
            clubs_by_name=ids.get("clubs", {}),
            default_value=None,
        )
    else:
        fact["club_id"] = None
    fact = fact.dropna(subset=["competition_id", "season_id", "player_id"])

    tm_cols = [c for c in fact.columns if c.startswith("tm_")]
    if tm_cols:
        with engine.begin() as conn:
            existing = conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'player_seasons'")
            ).fetchall()
            existing_cols = {r[0] for r in existing}
            for col in tm_cols:
                if col in existing_cols:
                    continue
                if pd.api.types.is_numeric_dtype(fact[col]):
                    col_type = "DOUBLE PRECISION"
                else:
                    col_type = "TEXT"
                conn.execute(text(f'ALTER TABLE player_seasons ADD COLUMN "{col}" {col_type}'))

    cols = [
        "player_id",
        "competition_id",
        "season_id",
        "club_id",
        "calendar",
        "team_in_selected_period",
        "position",
        "second_position",
        "minutes_played",
        "matches_played",
        "assigned_role",
        "assigned_role_pct_league",
        "assigned_role_pct_global",
        "global_score_adjusted",
        "league_strength_factor",
    ]
    cols.extend(tm_cols)
    rows = fact[cols].itertuples(index=False, name=None)

    update_cols = [
        "position",
        "second_position",
        "minutes_played",
        "matches_played",
        "assigned_role",
        "assigned_role_pct_league",
        "assigned_role_pct_global",
        "global_score_adjusted",
        "league_strength_factor",
        "team_in_selected_period",
    ] + tm_cols

    with engine.begin() as conn:
        _execute_upsert(
            conn,
            "player_seasons",
            cols,
            rows,
            ["player_id", "competition_id", "season_id", "club_id"],
            update_cols,
        )
        # Return id map for player_seasons
        map_rows = conn.execute(
            text("SELECT id, player_id, competition_id, season_id, COALESCE(club_id, -1) AS club_id FROM player_seasons")
        ).fetchall()
    index = {(r.player_id, r.competition_id, r.season_id, r.club_id): r.id for r in map_rows}
    return index


def _copy_dataframe(conn, table: str, df: pd.DataFrame, columns: Sequence[str], chunk_size: int = 5000):
    col_list = ", ".join([f'"{c}"' for c in columns])
    copy_sql = f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
    raw = conn.connection
    with raw.cursor() as cur:
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            buffer = StringIO()
            chunk.to_csv(buffer, index=False, header=False, na_rep="\\N")
            buffer.seek(0)
            if hasattr(cur, "copy"):
                with cur.copy(copy_sql) as copy:
                    copy.write(buffer.getvalue())
            else:
                cur.copy_expert(copy_sql, buffer)


def _copy_upsert_dataframe(
    conn,
    table: str,
    df: pd.DataFrame,
    columns: Sequence[str],
    conflict_cols: Sequence[str],
    update_cols: Sequence[str],
):
    if df.empty:
        return
    stage = f"_stage_{table}"
    quoted_table = _quote_ident(table)
    quoted_stage = _quote_ident(stage)
    col_list = ", ".join(_quote_ident(c) for c in columns)
    conflict = ", ".join(_quote_ident(c) for c in conflict_cols)
    update_parts = [f"{_quote_ident(c)}=EXCLUDED.{_quote_ident(c)}" for c in update_cols]
    if update_cols and table != "player_similarity":
        update_parts.append("updated_at=NOW()")
    update = ", ".join(update_parts)
    conn.execute(text(f"DROP TABLE IF EXISTS {quoted_stage}"))
    conn.execute(text(f"CREATE TEMP TABLE {quoted_stage} (LIKE {quoted_table} INCLUDING DEFAULTS) ON COMMIT DROP"))
    _copy_dataframe(conn, stage, df, columns)
    sql = f"INSERT INTO {quoted_table} ({col_list}) SELECT {col_list} FROM {quoted_stage}"
    if conflict_cols:
        sql += f" ON CONFLICT ({conflict})"
        if update:
            sql += f" DO UPDATE SET {update}"
            distinct_checks = [
                f"{quoted_table}.{_quote_ident(c)} IS DISTINCT FROM EXCLUDED.{_quote_ident(c)}"
                for c in update_cols
            ]
            if distinct_checks:
                sql += " WHERE " + " OR ".join(distinct_checks)
        else:
            sql += " DO NOTHING"
    conn.execute(text(sql))


def upsert_player_metrics(
    engine: Engine,
    metrics: pd.DataFrame,
    season_index: dict,
    ids: dict,
    replace: bool = False,
    use_copy: bool = False,
):
    if metrics.empty:
        print("[DB] player_metrics empty; skip.")
        return
    metrics = metrics.copy()
    print(f"[DB] player_metrics input rows={len(metrics)} cols={len(metrics.columns)}")
    metrics["player_id"] = metrics["wyscout_id"].astype(str).map(ids["players"])
    metrics["competition_id"] = metrics["competition_name"].map(ids["competitions"])
    metrics["season_id"] = metrics["calendar"].map(ids["seasons"])
    if "team_in_selected_period" in metrics.columns:
        metrics["club_id"] = _map_club_ids(
            metrics,
            team_col="team_in_selected_period",
            competition_col="competition_name",
            clubs_by_competition=ids.get("clubs_by_competition", {}),
            clubs_by_name=ids.get("clubs", {}),
            default_value=-1,
        )
    else:
        metrics["club_id"] = -1
    metrics["player_season_id"] = list(
        zip(
            metrics["player_id"],
            metrics["competition_id"],
            metrics["season_id"],
            metrics["club_id"].fillna(-1),
        )
    )
    metrics["player_season_id"] = metrics["player_season_id"].map(season_index)
    before_rows = len(metrics)
    metrics = metrics.dropna(subset=["player_season_id"])
    metrics["player_season_id"] = pd.to_numeric(metrics["player_season_id"], errors="coerce")
    metrics = metrics.dropna(subset=["player_season_id"])
    metrics["player_season_id"] = metrics["player_season_id"].astype("Int64")
    dropped = before_rows - len(metrics)
    print(f"[DB] player_metrics mapped rows={len(metrics)} dropped={dropped}")
    # Remove helper columns
    metrics = metrics.drop(columns=["club_id", "player_id", "competition_id", "season_id"], errors="ignore")
    metrics = metrics.drop(columns=["wyscout_id", "competition_name", "calendar", "team_in_selected_period", "team"], errors="ignore")

    metric_cols = [c for c in metrics.columns if c != "player_season_id"]
    print(f"[DB] player_metrics columns={len(metric_cols)}")
    if not metric_cols:
        print("[WARN] player_metrics has no numeric columns after cleanup")
        return
    effective_use_copy = use_copy or len(metric_cols) > 200
    # Ensure columns exist and optionally prune stale derived columns.
    if metric_cols:
        with engine.begin() as conn:
            existing = conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'player_metrics'")
            ).fetchall()
            existing_cols = {r[0] for r in existing}
            for col in metric_cols:
                if col not in existing_cols:
                    conn.execute(text(f"ALTER TABLE player_metrics ADD COLUMN {_quote_ident(col)} DOUBLE PRECISION"))
            prune_columns = os.getenv("PIPELINE_PRUNE_METRIC_COLUMNS", "1").lower() not in {"0", "false", "no"}
            if prune_columns:
                keep_cols = set(metric_cols).union(RESERVED_PLAYER_METRIC_COLUMNS)
                stale_cols = sorted(col for col in existing_cols if col not in keep_cols)
                for col in stale_cols:
                    conn.execute(text(f"ALTER TABLE player_metrics DROP COLUMN IF EXISTS {_quote_ident(col)}"))
                if stale_cols:
                    print(f"[DB] player_metrics pruned stale columns={len(stale_cols)}")

    metrics = metrics.where(pd.notna(metrics), None)

    with engine.begin() as conn:
        if replace:
            conn.execute(text("TRUNCATE player_metrics"))
        _copy_upsert_dataframe(
            conn,
            "player_metrics",
            metrics,
            metrics.columns.tolist(),
            ["player_season_id"],
            metric_cols,
        )

        total = conn.execute(text("SELECT COUNT(*) FROM player_metrics")).scalar()
        print(f"[DB] player_metrics total rows={total}")


def upsert_metric_percentiles(
    engine: Engine,
    percentiles: pd.DataFrame,
    season_index: dict,
    ids: dict,
    *,
    scope: str,
    replace: bool = False,
):
    if percentiles.empty:
        print(f"[DB] player_metric_percentiles_{scope} empty; skip.")
        return
    if scope not in {"global", "league"}:
        raise ValueError(f"unknown metric percentile scope: {scope}")

    table = f"player_metric_percentiles_{scope}"
    percentiles = percentiles.copy()
    print(f"[DB] {table} input rows={len(percentiles)}")
    percentiles["player_id"] = percentiles["wyscout_id"].astype(str).map(ids["players"])
    percentiles["competition_id"] = percentiles["competition_name"].map(ids["competitions"])
    percentiles["season_id"] = percentiles["calendar"].map(ids["seasons"])
    percentiles["club_id"] = _map_club_ids(
        percentiles,
        team_col="team_in_selected_period",
        competition_col="competition_name",
        clubs_by_competition=ids.get("clubs_by_competition", {}),
        clubs_by_name=ids.get("clubs", {}),
        default_value=-1,
    )
    percentiles["player_season_id"] = list(
        zip(
            percentiles["player_id"],
            percentiles["competition_id"],
            percentiles["season_id"],
            percentiles["club_id"].fillna(-1),
        )
    )
    percentiles["player_season_id"] = percentiles["player_season_id"].map(season_index)
    before_rows = len(percentiles)
    percentiles = percentiles.dropna(subset=["player_season_id", "season_id", "metric_key", "position_group"])
    if scope == "league":
        percentiles = percentiles.dropna(subset=["competition_id"])
    percentiles["player_season_id"] = pd.to_numeric(percentiles["player_season_id"], errors="coerce")
    percentiles["season_id"] = pd.to_numeric(percentiles["season_id"], errors="coerce")
    percentiles["competition_id"] = pd.to_numeric(percentiles["competition_id"], errors="coerce")
    percentiles["sample_size"] = pd.to_numeric(percentiles["sample_size"], errors="coerce")
    percentiles = percentiles.dropna(subset=["player_season_id", "season_id", "sample_size"])
    if scope == "league":
        percentiles = percentiles.dropna(subset=["competition_id"])
    percentiles["player_season_id"] = percentiles["player_season_id"].astype("Int64")
    percentiles["season_id"] = percentiles["season_id"].astype("Int64")
    percentiles["sample_size"] = percentiles["sample_size"].astype("Int64")
    if scope == "league":
        percentiles["competition_id"] = percentiles["competition_id"].astype("Int64")
    dropped = before_rows - len(percentiles)
    print(f"[DB] {table} mapped rows={len(percentiles)} dropped={dropped}")

    if scope == "global":
        cols = [
            "player_season_id",
            "season_id",
            "metric_key",
            "raw_value",
            "percentile",
            "position_group",
            "sample_size",
            "lower_is_better",
        ]
    else:
        cols = [
            "player_season_id",
            "season_id",
            "competition_id",
            "metric_key",
            "raw_value",
            "percentile",
            "position_group",
            "sample_size",
            "lower_is_better",
        ]
    percentiles = percentiles[cols].drop_duplicates(subset=["player_season_id", "metric_key"], keep="last")
    percentiles = percentiles.where(pd.notna(percentiles), None)
    update_cols = [col for col in cols if col not in {"player_season_id", "metric_key"}]

    with engine.begin() as conn:
        if replace:
            conn.execute(text(f"TRUNCATE {table}"))
        if len(percentiles) >= 50_000:
            _copy_upsert_dataframe(conn, table, percentiles, cols, ["player_season_id", "metric_key"], update_cols)
        else:
            rows = percentiles[cols].itertuples(index=False, name=None)
            _execute_upsert(
                conn,
                table,
                cols,
                rows,
                ["player_season_id", "metric_key"],
                update_cols,
            )
        total = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        print(f"[DB] {table} total rows={total}")


def upsert_role_scores(engine: Engine, role_scores: pd.DataFrame, season_index: dict, ids: dict):
    if role_scores.empty:
        return
    role_scores = role_scores.copy()
    if "team_in_selected_period" in role_scores.columns:
        role_scores["club_id"] = _map_club_ids(
            role_scores,
            team_col="team_in_selected_period",
            competition_col="competition_name",
            clubs_by_competition=ids.get("clubs_by_competition", {}),
            clubs_by_name=ids.get("clubs", {}),
            default_value=-1,
        )
    else:
        role_scores["club_id"] = -1
    role_scores["player_season_id"] = list(
        zip(
            role_scores["wyscout_id"].astype(str).map(ids["players"]),
            role_scores["competition_name"].map(ids["competitions"]),
            role_scores["calendar"].map(ids["seasons"]),
            role_scores["club_id"].fillna(-1),
        )
    )
    role_scores["player_season_id"] = role_scores["player_season_id"].map(season_index)
    role_scores = role_scores.dropna(subset=["player_season_id"])
    role_scores["player_season_id"] = pd.to_numeric(role_scores["player_season_id"], errors="coerce")
    role_scores = role_scores.dropna(subset=["player_season_id"])
    role_scores["player_season_id"] = role_scores["player_season_id"].astype("Int64")
    cols = ["player_season_id", "profile", "raw_score", "pct_league", "pct_global", "pct_global_adjusted"]
    rows = role_scores[cols].itertuples(index=False, name=None)
    with engine.begin() as conn:
        _execute_upsert(
            conn,
            "role_scores",
            cols,
            rows,
            ["player_season_id", "profile"],
            ["raw_score", "pct_league", "pct_global", "pct_global_adjusted"],
        )


def upsert_similarity(
    engine: Engine,
    similarity: pd.DataFrame,
    ids: dict,
    season_index: dict,
    replace: bool = False,
    use_copy: bool = False,
):
    if similarity.empty:
        return
    similarity = similarity.copy()
    print(f"[DB] player_similarity input rows={len(similarity)} cols={len(similarity.columns)}")
    if "player_a_id" in similarity.columns:
        similarity["player_a_id"] = similarity["player_a_id"].astype(str).map(ids["players"])
    else:
        similarity["player_a_id"] = similarity["player_a"].astype(str).map(ids["players"])
    if "player_b_id" in similarity.columns:
        similarity["player_b_id"] = similarity["player_b_id"].astype(str).map(ids["players"])
    else:
        similarity["player_b_id"] = similarity["player_b"].astype(str).map(ids["players"])
    print(f"[DB] player_similarity ids mapped a={similarity['player_a_id'].notna().sum()} b={similarity['player_b_id'].notna().sum()}")
    competition_a = similarity["competition_a"] if "competition_a" in similarity.columns else pd.Series(index=similarity.index, dtype="object")
    competition_b = similarity["competition_b"] if "competition_b" in similarity.columns else pd.Series(index=similarity.index, dtype="object")
    season_a = similarity["calendar_a"] if "calendar_a" in similarity.columns else pd.Series(index=similarity.index, dtype="object")
    season_b = similarity["calendar_b"] if "calendar_b" in similarity.columns else pd.Series(index=similarity.index, dtype="object")
    similarity["competition_a_id"] = competition_a.map(ids["competitions"])
    similarity["competition_b_id"] = competition_b.map(ids["competitions"])
    similarity["season_a_id"] = season_a.map(ids["seasons"])
    similarity["season_b_id"] = season_b.map(ids["seasons"])
    similarity["club_a_id"] = _map_club_ids(
        similarity,
        team_col="team_a",
        competition_col="competition_a",
        clubs_by_competition=ids.get("clubs_by_competition", {}),
        clubs_by_name=ids.get("clubs", {}),
        default_value=-1,
    )
    similarity["club_b_id"] = _map_club_ids(
        similarity,
        team_col="team_b",
        competition_col="competition_b",
        clubs_by_competition=ids.get("clubs_by_competition", {}),
        clubs_by_name=ids.get("clubs", {}),
        default_value=-1,
    )

    similarity["player_a_season_id"] = list(
        zip(
            similarity["player_a_id"],
            similarity["competition_a_id"],
            similarity["season_a_id"],
            similarity["club_a_id"],
        )
    )
    similarity["player_b_season_id"] = list(
        zip(
            similarity["player_b_id"],
            similarity["competition_b_id"],
            similarity["season_b_id"],
            similarity["club_b_id"],
        )
    )
    similarity["player_a_season_id"] = similarity["player_a_season_id"].map(season_index)
    similarity["player_b_season_id"] = similarity["player_b_season_id"].map(season_index)
    similarity = similarity.dropna(subset=["player_a_id", "player_b_id", "player_a_season_id", "player_b_season_id", "profile"])
    print(f"[DB] player_similarity rows after id mapping={len(similarity)}")
    for col in ("player_a_id", "player_b_id", "player_a_season_id", "player_b_season_id"):
        if col in similarity.columns:
            similarity[col] = similarity[col].apply(
                lambda val: int(val) if pd.notna(val) else None
            )
    cols = ["profile", "player_a_id", "player_b_id", "player_a_season_id", "player_b_season_id", "similarity"]
    similarity = similarity[cols].where(pd.notna(similarity[cols]), None)
    similarity = similarity.drop_duplicates(subset=["player_a_season_id", "player_b_season_id", "profile"], keep="last")
    topk = int(os.getenv("SIM_TOPK", "10") or "10")
    with engine.begin() as conn:
        if replace:
            conn.execute(text("TRUNCATE player_similarity"))
        if use_copy or len(similarity) >= 50_000:
            _copy_upsert_dataframe(
                conn,
                "player_similarity",
                similarity,
                cols,
                ["player_a_season_id", "player_b_season_id", "profile"],
                ["similarity", "player_a_id", "player_b_id"],
            )
        else:
            rows = similarity[cols].itertuples(index=False, name=None)
            _execute_upsert(
                conn,
                "player_similarity",
                cols,
                rows,
                ["player_a_season_id", "player_b_season_id", "profile"],
                ["similarity", "player_a_id", "player_b_id"],
            )
        if topk > 0:
            deleted = conn.execute(
                text(
                    """
                    DELETE FROM player_similarity ps
                    USING (
                        SELECT id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY player_a_season_id, profile
                                   ORDER BY similarity DESC NULLS LAST, id ASC
                               ) AS rn
                        FROM player_similarity
                        WHERE player_a_season_id IS NOT NULL
                          AND profile IS NOT NULL
                    ) ranked
                    WHERE ps.id = ranked.id
                      AND ranked.rn > :topk
                    """
                ),
                {"topk": topk},
            ).rowcount
            if deleted:
                print(f"[DB] player_similarity pruned rows above topk={topk}: {deleted}")
        total = conn.execute(text("SELECT COUNT(*) FROM player_similarity")).scalar()
        print(f"[DB] player_similarity total rows={total}")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in {"", "0", "false", "no", "off"}


def _split_env_values(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace(";", ",").replace("|", ",")
    items: list[str] = []
    for chunk in normalized.split(","):
        for item in chunk.split():
            item = item.strip()
            if item:
                items.append(item)
    return items


def _snapshot_target_seasons() -> set[str]:
    raw = os.getenv("SCORE_SNAPSHOT_SEASONS") or os.getenv("DATA_FRESHNESS_EXPECT_CALENDARS")
    values = _split_env_values(raw)
    if not values:
        values = ["2026/2027", "2026"]
    return set(values)


def _snapshot_bucket(snapshot_date: dt.date, cadence: str) -> tuple[str, dt.date]:
    cadence = (cadence or "biweekly").strip().lower()
    if cadence == "monthly":
        return (snapshot_date.strftime("%Y-%m"), snapshot_date.replace(day=1))
    if cadence == "daily":
        return (snapshot_date.isoformat(), snapshot_date)
    if cadence == "run":
        return (snapshot_date.isoformat(), snapshot_date)

    iso_year, iso_week, _ = snapshot_date.isocalendar()
    bucket_week = int(iso_week) - ((int(iso_week) - 1) % 2)
    bucket_start = dt.date.fromisocalendar(int(iso_year), bucket_week, 1)
    return (f"{iso_year}-W{bucket_week:02d}", bucket_start)


def _snapshot_date() -> dt.date:
    raw = os.getenv("SCORE_SNAPSHOT_DATE", "").strip()
    if raw:
        return dt.date.fromisoformat(raw)
    return dt.datetime.now(dt.timezone.utc).date()


def _scoring_model_payload() -> dict:
    return {
        "version": scoring_v2.SCORE_VERSION,
        "position_groups": [
            {
                "key": group.key,
                "display_name": group.display_name,
                "positions": list(group.positions),
                "metrics": [
                    {
                        "key": spec.key,
                        "weight": spec.weight,
                        "metrics": list(spec.metrics),
                        "family": spec.family,
                        "lower_is_better": spec.lower_is_better,
                    }
                    for spec in group.metrics
                ],
            }
            for group in scoring_v2.POSITION_GROUPS
        ],
    }


def _scoring_model_hash() -> str:
    payload = json.dumps(_scoring_model_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _snapshot_metric_specs() -> dict[str, dict[str, dict[str, object]]]:
    specs: dict[str, dict[str, dict[str, object]]] = {}
    for group in scoring_v2.POSITION_GROUPS:
        group_specs: dict[str, dict[str, object]] = {}
        for spec in group.metrics:
            per_metric_weight = float(spec.weight) / max(1, len(spec.metrics))
            for metric_key in spec.metrics:
                current = group_specs.get(metric_key)
                if current:
                    current["metric_weight"] = float(current["metric_weight"]) + per_metric_weight
                    continue
                group_specs[metric_key] = {
                    "metric_weight": per_metric_weight,
                    "metric_family": spec.family,
                    "lower_is_better": bool(spec.lower_is_better),
                }
        specs[group.display_name] = group_specs
        specs[group.key] = group_specs
    return specs


def _with_player_season_ids(frame: pd.DataFrame, season_index: dict, ids: dict) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mapped = frame.copy()
    mapped["player_id"] = mapped["wyscout_id"].astype(str).map(ids["players"])
    mapped["competition_id"] = mapped["competition_name"].map(ids["competitions"])
    mapped["season_id"] = mapped["calendar"].map(ids["seasons"])
    mapped["club_id"] = _map_club_ids(
        mapped,
        team_col="team_in_selected_period",
        competition_col="competition_name",
        clubs_by_competition=ids.get("clubs_by_competition", {}),
        clubs_by_name=ids.get("clubs", {}),
        default_value=-1,
    )
    mapped["player_season_id"] = list(
        zip(
            mapped["player_id"],
            mapped["competition_id"],
            mapped["season_id"],
            mapped["club_id"].fillna(-1),
        )
    )
    mapped["player_season_id"] = mapped["player_season_id"].map(season_index)
    mapped = mapped.dropna(subset=["player_id", "competition_id", "season_id", "player_season_id"])
    for col in ("player_id", "competition_id", "season_id", "player_season_id"):
        mapped[col] = pd.to_numeric(mapped[col], errors="coerce")
    mapped = mapped.dropna(subset=["player_id", "competition_id", "season_id", "player_season_id"])
    for col in ("player_id", "competition_id", "season_id", "player_season_id"):
        mapped[col] = mapped[col].astype("Int64")
    mapped["club_id"] = pd.to_numeric(mapped["club_id"].replace(-1, pd.NA), errors="coerce").astype("Int64")
    return mapped


def _percentile_lookup(percentiles: pd.DataFrame, season_index: dict, ids: dict) -> dict[tuple[int, str], float]:
    if percentiles.empty or "metric_key" not in percentiles.columns or "percentile" not in percentiles.columns:
        return {}
    mapped = _with_player_season_ids(percentiles, season_index, ids)
    if mapped.empty:
        return {}
    mapped = mapped.dropna(subset=["player_season_id", "metric_key"])
    return {
        (int(row.player_season_id), str(row.metric_key)): float(row.percentile)
        for row in mapped.itertuples(index=False)
        if pd.notna(row.percentile)
    }


def snapshot_current_season_scores(
    engine: Engine,
    *,
    run_id: str,
    source_uri: str,
    fact: pd.DataFrame,
    metrics: pd.DataFrame,
    metric_percentiles_global: pd.DataFrame,
    metric_percentiles_league: pd.DataFrame,
    season_index: dict,
    ids: dict,
) -> None:
    if not _env_flag("SCORE_SNAPSHOT_ENABLED", "0"):
        print("[DB] score snapshots disabled; skip.")
        return
    if fact.empty or "calendar" not in fact.columns:
        print("[DB] score snapshots empty fact; skip.")
        return

    target_seasons = _snapshot_target_seasons()
    fact_target = fact[fact["calendar"].astype(str).isin(target_seasons)].copy()
    if fact_target.empty:
        print(f"[DB] score snapshots no target seasons found targets={sorted(target_seasons)}")
        return

    cadence = os.getenv("SCORE_SNAPSHOT_CADENCE", "biweekly").strip().lower() or "biweekly"
    bucket_key, bucket_date = _snapshot_bucket(_snapshot_date(), cadence)
    if cadence == "run":
        bucket_key = f"{bucket_key}-{run_id}"
    model_hash = _scoring_model_hash()
    model_version = scoring_v2.SCORE_VERSION
    metric_specs = _snapshot_metric_specs()

    fact_target = _with_player_season_ids(fact_target, season_index, ids)
    if fact_target.empty:
        print("[DB] score snapshots no rows after id mapping; skip.")
        return

    metrics_target = _with_player_season_ids(metrics, season_index, ids) if not metrics.empty else pd.DataFrame()
    score_extra = pd.DataFrame()
    if not metrics_target.empty:
        extra_cols = [
            col
            for col in ("team_strength_z", "club_strength_modifier", "minutes_regularity_modifier")
            if col in metrics_target.columns
        ]
        if extra_cols:
            score_extra = (
                metrics_target[["player_season_id", *extra_cols]]
                .drop_duplicates(subset=["player_season_id"], keep="last")
                .set_index("player_season_id")
            )

    if "matches_played" in fact_target.columns:
        max_matches = pd.to_numeric(fact_target["matches_played"], errors="coerce").groupby(
            [fact_target["competition_name"], fact_target["calendar"]], dropna=False
        ).transform("max")
        minutes_possible = max_matches * 90
    else:
        minutes_possible = pd.Series(np.nan, index=fact_target.index)
    fallback_possible = pd.to_numeric(fact_target.get("minutes_played", pd.Series(np.nan, index=fact_target.index)), errors="coerce").groupby(
        [fact_target["competition_name"], fact_target["calendar"]], dropna=False
    ).transform("max")
    minutes_possible = minutes_possible.where(minutes_possible > 0, fallback_possible)
    fact_target["minutes_possible"] = minutes_possible
    minutes_played = pd.to_numeric(fact_target.get("minutes_played", pd.Series(np.nan, index=fact_target.index)), errors="coerce")
    fact_target["minutes_ratio"] = (minutes_played / fact_target["minutes_possible"]).where(fact_target["minutes_possible"] > 0)

    global_pct = _percentile_lookup(metric_percentiles_global, season_index, ids)
    league_pct = _percentile_lookup(metric_percentiles_league, season_index, ids)

    metric_values = pd.DataFrame()
    if not metrics_target.empty:
        metric_values = metrics_target.drop_duplicates(subset=["player_season_id"], keep="last").set_index("player_season_id")

    for season_label, season_rows in fact_target.groupby("calendar", dropna=False):
        season_label = str(season_label)
        season_id = ids.get("seasons", {}).get(season_label)
        if season_id is None:
            continue

        score_rows = season_rows.copy()
        if not score_extra.empty:
            score_rows = score_rows.join(score_extra, on="player_season_id")
        score_rows["position_group"] = score_rows.get("assigned_role")
        score_rows = score_rows.drop_duplicates(subset=["player_season_id"], keep="last")

        with engine.begin() as conn:
            snapshot_run_id = conn.execute(
                text(
                    """
                    INSERT INTO scoring_snapshot_runs (
                        run_id, snapshot_key, snapshot_date, season_id, season_label,
                        cadence, source_uri, scoring_model_version, scoring_model_hash
                    )
                    VALUES (
                        :run_id, :snapshot_key, :snapshot_date, :season_id, :season_label,
                        :cadence, :source_uri, :scoring_model_version, :scoring_model_hash
                    )
                    ON CONFLICT (season_id, snapshot_key, scoring_model_hash) DO UPDATE SET
                        run_id = EXCLUDED.run_id,
                        snapshot_at = NOW(),
                        cadence = EXCLUDED.cadence,
                        source_uri = EXCLUDED.source_uri,
                        scoring_model_version = EXCLUDED.scoring_model_version,
                        updated_at = NOW()
                    RETURNING id
                    """
                ),
                {
                    "run_id": run_id,
                    "snapshot_key": bucket_key,
                    "snapshot_date": bucket_date,
                    "season_id": int(season_id),
                    "season_label": season_label,
                    "cadence": cadence,
                    "source_uri": source_uri,
                    "scoring_model_version": model_version,
                    "scoring_model_hash": model_hash,
                },
            ).scalar_one()

            score_cols = [
                "snapshot_run_id",
                "player_season_id",
                "player_id",
                "competition_id",
                "club_id",
                "position",
                "position_group",
                "minutes_played",
                "matches_played",
                "minutes_possible",
                "minutes_ratio",
                "global_score_adjusted",
                "assigned_role_pct_league",
                "assigned_role_pct_global",
                "league_strength_factor",
                "team_strength_z",
                "club_strength_modifier",
                "minutes_regularity_modifier",
            ]
            for col in score_cols:
                if col not in score_rows.columns:
                    score_rows[col] = None
            score_rows["snapshot_run_id"] = int(snapshot_run_id)
            score_frame = score_rows[score_cols].where(pd.notna(score_rows[score_cols]), None)
            _copy_upsert_dataframe(
                conn,
                "player_score_snapshots",
                score_frame,
                score_cols,
                ["snapshot_run_id", "player_season_id"],
                [col for col in score_cols if col not in {"snapshot_run_id", "player_season_id"}],
            )

            snapshot_ids = {
                int(row.player_season_id): int(row.id)
                for row in conn.execute(
                    text(
                        """
                        SELECT id, player_season_id
                        FROM player_score_snapshots
                        WHERE snapshot_run_id = :snapshot_run_id
                        """
                    ),
                    {"snapshot_run_id": int(snapshot_run_id)},
                ).fetchall()
            }

            metric_rows: list[dict[str, object]] = []
            for row in score_rows.itertuples(index=False):
                player_season_id = int(getattr(row, "player_season_id"))
                score_snapshot_id = snapshot_ids.get(player_season_id)
                if not score_snapshot_id:
                    continue
                position_group = str(getattr(row, "position_group", "") or "")
                specs = metric_specs.get(position_group, {})
                if not specs:
                    continue
                source = metric_values.loc[player_season_id] if player_season_id in metric_values.index else None
                for metric_key, spec in specs.items():
                    raw_value = None
                    if source is not None and metric_key in metric_values.columns:
                        value = source.get(metric_key)
                        if pd.notna(value):
                            raw_value = float(value)
                    metric_rows.append(
                        {
                            "score_snapshot_id": score_snapshot_id,
                            "metric_key": metric_key,
                            "raw_value": raw_value,
                            "percentile_global": global_pct.get((player_season_id, metric_key)),
                            "percentile_league": league_pct.get((player_season_id, metric_key)),
                            "metric_weight": spec.get("metric_weight"),
                            "metric_family": spec.get("metric_family"),
                            "lower_is_better": spec.get("lower_is_better"),
                            "scoring_model_version": model_version,
                        }
                    )

            metric_frame = pd.DataFrame(metric_rows)
            metric_count = 0
            if not metric_frame.empty:
                metric_cols = [
                    "score_snapshot_id",
                    "metric_key",
                    "raw_value",
                    "percentile_global",
                    "percentile_league",
                    "metric_weight",
                    "metric_family",
                    "lower_is_better",
                    "scoring_model_version",
                ]
                metric_frame = metric_frame[metric_cols].drop_duplicates(
                    subset=["score_snapshot_id", "metric_key"],
                    keep="last",
                )
                metric_frame = metric_frame.where(pd.notna(metric_frame), None)
                _copy_upsert_dataframe(
                    conn,
                    "player_metric_snapshots",
                    metric_frame,
                    metric_cols,
                    ["score_snapshot_id", "metric_key"],
                    [col for col in metric_cols if col not in {"score_snapshot_id", "metric_key"}],
                )
                metric_count = len(metric_frame)

            conn.execute(
                text(
                    """
                    UPDATE scoring_snapshot_runs
                    SET rows_snapshotted = :rows_snapshotted,
                        metric_rows_snapshotted = :metric_rows_snapshotted,
                        updated_at = NOW()
                    WHERE id = :snapshot_run_id
                    """
                ),
                {
                    "rows_snapshotted": len(score_frame),
                    "metric_rows_snapshotted": metric_count,
                    "snapshot_run_id": int(snapshot_run_id),
                },
            )
        print(
            "[DB] score snapshots:"
            f" season={season_label} key={bucket_key} rows={len(score_frame)} metrics={metric_count}"
            f" model={model_version}/{model_hash}"
        )


def insert_pipeline_run(engine: Engine, run_id: str, status: str, source_uri: str, rows_processed: int, message: str = ""):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pipeline_runs (run_id, started_at, ended_at, status, source_uri, rows_processed, message)
                VALUES (:run_id, NOW(), NOW(), :status, :source_uri, :rows_processed, :message)
                ON CONFLICT (run_id) DO UPDATE SET
                    ended_at = NOW(),
                    status = EXCLUDED.status,
                    rows_processed = EXCLUDED.rows_processed,
                    message = EXCLUDED.message
                """
            ),
            {"run_id": run_id, "status": status, "source_uri": source_uri, "rows_processed": rows_processed, "message": message},
        )
