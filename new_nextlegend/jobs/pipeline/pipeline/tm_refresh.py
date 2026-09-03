from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from . import db
from .transfermarkt_matching import MatchConfig, build_match_candidates, prepare_transfermarkt_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly Transfermarkt refresh and Wyscout matching.")
    parser.add_argument(
        "--tm-profiles",
        default=os.getenv("TM_PROFILES_PATH", "/helpers/csv/transfermarkt_profiles.csv"),
        help="CSV produced by the Transfermarkt scraper/API.",
    )
    parser.add_argument(
        "--player-map",
        default=os.getenv("TM_PLAYER_MAP_PATH", "/helpers/csv/player_matching_reference.csv"),
        help="Optional curated Wyscout to Transfermarkt player mapping CSV.",
    )
    parser.add_argument(
        "--snapshot-date",
        default=os.getenv("TM_SNAPSHOT_DATE", dt.datetime.now(dt.timezone.utc).date().isoformat()),
        help="Market-value snapshot date, ISO format.",
    )
    parser.add_argument(
        "--season-label",
        default=os.getenv("TM_REFRESH_SEASON_LABEL", "2026/2027,2026"),
        help="Preferred Wyscout season labels for matching context, comma/space separated. Empty means all seasons.",
    )
    parser.add_argument("--run-id", default=os.getenv("TM_REFRESH_RUN_ID", str(uuid.uuid4())))
    parser.add_argument("--limit", type=int, default=int(os.getenv("TM_REFRESH_LIMIT", "0") or "0") or None)
    parser.add_argument("--dry-run", action="store_true", default=os.getenv("TM_REFRESH_DRY_RUN", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--min-candidate-score", type=float, default=float(os.getenv("TM_MIN_CANDIDATE_SCORE", "0.72")))
    parser.add_argument("--auto-accept-score", type=float, default=float(os.getenv("TM_AUTO_ACCEPT_SCORE", "0.90")))
    parser.add_argument("--auto-accept-margin", type=float, default=float(os.getenv("TM_AUTO_ACCEPT_MARGIN", "0.035")))
    return parser.parse_args()


def _clean_id(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text_value = str(value).strip().replace(".0", "")
    return text_value or None


def _split_labels(value: Optional[str]) -> list[str]:
    if not value:
        return []
    labels: list[str] = []
    for chunk in str(value).replace(";", ",").replace("|", ",").split(","):
        for part in chunk.split():
            part = part.strip()
            if part and part not in labels:
                labels.append(part)
    return labels


def apply_curated_player_map(wyscout: pd.DataFrame, player_map_path: str) -> pd.DataFrame:
    path = Path(player_map_path).expanduser()
    if not path.exists():
        return wyscout
    mapped = wyscout.copy()
    reference = pd.read_csv(path, low_memory=False)
    reference.columns = [str(col).strip().lower() for col in reference.columns]
    if "tm_player_id" not in reference.columns:
        return mapped
    mapped["existing_tm_id"] = mapped["existing_tm_id"].apply(_clean_id)
    reference["tm_player_id"] = reference["tm_player_id"].apply(_clean_id)
    reference = reference[reference["tm_player_id"].notna()]
    if reference.empty:
        return mapped

    direct_col = "wyscout_player_id" if "wyscout_player_id" in reference.columns else None
    if direct_col:
        direct = reference[[direct_col, "tm_player_id"]].dropna().drop_duplicates(subset=[direct_col], keep="last")
        direct[direct_col] = direct[direct_col].astype(str).str.strip()
        lookup = direct.set_index(direct_col)["tm_player_id"]
        mapped_id = mapped["wyscout_id"].astype(str).str.strip().map(lookup)
        mapped["existing_tm_id"] = mapped["existing_tm_id"].fillna(mapped_id)

    exact_cols = [col for col in ["name", "club_name", "competition_name", "calendar"] if col in mapped.columns]
    ref_rename = {}
    if "player" in reference.columns:
        ref_rename["player"] = "name"
    if "team" in reference.columns:
        ref_rename["team"] = "club_name"
    exact_ref_cols = [ref_rename.get(col, col) for col in ["player", "team", "competition_name", "calendar"] if col in reference.columns]
    if set(exact_cols).issubset(set(exact_ref_cols)) and exact_cols:
        ref = reference.rename(columns=ref_rename)
        ref = ref[exact_cols + ["tm_player_id"]].dropna().drop_duplicates(subset=exact_cols, keep="last")
        probe = mapped[exact_cols].reset_index().merge(ref, on=exact_cols, how="left")
        mapped_id = probe.drop_duplicates(subset=["index"], keep="last").set_index("index")["tm_player_id"]
        mapped["existing_tm_id"] = mapped["existing_tm_id"].fillna(mapped.index.to_series().map(mapped_id))

    mapped["existing_tm_id"] = mapped["existing_tm_id"].apply(_clean_id)
    return mapped


def load_wyscout_players(engine, season_label: Optional[str], limit: Optional[int]) -> pd.DataFrame:
    season_labels = _split_labels(season_label)
    with engine.begin() as conn:
        player_columns = {
            row[0]
            for row in conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'players'")
            ).fetchall()
        }
        season_columns = {
            row[0]
            for row in conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'player_seasons'")
            ).fetchall()
        }
    player_age = "p.age" if "age" in player_columns else "NULL::DOUBLE PRECISION"
    season_age = "ps.age" if "age" in season_columns else "NULL::DOUBLE PRECISION"
    age_expr = f"COALESCE({season_age}, {player_age})"
    country_expr = "p.country" if "country" in player_columns else "NULL::TEXT"
    birth_expr = "p.birth_date" if "birth_date" in player_columns else "NULL::DATE"
    params = {}
    if season_labels:
        placeholders = []
        priority_parts = []
        for idx, label in enumerate(season_labels):
            key = f"season_label_{idx}"
            params[key] = label
            placeholders.append(f":{key}")
            priority_parts.append(f"WHEN ps.calendar = :{key} THEN {idx}")
        where_clause = f"WHERE ps.calendar IN ({', '.join(placeholders)})"
        season_priority = f"CASE {' '.join(priority_parts)} ELSE {len(season_labels)} END,"
    else:
        where_clause = ""
        season_priority = ""
    sql = """
    WITH ranked AS (
        SELECT
            p.id AS player_id,
            p.wyscout_id,
            p.name,
            {country_expr} AS country,
            {birth_expr} AS birth_date,
            {age_expr} AS age,
            ps.position,
            COALESCE(c.name, ps.team_in_selected_period) AS club_name,
            comp.name AS competition_name,
            ps.calendar,
            p.tm_id AS existing_tm_id,
            ROW_NUMBER() OVER (
                PARTITION BY p.id
                ORDER BY
                    {season_priority}
                    ps.calendar DESC NULLS LAST,
                    ps.minutes_played DESC NULLS LAST,
                    ps.id DESC NULLS LAST
            ) AS rn
        FROM players p
        LEFT JOIN player_seasons ps ON ps.player_id = p.id
        LEFT JOIN clubs c ON c.id = ps.club_id
        LEFT JOIN competitions comp ON comp.id = ps.competition_id
        {where_clause}
    )
    SELECT *
    FROM ranked
    WHERE rn = 1
    ORDER BY player_id
    """.format(
        country_expr=country_expr,
        birth_expr=birth_expr,
        age_expr=age_expr,
        season_priority=season_priority,
        where_clause=where_clause,
    )
    if limit:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)
    return pd.read_sql(text(sql), engine, params=params)


def write_review_file(matches: pd.DataFrame, run_id: str) -> None:
    output = os.getenv("TM_MATCH_REVIEW_OUTPUT", "").strip()
    if not output or matches.empty:
        return
    path = Path(output).expanduser()
    if output.endswith("/") or path.is_dir():
        path = path / f"transfermarkt_match_review_{run_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    review_cols = [
        "player_id",
        "wyscout_id",
        "wyscout_name",
        "wyscout_club",
        "tm_player_id",
        "tm_player_name",
        "tm_club_name",
        "confidence_score",
        "score_margin",
        "method",
        "status",
        "is_primary",
    ]
    matches[review_cols].to_csv(path, index=False)
    print(f"[TM] review file written: {path}")


def main() -> None:
    args = parse_args()
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set; cannot refresh Transfermarkt data.")

    tm_path = Path(args.tm_profiles).expanduser()
    if not tm_path.exists():
        raise FileNotFoundError(f"Transfermarkt profiles CSV not found: {tm_path}")

    snapshot_date = dt.date.fromisoformat(args.snapshot_date)
    print(f"[TM] run_id={args.run_id} snapshot_date={snapshot_date} profiles={tm_path}")

    engine = create_engine(db_url)
    db.ensure_schema(engine)

    tm_raw = pd.read_csv(tm_path, low_memory=False)
    if args.limit:
        tm_raw = tm_raw.head(args.limit)
    tm_profiles = prepare_transfermarkt_profiles(tm_raw)
    wyscout = load_wyscout_players(engine, args.season_label, args.limit)
    wyscout = apply_curated_player_map(wyscout, args.player_map)
    print(f"[TM] prepared profiles={len(tm_profiles)} wyscout_players={len(wyscout)} season={args.season_label or 'all'}")

    config = MatchConfig(
        min_candidate_score=args.min_candidate_score,
        auto_accept_score=args.auto_accept_score,
        auto_accept_margin=args.auto_accept_margin,
    )
    matches = build_match_candidates(wyscout, tm_profiles, config=config)
    accepted = int(((matches.get("status") == "accepted") & (matches.get("is_primary") == True)).sum()) if not matches.empty else 0
    review = int((matches.get("status") == "review").sum()) if not matches.empty else 0
    print(f"[TM] match candidates={len(matches)} accepted={accepted} review={review}")
    write_review_file(matches, args.run_id)

    if args.dry_run:
        print("[DRY-RUN] skipping DB writes.")
        return

    stats = db.upsert_transfermarkt_refresh(
        engine,
        tm_profiles=tm_profiles,
        matches=matches,
        snapshot_date=snapshot_date,
        source="transfermarkt-api",
        update_player_seasons=True,
    )
    db.insert_pipeline_run(
        engine,
        args.run_id,
        status="success",
        source_uri=str(tm_path),
        rows_processed=len(tm_profiles),
        message=json.dumps({"transfermarkt_refresh": stats}, sort_keys=True),
    )


if __name__ == "__main__":
    main()
