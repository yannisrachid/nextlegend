from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


COL_MAP = {
    "competition_name": "competition_name",
    "competition_slug": "competition_slug",
    "calendar": "calendar",
    "page_number": "page_number",
    "row_number": "row_number",
    "header_name": "player",
    "header_current_team_name": "team",
    "header_la_t_club_name": "team_in_selected_period",
    "header_po_ition": "position",
    "header_age": "age",
    "header_market_value": "market_value",
    "header_contract_expire": "contract_expires",
    "header_total_matche": "matches_played",
    "header_minute_on_field": "minutes_played",
    "header_goal": "goals",
    "header_xg_hot": "xg",
    "header_a_i_t": "assists",
    "header_xg_a_i_t": "xa",
    "header_duel_avg": "duels_per_90",
    "header_duel_won": "duels_won_percent",
    "header_birth_country_name": "birth_country",
    "header_pa_port_country_name": "passport_country",
    "header_foot": "foot",
    "header_height": "height",
    "header_weight": "weight",
    "header_on_loan": "on_loan",
    "header_ucce_ful_defen_ive_action_avg": "successful_def_actions_per_90",
    "header_defen_ive_duel_avg": "def_duels_per_90",
    "header_defen_ive_duel_won": "def_duels_won_percent",
    "header_aerial_duel_avg": "aerial_duels_per_90",
    "header_aerial_duel_won": "aerial_duels_won_percent",
    "header_tackle_avg": "sliding_tackles_per_90",
    "header_po_e_ion_adju_ted_tackle": "sliding_tackles_padj",
    "header_hot_block_avg": "blocked_shots_per_90",
    "header_interception_avg": "interceptions_per_90",
    "header_po_e_ion_adju_ted_interception": "interceptions_padj",
    "header_foul_avg": "fouls_per_90",
    "header_yellow_card": "yellow_cards",
    "header_yellow_card_avg": "yellow_cards_per_90",
    "header_red_card": "red_cards",
    "header_red_card_avg": "red_cards_per_90",
    "header_ucce_ful_attacking_action_avg": "successful_attacks_per_90",
    "header_goal_avg": "goals_per_90",
    "header_non_penalty_goal": "non_penalty_goals",
    "header_non_penalty_goal_avg": "non_penalty_goals_per_90",
    "header_xg_hot_avg": "xg_per_90",
    "header_head_goal": "headed_goals",
    "header_head_goal_avg": "headed_goals_per_90",
    "header_hot": "shots",
    "header_hot_avg": "shots_per_90",
    "header_hot_on_target_percent": "shots_on_target_percent",
    "header_goal_conver_ion_percent": "goal_conversion_rate",
    "header_a_i_t_avg": "assists_per_90",
    "header_cro_e_avg": "crosses_per_90",
    "header_accurate_cro_e_percent": "accurate_crosses_percent",
    "header_cro_from_left_avg": "left_flank_crosses_per_90",
    "header_ucce_ful_cro_from_left_percent": "accurate_left_flank_crosses_percent",
    "header_cro_from_right_avg": "right_flank_crosses_per_90",
    "header_ucce_ful_cro_from_right_percent": "accurate_right_flank_crosses_percent",
    "header_cro_to_goalie_box_avg": "goal_area_crosses_per_90",
    "header_dribble_avg": "dribbles_per_90",
    "header_ucce_ful_dribble_percent": "successful_dribbles_percent",
    "header_offen_ive_duel_avg": "offensive_duels_per_90",
    "header_offen_ive_duel_won": "marking_duels_percent",
    "header_touch_in_box_avg": "touches_in_penalty_area_per_90",
    "header_progre_ive_run_avg": "progressive_runs_per_90",
    "header_acceleration_avg": "accelerations_per_90",
    "header_received_pa_avg": "passes_received_per_90",
    "header_received_long_pa_avg": "long_passes_received_per_90",
    "header_foul_uffered_avg": "fouls_suffered_per_90",
    "header_pa_e_avg": "passes_per_90",
    "header_accurate_pa_e_percent": "accurate_passes_percent",
    "header_forward_pa_e_avg": "forward_passes_per_90",
    "header_ucce_ful_forward_pa_e_percent": "accurate_forward_passes_percent",
    "header_back_pa_e_avg": "backward_passes_per_90",
    "header_ucce_ful_back_pa_e_percent": "accurate_backward_passes_percent",
    "header_vertical_pa_e_avg": "lateral_passes_per_90",
    "header_ucce_ful_vertical_pa_e_percent": "accurate_lateral_passes_percent",
    "header_hort_medium_pa_avg": "short_medium_passes_per_90",
    "header_accurate_hort_medium_pa_percent": "accurate_short_medium_passes_percent",
    "header_long_pa_e_avg": "long_passes_per_90",
    "header_ucce_ful_long_pa_e_percent": "accurate_long_passes_percent",
    "header_average_pa_length": "avg_pass_length_m",
    "header_average_long_pa_length": "avg_long_pass_length_m",
    "header_xg_a_i_t_avg": "xa_per_90",
    "header_hot_a_i_t_avg": "shot_assists_per_90",
    "header_pre_a_i_t_avg": "second_assists_per_90",
    "header_pre_pre_a_i_t_avg": "third_assists_per_90",
    "header_mart_pa_e_avg": "smart_passes_per_90",
    "header_accurate_mart_pa_e_percent": "accurate_smart_passes_percent",
    "header_key_pa_e_avg": "key_passes_per_90",
    "header_pa_e_to_final_third_avg": "passes_to_final_third_per_90",
    "header_accurate_pa_e_to_final_third_percent": "accurate_passes_to_final_third_percent",
    "header_pa_to_penalty_area_avg": "passes_to_penalty_area_per_90",
    "header_accurate_pa_to_penalty_area_percent": "accurate_passes_to_penalty_area_percent",
    "header_through_pa_e_avg": "through_passes_per_90",
    "header_ucce_ful_through_pa_e_percent": "accurate_through_passes_percent",
    "header_deep_completed_pa_avg": "deep_completions_per_90",
    "header_deep_completed_cro_avg": "deep_crosses_per_90",
    "header_progre_ive_pa_avg": "progressive_passes_per_90",
    "header_ucce_ful_progre_ive_pa_percent": "accurate_progressive_passes_percent",
    "header_conceded_goal": "goals_conceded",
    "header_conceded_goal_avg": "goals_conceded_per_90",
    "header_hot_again_t": "shots_against",
    "header_hot_again_t_avg": "shots_against_per_90",
    "header_clean_heet": "clean_sheets",
    "header_ave_percent": "save_percent",
    "header_xg_ave": "xg_against",
    "header_xg_ave_avg": "xg_against_per_90",
    "header_prevented_goal": "goals_prevented",
    "header_prevented_goal_avg": "goals_prevented_per_90",
    "header_back_pa_to_gk_avg": "back_passes_to_gk_per_90",
    "header_goalkeeper_exit_avg": "exits_per_90",
    "header_gk_aerial_duel_avg": "aerial_duels_gk_per_90",
    "header_free_kick_taken_avg": "free_kicks_per_90",
    "header_direct_free_kick_taken_avg": "direct_free_kicks_per_90",
    "header_direct_free_kick_on_target_percent": "direct_fk_on_target_percent",
    "header_corner_taken_avg": "corners_per_90",
    "header_penaltie_taken": "penalties_taken",
    "header_penaltie_conver_ion_percent": "penalty_conversion_percent",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the final column-renaming step."""
    parser = argparse.ArgumentParser(description="Renomme les colonnes du CSV Wyscout.")
    parser.add_argument(
        "--input",
        default="data/wyscout_players.csv",
        help="CSV source brut. Defaut: data/wyscout_players.csv",
    )
    parser.add_argument(
        "--output",
        default="data/wyscout_players_finale.csv",
        help="CSV de sortie avec colonnes renommees. Defaut: data/wyscout_players_finale.csv",
    )
    return parser.parse_args(argv)


def resolve_path(raw: str) -> Path:
    """Resolve a file path relative to the repository root when needed."""
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def main(argv: list[str] | None = None) -> int:
    """Rename the scraped CSV columns and write the final StatsBomb-style output."""
    args = parse_args(argv)
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        source_fieldnames = list(reader.fieldnames or [])
        if not source_fieldnames:
            raise SystemExit(f"CSV source vide ou sans entete: {input_path}")
        output_fieldnames = [COL_MAP.get(field, field) for field in source_fieldnames]
        with output_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=output_fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow({COL_MAP.get(field, field): value for field, value in row.items()})
    print(f"Colonnes renommees -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
