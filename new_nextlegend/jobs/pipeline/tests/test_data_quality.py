from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pandas as pd

from pipeline import data_quality


def base_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player": ["Player A", "Player B"],
            "competition_name": ["France. Ligue 1", "France. Ligue 1"],
            "calendar": ["2026/2027", "2026/2027"],
            "team_in_selected_period": ["Club A", "Club A"],
            "position": ["CF", "CF"],
            "minutes_played": [300, 280],
            "matches_played": [4, 4],
        }
    )


def base_fact() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wyscout_id": ["1", "2"],
            "competition_name": ["France. Ligue 1", "France. Ligue 1"],
            "calendar": ["2026/2027", "2026/2027"],
            "team_in_selected_period": ["Club A", "Club A"],
            "position": ["CF", "CF"],
            "minutes_played": [300, 280],
            "matches_played": [4, 4],
            "global_score_adjusted": [78.0, 71.0],
            "assigned_role": ["Centre Forwards", "Centre Forwards"],
        }
    )


def base_similarity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_a_id": ["1", "2"],
            "player_b_id": ["2", "1"],
            "competition_a": ["France. Ligue 1", "France. Ligue 1"],
            "competition_b": ["France. Ligue 1", "France. Ligue 1"],
            "calendar_a": ["2026/2027", "2026/2027"],
            "calendar_b": ["2026/2027", "2026/2027"],
            "team_a": ["Club A", "Club A"],
            "team_b": ["Club A", "Club A"],
            "profile": ["Centre Forwards", "Centre Forwards"],
            "similarity": [0.9, 0.9],
        }
    )


def base_artifacts() -> dict[str, pd.DataFrame]:
    return {
        "player_seasons": base_fact(),
        "player_metrics": pd.DataFrame({"wyscout_id": ["1", "2"], "minutes_played": [300, 280]}),
        "role_scores": pd.DataFrame({"wyscout_id": ["1", "2"], "profile": ["Centre Forwards", "Centre Forwards"]}),
        "player_similarity": base_similarity(),
    }


class DataQualityTests(unittest.TestCase):
    def test_valid_artifacts_pass(self) -> None:
        with patch.dict(os.environ, {"DATA_FRESHNESS_EXPECT_CALENDARS": "2026/2027", "DATA_QUALITY_SIM_TOPK": "10"}, clear=False):
            report = data_quality.validate_artifacts(raw=base_raw(), artifacts=base_artifacts(), strict=True)
        self.assertEqual(report["failures"], [])

    def test_duplicate_fact_key_fails(self) -> None:
        artifacts = base_artifacts()
        artifacts["player_seasons"] = pd.concat([base_fact(), base_fact().iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            data_quality.validate_artifacts(raw=base_raw(), artifacts=artifacts, strict=True)

    def test_score_out_of_bounds_fails(self) -> None:
        artifacts = base_artifacts()
        artifacts["player_seasons"].loc[0, "global_score_adjusted"] = 101.0
        with self.assertRaises(ValueError):
            data_quality.validate_artifacts(raw=base_raw(), artifacts=artifacts, strict=True)

    def test_similarity_topk_fails(self) -> None:
        artifacts = base_artifacts()
        rows = []
        for idx in range(11):
            rows.append(
                {
                    "player_a_id": "1",
                    "player_b_id": str(idx + 10),
                    "competition_a": "France. Ligue 1",
                    "competition_b": "France. Ligue 1",
                    "calendar_a": "2026/2027",
                    "calendar_b": "2026/2027",
                    "team_a": "Club A",
                    "team_b": f"Club {idx}",
                    "profile": "Centre Forwards",
                    "similarity": 0.8,
                }
            )
        artifacts["player_similarity"] = pd.DataFrame(rows)
        with patch.dict(os.environ, {"DATA_QUALITY_SIM_TOPK": "10"}, clear=False):
            with self.assertRaises(ValueError):
                data_quality.validate_artifacts(raw=base_raw(), artifacts=artifacts, strict=True)

    def test_expected_calendar_fails_when_missing(self) -> None:
        with patch.dict(os.environ, {"DATA_FRESHNESS_EXPECT_CALENDARS": "2027/2028"}, clear=False):
            with self.assertRaises(ValueError):
                data_quality.validate_artifacts(raw=base_raw(), artifacts=base_artifacts(), strict=True)


if __name__ == "__main__":
    unittest.main()
