import json
import sys
import unittest
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from scripts.import_scoutyourlegend_reports import (
    extract_eyeball_player_id,
    normalize_matches,
    normalize_star_rating,
    to_int,
)


class ScoutYourLegendImportTest(unittest.TestCase):
    def test_extract_eyeball_player_id_from_portal_url(self):
        self.assertEqual(extract_eyeball_player_id("https://portal.eyeball.club/player/14180"), "14180")
        self.assertEqual(extract_eyeball_player_id("https://portal.eyeball.club/search?playerId=244037"), "244037")
        self.assertEqual(extract_eyeball_player_id("131648"), "131648")
        self.assertIsNone(extract_eyeball_player_id(""))

    def test_normalize_matches_keeps_structured_observations(self):
        raw = json.dumps(
            [
                {
                    "team_a": "Nantes",
                    "score_a": "2",
                    "score_b": "1",
                    "team_b": "Rennes",
                    "competition": "U19 National",
                    "match_date": "2026-08-30",
                    "player_rating": "7.5",
                }
            ]
        )

        self.assertEqual(
            normalize_matches(raw),
            [
                {
                    "team_a": "Nantes",
                    "score_a": 2.0,
                    "score_b": 1.0,
                    "team_b": "Rennes",
                    "competition": "U19 National",
                    "match_date": "2026-08-30",
                    "player_rating": 7.5,
                }
            ],
        )

    def test_normalize_matches_supports_legacy_pipe_text(self):
        self.assertEqual(
            normalize_matches("Nantes - Rennes | Lorient - Nantes"),
            [
                {
                    "team_a": "Nantes - Rennes",
                    "score_a": None,
                    "score_b": None,
                    "team_b": "",
                    "competition": "",
                    "match_date": "",
                    "player_rating": None,
                },
                {
                    "team_a": "Lorient - Nantes",
                    "score_a": None,
                    "score_b": None,
                    "team_b": "",
                    "competition": "",
                    "match_date": "",
                    "player_rating": None,
                },
            ],
        )

    def test_rating_and_year_normalization(self):
        self.assertEqual(normalize_star_rating({"star_rating": "4"}), 4)
        self.assertEqual(normalize_star_rating({"overall_rating": "8.2"}), 4)
        self.assertEqual(normalize_star_rating({"technical_rating": "8", "physical_rating": "6"}), 4)
        self.assertIsNone(to_int(""))
        self.assertEqual(to_int("2008.0"), 2008)


if __name__ == "__main__":
    unittest.main()
