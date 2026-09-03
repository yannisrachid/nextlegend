from __future__ import annotations

import unittest

import pandas as pd

from pipeline.transfermarkt_matching import (
    build_match_candidates,
    encoded_birth_date,
    infer_birth_year_candidates,
    parse_birth_year,
    parse_market_value,
    prepare_transfermarkt_profiles,
)


class TransfermarktMatchingTests(unittest.TestCase):
    def test_birth_date_encoding_matches_players_matcher_pattern(self) -> None:
        self.assertEqual(encoded_birth_date("1997-06-15"), "15F97")

    def test_market_value_parser_handles_transfermarkt_labels(self) -> None:
        self.assertEqual(parse_market_value("€8.00m"), 8_000_000)
        self.assertEqual(parse_market_value("€250k"), 250_000)
        self.assertEqual(parse_market_value(1250000), 1_250_000)

    def test_birth_year_parser_handles_wyscout_age_label(self) -> None:
        self.assertEqual(parse_birth_year("'97 (28)"), 1997)
        self.assertEqual(infer_birth_year_candidates(28, "2026"), {1997, 1998, 1999})

    def test_prepare_transfermarkt_profiles_normalizes_expected_columns(self) -> None:
        raw = pd.DataFrame(
            {
                "player_id": ["1013991"],
                "player_name": ["Aaro Soiniemi"],
                "profile_url": ["https://www.transfermarkt.com/aaro/profil/spieler/1013991"],
                "profile_description": ["Aaro Soiniemi, 20, from Finland * 08/08/2005 in ,"],
                "club_id": ["1008.0"],
                "club_name": ["HJK Helsinki"],
                "market_value": ["€25k"],
            }
        )

        prepared = prepare_transfermarkt_profiles(raw)

        self.assertEqual(prepared.loc[0, "tm_player_id"], "1013991")
        self.assertEqual(prepared.loc[0, "tm_birth_date"], "2005-08-08")
        self.assertEqual(prepared.loc[0, "tm_market_value_eur"], 25_000)

    def test_bidirectional_matching_keeps_best_birth_date_candidate(self) -> None:
        wyscout = pd.DataFrame(
            {
                "player_id": [1],
                "wyscout_id": ["w1"],
                "name": ["Alex Martin"],
                "birth_date": ["2003-02-01"],
                "age": [23],
                "country": ["France"],
                "club_name": ["Paris FC"],
                "position": ["CF"],
            }
        )
        tm = pd.DataFrame(
            {
                "player_id": ["tm-old", "tm-good"],
                "player_name": ["Alex Martin", "Alex Martin"],
                "date_of_birth": ["1993-02-01", "2003-02-01"],
                "club_name": ["Paris FC", "Paris FC"],
                "nationality": ["France", "France"],
                "position": ["Centre-Forward", "Centre-Forward"],
                "market_value": ["€1m", "€2m"],
            }
        )

        matches = build_match_candidates(wyscout, tm)
        accepted = matches[matches["status"] == "accepted"]

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted.iloc[0]["tm_player_id"], "tm-good")

    def test_abbreviated_wyscout_name_can_match_when_club_is_clear(self) -> None:
        wyscout = pd.DataFrame(
            {
                "player_id": [1],
                "wyscout_id": ["w1"],
                "name": ["B. Sánchez"],
                "birth_date": [None],
                "age": [None],
                "country": [None],
                "club_name": ["Troyes"],
                "position": ["RW"],
            }
        )
        tm = pd.DataFrame(
            {
                "player_id": ["655637"],
                "player_name": ["Brayan Sánchez"],
                "club_name": ["Troyes"],
                "market_value": ["€500k"],
            }
        )

        matches = build_match_candidates(wyscout, tm)

        self.assertEqual(matches.iloc[0]["status"], "accepted")
        self.assertEqual(matches.iloc[0]["tm_player_id"], "655637")
        self.assertEqual(matches.iloc[0]["evidence"]["name_pattern"], "abbreviated")

    def test_birth_year_candidate_supports_abbreviated_name_matching(self) -> None:
        wyscout = pd.DataFrame(
            {
                "player_id": [1],
                "wyscout_id": ["w1"],
                "name": ["A. Martin"],
                "birth_year": [2003],
                "age": [23],
                "country": ["France"],
                "club_name": ["Paris FC"],
                "position": ["CF"],
            }
        )
        tm = pd.DataFrame(
            {
                "player_id": ["tm-old", "tm-good"],
                "player_name": ["Alex Martin", "Alex Martin"],
                "date_of_birth": ["1993-02-01", "2003-09-12"],
                "club_name": ["Paris FC", "Paris FC"],
                "nationality": ["France", "France"],
                "position": ["Centre-Forward", "Centre-Forward"],
                "market_value": ["€1m", "€2m"],
            }
        )

        matches = build_match_candidates(wyscout, tm)
        accepted = matches[matches["status"] == "accepted"]

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted.iloc[0]["tm_player_id"], "tm-good")


if __name__ == "__main__":
    unittest.main()
