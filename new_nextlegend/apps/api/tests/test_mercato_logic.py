import unittest

from mercato_logic import calculate_calibrated_level


LEAGUE_META = {
    "England. Premier League": {"difficulty": 9.099},
    "France. Ligue 2": {"difficulty": 6.95},
    "Netherlands. Eredivisie": {"difficulty": 7.516},
    "Bulgaria. Second League": {"difficulty": 4.9},
}

CONFIG = {
    "bands": [
        {"label": "Premier League", "difficulty_min": 8.9, "coefficient": 1.0, "cap": 98},
        {"label": "Championship / Eredivisie / Liga Portugal", "difficulty_min": 7.4, "coefficient": 0.78, "cap": 86},
        {"label": "Ligue 2 / D2 top pays", "difficulty_min": 6.8, "coefficient": 0.68, "cap": 80},
        {"label": "D2 Bulgarie / championnat tres faible", "difficulty_min": 0, "coefficient": 0.45, "cap": 70},
    ],
    "exact_overrides": [
        {
            "competition": "Bulgaria. Second League",
            "label": "D2 Bulgarie / championnat tres faible",
            "coefficient": 0.45,
            "cap": 70,
        }
    ],
}


class MercatoLogicTest(unittest.TestCase):
    def test_premier_league_keeps_elite_score(self):
        result = calculate_calibrated_level(
            {
                "raw_player_level": 95,
                "competition_name": "England. Premier League",
                "minutes_played": 3000,
            },
            LEAGUE_META,
            CONFIG,
        )
        self.assertAlmostEqual(result["league_coefficient"], 1.0)
        self.assertEqual(result["league_cap"], 98)
        self.assertGreaterEqual(result["calibrated_player_level"], 95)

    def test_bulgarian_second_league_is_capped(self):
        result = calculate_calibrated_level(
            {
                "raw_player_level": 95,
                "competition_name": "Bulgaria. Second League",
                "minutes_played": 3000,
            },
            LEAGUE_META,
            CONFIG,
        )
        self.assertEqual(result["league_coefficient"], 0.45)
        self.assertEqual(result["league_cap"], 70)
        self.assertLessEqual(result["calibrated_player_level"], 70)

    def test_ligue_2_is_penalized(self):
        result = calculate_calibrated_level(
            {
                "raw_player_level": 82,
                "competition_name": "France. Ligue 2",
                "minutes_played": 1200,
            },
            LEAGUE_META,
            CONFIG,
        )
        self.assertEqual(result["league_coefficient"], 0.68)
        self.assertLess(result["calibrated_player_level"], 70)

    def test_eredivisie_gets_mid_high_band(self):
        result = calculate_calibrated_level(
            {
                "raw_player_level": 88,
                "competition_name": "Netherlands. Eredivisie",
                "minutes_played": 1800,
            },
            LEAGUE_META,
            CONFIG,
        )
        self.assertEqual(result["league_coefficient"], 0.78)
        self.assertGreaterEqual(result["calibrated_player_level"], 69)
        self.assertLessEqual(result["calibrated_player_level"], 86)

    def test_low_first_divisions_are_strongly_capped(self):
        config = {
            "bands": CONFIG["bands"],
            "exact_overrides": [
                {
                    "competition": "Egypt. Premier League",
                    "label": "Medium/low first division - Egypt. Premier League",
                    "coefficient": 0.58,
                    "cap": 74,
                },
                {
                    "competition": "Moldova. Super Liga",
                    "label": "Low first division - Moldova. Super Liga",
                    "coefficient": 0.48,
                    "cap": 70,
                },
            ],
        }
        for competition, cap in (
            ("Egypt. Premier League", 74),
            ("Moldova. Super Liga", 70),
        ):
            result = calculate_calibrated_level(
                {
                    "raw_player_level": 95,
                    "competition_name": competition,
                    "minutes_played": 3000,
                },
                {},
                config,
            )
            self.assertEqual(result["league_cap"], cap)
            self.assertLessEqual(result["calibrated_player_level"], cap)


if __name__ == "__main__":
    unittest.main()
