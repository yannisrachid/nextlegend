import json
import unittest

from scripts.opta_bundle import parse_last_updated, parse_rankings


def ranking(name: str) -> dict:
    return {
        "contestantId": name,
        "contestantName": name,
        "currentGlobalRank": 1,
        "currentRating": 90,
    }


class OptaBundleTest(unittest.TestCase):
    def test_rankings_follow_dataset_shape_when_minified_names_change(self):
        men = [ranking(f"Men {i}") for i in range(5)]
        women = [ranking(f"Women {i}") for i in range(2)]
        bundle = (
            f"const x9=JSON.parse(`{json.dumps(women)}`),"
            f"z4=JSON.parse(`{json.dumps(men)}`);"
        )
        self.assertEqual(parse_rankings(bundle, "mens"), men)
        self.assertEqual(parse_rankings(bundle, "womens"), women)

    def test_ambiguous_rankings_fail_closed(self):
        bundle = f"const z4=JSON.parse(`{json.dumps([ranking('Only one')])}`);"
        with self.assertRaises(RuntimeError):
            parse_rankings(bundle, "mens")

    def test_updated_date_follows_renamed_variable(self):
        self.assertEqual(
            parse_last_updated('const hb="Oct 14, 2026",Qt={lastUpdated:hb};'),
            ("Oct 14, 2026", "2026-10-14"),
        )


if __name__ == "__main__":
    unittest.main()
