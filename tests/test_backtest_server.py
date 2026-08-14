import unittest
from datetime import datetime, timedelta, timezone

from backtest import calibration_rows, chronological_splits, evaluate
from server import Handler, confirmed_starting_lineup, hitter_context_metrics, hitter_power_metrics, local_game_time, odds_ttl_seconds


class BacktestAndServerTests(unittest.TestCase):
    def test_local_server_disables_stale_dashboard_caching(self):
        self.assertTrue(any("no-store" in value for value in Handler.end_headers.__code__.co_consts if isinstance(value, str)))

    def test_dashboard_has_a_unique_cache_busting_route(self):
        constants = Handler.do_GET.__code__.co_consts
        self.assertTrue(any(isinstance(value, tuple) and "/dashboard" in value for value in constants))
        self.assertIn("/index.html", constants)

    def test_walk_forward_split(self):
        rows = [{"as_of": f"2026-04-{day:02d}", "prediction_id": day} for day in range(1, 7)]
        splits = list(chronological_splits(rows, minimum_training=3))
        self.assertEqual(len(splits), 3)
        self.assertTrue(all(max(row["as_of"] for row in train) < test["as_of"] for train, test in splits))

    def test_calibration_and_metrics(self):
        rows = [
            {"projection": 6, "actual_value": 7, "line": 5.5, "probability_over": .7,
             "expected_value_over": .1, "over_price": -110, "confidence": "low", "month": "2026-04",
             "lineup_status": "confirmed", "pitcher_throws": "R"},
            {"projection": 5, "actual_value": 4, "line": 5.5, "probability_over": .3,
             "expected_value_over": None, "over_price": -110, "confidence": "low", "month": "2026-04",
             "lineup_status": "confirmed", "pitcher_throws": "L"},
        ]
        report = evaluate(rows)
        self.assertEqual(report["count"], 2)
        self.assertAlmostEqual(report["mae"], 1)
        self.assertIsNotNone(report["brier"])
        self.assertEqual(sum(bucket["count"] for bucket in report["calibration"]), 2)

    def test_confirmed_lineup_requires_nine_unique_players(self):
        feed = {"liveData": {"boxscore": {"teams": {"away": {
            "battingOrder": list(range(1, 10)),
            "players": {f"ID{i}": {"person": {"id": i, "fullName": f"P{i}"}} for i in range(1, 10)},
        }}}}}
        lineup = confirmed_starting_lineup(feed, "away")
        self.assertEqual(len(lineup), 9)
        feed["liveData"]["boxscore"]["teams"]["away"]["battingOrder"] = list(range(1, 9))
        self.assertIsNone(confirmed_starting_lineup(feed, "away"))

    def test_odds_freshness_and_timezone_helpers(self):
        now = datetime.now(timezone.utc)
        self.assertEqual(odds_ttl_seconds([{"start_time": (now + timedelta(minutes=10)).isoformat()}]), 60)
        self.assertNotEqual(local_game_time("2026-04-10T23:00:00Z"), "Time TBD")

    def test_hitter_power_metrics_are_derived_from_final_pitch_counts(self):
        metrics = hitter_power_metrics({"at_bats": 20, "hits": 6, "doubles": 2, "triples": 1, "hr": 1, "total_bases": 14})
        self.assertEqual(metrics, {"slg": .7, "iso": .4, "xbh": 4, "doubles": 2, "triples": 1, "hr": 1})
        self.assertIsNone(hitter_power_metrics({"at_bats": 20, "hits": 6, "total_bases": 0}))

    def test_hitter_context_weights_count_and_zone_to_the_starter(self):
        base = {"at_bats": 20, "hits": 5, "doubles": 2, "triples": 0, "hr": 0, "total_bases": 8}
        splits = [{
            "count_bucket": "pitcher_ahead", "zone": 4,
            "pa": 10, "at_bats": 10, "hits": 5, "total_bases": 9,
        }]
        starter_mix = [{"count_bucket": "pitcher_ahead", "zone": 4, "pitches": 80}]
        result = hitter_context_metrics(base, splits, starter_mix)
        self.assertGreater(result["adjusted_avg"], .25)
        self.assertGreater(result["adjusted_slg"], .4)
        self.assertGreater(result["coverage"], 0)


if __name__ == "__main__":
    unittest.main()
