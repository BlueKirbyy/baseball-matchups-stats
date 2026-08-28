import unittest
from pathlib import Path
import tempfile
from datetime import datetime, timedelta, timezone

from analytics_store import connect, initialize
from backtest import calibration_rows, chronological_splits, evaluate, evaluate_hitter_recent_form
from server import (
    Handler, active_slate_date_and_scoreboard, confirmed_starting_lineup,
    hitter_context_metrics, hitter_power_metrics, hitter_spray_profile,
    local_game_time, odds_ttl_seconds, summarize_espn_odds,
)


class BacktestAndServerTests(unittest.TestCase):
    def test_spray_profile_infers_batting_side_for_lookahead_rosters(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "spray.db"
            initialize(db_path)
            rows = [
                (2026, 42, "L", "R", "FF", "LF", 5, 1, 0, 0, 430, 55),
                (2026, 42, "L", "R", "FF", "CF", 10, 3, 1, 0, 900, 120),
                (2026, 42, "L", "R", "FF", "RF", 35, 15, 5, 3, 3250, 600),
                (2026, 42, "R", "L", "FF", "LF", 4, 1, 0, 0, 350, 40),
            ]
            with connect(db_path) as db:
                db.executemany(
                    """INSERT INTO gameday_batter_spray VALUES
                       (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                profile = hitter_spray_profile(
                    db, 2026, 42, None, "R", [{"code": "FF", "usage": 100}],
                )
            self.assertEqual(profile["bat_side"], "L")
            self.assertGreater(profile["pull_rate"], .5)
            self.assertEqual(profile["exact_hand_batted_balls"], 50)

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
             "lineup_status": "confirmed", "pitcher_throws": "R", "model_version": "v5"},
            {"projection": 5, "actual_value": 4, "line": 5.5, "probability_over": .3,
             "expected_value_over": None, "over_price": -110, "confidence": "low", "month": "2026-04",
             "lineup_status": "confirmed", "pitcher_throws": "L", "model_version": "v5"},
        ]
        report = evaluate(rows)
        self.assertEqual(report["count"], 2)
        self.assertAlmostEqual(report["mae"], 1)
        self.assertIsNotNone(report["brier"])
        self.assertEqual(sum(bucket["count"] for bucket in report["calibration"]), 2)
        self.assertEqual(report["groups"]["model_version"]["v5"]["count"], 2)
        self.assertEqual(report["groups"]["projection_band"]["5.0-5.9"]["count"], 1)

    def test_recent_form_backtest_compares_frozen_with_and_without_scores(self):
        rows = [
            {"outcome": "hit", "actual": 1, "score": .70, "recent_form_adjustment": .05,
             "recent_form_score": .25, "confidence": "medium", "coverage_band": "35-49%",
             "recent_form_pa_band": "35+", "window_days": 14},
            {"outcome": "hit", "actual": 0, "score": -.10, "recent_form_adjustment": -.04,
             "recent_form_score": -.20, "confidence": "limited", "coverage_band": "25-34%",
             "recent_form_pa_band": "20-34", "window_days": 21},
        ]
        report = evaluate_hitter_recent_form(rows)
        self.assertEqual(report["count"], 2)
        self.assertIn("brier_with_form", report)
        self.assertEqual(set(report["by_window_days"]), {"14", "21"})
        self.assertEqual(report["cap_review"]["current_cap"], .10)

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

    def test_espn_odds_nested_moneyline_is_devigged(self):
        summary = summarize_espn_odds([{
            "overUnder": 9,
            "provider": {"name": "Test Book"},
            "homeTeamOdds": {"current": {"moneyLine": -150}},
            "awayTeamOdds": {"current": {"moneyLine": 130}},
        }], "Home Club", "Away Club")
        self.assertEqual(summary["favorite"]["team"], "Home Club")
        self.assertGreater(summary["favorite"]["probability"], .5)
        self.assertLess(summary["favorite"]["probability"], .6)

    def test_espn_favorite_flag_is_used_when_moneyline_is_missing(self):
        summary = summarize_espn_odds([{
            "overUnder": 8.5,
            "homeTeamOdds": {"favorite": True},
            "awayTeamOdds": {"favorite": False},
        }], "Home Club", "Away Club")
        self.assertEqual(summary["favorite"]["team"], "Home Club")
        self.assertEqual(summary["favorite"]["probability"], .55)

    def test_active_slate_stays_today_while_a_local_game_is_unstarted(self):
        calls = []
        def load(_path, query):
            calls.append(query["dates"])
            return {"events": [
                {"status": {"type": {"state": "post", "completed": True}}},
                {"status": {"type": {"state": "pre", "completed": False}}},
            ]}
        day, _scoreboard, lookahead = active_slate_date_and_scoreboard(
            datetime(2026, 4, 10, 18, tzinfo=timezone.utc), load,
        )
        self.assertEqual(day.isoformat(), "2026-04-10")
        self.assertFalse(lookahead)
        self.assertEqual(calls, ["20260410"])

    def test_active_slate_rolls_to_tomorrow_when_every_game_has_started(self):
        calls = []
        def load(_path, query):
            calls.append(query["dates"])
            if query["dates"] == "20260410":
                return {"events": [
                    {"status": {"type": {"state": "post", "completed": True}}},
                    {"status": {"type": {"state": "in", "completed": False, "detail": "Top 7th"}}},
                ]}
            return {"events": [{"status": {"type": {"state": "pre", "completed": False}}}]}
        day, scoreboard, lookahead = active_slate_date_and_scoreboard(
            datetime(2026, 4, 10, 18, tzinfo=timezone.utc), load,
        )
        self.assertEqual(day.isoformat(), "2026-04-11")
        self.assertTrue(lookahead)
        self.assertEqual(len(scoreboard["events"]), 1)
        self.assertEqual(calls, ["20260410", "20260411"])

    def test_active_slate_rolls_to_tomorrow_after_every_game_finishes(self):
        calls = []
        def load(_path, query):
            calls.append(query["dates"])
            if query["dates"] == "20260410":
                return {"events": [
                    {"status": {"type": {"state": "post", "completed": True, "detail": "Final"}}},
                    {"status": {"type": {"detail": "Postponed"}}},
                ]}
            return {"events": [{"status": {"type": {"state": "pre", "completed": False}}}]}
        day, scoreboard, lookahead = active_slate_date_and_scoreboard(
            datetime(2026, 4, 10, 18, tzinfo=timezone.utc), load,
        )
        self.assertEqual(day.isoformat(), "2026-04-11")
        self.assertTrue(lookahead)
        self.assertEqual(len(scoreboard["events"]), 1)
        self.assertEqual(calls, ["20260410", "20260411"])

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
