import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from analytics_store import connect, initialize
from prediction_store import (
    add_market_snapshot, add_workload_override, ensure_pregame_prediction, latest_markets,
    latest_bullpen_snapshot, latest_workload_override, record_bullpen_snapshot,
    prediction_tracking_summary, save_prediction, settle_game_predictions,
    settle_prediction,
)
from collections import defaultdict

from sync_matchup_data import batter_context_store, batter_discipline_record, count_bucket, completed_game_observations, game_log_pks_from_payload, outcome_counts, process_feed


FIXTURE = Path(__file__).parent / "fixtures" / "completed_game.json"


class StorageAndSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "test.db"

    def tearDown(self):
        self.temp.cleanup()

    def test_non_destructive_migration(self):
        raw = sqlite3.connect(self.db_path)
        raw.execute("CREATE TABLE legacy(value TEXT)")
        raw.execute("INSERT INTO legacy VALUES ('preserve me')")
        raw.commit(); raw.close()
        initialize(self.db_path)
        initialize(self.db_path)
        with connect(self.db_path) as db:
            self.assertEqual(db.execute("SELECT value FROM legacy").fetchone()[0], "preserve me")
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 8)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0], 8)
            columns = {row[1] for row in db.execute("PRAGMA table_info(gameday_batter_pitch_velocity)")}
            self.assertTrue({"doubles", "triples", "total_bases"}.issubset(columns))
            context_columns = {row[1] for row in db.execute("PRAGMA table_info(gameday_batter_pitch_context)")}
            self.assertTrue({"pitcher_throws", "count_bucket", "zone", "total_bases"}.issubset(context_columns))
            self.assertTrue(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gameday_pitcher_arsenal_context'").fetchone())
            self.assertTrue(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='pitcher_game_results'").fetchone())
            self.assertTrue(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gameday_batter_discipline'").fetchone())
            self.assertTrue(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='workload_overrides'").fetchone())
            self.assertTrue(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='bullpen_snapshots'").fetchone())

    def test_bullpen_snapshots_are_append_only_and_read_as_one_unit(self):
        initialize(self.db_path)
        reliever = {
            "player_id": 300, "player_name": "Reliever", "throws": "R",
            "role": "Short relief", "readiness_score": 82,
            "readiness_status": "fresh", "mix_weight": .8,
            "pitches_today": 0, "pitches_yesterday": 12,
            "pitches_two_days_ago": 0, "three_day_pitches": 12,
            "consecutive_days": 1, "days_rest": 1,
            "recent_appearances": 5, "recent_starts": 0,
            "arsenal_available": True,
        }
        ids = record_bullpen_snapshot(
            42, 10, "Test Team", "2099-04-10T23:00:00+00:00",
            "2099-04-10T16:00:00+00:00", [reliever], self.db_path,
        )
        self.assertEqual(len(ids), 1)
        rows = latest_bullpen_snapshot(42, 10, "2099-04-10T17:00:00+00:00", self.db_path)
        self.assertEqual(rows[0]["player_name"], "Reliever")
        with connect(self.db_path) as db, self.assertRaises(sqlite3.IntegrityError):
            db.execute("UPDATE bullpen_snapshots SET readiness_score=10 WHERE bullpen_snapshot_id=?", (ids[0],))

    def test_pitch_outcomes_include_total_bases(self):
        counts = outcome_counts(["single", "double", "triple", "home_run", "walk", "field_out"])
        self.assertEqual(counts[6:], (1, 1, 10))

    def test_count_bucket_uses_the_pre_pitch_count(self):
        self.assertEqual(count_bucket({"count": {"balls": 0, "strikes": 2}}), "pitcher_ahead")
        self.assertEqual(count_bucket({"count": {"balls": 3, "strikes": 1}}), "hitter_ahead")
        self.assertEqual(count_bucket({"count": {"balls": 1, "strikes": 1}}), "even")
        self.assertEqual(count_bucket({"count": {"balls": "bad", "strikes": 1}}), "unknown")

    def test_feed_context_events_are_nested_through_zone(self):
        feed = {"gameData": {"datetime": {"officialDate": "2026-04-10"}}, "liveData": {"plays": {"allPlays": [{
            "matchup": {"pitcher": {"id": 200}, "batter": {"id": 100}, "pitchHand": {"code": "R"}},
            "result": {"eventType": "double"},
            "playEvents": [{
                "isPitch": True,
                "details": {"type": {"code": "SL", "description": "Slider"}, "description": "In play"},
                "count": {"balls": 1, "strikes": 1},
                "pitchData": {"startSpeed": 86.2, "coordinates": {"pX": 0.0, "pZ": 2.6}},
                "hitData": {"launchSpeed": 101.0, "launchAngle": 20.0},
            }],
        }]}}}
        pitcher_data = defaultdict(lambda: defaultdict(lambda: {"name": "Pitch", "speeds": [], "zones": [0] * 9}))
        pitcher_context = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        batter_events = defaultdict(list)
        pitch_events = defaultdict(lambda: defaultdict(list))
        pitch_zones = defaultdict(lambda: defaultdict(lambda: [0] * 9))
        velocity_events = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        context_events = batter_context_store()
        quality = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        workloads = defaultdict(list)
        discipline = defaultdict(batter_discipline_record)
        process_feed(feed, {200}, {100}, pitcher_data, pitcher_context, batter_events, pitch_events,
                     pitch_zones, velocity_events, context_events, quality, workloads, discipline)
        self.assertEqual(context_events[100]["SL"][86]["R"]["even"][4], ["double"])
        self.assertEqual(discipline[100]["plate_appearances"], 1)
        self.assertEqual(discipline[100]["pitches_seen"], 1)
        self.assertEqual(discipline[100]["total_bases"], 2)

    def test_as_of_game_log_cutoff_keeps_traded_history(self):
        payload = {"stats": [{"splits": [
            {"date": "2026-04-01", "game": {"gamePk": 1}, "team": {"id": 10}},
            {"date": "2026-05-01", "game": {"gamePk": 2}, "team": {"id": 20}},
            {"date": "2026-06-01", "game": {"gamePk": 3}, "team": {"id": 20}},
        ]}]}
        self.assertEqual(game_log_pks_from_payload(payload, "2026-06-01"), {1, 2})

    def test_starter_relief_and_handedness_fixture(self):
        rows = completed_game_observations(json.loads(FIXTURE.read_text()), "2026-04-11T00:00:00+00:00")
        by_player = {row["player_id"]: row for row in rows}
        self.assertEqual(by_player[200]["is_start"], 1)
        self.assertEqual(by_player[201]["is_start"], 0)
        self.assertEqual(by_player[200]["throws"], "R")
        self.assertEqual(by_player[100]["stands"], "L")
        self.assertEqual(by_player[400]["total_bases"], 2)

    def test_prediction_and_markets_are_append_only(self):
        initialize(self.db_path)
        market = {
            "captured_at": "2026-04-10T15:00:00+00:00", "provider": "Book",
            "platform_type": "sportsbook", "game_pk": 42, "player_id": 200,
            "player_name": "Home Starter", "prop_type": "pitcher_strikeouts",
            "line": 5.5, "over_price": -110, "under_price": -110, "source": "test",
        }
        first = add_market_snapshot(market, self.db_path)
        second = add_market_snapshot(market, self.db_path)
        self.assertNotEqual(first, second)
        with connect(self.db_path) as db, self.assertRaises(sqlite3.IntegrityError):
            db.execute("UPDATE market_snapshots SET line=6.5 WHERE market_snapshot_id=?", (first,))
        name_only = dict(market, player_id=None, captured_at="2026-04-10T16:00:00+00:00", line=6.5)
        name_only_id = add_market_snapshot(name_only, self.db_path)
        matches = latest_markets(42, player_id=200, player_name="Home Starter", as_of="2026-04-10T17:00:00+00:00", db_path=self.db_path)
        self.assertEqual({row["market_snapshot_id"] for row in matches}, {second})
        other_provider_id = add_market_snapshot(dict(name_only, provider="Other Book"), self.db_path)
        matches = latest_markets(42, player_id=200, player_name="Home Starter", as_of="2026-04-10T17:00:00+00:00", db_path=self.db_path)
        self.assertEqual({row["market_snapshot_id"] for row in matches}, {second, other_provider_id})
        prediction = {
            "as_of": "2026-04-10T16:00:00+00:00", "scheduled_start": "2026-04-10T23:00:00+00:00",
            "player_id": 200, "player_name": "Home Starter", "prop_type": "pitcher_strikeouts",
            "model_version": "v", "feature_version": "f", "projection": 6.0, "median": 6,
            "interval_low": 2, "interval_high": 10, "probability_over": .55,
            "probability_under": .45, "fair_over_price": -122, "fair_under_price": 122,
            "confidence": "low", "arsenal_coverage": .5, "effective_sample_size": 30,
            "lineup_confirmed": True, "decision": "RESEARCH_ONLY_UNVALIDATED", "factors": [],
        }
        prediction_id = save_prediction(42, prediction, {"market_snapshot_id": first}, db_path=self.db_path)
        settle_prediction(prediction_id, 7, db_path=self.db_path)
        with self.assertRaises(sqlite3.IntegrityError):
            settle_prediction(prediction_id, 8, db_path=self.db_path)

    def test_manual_pitch_cap_is_append_only_and_latest_wins(self):
        initialize(self.db_path)
        base = {
            "captured_at": "2026-04-10T15:00:00+00:00", "game_pk": 42,
            "player_id": 200, "player_name": "Home Starter", "pitch_limit": 80,
            "source": "manager report", "note": "returning from IL",
        }
        first = add_workload_override(base, self.db_path)
        second = add_workload_override(dict(base, captured_at="2026-04-10T16:00:00+00:00", pitch_limit=75), self.db_path)
        self.assertNotEqual(first, second)
        self.assertEqual(latest_workload_override(42, 200, "2026-04-10T17:00:00+00:00", self.db_path)["pitch_limit"], 75)
        with connect(self.db_path) as db, self.assertRaises(sqlite3.IntegrityError):
            db.execute("UPDATE workload_overrides SET pitch_limit=90 WHERE workload_override_id=?", (second,))

    def test_post_start_prediction_is_rejected(self):
        initialize(self.db_path)
        prediction = {
            "as_of": "2026-04-11T00:00:00+00:00", "scheduled_start": "2026-04-10T23:00:00+00:00",
            "player_id": 200, "player_name": "Home Starter", "prop_type": "pitcher_strikeouts",
            "model_version": "v", "feature_version": "f", "projection": 6.0, "median": 6,
            "interval_low": 2, "interval_high": 10, "confidence": "low", "arsenal_coverage": .5,
            "effective_sample_size": 30, "lineup_confirmed": True, "decision": "RESEARCH_ONLY", "factors": [],
        }
        with self.assertRaisesRegex(ValueError, "earlier"):
            save_prediction(42, prediction, db_path=self.db_path)

    def test_confirmed_projection_is_captured_once_and_auto_settled_with_components(self):
        initialize(self.db_path)
        prediction = {
            "as_of": "2099-04-10T16:00:00+00:00", "scheduled_start": "2099-04-10T23:00:00+00:00",
            "player_id": 200, "player_name": "Home Starter", "prop_type": "pitcher_strikeouts",
            "model_version": "v3", "feature_version": "f3", "projection": 6.0, "median": 6,
            "interval_low": 2, "interval_high": 10, "confidence": "low", "arsenal_coverage": .5,
            "effective_sample_size": 30, "lineup_confirmed": True, "decision": "RESEARCH_ONLY",
            "factors": [], "expected_batters_faced": 23.0, "expected_pitches": 92.0,
            "performance_outlook": {"expected_outs": 17.0}, "k_rate": .261,
        }
        first = ensure_pregame_prediction(42, prediction, db_path=self.db_path)
        second = ensure_pregame_prediction(42, prediction, db_path=self.db_path)
        self.assertEqual((first["status"], second["status"]), ("saved", "already_saved"))
        settled = settle_game_predictions(42, {200: {
            "strikeouts": 7, "batters_faced": 24, "pitches": 95, "outs": 18,
            "runs": 2, "earned_runs": 2, "hits": 5, "walks": 1,
        }}, db_path=self.db_path)
        self.assertEqual(settled, [first["prediction_id"]])
        summary = prediction_tracking_summary(self.db_path)
        self.assertEqual((summary["predictions"], summary["settled"]), (1, 1))
        self.assertEqual((summary["k_mae"], summary["bf_mae"]), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
