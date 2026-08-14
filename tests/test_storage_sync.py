import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from analytics_store import connect, initialize
from prediction_store import add_market_snapshot, latest_markets, save_prediction, settle_prediction
from collections import defaultdict

from sync_matchup_data import batter_context_store, count_bucket, completed_game_observations, game_log_pks_from_payload, outcome_counts, process_feed


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
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0], 4)
            columns = {row[1] for row in db.execute("PRAGMA table_info(gameday_batter_pitch_velocity)")}
            self.assertTrue({"doubles", "triples", "total_bases"}.issubset(columns))
            context_columns = {row[1] for row in db.execute("PRAGMA table_info(gameday_batter_pitch_context)")}
            self.assertTrue({"pitcher_throws", "count_bucket", "zone", "total_bases"}.issubset(context_columns))
            self.assertTrue(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gameday_pitcher_arsenal_context'").fetchone())

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
        process_feed(feed, {200}, {100}, pitcher_data, pitcher_context, batter_events, pitch_events,
                     pitch_zones, velocity_events, context_events, quality, workloads)
        self.assertEqual(context_events[100]["SL"][86]["R"]["even"][4], ["double"])

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


if __name__ == "__main__":
    unittest.main()
