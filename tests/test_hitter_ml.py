import json
from pathlib import Path
import unittest

from hitter_ml import (
    FEATURE_NAMES, PregameState, current_feature_vector, game_pa_rows,
    predict_exported,
)


FIXTURE = Path(__file__).parent / "fixtures" / "completed_game.json"


class HitterMLTests(unittest.TestCase):
    def test_features_are_snapshotted_before_game_updates(self):
        state = PregameState()
        before = state.feature_vector(10, 20, "R", "L", "FF", 2, True, 5)
        play = {
            "batter_id": 10, "pitcher_id": 20, "pitcher_throws": "R",
            "pitch_code": "FF", "pitches": [{"code": "FF", "velo": 96.0}],
            "at_bat": 1, "hit": 1, "total_bases": 2, "home_run": 0,
            "strikeout": 0, "extra_base_hit": 1, "walk": 0,
        }
        state.update_game([play])
        after = state.feature_vector(10, 20, "R", "L", "FF", 2, True, 5)
        self.assertEqual(before["batter_log_pa"], 0.0)
        self.assertGreater(after["batter_log_pa"], before["batter_log_pa"])
        self.assertGreater(after["platoon_avg"], before["platoon_avg"])

    def test_completed_feed_yields_plate_appearance_targets(self):
        rows = game_pa_rows(json.loads(FIXTURE.read_text(encoding="utf-8")))
        self.assertTrue(rows)
        self.assertTrue(all(row["batter_id"] and row["pitcher_id"] for row in rows))
        self.assertTrue(all(row["total_bases"] >= 0 for row in rows))

    def test_pure_python_export_predictors_are_bounded(self):
        features = {name: 0.0 for name in FEATURE_NAMES}
        logistic = {
            "kind": "logistic", "feature_names": list(FEATURE_NAMES),
            "medians": [0.0] * len(FEATURE_NAMES), "means": [0.0] * len(FEATURE_NAMES),
            "scales": [1.0] * len(FEATURE_NAMES), "intercept": 0.0,
            "coefficients": [0.0] * len(FEATURE_NAMES),
            "calibration": {"intercept": 0.0, "coefficient": 1.0},
        }
        histogram = {
            "kind": "hist_gradient_boosting", "feature_names": list(FEATURE_NAMES),
            "medians": [0.0] * len(FEATURE_NAMES), "initial_log_odds": 0.0,
            "trees": [{"left": [0], "right": [0], "feature": [0],
                       "threshold": [0.0], "value": [.2], "is_leaf": [1],
                       "missing_go_to_left": [0]}],
            "calibration": {"intercept": 0.0, "coefficient": 1.0},
        }
        self.assertAlmostEqual(predict_exported(logistic, features), .5)
        self.assertGreater(predict_exported(histogram, features), .5)

    def test_live_feature_map_contains_every_training_feature(self):
        hitter = {
            "lineup_order": 3, "bat_side": "L",
            "season": {"pa": 400, "avg": ".260", "hr": 20},
            "discipline": {"plate_appearances": 400, "hits": 100,
                           "total_bases": 170, "walks": 40},
            "platoon": {"pa": 150, "posterior_avg": .250,
                        "posterior_slg": .410, "posterior_iso": .160},
            "k_profile": {"research": {"posterior": .22}},
            "full_game_research": {"expected_average": .255,
                                   "expected_slg": .420, "coverage": .50},
        }
        side = {"pitcher_throws": "R", "workload": {"batters_faced": 400,
                "strikeouts": 100}, "arsenal": [{"code": "FF", "usage": 100,
                "velo": 95}], "appearance_history": []}
        self.assertEqual(set(current_feature_vector(hitter, side)), set(FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
