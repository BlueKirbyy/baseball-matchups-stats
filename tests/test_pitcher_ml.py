import unittest

from pitcher_ml import (
    FEATURE_NAMES, PregameWorkloadState, current_feature_vector,
    predict_exported_classifier, predict_exported_regressor,
    shadow_workload_prediction,
)


def ridge(intercept):
    return {
        "kind": "ridge", "feature_names": list(FEATURE_NAMES),
        "medians": [0.0] * len(FEATURE_NAMES),
        "means": [0.0] * len(FEATURE_NAMES),
        "scales": [1.0] * len(FEATURE_NAMES),
        "intercept": float(intercept),
        "coefficients": [0.0] * len(FEATURE_NAMES),
    }


class PitcherMLTests(unittest.TestCase):
    def test_workload_features_use_only_prior_games(self):
        state = PregameWorkloadState()
        starter = {"player_id": 200, "team_id": 10}
        before = state.feature_vector(starter, [], "2026-04-10")
        row = {
            "role": "pitcher", "is_start": 1, "player_id": 200,
            "team_id": 10, "game_date": "2026-04-10",
            "batters_faced": 27, "pitches": 100, "outs": 21,
            "strikeouts": 8, "walks_allowed": 2, "hits_allowed": 5,
        }
        state.update_game([row], "2026-04-10")
        after = state.feature_vector(starter, [], "2026-04-16")
        self.assertEqual(before["pitcher_log_starts"], 0.0)
        self.assertGreater(after["pitcher_log_starts"], 0.0)
        self.assertGreater(after["pitcher_avg_bf"], before["pitcher_avg_bf"])

    def test_pure_python_workload_exports(self):
        features = {name: 0.0 for name in FEATURE_NAMES}
        self.assertEqual(predict_exported_regressor(ridge(23), features), 23)
        classifier = ridge(0)
        classifier["calibration"] = {"intercept": 0.0, "coefficient": 1.0}
        self.assertAlmostEqual(predict_exported_classifier(classifier, features), .5)

    def test_live_features_and_shadow_distribution_are_bounded(self):
        side = {
            "pitcher_id": 200, "pitching_team_id": 10,
            "official_date": "2026-08-16", "appearance_history": [],
            "team_workload_history": [], "bullpen_context": {},
            "batters": [], "workload_override": {"pitch_limit": 80},
        }
        self.assertEqual(set(current_feature_vector(side)), set(FEATURE_NAMES))
        targets = {}
        for target, value, low, high in (
            ("batters_faced", 24, 18, 29), ("pitches", 90, 70, 105),
            ("outs", 18, 12, 23),
        ):
            targets[target] = {
                "selected": "ridge", "candidates": {"ridge": ridge(value)},
                "interval_models": {"low": ridge(low), "high": ridge(high)},
                "interval_calibration": 0, "metrics": {},
            }
        early = ridge(0)
        early["calibration"] = {"intercept": 0.0, "coefficient": 1.0}
        targets["early_exit"] = {
            "selected": "logistic", "candidates": {"logistic": early},
            "metrics": {},
        }
        registry = {
            "model_version": "test", "feature_version": "pitcher-game-pre-event-v1",
            "trained_through": "2026-08-15", "targets": targets,
        }
        result = shadow_workload_prediction(side, registry)
        self.assertTrue(result["available"])
        self.assertEqual(result["expected_pitches"], 80)
        self.assertLess(result["expected_batters_faced"], 24)
        self.assertAlmostEqual(result["early_exit_probability"], .5)


if __name__ == "__main__":
    unittest.main()
