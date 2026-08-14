import json
import math
import unittest

from modeling import (
    american_from_probability, distribution_summary, expected_value,
    hitter_arsenal_summary, hitter_k_risk, hitter_market_context, hitter_pitch_summary, implied_probability, k_data_grade,
    lineup_k_evidence, no_vig_probabilities, pitch_mix_evidence, pitcher_k_projection, shrunk_rate,
)


class ModelingTests(unittest.TestCase):
    def test_american_odds_and_no_vig(self):
        self.assertAlmostEqual(implied_probability(-110), 110 / 210)
        self.assertAlmostEqual(implied_probability(150), 0.4)
        no_vig = no_vig_probabilities(-110, -110)
        self.assertAlmostEqual(no_vig["over"], 0.5)
        self.assertEqual(american_from_probability(0.6), -150)
        self.assertGreater(expected_value(0.6, -110), 0)

    def test_small_samples_shrink_more(self):
        distances = []
        for pa in (3, 10, 25, 100):
            rate = shrunk_rate(pa, pa, 0.225, 60)
            distances.append(rate - 0.225)
        self.assertEqual(distances, sorted(distances))
        self.assertLess(shrunk_rate(3, 3, 0.225, 60), 0.27)

    def test_arsenal_coverage_does_not_renormalize_missing_pitch(self):
        side = {
            "arsenal": [{"code": "FF", "usage": 20}, {"code": "SL", "usage": 80}],
            "batters": [{"lineup_order": 1, "vs_pitches": {"FF": {"pa": 100, "strikeouts": 100}}}],
        }
        evidence = pitch_mix_evidence(side, 0.2)
        self.assertLess(evidence["coverage"], 0.2)
        self.assertLess(evidence["rate"], 0.31)

    def test_hitter_three_pa_never_high_confidence(self):
        batter = {"vs_pitches": {"FF": {"pa": 3, "avg": "1.000"}}}
        result = hitter_arsenal_summary(batter, [{"code": "FF", "usage": 100}])
        self.assertEqual(result["label"], "insufficient")
        self.assertEqual(result["tier"], "watchlist")
        self.assertLess(result["expected_average"], 0.29)

    def test_hitter_favorable_tier_needs_coverage_but_not_strong_delta(self):
        batter = {"vs_pitches": {"FF": {"pa": 60, "avg": "0.280"}}}
        result = hitter_arsenal_summary(batter, [{"code": "FF", "usage": 100}])
        self.assertEqual(result["label"], "favorable contact research")
        self.assertEqual((result["tier"], result["tone"]), ("favorable", "good"))

    def test_power_is_part_of_the_hitter_fit_score(self):
        contact_only = hitter_arsenal_summary(
            {"vs_pitches": {"FF": {"pa": 60, "avg": ".260"}}},
            [{"code": "FF", "usage": 100}],
        )
        power_matchup = hitter_arsenal_summary(
            {"vs_pitches": {"FF": {
                "pa": 60, "avg": ".260", "advanced": {"slg": .700, "iso": .440},
            }}},
            [{"code": "FF", "usage": 100}],
        )
        self.assertGreater(power_matchup["score"], contact_only["score"])
        self.assertGreater(power_matchup["expected_slg"], contact_only["expected_slg"])

    def test_hitter_opportunities_separate_contact_from_one_hit_power(self):
        batter = {
            "lineup_order": 3,
            "k_profile": {"pa": 220, "strikeouts": 45},
            "vs_pitches": {
                "FF": {
                    "pa": 70, "avg": ".250", "hr": 8,
                    "advanced": {"slg": .700, "iso": .450},
                    "quality": {"batted_balls": 45, "hard_hits": 25, "barrel_proxy": 9},
                },
                "SL": {
                    "pa": 40, "avg": ".225", "hr": 5,
                    "advanced": {"slg": .575, "iso": .350},
                    "quality": {"batted_balls": 24, "hard_hits": 13, "barrel_proxy": 5},
                },
            },
        }
        market = hitter_market_context({"total": 9.5, "favorite": {"team": "Test"}}, "Test")
        result = hitter_arsenal_summary(
            batter,
            [{"code": "FF", "usage": 60}, {"code": "SL", "usage": 40}],
            market_context=market,
        )
        opportunities = result["opportunities"]["items"]
        self.assertEqual(set(opportunities), {"overall", "hit", "total_bases", "home_run", "runs_rbi"})
        self.assertEqual(opportunities["hit"]["tier"], "neutral")
        self.assertEqual(opportunities["total_bases"]["tier"], "strong")
        self.assertEqual(opportunities["home_run"]["tier"], "strong")
        self.assertEqual(result["opportunities"]["primary"], "home_run")
        self.assertIn("Runs/RBIs also depend", opportunities["runs_rbi"]["risks"][0])

    def test_game_total_and_favorite_are_a_small_explicit_hitter_adjustment(self):
        context = hitter_market_context(
            {"total": 10.0, "favorite": {"team": "Test Team", "probability": .62}},
            "Test Team",
        )
        self.assertTrue(context["available"])
        self.assertGreater(context["adjustment"], 0)
        self.assertLessEqual(context["adjustment"], .005)
        batter = {"vs_pitches": {"FF": {"pa": 60, "avg": ".280"}}}
        baseline = hitter_arsenal_summary(batter, [{"code": "FF", "usage": 100}])
        adjusted = hitter_arsenal_summary(batter, [{"code": "FF", "usage": 100}], market_context=context)
        self.assertAlmostEqual(adjusted["base_score"], baseline["score"])
        self.assertGreater(adjusted["score"], baseline["score"])

    def test_pitch_contact_colors_require_sample_and_use_shrinkage(self):
        tiny = hitter_pitch_summary({"pa": 3, "avg": "1.000"})
        favorable = hitter_pitch_summary({"pa": 25, "avg": "0.500"})
        poor = hitter_pitch_summary({"pa": 25, "avg": "0.000"})
        neutral = hitter_pitch_summary({"pa": 25, "avg": "0.250"})
        self.assertEqual((tiny["label"], tiny["tone"]), ("low data", "neutral"))
        self.assertEqual((favorable["label"], favorable["tone"]), ("favorable", "good"))
        self.assertEqual((poor["label"], poor["tone"]), ("poor", "bad"))
        self.assertEqual((neutral["label"], neutral["tone"]), ("neutral", "neutral"))
        self.assertLess(favorable["shrunk_average"], 0.5)

    def test_lineup_k_risk_is_broader_than_sparse_pitch_mix_evidence(self):
        side = {
            "batters": [
                {"lineup_order": 1, "k_profile": {"pa": 180, "strikeouts": 58}},
                {"lineup_order": 2, "k_profile": {"pa": 160, "strikeouts": 23}},
            ]
        }
        evidence = lineup_k_evidence(side)
        self.assertGreater(evidence["coverage"], .5)
        self.assertEqual(evidence["hitters_covered"], 2)
        self.assertEqual(hitter_k_risk({"pa": 180, "strikeouts": 58})["tone"], "good")
        self.assertEqual(k_data_grade(True, 10, evidence, {"coverage": .1}, False)["grade"], "B")

    def test_k_projection_explains_lineup_and_workload_components(self):
        side = {
            "pitcher_id": 1, "pitcher": "Test Pitcher", "lineup_confirmed": True,
            "data_freshness_seconds": 0,
            "workload": {"appearances": 10, "batters_faced": 220, "strikeouts": 60,
                         "recent_appearances": 3, "recent_batters_faced": 66, "recent_strikeouts": 20},
            "appearance_history": [
                {"is_start": 1, "batters_faced": 22, "pitches": 91},
                {"is_start": 1, "batters_faced": 23, "pitches": 95},
                {"is_start": 1, "batters_faced": 24, "pitches": 98},
                {"is_start": 1, "batters_faced": 23, "pitches": 94},
                {"is_start": 1, "batters_faced": 22, "pitches": 90},
                {"is_start": 1, "batters_faced": 24, "pitches": 99},
            ],
            "arsenal": [{"code": "FF", "usage": 100}],
            "batters": [
                {"lineup_order": order, "k_profile": {"pa": 180, "strikeouts": 58}, "vs_pitches": {}}
                for order in range(1, 10)
            ],
        }
        result = pitcher_k_projection(side)
        self.assertEqual(result["data_grade"]["grade"], "B")
        self.assertGreater(result["components"]["lineup_adjustment"], 0)
        self.assertIn(result["workload_read"]["label"], {"stable workload", "recently extended"})
        self.assertIn("opportunity", result)

    def test_count_distribution_is_valid_and_overdispersed(self):
        result = distribution_summary(6.0, 5.5, "negative_binomial")
        self.assertAlmostEqual(result["probability_over"] + result["probability_under"], 1.0, places=8)
        self.assertLessEqual(result["interval_low"], result["median"])
        self.assertGreaterEqual(result["interval_high"], result["median"])

    def test_lineup_and_price_guardrails(self):
        side = {
            "pitcher_id": 1, "pitcher": "Test Pitcher", "lineup_confirmed": False,
            "data_freshness_seconds": 0,
            "workload": {"appearances": 10, "batters_faced": 220, "strikeouts": 55,
                         "recent_appearances": 3, "recent_batters_faced": 66, "recent_strikeouts": 17},
            "arsenal": [{"code": "FF", "usage": 100}], "batters": [],
        }
        self.assertEqual(pitcher_k_projection(side)["decision"], "WAIT_FOR_LINEUP")
        side["lineup_confirmed"] = True
        self.assertEqual(pitcher_k_projection(side)["decision"], "NEED_PRICE")
        pickem = {"platform_type": "pickem", "provider": "P", "line": 5.5, "market_snapshot_id": 1}
        result = pitcher_k_projection(side, pickem)
        self.assertEqual(result["decision"], "RESEARCH_ONLY")
        self.assertIsNone(result["market"]["expected_value_over"])
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
