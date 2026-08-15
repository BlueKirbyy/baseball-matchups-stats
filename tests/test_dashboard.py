from pathlib import Path
import unittest


INDEX = Path(__file__).resolve().parents[1] / "index.html"


class DashboardSpotlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = INDEX.read_text(encoding="utf-8")

    def test_both_spotlights_exist(self):
        self.assertIn('id="pitcherSpotlight"', self.page)
        self.assertIn('id="batterSpotlight"', self.page)
        self.assertIn("renderSpotlights()", self.page)

    def test_player_strikeout_prop_board_is_removed(self):
        self.assertNotIn('id="propBoard"', self.page)
        self.assertNotIn('id="propRows"', self.page)
        self.assertNotIn("Daily prop board", self.page)
        self.assertNotIn("Line / price", self.page)
        self.assertNotIn("renderPropRows", self.page)
        self.assertIn("async function loadSlateResearch()", self.page)

    def test_spotlights_only_consider_unstarted_games(self):
        self.assertIn("function spotlightEligibleGame(game)", self.page)
        self.assertIn("const upcomingProfiles=slateProfiles.filter(({game})=>spotlightEligibleGame(game))", self.page)
        self.assertIn("const pitchers=upcomingProfiles.flatMap", self.page)
        self.assertIn("const batters=upcomingProfiles.flatMap", self.page)
        self.assertIn("No unstarted games have saved pitcher profiles", self.page)

    def test_page_does_not_horizontally_overscroll_outside_tables(self):
        self.assertIn("html,body{max-width:100%;overflow-x:hidden;overscroll-behavior-x:none}", self.page)
        self.assertIn(".table-wrap{width:100%;max-width:100%;overflow:clip", self.page)
        self.assertIn("table-layout:fixed", self.page)
        self.assertIn(".research-table{min-width:0}", self.page)
        self.assertIn(".layout{display:block;width:100%}", self.page)

    def test_dashboard_is_full_width_without_inline_matchup_preview(self):
        self.assertIn("main{width:100%;max-width:none", self.page)
        self.assertIn("body:not(.matchup-view) #detail{display:none}", self.page)
        self.assertIn("grid-template-columns:repeat(auto-fit,minmax(360px,1fr))", self.page)
        self.assertIn("renderGames();if(matchupView){renderDetail();", self.page)
        self.assertNotIn("renderGames();renderDetail()", self.page)

    def test_matchup_view_is_full_width_without_panel_bars(self):
        self.assertIn(".matchup-view main{width:100%;max-width:none;padding:0}", self.page)
        self.assertIn(".matchup-view #detail{width:100%;max-width:none;margin:0", self.page)
        self.assertIn("border:0;border-radius:0", self.page)
        self.assertIn("wrapper.scrollLeft=0", self.page)

    def test_slate_games_open_a_dedicated_preloaded_matchup_view(self):
        self.assertIn("function openMatchupPage(gamePk)", self.page)
        self.assertIn("window.location.assign(`/matchup?game=${encodeURIComponent(gamePk)}`)", self.page)
        self.assertIn("const matchupView=Number.isInteger(requestedGamePk)&&requestedGamePk>0", self.page)
        self.assertIn("if(matchupView){renderDetail();if(selected?.gamePk===requestedGamePk)", self.page)
        self.assertIn("await loadResearch()", self.page)
        self.assertIn("← Back to slate", self.page)

    def test_spotlights_show_evidence_not_arbitrary_scores(self):
        self.assertIn("lineup_k_evidence?.coverage", self.page)
        self.assertIn("data_grade?.grade", self.page)
        self.assertIn("effective_sample_size)>=10", self.page)
        self.assertIn("expected BF", self.page)
        self.assertNotIn("fitRating", self.page)
        self.assertNotIn("91+", self.page)

    def test_cards_escape_external_text(self):
        self.assertIn("esc(side.pitcher)", self.page)
        self.assertIn("esc(batter.name)", self.page)
        self.assertIn("esc(game.away.abbr)", self.page)

    def test_hitter_pitch_performance_has_accessible_color_cues(self):
        self.assertIn("Lime: higher K risk / favorable hitter matchup", self.page)
        self.assertIn("Red: lower K risk / tough hitter matchup", self.page)
        self.assertIn("Gray: average / limited", self.page)
        self.assertIn("stat.research", self.page)
        self.assertIn("performance-label", self.page)

    def test_pitcher_board_shows_opportunity_and_the_projection_pathway(self):
        self.assertIn("K environment", self.page)
        self.assertIn("Data grade", self.page)
        self.assertIn("Projection pathway", self.page)
        self.assertIn("Confirmed-lineup K risk", self.page)
        self.assertIn("Starter leash", self.page)
        self.assertIn("K risk", self.page)

    def test_matchup_adjusted_workload_pathway_and_manual_cap_are_visible(self):
        self.assertIn("Why ${Number(p.expected_batters_faced).toFixed(1)} expected BF?", self.page)
        self.assertIn("Lineup patience", self.page)
        self.assertIn("Matchup early-hook adjustment", self.page)
        self.assertIn("80% matchup pitch-budget conversion", self.page)
        self.assertIn("Manual pitch cap", self.page)
        self.assertIn("/api/workload-overrides", self.page)

    def test_game_total_and_favorite_are_visible_in_research(self):
        self.assertIn("Game total", self.page)
        self.assertIn("Moneyline favorite", self.page)
        self.assertIn("const marketRead=market", self.page)
        self.assertIn("${esc(marketRead(game.odds))}", self.page)
        self.assertIn("Game market:", self.page)
        self.assertIn("small context points", self.page)

    def test_batter_spotlight_has_outcome_specific_opportunities(self):
        self.assertIn("Best hitter opportunities", self.page)
        for outcome in ("overall", "hit", "total_bases", "home_run", "runs_rbi"):
            self.assertIn(f'data-hitter-outcome="{outcome}"', self.page)
        self.assertIn("['strong','favorable'].includes(opportunity.tier)", self.page)
        self.assertIn("['usable','strong'].includes(opportunity.evidence)", self.page)
        self.assertIn("opportunity.drivers", self.page)
        self.assertIn("Risk:", self.page)
        self.assertIn("not calibrated prop probabilities", self.page)

    def test_batter_spotlight_lists_every_strong_matchup_for_selected_outcome(self):
        self.assertIn('id="allStrongBatterRows"', self.page)
        self.assertIn('id="allStrongBatterTitle"', self.page)
        self.assertIn("strongBatters=qualifiedBatters.filter(item=>item.opportunity.tier==='strong')", self.page)
        self.assertIn("strongBatters.map", self.page)
        self.assertNotIn("strongBatters.slice", self.page)
        self.assertIn("No upcoming confirmed-lineup batters have a strong matchup for this outcome yet.", self.page)

    def test_bullpen_readiness_and_full_game_hitter_blend_are_visible(self):
        self.assertIn("Estimated bullpen readiness", self.page)
        self.assertIn("Likely fresh", self.page)
        self.assertIn("Recent pitches: today", self.page)
        self.assertIn("Hitter fit versus this reliever", self.page)
        self.assertIn("full_game_research||batter.arsenal_research", self.page)
        self.assertIn("Full-game matchup & exposure", self.page)
        self.assertIn("Starter pitch cells remain starter-only", self.page)
        self.assertIn("bullpen_effect", self.page)


if __name__ == "__main__":
    unittest.main()
