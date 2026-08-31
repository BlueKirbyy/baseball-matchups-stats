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
        self.assertIn("No qualified pitcher spots yet", self.page)

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

    def test_research_shell_accounts_for_the_fixed_navigation_rail(self):
        self.assertIn("main{width:calc(100% - var(--rail));margin-left:var(--rail)", self.page)
        self.assertIn("@media(max-width:1000px)", self.page)
        self.assertIn("main{width:100%}", self.page)
        self.assertIn(".game{width:100%;max-width:100%;overflow:hidden", self.page)

    def test_scorebug_does_not_fake_a_pregame_count(self):
        self.assertIn("Pregame · count opens at first pitch", self.page)
        self.assertIn("isLive=/in progress|live/i.test(status)", self.page)
        self.assertIn("scorebug.classList.toggle('live',isLive)", self.page)
        self.assertNotIn(".count-dots.lime i:first-child", self.page)
        self.assertNotIn(".count-dots.red i:first-child", self.page)

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
        self.assertIn("opportunity.requirements", self.page)
        self.assertIn("effective_pa", self.page)
        self.assertIn("Expected to face ${Number(p.expected_batters_faced||0).toFixed(1)} hitters", self.page)
        self.assertNotIn("fitRating", self.page)
        self.assertNotIn("91+", self.page)

    def test_hitter_evidence_cell_uses_separate_readable_rows(self):
        self.assertIn("Evidence quality", self.page)
        self.assertIn("evidence-stack", self.page)
        self.assertIn("Bullpen ${pct(bullpen.modeled_weight)} modeled", self.page)

    def test_prior_dominated_pitch_cells_are_labeled_explicitly(self):
        self.assertIn("read.label==='prior-driven'?'Prior-driven'", self.page)

    def test_unstable_hitter_signals_have_explicit_red_risk_treatment(self):
        self.assertIn("function hitterRisk", self.page)
        self.assertIn("function riskMarkup", self.page)
        self.assertIn("Risk · Unstable signal", self.page)
        self.assertIn("risk-banner", self.page)
        self.assertIn("risk-high", self.page)
        self.assertIn("Red Risk banner: unstable evidence", self.page)
        self.assertIn("item.risk.level!=='high'", self.page)
        self.assertNotIn("false positive", self.page.lower())

    def test_compact_slate_hides_watchlists_until_requested(self):
        self.assertIn('id="watchlistToggle"', self.page)
        self.assertIn('aria-pressed="false"', self.page)
        self.assertIn("showWatchlist=false", self.page)
        self.assertIn("promisingBatters=batters.filter(item=>item.promising)", self.page)
        self.assertIn("primaryPromising=promisingBatters.slice(0,Math.max(0,6-primaryQualified.length))", self.page)
        self.assertIn("watchlistBatters=eligibleBatters.filter", self.page)
        self.assertIn("shownPitchers=[...qualifiedPitchers.slice(0,6),...(showWatchlist?pitcherWatchlist.slice(0,6):[])]", self.page)
        self.assertIn("Promising — positive direction, evidence still developing", self.page)
        self.assertIn("Research watchlist", self.page)
        self.assertIn("Limited-evidence pitcher watchlist", self.page)
        self.assertIn("showWatchlist=!showWatchlist;renderSpotlights()", self.page)

    def test_overall_tab_keeps_outcome_specific_strong_matchups_visible(self):
        self.assertIn("hitterOutcome==='overall'?Object.values", self.page)
        self.assertIn("'All strong matchups'", self.page)

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
        self.assertIn("How the K projection is built", self.page)
        self.assertIn("Strikeout rate describes efficiency", self.page)
        self.assertIn("Expected opportunity", self.page)
        self.assertIn("kAdjustmentMarkup", self.page)

    def test_matchup_adjusted_workload_pathway_and_manual_cap_are_visible(self):
        self.assertIn("How workload becomes ${Number(p.expected_batters_faced).toFixed(1)} hitters faced", self.page)
        self.assertIn("pitches per hitter", self.page)
        self.assertIn("Earlier exit more likely", self.page)
        self.assertIn("More hitters create more strikeout and out opportunities", self.page)

    def test_pitcher_workload_challenger_is_visible_and_explicitly_shadow(self):
        self.assertIn("Machine-learning workload challenger · shadow only", self.page)
        self.assertIn("ML + matchup K distribution", self.page)
        self.assertIn("ML output required safety limits", self.page)
        self.assertIn("Manual pitch cap", self.page)
        self.assertIn("/api/workload-overrides", self.page)

    def test_pitcher_workload_avoids_unexplained_abbreviations(self):
        self.assertNotIn("expected BF", self.page)
        self.assertNotIn("P/BF", self.page)
        self.assertNotIn("P/PA", self.page)
        self.assertNotIn(" pp", self.page)
        self.assertIn("percentage points", self.page)

    def test_bullpen_cards_have_compact_aligned_statuses_and_guidance(self):
        self.assertIn(".bullpen-list{grid-template-columns:repeat(3,minmax(0,1fr))", self.page)
        self.assertIn(".readiness-chip{align-self:start;justify-self:end", self.page)
        self.assertIn("View pitch arsenal and hitter fit", self.page)
        self.assertIn("How to use this:", self.page)

    def test_game_total_and_favorite_are_visible_in_research(self):
        self.assertIn("Game total", self.page)
        self.assertIn("Moneyline favorite", self.page)
        self.assertIn("const marketRead=market", self.page)
        self.assertIn("${esc(marketRead(game.odds))}", self.page)
        self.assertIn("Game market:", self.page)
        self.assertIn("small context points", self.page)

    def test_directional_park_and_weather_fit_is_visible_and_scoped(self):
        self.assertIn("function environmentMarkup", self.page)
        self.assertIn("environment.total_bases_multiplier", self.page)
        self.assertIn("environment.home_run_multiplier", self.page)
        self.assertIn("Park factors & geometry", self.page)
        self.assertIn("context.park?.dimensions", self.page)
        self.assertIn("separate total-base and home-run baselines", self.page)
        self.assertIn("Historical AVG/SLG and strikeout calculations are not rewritten", self.page)

    def test_batter_spotlight_has_outcome_specific_opportunities(self):
        self.assertIn("Best hitter opportunities", self.page)
        for outcome in ("overall", "hit", "total_bases", "home_run", "runs_rbi"):
            self.assertIn(f'data-hitter-outcome="{outcome}"', self.page)
        self.assertIn("['strong','favorable'].includes(direction)", self.page)
        self.assertIn("opportunity.qualified===true", self.page)
        self.assertIn("opportunity.promising===true", self.page)
        self.assertIn("opportunity.drivers", self.page)
        self.assertIn("Risk:", self.page)
        self.assertIn("Why not stronger:", self.page)
        self.assertIn("not calibrated prop probabilities", self.page)

    def test_batter_spotlight_lists_every_strong_matchup_for_selected_outcome(self):
        self.assertIn('id="allStrongBatterRows"', self.page)
        self.assertIn('id="allStrongBatterTitle"', self.page)
        self.assertIn("const strongBatters=batters.flatMap", self.page)
        self.assertIn("opportunity?.confidence==='high'&&opportunity?.qualified===true", self.page)
        self.assertIn("strongBatters.map", self.page)
        self.assertNotIn("strongBatters.slice", self.page)
        self.assertIn("No upcoming confirmed-lineup batters have a strong matchup for this outcome yet.", self.page)

    def test_recent_form_and_spotlight_diagnostics_are_explained(self):
        self.assertIn('id="batterSpotlightDiagnostics"', self.page)
        self.assertIn("function recentFormMarkup", self.page)
        self.assertIn("small, volatile supporting signal", self.page)
        self.assertIn("Recent form pending — not used as an exclusion", self.page)
        self.assertIn("function modelComponentsMarkup", self.page)
        self.assertIn("Lime: qualified positive matchup", self.page)
        self.assertIn("Blue: promising, lower confidence", self.page)

    def test_bullpen_readiness_and_full_game_hitter_blend_are_visible(self):
        self.assertIn("Estimated bullpen readiness", self.page)
        self.assertIn("Likely fresh", self.page)
        self.assertIn("Recent workload:", self.page)
        self.assertIn("Hitter fit versus this reliever", self.page)
        self.assertIn("full_game_research||batter.arsenal_research", self.page)
        self.assertIn("Full-game matchup & exposure", self.page)
        self.assertIn("Starter pitch cells remain starter-only", self.page)
        self.assertIn("bullpen_effect", self.page)

    def test_hitter_shadow_probabilities_and_platoon_anchor_are_visible(self):
        self.assertIn("mlProbabilityMarkup", self.page)
        self.assertIn("excluded from ranking while in shadow", self.page)
        self.assertIn("1+ H", self.page)
        self.assertIn("Exp TB", self.page)
        self.assertIn("platoon.label", self.page)
        self.assertIn("evidence_source", self.page)


if __name__ == "__main__":
    unittest.main()
