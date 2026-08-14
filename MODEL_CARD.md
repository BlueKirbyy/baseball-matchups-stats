# Diamond Intel model card

## Pitcher strikeouts: `pitcher-k-eb-v2`

Feature version: `gameday-features-v2`

Status: unvalidated research baseline. It is not approved to emit a bet recommendation or claim profitability.

### Target

Starting-pitcher strikeouts in one MLB game. The displayed probabilities apply to a timestamped market line; price-aware EV is calculated only for a two-sided sportsbook market.

### Feature cutoff

All game observations must have `game_date < target officialDate`, and every saved prediction has `as_of < scheduled_start`. Player-specific game logs retain pre-trade history. Pregame lineup and market values are stored with their capture timestamps. Closing values are evaluation-only and never prediction features.

### Method

- Pitcher strikeouts per batter faced use a beta-binomial posterior with an 80-BF league prior.
- Recent performance contributes at one-quarter weight and is still shrunk.
- Confirmed lineup K susceptibility uses each hitter's broader final-pitch PA history against the probable starter's throwing hand, with a 120-PA league K% prior and lineup-order weights.
- Batter/pitch-type strikeout evidence uses a 60-PA prior per pitch as a smaller pitch-mix refinement, rather than the primary matchup gate.
- Missing arsenal evidence retains the pitcher posterior instead of being discarded and renormalized.
- Expected batters faced prefers start-only player-game observations and shrinks toward 22 BF per start. Until those rows are populated, an explicitly labeled mixed-role aggregate fallback is used.
- Game strikeouts use a negative-binomial distribution with dispersion 12. The backtest reports an out-of-sample Poisson comparison; the distribution choice must be revisited when enough settled predictions exist.
- Raw umpire outcome rates, park geometry, weather, and game moneyline are not numerical adjustments in v1 because their effects have not been calibrated here.

### Confidence

The dashboard reports a separate A–D data grade. A requires a confirmed lineup, fresh data, at least 10 starts, at least 55% broader lineup K coverage, and 35% pitch-mix coverage; B requires a confirmed lineup, fresh data, at least six starts, and 35% lineup coverage; C is a limited but visible confirmed-lineup research read. `medium` requires A-grade data plus at least 35 effective pitch-mix PA; B/C are `low`; D is `insufficient`. These describe data quality, not the chance that a wager wins.

### Training and evaluation

No fitted training optimization is performed in v1. Priors are conservative documented assumptions. Evaluation is chronological over immutable predictions and includes count error, calibration, distribution comparison, rolling baselines, market baselines, price-aware ROI, and same-line CLV.

Current settled out-of-sample sample size: 0 at release. Calibration, ROI, and CLV are therefore unavailable. The decision system remains locked to research-only states.

### Release gate for actionable labels

Do not enable a `BET` state merely because historical ROI is positive. At minimum require a predeclared, chronologically held-out sample, acceptable calibration, improvement over simple and market baselines, sufficient bets across multiple months, stable sensitivity results, and positive CLV after accounting for vig. Document the exact thresholds and results in a new model version.

### Known failure modes

- Probable starters, roles, pitch limits, injuries, and lineups can change after capture.
- The legacy aggregate fallback can mix starts and relief appearances until observation history is populated.
- Pitch-type PA attribution uses the final pitch and is descriptive rather than causal.
- The K-opportunity label is a descriptive matchup environment based on projected Ks, K rate, and starter workload; it is not a probability of beating a prop line.
- Starter-leash labels are descriptive comparisons of recent versus season batters faced and pitch counts; they are not an independently calibrated injury, coach, or pitch-limit forecast.
- Hitter pitch cells select historical results against the probable starter's throwing hand and reweight them to that starter's coarse count state (pitcher ahead/even/hitter ahead) and 3×3 strike-zone mix. Those cells are sparse and use a 20-AB prior back to the hitter's broader same-pitch, same-velocity sample; they are descriptive, not causal.
- Hitter pitch cells report descriptive AVG, SLG, ISO, and XBH from final-pitch outcomes. They are not player-prop probabilities or projected game totals.
- When ESPN publishes it, the game total supplies a capped ±3.5-point run-environment adjustment to the hitter fit, and the listed moneyline favorite receives a 1.5-point team-scoring-context adjustment. These are transparent ranking context, not calibrated hit/total-base probabilities; absent markets add zero.
- Bullpen exposure is not modeled for hitter props.
- Park, weather, and umpire inputs are descriptive, not calibrated adjustments.
- Public feeds and manually imported markets may be delayed, incomplete, or incorrect.
- Negative-binomial dispersion is provisional until evaluated on a sufficient walk-forward sample.

### Unsupported uses

- Claiming a proven betting advantage
- Automated wagering
- Treating pick'em legs as independent sportsbook odds
- Using active-roster research as a confirmed-lineup signal
- Inferring official xwOBA, injury status, or a live sportsbook price when it was not supplied
- Using hitter contact research as a probability for hits, total bases, or home runs
