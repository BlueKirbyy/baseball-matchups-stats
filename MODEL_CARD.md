# Diamond Intel model card

## Pitcher strikeouts: `pitcher-k-workload-v4`

Feature version: `gameday-features-v4`

Status: unvalidated research baseline. It is not approved to emit a bet recommendation or claim profitability.

### Target

Starting-pitcher strikeouts in one MLB game. The displayed probabilities apply to a timestamped market line; price-aware EV is calculated only for a two-sided sportsbook market.

### Feature cutoff

All game observations must have `game_date < target officialDate`, and every saved prediction has `as_of < scheduled_start`. Player-specific game logs retain pre-trade history. Pregame lineup and market values are stored with their capture timestamps. Closing values are evaluation-only and never prediction features.

### Method

- Pitcher strikeouts per batter faced use a beta-binomial posterior with an 80-BF league prior.
- Recent performance contributes at one-quarter weight and is still shrunk.
- Confirmed lineup K susceptibility uses each hitter's broader final-pitch PA history against the probable starter's throwing hand, with a 120-PA league K% prior, lineup-order weights, and an evidence-coverage weight.
- Batter/pitch-type strikeout evidence uses a 60-PA prior per pitch and is capped at a 1.2 percentage-point adjustment because this sparse feature was too influential before settled calibration data existed.
- Missing arsenal evidence retains the pitcher posterior instead of being discarded and renormalized.
- Workload begins with a start-only pitch-budget prior. Confirmed-lineup pitches per PA, on-base proxy, total bases per PA, and out rate are shrunk with 120-PA league priors and lineup-order weighted.
- The free ESPN game total and moneyline contribute a capped team-run context. Opponent patience, traffic, power, and run context adjust early-hook risk and the available pitch budget; every intermediate value is displayed.
- Expected BF is 80% opponent-adjusted pitch budget divided by matchup pitches per batter and 20% historical BF stabilization. Expected outs use the opponent-adjusted out rate. A timestamped manual pitch cap can lower, but never raise, the model pitch budget.
- Until start-only and batter-discipline rows are populated, explicitly labeled aggregate or league-prior fallbacks are used.
- Game strikeouts mix a workload distribution with Beta-binomial K-rate uncertainty. Workload, pitch-count, outs, and K-rate error are evaluated separately once final MLB results arrive.
- Raw umpire outcome rates, park geometry, and weather are not numerical adjustments in v4 because their effects have not been calibrated here.

### Confidence

The dashboard reports a separate A–D data grade. A requires a confirmed lineup, fresh data, at least 10 starts, at least 55% broader lineup K coverage, and 35% pitch-mix coverage; B requires a confirmed lineup, fresh data, at least six starts, and 35% lineup coverage; C is a limited but visible confirmed-lineup research read. `medium` requires A-grade data plus at least 35 effective pitch-mix PA; B/C are `low`; D is `insufficient`. These describe data quality, not the chance that a wager wins.

### Training and evaluation

No fitted training optimization is performed in v4. Priors and adjustment caps are conservative documented assumptions. Evaluation is chronological over immutable predictions and includes count error, calibration, distribution comparison, rolling baselines, market baselines, price-aware ROI, and same-line CLV.

Current settled out-of-sample sample size: 0 at release. Calibration, ROI, and CLV are therefore unavailable. The decision system remains locked to research-only states.

### Release gate for actionable labels

Do not enable a `BET` state merely because historical ROI is positive. At minimum require a predeclared, chronologically held-out sample, acceptable calibration, improvement over simple and market baselines, sufficient bets across multiple months, stable sensitivity results, and positive CLV after accounting for vig. Document the exact thresholds and results in a new model version.

### Known failure modes

- Probable starters, roles, pitch limits, injuries, and lineups can change after capture.
- The legacy aggregate fallback can mix starts and relief appearances until observation history is populated.
- Pitch-type PA attribution uses the final pitch and is descriptive rather than causal.
- The K-opportunity label is a descriptive matchup environment based on projected Ks, K rate, and starter workload; it is not a probability of beating a prop line.
- Workload matchup adjustments are transparent heuristics, not a fitted survival model. They do not know unreported injuries, bullpen availability, or manager intent.
- A manual pitch cap is user-supplied scenario context. It must represent an informed restriction and cannot increase the model's historical pitch budget.
- Hitter pitch cells select historical results against the probable starter's throwing hand and reweight them to that starter's coarse count state (pitcher ahead/even/hitter ahead) and 3×3 strike-zone mix. Those cells are sparse and use a 20-AB prior back to the hitter's broader same-pitch, same-velocity sample; they are descriptive, not causal.
- Hitter pitch cells report descriptive AVG, SLG, ISO, and XBH from final-pitch outcomes. They are not player-prop probabilities or projected game totals.
- When ESPN publishes it, the game total supplies a capped ±3.5-point run-environment adjustment to the hitter fit, and the listed moneyline favorite receives a 1.5-point team-scoring-context adjustment. These are transparent ranking context, not calibrated hit/total-base probabilities; absent markets add zero.
- Hitter research estimates starter plate appearances from batting order and the starter's projected batters-faced distribution, then assigns the remaining expected appearances to a readiness/role-weighted relief mix. Each reliever is evaluated with the same pitch-type, velocity, handedness, count, and zone framework as the starter. Missing reliever arsenal mass is retained as league-average uncertainty rather than redistributed to known pitchers. This is descriptive full-game research, not a calibrated hitter-prop forecast.
- Bullpen readiness uses recent pitches and consecutive-day use. It does not observe warmups, injuries, transactions after capture, or manager intent, so it must never be read as confirmed availability.
- Park, weather, and umpire inputs are descriptive, not calibrated adjustments.
- Public feeds and manually imported markets may be delayed, incomplete, or incorrect.
- Beta-binomial rate uncertainty and the workload mixture remain provisional until evaluated on a sufficient walk-forward sample.

### Unsupported uses

- Claiming a proven betting advantage
- Automated wagering
- Treating pick'em legs as independent sportsbook odds
- Using active-roster research as a confirmed-lineup signal
- Inferring official xwOBA, injury status, or a live sportsbook price when it was not supplied
- Using hitter contact research as a probability for hits, total bases, or home runs
