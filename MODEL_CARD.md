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

### Workload challenger: `pitcher-workload-challenger-v1`

Feature version: `pitcher-game-pre-event-v1`. Status: shadow; it is displayed beside the production empirical workload but cannot change rankings, confidence, or decisions.

One immutable example is built per historical starter-game. Every feature is reconstructed from games ending before the target game, and state is updated only after the whole game is labeled. Features cover prior and recent BF/pitches/outs, pitcher efficiency and command, days rest, team hook history, opposing-lineup K/traffic/power, recent bullpen use, and season progress. Whole games are split chronologically into 65% train, 17% calibration, and 18% untouched test. Ridge regression is compared with histogram gradient boosting; separate quantile models form workload intervals, and calibrated logistic/gradient challengers estimate early exit.

The August 16 artifact used 2410 train, 630 calibration, and 668 test starts through August 15. Gradient boosting reduced held-out MAE versus the prior-start historical baseline from 3.221 to 2.931 BF (9.0%), 10.878 to 9.767 pitches (10.2%), and 3.104 to 2.919 outs (6.0%). Empirical coverage for the nominal 80% intervals was 78.3%, 81.3%, and 78.6%, respectively. Early-exit log loss improved from 0.669 to 0.615. These results make the challenger suitable for prospective shadow tracking, not production promotion.

At runtime, the shadow BF distribution is combined with the same transparent opponent-adjusted K rate used by v4 to produce a second K distribution. Outputs that move implausibly far from recent-start workload are capped and carry an explicit red risk warning. This safety rule is not evidence that the prediction is correct; it prevents an experimental tail estimate from being presented without context.

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

## Hitter challengers: `hitter-challenger-v1`

Feature version: `hitter-pa-pre-event-v1`

Status: shadow. Shadow outputs are displayed beside the empirical-Bayes read but do not change spotlight ordering, strong/favorable labels, or a wagering decision.

### Leakage controls and features

Each example is one completed regular-season plate appearance. Games are read chronologically; features for every PA in a game are created from state ending before that game, and state is updated only after the entire game has been labeled. The split is by whole game and time: 65% train, the next 17% calibration, and the latest 18% untouched test. Features include season-to-date hitter rates, opposing-hand platoon rates, pitch-mix fit and coverage, pitcher results and repertoire, lineup position, same-side status, velocity, and starter/reliever role. Current-day outcomes, closing markets, and future games are excluded.

The production empirical model now anchors every pitch/velocity cell to the batter's opposing-hand platoon posterior. A same-hand, velocity-matched cell is preferred; an all-hand fallback is explicitly labeled. Missing arsenal mass keeps the platoon prior. A strong label requires at least 35% reliable full-game arsenal coverage, 30 effective PA, and 3% exact count/zone context coverage. Those percentages are not interchangeable: full-game coverage is diluted by every unmodeled pitch and reliever, while context coverage is the much narrower pitcher-hand/count/zone intersection. On the August 15 validation slate, the former policy's 55%/25 PA/20% thresholds admitted 0 of 180 confirmed-lineup hitters because observed maxima were 37% full-game and 8% exact-context coverage; the revised joint gate admitted only two top-tail evidence profiles before matchup direction was considered. A high score with merely usable evidence is capped at favorable. A 100+ AB raw platoon split that materially disagrees with the pitch-fit estimate blocks a strong label and is shown as a downgrade risk.

Matchup direction and stability risk are separate outputs. A signal is marked high risk when it has a critical evidence failure or at least two instability flags: under 15% arsenal coverage, under 10 effective PA, under 2% exact-context coverage, under 50 opposing-hand AB, or under 100 current-season PA. More severe cutoffs—under 10% coverage, five effective PA, 25 opposing-hand AB, or 60 season PA—trigger high risk on their own. A platoon conflict also triggers high risk. The UI renders these cases in red, lists the exact failing inputs, keeps them out of Strong lists, and ranks them behind better-supported watchlist signals.

### Challengers, calibration, and promotion

For hit, extra-base hit, home run, and strikeout targets, training compares a regularized logistic baseline with a histogram gradient-boosting challenger. Both are sigmoid-calibrated on the middle time block. The final JSON export is evaluated by the dependency-free server; training verifies exported predictions against scikit-learn before saving it.

Promotion is per target and requires at least 1,000 out-of-time test PA, at least 100 positives, at least 0.5% log-loss improvement over a constant-rate baseline, Brier score no worse than baseline, and expected calibration error no greater than 0.04. Passing those numerical gates only marks a target eligible. It remains shadow unless training is run with `--promote-eligible`. The August 15 build used 90,672 train, 23,716 calibration, and 25,016 test PA through August 14; strikeout was eligible, while hit, XBH, and HR did not clear the predeclared improvement threshold. All remain shadow.

The UI converts per-PA hit, HR, and strikeout estimates to at-least-one event probabilities using expected PA and displays a range based on challenger disagreement and out-of-time calibration error. Expected total bases is a transparent approximation from the hit/XBH/HR probabilities. These are model estimates, not sportsbook-edge or profitability claims.
