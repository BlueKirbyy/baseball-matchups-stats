# Diamond Intel

Diamond Intel is a local MLB matchup and player-prop research board. It combines MLB Gameday pitch feeds, confirmed lineups, pitcher workload, pitch-mix evidence, game context, and timestamped prop prices.

The current pitcher-strikeout model is an unvalidated research baseline. The application intentionally does not call a projection a profitable edge until walk-forward results demonstrate calibration and value.

## Architecture

- `sync_matchup_data.py`: downloads and caches MLB Gameday feeds, builds current matchup aggregates, and records immutable player-game observations.
- `analytics_store.py`: SQLite schema plus numbered, non-destructive migrations.
- `modeling.py`: empirical-Bayes shrinkage, arsenal coverage, directional park/weather fit, bullpen readiness/exposure, pitcher-K workload, count distributions, and market math.
- `park_factors.py`: versioned, handedness-specific Statcast venue factors for separate total-base and home-run context.
- `pitcher_ml.py` / `train_pitcher_ml.py`: chronological starter-game features, workload challengers, dependency-free shadow inference, and the workload-driven K distribution.
- `prediction_store.py`: append-only pregame, market, prediction, and result records.
- `market_data.py`: sportsbook/pick'em CSV and manual market import.
- `backtest.py`: chronological evaluation, calibration, distribution comparison, ROI, and same-line CLV.
- `server.py`: localhost API and static server.
- `index.html`: responsive daily prop board and detailed pitch research. It displays server-computed statistics; model formulas no longer live in browser JavaScript.
- `diamond_intel.db`: the existing local database. Initialization migrates it in place and does not delete existing tables.

## Setup and start

Python 3.9 or newer is sufficient; runtime code uses the standard library.

```bash
python3 server.py
```

Open http://127.0.0.1:8000. The server deliberately binds only to localhost.

Run tests without network access:

```bash
python3 -m unittest discover -s tests -v
```

## Database migrations

Starting the server or a CLI calls `analytics_store.initialize()`. Migrations are recorded in `schema_migrations`; the current version is 11. Existing research tables and rows are retained. Database triggers reject updates and deletes on observations, pregame snapshots, ML feature snapshots, settled outcomes, training examples, bullpen snapshots, market snapshots, workload overrides, predictions, and results.

To inspect the database safely:

```bash
sqlite3 diamond_intel.db 'PRAGMA integrity_check; PRAGMA user_version; SELECT * FROM schema_migrations;'
```

Back up `diamond_intel.db` before any manual SQL changes. The application does not provide a destructive downgrade.

## Sync research data

Sync one upcoming game:

```bash
python3 sync_statcast.py --game-pk 823917
```

Sync the active slate:

```bash
python3 sync_statcast.py --all
```

The active slate follows the computer's local date while any game remains
unstarted. As soon as every game has begun, been postponed, or been canceled,
both the dashboard and `--all` automatically switch to tomorrow's schedule,
even if the final game is still in progress.
Tomorrow's games remain visible when probable starters or confirmed lineups
have not been announced yet.

The sync reuses `.gameday_cache`. It enforces a strict `game_date < target_game_date` historical cutoff. Team schedules are supplemented with player-specific game logs, which preserves games played for former teams after a trade. Completed feeds create game-level pitcher and batter observations, including starter classification and handedness.

Pitch-matchup cells include descriptive AVG, SLG, ISO, and XBH results from the
same final-pitch sample. The matchup read also reweights those results to the
probable starter's throwing hand and historical three-state count / 3×3 zone
mix; sparse split cells shrink back to the hitter's same-pitch, same-velocity
sample. After this update, run `python3 sync_statcast.py --all` once to populate
the added power and count/hand/zone context fields. Until then, the board
explicitly marks power metrics as pending.

Each game card also displays ESPN's best-effort public total and moneyline
favorite when available. Hitter ranking turns those two inputs into a clearly
shown, team-specific run expectation: the moneyline favorite receives the
larger share of the total, while games without a usable favorite are split
evenly. The resulting market adjustment is capped; pitch, power, lineup, and
sample evidence remain the primary inputs, and missing markets produce no
adjustment.

The same sync also creates a timestamped bullpen snapshot for each team. It
identifies active relief candidates, estimates readiness from pitches thrown
today and over the prior two days, infers a broad short/long-relief role, and
builds pitch profiles from the same strictly pregame Gameday history. These are
workload estimates—not official availability or manager intent.

The sync also saves each hitter's five-sector spray distribution (LF, LCF, CF,
RCF, RF) by batting side, pitcher hand, and pitch type. The matchup model starts
with handedness-specific Baseball Savant Statcast park factors for all 30
current MLB venues. Standard venues use the 2024-2026 three-year window;
Sutter Health Park uses 2025-2026 because it lacks a third MLB season. Separate
home-run and total-base baselines preserve important differences such as parks
that suppress homers but create extra doubles or triples. Total-base factors
combine the official 1B/2B/3B/HR indices using fixed league-level total-base
contribution weights documented in `park_factors.py`.

The hitter's pitch-mix-weighted spray, wall geometry, roof status, temperature,
and directional wind then make smaller game-specific adjustments. Dimensions
are deliberately a residual—not the park baseline—because the empirical
Statcast factors already contain the park's average geometry, altitude, and
typical conditions. This context is used only for **total-base** and **home-run**
opportunity; it does not rewrite historical AVG/SLG or alter hit and strikeout
reads. The UI shows both multipliers, batter side, pull rate, sample size,
window, and confidence so the adjustment is auditable. Run
`python3 sync_statcast.py --all` once after upgrading to backfill the spray
table; normal future syncs keep it current.

For an old game, the target game itself and later dates are excluded. Cached feeds are raw source responses; prediction features still apply the as-of cutoff.

## Import prop markets

The import format is demonstrated in `examples/markets.csv`. Required columns are:

- `captured_at`: ISO-8601 timestamp with timezone
- `provider`
- `platform_type`: `sportsbook` or `pickem`
- `player_name`
- `prop_type`
- `line`

Recommended identifiers are `game_pk` and `player_id`.

Every import is a new immutable snapshot. Use `--closing` only for the final pregame observation you intend to treat as the closing price.

The matchup page also has a local **Save K line** form. Enter the sportsbook/source and strikeout line; over/under prices are optional. This does not call a paid API and does not replace the app's model projection. The API accepts the same fields with `POST /api/markets`.

The separate **Save pitch cap** form is for a real workload restriction such as an injury return or announced manager limit. It is stored locally and immutably, can only lower the model's historical pitch budget, and is not a sportsbook projection. Leave it empty when no credible restriction is known.

## Prediction interpretation

The version `pitcher-k-workload-v4` estimates:

1. A pitcher K rate, shrunk toward a league prior.
2. Confirmed-opponent K susceptibility against the probable starter's throwing hand, shrunk per hitter and weighted by lineup order; exact pitch-mix K evidence is a smaller refinement.
3. A baseline pitch budget from start-only history, with extra but still-shrunk weight on the five most recent starts.
4. Tonight's lineup patience, baserunner rate, power, and out conversion from free MLB Gameday outcomes, shrunk toward league priors and weighted by batting order.
5. A capped game-total/favorite adjustment, matchup early-hook risk, optional manual pitch cap, expected BF, and expected outs.
6. A workload/Beta-binomial strikeout distribution with an 80% interval and milestone probabilities.
7. Separate workload, out-conversion, baserunner/command, run-suppression, and early-exit reads.

Unobserved pitch types retain the pitcher prior; the model does not renormalize a small observed slice into a complete arsenal. Three plate appearances cannot create high confidence. The pitcher display separates a K-opportunity read from a data grade: a useful matchup environment can be shown with limited data, but that data grade remains visible. `arsenal coverage` incorporates pitch usage and sample reliability, while lineup K coverage reports the broader handedness-matched opponent sample.

Current decision states:

- `WAIT_FOR_LINEUP`: no official opposing lineup
- `NEED_PRICE`: projection exists but no prop price exists
- `STALE_MARKET`: imported market is more than six hours old
- `STALE_DATA`: matchup sync is missing or more than 36 hours old
- `RESEARCH_ONLY`: pick'em or incomplete two-sided sportsbook price
- `PASS_LOW_DATA`: insufficient coverage/workload
- `RESEARCH_ONLY_UNVALIDATED`: enough inputs for comparison, but the model has not passed a profitability/calibration gate

No current state is a bet recommendation.

### Slate spotlights

The dashboard includes separate pitcher and batter spotlights. Pitcher cards show the K environment, a transparent A–D data grade, expected batters faced, pitch budget, and early-exit risk. Research-ready K cards require a confirmed lineup, data grade A/B, and a favorable or high K environment; lower-grade cards can still appear as clearly labeled context rather than being hidden as generic watchlists.

The batter spotlight separates overall offense, 1+ hit, total-base power, home-run power, and runs/RBIs. Each lens ranks only its relevant descriptive evidence: contact and K history for hits; SLG, ISO, and hard-hit evidence for total bases; ISO, barrel proxy, and pitch-ending HR rate for home-run power; and lineup position, expected PA, game total, favorite status, contact, and power for runs/RBIs. A hitter can therefore surface as a strong power matchup without being mislabeled as an elite contact matchup. Spotlight rankings use the full-game blend when a bullpen snapshot is available and explicitly label whether the relief group boosts, supports, lowers, or leaves the starter-only outlook uncertain.

Batters qualify only when the official lineup is confirmed, the selected outcome read is favorable, evidence is usable or strong, coverage is at least 35%, and effective evidence is at least 10 PA. Cards state why the hitter surfaced, the most important risk, coverage, and effective sample. The categories are research rankings—not calibrated prop probabilities. If nobody qualifies, the panel clearly switches to a lower-confidence watchlist rather than disappearing or relaxing the thresholds silently.

### Train the hitter challengers

The normal server still runs with system Python and no ML dependency. Training uses an isolated virtual environment and writes the large immutable example set to ignored `hitter_training.db`; the small pure-JSON model artifact is loaded by the regular server.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train_hitter_ml.py
```

The command scans saved completed MLB Gameday feeds in chronological order, builds pre-event features, trains logistic and histogram-gradient-boosting challengers, calibrates them on a later block, tests on the latest block, verifies the dependency-free export, and writes `models/hitter_ml_registry.json`. It defaults to shadow mode. To promote only targets that pass every documented out-of-time gate:

```bash
.venv/bin/python train_hitter_ml.py --skip-build --promote-eligible
```

Do not promote a target just because one metric improved. Review the generated Brier, log-loss, ECE, sample, and positive-event counts in the registry and update the model card for each release.

### Train the pitcher workload challenger

The workload trainer reconstructs features before every historical start from
the immutable observation store, then trains batters-faced, pitch-count, outs,
and early-exit models with chronological train/calibration/test blocks.

```bash
.venv/bin/python train_pitcher_ml.py
```

The exported `models/pitcher_workload_registry.json` is loaded by the normal
dependency-free server. Its BF/pitch/outs estimates and the resulting K
distribution are displayed as **shadow only** and cannot change rankings or a
research decision. Extreme disagreements with recent-start workload are
limited and explicitly shown as a red risk warning.

## Save and settle predictions

Save the projection visible for a game/player:

```bash
curl -X POST http://127.0.0.1:8000/api/predictions \
  -H 'Content-Type: application/json' \
  -d '{"game_pk":823994,"player_id":123456}'
```

Confirmed-lineup predictions are saved automatically the first time slate or matchup research loads. After a game becomes final, the server settles strikeouts, batters faced, pitches, outs, runs, earned runs, hits, and walks from the free MLB box score. Predictions and results are append-only and cannot be overwritten.

## Backtest

Evaluate all settled predictions:

```bash
python3 backtest.py --json backtest-report.json
```

Optional chronological range:

```bash
python3 backtest.py --start 2026-05-01T00:00:00Z --end 2026-07-01T00:00:00Z
```

The evaluator excludes predictions whose `as_of` is not earlier than scheduled first pitch. It reports K MAE/RMSE plus separate batters-faced, pitch-count, outs, and K-rate error. Probability calibration and market comparisons are added when a timestamped line/price exists.

An empty report is expected until immutable predictions are saved and settled. Historical season aggregates are not silently treated as valid historical predictions.

## Metric glossary

- **Projection:** model’s expected result, not a guarantee.
- **Prediction interval:** range intended to convey outcome uncertainty.
- **Probability:** model-estimated chance of the listed side at the supplied line.
- **Fair odds:** American price corresponding to model probability before a safety margin.
- **Market implied probability:** probability encoded by the offered odds, including vig.
- **No-vig probability:** the two sides normalized to remove the book’s margin.
- **Estimated EV:** expected profit per unit using the actual price. It is unavailable for pick'em legs or incomplete prices.
- **Edge:** model probability minus comparable market probability. It should be interpreted only after calibration.
- **ROI:** realized profit divided by units risked in historical evaluation.
- **CLV:** movement from the selected price toward the designated closing market, compared only at the same line here.
- **Arsenal coverage:** how much of the starter’s primary mix has reliable hitter evidence.
- **Bullpen readiness:** a transparent workload score based on recent pitches and consecutive days; it is not confirmed availability.
- **Modeled bullpen mix:** the share of the estimated relief appearance mix backed by a saved pitcher arsenal. Unmodeled share remains league-average uncertainty instead of being redistributed to known pitchers.
- **SP/BP exposure:** expected plate appearances against the starter and bullpen, estimated from batting order and the starter’s projected batters faced.

## Data sources and limitations

- MLB Stats API/Gameday supplies schedules, probable pitchers, lineups, pitch type, velocity, coarse coordinates, handedness, and outcomes. It is a public web service without an application uptime guarantee.
- ESPN’s public scoreboard provides best-effort game totals and moneylines, not player props.
- The final pitch of a plate appearance is used to group the PA outcome by pitch type. This is descriptive attribution, not proof that the pitch caused the outcome.
- Pitch-cell SLG, ISO, and XBH are descriptive final-pitch outcomes, not hit or total-base prop probabilities.
- The displayed barrel value is explicitly a conservative proxy, not MLB’s official Barrel metric. No xwOBA is fabricated.
- Weather is labeled as the MLB feed value, not a dedicated forecast.
- Park geometry and umpire history remain descriptive in model v4. The raw umpire K% is excluded because it is confounded by assigned players. Hitter pitch cells and their priors use the opposing pitcher's throwing-hand split, with clearly labeled all-hand fallback when the split is too sparse.
- Hitter contact research is shrinkage-aware. Separate hit/XBH/HR/K challengers run in shadow with calibrated out-of-time probabilities, but they do not affect the empirical rankings until a target clears its promotion gate. Bullpen exposure remains a weighted descriptive blend, not a calibrated simulation of manager decisions.

Use this tool as research support.
