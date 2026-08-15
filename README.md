# Diamond Intel

Diamond Intel is a local MLB matchup and player-prop research board. It combines MLB Gameday pitch feeds, confirmed lineups, pitcher workload, pitch-mix evidence, game context, and timestamped prop prices.

The current pitcher-strikeout model is an unvalidated research baseline. The application intentionally does not call a projection a profitable edge until walk-forward results demonstrate calibration and value.

## Architecture

- `sync_matchup_data.py`: downloads and caches MLB Gameday feeds, builds current matchup aggregates, and records immutable player-game observations.
- `analytics_store.py`: SQLite schema plus numbered, non-destructive migrations.
- `modeling.py`: empirical-Bayes shrinkage, arsenal coverage, bullpen readiness/exposure, pitcher-K workload, count distributions, and market math.
- `prediction_store.py`: append-only pregame, market, prediction, and result records.
- `market_data.py`: sportsbook/pick'em CSV and manual market import.
- `backtest.py`: chronological evaluation, calibration, distribution comparison, ROI, and same-line CLV.
- `server.py`: localhost API and static server.
- `index.html`: responsive daily prop board and detailed pitch research. It displays server-computed statistics; model formulas no longer live in browser JavaScript.
- `diamond_intel.db`: the existing local database. Initialization migrates it in place and does not delete existing tables.

The unused PrizePicks Node proxy was removed. Market providers now enter through the explicit adapter/import boundary rather than an undocumented upstream endpoint.

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

Starting the server or a CLI calls `analytics_store.initialize()`. Migrations are recorded in `schema_migrations`; the current version is 8. Existing research tables and rows are retained. Database triggers reject updates and deletes on observations, pregame snapshots, bullpen snapshots, market snapshots, workload overrides, predictions, and results.

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

Sync today’s slate:

```bash
python3 sync_statcast.py --all
```

The sync reuses `.gameday_cache`. It enforces a strict `game_date < target_game_date` historical cutoff. Team schedules are supplemented with player-specific game logs, which preserves games played for former teams after a trade. Completed feeds create game-level pitcher and batter observations, including starter classification and handedness.

Pitch-matchup cells include descriptive AVG, SLG, ISO, and XBH results from the
same final-pitch sample. The matchup read also reweights those results to the
probable starter's throwing hand and historical three-state count / 3×3 zone
mix; sparse split cells shrink back to the hitter's same-pitch, same-velocity
sample. After this update, run `python3 sync_statcast.py --all` once to populate
the added power and count/hand/zone context fields. Until then, the board
explicitly marks power metrics as pending.

Each game card also displays ESPN's best-effort public total and moneyline
favorite when available. Hitter ranking includes a clearly shown, capped market
context adjustment: a higher/lower total can move the fit by at most 3.5 points,
and a favorite gets a further 1.5-point team-scoring-context adjustment. Pitch,
power, lineup, and sample evidence remain the primary inputs; missing markets
produce no adjustment.

The same sync also creates a timestamped bullpen snapshot for each team. It
identifies active relief candidates, estimates readiness from pitches thrown
today and over the prior two days, infers a broad short/long-relief role, and
builds pitch profiles from the same strictly pregame Gameday history. These are
workload estimates—not official availability or manager intent.

For an old game, the target game itself and later dates are excluded. Cached feeds are raw source responses; prediction features still apply the as-of cutoff.

## Import prop markets

The import format is demonstrated in `examples/markets.csv`. Required columns are:

- `captured_at`: ISO-8601 timestamp with timezone
- `provider`
- `platform_type`: `sportsbook` or `pickem`
- `player_name`
- `prop_type`
- `line`

Recommended identifiers are `game_pk` and `player_id`. Sportsbook rows should contain both `over_price` and `under_price`. Pick'em payout structure belongs in `payout_json`; it is never converted into fictional single-leg odds.

```bash
python3 market_data.py import examples/markets.csv
```

Manual sportsbook example:

```bash
python3 market_data.py add \
  --captured-at 2026-08-12T18:00:00-04:00 \
  --provider "Example Book" \
  --platform-type sportsbook \
  --game-pk 823994 \
  --player-id 123456 \
  --player-name "Example Pitcher" \
  --line 5.5 \
  --over-price -110 \
  --under-price -110
```

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
- Prop markets must be imported or entered manually unless a new provider adapter is implemented.
- The final pitch of a plate appearance is used to group the PA outcome by pitch type. This is descriptive attribution, not proof that the pitch caused the outcome.
- Pitch-cell SLG, ISO, and XBH are descriptive final-pitch outcomes, not hit or total-base prop probabilities.
- The displayed barrel value is explicitly a conservative proxy, not MLB’s official Barrel metric. No xwOBA is fabricated.
- Weather is labeled as the MLB feed value, not a dedicated forecast.
- Park geometry and umpire history remain descriptive in model v4. The raw umpire K% is excluded because it is confounded by assigned players. Hitter pitch cells use the opposing pitcher's throwing-hand split; batter-side and full platoon adjustments remain future work.
- Hitter contact research is shrinkage-aware but is not yet a prop probability model. Bullpen exposure is a weighted descriptive blend, not a calibrated simulation of manager decisions. Batter strikeouts, hits, total bases, and rare-event home-run forecasts require their own validated targets.

## Adding a provider or prop

Normalize a provider response to the `market_snapshots` fields and call `add_market_snapshot`; keep network/authentication details outside model code. A new prop needs a versioned server-side target model, immutable prediction schema compatibility, settlement mapping, walk-forward evaluation, tests, UI labeling, and a model-card update before any action label is enabled.

Use this tool as research support, shop prices independently, and size risk conservatively. Short historical success does not prove a durable betting advantage.
