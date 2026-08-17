"""Persistence helpers for immutable market and model records."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from analytics_store import connect


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def is_before_start(captured_at, scheduled_start):
    if not scheduled_start:
        return False
    try:
        return parse_timestamp(captured_at) < parse_timestamp(scheduled_start)
    except (TypeError, ValueError):
        return False


def record_game(game, observed_at=None, db_path=None):
    observed_at = observed_at or utc_now()
    with connect(db_path) as db:
        db.execute(
            """INSERT INTO games(
                 game_pk, scheduled_start, official_date, away_team_id,
                 away_team_name, home_team_id, home_team_name, venue_name,
                 first_seen_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(game_pk) DO UPDATE SET
                 scheduled_start=excluded.scheduled_start,
                 official_date=excluded.official_date,
                 away_team_id=excluded.away_team_id,
                 away_team_name=excluded.away_team_name,
                 home_team_id=excluded.home_team_id,
                 home_team_name=excluded.home_team_name,
                 venue_name=excluded.venue_name,
                 updated_at=excluded.updated_at""",
            (
                game["game_pk"], game.get("scheduled_start"), game.get("official_date"),
                game.get("away_team_id"), game.get("away_team_name"),
                game.get("home_team_id"), game.get("home_team_name"),
                game.get("venue_name"), observed_at, observed_at,
            ),
        )


def record_pregame_snapshot(game_pk, captured_at, scheduled_start, probable, lineups, db_path=None):
    with connect(db_path) as db:
        cursor = db.execute(
            """INSERT INTO pregame_snapshots(
                 game_pk, captured_at, scheduled_start, away_probable_pitcher_id,
                 home_probable_pitcher_id, away_lineup_json, home_lineup_json,
                 away_lineup_confirmed, home_lineup_confirmed
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_pk, captured_at, scheduled_start,
                (probable.get("away") or {}).get("id"), (probable.get("home") or {}).get("id"),
                json.dumps(lineups.get("away") or []), json.dumps(lineups.get("home") or []),
                int(bool(lineups.get("away"))), int(bool(lineups.get("home"))),
            ),
        )
        return cursor.lastrowid


def record_ml_feature_snapshot(game_pk, captured_at, scheduled_start, player_id,
                               player_name, target, model_version,
                               feature_version, features, lineup_confirmed=False,
                               source="diamond-intel-pregame", db_path=None):
    """Save the first reproducible pregame feature vector for a player/game.

    Browser refreshes are intentionally idempotent: the earliest snapshot for
    a model feature version wins and can never be edited after first pitch.
    """
    if not is_before_start(captured_at, scheduled_start):
        return None
    with connect(db_path) as db:
        cursor = db.execute(
            """INSERT OR IGNORE INTO ml_feature_snapshots(
                 captured_at, scheduled_start, game_pk, player_id, player_name,
                 target, model_version, feature_version, lineup_confirmed,
                 features_json, source
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                captured_at, scheduled_start, int(game_pk), int(player_id),
                player_name, target, model_version, feature_version,
                int(bool(lineup_confirmed)), json.dumps(features, sort_keys=True),
                source,
            ),
        )
        if cursor.rowcount:
            return cursor.lastrowid
        row = db.execute(
            """SELECT ml_snapshot_id FROM ml_feature_snapshots
               WHERE game_pk=? AND player_id=? AND target=? AND feature_version=?""",
            (int(game_pk), int(player_id), target, feature_version),
        ).fetchone()
        return row["ml_snapshot_id"] if row else None


def record_settled_player_outcome(game_pk, player_id, player_name,
                                  target_group, outcomes, game_date=None,
                                  source="mlb-gameday", settled_at=None,
                                  db_path=None):
    """Append a normalized final result shared by future ML pipelines."""
    with connect(db_path) as db:
        cursor = db.execute(
            """INSERT OR IGNORE INTO settled_player_outcomes(
                 game_pk, game_date, player_id, player_name, target_group,
                 outcomes_json, settled_at, source
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(game_pk), game_date, int(player_id), player_name,
                target_group, json.dumps(outcomes, sort_keys=True),
                settled_at or utc_now(), source,
            ),
        )
        return cursor.rowcount


def record_bullpen_snapshot(game_pk, team_id, team_name, scheduled_start,
                            captured_at, relievers, db_path=None):
    """Append one reproducible pregame bullpen-readiness snapshot."""
    if not is_before_start(captured_at, scheduled_start):
        return []
    identifiers = []
    with connect(db_path) as db:
        for reliever in relievers:
            cursor = db.execute(
                """INSERT INTO bullpen_snapshots(
                     game_pk, team_id, team_name, scheduled_start, captured_at,
                     player_id, player_name, throws, role, readiness_score,
                     readiness_status, mix_weight, pitches_today,
                     pitches_yesterday, pitches_two_days_ago,
                     three_day_pitches, consecutive_days, days_rest,
                     recent_appearances, recent_starts, arsenal_available
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    game_pk, team_id, team_name, scheduled_start, captured_at,
                    reliever["player_id"], reliever["player_name"], reliever.get("throws"),
                    reliever["role"], reliever["readiness_score"], reliever["readiness_status"],
                    reliever["mix_weight"], reliever.get("pitches_today", 0),
                    reliever.get("pitches_yesterday", 0), reliever.get("pitches_two_days_ago", 0),
                    reliever.get("three_day_pitches", 0), reliever.get("consecutive_days", 0),
                    reliever.get("days_rest"), reliever.get("recent_appearances", 0),
                    reliever.get("recent_starts", 0), int(bool(reliever.get("arsenal_available"))),
                ),
            )
            identifiers.append(cursor.lastrowid)
    return identifiers


def latest_bullpen_snapshot(game_pk, team_id, as_of=None, db_path=None):
    """Return every reliever from the latest eligible snapshot as one unit."""
    params = [game_pk, team_id]
    cutoff = ""
    if as_of:
        cutoff = " AND captured_at<=?"
        params.append(as_of)
    with connect(db_path) as db:
        captured = db.execute(
            f"""SELECT MAX(captured_at) AS captured_at FROM bullpen_snapshots
                 WHERE game_pk=? AND team_id=?{cutoff}""",
            params,
        ).fetchone()
        if not captured or not captured["captured_at"]:
            return []
        return [dict(row) for row in db.execute(
            """SELECT * FROM bullpen_snapshots
               WHERE game_pk=? AND team_id=? AND captured_at=?
               ORDER BY mix_weight DESC, player_name""",
            (game_pk, team_id, captured["captured_at"]),
        )]


def add_market_snapshot(row, db_path=None):
    now = utc_now()
    with connect(db_path) as db:
        cursor = db.execute(
            """INSERT INTO market_snapshots(
                 captured_at, provider, platform_type, game_pk, player_id,
                 player_name, prop_type, line, over_price, under_price,
                 payout_json, is_closing, source, imported_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.get("captured_at") or now, row["provider"], row.get("platform_type", "sportsbook"),
                row.get("game_pk"), row.get("player_id"), row["player_name"], row["prop_type"],
                float(row["line"]), row.get("over_price"), row.get("under_price"),
                json.dumps(row.get("payout")) if row.get("payout") is not None else row.get("payout_json"),
                int(bool(row.get("is_closing"))), row.get("source", "manual"), now,
            ),
        )
        return cursor.lastrowid


def latest_markets(game_pk, prop_type="pitcher_strikeouts", as_of=None, player_id=None, player_name=None, db_path=None):
    as_of = as_of or utc_now()
    player_clause = "1=1"
    player_params = []
    if player_id is not None:
        player_clause = "(candidate.player_id=? OR (candidate.player_id IS NULL AND lower(candidate.player_name)=lower(?)))"
        player_params = [player_id, player_name or ""]
    query = f"""
      SELECT candidate.* FROM market_snapshots candidate
      WHERE candidate.game_pk=? AND candidate.prop_type=? AND candidate.captured_at<=?
        AND {player_clause}
        AND candidate.market_snapshot_id=(
          SELECT latest.market_snapshot_id FROM market_snapshots latest
          WHERE latest.game_pk=candidate.game_pk AND latest.prop_type=candidate.prop_type
            AND latest.provider=candidate.provider AND latest.platform_type=candidate.platform_type
            AND (latest.player_id=candidate.player_id OR (
              latest.player_id IS NULL AND candidate.player_id IS NULL
              AND lower(latest.player_name)=lower(candidate.player_name)
            ))
            AND latest.captured_at<=?
          ORDER BY latest.captured_at DESC, latest.market_snapshot_id DESC LIMIT 1
        )
      ORDER BY candidate.provider, candidate.platform_type
    """
    params = [game_pk, prop_type, as_of, *player_params, as_of]
    with connect(db_path) as db:
        rows = [dict(row) for row in db.execute(query, params)]
    chosen = {}
    for row in rows:
        key = (row["provider"], row["platform_type"])
        current = chosen.get(key)
        row_exact = player_id is not None and row.get("player_id") == player_id
        current_exact = current and player_id is not None and current.get("player_id") == player_id
        if current is None or (row_exact and not current_exact) or (row_exact == current_exact and row["captured_at"] > current["captured_at"]):
            chosen[key] = row
    return list(chosen.values())


def add_workload_override(row, db_path=None):
    """Append a pregame manual pitch cap without replacing prior records."""
    now = utc_now()
    pitch_limit = float(row["pitch_limit"])
    if not 20 <= pitch_limit <= 130:
        raise ValueError("pitch_limit must be between 20 and 130")
    with connect(db_path) as db:
        cursor = db.execute(
            """INSERT INTO workload_overrides(
                 captured_at, game_pk, player_id, player_name, pitch_limit,
                 source, note, imported_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.get("captured_at") or now, int(row["game_pk"]), int(row["player_id"]),
                row["player_name"], pitch_limit, row.get("source", "manual"),
                row.get("note"), now,
            ),
        )
        return cursor.lastrowid


def latest_workload_override(game_pk, player_id, as_of=None, db_path=None):
    """Return the newest pitch-cap record available at the requested time."""
    as_of = as_of or utc_now()
    with connect(db_path) as db:
        row = db.execute(
            """SELECT * FROM workload_overrides
               WHERE game_pk=? AND player_id=? AND captured_at<=?
               ORDER BY captured_at DESC, workload_override_id DESC LIMIT 1""",
            (int(game_pk), int(player_id), as_of),
        ).fetchone()
    return dict(row) if row else None


def save_prediction(game_pk, prediction, market=None, data_freshness_seconds=None, db_path=None):
    """Append a prediction; intentionally no update path exists."""
    market = market or prediction.get("market") or {}
    if data_freshness_seconds is None:
        data_freshness_seconds = prediction.get("data_freshness_seconds")
    if not is_before_start(prediction["as_of"], prediction.get("scheduled_start")):
        raise ValueError("prediction as_of must be earlier than scheduled_start")
    with connect(db_path) as db:
        cursor = db.execute(
            """INSERT INTO model_predictions(
                 created_at, as_of, game_pk, scheduled_start, player_id, player_name,
                 prop_type, model_version, feature_version, projection, median,
                 interval_low, interval_high, probability_over, probability_under,
                 fair_over_price, fair_under_price, market_snapshot_id, no_vig_over,
                 expected_value_over, expected_value_under, confidence,
                 arsenal_coverage, effective_sample_size, lineup_confirmed,
                 data_freshness_seconds, decision, factors_json, inputs_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                utc_now(), prediction["as_of"], game_pk, prediction.get("scheduled_start"),
                prediction["player_id"], prediction["player_name"], prediction["prop_type"],
                prediction["model_version"], prediction["feature_version"], prediction["projection"],
                prediction["median"], prediction["interval_low"], prediction["interval_high"],
                prediction.get("probability_over"), prediction.get("probability_under"),
                prediction.get("fair_over_price"), prediction.get("fair_under_price"),
                market.get("market_snapshot_id"), market.get("no_vig_over"),
                market.get("expected_value_over"), market.get("expected_value_under"),
                prediction["confidence"], prediction["arsenal_coverage"],
                prediction["effective_sample_size"], int(prediction["lineup_confirmed"]),
                data_freshness_seconds, prediction["decision"],
                json.dumps(prediction.get("factors", [])), json.dumps(prediction, sort_keys=True),
            ),
        )
        return cursor.lastrowid


def ensure_pregame_prediction(game_pk, prediction, market=None, db_path=None):
    """Capture one confirmed-lineup forecast per model/player/game.

    Matchup research is requested repeatedly by the slate and detail pages. A
    read should therefore be idempotent instead of manufacturing duplicate
    backtest rows every time the browser refreshes.
    """
    if not prediction.get("lineup_confirmed"):
        return {"status": "waiting_for_lineup", "prediction_id": None}
    if not is_before_start(prediction.get("as_of"), prediction.get("scheduled_start")):
        return {"status": "game_started", "prediction_id": None}
    with connect(db_path) as db:
        existing = db.execute(
            """SELECT prediction_id FROM model_predictions
               WHERE game_pk=? AND player_id=? AND prop_type=?
                 AND model_version=? AND feature_version=? AND lineup_confirmed=1
               ORDER BY prediction_id LIMIT 1""",
            (
                game_pk, prediction.get("player_id"), prediction.get("prop_type"),
                prediction.get("model_version"), prediction.get("feature_version"),
            ),
        ).fetchone()
    if existing:
        return {"status": "already_saved", "prediction_id": existing["prediction_id"]}
    return {
        "status": "saved",
        "prediction_id": save_prediction(game_pk, prediction, market, db_path=db_path),
    }


def pending_prediction_game_pks(db_path=None):
    """Games that have immutable forecasts but no recorded final result."""
    with connect(db_path) as db:
        rows = db.execute(
                """SELECT p.game_pk, MIN(p.scheduled_start) AS scheduled_start
                   FROM model_predictions p
                   LEFT JOIN prediction_results r ON r.prediction_id=p.prediction_id
                   WHERE r.prediction_id IS NULL
                   GROUP BY p.game_pk
                   ORDER BY p.game_pk"""
            ).fetchall()
    now = datetime.now(timezone.utc)
    pending = []
    for row in rows:
        try:
            start = parse_timestamp(row["scheduled_start"])
        except (TypeError, ValueError):
            continue
        if start <= now:
            pending.append(int(row["game_pk"]))
    return pending


def settle_prediction(prediction_id, actual_value, source="manual", settled_at=None, db_path=None, actuals=None):
    """Settle once; replacing a result would undermine the audit trail."""
    actuals = actuals or {}
    with connect(db_path) as db:
        db.execute(
            """INSERT INTO prediction_results(
                 prediction_id, settled_at, actual_value, result_source,
                 actual_batters_faced, actual_pitches, actual_outs, actual_runs,
                 actual_earned_runs, actual_hits, actual_walks
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id, settled_at or utc_now(), float(actual_value), source,
                actuals.get("batters_faced"), actuals.get("pitches"), actuals.get("outs"),
                actuals.get("runs"), actuals.get("earned_runs"), actuals.get("hits"),
                actuals.get("walks"),
            ),
        )


def settle_game_predictions(game_pk, pitching_lines, source="mlb-gameday", db_path=None):
    """Settle every saved starter projection found in a final MLB box score."""
    normalized = {int(player_id): line for player_id, line in (pitching_lines or {}).items()}
    settled = []
    with connect(db_path) as db:
        pending = db.execute(
            """SELECT p.prediction_id, p.player_id
               FROM model_predictions p
               LEFT JOIN prediction_results r ON r.prediction_id=p.prediction_id
               WHERE p.game_pk=? AND r.prediction_id IS NULL""",
            (game_pk,),
        ).fetchall()
        for prediction in pending:
            line = normalized.get(int(prediction["player_id"]))
            if not line or line.get("strikeouts") is None:
                continue
            db.execute(
                """INSERT INTO prediction_results(
                     prediction_id, settled_at, actual_value, result_source,
                     actual_batters_faced, actual_pitches, actual_outs, actual_runs,
                     actual_earned_runs, actual_hits, actual_walks
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction["prediction_id"], utc_now(), float(line["strikeouts"]), source,
                    line.get("batters_faced"), line.get("pitches"), line.get("outs"),
                    line.get("runs"), line.get("earned_runs"), line.get("hits"), line.get("walks"),
                ),
            )
            db.execute(
                """INSERT OR IGNORE INTO settled_player_outcomes(
                     game_pk, game_date, player_id, player_name, target_group,
                     outcomes_json, settled_at, source
                   )
                   SELECT ?, g.official_date, ?, p.player_name, 'pitcher_game',
                          ?, ?, ?
                     FROM model_predictions p
                     LEFT JOIN games g ON g.game_pk=p.game_pk
                    WHERE p.prediction_id=?""",
                (
                    int(game_pk), int(prediction["player_id"]),
                    json.dumps(line, sort_keys=True), utc_now(), source,
                    prediction["prediction_id"],
                ),
            )
            settled.append(prediction["prediction_id"])
    return settled


def prediction_tracking_summary(db_path=None):
    """Small transparent scorecard for the model's immutable forecasts."""
    with connect(db_path) as db:
        rows = db.execute(
            """SELECT p.projection, p.inputs_json, r.actual_value,
                      r.actual_batters_faced, r.actual_pitches, r.actual_outs
               FROM model_predictions p
               LEFT JOIN prediction_results r ON r.prediction_id=p.prediction_id
               ORDER BY p.as_of"""
        ).fetchall()
    settled = [row for row in rows if row["actual_value"] is not None]

    def mean_absolute(pairs):
        pairs = [(float(left), float(right)) for left, right in pairs if left is not None and right is not None]
        return sum(abs(left - right) for left, right in pairs) / len(pairs) if pairs else None

    inputs = []
    for row in settled:
        try:
            payload = json.loads(row["inputs_json"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        inputs.append((row, payload))
    return {
        "predictions": len(rows),
        "settled": len(settled),
        "k_mae": mean_absolute((row["projection"], row["actual_value"]) for row in settled),
        "bf_mae": mean_absolute((payload.get("expected_batters_faced"), row["actual_batters_faced"]) for row, payload in inputs),
        "pitch_count_mae": mean_absolute((payload.get("expected_pitches"), row["actual_pitches"]) for row, payload in inputs),
        "outs_mae": mean_absolute((payload.get("performance_outlook", {}).get("expected_outs"), row["actual_outs"]) for row, payload in inputs),
        "status": "tracking" if settled else "collecting",
    }
