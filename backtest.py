"""Walk-forward evaluation for immutable Diamond Intel predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from math import exp, log, sqrt

from analytics_store import connect, initialize
from modeling import distribution_summary, no_vig_probabilities


def chronological_splits(rows, minimum_training=30):
    """Yield expanding-window train/test pairs; never mixes future into training."""
    ordered = sorted(rows, key=lambda row: (row["as_of"], row.get("prediction_id", 0)))
    for index in range(minimum_training, len(ordered)):
        yield ordered[:index], ordered[index]


def calibration_rows(rows, bucket_width=0.1):
    buckets = defaultdict(list)
    for row in rows:
        probability = row.get("probability_over")
        outcome = row.get("over_outcome")
        if probability is None or outcome is None:
            continue
        lower = min(0.9, int(float(probability) / bucket_width) * bucket_width)
        buckets[lower].append((float(probability), float(outcome)))
    return [
        {
            "bucket": f"{lower:.1f}-{lower + bucket_width:.1f}",
            "count": len(values),
            "mean_probability": sum(value[0] for value in values) / len(values),
            "observed_rate": sum(value[1] for value in values) / len(values),
        }
        for lower, values in sorted(buckets.items())
    ]


def maximum_drawdown(profits):
    bankroll = peak = drawdown = 0.0
    for profit in profits:
        bankroll += profit
        peak = max(peak, bankroll)
        drawdown = max(drawdown, peak - bankroll)
    return drawdown


def evaluate_hitter_recent_form(rows):
    """Compare frozen hitter scores with and without the recent-form feature.

    The opportunity model is descriptive rather than calibrated.  The Brier
    values below therefore compare two versions of the same monotonic score
    transform; they are promotion evidence, not betting-probability claims.
    """
    outcome_key = {
        "hit": "one_plus_hit", "total_bases": "one_plus_total_base",
        "home_run": "home_run", "runs_rbi": "run_or_rbi",
        "overall": "one_plus_hit",
    }
    observations = []
    for raw in rows:
        row = dict(raw)
        outcome = row.get("outcome")
        actual = row.get("actual")
        if outcome not in outcome_key or actual is None:
            continue
        with_score = float(row.get("score") or 0.0)
        recent = float(row.get("recent_form_adjustment") or 0.0)
        without_score = with_score - recent
        # Center the research score on the favorable threshold. This is only a
        # common comparison transform and is deliberately not shown as a prop
        # probability in the product.
        probability_with = 1.0 / (1.0 + exp(-(with_score - .25)))
        probability_without = 1.0 / (1.0 + exp(-(without_score - .25)))
        observations.append({
            **row, "actual": float(actual),
            "probability_with": probability_with,
            "probability_without": probability_without,
        })
    if not observations:
        return {
            "status": "collecting", "count": 0,
            "note": "Pregame hitter snapshots and settled hitter games are still accumulating.",
        }

    def summarize(items):
        if not items:
            return {"count": 0, "brier_with_form": None, "brier_without_form": None, "brier_delta": None}
        with_brier = sum((item["probability_with"] - item["actual"]) ** 2 for item in items) / len(items)
        without_brier = sum((item["probability_without"] - item["actual"]) ** 2 for item in items) / len(items)
        return {
            "count": len(items), "brier_with_form": with_brier,
            "brier_without_form": without_brier,
            "brier_delta": with_brier - without_brier,
            "recent_form_helped": with_brier < without_brier,
        }

    def grouped(key):
        groups = defaultdict(list)
        for item in observations:
            groups[str(item.get(key) or "unknown")].append(item)
        return {name: summarize(items) for name, items in sorted(groups.items())}

    report = summarize(observations)
    report.update({
        "status": "ok",
        "by_outcome": grouped("outcome"),
        "by_confidence": grouped("confidence"),
        "by_coverage_band": grouped("coverage_band"),
        "by_recent_form_pa_band": grouped("recent_form_pa_band"),
        "by_window_days": grouped("window_days"),
        "outside_extreme_hot_streaks": summarize([
            item for item in observations if abs(float(item.get("recent_form_score") or 0.0)) < .35
        ]),
        "cap_review": {
            "observed_max_adjustment": max(abs(float(item.get("recent_form_adjustment") or 0.0)) for item in observations),
            "current_cap": .10,
            "note": "Reduce the cap only after enough walk-forward rows show worse out-of-sample Brier score with form.",
        },
        "validation": "chronological frozen pregame snapshots only; descriptive score comparison, not calibrated wagering probabilities",
    })
    return report


def evaluate(rows):
    settled = []
    for raw in rows:
        row = dict(raw)
        actual = row.get("actual_value")
        if actual is None:
            continue
        projection = float(row["projection"])
        actual = float(actual)
        row["error"] = projection - actual
        row["projection_band"] = (
            "under 4" if projection < 4.0 else
            "4.0-4.9" if projection < 5.0 else
            "5.0-5.9" if projection < 6.0 else
            "6.0-6.9" if projection < 7.0 else "7.0+"
        )
        if row.get("line") is not None:
            line = float(row["line"])
            row["over_outcome"] = 1 if actual > line else 0 if actual < line else None
        settled.append(row)
    count = len(settled)
    if not count:
        return {"status": "no_settled_predictions", "count": 0, "note": "Settle immutable predictions before evaluating."}
    probabilities = [row for row in settled if row.get("probability_over") is not None and row.get("over_outcome") is not None]
    brier = sum((float(row["probability_over"]) - row["over_outcome"]) ** 2 for row in probabilities) / len(probabilities) if probabilities else None
    log_loss = None
    if probabilities:
        losses = []
        for row in probabilities:
            probability = max(1e-9, min(1 - 1e-9, float(row["probability_over"])))
            outcome = row["over_outcome"]
            losses.append(-(outcome * log(probability) + (1 - outcome) * log(1 - probability)))
        log_loss = sum(losses) / len(losses)
    bets, profits = [], []
    for row in probabilities:
        ev_over = row.get("expected_value_over")
        price = row.get("over_price")
        if ev_over is None or price is None or float(ev_over) <= 0:
            continue
        profit = (float(price) / 100 if float(price) > 0 else 100 / -float(price)) if row["over_outcome"] else -1.0
        bets.append(row)
        profits.append(profit)
    grouped = {}
    for dimension in (
        "model_version", "projection_band", "confidence", "season", "month",
        "lineup_status", "pitcher_throws",
    ):
        groups = defaultdict(list)
        for row in settled:
            groups[str(row.get(dimension) or "unknown")].append(abs(row["error"]))
        grouped[dimension] = {key: {"count": len(values), "mae": sum(values) / len(values)} for key, values in sorted(groups.items())}
    candidate_scores = {}
    for distribution in ("poisson", "negative_binomial"):
        candidate_rows = []
        for row in settled:
            if row.get("line") is None or row.get("over_outcome") is None:
                continue
            probability = distribution_summary(float(row["projection"]), float(row["line"]), distribution).get("probability_over")
            candidate_rows.append((probability, row["over_outcome"]))
        if candidate_rows:
            candidate_scores[distribution] = {
                "count": len(candidate_rows),
                "brier": sum((probability - outcome) ** 2 for probability, outcome in candidate_rows) / len(candidate_rows),
            }
    rolling_errors = []
    ordered = sorted(settled, key=lambda row: row.get("as_of", ""))
    history = []
    for row in ordered:
        if history:
            baseline = sum(history) / len(history)
            rolling_errors.append(abs(baseline - float(row["actual_value"])))
        history.append(float(row["actual_value"]))
    market_rows = []
    for row in probabilities:
        no_vig = no_vig_probabilities(row.get("over_price"), row.get("under_price"))
        if no_vig:
            market_rows.append((no_vig["over"], row["over_outcome"]))
    closing_differences = []
    for row in probabilities:
        bet_no_vig = no_vig_probabilities(row.get("over_price"), row.get("under_price"))
        close_no_vig = no_vig_probabilities(row.get("closing_over_price"), row.get("closing_under_price"))
        if bet_no_vig and close_no_vig and row.get("closing_line") == row.get("line"):
            closing_differences.append(close_no_vig["over"] - bet_no_vig["over"])
    def component_error(projected_key, actual_key):
        pairs = [
            (float(row[projected_key]), float(row[actual_key]))
            for row in settled
            if row.get(projected_key) is not None and row.get(actual_key) is not None
        ]
        return {
            "count": len(pairs),
            "mae": sum(abs(projected - actual) for projected, actual in pairs) / len(pairs) if pairs else None,
            "bias": sum(projected - actual for projected, actual in pairs) / len(pairs) if pairs else None,
        }

    k_rate_pairs = [
        (float(row["projected_k_rate"]), float(row["actual_value"]) / float(row["actual_batters_faced"]))
        for row in settled
        if row.get("projected_k_rate") is not None and row.get("actual_batters_faced") not in (None, 0)
    ]
    report = {
        "status": "ok",
        "count": count,
        "mae": sum(abs(row["error"]) for row in settled) / count,
        "rmse": sqrt(sum(row["error"] ** 2 for row in settled) / count),
        "components": {
            "workload_batters_faced": component_error("projected_batters_faced", "actual_batters_faced"),
            "pitch_count": component_error("projected_pitches", "actual_pitches"),
            "outs": component_error("projected_outs", "actual_outs"),
            "k_rate": {
                "count": len(k_rate_pairs),
                "mae": sum(abs(projected - actual) for projected, actual in k_rate_pairs) / len(k_rate_pairs) if k_rate_pairs else None,
                "bias": sum(projected - actual for projected, actual in k_rate_pairs) / len(k_rate_pairs) if k_rate_pairs else None,
            },
        },
        "brier": brier,
        "log_loss": log_loss,
        "distribution_comparison": candidate_scores,
        "recommended_distribution": min(candidate_scores, key=lambda name: candidate_scores[name]["brier"]) if candidate_scores else None,
        "calibration": calibration_rows(probabilities),
        "groups": grouped,
        "baselines": {
            "rolling_actual_mean_mae": sum(rolling_errors) / len(rolling_errors) if rolling_errors else None,
            "market_no_vig_brier": sum((probability - outcome) ** 2 for probability, outcome in market_rows) / len(market_rows) if market_rows else None,
        },
        "bets": {
            "count": len(bets),
            "profit_units": sum(profits),
            "roi": sum(profits) / len(bets) if bets else None,
            "win_rate": sum(row["over_outcome"] for row in bets) / len(bets) if bets else None,
            "average_over_price": sum(float(row["over_price"]) for row in bets) / len(bets) if bets else None,
            "maximum_drawdown_units": maximum_drawdown(profits),
        },
        "clv": {
            "available": bool(closing_differences),
            "count": len(closing_differences),
            "mean_probability_improvement": sum(closing_differences) / len(closing_differences) if closing_differences else None,
            "note": "Compared only when bet-time and closing snapshots use the same line.",
        },
        "validation": "out-of-sample records only; this report does not itself establish profitability",
    }
    return report


def load_rows(db_path=None, start=None, end=None):
    clauses = ["r.actual_value IS NOT NULL"]
    params = []
    if start:
        clauses.append("p.as_of>=?")
        params.append(start)
    if end:
        clauses.append("p.as_of<?")
        params.append(end)
    query = f"""
      SELECT p.*, r.actual_value, r.actual_batters_faced, r.actual_pitches, r.actual_outs,
             r.actual_runs, r.actual_earned_runs, r.actual_hits, r.actual_walks,
             json_extract(p.inputs_json, '$.expected_batters_faced') AS projected_batters_faced,
             json_extract(p.inputs_json, '$.expected_pitches') AS projected_pitches,
             json_extract(p.inputs_json, '$.performance_outlook.expected_outs') AS projected_outs,
             json_extract(p.inputs_json, '$.k_rate') AS projected_k_rate,
             m.line, m.over_price, m.under_price,
             closing.line AS closing_line, closing.over_price AS closing_over_price,
             closing.under_price AS closing_under_price,
             substr(p.as_of, 1, 4) AS season,
             substr(p.as_of, 1, 7) AS month,
             CASE WHEN p.lineup_confirmed=1 THEN 'confirmed' ELSE 'unconfirmed' END AS lineup_status,
             json_extract(p.inputs_json, '$.pitcher_throws') AS pitcher_throws
      FROM model_predictions p
      JOIN prediction_results r ON r.prediction_id=p.prediction_id
      LEFT JOIN market_snapshots m ON m.market_snapshot_id=p.market_snapshot_id
      LEFT JOIN market_snapshots closing ON closing.market_snapshot_id=(
        SELECT candidate.market_snapshot_id FROM market_snapshots candidate
        WHERE candidate.game_pk=p.game_pk AND candidate.player_id=p.player_id
          AND candidate.prop_type=p.prop_type AND candidate.is_closing=1
        ORDER BY candidate.captured_at DESC LIMIT 1
      )
      WHERE {' AND '.join(clauses)}
        AND (p.scheduled_start IS NULL OR p.as_of < p.scheduled_start)
      ORDER BY p.as_of, p.prediction_id
    """
    with connect(db_path) as db:
        return [dict(row) for row in db.execute(query, params)]


def load_hitter_recent_form_rows(db_path=None, start=None, end=None):
    """Expand immutable pregame hitter snapshots into evaluation rows."""
    clauses = ["s.target='hitter_spotlight'", "o.target_group='hitter_game'", "s.captured_at<s.scheduled_start"]
    params = []
    if start:
        clauses.append("s.captured_at>=?")
        params.append(start)
    if end:
        clauses.append("s.captured_at<?")
        params.append(end)
    with connect(db_path) as db:
        joined = db.execute(
            f"""SELECT s.game_pk, s.player_id, s.captured_at, s.scheduled_start,
                       s.features_json, o.outcomes_json
                  FROM ml_feature_snapshots s
                  JOIN settled_player_outcomes o
                    ON o.game_pk=s.game_pk AND o.player_id=s.player_id
                 WHERE {' AND '.join(clauses)}
                 ORDER BY s.scheduled_start, s.game_pk, s.player_id""",
            params,
        ).fetchall()
    rows = []
    for joined_row in joined:
        features = json.loads(joined_row["features_json"])
        outcomes = json.loads(joined_row["outcomes_json"])
        form = features.get("recent_form") or {}
        items = ((features.get("opportunities") or {}).get("items") or {})
        for outcome, opportunity in items.items():
            actual = {
                "hit": outcomes.get("one_plus_hit"),
                "total_bases": outcomes.get("one_plus_total_base"),
                "home_run": int((outcomes.get("home_runs") or 0) > 0),
                "runs_rbi": outcomes.get("run_or_rbi"),
                "overall": outcomes.get("one_plus_hit"),
            }.get(outcome)
            coverage = float((opportunity.get("requirements") or {}).get("coverage") or 0.0)
            actual_coverage = float(opportunity.get("coverage") or features.get("coverage") or 0.0)
            form_pa = int(form.get("pa") or 0)
            rows.append({
                "game_pk": joined_row["game_pk"], "player_id": joined_row["player_id"],
                "outcome": outcome, "actual": actual,
                "score": opportunity.get("score"),
                "recent_form_adjustment": opportunity.get("recent_form_adjustment"),
                "recent_form_score": form.get("score"),
                "confidence": opportunity.get("confidence"),
                "coverage_band": (
                    "below_gate" if actual_coverage < coverage else
                    "25-34%" if actual_coverage < .35 else
                    "35-49%" if actual_coverage < .50 else "50%+"
                ),
                "recent_form_pa_band": "<20" if form_pa < 20 else "20-34" if form_pa < 35 else "35+",
                "window_days": form.get("window_days"),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate immutable pitcher-prop predictions chronologically")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--json", dest="json_path", help="Write the machine-readable report")
    args = parser.parse_args()
    initialize()
    rows = load_rows(start=args.start, end=args.end)
    report = evaluate(rows)
    report["hitter_recent_form"] = evaluate_hitter_recent_form(
        load_hitter_recent_form_rows(start=args.start, end=args.end)
    )
    with connect() as db:
        versions = db.execute("SELECT model_version, feature_version FROM model_predictions ORDER BY prediction_id DESC LIMIT 1").fetchone()
        if versions:
            db.execute(
                "INSERT INTO model_evaluations(created_at, model_version, feature_version, evaluation_start, evaluation_end, report_json) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), versions[0], versions[1], args.start, args.end, json.dumps(report, sort_keys=True)),
            )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
