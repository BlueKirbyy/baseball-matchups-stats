"""Walk-forward evaluation for immutable Diamond Intel predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from math import log, sqrt

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
    for dimension in ("confidence", "season", "month", "lineup_status", "pitcher_throws"):
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
    report = {
        "status": "ok",
        "count": count,
        "mae": sum(abs(row["error"]) for row in settled) / count,
        "rmse": sqrt(sum(row["error"] ** 2 for row in settled) / count),
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
      SELECT p.*, r.actual_value, m.line, m.over_price, m.under_price,
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate immutable pitcher-prop predictions chronologically")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--json", dest="json_path", help="Write the machine-readable report")
    args = parser.parse_args()
    initialize()
    rows = load_rows(start=args.start, end=args.end)
    report = evaluate(rows)
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
