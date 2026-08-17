#!/usr/bin/env python3
"""Build chronological hitter examples and train calibrated shadow challengers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from math import log
from pathlib import Path

from analytics_store import connect
from hitter_ml import (
    FEATURE_NAMES, FEATURE_VERSION, REGISTRY_PATH, TARGETS, TRAINING_DB_PATH,
    build_training_examples, predict_exported,
)


def matrix(db_path):
    rows = []
    with connect(db_path) as db:
        for row in db.execute(
            """SELECT game_date, game_pk, features_json, hit, extra_base_hit,
                      home_run, strikeout
               FROM hitter_ml_examples WHERE feature_version=?
               ORDER BY game_date, game_pk, pa_index""",
            (FEATURE_VERSION,),
        ):
            features = json.loads(row["features_json"])
            rows.append((
                row["game_date"], row["game_pk"],
                [float(features.get(name, 0.0)) for name in FEATURE_NAMES],
                {target: int(row[target]) for target in TARGETS},
            ))
    return rows


def chronological_split(rows):
    games = []
    for row in rows:
        key = (row[0], row[1])
        if not games or games[-1] != key:
            games.append(key)
    if len(games) < 30:
        raise RuntimeError("At least 30 completed games are required for chronological training.")
    train_end = games[max(1, int(len(games) * .65)) - 1]
    calibration_end = games[max(2, int(len(games) * .82)) - 1]
    train = [row for row in rows if (row[0], row[1]) <= train_end]
    calibration = [row for row in rows if train_end < (row[0], row[1]) <= calibration_end]
    test = [row for row in rows if (row[0], row[1]) > calibration_end]
    return train, calibration, test


def ece_score(y, probability, bins=10):
    total, error = len(y), 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, value in enumerate(probability) if low <= value < high or (index == bins - 1 and value == 1)]
        if members:
            predicted = sum(probability[i] for i in members) / len(members)
            observed = sum(y[i] for i in members) / len(members)
            error += len(members) / total * abs(predicted - observed)
    return error


def metrics(y, probability):
    clipped = [min(.999999, max(.000001, float(value))) for value in probability]
    brier = sum((value - actual) ** 2 for actual, value in zip(y, clipped)) / len(y)
    log_loss = -sum(actual * log(value) + (1 - actual) * log(1 - value) for actual, value in zip(y, clipped)) / len(y)
    return {
        "brier": round(brier, 6), "log_loss": round(log_loss, 6),
        "ece": round(ece_score(y, clipped), 6), "examples": len(y),
        "positives": int(sum(y)), "base_rate": round(sum(y) / len(y), 6),
    }


def calibrate(raw_calibration, y_calibration, raw_test):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    calibrator = LogisticRegression(C=1.0, solver="liblinear")
    calibrator.fit(np.asarray(raw_calibration).reshape(-1, 1), y_calibration)
    export = {
        "intercept": float(calibrator.intercept_[0]),
        "coefficient": float(calibrator.coef_[0][0]),
    }
    calibrated_raw = export["intercept"] + export["coefficient"] * np.asarray(raw_test)
    probability = 1.0 / (1.0 + np.exp(-np.clip(calibrated_raw, -35, 35)))
    return probability, export


def export_logistic(model, medians, means, scales, calibration):
    return {
        "kind": "logistic", "feature_names": list(FEATURE_NAMES),
        "medians": medians.tolist(), "means": means.tolist(), "scales": scales.tolist(),
        "intercept": float(model.intercept_[0]),
        "coefficients": model.coef_[0].tolist(), "calibration": calibration,
    }


def export_gradient(model, medians, calibration):
    trees = []
    for stage in model._predictors:
        nodes = stage[0].nodes
        trees.append({
            "left": nodes["left"].astype(int).tolist(),
            "right": nodes["right"].astype(int).tolist(),
            "feature": nodes["feature_idx"].astype(int).tolist(),
            "threshold": nodes["num_threshold"].astype(float).tolist(),
            "value": nodes["value"].astype(float).tolist(),
            "is_leaf": nodes["is_leaf"].astype(int).tolist(),
            "missing_go_to_left": nodes["missing_go_to_left"].astype(int).tolist(),
        })
    return {
        "kind": "hist_gradient_boosting", "feature_names": list(FEATURE_NAMES),
        "medians": medians.tolist(),
        "initial_log_odds": float(model._baseline_prediction[0, 0]), "trees": trees,
        "calibration": calibration,
    }


def train_target(target, train, calibration_rows, test):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    x_train = np.asarray([row[2] for row in train], dtype=float)
    x_cal = np.asarray([row[2] for row in calibration_rows], dtype=float)
    x_test = np.asarray([row[2] for row in test], dtype=float)
    y_train = np.asarray([row[3][target] for row in train], dtype=int)
    y_cal = np.asarray([row[3][target] for row in calibration_rows], dtype=int)
    y_test = np.asarray([row[3][target] for row in test], dtype=int)
    medians = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_cal = np.where(np.isnan(x_cal), medians, x_cal)
    x_test = np.where(np.isnan(x_test), medians, x_test)

    means, scales = x_train.mean(axis=0), x_train.std(axis=0)
    scales[scales < 1e-6] = 1.0
    logistic = LogisticRegression(C=.5, max_iter=1000, solver="liblinear")
    x_train_scaled = (x_train - means) / scales
    logistic.fit(x_train_scaled, y_train)
    logistic_cal_raw = logistic.intercept_[0] + np.einsum("ij,j->i", (x_cal - means) / scales, logistic.coef_[0])
    logistic_test_raw = logistic.intercept_[0] + np.einsum("ij,j->i", (x_test - means) / scales, logistic.coef_[0])
    logistic_probability, logistic_calibration = calibrate(logistic_cal_raw, y_cal, logistic_test_raw)

    gradient = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=.06, max_depth=4, min_samples_leaf=100,
        l2_regularization=1.0, random_state=42,
    )
    gradient.fit(x_train, y_train)
    gradient_cal_raw = gradient.decision_function(x_cal)
    gradient_test_raw = gradient.decision_function(x_test)
    gradient_probability, gradient_calibration = calibrate(gradient_cal_raw, y_cal, gradient_test_raw)

    baseline_rate = min(.999, max(.001, float(y_train.mean())))
    reports = {
        "baseline": metrics(y_test, [baseline_rate] * len(y_test)),
        "logistic": metrics(y_test, logistic_probability),
        "gradient_boosting": metrics(y_test, gradient_probability),
    }
    selected = min(("logistic", "gradient_boosting"), key=lambda name: (reports[name]["log_loss"], reports[name]["brier"]))
    chosen, baseline = reports[selected], reports["baseline"]
    relative_improvement = (baseline["log_loss"] - chosen["log_loss"]) / baseline["log_loss"]
    eligible = (
        chosen["examples"] >= 1000 and chosen["positives"] >= 100
        and relative_improvement >= .005
        and chosen["brier"] <= baseline["brier"]
        and chosen["ece"] <= .04
    )
    candidates = {
        "logistic": export_logistic(logistic, medians, means, scales, logistic_calibration),
        "gradient_boosting": export_gradient(gradient, medians, gradient_calibration),
    }
    reference = {"logistic": logistic_probability, "gradient_boosting": gradient_probability}
    for name, exported in candidates.items():
        for index in range(min(25, len(x_test))):
            feature_map = {feature: float(x_test[index, position]) for position, feature in enumerate(FEATURE_NAMES)}
            difference = abs(predict_exported(exported, feature_map) - float(reference[name][index]))
            if difference > 1e-7:
                raise RuntimeError(f"Pure-Python {name} export differs from sklearn by {difference:.3g}")
    return {
        "selected": selected,
        "eligible_for_promotion": bool(eligible),
        "promotion_gate": {
            "minimum_test_examples": 1000, "minimum_positives": 100,
            "minimum_log_loss_improvement": .005, "maximum_ece": .04,
            "actual_log_loss_improvement": round(relative_improvement, 6),
        },
        "metrics": {**reports, "selected": chosen},
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(TRAINING_DB_PATH))
    parser.add_argument("--cache-dir", default=str(Path(__file__).parent.parent / ".gameday_cache"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--limit-games", type=int)
    parser.add_argument("--promote-eligible", action="store_true",
                        help="Promote only targets that pass every out-of-time gate; default is shadow mode.")
    parser.add_argument("--output", default=str(REGISTRY_PATH))
    args = parser.parse_args()
    if not args.skip_build:
        result = build_training_examples(args.cache_dir, args.db, args.limit_games)
        print(f"Built chronological examples from {result['games']} games; {result['inserted']} new rows.")
    rows = matrix(args.db)
    train, calibration_rows, test = chronological_split(rows)
    print(f"Training on {len(train):,} PA; calibrating on {len(calibration_rows):,}; testing on {len(test):,}.")
    targets = {}
    for target in TARGETS:
        print(f"Training {target} challengers…")
        record = train_target(target, train, calibration_rows, test)
        record["status"] = "promoted" if args.promote_eligible and record["eligible_for_promotion"] else "shadow"
        targets[target] = record
        selected = record["selected"]
        report = record["metrics"][selected]
        print(f"  {selected}: log loss {report['log_loss']:.4f}, Brier {report['brier']:.4f}, ECE {report['ece']:.4f}; {record['status']}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    registry = {
        "model_version": "hitter-challenger-v1",
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trained_through": rows[-1][0],
        "split": {
            "train_through": train[-1][0], "calibration_through": calibration_rows[-1][0],
            "test_through": test[-1][0], "train_examples": len(train),
            "calibration_examples": len(calibration_rows), "test_examples": len(test),
        },
        "default_mode": "shadow", "targets": targets,
    }
    output.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
