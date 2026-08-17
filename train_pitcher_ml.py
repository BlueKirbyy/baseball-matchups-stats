#!/usr/bin/env python3
"""Build chronological starter-game examples and train workload challengers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from math import log, sqrt
from pathlib import Path
import json

from analytics_store import DB_PATH, connect
from pitcher_ml import (
    FEATURE_NAMES, FEATURE_VERSION, MODEL_VERSION, REGISTRY_PATH,
    build_training_examples, predict_exported_classifier,
    predict_exported_regressor,
)


TARGETS = ("batters_faced", "pitches", "outs")


def matrix(db_path):
    rows = []
    with connect(db_path) as db:
        for row in db.execute(
            """SELECT game_date, game_pk, player_id, features_json,
                      batters_faced, pitches, outs, early_exit
                 FROM pitcher_game_ml_examples WHERE feature_version=?
                ORDER BY game_date, game_pk, player_id""",
            (FEATURE_VERSION,),
        ):
            features = json.loads(row["features_json"])
            rows.append((
                row["game_date"], row["game_pk"], row["player_id"],
                [float(features.get(name, 0.0)) for name in FEATURE_NAMES],
                {target: float(row[target]) for target in TARGETS}
                | {"early_exit": int(row["early_exit"])},
            ))
    return rows


def chronological_split(rows):
    games = []
    for row in rows:
        key = (row[0], row[1])
        if not games or games[-1] != key:
            games.append(key)
    if len(games) < 100:
        raise RuntimeError("At least 100 completed games are required for workload training.")
    train_end = games[max(1, int(len(games) * .65)) - 1]
    calibration_end = games[max(2, int(len(games) * .82)) - 1]
    train = [row for row in rows if (row[0], row[1]) <= train_end]
    calibration = [row for row in rows if train_end < (row[0], row[1]) <= calibration_end]
    test = [row for row in rows if (row[0], row[1]) > calibration_end]
    return train, calibration, test


def regression_metrics(actual, predicted):
    errors = [float(prediction) - float(value) for value, prediction in zip(actual, predicted)]
    return {
        "examples": len(errors),
        "mae": round(sum(abs(value) for value in errors) / len(errors), 6),
        "rmse": round(sqrt(sum(value * value for value in errors) / len(errors)), 6),
        "bias": round(sum(errors) / len(errors), 6),
    }


def classification_metrics(actual, probability):
    clipped = [min(.999999, max(.000001, float(value))) for value in probability]
    return {
        "examples": len(actual), "positives": int(sum(actual)),
        "brier": round(sum((p - y) ** 2 for y, p in zip(actual, clipped)) / len(actual), 6),
        "log_loss": round(-sum(y * log(p) + (1 - y) * log(1 - p) for y, p in zip(actual, clipped)) / len(actual), 6),
    }


def _export_trees(model):
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
    return trees


def export_histogram(model, medians, kind="hist_gradient_boosting", calibration=None):
    exported = {
        "kind": kind, "feature_names": list(FEATURE_NAMES),
        "medians": medians.tolist(),
        "initial_value": float(model._baseline_prediction[0, 0]),
        "trees": _export_trees(model),
    }
    if calibration:
        exported["calibration"] = calibration
    return exported


def export_ridge(model, medians, means, scales):
    return {
        "kind": "ridge", "feature_names": list(FEATURE_NAMES),
        "medians": medians.tolist(), "means": means.tolist(),
        "scales": scales.tolist(), "intercept": float(model.intercept_),
        "coefficients": model.coef_.tolist(),
    }


def _arrays(rows):
    import numpy as np
    return np.asarray([row[3] for row in rows], dtype=float)


def train_regression_target(target, train, calibration, test):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    x_train, x_cal, x_test = _arrays(train), _arrays(calibration), _arrays(test)
    y_train = np.asarray([row[4][target] for row in train], dtype=float)
    y_cal = np.asarray([row[4][target] for row in calibration], dtype=float)
    y_test = np.asarray([row[4][target] for row in test], dtype=float)
    medians = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_cal = np.where(np.isnan(x_cal), medians, x_cal)
    x_test = np.where(np.isnan(x_test), medians, x_test)
    means, scales = x_train.mean(axis=0), x_train.std(axis=0)
    scales[scales < 1e-6] = 1.0

    ridge = Ridge(alpha=10.0, solver="lsqr").fit((x_train - means) / scales, y_train)
    ridge_test = ridge.predict((x_test - means) / scales)
    gradient = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=120, learning_rate=.045,
        max_depth=3, min_samples_leaf=60, l2_regularization=5.0,
        random_state=42,
    ).fit(x_train, y_train)
    gradient_test = gradient.predict(x_test)
    baseline_feature = {
        "batters_faced": "pitcher_avg_bf", "pitches": "pitcher_avg_pitches",
        "outs": "pitcher_avg_outs",
    }[target]
    baseline_index = FEATURE_NAMES.index(baseline_feature)
    reports = {
        "historical_baseline": regression_metrics(y_test, x_test[:, baseline_index]),
        "ridge": regression_metrics(y_test, ridge_test),
        "gradient_boosting": regression_metrics(y_test, gradient_test),
    }
    selected = min(("ridge", "gradient_boosting"), key=lambda name: reports[name]["mae"])
    candidates = {
        "ridge": export_ridge(ridge, medians, means, scales),
        "gradient_boosting": export_histogram(gradient, medians),
    }

    interval_models = {}
    calibration_predictions = []
    for label, quantile in (("low", .10), ("high", .90)):
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, max_iter=150,
            learning_rate=.045, max_depth=3, min_samples_leaf=60,
            l2_regularization=5.0, random_state=42,
        ).fit(x_train, y_train)
        interval_models[label] = export_histogram(model, medians, kind="hist_quantile")
        calibration_predictions.append(model.predict(x_cal))
    low_cal, high_cal = calibration_predictions
    miss = np.maximum.reduce((low_cal - y_cal, y_cal - high_cal, np.zeros(len(y_cal))))
    padding = float(np.quantile(miss, .80, method="higher"))
    low_test = np.asarray([predict_exported_regressor(interval_models["low"], dict(zip(FEATURE_NAMES, row))) for row in x_test]) - padding
    high_test = np.asarray([predict_exported_regressor(interval_models["high"], dict(zip(FEATURE_NAMES, row))) for row in x_test]) + padding
    coverage = float(np.mean((y_test >= low_test) & (y_test <= high_test)))
    width = float(np.mean(high_test - low_test))

    reference = {"ridge": ridge_test, "gradient_boosting": gradient_test}
    for name, exported in candidates.items():
        for index in range(min(25, len(x_test))):
            feature_map = dict(zip(FEATURE_NAMES, x_test[index]))
            difference = abs(predict_exported_regressor(exported, feature_map) - float(reference[name][index]))
            if difference > 1e-7:
                raise RuntimeError(f"Pure-Python {target}/{name} export differs by {difference:.3g}")

    baseline_mae = reports["historical_baseline"]["mae"]
    improvement = (baseline_mae - reports[selected]["mae"]) / baseline_mae
    return {
        "selected": selected, "status": "shadow",
        "eligible_for_promotion": bool(len(test) >= 500 and improvement >= .01 and .75 <= coverage <= .88),
        "metrics": {**reports, "selected": reports[selected],
                    "relative_mae_improvement": round(improvement, 6),
                    "interval_80_coverage": round(coverage, 6),
                    "interval_mean_width": round(width, 6)},
        "candidates": candidates, "interval_models": interval_models,
        "interval_calibration": round(padding, 8),
    }


def _sigmoid_calibrate(raw_cal, y_cal, raw_test):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    calibrator = LogisticRegression(C=1.0, solver="liblinear")
    calibrator.fit(np.asarray(raw_cal).reshape(-1, 1), y_cal)
    export = {"intercept": float(calibrator.intercept_[0]),
              "coefficient": float(calibrator.coef_[0][0])}
    raw = export["intercept"] + export["coefficient"] * np.asarray(raw_test)
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -35, 35))), export


def train_early_exit(train, calibration, test):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    x_train, x_cal, x_test = _arrays(train), _arrays(calibration), _arrays(test)
    y_train = np.asarray([row[4]["early_exit"] for row in train], dtype=int)
    y_cal = np.asarray([row[4]["early_exit"] for row in calibration], dtype=int)
    y_test = np.asarray([row[4]["early_exit"] for row in test], dtype=int)
    medians = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_cal = np.where(np.isnan(x_cal), medians, x_cal)
    x_test = np.where(np.isnan(x_test), medians, x_test)
    means, scales = x_train.mean(axis=0), x_train.std(axis=0)
    scales[scales < 1e-6] = 1.0

    logistic = LogisticRegression(C=.5, max_iter=1000, solver="liblinear").fit((x_train - means) / scales, y_train)
    logistic_cal_raw = logistic.intercept_[0] + ((x_cal - means) / scales) @ logistic.coef_[0]
    logistic_test_raw = logistic.intercept_[0] + ((x_test - means) / scales) @ logistic.coef_[0]
    logistic_probability, logistic_calibration = _sigmoid_calibrate(logistic_cal_raw, y_cal, logistic_test_raw)
    logistic_export = {
        "kind": "ridge", "feature_names": list(FEATURE_NAMES),
        "medians": medians.tolist(), "means": means.tolist(),
        "scales": scales.tolist(), "intercept": float(logistic.intercept_[0]),
        "coefficients": logistic.coef_[0].tolist(),
        "calibration": logistic_calibration,
    }

    gradient = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.045, max_depth=3,
        min_samples_leaf=60, l2_regularization=5.0, random_state=42,
    ).fit(x_train, y_train)
    gradient_probability, gradient_calibration = _sigmoid_calibrate(
        gradient.decision_function(x_cal), y_cal, gradient.decision_function(x_test),
    )
    gradient_export = export_histogram(gradient, medians, calibration=gradient_calibration)
    reports = {
        "baseline": classification_metrics(y_test, [float(y_train.mean())] * len(y_test)),
        "logistic": classification_metrics(y_test, logistic_probability),
        "gradient_boosting": classification_metrics(y_test, gradient_probability),
    }
    selected = min(("logistic", "gradient_boosting"), key=lambda name: reports[name]["log_loss"])
    candidates = {"logistic": logistic_export, "gradient_boosting": gradient_export}
    reference = {"logistic": logistic_probability, "gradient_boosting": gradient_probability}
    for name, exported in candidates.items():
        for index in range(min(25, len(x_test))):
            difference = abs(predict_exported_classifier(exported, dict(zip(FEATURE_NAMES, x_test[index]))) - float(reference[name][index]))
            if difference > 1e-7:
                raise RuntimeError(f"Pure-Python early-exit/{name} export differs by {difference:.3g}")
    improvement = (reports["baseline"]["log_loss"] - reports[selected]["log_loss"]) / reports["baseline"]["log_loss"]
    return {
        "selected": selected, "status": "shadow", "eligible_for_promotion": False,
        "metrics": {**reports, "selected": reports[selected],
                    "relative_log_loss_improvement": round(improvement, 6)},
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the shadow pitcher workload model")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--limit-games", type=int)
    parser.add_argument("--output", default=str(REGISTRY_PATH))
    args = parser.parse_args()
    if not args.skip_build:
        result = build_training_examples(args.db, args.limit_games)
        print(f"Built {result['inserted']:,} new starter examples and {result['outcomes']:,} normalized outcomes from {result['games']:,} games.")
    rows = matrix(args.db)
    train, calibration, test = chronological_split(rows)
    print(f"Training on {len(train):,} starts; calibrating on {len(calibration):,}; testing on {len(test):,}.")
    targets = {target: train_regression_target(target, train, calibration, test) for target in TARGETS}
    targets["early_exit"] = train_early_exit(train, calibration, test)
    for target, record in targets.items():
        selected = record["selected"]
        report = record["metrics"][selected]
        metric = report.get("mae", report.get("log_loss"))
        print(f"  {target}: {selected} ({metric:.4f}); shadow")
    now = datetime.now(timezone.utc).isoformat()
    registry = {
        "model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
        "created_at": now, "trained_through": rows[-1][0],
        "default_mode": "shadow",
        "split": {
            "train_through": train[-1][0], "calibration_through": calibration[-1][0],
            "test_through": test[-1][0], "train_examples": len(train),
            "calibration_examples": len(calibration), "test_examples": len(test),
        },
        "targets": targets,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    with connect(args.db) as db:
        db.execute(
            """INSERT OR IGNORE INTO ml_model_registry(
                 model_version, feature_version, target_group, created_at,
                 trained_through, status, artifact_path, metrics_json
               ) VALUES (?, ?, 'pitcher_workload', ?, ?, 'shadow', ?, ?)""",
            (MODEL_VERSION, FEATURE_VERSION, now, rows[-1][0], str(output),
             json.dumps({key: value["metrics"] for key, value in targets.items()}, sort_keys=True)),
        )
    print(f"Saved shadow registry to {output}")


if __name__ == "__main__":
    main()
