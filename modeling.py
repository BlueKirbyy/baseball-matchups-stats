"""Versioned, server-side models and market math for Diamond Intel.

The initial model is intentionally a conservative empirical-Bayes baseline. It
is a research model until walk-forward results establish calibration and value.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import erf, exp, floor, isfinite, lgamma, log, sqrt
from statistics import median
import re

from park_factors import venue_factor
from pitcher_ml import shadow_workload_prediction

MODEL_VERSION = "pitcher-k-workload-v5"
FEATURE_VERSION = "gameday-features-v5"
LEAGUE_K_RATE = 0.225
LEAGUE_BF_PER_START = 22.0
LEAGUE_HITTER_AVERAGE = 0.245
LEAGUE_HITTER_SLG = 0.400
LEAGUE_HITTER_ISO = LEAGUE_HITTER_SLG - LEAGUE_HITTER_AVERAGE
LEAGUE_HITTER_OBP = 0.320
LEAGUE_HITTER_HR_RATE = 0.030
LEAGUE_HARD_HIT_RATE = 0.390
LEAGUE_BARREL_PROXY_RATE = 0.075
HITTER_PITCH_PRIOR_PA = 60.0
HITTER_PLATOON_PRIOR_PA = 80.0
HITTER_PITCH_MIN_PA = 10.0
HITTER_PITCH_MIN_RELIABILITY = 0.25
HITTER_CONTACT_DELTA = 0.025
HITTER_CONTACT_FAVORABLE_DELTA = 0.012
HITTER_ARSENAL_MIN_COVERAGE = 0.35
HITTER_ARSENAL_MIN_EFFECTIVE_PA = 10.0
# Full-game coverage is deliberately diluted by unmodeled arsenal mass and the
# reliever blend. Count/zone coverage is an even narrower exact-context
# intersection, so it naturally lives in the low single digits. These gates
# identify the top evidence tail without requiring impossible percentages.
HITTER_STRONG_MIN_COVERAGE = 0.35
HITTER_STRONG_MIN_EFFECTIVE_PA = 30.0
HITTER_STRONG_MIN_CONTEXT_COVERAGE = 0.03
HITTER_RISK_MIN_COVERAGE = 0.15
HITTER_RISK_MIN_EFFECTIVE_PA = 10.0
HITTER_RISK_MIN_CONTEXT_COVERAGE = 0.02
HITTER_RISK_MIN_PLATOON_AB = 50
HITTER_RISK_MIN_SEASON_PA = 100
HITTER_PROMISING_MIN_COVERAGE = 0.25
HITTER_PROMISING_MIN_EFFECTIVE_PA = 20.0
HITTER_RECENT_FORM_MIN_PA = 20
HITTER_RECENT_FORM_USABLE_PA = 35
HITTER_RECENT_FORM_MAX_ADJUSTMENT = 0.10
# Evidence needs differ by outcome.  Direction is scored separately, so missing
# one of these gates lowers confidence without erasing a genuinely positive read.
HITTER_OUTCOME_EVIDENCE = {
    "hit": {"coverage": 0.25, "effective_pa": 20.0, "context": 0.0, "quality": 0.0, "lineup": False},
    "total_bases": {"coverage": 0.30, "effective_pa": 25.0, "context": 0.0, "quality": 0.10, "lineup": False},
    "home_run": {"coverage": 0.30, "effective_pa": 30.0, "context": 0.0, "quality": 0.20, "lineup": False},
    "runs_rbi": {"coverage": 0.20, "effective_pa": 20.0, "context": 0.0, "quality": 0.0, "lineup": True},
    "overall": {"coverage": 0.30, "effective_pa": 25.0, "context": 0.0, "quality": 0.0, "lineup": False},
}
LEAGUE_GAME_TOTAL = 8.5
LEAGUE_TEAM_RUNS = LEAGUE_GAME_TOTAL / 2.0
LEAGUE_STARTER_ER9 = 4.30
LEAGUE_STARTER_WHIP = 1.30
LEAGUE_PITCHES_PER_PA = 3.9
LEAGUE_ON_BASE_PROXY = 0.320
LEAGUE_OUT_RATE = 0.675
LEAGUE_TOTAL_BASES_PER_PA = 0.360


def _number(value, default=None):
    try:
        result = float(value)
        return result if isfinite(result) else default
    except (TypeError, ValueError):
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def shrunk_rate(successes, trials, prior_rate, prior_strength):
    """Beta-binomial posterior mean with an explicit effective prior sample."""
    successes = max(0.0, _number(successes, 0.0))
    trials = max(successes, _number(trials, 0.0))
    prior_rate = clamp(_number(prior_rate, LEAGUE_K_RATE), 0.001, 0.999)
    prior_strength = max(0.0, _number(prior_strength, 0.0))
    return (successes + prior_rate * prior_strength) / (trials + prior_strength) if trials + prior_strength else prior_rate


def hitter_recent_form_adjustment(outcome, recent_form):
    """Return a deliberately small, sample-shrunk recent-form contribution."""
    recent_form = recent_form or {}
    pa = max(0.0, _number(recent_form.get("pa"), 0.0))
    if pa < HITTER_RECENT_FORM_MIN_PA:
        return 0.0
    component_scores = recent_form.get("component_scores") or {}
    score = clamp(_number(component_scores.get(outcome), _number(recent_form.get("score"), 0.0)), -1.0, 1.0)
    weights = {
        "hit": 0.07,
        "total_bases": 0.10,
        "home_run": 0.08 if pa >= HITTER_RECENT_FORM_USABLE_PA else 0.04,
        "runs_rbi": 0.06,
        "overall": 0.08,
    }
    return clamp(score * weights.get(outcome, 0.08), -HITTER_RECENT_FORM_MAX_ADJUSTMENT,
                 HITTER_RECENT_FORM_MAX_ADJUSTMENT)


def calculate_hitter_recent_form(rows, as_of, primary_days=14, fallback_days=21):
    """Summarize completed pregame batting lines without same-game leakage."""
    try:
        cutoff = datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return {"window_days": primary_days, "games": 0, "pa": 0, "label": "Insufficient recent data", "score": 0.0, "adjustment": 0.0, "drivers": []}
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)

    def played_at(row):
        raw = row.get("scheduled_start")
        if raw:
            try:
                value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass
        try:
            # Unknown same-day start times are placed at the end of the day so a
            # doubleheader cannot accidentally consume a future result.
            return datetime.fromisoformat(f"{row.get('game_date')}T23:59:59+00:00")
        except (TypeError, ValueError):
            return None

    prior = [(played_at(row), dict(row)) for row in rows]
    # Same-day results are excluded even when an earlier game in a
    # doubleheader has finished. This deliberately conservative rule prevents
    # any current-slate outcome from entering a pregame feature vector.
    prior = sorted(
        (played, row) for played, row in prior
        if played and played < cutoff and played.date() < cutoff.date()
    )
    season_rows = [row for played, row in prior if played.year == cutoff.year]

    def within(days):
        start = cutoff - timedelta(days=days)
        return [row for played, row in prior if start <= played < cutoff]

    window_days = primary_days
    recent_rows = within(primary_days)
    if sum(_number(row.get("plate_appearances"), 0.0) for row in recent_rows) < 25:
        window_days = fallback_days
        recent_rows = within(fallback_days)

    def summarize(items):
        totals = {
            key: sum(max(0.0, _number(row.get(key), 0.0)) for row in items)
            for key in (
                "plate_appearances", "at_bats", "hits", "walks", "hit_by_pitch",
                "sacrifice_flies", "total_bases", "doubles", "triples",
                "home_runs", "strikeouts",
            )
        }
        pa, ab = totals["plate_appearances"], totals["at_bats"]
        obp_denominator = ab + totals["walks"] + totals["hit_by_pitch"] + totals["sacrifice_flies"]
        avg = totals["hits"] / ab if ab else None
        obp = (totals["hits"] + totals["walks"] + totals["hit_by_pitch"]) / obp_denominator if obp_denominator else None
        slg = totals["total_bases"] / ab if ab else None
        return {
            **totals, "games": len(items), "avg": avg, "obp": obp, "slg": slg,
            "iso": max(0.0, slg - avg) if slg is not None and avg is not None else None,
            "strikeout_rate": totals["strikeouts"] / pa if pa else None,
            "extra_base_hits": totals["doubles"] + totals["triples"] + totals["home_runs"],
            "home_run_rate": totals["home_runs"] / pa if pa else None,
            "extra_base_hit_rate": (totals["doubles"] + totals["triples"] + totals["home_runs"]) / pa if pa else None,
        }

    recent = summarize(recent_rows)
    season = summarize(season_rows)
    hit_streak = 0
    for _played, row in reversed(prior):
        if _number(row.get("hits"), 0.0) <= 0:
            break
        hit_streak += 1
    games_with_hit = sum(_number(row.get("hits"), 0.0) > 0 for row in recent_rows)
    pa = recent["plate_appearances"]
    reliability = pa / (pa + 50.0) if pa else 0.0

    season_avg = season["avg"] if season["avg"] is not None else LEAGUE_HITTER_AVERAGE
    season_obp = season["obp"] if season["obp"] is not None else LEAGUE_HITTER_OBP
    season_slg = season["slg"] if season["slg"] is not None else LEAGUE_HITTER_SLG
    season_iso = season["iso"] if season["iso"] is not None else LEAGUE_HITTER_ISO
    season_k = season["strikeout_rate"] if season["strikeout_rate"] is not None else LEAGUE_K_RATE
    season_hr_rate = season["home_run_rate"] if season["home_run_rate"] is not None else LEAGUE_HITTER_HR_RATE
    season_xbh_rate = season["extra_base_hit_rate"] if season["extra_base_hit_rate"] is not None else .080
    avg_signal = clamp(((recent["avg"] or season_avg) - season_avg) / .055, -1.0, 1.0)
    slg_signal = clamp(((recent["slg"] or season_slg) - season_slg) / .120, -1.0, 1.0)
    obp_signal = clamp(((recent["obp"] or season_obp) - season_obp) / .060, -1.0, 1.0)
    iso_signal = clamp(((recent["iso"] or season_iso) - season_iso) / .080, -1.0, 1.0)
    k_signal = clamp((season_k - (recent["strikeout_rate"] if recent["strikeout_rate"] is not None else season_k)) / .060, -1.0, 1.0)
    hit_game_rate = games_with_hit / len(recent_rows) if recent_rows else 0.0
    streak_signal = clamp(
        0.55 * clamp((hit_game_rate - .50) / .30, -1.0, 1.0)
        + 0.45 * clamp((hit_streak - 2.0) / 6.0, -1.0, 1.0),
        -1.0, 1.0,
    )
    raw_score = (
        .35 * slg_signal + .25 * obp_signal + .20 * iso_signal
        + .10 * k_signal + .10 * streak_signal
    )
    hr_signal = clamp(((recent["home_run_rate"] or season_hr_rate) - season_hr_rate) / .025, -1.0, 1.0)
    xbh_signal = clamp(((recent["extra_base_hit_rate"] or season_xbh_rate) - season_xbh_rate) / .050, -1.0, 1.0)
    raw_components = {
        "hit": .45 * avg_signal + .30 * streak_signal + .20 * k_signal + .05 * obp_signal,
        "total_bases": .55 * slg_signal + .35 * iso_signal + .10 * xbh_signal,
        "home_run": .50 * iso_signal + .30 * hr_signal + .20 * xbh_signal,
        "runs_rbi": .45 * obp_signal + .25 * streak_signal + .20 * slg_signal + .10 * k_signal,
        "overall": raw_score,
    }
    component_scores = {
        key: clamp(value * reliability, -1.0, 1.0) if pa >= HITTER_RECENT_FORM_MIN_PA else 0.0
        for key, value in raw_components.items()
    }
    score = component_scores["overall"]
    if pa < HITTER_RECENT_FORM_MIN_PA:
        label = "Insufficient recent data"
    elif score >= .16:
        label = "Hot"
    elif score >= .04:
        label = "Positive"
    elif score <= -.08:
        label = "Cold"
    else:
        label = "Neutral"
    drivers = []
    if recent["obp"] is not None and recent["slg"] is not None:
        drivers.append(f"Last {window_days} days: {recent['obp']:.3f} OBP · {recent['slg']:.3f} SLG")
    if hit_streak >= 2:
        drivers.append(f"{hit_streak}-game hit streak")
    drivers.append(f"{pa:.0f} PA · {reliability:.0%} recent-form reliability")
    return {
        "window_days": window_days, "window_start": (cutoff - timedelta(days=window_days)).date().isoformat(),
        "as_of": cutoff.isoformat(), "games": recent["games"], "pa": int(pa),
        "avg": recent["avg"], "obp": recent["obp"], "slg": recent["slg"],
        "iso": recent["iso"], "strikeout_rate": recent["strikeout_rate"],
        "hits": int(recent["hits"]), "games_with_hit": games_with_hit,
        "hit_streak": hit_streak, "extra_base_hits": int(recent["extra_base_hits"]),
        "home_runs": int(recent["home_runs"]), "reliability": round(reliability, 4),
        "raw_score": round(raw_score, 4), "score": round(score, 4),
        "component_scores": {key: round(value, 4) for key, value in component_scores.items()},
        "adjustment": round(hitter_recent_form_adjustment("overall", {"pa": pa, "score": score}), 4),
        "label": label, "drivers": drivers,
        "source_game_pks": [int(row["game_pk"]) for row in recent_rows if row.get("game_pk") is not None],
        "source_games": [
            {
                key: row.get(key) for key in (
                    "game_pk", "game_date", "scheduled_start", "plate_appearances",
                    "at_bats", "hits", "walks", "hit_by_pitch", "sacrifice_flies",
                    "total_bases", "doubles", "triples", "home_runs", "strikeouts",
                )
            }
            for row in recent_rows
        ],
        "season_baseline": {
            "pa": int(season["plate_appearances"]), "avg": season_avg, "obp": season_obp,
            "slg": season_slg, "iso": season_iso, "strikeout_rate": season_k,
        },
    }


def hitter_ranking_reliability(batter, metrics):
    """Blend independent evidence dimensions without an artificial coverage floor."""
    coverage = clamp(_number(metrics.get("coverage"), 0.0), 0.0, 1.0)
    sample = max(0.0, _number(metrics.get("effective_sample_size"), 0.0))
    context = clamp(_number(metrics.get("context_coverage"), 0.0) / 0.05, 0.0, 1.0)
    platoon_ab = max(0.0, _number((metrics.get("platoon") or batter.get("platoon") or {}).get("at_bats"), 0.0))
    bullpen = metrics.get("bullpen") or {}
    bullpen_coverage = clamp(
        _number(bullpen.get("coverage"), _number(bullpen.get("modeled_weight"), 0.0)),
        0.0, 1.0,
    )
    sample_reliability = sample / (sample + 30.0) if sample else 0.0
    platoon_reliability = platoon_ab / (platoon_ab + 80.0) if platoon_ab else 0.0
    return clamp(
        0.35 * coverage + 0.25 * sample_reliability + 0.15 * context
        + 0.15 * platoon_reliability + 0.10 * bullpen_coverage,
        0.0, 1.0,
    )


def hitter_outcome_confidence(outcome, batter, metrics, risk=None):
    """Classify evidence independently from positive/negative matchup direction."""
    requirements = HITTER_OUTCOME_EVIDENCE[outcome]
    risk = risk or hitter_signal_risk(batter, metrics)
    lineup_confirmed = bool(batter.get("lineup_order"))
    meets_outcome_gate = (
        _number(metrics.get("coverage"), 0.0) >= requirements["coverage"]
        and _number(metrics.get("effective_sample_size"), 0.0) >= requirements["effective_pa"]
        and _number(metrics.get("context_coverage"), 0.0) >= requirements["context"]
        and _number(metrics.get("quality_coverage"), 0.0) >= requirements["quality"]
        and (not requirements["lineup"] or lineup_confirmed)
    )
    high = (
        hitter_strong_evidence(metrics)
        and lineup_confirmed
        and risk.get("level") != "high"
        and _number(metrics.get("quality_coverage"), 0.0) >= requirements["quality"]
    )
    meets_promising_gate = (
        _number(metrics.get("coverage"), 0.0) >= HITTER_PROMISING_MIN_COVERAGE
        and _number(metrics.get("effective_sample_size"), 0.0) >= HITTER_PROMISING_MIN_EFFECTIVE_PA
    )
    blockers = []
    if _number(metrics.get("coverage"), 0.0) < requirements["coverage"]:
        blockers.append(f"needs {requirements['coverage']:.0%} coverage")
    if _number(metrics.get("effective_sample_size"), 0.0) < requirements["effective_pa"]:
        blockers.append(f"needs {requirements['effective_pa']:.0f} effective PA")
    if _number(metrics.get("context_coverage"), 0.0) < requirements["context"]:
        blockers.append(f"needs {requirements['context']:.0%} exact-context coverage")
    if _number(metrics.get("quality_coverage"), 0.0) < requirements["quality"]:
        blockers.append(f"needs {requirements['quality']:.0%} quality-of-contact coverage")
    if requirements["lineup"] and not lineup_confirmed:
        blockers.append("needs a confirmed lineup position")
    if risk.get("level") == "high":
        blockers.append("high evidence risk")
    if high:
        confidence = "high"
    elif meets_outcome_gate and risk.get("level") != "high":
        confidence = "medium"
    elif meets_promising_gate and risk.get("level") != "high":
        confidence = "medium"
    else:
        confidence = "limited"
    return {
        "confidence": confidence,
        "qualified": (
            meets_outcome_gate and confidence in {"high", "medium"}
            and lineup_confirmed and risk.get("level") != "high"
        ),
        "requirements": dict(requirements),
        "meets_outcome_gate": meets_outcome_gate,
        "blockers": blockers,
    }


def implied_probability(american_price):
    price = _number(american_price)
    if price is None or price == 0:
        return None
    return -price / (-price + 100.0) if price < 0 else 100.0 / (price + 100.0)


def no_vig_probabilities(over_price, under_price):
    over = implied_probability(over_price)
    under = implied_probability(under_price)
    if over is None or under is None or over + under <= 0:
        return None
    return {"over": over / (over + under), "under": under / (over + under)}


def american_from_probability(probability):
    probability = _number(probability)
    if probability is None or not 0 < probability < 1:
        return None
    price = -100 * probability / (1 - probability) if probability >= 0.5 else 100 * (1 - probability) / probability
    return int(round(price))


def expected_value(probability, american_price):
    probability = _number(probability)
    price = _number(american_price)
    if probability is None or price is None or price == 0:
        return None
    profit = price / 100.0 if price > 0 else 100.0 / -price
    return probability * profit - (1.0 - probability)


def poisson_pmf(k, mean):
    if k < 0 or mean < 0:
        return 0.0
    if mean == 0:
        return 1.0 if k == 0 else 0.0
    return exp(-mean + k * log(mean) - lgamma(k + 1))


def negative_binomial_pmf(k, mean, dispersion=12.0):
    """NB2 PMF where variance = mean + mean^2 / dispersion."""
    if k < 0 or mean < 0 or dispersion <= 0:
        return 0.0
    if mean == 0:
        return 1.0 if k == 0 else 0.0
    r = float(dispersion)
    p = r / (r + mean)
    return exp(lgamma(k + r) - lgamma(r) - lgamma(k + 1) + r * log(p) + k * log(1 - p))


def count_distribution(mean, kind="negative_binomial", dispersion=12.0, maximum=50):
    pmf = negative_binomial_pmf if kind == "negative_binomial" else poisson_pmf
    values = [pmf(k, mean, dispersion) if kind == "negative_binomial" else pmf(k, mean) for k in range(maximum + 1)]
    total = sum(values)
    return [value / total for value in values] if total else [1.0] + [0.0] * maximum


def distribution_summary(mean, line=None, kind="negative_binomial", dispersion=12.0):
    probabilities = count_distribution(mean, kind, dispersion)

    def quantile(target):
        cumulative = 0.0
        for value, probability in enumerate(probabilities):
            cumulative += probability
            if cumulative >= target:
                return value
        return len(probabilities) - 1

    result = {
        "distribution": kind,
        "median": quantile(0.5),
        "interval_low": quantile(0.1),
        "interval_high": quantile(0.9),
    }
    if line is not None:
        line = float(line)
        result["probability_over"] = sum(probabilities[floor(line) + 1 :])
        result["probability_under"] = sum(probabilities[: max(0, int(line) if line.is_integer() else floor(line) + 1)])
        result["probability_push"] = probabilities[int(line)] if line.is_integer() and 0 <= int(line) < len(probabilities) else 0.0
        decisive = result["probability_over"] + result["probability_under"]
        if decisive:
            result["fair_over_price"] = american_from_probability(result["probability_over"] / decisive)
            result["fair_under_price"] = american_from_probability(result["probability_under"] / decisive)
    return result


def beta_binomial_pmf(k, trials, alpha, beta):
    """Beta-binomial PMF used to separate K-rate from workload uncertainty."""
    if k < 0 or k > trials or alpha <= 0 or beta <= 0:
        return 0.0
    return exp(
        lgamma(trials + 1) - lgamma(k + 1) - lgamma(trials - k + 1)
        + lgamma(k + alpha) + lgamma(trials - k + beta) - lgamma(trials + alpha + beta)
        + lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
    )


def workload_k_distribution(k_rate, expected_bf, bf_low, bf_high, line=None,
                            rate_strength=120.0, workload_scenarios=None):
    """Mix K outcomes over explicit short, normal, and extended outings.

    The point workload estimate remains the mean.  Scenario mixing adds the
    asymmetric downside that a single bell curve hid when a starter was
    ineffective or hooked early.
    """
    spread = max(2.0, (max(expected_bf, bf_high) - min(expected_bf, bf_low)) / 2.56)
    bf_values = list(range(8, 37))
    scenarios = workload_scenarios or [
        {"key": "normal", "probability": 1.0, "batters_faced": expected_bf, "spread": spread}
    ]
    bf_weights = [0.0] * len(bf_values)
    for scenario in scenarios:
        probability = max(0.0, _number(scenario.get("probability"), 0.0))
        scenario_mean = clamp(_number(scenario.get("batters_faced"), expected_bf), 8.0, 36.0)
        scenario_spread = max(1.4, _number(scenario.get("spread"), spread))
        raw = [exp(-0.5 * ((value - scenario_mean) / scenario_spread) ** 2) for value in bf_values]
        raw_total = sum(raw) or 1.0
        for index, value in enumerate(raw):
            bf_weights[index] += probability * value / raw_total
    weight_total = sum(bf_weights) or 1.0
    alpha = clamp(k_rate, .01, .99) * max(20.0, rate_strength)
    beta = (1.0 - clamp(k_rate, .01, .99)) * max(20.0, rate_strength)
    probabilities = [0.0] * 37
    for batters_faced, weight in zip(bf_values, bf_weights):
        normalized_weight = weight / weight_total
        for strikeouts in range(batters_faced + 1):
            probabilities[strikeouts] += normalized_weight * beta_binomial_pmf(
                strikeouts, batters_faced, alpha, beta,
            )
    total = sum(probabilities) or 1.0
    probabilities = [value / total for value in probabilities]

    def quantile(target):
        cumulative = 0.0
        for value, probability in enumerate(probabilities):
            cumulative += probability
            if cumulative >= target:
                return value
        return len(probabilities) - 1

    result = {
        "distribution": "workload_mixture_beta_binomial",
        "median": quantile(.5),
        "interval_low": quantile(.1),
        "interval_high": quantile(.9),
        "probability_three_or_fewer": sum(probabilities[:4]),
        "workload_scenarios": scenarios,
        "milestones": {
            str(target): sum(probabilities[target:]) for target in (4, 5, 6, 7, 8)
        },
    }
    if line is not None:
        line = float(line)
        result["probability_over"] = sum(probabilities[floor(line) + 1:])
        result["probability_under"] = sum(probabilities[:max(0, int(line) if line.is_integer() else floor(line) + 1)])
        result["probability_push"] = probabilities[int(line)] if line.is_integer() and 0 <= int(line) < len(probabilities) else 0.0
        decisive = result["probability_over"] + result["probability_under"]
        if decisive:
            result["fair_over_price"] = american_from_probability(result["probability_over"] / decisive)
            result["fair_under_price"] = american_from_probability(result["probability_under"] / decisive)
    return result


def _lineup_weight(order):
    expected_pa = (4.75, 4.65, 4.55, 4.45, 4.35, 4.25, 4.15, 4.05, 3.95)
    try:
        return expected_pa[int(order) - 1]
    except (TypeError, ValueError, IndexError):
        return 4.2


def bullpen_readiness(pitches_today=0, pitches_yesterday=0, pitches_two_days_ago=0,
                      consecutive_days=0, three_day_pitches=None):
    """Return a transparent, workload-only estimate of reliever readiness.

    This is deliberately an estimate rather than an available/unavailable fact.
    Recent pitches and consecutive-day use are observable; injuries, warm-up
    activity, and manager intent are not. The score exists mainly to create
    stable appearance-mix weights and an explainable UI status.
    """
    today = max(0.0, _number(pitches_today, 0.0))
    yesterday = max(0.0, _number(pitches_yesterday, 0.0))
    two_days = max(0.0, _number(pitches_two_days_ago, 0.0))
    consecutive = max(0, int(_number(consecutive_days, 0.0)))
    three_day = max(0.0, _number(three_day_pitches, today + yesterday + two_days))
    score = 100.0 - 2.0 * today - 0.9 * yesterday - 0.35 * two_days
    if consecutive >= 2:
        score -= 12.0
    if consecutive >= 3:
        score -= 20.0
    if yesterday >= 30:
        score -= 10.0
    if three_day >= 55:
        score -= 10.0
    score = clamp(score, 0.0, 100.0)
    # "Fresh" is intentionally stricter than a high numeric score. A pitcher
    # who already worked today, threw 20+ pitches yesterday, or is on a
    # multi-day run may still be usable, but should not be presented as rested.
    if score >= 75 and today == 0 and yesterday < 20 and consecutive < 2 and three_day < 40:
        status, label = "fresh", "Likely fresh"
    elif score >= 50:
        status, label = "available", "Likely available"
    elif score >= 25:
        status, label = "limited", "Limited"
    else:
        status, label = "unlikely", "Unlikely"
    reasons = []
    if today:
        reasons.append(f"{int(today)} pitches today")
    if yesterday:
        reasons.append(f"{int(yesterday)} pitches yesterday")
    if consecutive >= 2:
        reasons.append(f"worked {consecutive} straight days")
    if not reasons:
        reasons.append("no meaningful recent workload flag")
    return {"score": round(score, 1), "status": status, "label": label, "reasons": reasons}


def expected_starter_plate_appearances(lineup_order, expected_batters_faced,
                                       batters_faced_interval=None):
    """Estimate how many times one lineup slot will face the starter.

    A hitter in slot ``i`` sees the starter when the starter reaches batter
    numbers i, i+9, i+18, and so on. The BF interval supplies uncertainty
    instead of treating a point estimate as a hard cutoff.
    """
    expected_bf = max(1.0, _number(expected_batters_faced, LEAGUE_BF_PER_START))
    try:
        order = int(lineup_order)
        if not 1 <= order <= 9:
            raise ValueError
    except (TypeError, ValueError):
        return round(_lineup_weight(None) * clamp(expected_bf / 38.0, 0.25, 0.80), 3)
    low = high = None
    if isinstance(batters_faced_interval, (list, tuple)) and len(batters_faced_interval) >= 2:
        low = _number(batters_faced_interval[0])
        high = _number(batters_faced_interval[1])
    standard_deviation = max(2.2, ((high - low) / 2.563) if low is not None and high is not None and high > low else 4.0)
    expected = 0.0
    for turn in range(5):
        threshold = order + 9 * turn
        z_score = (threshold - 0.5 - expected_bf) / (standard_deviation * sqrt(2.0))
        expected += 0.5 * (1.0 - erf(z_score))
    return round(clamp(expected, 0.0, _lineup_weight(order)), 3)


def pitch_mix_evidence(side, pitcher_rate):
    """Calculate shrinkage-aware opponent evidence without hiding missing mix."""
    pitches = side.get("arsenal", [])[:5]
    total_usage = sum(max(0.0, _number(pitch.get("usage"), 0.0)) for pitch in pitches)
    if not pitches or total_usage <= 0:
        return {"rate": pitcher_rate, "coverage": 0.0, "effective_sample_size": 0.0, "hitters_covered": 0}

    weighted_rates = 0.0
    lineup_weights = 0.0
    weighted_coverage = 0.0
    effective_sample = 0.0
    hitters_covered = 0
    for batter in side.get("batters", []):
        batter_rate = 0.0
        covered_usage = 0.0
        batter_sample = 0.0
        for pitch in pitches:
            usage = max(0.0, _number(pitch.get("usage"), 0.0)) / total_usage
            stat = (batter.get("vs_pitches") or {}).get(pitch.get("code"))
            pa = _number((stat or {}).get("pa"), 0.0)
            strikeouts = _number((stat or {}).get("strikeouts"), 0.0)
            if pa <= 0:
                batter_rate += usage * pitcher_rate
                continue
            reliability = pa / (pa + 60.0)
            posterior = shrunk_rate(strikeouts, pa, pitcher_rate, 60.0)
            batter_rate += usage * posterior
            covered_usage += usage * reliability
            batter_sample += min(pa, 35.0) * usage
        lineup_weight = _lineup_weight(batter.get("lineup_order"))
        weighted_rates += batter_rate * lineup_weight
        weighted_coverage += covered_usage * lineup_weight
        effective_sample += batter_sample * lineup_weight / 4.2
        lineup_weights += lineup_weight
        hitters_covered += int(covered_usage > 0)
    if not lineup_weights:
        return {"rate": pitcher_rate, "coverage": 0.0, "effective_sample_size": 0.0, "hitters_covered": 0}
    return {
        "rate": weighted_rates / lineup_weights,
        "coverage": clamp(weighted_coverage / lineup_weights, 0.0, 1.0),
        "effective_sample_size": effective_sample,
        "hitters_covered": hitters_covered,
    }


def pitcher_process_evidence(side):
    """Use arsenal whiffs as a small stabilizer, never a standalone K signal."""
    arsenal = side.get("arsenal") or []
    swings = sum(max(0.0, _number(pitch.get("swings"), 0.0)) for pitch in arsenal)
    whiffs = sum(max(0.0, _number(pitch.get("whiffs"), 0.0)) for pitch in arsenal)
    chases = sum(max(0.0, _number(pitch.get("chases"), 0.0)) for pitch in arsenal)
    if swings <= 0:
        return {
            "available": False, "swings": 0, "whiffs": 0, "chases": 0,
            "whiff_per_swing": None, "reliability": 0.0, "adjustment": 0.0,
            "note": "Pitch-level whiff history is unavailable.",
        }
    whiff_rate = clamp(whiffs / swings, 0.0, 1.0)
    reliability = swings / (swings + 400.0)
    # Approximately 25% whiffs per swing is a neutral MLB anchor.  The effect
    # is deliberately capped below one percentage point until walk-forward
    # results establish that the process signal improves K calibration.
    adjustment = clamp((whiff_rate - .25) * .12 * reliability, -.008, .008)
    return {
        "available": True, "swings": swings, "whiffs": whiffs,
        "chases": chases, "whiff_per_swing": whiff_rate,
        "reliability": reliability, "adjustment": adjustment,
        "note": "Whiff rate is a small process check; it cannot create a favorable K grade by itself.",
    }


def _batter_pitch_k_evidence(batter, pitches, pitcher_rate):
    """Return one hitter's shrunk K response to the starter's pitch mix."""
    total_usage = sum(max(0.0, _number(pitch.get("usage"), 0.0)) for pitch in pitches)
    if not pitches or total_usage <= 0:
        return {"rate": pitcher_rate, "coverage": 0.0, "effective_pa": 0.0}
    rate = coverage = effective_pa = 0.0
    for pitch in pitches:
        usage = max(0.0, _number(pitch.get("usage"), 0.0)) / total_usage
        stat = (batter.get("vs_pitches") or {}).get(pitch.get("code")) or {}
        pa = max(0.0, _number(stat.get("pa"), 0.0))
        strikeouts = max(0.0, _number(stat.get("strikeouts"), 0.0))
        if pa <= 0:
            rate += usage * pitcher_rate
            continue
        reliability = pa / (pa + 75.0)
        rate += usage * shrunk_rate(strikeouts, pa, pitcher_rate, 75.0)
        coverage += usage * reliability
        effective_pa += usage * min(pa, 40.0)
    return {
        "rate": rate, "coverage": clamp(coverage, 0.0, 1.0),
        "effective_pa": effective_pa,
    }


def _extreme_k_shrinkage(rate, reliability):
    """Regress fragile tail estimates more than ordinary matchup estimates."""
    rate = clamp(rate, .06, .45)
    reliability = clamp(reliability, 0.0, 1.0)
    if rate > .30:
        shrink = .25 + .35 * (1.0 - reliability)
        return .30 + (rate - .30) * (1.0 - shrink)
    if rate < .15:
        shrink = .15 + .25 * (1.0 - reliability)
        return .15 - (.15 - rate) * (1.0 - shrink)
    return rate


def hitter_by_hitter_k_projection(side, pitcher_rate, expected_bf, bf_interval,
                                  process_adjustment=0.0):
    """Project each lineup slot, then aggregate only expected starter matchups."""
    pitches = (side.get("arsenal") or [])[:5]
    rows = []
    raw_total = adjusted_total = expected_pa_total = 0.0
    broad_total = exact_total = 0.0
    lineup_coverage = pitch_coverage = 0.0
    for batter in side.get("batters") or []:
        profile = batter.get("k_profile") or {}
        pa = max(0.0, _number(profile.get("pa"), 0.0))
        strikeouts = max(0.0, _number(profile.get("strikeouts"), 0.0))
        batter_rate = shrunk_rate(strikeouts, pa, LEAGUE_K_RATE, 140.0)
        batter_reliability = pa / (pa + 140.0) if pa else 0.0
        broad_adjustment = (.35 + .20 * batter_reliability) * (batter_rate - LEAGUE_K_RATE)
        exact = _batter_pitch_k_evidence(batter, pitches, pitcher_rate)
        exact_adjustment = clamp(
            .20 * exact["coverage"] * (exact["rate"] - pitcher_rate), -.010, .010,
        )
        raw_rate = clamp(
            pitcher_rate + broad_adjustment + exact_adjustment + process_adjustment,
            .06, .45,
        )
        reliability = clamp(
            .55 * batter_reliability + .25 * exact["coverage"]
            + .20 * (max(0.0, _number((side.get("workload") or {}).get("batters_faced"), 0.0)) /
                     (max(0.0, _number((side.get("workload") or {}).get("batters_faced"), 0.0)) + 250.0)),
            0.0, 1.0,
        )
        adjusted_rate = _extreme_k_shrinkage(raw_rate, reliability)
        expected_pa = expected_starter_plate_appearances(
            batter.get("lineup_order"), expected_bf, bf_interval,
        )
        raw_total += raw_rate * expected_pa
        adjusted_total += adjusted_rate * expected_pa
        broad_total += broad_adjustment * expected_pa
        exact_total += exact_adjustment * expected_pa
        expected_pa_total += expected_pa
        lineup_coverage += batter_reliability * expected_pa
        pitch_coverage += exact["coverage"] * expected_pa
        rows.append({
            "player_id": batter.get("id"), "name": batter.get("name"),
            "lineup_order": batter.get("lineup_order"), "expected_pa": expected_pa,
            "broad_k_rate": batter_rate, "pitch_k_rate": exact["rate"],
            "pitch_coverage": exact["coverage"], "raw_k_rate": raw_rate,
            "adjusted_k_rate": adjusted_rate,
            "expected_strikeouts": adjusted_rate * expected_pa,
        })
    if expected_pa_total <= 0:
        return {
            "rate": pitcher_rate, "raw_rate": pitcher_rate, "rows": [],
            "lineup_coverage": 0.0, "pitch_coverage": 0.0,
            "lineup_adjustment": 0.0, "pitch_adjustment": 0.0,
            "extreme_shrinkage": 0.0,
        }
    raw_rate = raw_total / expected_pa_total
    adjusted_rate = adjusted_total / expected_pa_total
    return {
        "rate": adjusted_rate, "raw_rate": raw_rate, "rows": rows,
        "lineup_coverage": clamp(lineup_coverage / expected_pa_total, 0.0, 1.0),
        "pitch_coverage": clamp(pitch_coverage / expected_pa_total, 0.0, 1.0),
        "lineup_adjustment": broad_total / expected_pa_total,
        "pitch_adjustment": exact_total / expected_pa_total,
        "extreme_shrinkage": adjusted_rate - raw_rate,
    }


def lineup_k_evidence(side):
    """Summarize the confirmed opponent's broader, handedness-matched K risk.

    Unlike pitch-mix evidence, this intentionally uses each hitter's full
    final-pitch PA history against the probable starter's throwing hand. It is
    the stable primary matchup feature; pitch-mix history remains a smaller
    refinement because exact pitch/velocity samples are naturally sparse.
    """
    weighted_rate = weighted_coverage = effective_sample = lineup_weights = 0.0
    hitters_covered = 0
    for batter in side.get("batters", []):
        profile = batter.get("k_profile") or {}
        pa = max(0.0, _number(profile.get("pa"), 0.0))
        strikeouts = max(0.0, _number(profile.get("strikeouts"), 0.0))
        lineup_weight = _lineup_weight(batter.get("lineup_order"))
        posterior = shrunk_rate(strikeouts, pa, LEAGUE_K_RATE, 120.0)
        reliability = pa / (pa + 120.0) if pa else 0.0
        weighted_rate += posterior * lineup_weight
        weighted_coverage += reliability * lineup_weight
        effective_sample += min(pa, 120.0) * lineup_weight / 4.2
        lineup_weights += lineup_weight
        hitters_covered += int(pa > 0)
    if not lineup_weights:
        return {
            "rate": LEAGUE_K_RATE, "coverage": 0.0,
            "effective_sample_size": 0.0, "hitters_covered": 0,
        }
    return {
        "rate": weighted_rate / lineup_weights,
        "coverage": clamp(weighted_coverage / lineup_weights, 0.0, 1.0),
        "effective_sample_size": effective_sample,
        "hitters_covered": hitters_covered,
    }


def hitter_k_risk(profile):
    """Describe one hitter's K susceptibility with a conservative PA prior."""
    profile = profile or {}
    pa = max(0.0, _number(profile.get("pa"), 0.0))
    strikeouts = max(0.0, _number(profile.get("strikeouts"), 0.0))
    posterior = shrunk_rate(strikeouts, pa, LEAGUE_K_RATE, 120.0)
    if pa < 30:
        label, tone = "limited K sample", "neutral"
    elif posterior >= .255:
        label, tone = "high K risk", "good"
    elif posterior <= .195:
        label, tone = "low K risk", "bad"
    else:
        label, tone = "average K risk", "neutral"
    return {
        "label": label,
        "tone": tone,
        "posterior": posterior,
        "reliability": pa / (pa + 120.0) if pa else 0.0,
    }


def workload_read(side, estimate):
    """Return a descriptive starter-leash read; it is not an extra K adjustment."""
    expected_batters_faced = estimate["expected_batters_faced"]
    appearances = estimate["appearances"]
    source = estimate["source"]
    starts = [row for row in side.get("appearance_history", []) if row.get("is_start")]
    if starts:
        season_bf = sum(_number(row.get("batters_faced"), 0.0) for row in starts) / len(starts)
        season_pitches = sum(_number(row.get("pitches"), 0.0) for row in starts) / len(starts)
        recent = starts[-3:]
        recent_bf = sum(_number(row.get("batters_faced"), 0.0) for row in recent) / len(recent)
        recent_pitches = sum(_number(row.get("pitches"), 0.0) for row in recent) / len(recent)
    else:
        workload = side.get("workload") or {}
        season_bf = (_number(workload.get("batters_faced"), 0.0) / appearances) if appearances else expected_batters_faced
        season_pitches = (_number(workload.get("pitches"), 0.0) / appearances) if appearances else 0.0
        recent_count = max(1, int(_number(workload.get("recent_appearances"), 0.0)))
        recent_bf = (_number(workload.get("recent_batters_faced"), 0.0) / recent_count) if workload else season_bf
        recent_pitches = (_number(workload.get("recent_pitches"), 0.0) / recent_count) if workload else season_pitches
    delta = recent_bf - season_bf
    if appearances < 3:
        label, tone = "limited starter history", "neutral"
    elif estimate["early_exit_risk"] >= .38 or delta <= -2.0 or recent_pitches <= season_pitches - 12:
        label, tone = "recently shortened", "bad"
    elif delta >= 2.0 and recent_pitches >= season_pitches + 8:
        label, tone = "recently extended", "good"
    else:
        label, tone = "stable workload", "neutral"
    return {
        "label": label,
        "tone": tone,
        "expected_batters_faced": expected_batters_faced,
        "season_batters_faced": season_bf,
        "recent_batters_faced": recent_bf,
        "season_pitches": season_pitches,
        "recent_pitches": recent_pitches,
        "expected_pitches": estimate["expected_pitches"],
        "pitches_interval": estimate["pitches_interval"],
        "expected_outs": estimate["expected_outs"],
        "outs_interval": estimate["outs_interval"],
        "early_exit_risk": estimate["early_exit_risk"],
        "pitches_per_batter": estimate["pitches_per_batter"],
        "appearances": appearances,
        "source": source,
    }


def k_opportunity_read(k_rate, expected_ks, lineup_rate, workload,
                       downside_probability=None, early_exit_probability=None):
    """Classify K environment separately from evidence quality or a prop line."""
    downside = _number(downside_probability, .5)
    early_exit = _number(early_exit_probability, workload.get("early_exit_risk", .35))
    if (expected_ks >= 6.4 and k_rate >= .255 and workload["tone"] != "bad"
            and downside <= .22 and early_exit <= .28):
        label, tone = "high K environment", "good"
    elif (expected_ks >= 5.4 and k_rate >= .225 and workload["tone"] != "bad"
          and downside <= .32 and early_exit <= .36):
        label, tone = "favorable K environment", "good"
    elif expected_ks >= 5.4 and k_rate >= .225:
        label, tone = "volatile K ceiling", "neutral"
    elif expected_ks <= 4.4 or k_rate <= .185 or workload["tone"] == "bad":
        label, tone = "limited K environment", "bad"
    else:
        label, tone = "neutral K environment", "neutral"
    return {
        "label": label,
        "tone": tone,
        "k_rate": k_rate,
        "lineup_k_rate": lineup_rate,
        "expected_strikeouts": expected_ks,
        "probability_three_or_fewer": downside,
        "early_exit_probability": early_exit,
    }


def k_data_grade(lineup_confirmed, appearances, lineup_evidence, pitch_evidence, stale_data):
    """Grade input completeness without suppressing a useful research read."""
    if lineup_confirmed and not stale_data and appearances >= 10 and lineup_evidence["coverage"] >= .55 and pitch_evidence["coverage"] >= .35:
        grade = "A"
    elif lineup_confirmed and not stale_data and appearances >= 6 and lineup_evidence["coverage"] >= .35:
        grade = "B"
    elif lineup_confirmed and appearances >= 3 and lineup_evidence["hitters_covered"] >= 5:
        grade = "C"
    else:
        grade = "D"
    blockers = []
    if not lineup_confirmed:
        blockers.append("confirmed lineup")
    if stale_data:
        blockers.append("fresh matchup sync")
    if appearances < 6:
        blockers.append("starter history")
    if lineup_evidence["coverage"] < .35:
        blockers.append("opponent K sample")
    if pitch_evidence["coverage"] < .35:
        blockers.append("pitch-mix sample")
    return {"grade": grade, "blockers": blockers}


def lineup_workload_context(side):
    """Summarize opponent patience, traffic, power, and out conversion.

    These are full-season Gameday outcomes for the listed hitters, shrunk to
    league priors and weighted by batting order. Missing data remains neutral.
    """
    totals = {"ppa": 0.0, "on_base": 0.0, "out_rate": 0.0, "tb_per_pa": 0.0}
    weight_total = coverage = effective_pa = 0.0
    for batter in side.get("batters", []):
        discipline = batter.get("discipline") or {}
        pa = max(0.0, _number(discipline.get("plate_appearances"), 0.0))
        pitches = max(0.0, _number(discipline.get("pitches_seen"), 0.0))
        walks = max(0.0, _number(discipline.get("walks"), 0.0))
        hbp = max(0.0, _number(discipline.get("hit_by_pitch"), 0.0))
        hits = max(0.0, _number(discipline.get("hits"), 0.0))
        outs = max(0.0, _number(discipline.get("outs"), 0.0))
        total_bases = max(0.0, _number(discipline.get("total_bases"), 0.0))
        prior = 120.0
        ppa = (pitches + LEAGUE_PITCHES_PER_PA * prior) / (pa + prior)
        on_base = (hits + walks + hbp + LEAGUE_ON_BASE_PROXY * prior) / (pa + prior)
        out_rate = (outs + LEAGUE_OUT_RATE * prior) / (pa + prior)
        tb_per_pa = (total_bases + LEAGUE_TOTAL_BASES_PER_PA * prior) / (pa + prior)
        weight = _lineup_weight(batter.get("lineup_order"))
        reliability = pa / (pa + prior) if pa else 0.0
        for key, value in (("ppa", ppa), ("on_base", on_base), ("out_rate", out_rate), ("tb_per_pa", tb_per_pa)):
            totals[key] += value * weight
        coverage += reliability * weight
        effective_pa += min(pa, prior) * weight / 4.2
        weight_total += weight
    if not weight_total:
        return {
            "pitches_per_pa": LEAGUE_PITCHES_PER_PA, "on_base_rate": LEAGUE_ON_BASE_PROXY,
            "out_rate": LEAGUE_OUT_RATE, "total_bases_per_pa": LEAGUE_TOTAL_BASES_PER_PA,
            "coverage": 0.0, "effective_pa": 0.0,
        }
    return {
        "pitches_per_pa": totals["ppa"] / weight_total,
        "on_base_rate": totals["on_base"] / weight_total,
        "out_rate": totals["out_rate"] / weight_total,
        "total_bases_per_pa": totals["tb_per_pa"] / weight_total,
        "coverage": clamp(coverage / weight_total, 0.0, 1.0),
        "effective_pa": effective_pa,
    }


def _team_run_expectation(side):
    """Derive a small team-scoring context from the free ESPN game market."""
    context = side.get("market_context") or {}
    total = _number(context.get("total"))
    expected = total / 2.0 if total is not None else LEAGUE_GAME_TOTAL / 2.0
    favorite = context.get("favorite")
    probability = _number(context.get("favorite_probability"), .55)
    if favorite:
        favorite_edge = clamp((probability - .5) / .20, 0.0, 1.0)
        expected += (.35 if favorite == side.get("batting_team") else -.35) * favorite_edge
    return clamp(expected, 2.5, 6.5)


def _apply_matchup_workload(side, baseline):
    lineup = lineup_workload_context(side)
    coverage = lineup["coverage"]
    pitcher_ppa = baseline["pitcher_pitches_per_batter"]
    matchup_ppa = clamp(
        pitcher_ppa + .55 * coverage * (lineup["pitches_per_pa"] - LEAGUE_PITCHES_PER_PA),
        3.15, 4.65,
    )
    team_runs = _team_run_expectation(side)
    run_environment_adjustment = clamp((team_runs - LEAGUE_GAME_TOTAL / 2.0) * .006, -.012, .012)
    matchup_on_base = clamp(
        .65 * baseline["pitcher_baserunner_rate"] + .35 * lineup["on_base_rate"]
        + run_environment_adjustment,
        .24, .43,
    )
    matchup_out_rate = clamp(
        .65 * baseline["pitcher_out_rate"] + .35 * lineup["out_rate"]
        - .50 * run_environment_adjustment,
        .54, .78,
    )
    patience_hook = .18 * (matchup_ppa - pitcher_ppa)
    traffic_hook = .85 * (matchup_on_base - baseline["pitcher_baserunner_rate"])
    power_hook = .30 * coverage * (lineup["total_bases_per_pa"] - LEAGUE_TOTAL_BASES_PER_PA)
    run_hook = .025 * (team_runs - LEAGUE_GAME_TOTAL / 2.0)
    hook_adjustment = clamp(patience_hook + traffic_hook + power_hook + run_hook, -.12, .18)
    early_exit_risk = clamp(baseline["base_early_exit_risk"] + hook_adjustment, .05, .75)
    adjusted_pitch_budget = clamp(
        baseline["baseline_pitch_budget"] - 32.0 * hook_adjustment,
        35.0, 115.0,
    )
    override = side.get("workload_override") or {}
    manual_pitch_limit = _number(override.get("pitch_limit"))
    if manual_pitch_limit is not None:
        adjusted_pitch_budget = min(adjusted_pitch_budget, manual_pitch_limit)
    pitch_derived_bf = adjusted_pitch_budget / matchup_ppa
    expected_bf = .20 * baseline["direct_bf"] + .80 * pitch_derived_bf
    expected_outs = .25 * baseline["historical_outs"] + .75 * (expected_bf * matchup_out_rate)
    bf_spread = max(3.0, baseline["bf_spread"] + (1.0 - coverage) * .75 + abs(hook_adjustment) * 4.0)
    outs_spread = max(3.5, baseline["outs_spread"] + (1.0 - coverage) * .75)
    pitch_spread = baseline["pitch_spread"]
    pitch_high = min(125.0, adjusted_pitch_budget + 1.28 * pitch_spread)
    if manual_pitch_limit is not None:
        pitch_high = min(pitch_high, manual_pitch_limit)
    stages = {
        "baseline_pitch_budget": baseline["baseline_pitch_budget"],
        "manual_pitch_limit": manual_pitch_limit,
        "matchup_pitch_budget": adjusted_pitch_budget,
        "pitcher_pitches_per_batter": pitcher_ppa,
        "lineup_pitches_per_pa": lineup["pitches_per_pa"],
        "matchup_pitches_per_batter": matchup_ppa,
        "pitcher_baserunner_rate": baseline["pitcher_baserunner_rate"],
        "lineup_on_base_rate": lineup["on_base_rate"],
        "matchup_baserunner_rate": matchup_on_base,
        "matchup_out_rate": matchup_out_rate,
        "team_run_expectation": team_runs,
        "baseline_early_exit_risk": baseline["base_early_exit_risk"],
        "matchup_hook_adjustment": hook_adjustment,
        "lineup_coverage": coverage,
        "lineup_effective_pa": lineup["effective_pa"],
        "pitch_derived_bf": pitch_derived_bf,
        "historical_bf_stabilizer": baseline["direct_bf"],
    }
    return {
        "expected_batters_faced": expected_bf,
        "batters_faced_interval": [max(8.0, expected_bf - 1.28 * bf_spread), min(36.0, expected_bf + 1.28 * bf_spread)],
        "expected_pitches": adjusted_pitch_budget,
        "pitches_interval": [max(20.0, adjusted_pitch_budget - 1.28 * pitch_spread), max(20.0, pitch_high)],
        "expected_outs": clamp(expected_outs, 3.0, 27.0),
        "outs_interval": [max(3.0, expected_outs - 1.28 * outs_spread), min(27.0, expected_outs + 1.28 * outs_spread)],
        "early_exit_risk": early_exit_risk,
        "pitches_per_batter": matchup_ppa,
        "appearances": baseline["appearances"],
        "source": baseline["source"] + " plus opponent workload context",
        "lineup_workload_context": lineup,
        "workload_stages": stages,
    }


def _workload_estimate(side):
    starts = [row for row in side.get("appearance_history", []) if row.get("is_start")]
    starts = [row for row in starts if _number(row.get("batters_faced"), 0.0) > 0]
    if starts:
        recent = starts[-5:]

        def blended(key, prior, prior_weight=3.0):
            season_values = [_number(row.get(key)) for row in starts]
            season_values = [value for value in season_values if value is not None]
            recent_values = [_number(row.get(key)) for row in recent]
            recent_values = [value for value in recent_values if value is not None]
            numerator = prior * prior_weight + sum(season_values) + .65 * sum(recent_values)
            denominator = prior_weight + len(season_values) + .65 * len(recent_values)
            return numerator / denominator if denominator else prior

        batters = [_number(row.get("batters_faced"), 0.0) for row in starts]
        total_bf = sum(batters)
        pitch_rows = [row for row in starts if row.get("pitches") is not None]
        pitch_bf = sum(_number(row.get("batters_faced"), 0.0) for row in pitch_rows)
        total_pitches = sum(_number(row.get("pitches"), 0.0) for row in pitch_rows)
        command_rows = [row for row in starts if row.get("hits_allowed") is not None and row.get("walks_allowed") is not None]
        command_bf = sum(_number(row.get("batters_faced"), 0.0) for row in command_rows)
        hits_walks = sum(_number(row.get("hits_allowed"), 0.0) + _number(row.get("walks_allowed"), 0.0) for row in command_rows)
        out_rows = [row for row in starts if row.get("outs") is not None]
        out_bf = sum(_number(row.get("batters_faced"), 0.0) for row in out_rows)
        total_outs = sum(_number(row.get("outs"), 0.0) for row in out_rows)
        baseline_pitches = blended("pitches", 85.0)
        direct_bf = blended("batters_faced", LEAGUE_BF_PER_START)
        historical_outs = blended("outs", 16.5)
        pitch_values = [_number(row.get("pitches"), baseline_pitches) for row in starts]
        outs_values = [_number(row.get("outs"), historical_outs) for row in starts]
        shortened = sum(int(
            _number(row.get("batters_faced"), 0) < 18
            or (row.get("outs") is not None and _number(row.get("outs"), 0) < 15)
            or (row.get("pitches") is not None and _number(row.get("pitches"), 0) < 70)
        ) for row in starts)
        baseline = {
            "baseline_pitch_budget": baseline_pitches,
            "direct_bf": direct_bf,
            "historical_outs": historical_outs,
            "pitcher_pitches_per_batter": (total_pitches + LEAGUE_PITCHES_PER_PA * 80.0) / (pitch_bf + 80.0),
            "pitcher_baserunner_rate": shrunk_rate(hits_walks, command_bf, LEAGUE_ON_BASE_PROXY, 100.0),
            "pitcher_out_rate": shrunk_rate(total_outs, out_bf, LEAGUE_OUT_RATE, 100.0),
            "base_early_exit_risk": (shortened + 1.5) / (len(starts) + 6.0),
            "bf_spread": max(2.75, sqrt(sum((value - direct_bf) ** 2 for value in batters) / max(1, len(batters) - 1))),
            "pitch_spread": max(10.0, sqrt(sum((value - baseline_pitches) ** 2 for value in pitch_values) / max(1, len(pitch_values) - 1))),
            "outs_spread": max(3.0, sqrt(sum((value - historical_outs) ** 2 for value in outs_values) / max(1, len(outs_values) - 1))),
            "appearances": len(starts),
            "source": "start-only pitch budget",
        }
        return _apply_matchup_workload(side, baseline)

    workload = side.get("workload") or {}
    appearances = int(_number(workload.get("appearances"), 0))
    batters_faced = _number(workload.get("batters_faced"), 0.0)
    if appearances and batters_faced:
        baseline_pitches = _number(workload.get("pitches"), 0.0) / appearances
        direct_bf = batters_faced / appearances
        historical_outs = _number(workload.get("outs"), 0.0) / appearances
        baseline = {
            "baseline_pitch_budget": baseline_pitches or 85.0,
            "direct_bf": direct_bf,
            "historical_outs": historical_outs or 16.5,
            "pitcher_pitches_per_batter": (_number(workload.get("pitches"), 0.0) + LEAGUE_PITCHES_PER_PA * 80.0) / (batters_faced + 80.0),
            "pitcher_baserunner_rate": LEAGUE_ON_BASE_PROXY,
            "pitcher_out_rate": (historical_outs / direct_bf) if direct_bf else LEAGUE_OUT_RATE,
            "base_early_exit_risk": .35, "bf_spread": 3.5, "pitch_spread": 16.0,
            "outs_spread": 4.5, "appearances": appearances,
            "source": "mixed-role aggregate fallback",
        }
        return _apply_matchup_workload(side, baseline)
    return _apply_matchup_workload(side, {
        "baseline_pitch_budget": 85.0, "direct_bf": LEAGUE_BF_PER_START,
        "historical_outs": 16.5, "pitcher_pitches_per_batter": LEAGUE_PITCHES_PER_PA,
        "pitcher_baserunner_rate": LEAGUE_ON_BASE_PROXY, "pitcher_out_rate": LEAGUE_OUT_RATE,
        "base_early_exit_risk": .35, "bf_spread": 4.5, "pitch_spread": 18.0,
        "outs_spread": 5.0, "appearances": 0, "source": "league prior",
    })


def starter_workload_scenarios(estimate):
    """Create an asymmetric workload mixture while preserving the point mean."""
    expected = clamp(_number(estimate.get("expected_batters_faced"), LEAGUE_BF_PER_START), 8.0, 36.0)
    early_exit = clamp(_number(estimate.get("early_exit_risk"), .35), .05, .75)
    manual_limit = _number((estimate.get("workload_stages") or {}).get("manual_pitch_limit"))
    short_probability = clamp(.08 + .36 * early_exit, .10, .34)
    extended_probability = clamp(.18 - .18 * early_exit, .05, .16)
    if manual_limit is not None and manual_limit < 80:
        short_probability = clamp(short_probability + .08, .10, .42)
        extended_probability = max(.03, extended_probability - .06)
    normal_probability = max(.20, 1.0 - short_probability - extended_probability)
    total_probability = short_probability + normal_probability + extended_probability
    short_probability /= total_probability
    normal_probability /= total_probability
    extended_probability /= total_probability
    short_bf = max(8.0, expected - 7.0)
    extended_bf = min(36.0, expected + (2.5 if manual_limit is not None else 4.0))
    # Solve the normal scenario center so the mixture retains the transparent
    # point workload estimate instead of silently lowering every projection.
    normal_bf = (
        expected - short_probability * short_bf - extended_probability * extended_bf
    ) / normal_probability
    normal_bf = clamp(normal_bf, 12.0, 32.0)
    return [
        {
            "key": "short", "label": "Short/ineffective outing",
            "probability": short_probability, "batters_faced": short_bf,
            "spread": 2.2,
        },
        {
            "key": "normal", "label": "Normal outing",
            "probability": normal_probability, "batters_faced": normal_bf,
            "spread": 2.4,
        },
        {
            "key": "extended", "label": "Extended outing",
            "probability": extended_probability, "batters_faced": extended_bf,
            "spread": 2.2,
        },
    ]


def pitcher_performance_outlook(side, estimate):
    """Describe workload, command, and run suppression separately from Ks."""
    starts = [row for row in side.get("appearance_history", []) if row.get("is_start")]
    innings = sum(_number(row.get("outs"), 0.0) for row in starts) / 3.0
    walks = sum(_number(row.get("walks_allowed"), 0.0) for row in starts)
    hits = sum(_number(row.get("hits_allowed"), 0.0) for row in starts)
    earned_runs = sum(_number(row.get("earned_runs"), 0.0) for row in starts)
    command_sample = any(row.get("walks_allowed") is not None for row in starts)
    run_sample = any(row.get("earned_runs") is not None for row in starts)
    whip = (walks + hits) / innings if command_sample and innings else None
    earned_run_rate = earned_runs * 9.0 / innings if run_sample and innings else None
    early_exit = estimate["early_exit_risk"]
    if early_exit >= .38:
        exit_label, exit_tone = "elevated early-exit risk", "bad"
    elif early_exit <= .20:
        exit_label, exit_tone = "lower early-exit risk", "good"
    else:
        exit_label, exit_tone = "typical early-exit risk", "neutral"
    if whip is None:
        command_label, command_tone = "command history pending", "neutral"
    elif whip <= 1.15:
        command_label, command_tone = "limits baserunners", "good"
    elif whip >= 1.45:
        command_label, command_tone = "elevated traffic risk", "bad"
    else:
        command_label, command_tone = "average traffic risk", "neutral"
    if earned_run_rate is None:
        run_label, run_tone = "run history pending", "neutral"
    elif earned_run_rate <= 3.5:
        run_label, run_tone = "strong run suppression", "good"
    elif earned_run_rate >= 5.0:
        run_label, run_tone = "elevated run risk", "bad"
    else:
        run_label, run_tone = "average run suppression", "neutral"
    return {
        "expected_outs": estimate["expected_outs"],
        "outs_interval": estimate["outs_interval"],
        "expected_innings": estimate["expected_outs"] / 3.0,
        "expected_pitches": estimate["expected_pitches"],
        "pitches_interval": estimate["pitches_interval"],
        "early_exit_risk": early_exit,
        "early_exit_label": exit_label,
        "early_exit_tone": exit_tone,
        "whip_history": whip,
        "command_label": command_label,
        "command_tone": command_tone,
        "earned_runs_per_nine": earned_run_rate,
        "run_suppression_label": run_label,
        "run_suppression_tone": run_tone,
        "matchup_pitches_per_batter": estimate["workload_stages"]["matchup_pitches_per_batter"],
        "matchup_baserunner_rate": estimate["workload_stages"]["matchup_baserunner_rate"],
        "matchup_out_rate": estimate["workload_stages"]["matchup_out_rate"],
        "team_run_expectation": estimate["workload_stages"]["team_run_expectation"],
        "starts": len(starts),
    }


def evaluate_market(distribution, market):
    if not market:
        return {}
    no_vig = no_vig_probabilities(market.get("over_price"), market.get("under_price")) if market.get("platform_type") == "sportsbook" else None
    probability_over = distribution.get("probability_over")
    probability_under = distribution.get("probability_under")
    return {
        "market_snapshot_id": market.get("market_snapshot_id"),
        "provider": market.get("provider"),
        "captured_at": market.get("captured_at"),
        "platform_type": market.get("platform_type"),
        "line": market.get("line"),
        "over_price": market.get("over_price"),
        "under_price": market.get("under_price"),
        "no_vig_over": no_vig["over"] if no_vig else None,
        "no_vig_under": no_vig["under"] if no_vig else None,
        "expected_value_over": expected_value(probability_over, market.get("over_price")) if market.get("platform_type") == "sportsbook" else None,
        "expected_value_under": expected_value(probability_under, market.get("under_price")) if market.get("platform_type") == "sportsbook" else None,
    }


def pitcher_k_market_assessment(market, projection, distribution, data_grade,
                                lineup_confirmed, early_exit_probability,
                                lineup_coverage):
    """Apply strict research guardrails to a manually entered strikeout line."""
    line = _number((market or {}).get("line"))
    if line is None:
        return {
            "available": False, "label": "No line entered", "tone": "neutral",
            "direction": None, "edge": None,
            "reasons": ["Enter a current sportsbook line to evaluate the gap."],
        }
    edge = projection - line
    probability_over = _number(distribution.get("probability_over"), 0.0)
    probability_under = _number(distribution.get("probability_under"), 0.0)
    high_line = line >= 6.5
    required_edge = .8 if high_line else .6
    required_over = .60 if high_line else .58
    common_blockers = []
    if not lineup_confirmed:
        common_blockers.append("lineup is not confirmed")
    if data_grade not in ("A", "B"):
        common_blockers.append(f"data grade is {data_grade}")
    if lineup_coverage < (.45 if high_line else .35):
        common_blockers.append("opponent K coverage is limited")
    if early_exit_probability > (.25 if high_line else .32):
        common_blockers.append("early-exit downside is elevated")
    over_blockers = list(common_blockers)
    if high_line and edge < required_edge:
        over_blockers.append("a high line requires at least a 0.8-K model cushion")
    if not over_blockers and edge >= required_edge and probability_over >= required_over:
        return {
            "available": True, "label": "Favorable over research", "tone": "good",
            "direction": "over", "edge": edge, "probability_over": probability_over,
            "probability_under": probability_under,
            "reasons": ["Projection, workload downside, and evidence clear the conservative gate."],
        }
    # The larger cushion is an over-only guardrail. A high sportsbook line
    # can still create a conservative under research signal.
    if not common_blockers and edge <= -.6 and probability_under >= .60:
        return {
            "available": True, "label": "Favorable under research", "tone": "bad",
            "direction": "under", "edge": edge, "probability_over": probability_over,
            "probability_under": probability_under,
            "reasons": ["The line sits meaningfully above the conservative distribution."],
        }
    reasons = over_blockers or [
        "The projection is too close to the line after workload and K-rate uncertainty."
    ]
    return {
        "available": True, "label": "No clear edge", "tone": "neutral",
        "direction": None, "edge": edge, "probability_over": probability_over,
        "probability_under": probability_under, "reasons": reasons,
    }


def pitcher_k_projection(side, market=None, as_of=None, scheduled_start=None):
    """Return a conservative, versioned pitcher-K research forecast."""
    workload = side.get("workload") or {}
    bf = _number(workload.get("batters_faced"), 0.0)
    strikeouts = _number(workload.get("strikeouts"), 0.0)
    recent_bf = _number(workload.get("recent_batters_faced"), 0.0)
    recent_k = _number(workload.get("recent_strikeouts"), 0.0)
    pitcher_rate = shrunk_rate(strikeouts + 0.25 * recent_k, bf + 0.25 * recent_bf, LEAGUE_K_RATE, 80.0)
    pitch_evidence = pitch_mix_evidence(side, pitcher_rate)
    lineup_evidence = lineup_k_evidence(side)
    workload_estimate = _workload_estimate(side)
    expected_bf = workload_estimate["expected_batters_faced"]
    bf_low, bf_high = workload_estimate["batters_faced_interval"]
    process_evidence = pitcher_process_evidence(side)
    hitter_projection = hitter_by_hitter_k_projection(
        side, pitcher_rate, expected_bf, [bf_low, bf_high],
        process_adjustment=process_evidence["adjustment"],
    )
    # Every expected starter plate appearance is now modeled individually.
    # Tail estimates receive extra regression; ordinary estimates are left
    # untouched. This specifically addresses the observed instability above
    # seven projected strikeouts without suppressing the middle of the board.
    k_rate = clamp(hitter_projection["rate"], 0.06, 0.45)
    lineup_adjustment = hitter_projection["lineup_adjustment"]
    pitch_adjustment = hitter_projection["pitch_adjustment"]
    extreme_shrinkage = hitter_projection["extreme_shrinkage"]
    appearances = workload_estimate["appearances"]
    expected_ks = k_rate * expected_bf
    leash = workload_read(side, workload_estimate)
    performance_outlook = pitcher_performance_outlook(side, workload_estimate)
    line = _number((market or {}).get("line"))
    rate_strength = clamp(80.0 + .15 * bf + .25 * lineup_evidence["effective_sample_size"], 80.0, 240.0)
    workload_scenarios = starter_workload_scenarios(workload_estimate)
    distribution = workload_k_distribution(
        k_rate, expected_bf, bf_low, bf_high, line, rate_strength=rate_strength,
        workload_scenarios=workload_scenarios,
    )
    early_exit_probability = next(
        (scenario["probability"] for scenario in workload_scenarios if scenario["key"] == "short"),
        workload_estimate["early_exit_risk"],
    )
    performance_outlook["early_exit_probability"] = early_exit_probability
    if early_exit_probability >= .25:
        performance_outlook["early_exit_label"] = "elevated short-outing weight"
        performance_outlook["early_exit_tone"] = "bad"
    elif early_exit_probability <= .15:
        performance_outlook["early_exit_label"] = "lower short-outing weight"
        performance_outlook["early_exit_tone"] = "good"
    else:
        performance_outlook["early_exit_label"] = "typical short-outing weight"
        performance_outlook["early_exit_tone"] = "neutral"
    opportunity = k_opportunity_read(
        k_rate, expected_ks, lineup_evidence["rate"], leash,
        distribution["probability_three_or_fewer"], early_exit_probability,
    )
    try:
        workload_challenger = shadow_workload_prediction(side)
    except (KeyError, TypeError, ValueError, OSError) as error:
        workload_challenger = {
            "available": False, "status": "unavailable",
            "message": f"Workload challenger unavailable: {error}",
        }
    k_challenger = {"available": False, "status": workload_challenger.get("status", "collecting")}
    if workload_challenger.get("available"):
        ml_bf = workload_challenger["expected_batters_faced"]
        ml_low, ml_high = workload_challenger["batters_faced_interval"]
        ml_scenarios = starter_workload_scenarios({
            **workload_estimate, "expected_batters_faced": ml_bf,
            "batters_faced_interval": [ml_low, ml_high],
            "early_exit_risk": workload_challenger.get(
                "early_exit_probability", workload_estimate["early_exit_risk"],
            ),
        })
        ml_distribution = workload_k_distribution(
            k_rate, ml_bf, ml_low, ml_high, line, rate_strength=rate_strength,
            workload_scenarios=ml_scenarios,
        )
        k_challenger = {
            "available": True, "status": "shadow",
            "projection": k_rate * ml_bf,
            "median": ml_distribution["median"],
            "interval_low": ml_distribution["interval_low"],
            "interval_high": ml_distribution["interval_high"],
            "probability_over": ml_distribution.get("probability_over"),
            "probability_under": ml_distribution.get("probability_under"),
            "probability_push": ml_distribution.get("probability_push"),
            "probability_three_or_fewer": ml_distribution["probability_three_or_fewer"],
            "milestone_probabilities": ml_distribution["milestones"],
            "distribution": ml_distribution["distribution"],
            "note": "Uses the shadow ML workload with the same transparent matchup K rate.",
        }
    market_read = evaluate_market(distribution, market)
    lineup_confirmed = side.get("lineup_confirmed") is True
    data_freshness_seconds = _number(side.get("data_freshness_seconds"))
    stale_data = data_freshness_seconds is None or data_freshness_seconds > 36 * 60 * 60
    stale_market = False
    if market and market.get("captured_at"):
        try:
            market_time = datetime.fromisoformat(str(market["captured_at"]).replace("Z", "+00:00"))
            as_of_time = datetime.fromisoformat(str(as_of or datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00"))
            stale_market = (as_of_time - market_time).total_seconds() > 6 * 60 * 60
        except (TypeError, ValueError):
            stale_market = True
    pitch_coverage = pitch_evidence["coverage"]
    data_grade = k_data_grade(lineup_confirmed, appearances, lineup_evidence, pitch_evidence, stale_data)
    if data_grade["grade"] == "A" and pitch_evidence["effective_sample_size"] >= 35:
        confidence = "medium"
    elif data_grade["grade"] in ("A", "B", "C"):
        confidence = "low"
    else:
        confidence = "insufficient"
    market_assessment = pitcher_k_market_assessment(
        market, expected_ks, distribution, data_grade["grade"], lineup_confirmed,
        early_exit_probability, lineup_evidence["coverage"],
    )
    spotlight_qualified = bool(
        lineup_confirmed and data_grade["grade"] in ("A", "B")
        and opportunity["tone"] == "good"
        and early_exit_probability <= .28
        and distribution["probability_three_or_fewer"] <= .25
        and (line is None or market_assessment.get("direction") == "over")
    )
    if not lineup_confirmed:
        decision = "WAIT_FOR_LINEUP"
    elif stale_data:
        decision = "STALE_DATA"
    elif market is None:
        decision = "NEED_PRICE"
    elif stale_market:
        decision = "STALE_MARKET"
    elif market.get("platform_type") != "sportsbook" or market.get("over_price") is None or market.get("under_price") is None:
        decision = "RESEARCH_ONLY"
    elif data_grade["grade"] == "D":
        decision = "PASS_LOW_DATA"
    else:
        # No betting recommendation is emitted until MODEL_CARD documents a
        # successful out-of-sample validation gate.
        decision = "RESEARCH_ONLY_UNVALIDATED"
    factors = [
        f"Pitcher baseline K rate {pitcher_rate:.1%}",
        f"Lineup K-risk adjustment {lineup_adjustment:+.1%}",
        f"Pitch-mix adjustment {pitch_adjustment:+.1%}",
        f"Pitch-process adjustment {process_evidence['adjustment']:+.1%}",
        f"Extreme-rate regression {extreme_shrinkage:+.1%}",
        f"Expected workload {expected_bf:.1f} batters faced",
        f"Expected pitch budget {workload_estimate['expected_pitches']:.0f} pitches",
        f"Opponent-adjusted efficiency {workload_estimate['pitches_per_batter']:.2f} pitches per batter",
        f"Opponent workload coverage {workload_estimate['lineup_workload_context']['coverage']:.0%}",
        f"Early-exit risk {workload_estimate['early_exit_risk']:.0%}",
    ]
    downside_risks = []
    if distribution["probability_three_or_fewer"] >= .25:
        downside_risks.append(
            f"{distribution['probability_three_or_fewer']:.0%} modeled chance of three or fewer strikeouts"
        )
    if early_exit_probability >= .24:
        downside_risks.append(f"{early_exit_probability:.0%} short-outing scenario weight")
    if lineup_evidence["coverage"] < .45:
        downside_risks.append(f"only {lineup_evidence['coverage']:.0%} opponent K coverage")
    if extreme_shrinkage <= -.003:
        downside_risks.append("the raw matchup K rate required extra tail regression")
    if not process_evidence["available"]:
        downside_risks.append("pitch-level whiff process is unavailable")
    if (workload_estimate.get("workload_stages") or {}).get("manual_pitch_limit") is not None:
        downside_risks.append("a manual pitch restriction is active")
    if not downside_risks:
        downside_risks.append("ordinary start-to-start K-rate and workload variance remains")
    missing = []
    if not lineup_confirmed:
        missing.append("confirmed lineup")
    if not side.get("appearance_history"):
        missing.append("start-only appearance history")
    if lineup_evidence["coverage"] < .35:
        missing.append("broader opponent K sample")
    if pitch_coverage < .35:
        missing.append("pitch-mix K sample")
    if market is None:
        missing.append("priced market")
    elif stale_market:
        missing.append("fresh market price")
    if stale_data:
        missing.append("fresh matchup sync")
    return {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "as_of": as_of or datetime.now(timezone.utc).isoformat(),
        "scheduled_start": scheduled_start,
        "player_id": side.get("pitcher_id"),
        "player_name": side.get("pitcher"),
        "pitcher_throws": side.get("pitcher_throws"),
        "prop_type": "pitcher_strikeouts",
        "projection": expected_ks,
        "k_rate": k_rate,
        "expected_batters_faced": expected_bf,
        "expected_pitches": workload_estimate["expected_pitches"],
        "batters_faced_interval": [bf_low, bf_high],
        "median": distribution["median"],
        "interval_low": distribution["interval_low"],
        "interval_high": distribution["interval_high"],
        "probability_over": distribution.get("probability_over"),
        "probability_under": distribution.get("probability_under"),
        "probability_push": distribution.get("probability_push"),
        "fair_over_price": distribution.get("fair_over_price"),
        "fair_under_price": distribution.get("fair_under_price"),
        "distribution": distribution["distribution"],
        "probability_three_or_fewer": distribution["probability_three_or_fewer"],
        "workload_scenarios": workload_scenarios,
        "milestone_probabilities": distribution["milestones"],
        "confidence": confidence,
        "arsenal_coverage": pitch_coverage,
        "effective_sample_size": pitch_evidence["effective_sample_size"],
        "hitters_covered": pitch_evidence["hitters_covered"],
        "lineup_k_evidence": lineup_evidence,
        "pitch_mix_evidence": pitch_evidence,
        "pitch_process_evidence": process_evidence,
        "hitter_by_hitter_k": hitter_projection,
        "data_grade": data_grade,
        "opportunity": opportunity,
        "market_assessment": market_assessment,
        "spotlight_qualified": spotlight_qualified,
        "downside_risks": downside_risks,
        "workload_read": leash,
        "performance_outlook": performance_outlook,
        "workload_challenger": workload_challenger,
        "k_challenger": k_challenger,
        "workload_stages": workload_estimate["workload_stages"],
        "lineup_workload_context": workload_estimate["lineup_workload_context"],
        "components": {
            "baseline_k_rate": pitcher_rate,
            "lineup_adjustment": lineup_adjustment,
            "pitch_adjustment": pitch_adjustment,
            "process_adjustment": process_evidence["adjustment"],
            "raw_matchup_k_rate": hitter_projection["raw_rate"],
            "extreme_rate_shrinkage": extreme_shrinkage,
            "matchup_k_rate": k_rate,
            "expected_batters_faced": expected_bf,
            "expected_pitches": workload_estimate["expected_pitches"],
            "expected_outs": workload_estimate["expected_outs"],
            "early_exit_risk": workload_estimate["early_exit_risk"],
            "matchup_pitches_per_batter": workload_estimate["pitches_per_batter"],
            "matchup_baserunner_rate": workload_estimate["workload_stages"]["matchup_baserunner_rate"],
            "matchup_out_rate": workload_estimate["workload_stages"]["matchup_out_rate"],
            "matchup_hook_adjustment": workload_estimate["workload_stages"]["matchup_hook_adjustment"],
        },
        "lineup_confirmed": lineup_confirmed,
        "decision": decision,
        "factors": factors,
        "missing_inputs": missing,
        "market": market_read or None,
        "market_stale": stale_market,
        "data_freshness_seconds": data_freshness_seconds,
        "validated_edge": False,
    }


def hitter_pitch_summary(stat, league_average=LEAGUE_HITTER_AVERAGE, prior=None):
    """Return a context-aware, shrinkage-safe contact and power read."""
    stat = stat or {}
    prior = prior or {}
    prior_average = clamp(_number(prior.get("avg"), league_average), 0.0, 1.0)
    prior_slg = max(0.0, _number(prior.get("slg"), LEAGUE_HITTER_SLG))
    prior_iso = max(0.0, _number(prior.get("iso"), prior_slg - prior_average))
    pa = max(0.0, _number(stat.get("pa"), 0.0))
    context = stat.get("context") or {}
    average = clamp(_number(context.get("adjusted_avg"), _number(stat.get("avg"), prior_average)), 0.0, 1.0)
    advanced = stat.get("advanced") or {}
    power_value = _number(context.get("adjusted_slg"), _number(advanced.get("slg")))
    has_power = power_value is not None
    slugging = max(0.0, power_value if has_power else LEAGUE_HITTER_SLG)
    approximate_hits = average * pa
    posterior = shrunk_rate(approximate_hits, pa, prior_average, HITTER_PITCH_PRIOR_PA)
    reliability = pa / (pa + HITTER_PITCH_PRIOR_PA) if pa else 0.0
    slugging_posterior = (slugging * pa + prior_slg * HITTER_PITCH_PRIOR_PA) / (pa + HITTER_PITCH_PRIOR_PA) if pa else prior_slg
    iso_posterior = max(0.0, slugging_posterior - posterior) if has_power else prior_iso
    avg_delta = posterior - league_average
    iso_delta = iso_posterior - LEAGUE_HITTER_ISO
    delta = .70 * avg_delta + .30 * iso_delta
    context_coverage = clamp(_number(context.get("coverage"), 0.0), 0.0, 1.0)
    if pa < HITTER_PITCH_MIN_PA:
        label, tone = "low data", "neutral"
    elif reliability < HITTER_PITCH_MIN_RELIABILITY:
        label, tone = "prior-driven", "neutral"
    elif delta >= HITTER_CONTACT_DELTA:
        label, tone = "favorable", "good"
    elif delta <= -HITTER_CONTACT_DELTA:
        label, tone = "poor", "bad"
    else:
        label, tone = "neutral", "neutral"
    return {
        "label": label,
        "tone": tone,
        "shrunk_average": posterior,
        "shrunk_slg": slugging_posterior,
        "shrunk_iso": iso_posterior,
        "delta": delta,
        "avg_delta": avg_delta,
        "iso_delta": iso_delta,
        "reliability": reliability,
        "context_coverage": context_coverage,
        "pa": pa,
        "prior_average": prior_average,
        "prior_slg": prior_slg,
    }


def hitter_strong_evidence(metrics):
    """Return whether a hitter read clears the high-confidence evidence gate."""
    return (
        _number(metrics.get("coverage"), 0.0) >= HITTER_STRONG_MIN_COVERAGE
        and _number(metrics.get("effective_sample_size"), 0.0) >= HITTER_STRONG_MIN_EFFECTIVE_PA
        and _number(metrics.get("context_coverage"), 0.0) >= HITTER_STRONG_MIN_CONTEXT_COVERAGE
    )


def hitter_signal_risk(batter, metrics):
    """Describe when an attractive hitter point estimate is too unstable to trust.

    Matchup direction and evidence stability are intentionally separate. A high
    score can still be labeled high risk when it rests on sparse pitch,
    handedness, season, or exact-context evidence.
    """
    coverage = clamp(_number(metrics.get("coverage"), 0.0), 0.0, 1.0)
    sample = max(0.0, _number(metrics.get("effective_sample_size"), 0.0))
    context_coverage = clamp(_number(metrics.get("context_coverage"), 0.0), 0.0, 1.0)
    platoon = metrics.get("platoon") or batter.get("platoon") or {}
    season = batter.get("season") or {}
    platoon_ab = _number(platoon.get("at_bats"))
    season_pa = _number(season.get("pa"))
    reasons = []
    instability_flags = []

    def flag(key, reason):
        instability_flags.append(key)
        reasons.append(reason)

    if coverage < HITTER_RISK_MIN_COVERAGE:
        flag("arsenal_coverage", f"Only {coverage:.0%} arsenal coverage")
    if sample < HITTER_RISK_MIN_EFFECTIVE_PA:
        flag("effective_pa", f"Only {sample:.1f} effective PA")
    if context_coverage < HITTER_RISK_MIN_CONTEXT_COVERAGE:
        flag("context_coverage", f"Only {context_coverage:.0%} exact-context coverage")
    if platoon_ab is not None and platoon_ab < HITTER_RISK_MIN_PLATOON_AB:
        flag("platoon_sample", f"Only {platoon_ab:.0f} AB in the {platoon.get('label', 'handedness')} split")
    if season_pa is not None and season_pa < HITTER_RISK_MIN_SEASON_PA:
        flag("season_sample", f"Only {season_pa:.0f} current-season PA")
    if metrics.get("platoon_disagreement"):
        flag("platoon_disagreement", "Pitch-fit estimate conflicts with the stable platoon baseline")

    critical = (
        coverage < 0.10
        or sample < 5.0
        or (platoon_ab is not None and platoon_ab < 25)
        or (season_pa is not None and season_pa < 60)
        or bool(metrics.get("platoon_disagreement"))
    )
    if critical or len(instability_flags) >= 2:
        level, label = "high", "High risk"
    elif instability_flags:
        level, label = "elevated", "Elevated risk"
    else:
        level, label = "normal", "Standard risk"

    if not batter.get("lineup_order"):
        reasons.append("Starting lineup is not confirmed")
    bullpen_effect = metrics.get("bullpen_effect") or {}
    if bullpen_effect.get("key") in {"uncertain", "unknown"}:
        reasons.append(bullpen_effect.get("label") or "Bullpen uncertainty")

    return {
        "level": level,
        "label": label,
        "reasons": list(dict.fromkeys(reasons))[:6],
        "flags": instability_flags,
    }


def hitter_market_context(market, batting_team):
    """Turn the game market into a conservative team scoring expectation.

    A nine-run game does not imply 4.5 runs for both clubs. When a moneyline is
    available, split the total toward the favorite; this is an explanatory run
    proxy rather than a sportsbook team-total market. The resulting adjustment
    remains deliberately small relative to pitch and batted-ball evidence.
    """
    market = market or {}
    if isinstance(batting_team, dict):
        batting_team = batting_team.get("name") or batting_team.get("displayName")

    def same_team(left, right):
        normalize = lambda value: "".join(character for character in str(value or "").lower() if character.isalnum())
        return bool(normalize(left)) and normalize(left) == normalize(right)

    total = _number(market.get("total"))
    favorite = market.get("favorite") or {}
    favorite_team = favorite.get("team") if isinstance(favorite, dict) else None
    favorite_probability = _number(favorite.get("probability")) if isinstance(favorite, dict) else None
    favorite_probability = clamp(favorite_probability, .50, .80) if favorite_probability is not None else None
    is_favorite = same_team(favorite_team, batting_team) if favorite_team and batting_team else False
    team_role = "favorite" if is_favorite else ("underdog" if favorite_team and batting_team else "unknown")

    team_runs = None
    opponent_runs = None
    if total is not None:
        probability = favorite_probability if favorite_probability is not None else (.55 if favorite_team else .50)
        run_shift = clamp((probability - .50) * total * .50, 0.0, 1.25) if favorite_team else 0.0
        team_runs = total / 2.0 + run_shift if is_favorite else total / 2.0 - run_shift
        opponent_runs = total - team_runs
    total_adjustment = (
        clamp(((team_runs - LEAGUE_TEAM_RUNS) / 1.25) * .005, -.005, .005)
        if team_runs is not None else 0.0
    )
    return {
        "total": total,
        "favorite": favorite_team,
        "favorite_probability": favorite_probability,
        "batting_team": batting_team,
        "team_role": team_role,
        "is_favorite": is_favorite,
        "team_run_expectation": round(team_runs, 2) if team_runs is not None else None,
        "opponent_run_expectation": round(opponent_runs, 2) if opponent_runs is not None else None,
        "total_adjustment": total_adjustment,
        # Kept for old cached clients; team allocation now lives in total_adjustment.
        "favorite_adjustment": 0.0,
        "adjustment": total_adjustment,
        "available": total is not None or favorite_team is not None,
    }


def hitter_starter_quality_context(performance, starter_share=1.0):
    """Return a capped hitter adjustment from the starter's run prevention.

    ER/9 and WHIP are noisy, so neither is treated as a projection by itself.
    They are blended, capped, and scaled by the share of plate appearances the
    hitter is expected to take against the starter.
    """
    performance = performance or {}
    er9 = _number(performance.get("earned_runs_per_nine"))
    whip = _number(performance.get("whip_history"))
    components = []
    if er9 is not None:
        components.append((.60, clamp((LEAGUE_STARTER_ER9 - er9) / 1.50, -1.0, 1.0)))
    if whip is not None:
        components.append((.40, clamp((LEAGUE_STARTER_WHIP - whip) / .25, -1.0, 1.0)))
    total_weight = sum(weight for weight, _value in components)
    quality = sum(weight * value for weight, value in components) / total_weight if total_weight else 0.0
    starter_share = clamp(_number(starter_share, 1.0), 0.0, 1.0)
    # Positive quality means a tougher starter; hitter score adjustment is inverse.
    score_adjustment = -0.55 * quality * starter_share
    if quality >= .30:
        label, tone = "tough starter", "bad"
    elif quality <= -.30:
        label, tone = "vulnerable starter", "good"
    else:
        label, tone = "average starter quality", "neutral"
    return {
        "quality": round(quality, 3),
        "score_adjustment": round(score_adjustment, 3),
        "label": label,
        "tone": tone,
        "earned_runs_per_nine": er9,
        "whip": whip,
        "starter_share": round(starter_share, 3),
        "available": bool(components),
    }


FIELD_SECTORS = ("LF", "LCF", "CF", "RCF", "RF")
NEUTRAL_PARK_DISTANCES = {"LF": 330.0, "LCF": 375.0, "CF": 400.0, "RCF": 375.0, "RF": 330.0}


def park_weather_fit(spray_profile, park, weather):
    """Estimate outcome-specific park and weather modifiers.

    Handedness-specific multi-year Statcast park factors establish the venue
    baseline. Broad spray sectors and wall dimensions provide only a small
    player-specific residual so dimensions are not double-counted. Directional
    wind and temperature are then applied as game-day overlays. Historical
    hitter results are never rewritten.
    """
    spray_profile = spray_profile or {}
    park = park or {}
    weather = weather or {}
    sectors = spray_profile.get("sectors") or {}
    sector_total = sum(max(0.0, _number(sectors.get(sector), 0.0)) for sector in FIELD_SECTORS)
    if sector_total <= 0:
        sectors = {sector: .20 for sector in FIELD_SECTORS}
        sector_total = 1.0
    sectors = {sector: max(0.0, _number(sectors.get(sector), 0.0)) / sector_total for sector in FIELD_SECTORS}
    batted_balls = max(0.0, _number(spray_profile.get("batted_balls"), 0.0))

    empirical = venue_factor(park.get("venue_id"), spray_profile.get("bat_side"))
    empirical_total_bases_effect = (
        empirical["total_bases_multiplier"] - 1.0 if empirical else 0.0
    )
    empirical_home_run_effect = (
        empirical["home_run_multiplier"] - 1.0 if empirical else 0.0
    )

    distances = park.get("distances") or {}
    raw_distance_effect = 0.0
    dimension_count = 0
    for sector in FIELD_SECTORS:
        distance = _number(distances.get(sector))
        if distance is None:
            continue
        dimension_count += 1
        neutral = NEUTRAL_PARK_DISTANCES[sector]
        raw_distance_effect += sectors[sector] * clamp((neutral - distance) / neutral * 1.20, -.08, .08)
    # Statcast already captures the park's average geometry. This residual only
    # asks whether this hitter's directional spray fits that geometry unusually
    # well or poorly. Unknown venues retain a larger dimension-only fallback.
    dimension_effect = clamp(raw_distance_effect * (.25 if empirical else .65), -.025, .025)

    roof = str(park.get("roof") or "").lower()
    weather_limited = roof not in {"open", "unknown", "outdoor"} and bool(roof)
    wind_text = str(weather.get("wind") or "")
    wind_speed_match = re.search(r"(\d+(?:\.\d+)?)\s*mph", wind_text, re.IGNORECASE)
    wind_speed = clamp(_number(wind_speed_match.group(1), 0.0), 0.0, 25.0) if wind_speed_match else 0.0
    upper_wind = wind_text.upper().replace("RIGHT", "RF").replace("LEFT", "LF").replace("CENTER", "CF")
    wind_sign = 1.0 if "OUT TO" in upper_wind else -1.0 if "IN FROM" in upper_wind else 0.0
    target = next((sector for sector in ("RCF", "LCF", "RF", "LF", "CF") if sector in upper_wind), None)
    target_index = FIELD_SECTORS.index(target) if target else None
    directional_share = 0.0
    if target_index is not None:
        for index, sector in enumerate(FIELD_SECTORS):
            distance = abs(index - target_index)
            directional_share += sectors[sector] * (1.0 if distance == 0 else .45 if distance == 1 else 0.0)
    wind_effect = 0.0 if weather_limited else wind_sign * wind_speed * .004 * directional_share

    temperature = _number(weather.get("temp"))
    temperature_effect = (
        0.0 if weather_limited or temperature is None
        else clamp((temperature - 72.0) * .0015, -.03, .03)
    )
    total_bases_effect = clamp(
        empirical_total_bases_effect
        + .35 * dimension_effect
        + .55 * wind_effect
        + .50 * temperature_effect,
        -.20, .20,
    )
    home_run_effect = clamp(
        empirical_home_run_effect + dimension_effect + wind_effect + temperature_effect,
        -.20, .20,
    )
    total_bases_multiplier = 1.0 + total_bases_effect
    home_run_multiplier = 1.0 + home_run_effect

    if batted_balls >= 100 and dimension_count >= 4 and (weather_limited or wind_speed_match):
        confidence = "strong"
    elif batted_balls >= 35 and dimension_count >= 3:
        confidence = "usable"
    else:
        confidence = "limited"
    available = bool(empirical) or (batted_balls > 0 and dimension_count >= 3)
    if not available:
        total_bases_multiplier = home_run_multiplier = 1.0
        total_bases_effect = home_run_effect = 0.0
        confidence = "limited"

    def outcome_label(effect, outcome):
        if effect >= .045:
            return f"favorable {outcome} environment", "good"
        if effect <= -.045:
            return f"suppressive {outcome} environment", "bad"
        return f"neutral {outcome} environment", "neutral"

    total_bases_label, total_bases_tone = outcome_label(total_bases_effect, "total-base")
    home_run_label, home_run_tone = outcome_label(home_run_effect, "home-run")
    if total_bases_tone == home_run_tone:
        label = total_bases_label if total_bases_tone != "neutral" else "neutral power environment"
        tone = total_bases_tone
    else:
        label, tone = "mixed park effects by outcome", "neutral"
    factors = []
    if empirical:
        factors.append(
            f"Statcast {empirical['years']} · TB {empirical['total_bases_index']:.0f} · "
            f"HR {empirical['home_run_index']:.0f} ({empirical['bat_side']}HB)"
        )
    if dimension_count:
        factors.append(f"Directional wall residual {dimension_effect:+.1%}")
    if weather_limited:
        factors.append("Roof limits weather impact")
    elif wind_sign and target:
        factors.append(f"Wind {wind_effect:+.1%} toward {target}")
    elif wind_text:
        factors.append("No reliable outfield wind vector")
    if temperature is not None and not weather_limited:
        factors.append(f"Temperature carry {temperature_effect:+.1%}")
    return {
        "available": available,
        "label": label,
        "tone": tone,
        # Legacy aliases intentionally point at the total-base context. New
        # consumers should use the outcome-specific fields below.
        "multiplier": round(total_bases_multiplier, 3),
        "effect": round(total_bases_effect, 3),
        "total_bases_multiplier": round(total_bases_multiplier, 3),
        "total_bases_effect": round(total_bases_effect, 3),
        "total_bases_label": total_bases_label,
        "total_bases_tone": total_bases_tone,
        "home_run_multiplier": round(home_run_multiplier, 3),
        "home_run_effect": round(home_run_effect, 3),
        "home_run_label": home_run_label,
        "home_run_tone": home_run_tone,
        "statcast_park_factor": empirical,
        "statcast_total_bases_effect": round(empirical_total_bases_effect, 3),
        "statcast_home_run_effect": round(empirical_home_run_effect, 3),
        "park_effect": round(dimension_effect, 3),
        "raw_dimension_effect": round(raw_distance_effect, 3),
        "wind_effect": round(wind_effect, 3),
        "temperature_effect": round(temperature_effect, 3),
        "wind_target": target,
        "wind_speed": wind_speed,
        "confidence": confidence,
        "batted_balls": int(batted_balls),
        "bat_side": spray_profile.get("bat_side"),
        "pull_rate": _number(spray_profile.get("pull_rate")),
        "center_rate": _number(spray_profile.get("center_rate")),
        "opposite_rate": _number(spray_profile.get("opposite_rate")),
        "sectors": {sector: round(sectors[sector], 4) for sector in FIELD_SECTORS},
        "factors": factors,
        "method": "statcast venue + directional weather v2",
    }


def hitter_opportunity_reads(batter, metrics, market_context):
    """Describe distinct hitter outcomes without claiming prop probabilities.

    One broad contact score can hide a useful power-only matchup. These reads
    deliberately keep hit, total-base, home-run, and run-production evidence
    separate. The scores rank research cards only; the displayed values and
    drivers remain interpretable descriptive statistics.
    """
    coverage = metrics["coverage"]
    sample = metrics["effective_sample_size"]
    context_coverage = metrics["context_coverage"]
    quality_coverage = metrics["quality_coverage"]
    expected_average = metrics["expected_average"]
    expected_slg = metrics["expected_slg"]
    expected_iso = metrics["expected_iso"]
    hr_rate = metrics["hr_rate"]
    hard_hit_rate = metrics["hard_hit_rate"]
    barrel_rate = metrics["barrel_rate"]
    k_rate = _number(
        metrics.get("k_rate"),
        hitter_k_risk(batter.get("k_profile"))["posterior"],
    )
    lineup_order = batter.get("lineup_order")
    lineup_number = _number(lineup_order)
    projected_pa = _lineup_weight(lineup_order)
    total = _number(market_context.get("total"))
    team_runs = _number(market_context.get("team_run_expectation"))
    favorite = bool(market_context.get("is_favorite"))
    starter_quality = metrics.get("starter_quality") or {}
    starter_adjustment = _number(starter_quality.get("score_adjustment"), 0.0)
    environment = metrics.get("environment") or {}
    total_bases_environment_effect = (
        _number(environment.get("total_bases_effect"), _number(environment.get("effect"), 0.0))
        if environment.get("available") else 0.0
    )
    home_run_environment_effect = (
        _number(environment.get("home_run_effect"), _number(environment.get("effect"), 0.0))
        if environment.get("available") else 0.0
    )
    power_environment_adjustment = clamp(total_bases_environment_effect / .10 * .20, -.30, .30)
    home_run_environment_adjustment = clamp(home_run_environment_effect / .10 * .45, -.60, .60)
    platoon = metrics.get("platoon") or {}
    platoon_ab = max(0.0, _number(platoon.get("at_bats"), 0.0))
    platoon_reliability = platoon_ab / (platoon_ab + 120.0) if platoon_ab else 0.0
    raw_platoon_avg = _number(platoon.get("raw_avg"))
    raw_platoon_slg = _number(platoon.get("raw_slg"))
    raw_platoon_iso = (
        max(0.0, raw_platoon_slg - raw_platoon_avg)
        if raw_platoon_avg is not None and raw_platoon_slg is not None else None
    )
    # The pitch model already uses a shrunk handedness prior. These smaller
    # terms make a stable left/right split explicit without double-counting it.
    platoon_contact = (
        .14 * platoon_reliability * clamp((raw_platoon_avg - LEAGUE_HITTER_AVERAGE) / .040, -1.0, 1.0)
        if raw_platoon_avg is not None else 0.0
    )
    platoon_power = (
        .18 * platoon_reliability * clamp((raw_platoon_slg - LEAGUE_HITTER_SLG) / .100, -1.0, 1.0)
        if raw_platoon_slg is not None else 0.0
    )
    platoon_home_run = (
        .15 * platoon_reliability * clamp((raw_platoon_iso - LEAGUE_HITTER_ISO) / .080, -1.0, 1.0)
        if raw_platoon_iso is not None else 0.0
    )

    contact_core = (
        0.72 * (expected_average - LEAGUE_HITTER_AVERAGE) / 0.025
        + 0.28 * (LEAGUE_K_RATE - k_rate) / 0.040
    )
    power_core = (
        0.52 * (expected_slg - LEAGUE_HITTER_SLG) / 0.080
        + 0.30 * (expected_iso - LEAGUE_HITTER_ISO) / 0.060
        + 0.18 * (hard_hit_rate - LEAGUE_HARD_HIT_RATE) / 0.060
    )
    home_run_core = (
        0.42 * (hr_rate - LEAGUE_HITTER_HR_RATE) / 0.015
        + 0.33 * (expected_iso - LEAGUE_HITTER_ISO) / 0.060
        + 0.25 * (barrel_rate - LEAGUE_BARREL_PROXY_RATE) / 0.030
    )
    contact_context = starter_adjustment + platoon_contact
    power_context = starter_adjustment + platoon_power + power_environment_adjustment
    home_run_context = starter_adjustment + platoon_home_run + home_run_environment_adjustment
    contact_score = contact_core + contact_context
    power_score = power_core + power_context
    home_run_score = home_run_core + home_run_context
    lineup_score = clamp((projected_pa - 4.2) / 0.45, -1.0, 1.0)
    environment_score = clamp(
        ((team_runs - LEAGUE_TEAM_RUNS) / .75) if team_runs is not None else 0.0,
        -1.0, 1.0,
    )
    run_core = 0.30 * contact_core + 0.25 * power_core
    run_context = (
        0.30 * contact_context + 0.25 * power_context
        + 0.30 * lineup_score + 0.15 * environment_score
    )
    run_score = run_core + run_context

    base_risks = []
    if coverage < 0.45:
        base_risks.append(f"Only {coverage:.0%} of arsenal evidence is reliable")
    if sample < 20:
        base_risks.append(f"Small {sample:.0f} effective-PA sample")
    if not lineup_order:
        base_risks.append("Starting lineup is not confirmed")
    elif lineup_number is not None and lineup_number >= 7:
        base_risks.append("Lower-order plate-appearance risk")
    if k_rate >= 0.26:
        base_risks.append(f"Elevated {k_rate:.1%} descriptive K rate")
    if metrics.get("platoon_disagreement"):
        base_risks.append(
            f"Pitch-fit estimate disagrees with the {_number(platoon.get('at_bats'), 0):.0f}-AB "
            f"{platoon.get('label', 'platoon')} baseline"
        )
    if starter_quality.get("available") and _number(starter_quality.get("quality"), 0.0) >= .30:
        er9 = _number(starter_quality.get("earned_runs_per_nine"))
        whip = _number(starter_quality.get("whip"))
        detail = " · ".join(value for value in (
            f"{er9:.2f} ER/9" if er9 is not None else None,
            f"{whip:.2f} WHIP" if whip is not None else None,
        ) if value)
        base_risks.append(f"Tough starter run prevention{': ' + detail if detail else ''}")

    signal_risk = hitter_signal_risk(batter, metrics)
    reliability = hitter_ranking_reliability(batter, metrics)
    recent_form = batter.get("recent_form") or metrics.get("recent_form") or {}

    def build(key, title, base_score, value, drivers, extra_risks=(),
              context_adjustment=0.0, recent_adjustment=None):
        confidence_read = hitter_outcome_confidence(key, batter, metrics, signal_risk)
        confidence = confidence_read["confidence"]
        recent_adjustment = (
            hitter_recent_form_adjustment(key, recent_form)
            if recent_adjustment is None else recent_adjustment
        )
        adjusted_score = base_score + recent_adjustment
        if base_score >= 0.75 and confidence == "high" and not metrics.get("platoon_disagreement"):
            direction, tone = "strong", "good"
        elif base_score >= 0.25:
            direction, tone = "favorable", "good"
        elif base_score <= -0.40:
            direction, tone = "tough", "bad"
        else:
            direction, tone = "neutral", "neutral"
        # A marginal-data hitter signal should not be called strong against a
        # demonstrably strong run suppressor. It remains visible as favorable.
        if (
            direction == "strong"
            and _number(starter_quality.get("quality"), 0.0) >= .30
            and coverage < .50
        ):
            direction = "favorable"
        qualified = (
            confidence_read["qualified"]
            and direction in {"strong", "favorable"}
        )
        promising = (
            not qualified
            and direction in {"strong", "favorable"}
            and bool(lineup_order)
            and signal_risk.get("level") != "high"
            and coverage >= HITTER_PROMISING_MIN_COVERAGE
            and sample >= HITTER_PROMISING_MIN_EFFECTIVE_PA
        )
        why_not_stronger = confidence_read["blockers"][0] if confidence_read["blockers"] else (
            "matchup score is below the strong threshold" if direction != "strong" else "clears the strong evidence bar"
        )
        return {
            "key": key,
            "title": title,
            "score": round(adjusted_score, 3),
            "base_opportunity_score": round(base_score, 3),
            "raw_matchup_score": round(base_score - context_adjustment, 3),
            "recent_form_adjustment": round(recent_adjustment, 3),
            "context_adjustment": round(context_adjustment, 3),
            "reliability": round(reliability, 3),
            "ranking_score": round(adjusted_score * reliability, 3),
            "direction": direction,
            "confidence": confidence,
            # Compatibility aliases for older clients. New consumers should use
            # direction/confidence and must not infer confidence from tier.
            "tier": direction,
            "tone": tone,
            "value": value,
            "drivers": drivers[:3],
            "risks": list(dict.fromkeys([*base_risks, *extra_risks]))[:3],
            "evidence": {"high": "strong", "medium": "usable", "limited": "limited"}[confidence],
            "qualified": qualified,
            "promising": promising,
            "why_not_stronger": why_not_stronger,
            "requirements": confidence_read["requirements"],
            "coverage": round(coverage, 4),
            "effective_pa": round(sample, 2),
            "context_coverage": round(context_coverage, 4),
            "quality_coverage": round(quality_coverage, 4),
        }

    quality_risks = [] if quality_coverage >= 0.20 else ["Limited hard-hit and barrel evidence"]
    environment_risks = (
        ["Limited directional park/weather evidence"]
        if environment.get("confidence") == "limited" else []
    )
    platoon_driver = (
        f"{platoon.get('label', 'Platoon')} {raw_platoon_avg:.3f} AVG / {raw_platoon_slg:.3f} SLG ({platoon_ab:.0f} AB)"
        if raw_platoon_avg is not None and raw_platoon_slg is not None
        else f"{platoon.get('label', 'Platoon')} data pending"
    )
    starter_driver = (
        f"Starter quality {starter_quality.get('label')} ({starter_adjustment:+.2f})"
        if starter_quality.get("available") else "Starter quality pending"
    )
    total_bases_environment_driver = (
        f"Park/weather TB {environment.get('total_bases_multiplier', environment.get('multiplier', 1.0)):.2f}× · "
        f"{environment.get('total_bases_label', environment.get('label', 'neutral'))}"
        if environment.get("available") else "Park/weather directional profile pending"
    )
    home_run_environment_driver = (
        f"Park/weather HR {environment.get('home_run_multiplier', environment.get('multiplier', 1.0)):.2f}× · "
        f"{environment.get('home_run_label', environment.get('label', 'neutral'))}"
        if environment.get("available") else "Park/weather directional profile pending"
    )
    environment_driver = (
        f"Park/weather TB {environment.get('total_bases_multiplier', environment.get('multiplier', 1.0)):.2f}× · "
        f"HR {environment.get('home_run_multiplier', environment.get('multiplier', 1.0)):.2f}×"
        if environment.get("available") else "Park/weather directional profile pending"
    )
    opportunities = {
        "hit": build(
            "hit", "Hit opportunity", contact_score,
            f"{expected_average:.3f} matchup AVG",
            [f"Matchup AVG {expected_average:.3f} vs {LEAGUE_HITTER_AVERAGE:.3f} MLB",
             platoon_driver, starter_driver],
            context_adjustment=contact_context,
        ),
        "total_bases": build(
            "total_bases", "Total-base power", power_score,
            f"{expected_slg:.3f} matchup SLG",
            [f"Matchup SLG {expected_slg:.3f} vs {LEAGUE_HITTER_SLG:.3f} MLB",
             f"Matchup ISO {expected_iso:.3f}", total_bases_environment_driver],
            [*quality_risks, *environment_risks],
            context_adjustment=power_context,
        ),
        "home_run": build(
            "home_run", "Home-run power", home_run_score,
            f"{expected_iso:.3f} matchup ISO",
            [f"Matchup ISO {expected_iso:.3f} vs {LEAGUE_HITTER_ISO:.3f} MLB",
             f"Barrel proxy {barrel_rate:.1%}", home_run_environment_driver],
            [*quality_risks, *environment_risks],
            context_adjustment=home_run_context,
        ),
        "runs_rbi": build(
            "runs_rbi", "Runs + RBI opportunity", run_score,
            f"{projected_pa:.1f} expected PA",
            [f"Batting {lineup_order or 'order pending'} · about {projected_pa:.1f} PA",
             (f"Team run expectation {team_runs:.2f} of {total:.1f}" if team_runs is not None and total is not None
              else "Team run expectation unavailable"),
             ("Batting team is favored" if favorite else
              ("Batting team is the underdog" if market_context.get("team_role") == "underdog" else f"Matchup SLG {expected_slg:.3f}"))],
            ["Runs/RBIs also depend on surrounding hitters and bullpen"] if lineup_order else [],
            context_adjustment=run_context,
        ),
    }
    primary_key = max(opportunities, key=lambda key: opportunities[key]["ranking_score"])
    primary = opportunities[primary_key]
    strongest_paths = sorted(
        opportunities.values(), key=lambda item: item["base_opportunity_score"], reverse=True,
    )[:2]
    overall_base = 0.65 * strongest_paths[0]["base_opportunity_score"] + 0.35 * strongest_paths[1]["base_opportunity_score"]
    overall_context = 0.65 * strongest_paths[0]["context_adjustment"] + 0.35 * strongest_paths[1]["context_adjustment"]
    overall_recent = clamp(
        0.65 * strongest_paths[0]["recent_form_adjustment"]
        + 0.35 * strongest_paths[1]["recent_form_adjustment"],
        -HITTER_RECENT_FORM_MAX_ADJUSTMENT, HITTER_RECENT_FORM_MAX_ADJUSTMENT,
    )
    opportunities["overall"] = build(
        "overall", "Overall offensive opportunity", overall_base,
        f"{primary['title']}",
        [f"Best path: {primary['title']}", primary["drivers"][0],
         environment_driver if environment.get("available") else primary["drivers"][1]],
        primary["risks"],
        context_adjustment=overall_context,
        recent_adjustment=overall_recent,
    )
    confidence_order = {"limited": 0, "medium": 1, "high": 2}
    underlying_confidence = min(
        (item["confidence"] for item in strongest_paths),
        key=lambda value: confidence_order[value],
    )
    overall = opportunities["overall"]
    if confidence_order[overall["confidence"]] > confidence_order[underlying_confidence]:
        overall["confidence"] = underlying_confidence
        overall["evidence"] = {"high": "strong", "medium": "usable", "limited": "limited"}[underlying_confidence]
        overall["qualified"] = (
            underlying_confidence in {"high", "medium"}
            and bool(lineup_order) and signal_risk.get("level") != "high"
            and overall["direction"] in {"strong", "favorable"}
        )
        overall["promising"] = (
            not overall["qualified"] and bool(lineup_order)
            and signal_risk.get("level") != "high"
            and overall["direction"] in {"strong", "favorable"}
            and coverage >= HITTER_PROMISING_MIN_COVERAGE
            and sample >= HITTER_PROMISING_MIN_EFFECTIVE_PA
        )
        if not overall["qualified"]:
            overall["why_not_stronger"] = "underlying outcome evidence is limited"
    recent_form_payload = dict(recent_form)
    recent_form_payload["adjustment"] = round(overall_recent, 3)
    return {
        "primary": primary_key,
        "projected_pa": projected_pa,
        "items": opportunities,
        "risk": signal_risk,
        "recent_form": recent_form_payload,
    }


def hitter_arsenal_summary(batter, pitches, league_average=LEAGUE_HITTER_AVERAGE, market_context=None):
    """Shrink descriptive contact results into transparent strong/favorable tiers.

    These are research tiers, not hit-prop probabilities or wagering signals.
    Strong retains the original +25-point standard. Favorable gives confirmed
    lineups a useful middle tier once coverage is broad enough.
    """
    total_usage = sum(max(0.0, _number(p.get("usage"), 0.0)) for p in pitches)
    market_context = market_context or hitter_market_context(None, None)
    platoon = batter.get("platoon") or {}
    platoon_prior = {
        "avg": _number(platoon.get("posterior_avg"), league_average),
        "slg": _number(platoon.get("posterior_slg"), LEAGUE_HITTER_SLG),
        "iso": _number(platoon.get("posterior_iso"), LEAGUE_HITTER_ISO),
    }
    if total_usage <= 0:
        empty = {"label": "insufficient", "tier": "watchlist", "tone": "neutral", "coverage": 0.0, "effective_sample_size": 0.0, "expected_average": platoon_prior["avg"], "expected_slg": platoon_prior["slg"], "expected_iso": platoon_prior["iso"], "delta": platoon_prior["avg"] - league_average, "base_score": 0.0, "score": market_context["adjustment"], "market_context": market_context, "context_coverage": 0.0, "quality_coverage": 0.0, "hard_hit_rate": LEAGUE_HARD_HIT_RATE, "barrel_rate": LEAGUE_BARREL_PROXY_RATE, "hr_rate": LEAGUE_HITTER_HR_RATE, "platoon": platoon, "environment": batter.get("environment") or {}, "recent_form": batter.get("recent_form") or {}}
        empty["opportunities"] = hitter_opportunity_reads(batter, empty, market_context)
        empty["risk"] = empty["opportunities"]["risk"]
        return empty
    expected = expected_slg = expected_iso = coverage = context_coverage = sample = score = 0.0
    hard_hit_rate = barrel_rate = hr_rate = quality_coverage = 0.0
    for pitch in pitches:
        usage = max(0.0, _number(pitch.get("usage"), 0.0)) / total_usage
        stat = (batter.get("vs_pitches") or {}).get(pitch.get("code"))
        pitch_read = hitter_pitch_summary(stat, league_average, platoon_prior)
        pa = pitch_read["pa"]
        posterior = pitch_read["shrunk_average"]
        slg_posterior = pitch_read["shrunk_slg"]
        iso_posterior = pitch_read["shrunk_iso"]
        reliability = pitch_read["reliability"]
        quality = (stat or {}).get("quality") or {}
        batted_balls = max(0.0, _number(quality.get("batted_balls"), 0.0))
        quality_reliability = batted_balls / (batted_balls + 40.0) if batted_balls else 0.0
        pitch_hard_hit = shrunk_rate(quality.get("hard_hits"), batted_balls, LEAGUE_HARD_HIT_RATE, 40.0)
        pitch_barrel = shrunk_rate(quality.get("barrel_proxy"), batted_balls, LEAGUE_BARREL_PROXY_RATE, 50.0)
        pitch_hr_rate = shrunk_rate((stat or {}).get("hr"), pa, LEAGUE_HITTER_HR_RATE, 100.0)
        expected += usage * posterior
        expected_slg += usage * slg_posterior
        expected_iso += usage * iso_posterior
        hard_hit_rate += usage * pitch_hard_hit
        barrel_rate += usage * pitch_barrel
        hr_rate += usage * pitch_hr_rate
        quality_coverage += usage * quality_reliability
        coverage += usage * reliability
        context_coverage += usage * pitch_read["context_coverage"]
        sample += min(pa, 60.0) * usage
        score += usage * pitch_read["delta"]
    delta = expected - league_average
    base_score = score
    score = base_score + market_context["adjustment"]
    strong_evidence = hitter_strong_evidence({
        "coverage": coverage,
        "effective_sample_size": sample,
        "context_coverage": context_coverage,
    })
    platoon_ab = max(0.0, _number(platoon.get("at_bats"), 0.0))
    platoon_disagreement = (
        platoon_ab >= 100
        and abs(expected - _number(platoon.get("raw_avg"), expected)) >= 0.040
    )
    if coverage < HITTER_ARSENAL_MIN_COVERAGE or sample < HITTER_ARSENAL_MIN_EFFECTIVE_PA:
        label, tier, tone = "insufficient", "watchlist", "neutral"
    elif score >= HITTER_CONTACT_DELTA and strong_evidence and not platoon_disagreement:
        label, tier, tone = "strong contact research", "strong", "good"
    elif score >= HITTER_CONTACT_FAVORABLE_DELTA:
        label, tier, tone = "favorable contact research", "favorable", "good"
    elif score <= -HITTER_CONTACT_DELTA:
        label, tier, tone = "tough contact research", "tough", "bad"
    else:
        label, tier, tone = "neutral contact research", "neutral", "neutral"
    result = {
        "label": label,
        "tier": tier,
        "tone": tone,
        "coverage": clamp(coverage, 0.0, 1.0),
        "effective_sample_size": sample,
        "expected_average": expected,
        "expected_slg": expected_slg,
        "expected_iso": expected_iso,
        "delta": delta,
        "base_score": base_score,
        "score": score,
        "market_context": market_context,
        "context_coverage": clamp(context_coverage, 0.0, 1.0),
        "quality_coverage": clamp(quality_coverage, 0.0, 1.0),
        "hard_hit_rate": hard_hit_rate,
        "barrel_rate": barrel_rate,
        "hr_rate": hr_rate,
        "platoon": platoon,
        "platoon_disagreement": platoon_disagreement,
        "strong_evidence": strong_evidence,
        "environment": batter.get("environment") or {},
        "recent_form": batter.get("recent_form") or {},
    }
    result["opportunities"] = hitter_opportunity_reads(batter, result, market_context)
    result["risk"] = result["opportunities"]["risk"]
    return result


def blend_full_game_hitter_matchup(batter, starter_summary, bullpen_entries,
                                   expected_batters_faced,
                                   batters_faced_interval=None,
                                   market_context=None,
                                   starter_performance=None):
    """Blend starter and appearance-weighted reliever evidence by expected PA.

    ``bullpen_entries`` contains ``weight`` plus an optional reliever ``summary``.
    Missing reliever arsenals keep their probability mass and contribute a
    league-average prior, preventing a thin modeled subset from being
    renormalized into false certainty.
    """
    starter = dict(starter_summary or {})
    market_context = market_context or starter.get("market_context") or hitter_market_context(None, None)
    projected_pa = _lineup_weight(batter.get("lineup_order"))
    starter_pa = min(
        projected_pa,
        expected_starter_plate_appearances(
            batter.get("lineup_order"), expected_batters_faced, batters_faced_interval,
        ),
    )
    bullpen_pa = max(0.0, projected_pa - starter_pa)
    starter_share = starter_pa / projected_pa if projected_pa else 1.0
    bullpen_share = 1.0 - starter_share

    entries = list(bullpen_entries or [])
    raw_total = sum(max(0.0, _number(entry.get("weight"), 0.0)) for entry in entries)
    metric_defaults = {
        "expected_average": LEAGUE_HITTER_AVERAGE,
        "expected_slg": LEAGUE_HITTER_SLG,
        "expected_iso": LEAGUE_HITTER_ISO,
        "hard_hit_rate": LEAGUE_HARD_HIT_RATE,
        "barrel_rate": LEAGUE_BARREL_PROXY_RATE,
        "hr_rate": LEAGUE_HITTER_HR_RATE,
        "base_score": 0.0,
        "coverage": 0.0,
        "context_coverage": 0.0,
        "quality_coverage": 0.0,
        "effective_sample_size": 0.0,
    }
    bullpen = dict(metric_defaults)
    starter_k_rate = hitter_k_risk(batter.get("k_profile"))["posterior"]
    bullpen_k_rate = LEAGUE_K_RATE
    starter_environment = starter.get("environment") or {}
    bullpen_total_bases_environment_effect = 0.0
    bullpen_home_run_environment_effect = 0.0
    bullpen_environment_sample = 0.0
    modeled_weight = 0.0
    mix = []
    if raw_total > 0:
        for entry in entries:
            normalized = max(0.0, _number(entry.get("weight"), 0.0)) / raw_total
            summary = entry.get("summary") or {}
            if summary:
                modeled_weight += normalized
            reliever_k = hitter_k_risk(entry.get("k_profile"))["posterior"] if entry.get("k_profile") else LEAGUE_K_RATE
            bullpen_k_rate += normalized * (reliever_k - LEAGUE_K_RATE)
            reliever_environment = summary.get("environment") or {}
            if reliever_environment.get("available"):
                bullpen_total_bases_environment_effect += normalized * _number(
                    reliever_environment.get("total_bases_effect"),
                    _number(reliever_environment.get("effect"), 0.0),
                )
                bullpen_home_run_environment_effect += normalized * _number(
                    reliever_environment.get("home_run_effect"),
                    _number(reliever_environment.get("effect"), 0.0),
                )
                bullpen_environment_sample += normalized * _number(reliever_environment.get("batted_balls"), 0.0)
            for metric, default in metric_defaults.items():
                value = _number(summary.get(metric), default)
                bullpen[metric] += normalized * (value - default)
            mix.append({
                "player_id": entry.get("player_id"),
                "name": entry.get("name"),
                "weight": round(normalized, 4),
                "status": entry.get("status"),
                "role": entry.get("role"),
                "modeled": bool(summary),
            })

    if raw_total <= 0:
        # No readiness snapshot yet: show the starter-only result without
        # pretending an unknown bullpen is league average evidence.
        bullpen_share = 0.0
        starter_share = 1.0
        starter_pa = projected_pa
        bullpen_pa = 0.0

    result = {}
    for metric, default in metric_defaults.items():
        starter_value = _number(starter.get(metric), default)
        result[metric] = starter_share * starter_value + bullpen_share * bullpen[metric]
    result["coverage"] = clamp(result["coverage"], 0.0, 1.0)
    result["context_coverage"] = clamp(result["context_coverage"], 0.0, 1.0)
    result["quality_coverage"] = clamp(result["quality_coverage"], 0.0, 1.0)
    result["delta"] = result["expected_average"] - LEAGUE_HITTER_AVERAGE
    result["k_rate"] = starter_share * starter_k_rate + bullpen_share * bullpen_k_rate
    result["score"] = result["base_score"] + _number(market_context.get("adjustment"), 0.0)
    result["market_context"] = market_context
    result["platoon"] = starter.get("platoon") or batter.get("platoon") or {}
    result["platoon_disagreement"] = bool(starter.get("platoon_disagreement"))
    starter_total_bases_environment_effect = (
        _number(starter_environment.get("total_bases_effect"), _number(starter_environment.get("effect"), 0.0))
        if starter_environment.get("available") else 0.0
    )
    starter_home_run_environment_effect = (
        _number(starter_environment.get("home_run_effect"), _number(starter_environment.get("effect"), 0.0))
        if starter_environment.get("available") else 0.0
    )
    blended_total_bases_environment_effect = (
        starter_share * starter_total_bases_environment_effect
        + bullpen_share * bullpen_total_bases_environment_effect
    )
    blended_home_run_environment_effect = (
        starter_share * starter_home_run_environment_effect
        + bullpen_share * bullpen_home_run_environment_effect
    )

    def blended_environment_label(effect, outcome):
        if effect >= .045:
            return f"favorable {outcome} environment", "good"
        if effect <= -.045:
            return f"suppressive {outcome} environment", "bad"
        return f"neutral {outcome} environment", "neutral"

    total_bases_environment_label, total_bases_environment_tone = blended_environment_label(
        blended_total_bases_environment_effect, "total-base",
    )
    home_run_environment_label, home_run_environment_tone = blended_environment_label(
        blended_home_run_environment_effect, "home-run",
    )
    if total_bases_environment_tone == home_run_environment_tone:
        environment_label = (
            total_bases_environment_label
            if total_bases_environment_tone != "neutral" else "neutral power environment"
        )
        environment_tone = total_bases_environment_tone
    else:
        environment_label, environment_tone = "mixed park effects by outcome", "neutral"
    result["environment"] = {
        **starter_environment,
        "available": bool(starter_environment.get("available")) or bullpen_environment_sample > 0,
        "effect": round(blended_total_bases_environment_effect, 3),
        "multiplier": round(1.0 + blended_total_bases_environment_effect, 3),
        "total_bases_effect": round(blended_total_bases_environment_effect, 3),
        "total_bases_multiplier": round(1.0 + blended_total_bases_environment_effect, 3),
        "total_bases_label": total_bases_environment_label,
        "total_bases_tone": total_bases_environment_tone,
        "home_run_effect": round(blended_home_run_environment_effect, 3),
        "home_run_multiplier": round(1.0 + blended_home_run_environment_effect, 3),
        "home_run_label": home_run_environment_label,
        "home_run_tone": home_run_environment_tone,
        "label": environment_label,
        "tone": environment_tone,
        "batted_balls": int(_number(starter_environment.get("batted_balls"), 0.0) + bullpen_environment_sample),
        "method": "starter/bullpen Statcast park-weather blend v2",
    }
    result["starter_quality"] = hitter_starter_quality_context(starter_performance, starter_share)
    result["recent_form"] = batter.get("recent_form") or starter.get("recent_form") or {}
    # Make bullpen evidence available while the opportunity reliability score is
    # built; the public payload is completed with its readable mix below.
    result["bullpen"] = {
        **bullpen,
        "k_rate": bullpen_k_rate,
        "modeled_weight": round(modeled_weight, 4),
        "mix": sorted(mix, key=lambda item: item["weight"], reverse=True),
    }
    result["strong_evidence"] = hitter_strong_evidence(result)
    if result["coverage"] < HITTER_ARSENAL_MIN_COVERAGE or result["effective_sample_size"] < HITTER_ARSENAL_MIN_EFFECTIVE_PA:
        result.update({"label": "insufficient", "tier": "watchlist", "tone": "neutral"})
    elif result["score"] >= HITTER_CONTACT_DELTA and result["strong_evidence"] and not result["platoon_disagreement"]:
        result.update({"label": "strong full-game research", "tier": "strong", "tone": "good"})
    elif result["score"] >= HITTER_CONTACT_FAVORABLE_DELTA:
        result.update({"label": "favorable full-game research", "tier": "favorable", "tone": "good"})
    elif result["score"] <= -HITTER_CONTACT_DELTA:
        result.update({"label": "tough full-game research", "tier": "tough", "tone": "bad"})
    else:
        result.update({"label": "neutral full-game research", "tier": "neutral", "tone": "neutral"})

    opportunities = hitter_opportunity_reads(batter, result, market_context)
    starter_overall = ((starter.get("opportunities") or {}).get("items") or {}).get("overall", {})
    full_overall = opportunities["items"]["overall"]
    opportunity_delta = _number(full_overall.get("score"), 0.0) - _number(starter_overall.get("score"), 0.0)
    bullpen_coverage = clamp(_number(bullpen.get("coverage"), 0.0), 0.0, 1.0)
    if not raw_total:
        effect, effect_label = "unknown", "Bullpen not modeled"
    elif modeled_weight < 0.55 or bullpen_coverage < 0.20:
        effect, effect_label = "uncertain", "Bullpen uncertainty"
    elif opportunity_delta >= 0.18:
        effect, effect_label = "boost", "Bullpen boost"
    elif opportunity_delta <= -0.18:
        effect, effect_label = "downgrade", "Bullpen downgrade"
    elif opportunity_delta <= -0.08:
        effect, effect_label = "slight_downgrade", "Bullpen slightly lowers outlook"
    elif opportunity_delta >= -0.03 and full_overall.get("tier") in {"strong", "favorable"} and starter_overall.get("tier") in {"strong", "favorable"}:
        effect, effect_label = "supported", "Bullpen supported"
    else:
        effect, effect_label = "neutral", "Bullpen neutral"
    for opportunity in opportunities["items"].values():
        drivers = list(opportunity.get("drivers") or [])
        risks = list(opportunity.get("risks") or [])
        if effect in {"boost", "supported", "neutral"}:
            direction = "+" if opportunity_delta >= 0 else ""
            drivers.insert(1, f"{effect_label} ({direction}{opportunity_delta:.2f})")
        elif effect in {"downgrade", "slight_downgrade"}:
            risks.insert(0, f"{effect_label} ({opportunity_delta:.2f})")
        elif effect in {"uncertain", "unknown"}:
            risks.insert(0, effect_label)
        opportunity["drivers"] = list(dict.fromkeys(drivers))[:3]
        opportunity["risks"] = list(dict.fromkeys(risks))[:3]
    result["opportunities"] = opportunities
    overall_tier = opportunities["items"]["overall"]["direction"]
    result["direction"] = overall_tier
    result["confidence"] = opportunities["items"]["overall"]["confidence"]
    result["tier"] = overall_tier
    result["tone"] = opportunities["items"]["overall"]["tone"]
    result["label"] = {
        "strong": "strong full-game research",
        "favorable": "favorable full-game research",
        "tough": "tough full-game research",
        "neutral": "neutral full-game research",
    }.get(overall_tier, "neutral full-game research")
    result["exposure"] = {
        "projected_pa": round(projected_pa, 2),
        "starter_pa": round(starter_pa, 2),
        "bullpen_pa": round(bullpen_pa, 2),
        "starter_share": round(starter_share, 4),
        "bullpen_share": round(bullpen_share, 4),
    }
    result["starter"] = {
        "expected_average": _number(starter.get("expected_average"), LEAGUE_HITTER_AVERAGE),
        "expected_slg": _number(starter.get("expected_slg"), LEAGUE_HITTER_SLG),
        "expected_iso": _number(starter.get("expected_iso"), LEAGUE_HITTER_ISO),
        "score": _number(starter.get("score"), 0.0),
        "tier": starter.get("tier", "watchlist"),
        "tone": starter.get("tone", "neutral"),
    }
    result["bullpen"] = {
        **bullpen,
        "k_rate": bullpen_k_rate,
        "modeled_weight": round(modeled_weight, 4),
        "mix": sorted(mix, key=lambda item: item["weight"], reverse=True),
    }
    result["bullpen_effect"] = {
        "key": effect,
        "label": effect_label,
        "opportunity_delta": round(opportunity_delta, 3),
    }
    result["risk"] = hitter_signal_risk(batter, result)
    result["opportunities"]["risk"] = result["risk"]
    return result
