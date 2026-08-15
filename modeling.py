"""Versioned, server-side models and market math for Diamond Intel.

The initial model is intentionally a conservative empirical-Bayes baseline. It
is a research model until walk-forward results establish calibration and value.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import erf, exp, floor, isfinite, lgamma, log, sqrt
from statistics import median

MODEL_VERSION = "pitcher-k-workload-v4"
FEATURE_VERSION = "gameday-features-v4"
LEAGUE_K_RATE = 0.225
LEAGUE_BF_PER_START = 22.0
LEAGUE_HITTER_AVERAGE = 0.245
LEAGUE_HITTER_SLG = 0.400
LEAGUE_HITTER_ISO = LEAGUE_HITTER_SLG - LEAGUE_HITTER_AVERAGE
LEAGUE_HITTER_HR_RATE = 0.030
LEAGUE_HARD_HIT_RATE = 0.390
LEAGUE_BARREL_PROXY_RATE = 0.075
HITTER_PITCH_PRIOR_PA = 60.0
HITTER_PITCH_MIN_PA = 10.0
HITTER_CONTACT_DELTA = 0.025
HITTER_CONTACT_FAVORABLE_DELTA = 0.012
HITTER_ARSENAL_MIN_COVERAGE = 0.35
HITTER_ARSENAL_MIN_EFFECTIVE_PA = 10.0
LEAGUE_GAME_TOTAL = 8.5
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


def workload_k_distribution(k_rate, expected_bf, bf_low, bf_high, line=None, rate_strength=120.0):
    """Mix K outcomes over a distribution of possible starter workloads."""
    spread = max(2.0, (max(expected_bf, bf_high) - min(expected_bf, bf_low)) / 2.56)
    bf_values = list(range(8, 37))
    bf_weights = [exp(-0.5 * ((value - expected_bf) / spread) ** 2) for value in bf_values]
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
        "distribution": "workload_beta_binomial",
        "median": quantile(.5),
        "interval_low": quantile(.1),
        "interval_high": quantile(.9),
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


def k_opportunity_read(k_rate, expected_ks, lineup_rate, workload):
    """Classify K environment separately from evidence quality or a prop line."""
    if expected_ks >= 6.4 and k_rate >= .255 and workload["tone"] != "bad":
        label, tone = "high K environment", "good"
    elif expected_ks >= 5.4 and k_rate >= .225 and workload["tone"] != "bad":
        label, tone = "favorable K environment", "good"
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
    # Broader, handedness-matched hitter K history is the primary opponent
    # adjustment. Sparse exact-pitch evidence is deliberately a smaller
    # refinement instead of a prerequisite for a useful projection.
    # The confirmed-lineup split remains the primary opponent input, but its
    # influence now scales with evidence coverage. Exact-pitch final-PA data is
    # intentionally capped because it is much sparser and was over-influential
    # in v2 before any settled calibration sample existed.
    lineup_weight = .45 + .25 * lineup_evidence["coverage"]
    lineup_adjustment = lineup_weight * (lineup_evidence["rate"] - LEAGUE_K_RATE)
    pitch_weight = .25 * pitch_evidence["coverage"] * (
        pitch_evidence["effective_sample_size"] / (pitch_evidence["effective_sample_size"] + 180.0)
    )
    pitch_adjustment = clamp(pitch_weight * (pitch_evidence["rate"] - pitcher_rate), -.012, .012)
    k_rate = clamp(pitcher_rate + lineup_adjustment + pitch_adjustment, 0.06, 0.45)
    workload_estimate = _workload_estimate(side)
    expected_bf = workload_estimate["expected_batters_faced"]
    bf_low, bf_high = workload_estimate["batters_faced_interval"]
    appearances = workload_estimate["appearances"]
    expected_ks = k_rate * expected_bf
    leash = workload_read(side, workload_estimate)
    performance_outlook = pitcher_performance_outlook(side, workload_estimate)
    opportunity = k_opportunity_read(k_rate, expected_ks, lineup_evidence["rate"], leash)
    line = _number((market or {}).get("line"))
    rate_strength = clamp(80.0 + .15 * bf + .25 * lineup_evidence["effective_sample_size"], 80.0, 240.0)
    distribution = workload_k_distribution(
        k_rate, expected_bf, bf_low, bf_high, line, rate_strength=rate_strength,
    )
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
        f"Expected workload {expected_bf:.1f} batters faced",
        f"Expected pitch budget {workload_estimate['expected_pitches']:.0f} pitches",
        f"Opponent-adjusted efficiency {workload_estimate['pitches_per_batter']:.2f} pitches per batter",
        f"Opponent workload coverage {workload_estimate['lineup_workload_context']['coverage']:.0%}",
        f"Early-exit risk {workload_estimate['early_exit_risk']:.0%}",
    ]
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
        "milestone_probabilities": distribution["milestones"],
        "confidence": confidence,
        "arsenal_coverage": pitch_coverage,
        "effective_sample_size": pitch_evidence["effective_sample_size"],
        "hitters_covered": pitch_evidence["hitters_covered"],
        "lineup_k_evidence": lineup_evidence,
        "pitch_mix_evidence": pitch_evidence,
        "data_grade": data_grade,
        "opportunity": opportunity,
        "workload_read": leash,
        "performance_outlook": performance_outlook,
        "workload_stages": workload_estimate["workload_stages"],
        "lineup_workload_context": workload_estimate["lineup_workload_context"],
        "components": {
            "baseline_k_rate": pitcher_rate,
            "lineup_adjustment": lineup_adjustment,
            "pitch_adjustment": pitch_adjustment,
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


def hitter_pitch_summary(stat, league_average=LEAGUE_HITTER_AVERAGE):
    """Return a context-aware, shrinkage-safe contact and power read."""
    stat = stat or {}
    pa = max(0.0, _number(stat.get("pa"), 0.0))
    context = stat.get("context") or {}
    average = clamp(_number(context.get("adjusted_avg"), _number(stat.get("avg"), league_average)), 0.0, 1.0)
    advanced = stat.get("advanced") or {}
    power_value = _number(context.get("adjusted_slg"), _number(advanced.get("slg")))
    has_power = power_value is not None
    slugging = max(0.0, power_value if has_power else LEAGUE_HITTER_SLG)
    approximate_hits = average * pa
    posterior = shrunk_rate(approximate_hits, pa, league_average, HITTER_PITCH_PRIOR_PA)
    reliability = pa / (pa + HITTER_PITCH_PRIOR_PA) if pa else 0.0
    slugging_posterior = (slugging * pa + LEAGUE_HITTER_SLG * HITTER_PITCH_PRIOR_PA) / (pa + HITTER_PITCH_PRIOR_PA) if pa else LEAGUE_HITTER_SLG
    iso_posterior = max(0.0, slugging_posterior - posterior) if has_power else LEAGUE_HITTER_ISO
    avg_delta = posterior - league_average
    iso_delta = iso_posterior - LEAGUE_HITTER_ISO
    delta = .70 * avg_delta + .30 * iso_delta
    context_coverage = clamp(_number(context.get("coverage"), 0.0), 0.0, 1.0)
    if pa < HITTER_PITCH_MIN_PA:
        label, tone = "low data", "neutral"
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
    }


def hitter_market_context(market, batting_team):
    """Return a deliberately modest, transparent game-environment adjustment.

    The market total is a useful run-environment proxy and favorite status is a
    small proxy for team scoring opportunity. Neither turns pitch research into
    a hit-prop probability, so their combined contribution is capped at five
    points on the hitter fit scale.
    """
    market = market or {}
    total = _number(market.get("total"))
    total_adjustment = clamp(((total - LEAGUE_GAME_TOTAL) / 2.5) * .0035, -.0035, .0035) if total is not None else 0.0
    favorite = market.get("favorite") or {}
    favorite_team = favorite.get("team") if isinstance(favorite, dict) else None
    favorite_adjustment = .0015 if favorite_team and batting_team and favorite_team == batting_team else 0.0
    return {
        "total": total,
        "favorite": favorite_team,
        "favorite_probability": _number(favorite.get("probability")) if isinstance(favorite, dict) else None,
        "total_adjustment": total_adjustment,
        "favorite_adjustment": favorite_adjustment,
        "adjustment": total_adjustment + favorite_adjustment,
        "available": total is not None or favorite_team is not None,
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
    favorite = _number(market_context.get("favorite_adjustment"), 0.0) > 0

    contact_score = (
        0.72 * (expected_average - LEAGUE_HITTER_AVERAGE) / 0.025
        + 0.28 * (LEAGUE_K_RATE - k_rate) / 0.040
    )
    power_score = (
        0.52 * (expected_slg - LEAGUE_HITTER_SLG) / 0.080
        + 0.30 * (expected_iso - LEAGUE_HITTER_ISO) / 0.060
        + 0.18 * (hard_hit_rate - LEAGUE_HARD_HIT_RATE) / 0.060
    )
    home_run_score = (
        0.42 * (hr_rate - LEAGUE_HITTER_HR_RATE) / 0.015
        + 0.33 * (expected_iso - LEAGUE_HITTER_ISO) / 0.060
        + 0.25 * (barrel_rate - LEAGUE_BARREL_PROXY_RATE) / 0.030
    )
    lineup_score = clamp((projected_pa - 4.2) / 0.45, -1.0, 1.0)
    environment_score = clamp(((total - LEAGUE_GAME_TOTAL) / 1.5) if total is not None else 0.0, -1.0, 1.0)
    if favorite:
        environment_score = clamp(environment_score + 0.20, -1.0, 1.0)
    run_score = 0.30 * contact_score + 0.25 * power_score + 0.30 * lineup_score + 0.15 * environment_score
    offense_scores = sorted((contact_score, power_score, home_run_score), reverse=True)
    overall_score = 0.50 * offense_scores[0] + 0.30 * offense_scores[1] + 0.20 * run_score

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

    if coverage >= 0.55 and sample >= 25 and context_coverage >= 0.20:
        evidence = "strong"
    elif coverage >= HITTER_ARSENAL_MIN_COVERAGE and sample >= HITTER_ARSENAL_MIN_EFFECTIVE_PA:
        evidence = "usable"
    else:
        evidence = "limited"

    def build(key, title, score, value, drivers, extra_risks=()):
        if evidence == "limited":
            tier, tone = "watchlist", "neutral"
        elif score >= 0.75:
            tier, tone = "strong", "good"
        elif score >= 0.25:
            tier, tone = "favorable", "good"
        elif score <= -0.40:
            tier, tone = "tough", "bad"
        else:
            tier, tone = "neutral", "neutral"
        return {
            "key": key,
            "title": title,
            "score": round(score, 3),
            "tier": tier,
            "tone": tone,
            "value": value,
            "drivers": drivers[:3],
            "risks": list(dict.fromkeys([*base_risks, *extra_risks]))[:3],
            "evidence": evidence,
        }

    quality_risks = [] if quality_coverage >= 0.20 else ["Limited hard-hit and barrel evidence"]
    opportunities = {
        "hit": build(
            "hit", "Hit opportunity", contact_score,
            f"{expected_average:.3f} matchup AVG",
            [f"Matchup AVG {expected_average:.3f} vs {LEAGUE_HITTER_AVERAGE:.3f} MLB",
             f"Descriptive K rate {k_rate:.1%}", f"Arsenal coverage {coverage:.0%}"],
        ),
        "total_bases": build(
            "total_bases", "Total-base power", power_score,
            f"{expected_slg:.3f} matchup SLG",
            [f"Matchup SLG {expected_slg:.3f} vs {LEAGUE_HITTER_SLG:.3f} MLB",
             f"Matchup ISO {expected_iso:.3f}", f"Hard-hit read {hard_hit_rate:.1%}"],
            quality_risks,
        ),
        "home_run": build(
            "home_run", "Home-run power", home_run_score,
            f"{expected_iso:.3f} matchup ISO",
            [f"Matchup ISO {expected_iso:.3f} vs {LEAGUE_HITTER_ISO:.3f} MLB",
             f"Barrel proxy {barrel_rate:.1%}", f"Pitch-ending HR rate {hr_rate:.1%}"],
            quality_risks,
        ),
        "runs_rbi": build(
            "runs_rbi", "Runs + RBI opportunity", run_score,
            f"{projected_pa:.1f} expected PA",
            [f"Batting {lineup_order or 'order pending'} · about {projected_pa:.1f} PA",
             f"Game total {total:.1f}" if total is not None else "Game total unavailable",
             "Batting team is favored" if favorite else f"Matchup SLG {expected_slg:.3f}"],
            ["Runs/RBIs also depend on surrounding hitters and bullpen"] if lineup_order else [],
        ),
    }
    primary_key = max(opportunities, key=lambda key: opportunities[key]["score"])
    primary = opportunities[primary_key]
    opportunities["overall"] = build(
        "overall", "Overall offensive opportunity", overall_score,
        f"{primary['title']}",
        [f"Best path: {primary['title']}", *primary["drivers"][:2]],
        primary["risks"],
    )
    return {"primary": primary_key, "projected_pa": projected_pa, "items": opportunities}


def hitter_arsenal_summary(batter, pitches, league_average=LEAGUE_HITTER_AVERAGE, market_context=None):
    """Shrink descriptive contact results into transparent strong/favorable tiers.

    These are research tiers, not hit-prop probabilities or wagering signals.
    Strong retains the original +25-point standard. Favorable gives confirmed
    lineups a useful middle tier once coverage is broad enough.
    """
    total_usage = sum(max(0.0, _number(p.get("usage"), 0.0)) for p in pitches)
    market_context = market_context or hitter_market_context(None, None)
    if total_usage <= 0:
        empty = {"label": "insufficient", "tier": "watchlist", "tone": "neutral", "coverage": 0.0, "effective_sample_size": 0.0, "expected_average": league_average, "expected_slg": LEAGUE_HITTER_SLG, "expected_iso": LEAGUE_HITTER_ISO, "delta": 0.0, "base_score": 0.0, "score": market_context["adjustment"], "market_context": market_context, "context_coverage": 0.0, "quality_coverage": 0.0, "hard_hit_rate": LEAGUE_HARD_HIT_RATE, "barrel_rate": LEAGUE_BARREL_PROXY_RATE, "hr_rate": LEAGUE_HITTER_HR_RATE}
        empty["opportunities"] = hitter_opportunity_reads(batter, empty, market_context)
        return empty
    expected = expected_slg = expected_iso = coverage = context_coverage = sample = score = 0.0
    hard_hit_rate = barrel_rate = hr_rate = quality_coverage = 0.0
    for pitch in pitches:
        usage = max(0.0, _number(pitch.get("usage"), 0.0)) / total_usage
        stat = (batter.get("vs_pitches") or {}).get(pitch.get("code"))
        pitch_read = hitter_pitch_summary(stat, league_average)
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
    if coverage < HITTER_ARSENAL_MIN_COVERAGE or sample < HITTER_ARSENAL_MIN_EFFECTIVE_PA:
        label, tier, tone = "insufficient", "watchlist", "neutral"
    elif score >= HITTER_CONTACT_DELTA:
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
    }
    result["opportunities"] = hitter_opportunity_reads(batter, result, market_context)
    return result


def blend_full_game_hitter_matchup(batter, starter_summary, bullpen_entries,
                                   expected_batters_faced,
                                   batters_faced_interval=None,
                                   market_context=None):
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
    if result["coverage"] < HITTER_ARSENAL_MIN_COVERAGE or result["effective_sample_size"] < HITTER_ARSENAL_MIN_EFFECTIVE_PA:
        result.update({"label": "insufficient", "tier": "watchlist", "tone": "neutral"})
    elif result["score"] >= HITTER_CONTACT_DELTA:
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
    return result
