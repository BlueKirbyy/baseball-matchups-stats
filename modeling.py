"""Versioned, server-side models and market math for Diamond Intel.

The initial model is intentionally a conservative empirical-Bayes baseline. It
is a research model until walk-forward results establish calibration and value.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import exp, floor, isfinite, lgamma, log, sqrt
from statistics import median

MODEL_VERSION = "pitcher-k-eb-v2"
FEATURE_VERSION = "gameday-features-v2"
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


def _lineup_weight(order):
    expected_pa = (4.75, 4.65, 4.55, 4.45, 4.35, 4.25, 4.15, 4.05, 3.95)
    try:
        return expected_pa[int(order) - 1]
    except (TypeError, ValueError, IndexError):
        return 4.2


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


def workload_read(side, expected_batters_faced, appearances, source):
    """Return a descriptive starter-leash read; it is not an extra K adjustment."""
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
    elif delta <= -2.0 or recent_pitches <= season_pitches - 12:
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
        "appearances": appearances,
        "source": source,
    }


def k_opportunity_read(k_rate, expected_ks, lineup_rate, workload):
    """Classify K environment separately from evidence quality or a prop line."""
    if expected_ks >= 6.4 and k_rate >= .255 and workload["tone"] != "bad":
        label, tone = "excellent K opportunity", "good"
    elif expected_ks >= 5.4 and k_rate >= .225 and workload["tone"] != "bad":
        label, tone = "good K opportunity", "good"
    elif expected_ks <= 4.4 or k_rate <= .185 or workload["tone"] == "bad":
        label, tone = "limited K opportunity", "bad"
    else:
        label, tone = "neutral K opportunity", "neutral"
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


def _workload_estimate(side):
    starts = [row for row in side.get("appearance_history", []) if row.get("is_start")]
    if starts:
        season = [_number(row.get("batters_faced"), 0.0) for row in starts]
        recent = season[-3:]
        expected = (LEAGUE_BF_PER_START * 4 + sum(season) + 0.5 * sum(recent)) / (4 + len(season) + 0.5 * len(recent))
        spread = max(2.5, sqrt(sum((value - expected) ** 2 for value in season) / max(1, len(season) - 1)))
        return expected, max(8.0, expected - 1.28 * spread), expected + 1.28 * spread, len(starts), "start-only history"
    workload = side.get("workload") or {}
    appearances = int(_number(workload.get("appearances"), 0))
    batters_faced = _number(workload.get("batters_faced"), 0.0)
    recent_appearances = int(_number(workload.get("recent_appearances"), 0))
    recent_bf = _number(workload.get("recent_batters_faced"), 0.0)
    if not appearances or not batters_faced:
        return LEAGUE_BF_PER_START, 14.0, 28.0, 0, "league prior"
    season_mean = batters_faced / appearances
    recent_mean = recent_bf / recent_appearances if recent_appearances else season_mean
    expected = (LEAGUE_BF_PER_START * 4 + season_mean * appearances + recent_mean * min(recent_appearances, 3)) / (4 + appearances + min(recent_appearances, 3))
    spread = max(3.0, abs(season_mean - recent_mean) + 2.0)
    return expected, max(8.0, expected - spread), expected + spread, appearances, "mixed-role aggregate fallback"


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
    lineup_adjustment = .75 * (lineup_evidence["rate"] - LEAGUE_K_RATE)
    pitch_weight = pitch_evidence["effective_sample_size"] / (pitch_evidence["effective_sample_size"] + 120.0)
    pitch_adjustment = pitch_weight * (pitch_evidence["rate"] - pitcher_rate)
    k_rate = clamp(pitcher_rate + lineup_adjustment + pitch_adjustment, 0.06, 0.45)
    expected_bf, bf_low, bf_high, appearances, workload_source = _workload_estimate(side)
    expected_ks = k_rate * expected_bf
    leash = workload_read(side, expected_bf, appearances, workload_source)
    opportunity = k_opportunity_read(k_rate, expected_ks, lineup_evidence["rate"], leash)
    line = _number((market or {}).get("line"))
    distribution = distribution_summary(expected_ks, line, kind="negative_binomial", dispersion=12.0)
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
        "confidence": confidence,
        "arsenal_coverage": pitch_coverage,
        "effective_sample_size": pitch_evidence["effective_sample_size"],
        "hitters_covered": pitch_evidence["hitters_covered"],
        "lineup_k_evidence": lineup_evidence,
        "pitch_mix_evidence": pitch_evidence,
        "data_grade": data_grade,
        "opportunity": opportunity,
        "workload_read": leash,
        "components": {
            "baseline_k_rate": pitcher_rate,
            "lineup_adjustment": lineup_adjustment,
            "pitch_adjustment": pitch_adjustment,
            "matchup_k_rate": k_rate,
            "expected_batters_faced": expected_bf,
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
    k_rate = hitter_k_risk(batter.get("k_profile"))["posterior"]
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
