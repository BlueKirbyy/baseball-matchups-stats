"""Versioned empirical MLB venue factors used by the matchup model.

Values are Baseball Savant Statcast park-factor indices, where 100 is league
average.  Standard MLB venues use the 2024-2026 three-year window; Sutter
Health Park uses 2025-2026 because it does not have a third MLB season.

Tuple layout: (overall park factor, 1B, 2B, 3B, HR).
Source: https://baseballsavant.mlb.com/leaderboard/statcast-park-factors
Captured: 2026-08-17.
"""

STATCAST_PARK_FACTOR_SOURCE = (
    "Baseball Savant Statcast Park Factors · handedness split · rolling window"
)
TOTAL_BASE_EVENT_WEIGHTS = {"1B": .44, "2B": .27, "3B": .03, "HR": .26}


PARK_FACTORS = {
    1: {"name": "Angel Stadium", "years": "2024-2026", "L": (98, 99, 92, 80, 97), "R": (100, 99, 91, 92, 105)},
    2: {"name": "Oriole Park at Camden Yards", "years": "2024-2026", "L": (103, 102, 97, 87, 120), "R": (102, 104, 105, 150, 100)},
    3: {"name": "Fenway Park", "years": "2024-2026", "L": (103, 101, 138, 89, 83), "R": (102, 107, 105, 83, 89)},
    4: {"name": "Rate Field", "years": "2024-2026", "L": (100, 96, 89, 86, 103), "R": (98, 102, 98, 52, 90)},
    5: {"name": "Progressive Field", "years": "2024-2026", "L": (98, 99, 101, 58, 101), "R": (100, 100, 108, 76, 85)},
    7: {"name": "Kauffman Stadium", "years": "2024-2026", "L": (101, 103, 124, 182, 78), "R": (101, 100, 115, 152, 93)},
    12: {"name": "Tropicana Field", "years": "2024-2026", "L": (93, 96, 90, 98, 90), "R": (100, 96, 101, 125, 115)},
    14: {"name": "Rogers Centre", "years": "2024-2026", "L": (98, 94, 108, 70, 104), "R": (103, 104, 100, 69, 108)},
    15: {"name": "Chase Field", "years": "2024-2026", "L": (100, 104, 109, 208, 77), "R": (106, 102, 113, 213, 105)},
    17: {"name": "Wrigley Field", "years": "2024-2026", "L": (98, 99, 83, 134, 102), "R": (95, 96, 76, 101, 103)},
    19: {"name": "Coors Field", "years": "2024-2026", "L": (114, 114, 120, 200, 116), "R": (111, 116, 125, 214, 104)},
    22: {"name": "Dodger Stadium", "years": "2024-2026", "L": (98, 93, 91, 60, 119), "R": (103, 94, 95, 67, 131)},
    31: {"name": "PNC Park", "years": "2024-2026", "L": (104, 99, 130, 76, 97), "R": (99, 106, 108, 77, 71)},
    32: {"name": "American Family Field", "years": "2024-2026", "L": (96, 97, 84, 110, 94), "R": (98, 93, 90, 60, 110)},
    680: {"name": "T-Mobile Park", "years": "2024-2026", "L": (93, 93, 91, 31, 97), "R": (90, 88, 83, 46, 98)},
    2392: {"name": "Daikin Park", "years": "2024-2026", "L": (102, 99, 89, 61, 121), "R": (100, 97, 100, 106, 109)},
    2394: {"name": "Comerica Park", "years": "2024-2026", "L": (100, 101, 90, 204, 97), "R": (101, 100, 95, 70, 106)},
    2395: {"name": "Oracle Park", "years": "2024-2026", "L": (97, 102, 106, 141, 73), "R": (97, 103, 104, 139, 80)},
    2529: {"name": "Sutter Health Park", "years": "2025-2026", "L": (108, 99, 120, 52, 118), "R": (113, 108, 125, 67, 124)},
    2602: {"name": "Great American Ball Park", "years": "2024-2026", "L": (104, 97, 98, 85, 122), "R": (100, 93, 102, 63, 113)},
    2680: {"name": "Petco Park", "years": "2024-2026", "L": (96, 96, 89, 76, 93), "R": (98, 98, 85, 64, 115)},
    2681: {"name": "Citizens Bank Park", "years": "2024-2026", "L": (104, 98, 96, 87, 131), "R": (100, 105, 95, 126, 100)},
    2889: {"name": "Busch Stadium", "years": "2024-2026", "L": (95, 107, 105, 67, 72), "R": (98, 108, 109, 76, 78)},
    3289: {"name": "Citi Field", "years": "2024-2026", "L": (99, 100, 96, 70, 95), "R": (98, 92, 93, 90, 106)},
    3309: {"name": "Nationals Park", "years": "2024-2026", "L": (103, 107, 99, 96, 109), "R": (102, 104, 98, 92, 99)},
    3312: {"name": "Target Field", "years": "2024-2026", "L": (100, 104, 103, 85, 93), "R": (105, 103, 121, 104, 97)},
    3313: {"name": "Yankee Stadium", "years": "2024-2026", "L": (102, 90, 98, 59, 119), "R": (100, 91, 92, 85, 117)},
    4169: {"name": "loanDepot park", "years": "2024-2026", "L": (101, 105, 99, 135, 93), "R": (99, 98, 108, 140, 82)},
    4705: {"name": "Truist Park", "years": "2024-2026", "L": (100, 104, 92, 78, 105), "R": (101, 104, 99, 99, 96)},
    5325: {"name": "Globe Life Field", "years": "2024-2026", "L": (95, 99, 93, 85, 96), "R": (93, 94, 92, 85, 93)},
}


def venue_factor(venue_id, bat_side):
    """Return outcome-specific multipliers for a venue and hitter side."""
    try:
        profile = PARK_FACTORS.get(int(venue_id))
    except (TypeError, ValueError):
        profile = None
    if not profile:
        return None
    side = bat_side if bat_side in {"L", "R"} else None
    if side:
        values = profile[side]
    else:
        values = tuple((left + right) / 2 for left, right in zip(profile["L"], profile["R"]))
    overall, one_base, two_base, three_base, home_run = values
    total_bases = (
        TOTAL_BASE_EVENT_WEIGHTS["1B"] * one_base
        + TOTAL_BASE_EVENT_WEIGHTS["2B"] * two_base
        + TOTAL_BASE_EVENT_WEIGHTS["3B"] * three_base
        + TOTAL_BASE_EVENT_WEIGHTS["HR"] * home_run
    )
    return {
        "venue_id": int(venue_id),
        "venue": profile["name"],
        "years": profile["years"],
        "bat_side": side or "B",
        "overall_index": overall,
        "single_index": one_base,
        "double_index": two_base,
        "triple_index": three_base,
        "home_run_index": home_run,
        "total_bases_index": round(total_bases, 1),
        "home_run_multiplier": home_run / 100.0,
        "total_bases_multiplier": total_bases / 100.0,
        "source": STATCAST_PARK_FACTOR_SOURCE,
    }
