#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster: five quantiles per trophy, a week out.

For each trophy of Grand Theft Auto VI on PlayStation 5, forecast the share
of Exophase-tracked players who will hold it at the case's target, as five
quantiles (5%, 25%, 50%, 75%, 95%) on the percent scale. The score is a
pinball loss on the logit scale, so the work happens in logits and the
spread is an honest statement of uncertainty, not padding.

What the data looks like, and what the model does about it
-----------------------------------------------------------
A case's history is a sequence of reads of one public trophy page. Two
forces move every rate in opposite directions:

* players earn trophies as they play, which pushes each trophy's earned
  count up, and
* Exophase tracks more players over launch week, which dilutes every
  rate, because a newly tracked player has usually earned less than the
  early cohort.

A rate-only trend confuses the two. So the forecast blends two views of
the same history, both fitted on log time (hours since the release day
began, 2026-11-19T00:00Z, through ln(1 + hours), which slows down as the
week goes on, the way launch curves do):

1. a per-trophy least-squares line of logit(rate) on log time over the
   last 24 hours of reads, its slope shrunk toward the cross-trophy
   median slope (a trophy with three flat reads should not get a wild
   slope of its own), and
2. a count model: log earned-count and log players-tracked each get
   their own line on log time over the whole history, and the rate
   forecast is the ratio of the two extrapolated counts. This is the
   view that knows dilution: when the player base is doubling, earned
   counts must more than double just to hold a rate flat. The count
   view uses the full history, not just the last day, because the last
   day's growth overstates the week when the player base is near its
   fastest growth. It gets a quarter of the blend: the trend view is
   the steadier of the two when curves bend, and the count view is
   there to catch dilution the trend view misses.

The two medians are blended in logit space (three quarters trend, one
quarter count), then pulled a little way back toward the last observed
rate (shrinkage that grows with how much distinct history the trophy
actually has). The spread combines the house random-walk term (0.5 logit
per root-day) with the fit's residual spread, the slope's own
uncertainty over the extrapolation gap, and a penalty when the history
is stale.

The stale-history case matters: on launch night Exophase did not refresh
its counts for hours, so a whole history can be one frozen page. A flat
history is not evidence that rates stand still; it is evidence that the
next refresh has not landed yet. The kit documents what that first
refresh did: tracked players jumped 638 -> 3,868 while the trophies
earned outside the one everybody-held trophy barely moved (305 -> 408),
so every other trophy's rate fell to roughly a fifth of its frozen
value. When a trophy's history shows no real movement, the median is
therefore shifted down by about that scale of dilution (a little less
for the easy early trophies new players earn quickly, a little more for
the hard ones almost no newcomer holds), and the spread is widened,
because the exact size of the next refresh is genuinely unknown. The
one trophy everybody already holds (a 100% rate) is the exception: a
first-mission trophy stays near 100% as new players arrive.

Standard library only, no network, deterministic: the same case always
writes the same bytes. Reads every *.in file in $MAKEFUN_INPUT_DIR (else
/input) in name order and writes one <id>.out per case to
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at those levels.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
WINDOW_HOURS = 24.0        # the house curve baseline's lookback window
BASE_SIGMA = 0.5           # random-walk logit spread per root-day
SIGMA_FLOOR = 0.20
SIGMA_CAP = 1.80
SLOPE_CAP = 4.0            # logit per log-hour unit, either direction
MEDIAN_MOVE_CAP = 4.0      # furthest the median may move from last, logit
FLAT_SPREAD = 0.70         # extra logit spread when history is frozen
PLAYERS_GROWTH_CAP = 30.0  # players at target, as a multiple of last

# Expected refresh dilution (logit shift of the median) when a trophy's
# history is frozen. The documented first refresh took the average
# non-trivial trophy to about a fifth of its frozen rate (a logit move
# near -1.5); easy early trophies fall less, because new players earn
# them quickly, and the hardest trophies fall furthest.
FLAT_SHIFT = {
    "platinum": -1.80,
    "gold": -1.75,
    "silver": -1.60,
    "bronze_rare": -1.50,
    "bronze_common": -1.30,
}


def parse_ts(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def log_time(moment):
    """ln(1 + hours from the release day's start), floored at zero."""
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return math.log1p(max(hours, 0.0))


def clamp_rate(pct):
    """Clamp a percent into the range the logit is defined on."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    """Logit of a percent (0..100) after clamping."""
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    """Inverse logit, back to a percent in (0, 100)."""
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    top = math.exp(value)
    return 100.0 * top / (1.0 + top)


def round4(value):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    rounded = Decimal(repr(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP)
    as_float = float(rounded)
    return int(as_float) if as_float.is_integer() else as_float


def ols(points):
    """Least squares line through (x, y) points.

    Returns (slope, intercept, residual_sd, n, x_mean, sxx), or None
    when the points cannot define a line (too few, or no x spread).
    """
    n = len(points)
    if n < 2:
        return None
    x_mean = sum(p[0] for p in points) / n
    y_mean = sum(p[1] for p in points) / n
    sxx = sum((p[0] - x_mean) ** 2 for p in points)
    if sxx <= 1e-12:
        return None
    sxy = sum((p[0] - x_mean) * (p[1] - y_mean) for p in points)
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    if n > 2:
        resid_var = sum((p[1] - (intercept + slope * p[0])) ** 2
                        for p in points) / (n - 2)
    else:
        resid_var = 0.0
    return (slope, intercept, math.sqrt(max(resid_var, 0.0)), n,
            x_mean, sxx)


def median(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def distinct_count(values):
    """How many distinct values a series takes (rounded to 2 decimals)."""
    return len({round(v, 2) for v in values})


def flat_shift(trophy, last_rate):
    """Refresh-dilution logit shift for a trophy with frozen history."""
    if last_rate >= 99.0:
        return 0.0
    tier = (trophy.get("tier") or "bronze").lower()
    if tier == "bronze":
        key = "bronze_common" if last_rate >= 1.0 else "bronze_rare"
        return FLAT_SHIFT[key]
    return FLAT_SHIFT.get(tier, -0.50)


def series_for(case, index):
    """Per-read (time, x, rate, players) for one trophy, oldest first."""
    rows = []
    for read in case.get("history") or []:
        rates = read.get("rates") or []
        if index >= len(rates):
            continue
        rate = rates[index]
        if rate is None:
            continue
        try:
            moment = parse_ts(read["at"])
            players = float(read["players_tracked"])
        except (KeyError, TypeError, ValueError):
            continue
        if players <= 0:
            continue
        rows.append((moment, log_time(moment), float(rate), players))
    return rows


def window_points(rows, checkpoint, want):
    """(x, y) points in the last-24h window for one transform.

    want is a function of (moment, x, rate, players) giving y, or None
    to skip the read. Falls back to the full history when the window
    holds fewer than 3 points.
    """
    cutoff = checkpoint - timedelta(hours=WINDOW_HOURS)
    recent = [(x, want(m, x, r, p)) for (m, x, r, p) in rows
              if m >= cutoff]
    recent = [(x, y) for (x, y) in recent if y is not None]
    if len(recent) >= 3:
        return recent
    full = [(x, want(m, x, r, p)) for (m, x, r, p) in rows]
    return [(x, y) for (x, y) in full if y is not None]


def forecast_trophy(trophy, index, case, checkpoint, x_target,
                    horizon_days, global_slope, global_resid,
                    players_fit):
    """Five quantiles for one trophy; never raises."""
    rows = series_for(case, index)
    last_rate = None
    if rows:
        last_rate = rows[-1][2]
    if last_rate is None:
        # No usable read at all: a wide, low, honest guess.
        centre = logit(1.0)
        sigma = 1.4
        return quantiles_from(centre, sigma)

    last_logit = logit(last_rate)
    rates_seen = [r for (_, _, r, _) in rows]
    players_seen = [p for (_, _, _, p) in rows]
    n_distinct = distinct_count(rates_seen)
    frozen = n_distinct <= 1 and distinct_count(players_seen) <= 1

    # --- View 1: logit(rate) trend on log time, slope shrunk. ---
    trend_logit = None
    trend_resid = global_resid
    slope_unc = 0.0
    fit = ols(window_points(
        rows, checkpoint,
        lambda m, x, r, p: logit(r)))
    if fit is not None:
        slope, intercept, resid, n, x_mean, sxx = fit
        weight = n_distinct / (n_distinct + 3.0)
        slope = weight * slope + (1.0 - weight) * global_slope
        slope = max(-SLOPE_CAP, min(SLOPE_CAP, slope))
        trend_logit = intercept + slope * x_target
        trend_resid = resid
        # Standard error of the slope, carried over the extrapolation gap.
        gap = max(0.0, x_target - x_mean)
        if n > 2 and sxx > 0:
            slope_unc = (resid / math.sqrt(sxx)) * gap
        slope_unc = min(slope_unc, 0.8)

    # --- View 2: earned-count / players-count ratio at the target. ---
    count_logit = None
    if players_fit is not None and rows:
        last_players = rows[-1][3]
        last_count = max(last_rate / 100.0 * last_players, 0.25)
        # Full history for the count view: see the module docstring.
        earned_fit = ols([(x, math.log(max(r / 100.0 * p, 0.25)))
                          for (_, x, r, p) in rows])
        p_slope, p_intercept = players_fit
        players_target = math.exp(p_intercept + p_slope * x_target)
        players_target = max(last_players,
                             min(players_target,
                                 last_players * PLAYERS_GROWTH_CAP))
        if earned_fit is not None:
            e_slope, e_intercept = earned_fit[0], earned_fit[1]
            e_slope = max(-SLOPE_CAP, min(SLOPE_CAP, e_slope))
            earned_target = math.exp(e_intercept + e_slope * x_target)
            earned_target = max(earned_target, last_count)
            earned_target = min(earned_target, players_target)
            rate_count = 100.0 * earned_target / players_target
            count_logit = logit(rate_count)

    # --- Blend the views, shrink toward the last read. ---
    if trend_logit is not None and count_logit is not None:
        centre = 0.75 * trend_logit + 0.25 * count_logit
    elif trend_logit is not None:
        centre = trend_logit
    elif count_logit is not None:
        centre = count_logit
    else:
        centre = last_logit
    if centre != last_logit:
        shrink = min(n_distinct, 15) / (min(n_distinct, 15) + 1.0)
        centre = last_logit + shrink * (centre - last_logit)
    centre = last_logit + max(-MEDIAN_MOVE_CAP,
                              min(MEDIAN_MOVE_CAP, centre - last_logit))

    # --- Frozen history: expect the refresh, and admit ignorance. ---
    # With a frozen history both views above are just persistence, so
    # they carry no information beyond the last rate: the refresh prior
    # is the whole median. With only one move in the history the views
    # carry a little signal, so the prior gets a quarter weight there.
    extra_var = 0.0
    if frozen:
        centre = last_logit + flat_shift(trophy, last_rate)
        extra_var = FLAT_SPREAD ** 2
    elif n_distinct <= 2:
        # One move in the whole history: half the refresh prior, half
        # the flat-history penalty.
        prior = last_logit + flat_shift(trophy, last_rate)
        centre = 0.75 * centre + 0.25 * prior
        extra_var = (0.5 * FLAT_SPREAD) ** 2

    # --- Spread: random walk + fit residual + slope risk + flat risk. ---
    resid = 0.5 * (trend_resid + global_resid)
    variance = ((BASE_SIGMA * math.sqrt(horizon_days)) ** 2
                + resid ** 2 + slope_unc ** 2 + extra_var)
    # Tiny-count trophies carry extra sampling noise in their rates.
    if rows:
        last_count = max(last_rate / 100.0 * rows[-1][3], 1.0)
        variance += min(0.25, 0.25 / last_count)
    sigma = math.sqrt(variance)
    sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
    return quantiles_from(centre, sigma)


def quantiles_from(centre, sigma):
    """Five rounded, non-decreasing quantiles around a logit centre."""
    out = []
    prev = None
    for z in Z:
        value = round4(inv_logit(centre + z * sigma))
        if prev is not None and value < prev:
            value = prev
        out.append(value)
        prev = value
    return out


def persistence_rows(case):
    """Fallback forecast: last rate, house-style widening spread."""
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    sigma = max(SIGMA_FLOOR, BASE_SIGMA * math.sqrt(horizon_days))
    rows = []
    history = case.get("history") or []
    last_rates = history[-1].get("rates") if history else []
    for i, trophy in enumerate(case.get("trophies") or []):
        rate = None
        if i < len(last_rates or []):
            rate = last_rates[i]
        centre = logit(rate if rate is not None else 1.0)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from(centre, sigma)})
    return {"trophies": rows}


def forecast_case(case):
    """Forecast one case; falls back to persistence on any surprise."""
    try:
        checkpoint = parse_ts(case["at"])
        target = parse_ts(case["target"])
    except (KeyError, TypeError, ValueError):
        return persistence_rows(case)
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    x_target = log_time(target)

    trophies = case.get("trophies") or []

    # Cross-trophy slope and residual medians, for shrinkage targets.
    slopes = []
    resids = []
    for i in range(len(trophies)):
        rows = series_for(case, i)
        fit = ols(window_points(
            rows, checkpoint, lambda m, x, r, p: logit(r)))
        if fit is not None:
            slopes.append(max(-SLOPE_CAP, min(SLOPE_CAP, fit[0])))
            resids.append(fit[2])
    global_slope = median(slopes) if slopes else 0.0
    global_resid = median(resids) if resids else 0.3

    # Players-tracked trend on log time (the dilution view's denominator).
    players_fit = None
    player_rows = []
    for read in case.get("history") or []:
        try:
            moment = parse_ts(read["at"])
            players = float(read["players_tracked"])
        except (KeyError, TypeError, ValueError):
            continue
        if players > 0:
            player_rows.append(
                (moment, log_time(moment), math.log(players)))
    if len(player_rows) >= 2:
        # Full history: the last day's growth overstates the week when
        # the player base is near its fastest growth (a saturating
        # curve's inflection), so the dilution view uses the whole span.
        use = [(x, y) for (_, x, y) in player_rows]
        fit = ols(use)
        if fit is not None:
            players_fit = (max(-SLOPE_CAP, min(SLOPE_CAP, fit[0])),
                           fit[1])

    rows = []
    for i, trophy in enumerate(trophies):
        try:
            quantiles = forecast_trophy(
                trophy, i, case, checkpoint, x_target, horizon_days,
                global_slope, global_resid, players_fit)
        except Exception:
            history = case.get("history") or []
            last_rates = history[-1].get("rates") if history else []
            rate = (last_rates[i] if i < len(last_rates or [])
                    and last_rates[i] is not None else 1.0)
            sigma = max(SIGMA_FLOOR,
                        BASE_SIGMA * math.sqrt(horizon_days))
            quantiles = quantiles_from(logit(rate), sigma)
        rows.append({"id": trophy["id"], "q": quantiles})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir)
                   if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r",
                  encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
