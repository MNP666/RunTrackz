"""
runtrackz.treadmill_analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Treadmill-specific analysis with manually supplied gradient schedules.

Treadmill wearables often mis-record or omit incline data entirely.  This
module lets you supply the gradient as a function of elapsed time and computes
grade-adjusted pace (GAP) together with per-segment metrics.

Gradient schedule formats
-------------------------
Both a list of breakpoints **or** a callable are accepted:

1. **Step-function list** (most common for fixed-programme workouts)::

       gradient = [(0, 1.0), (300, 5.0), (1500, 1.0)]

   Each tuple ``(start_s, gradient_pct)`` says "from this elapsed second
   onward, the incline is X %".  The final entry stays in effect until the
   end of the run.

2. **Callable** ``f(elapsed_s: float) -> gradient_pct``::

       gradient = lambda t: 1.0 if t < 300 else 5.0 if t < 1500 else 1.0

Grade-adjusted pace (GAP)
--------------------------
GAP is computed using the **Minetti et al. (2002)** metabolic-cost model::

    E(i) = 155.4·i⁵ − 30.4·i⁴ − 43.3·i³ + 46.3·i² + 19.5·i + 3.6   [J/(kg·m)]

where *i* is the fractional grade (e.g. 0.05 for 5 %) and E(0) = 3.6.

    gap_factor = E(i) / E(0)
    GAP_pace   = actual_pace / gap_factor

A 5 % incline gives gap_factor ≈ 1.27, so a 5:00 /km effort at 5 % is
equivalent to running 3:56 /km on flat.

Example
-------
    import runtrackz
    from runtrackz import treadmill_analysis as ta

    run   = runtrackz.load("morning_treadmill.fit")
    hr    = runtrackz.hr_analysis.analyze(run, config=cfg)
    pace  = runtrackz.pace_analysis.analyze(run)

    # 5 min warm-up 1 %, 20 min main block at 5 %, 5 min cool-down 1 %
    stats = ta.analyze(run, hr, pace,
                       gradient=[(0, 1.0), (300, 5.0), (1500, 1.0)])
    print(stats.summary())

    # Quick reference table
    print(ta.gap_table([0, 1, 2, 3, 4, 5, 6, 8, 10]))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------

GradientSchedule = Union[
    List[Tuple[float, float]],  # [(start_elapsed_s, gradient_pct), ...]
    Callable[[float], float],   # f(elapsed_s) -> gradient_pct
]

# ---------------------------------------------------------------------------
# Minetti 2002 metabolic cost model
# ---------------------------------------------------------------------------

_E0 = 3.6  # J/(kg·m) — flat running metabolic cost


def _minetti_cost(i: float) -> float:
    """
    Metabolic cost of running at fractional grade *i* [J/(kg·m)].

    Minetti et al. (2002), valid for i ∈ [−0.45, +0.45].
    """
    return 155.4*i**5 - 30.4*i**4 - 43.3*i**3 + 46.3*i**2 + 19.5*i + _E0


def _gap_factor(gradient_pct: float) -> float:
    """
    Ratio of metabolic cost at *gradient_pct* % to flat cost.

    gap_factor > 1  → slope is harder than flat.
    Multiply actual speed by gap_factor to get flat-equivalent speed.
    Divide actual pace (min/km) by gap_factor to get GAP pace.
    """
    i = gradient_pct / 100.0
    return _minetti_cost(i) / _E0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GradientSegment:
    """Metrics for one contiguous gradient block."""

    index: int                   # 1-based segment number
    start_s: float               # elapsed seconds at block start
    end_s: float                 # elapsed seconds at block end
    duration_s: float
    gradient_pct: float
    distance_m: float
    avg_pace_min_km: float       # actual treadmill pace
    avg_gap_min_km: float        # grade-adjusted pace (flat equivalent)
    avg_hr: Optional[float]
    avg_power_w: Optional[float]

    @property
    def pace_str(self) -> str:
        return _fmt_pace(self.avg_pace_min_km)

    @property
    def gap_str(self) -> str:
        return _fmt_pace(self.avg_gap_min_km)

    def __str__(self) -> str:
        hr_str  = f"{self.avg_hr:.0f}" if self.avg_hr is not None else " -- "
        pwr_str = (f"{self.avg_power_w:.0f} W"
                   if self.avg_power_w is not None else "  --  ")
        dur_min = int(self.duration_s // 60)
        dur_sec = int(self.duration_s % 60)
        return (
            f"Seg {self.index:2d}  "
            f"grade {self.gradient_pct:+.1f}%  |  "
            f"{dur_min}:{dur_sec:02d}  {self.distance_m/1000:.2f} km  |  "
            f"pace {self.pace_str}  → GAP {self.gap_str}  |  "
            f"HR {hr_str} bpm  |  {pwr_str}"
        )


@dataclass
class TreadmillStats:
    """Aggregate results from a treadmill run with a known gradient schedule."""

    # Segment-level breakdown
    segments: List[GradientSegment]

    # Overall
    total_distance_m: float
    flat_equivalent_distance_m: float   # GAP-weighted equivalent
    avg_gradient_pct: float             # distance-weighted mean

    avg_pace_min_km: float
    avg_gap_min_km: float               # flat-equivalent pace
    gap_factor: float                   # actual_pace / GAP — >1 means harder than flat

    avg_hr: Optional[float]
    avg_power_w: Optional[float]

    # Gradient schedule stored for reference / later plotting
    gradient_breakpoints: List[Tuple[float, float]]

    def summary(self) -> str:
        dur_total = sum(s.duration_s for s in self.segments)
        lines = [
            "── Treadmill Analysis ─────────────────────────────────────────",
            f"  Total distance       : {self.total_distance_m/1000:.2f} km",
            f"  Flat-equiv. distance : {self.flat_equivalent_distance_m/1000:.2f} km"
            f"  (×{self.gap_factor:.2f} harder than flat)",
            f"  Avg gradient         : {self.avg_gradient_pct:+.1f}%",
            "",
            f"  Actual pace          : {_fmt_pace(self.avg_pace_min_km)}",
            f"  Grade-adjusted pace  : {_fmt_pace(self.avg_gap_min_km)}  (flat equiv.)",
        ]
        if self.avg_hr is not None:
            lines.append(f"  Avg HR               : {self.avg_hr:.0f} bpm")
        if self.avg_power_w is not None:
            lines.append(f"  Avg power (Stryd)    : {self.avg_power_w:.0f} W")

        if self.segments:
            lines += ["", "  Gradient segments:"]
            for seg in self.segments:
                lines.append(f"    {seg}")

        lines.append("───────────────────────────────────────────────────────────")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze(
    run,
    hr_stats,
    pace_stats,
    gradient: GradientSchedule,
    config=None,
) -> TreadmillStats:
    """
    Analyse a treadmill run with a manually supplied gradient schedule.

    Parameters
    ----------
    run : RunData
        Parsed run from :func:`RunData.from_dataframe`.
    hr_stats : HRStats
        Result from :func:`runtrackz.hr_analysis.analyze`.
    pace_stats : PaceStats
        Result from :func:`runtrackz.pace_analysis.analyze`.
    gradient : GradientSchedule
        The incline schedule — either a ``[(start_s, pct), ...]`` list or a
        callable ``f(elapsed_s) -> gradient_pct``.

        **List example** (5 min at 1 %, 20 min at 5 %, rest at 1 %)::

            gradient = [(0, 1.0), (300, 5.0), (1500, 1.0)]

        **Callable example**::

            gradient = lambda t: 1.0 if t < 300 else 5.0 if t < 1500 else 1.0

    config : Config, optional
        Reserved for future threshold configuration; not used currently.

    Returns
    -------
    TreadmillStats
    """
    df = run.df.copy()
    elapsed_s = df['elapsed_s'].values.astype(float)

    # ── 1. Resolve gradient percentage for every row ───────────────────────
    grad_arr = _resolve_schedule(gradient, elapsed_s)

    # ── 2. Per-row GAP ─────────────────────────────────────────────────────
    gap_factors  = np.vectorize(_gap_factor)(grad_arr)
    speed_ms     = df['speed_ms'].fillna(0.0).values
    gap_speed_ms = speed_ms * gap_factors

    df['gradient_pct']   = grad_arr
    df['gap_factor']     = gap_factors
    df['gap_speed_ms']   = gap_speed_ms

    # ── 3. Extract breakpoints for segment splitting ───────────────────────
    gradient_breakpoints = _extract_breakpoints(gradient, elapsed_s, grad_arr)

    # ── 4. Segment stats ───────────────────────────────────────────────────
    seg_starts = [b[0] for b in gradient_breakpoints]
    seg_pcts   = [b[1] for b in gradient_breakpoints]
    n_segs     = len(seg_starts)

    segments: List[GradientSegment] = []
    for k in range(n_segs):
        s_start = seg_starts[k]
        s_end   = seg_starts[k + 1] if k + 1 < n_segs else float(elapsed_s[-1]) + 1.0
        mask    = (elapsed_s >= s_start) & (elapsed_s < s_end)
        if not mask.any():
            continue

        seg_df   = df[mask]
        sp_vals  = seg_df['speed_ms'].dropna()
        gap_vals = seg_df['gap_speed_ms'].dropna()

        # Skip if no meaningful movement
        moving_sp = sp_vals[sp_vals > 0.5]
        if len(moving_sp) == 0:
            continue

        dur    = float(s_end - s_start)
        dist_m = float(sp_vals.mean() * dur)

        act_pace = _speed_to_pace(sp_vals.mean())
        gap_pace = _speed_to_pace(gap_vals[gap_vals > 0.5].mean()
                                  if len(gap_vals[gap_vals > 0.5]) > 0
                                  else sp_vals.mean())

        hr_s  = seg_df['heart_rate'].dropna()
        avg_hr = float(hr_s.mean()) if len(hr_s) > 0 else None

        pwr_s = (seg_df['power_w'].dropna()
                 if 'power_w' in seg_df.columns
                 else pd.Series(dtype=float))
        avg_power = float(pwr_s.mean()) if len(pwr_s) > 0 else None

        segments.append(GradientSegment(
            index           = len(segments) + 1,
            start_s         = s_start,
            end_s           = min(s_end, float(elapsed_s[-1])),
            duration_s      = dur,
            gradient_pct    = seg_pcts[k],
            distance_m      = dist_m,
            avg_pace_min_km = act_pace,
            avg_gap_min_km  = gap_pace,
            avg_hr          = avg_hr,
            avg_power_w     = avg_power,
        ))

    # ── 5. Overall stats ───────────────────────────────────────────────────
    total_dist = pace_stats.total_distance_m

    # Distance-weighted average gradient
    if segments:
        w_grad   = sum(s.gradient_pct * s.distance_m for s in segments)
        w_dist   = sum(s.distance_m for s in segments)
        avg_grad = w_grad / w_dist if w_dist > 0 else 0.0
    else:
        avg_grad = float(np.mean(grad_arr))

    # Overall GAP using moving seconds only
    moving_gap = gap_speed_ms[speed_ms > 0.5]
    if len(moving_gap) > 0 and moving_gap.mean() > 0:
        overall_gap_pace = _speed_to_pace(moving_gap.mean())
    else:
        overall_gap_pace = pace_stats.avg_pace_min_km

    flat_equiv = (float(moving_gap.mean()) * pace_stats.total_time_s
                  if len(moving_gap) > 0 else total_dist)

    gap_factor_overall = (pace_stats.avg_pace_min_km / overall_gap_pace
                          if overall_gap_pace > 0 else 1.0)

    hr_all  = df['heart_rate'].dropna()
    avg_hr_overall = float(hr_all.mean()) if len(hr_all) > 0 else None

    pwr_all = (df['power_w'].dropna()
               if 'power_w' in df.columns
               else pd.Series(dtype=float))
    avg_power_overall = float(pwr_all.mean()) if len(pwr_all) > 0 else None

    return TreadmillStats(
        segments                   = segments,
        total_distance_m           = total_dist,
        flat_equivalent_distance_m = flat_equiv,
        avg_gradient_pct           = avg_grad,
        avg_pace_min_km            = pace_stats.avg_pace_min_km,
        avg_gap_min_km             = overall_gap_pace,
        gap_factor                 = gap_factor_overall,
        avg_hr                     = avg_hr_overall,
        avg_power_w                = avg_power_overall,
        gradient_breakpoints       = gradient_breakpoints,
    )


# ---------------------------------------------------------------------------
# Utility / convenience
# ---------------------------------------------------------------------------

def gap_table(gradients: List[float]) -> pd.DataFrame:
    """
    Reference table showing GAP factor and flat-equivalent paces for a list
    of gradients.

    Parameters
    ----------
    gradients : list of float
        Gradient percentages to include.

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> import runtrackz.treadmill_analysis as ta
    >>> print(ta.gap_table([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]))
    """
    rows = []
    for g in gradients:
        gf = _gap_factor(g)
        rows.append({
            'gradient_%': g,
            'gap_factor': round(gf, 3),
            'flat_equiv_4:30/km': _fmt_pace(4.5 / gf),
            'flat_equiv_5:00/km': _fmt_pace(5.0 / gf),
            'flat_equiv_5:30/km': _fmt_pace(5.5 / gf),
            'flat_equiv_6:00/km': _fmt_pace(6.0 / gf),
            'flat_equiv_6:30/km': _fmt_pace(6.5 / gf),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_schedule(
    gradient: GradientSchedule,
    elapsed_s: np.ndarray,
) -> np.ndarray:
    """Map a gradient schedule to a float array aligned with *elapsed_s*."""
    if callable(gradient):
        return np.vectorize(gradient)(elapsed_s).astype(float)

    breakpoints = sorted(gradient, key=lambda t: t[0])
    if not breakpoints:
        return np.zeros_like(elapsed_s, dtype=float)

    starts = np.array([b[0] for b in breakpoints], dtype=float)
    pcts   = np.array([b[1] for b in breakpoints], dtype=float)

    idx = np.searchsorted(starts, elapsed_s, side='right') - 1
    idx = np.clip(idx, 0, len(pcts) - 1)
    return pcts[idx].astype(float)


def _extract_breakpoints(
    gradient: GradientSchedule,
    elapsed_s: np.ndarray,
    grad_arr: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Return a canonical ``[(start_s, pct)]`` list for segment splitting.

    For callables, derive breakpoints by scanning for changes in the resolved
    array (rounded to 4 dp to avoid float noise).
    """
    if callable(gradient):
        bp: List[Tuple[float, float]] = []
        prev: Optional[float] = None
        for t, g in zip(elapsed_s, grad_arr):
            g_r = round(float(g), 4)
            if g_r != prev:
                bp.append((float(t), g_r))
                prev = g_r
        return bp
    return sorted(gradient, key=lambda x: x[0])


def _speed_to_pace(speed_ms: float) -> float:
    """Convert m/s to min/km.  Returns NaN for zero / negative speed."""
    if speed_ms <= 0:
        return float('nan')
    return (1000.0 / 60.0) / speed_ms


def _fmt_pace(pace_min_km: float) -> str:
    """Format a decimal min/km value as ``MM:SS /km``."""
    if pace_min_km != pace_min_km:  # nan check
        return "--:-- /km"
    m = int(pace_min_km)
    s = int((pace_min_km - m) * 60)
    return f"{m}:{s:02d} /km"
