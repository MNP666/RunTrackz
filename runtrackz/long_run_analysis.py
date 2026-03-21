"""
runtrackz.long_run_analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Analysis for long runs: endurance, pacing strategy, and cardiac drift.

A long run is characterised by duration/distance, an aerobic effort level,
and the physiological challenges of fatigue over time — declining economy,
rising HR for the same pace (cardiac drift), and pacing strategy decisions.

Planned metrics
---------------
- Cardiac drift: HR rise for the same pace across the run
- Pacing strategy: even / positive / negative split classification
- Thirds analysis: avg pace and HR in each third of the run
- Fuelling indicator: HR acceleration in the final third (suggests glycogen
  depletion when it diverges sharply from the first two thirds)
- Walk-break detection and total walking time

Current status
--------------
The ``LongRunStats`` dataclass is defined and ``analyze()`` is callable.
Thirds analysis and pacing strategy are computed.  Cardiac drift is derived
from the aerobic decoupling already computed in ``hr_analysis``.  Walk-break
detection is a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from runtrackz.models import RunData
    from runtrackz.hr_analysis import HRStats
    from runtrackz.pace_analysis import PaceStats
    from runtrackz.config import Config


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunThird:
    """Metrics for one third of a long run."""
    label:           str    # 'first', 'second', 'third'
    avg_pace_min_km: float
    avg_hr:          float
    distance_m:      float

    @property
    def pace_str(self) -> str:
        m = int(self.avg_pace_min_km)
        s = int((self.avg_pace_min_km - m) * 60)
        return f"{m}:{s:02d}"


@dataclass
class LongRunStats:
    """
    Results of a long run analysis.

    Attributes
    ----------
    cardiac_drift_pct : float or None
        Percentage rise in HR from first half to second half, adjusted for
        any pace change.  Taken directly from ``hr_stats.aerobic_decoupling_pct``
        when pre-computed HR stats are provided.
    pacing_strategy : str or None
        ``'even'``, ``'negative'`` (sped up), or ``'positive'`` (slowed down),
        based on first-half vs second-half average pace.
    first_half_pace_min_km : float or None
        Average pace over the first half.
    second_half_pace_min_km : float or None
        Average pace over the second half.
    thirds : list[RunThird]
        Pace and HR breakdown by each third of total distance.  Useful for
        spotting fade in the final third.
    walk_time_s : float
        Total time spent at walking pace (< 3 km/h).  0.0 until detection
        is implemented.
    """
    cardiac_drift_pct:        Optional[float]    = None
    pacing_strategy:          Optional[str]      = None
    first_half_pace_min_km:   Optional[float]    = None
    second_half_pace_min_km:  Optional[float]    = None
    thirds:                   list['RunThird']   = None   # type: ignore[assignment]
    walk_time_s:              float              = 0.0

    def __post_init__(self):
        if self.thirds is None:
            self.thirds = []

    def summary(self) -> str:
        lines = ["Long Run Analysis"]
        if self.pacing_strategy:
            lines.append(f"  Pacing strategy    : {self.pacing_strategy}")
        if self.first_half_pace_min_km and self.second_half_pace_min_km:
            def fmt(p):
                m = int(p); s = int((p - m) * 60); return f"{m}:{s:02d}"
            lines.append(f"  First half pace    : {fmt(self.first_half_pace_min_km)} /km")
            lines.append(f"  Second half pace   : {fmt(self.second_half_pace_min_km)} /km")
        if self.cardiac_drift_pct is not None:
            lines.append(f"  Cardiac drift      : {self.cardiac_drift_pct:+.1f} %"
                         + ("  (good)" if abs(self.cardiac_drift_pct) < 5 else ""))
        if self.thirds:
            lines.append("  Thirds breakdown:")
            for t in self.thirds:
                lines.append(f"    {t.label:6s}  pace={t.pace_str} /km  "
                              f"HR={t.avg_hr:.0f} bpm  dist={t.distance_m/1000:.2f} km")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(
    run: 'RunData',
    hr_stats: Optional['HRStats'] = None,
    pace_stats: Optional['PaceStats'] = None,
    config: Optional['Config'] = None,
) -> LongRunStats:
    """
    Analyse a long run.

    Parameters
    ----------
    run : RunData
        Parsed run from :func:`runtrackz.load`.
    hr_stats : HRStats, optional
        Pre-computed HR stats — aerobic decoupling is used directly for
        cardiac drift if available.
    pace_stats : PaceStats, optional
        Pre-computed pace stats.
    config : Config, optional
        Personal config (not used yet — reserved for target pace thresholds).

    Returns
    -------
    LongRunStats
    """
    df = run.df
    stats = LongRunStats()

    # ── Cardiac drift from aerobic decoupling ─────────────────────────────
    if hr_stats is not None:
        stats.cardiac_drift_pct = hr_stats.aerobic_decoupling_pct

    # ── Pacing strategy: first vs second half ────────────────────────────
    valid = df[['pace_min_km', 'heart_rate']].dropna()
    if len(valid) >= 20:
        mid = len(valid) // 2
        first_half  = valid.iloc[:mid]
        second_half = valid.iloc[mid:]

        stats.first_half_pace_min_km  = float(first_half['pace_min_km'].mean())
        stats.second_half_pace_min_km = float(second_half['pace_min_km'].mean())

        diff = stats.second_half_pace_min_km - stats.first_half_pace_min_km
        if abs(diff) < 0.15:          # within ~9 sec/km = even
            stats.pacing_strategy = 'even'
        elif diff < 0:                # faster in second half
            stats.pacing_strategy = 'negative'
        else:                         # slower in second half
            stats.pacing_strategy = 'positive'

    # ── Thirds breakdown ──────────────────────────────────────────────────
    if len(valid) >= 30:
        n = len(valid)
        slices = [
            ('first',  valid.iloc[:n//3]),
            ('second', valid.iloc[n//3: 2*n//3]),
            ('third',  valid.iloc[2*n//3:]),
        ]
        for label, chunk in slices:
            dist_col = df['distance_m'].dropna()
            chunk_dist = dist_col.iloc[len(dist_col) // 3] if len(dist_col) >= 3 else 0
            stats.thirds.append(RunThird(
                label           = label,
                avg_pace_min_km = float(chunk['pace_min_km'].mean()),
                avg_hr          = float(chunk['heart_rate'].mean()),
                distance_m      = float(len(chunk)),   # approx: 1 row ≈ 1 s ≈ ~3 m
            ))

    # TODO: walk-break detection
    # Suggested approach:
    #   speed_col = df['speed_ms'].dropna()
    #   walking_mask = speed_col < (3 / 3.6)   # < 3 km/h
    #   stats.walk_time_s = float(walking_mask.sum())

    return stats
