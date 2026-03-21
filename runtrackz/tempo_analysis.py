"""
runtrackz.tempo_analysis
~~~~~~~~~~~~~~~~~~~~~~~~
Analysis for tempo runs: sustained efforts at or near lactate threshold.

A tempo run is characterised by a prolonged period of controlled discomfort —
steady pace, elevated HR (typically 80–90 % of max or close to LT HR), and
low pace variability.

Planned metrics
---------------
- Time and distance spent at/near lactate threshold (pace and HR windows)
- Pace consistency (CV, rolling std)
- HR drift over the tempo segment (rising HR for steady pace = fatigue)
- Time-in-zone breakdown for the core tempo segment
- Comparison of first-half vs second-half HR at the same pace

Current status
--------------
The ``TempoStats`` dataclass is defined and ``analyze()`` is callable.
Threshold-segment extraction is not yet implemented; all segment-specific
fields are computed from the full run until then.
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
class TempoStats:
    """
    Results of a tempo run analysis.

    Attributes
    ----------
    avg_pace_min_km : float or None
        Average pace over the full run (or tempo segment once extracted).
    pace_variability_cv : float or None
        Coefficient of variation of pace — lower means more even effort.
    avg_hr : float or None
        Average heart rate over the tempo segment.
    hr_drift_pct : float or None
        Percentage rise in HR from the first quarter to the last quarter at
        roughly the same pace.  Positive = HR crept up (normal fatigue).
    time_at_threshold_s : float or None
        Seconds spent within ±config.lactate_threshold.heart_rate bpm of the
        configured lactate threshold HR.  None when config is not provided.
    pct_at_threshold : float or None
        ``time_at_threshold_s`` as a percentage of total run time.
    hr_pct_of_max : float or None
        Average HR as a fraction of ``config.max_hr``.  None without config.

    Notes
    -----
    Segment extraction (identifying just the "tempo block" within a run that
    also has warm-up/cool-down) is not yet implemented.  All metrics currently
    reflect the full run.
    """
    avg_pace_min_km:     Optional[float] = None
    pace_variability_cv: Optional[float] = None
    avg_hr:              Optional[float] = None
    hr_drift_pct:        Optional[float] = None
    time_at_threshold_s: Optional[float] = None
    pct_at_threshold:    Optional[float] = None
    hr_pct_of_max:       Optional[float] = None

    def summary(self) -> str:
        lines = ["Tempo Run Analysis"]
        if self.avg_pace_min_km is not None:
            m = int(self.avg_pace_min_km)
            s = int((self.avg_pace_min_km - m) * 60)
            lines.append(f"  Avg pace           : {m}:{s:02d} /km")
        if self.pace_variability_cv is not None:
            lines.append(f"  Pace variability   : CV={self.pace_variability_cv:.3f}"
                         + ("  (good)" if self.pace_variability_cv < 0.05 else ""))
        if self.avg_hr is not None:
            lines.append(f"  Avg HR             : {self.avg_hr:.0f} bpm")
        if self.hr_pct_of_max is not None:
            lines.append(f"  HR % of max        : {self.hr_pct_of_max*100:.1f} %")
        if self.hr_drift_pct is not None:
            lines.append(f"  HR drift           : {self.hr_drift_pct:+.1f} %"
                         + ("  (normal)" if self.hr_drift_pct < 5 else "  (notable)"))
        if self.time_at_threshold_s is not None:
            m = int(self.time_at_threshold_s // 60)
            s = int(self.time_at_threshold_s % 60)
            lines.append(f"  Time at threshold  : {m}:{s:02d}"
                         + (f"  ({self.pct_at_threshold:.0f} %)" if self.pct_at_threshold else ""))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(
    run: 'RunData',
    hr_stats: Optional['HRStats'] = None,
    pace_stats: Optional['PaceStats'] = None,
    config: Optional['Config'] = None,
) -> TempoStats:
    """
    Analyse a tempo run.

    Parameters
    ----------
    run : RunData
        Parsed run from :func:`runtrackz.load`.
    hr_stats : HRStats, optional
        Pre-computed HR stats.
    pace_stats : PaceStats, optional
        Pre-computed pace stats.
    config : Config, optional
        Personal config — used for LT HR threshold and max HR.

    Returns
    -------
    TempoStats
    """
    df = run.df
    stats = TempoStats()

    # ── Pace metrics ─────────────────────────────────────────────────────
    pace_col = df['pace_min_km'].dropna()
    if len(pace_col) > 10:
        stats.avg_pace_min_km = float(pace_col.mean())
        stats.pace_variability_cv = float(pace_col.std() / pace_col.mean())

    # ── HR metrics ───────────────────────────────────────────────────────
    hr_col = df['heart_rate'].dropna()
    if len(hr_col) > 10:
        stats.avg_hr = float(hr_col.mean())

        # HR drift: compare first quarter vs last quarter
        q = len(hr_col) // 4
        if q > 0:
            hr_first = float(hr_col.iloc[:q].mean())
            hr_last  = float(hr_col.iloc[-q:].mean())
            if hr_first > 0:
                stats.hr_drift_pct = (hr_last - hr_first) / hr_first * 100

    if config is not None and stats.avg_hr is not None:
        stats.hr_pct_of_max = stats.avg_hr / config.max_hr

        # Time near lactate threshold HR (±8 bpm window)
        lt_hr = getattr(getattr(config, 'lactate_threshold', None), 'heart_rate', None)
        if lt_hr is not None:
            # TODO: restrict to tempo segment once segment extraction is implemented
            window = 8
            mask = hr_col.between(lt_hr - window, lt_hr + window)
            stats.time_at_threshold_s = float(mask.sum())
            total_s = df['elapsed_s'].max()
            if total_s > 0:
                stats.pct_at_threshold = stats.time_at_threshold_s / total_s * 100

    # TODO: extract the actual tempo block (excl. warm-up / cool-down)
    # Suggested approach:
    #   1. Find the longest continuous segment where pace < avg_pace + 1 tolerance
    #      AND HR > LT_HR * 0.85
    #   2. Re-compute all metrics on that segment only

    return stats
