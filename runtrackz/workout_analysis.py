"""
runtrackz.workout_analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~
Analysis for structured workout sessions: interval runs, speed work, fartlek.

Interval detection algorithm
-----------------------------
1. Apply a rolling median to the speed series (window = ``smooth_window``
   samples) to remove per-second noise without distorting interval edges.
2. Mark every sample as "effort" where smoothed speed exceeds
   ``effort_threshold_ms``.
3. Collect contiguous effort runs as candidate intervals.
4. Drop candidates shorter than ``min_interval_s`` seconds or
   ``min_interval_m`` metres (catches brief accelerations at end of warm-up).
5. The portion of the run before the first interval → warm-up.
   The portion after the last interval → cool-down.
6. Each gap between consecutive intervals → recovery segment.

Configuring the threshold
--------------------------
The default ``effort_threshold_ms = 3.5`` (3.5 m/s ≈ 4:45 /km) works for
most runners doing 1 km intervals at ~4:00–4:30 /km.  Adjust it if your
workout pace is substantially different::

    # Slower runner, 5:30 /km intervals (3.03 m/s)
    stats = analyze(run, hr, pace, effort_threshold_ms=2.9)

    # Express as a target pace instead of m/s
    pace_min_km = 5.0          # 5:00 /km threshold
    threshold   = (1000 / 60) / pace_min_km   # convert to m/s
    stats = analyze(run, hr, pace, effort_threshold_ms=threshold)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from runtrackz.models import RunData
    from runtrackz.hr_analysis import HRStats
    from runtrackz.pace_analysis import PaceStats
    from runtrackz.config import Config


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Interval:
    """A single effort interval detected within a workout."""
    index: int                   # 1-based
    start_s: float               # elapsed seconds at first sample
    end_s: float                 # elapsed seconds at last sample
    duration_s: float
    distance_m: float
    avg_pace_min_km: float
    avg_hr: Optional[float]
    avg_power_w: Optional[float] = None
    peak_hr: Optional[float] = None

    @property
    def duration_str(self) -> str:
        m, s = divmod(int(self.duration_s), 60)
        return f"{m}:{s:02d}"

    @property
    def pace_str(self) -> str:
        m = int(self.avg_pace_min_km)
        s = int((self.avg_pace_min_km - m) * 60)
        return f"{m}:{s:02d}"

    def __str__(self) -> str:
        hr_str   = f"{self.avg_hr:.0f}"   if self.avg_hr   is not None else " -- "
        peak_str = f"{self.peak_hr:.0f}"  if self.peak_hr  is not None else " -- "
        pwr_str  = f"{self.avg_power_w:.0f} W" if self.avg_power_w is not None else "  --  "
        return (
            f"Rep {self.index:2d}  "
            f"{self.duration_str}  {self.distance_m / 1000:.2f} km  "
            f"pace {self.pace_str} /km  "
            f"avg HR {hr_str} bpm  peak {peak_str} bpm  "
            f"{pwr_str}"
        )


@dataclass
class Recovery:
    """A recovery period between two consecutive intervals."""
    index: int                   # 1 = recovery after interval 1
    start_s: float
    end_s: float
    duration_s: float
    avg_hr: Optional[float]
    hr_at_start: Optional[float]  # max HR in first 15 s — how high you peaked
    hr_at_end: Optional[float]    # avg HR in last 15 s — how recovered you are
    hr_drop_bpm: Optional[float]  # hr_at_start − hr_at_end

    @property
    def duration_str(self) -> str:
        m, s = divmod(int(self.duration_s), 60)
        return f"{m}:{s:02d}"

    def __str__(self) -> str:
        hr_str   = f"{self.avg_hr:.0f}"    if self.avg_hr    is not None else " -- "
        drop_str = f"−{self.hr_drop_bpm:.0f}" if self.hr_drop_bpm is not None else " -- "
        end_str  = f"{self.hr_at_end:.0f}" if self.hr_at_end  is not None else " -- "
        return (
            f"Rec {self.index:2d}  "
            f"{self.duration_str}  "
            f"avg HR {hr_str} bpm  "
            f"HR drop {drop_str} bpm  "
            f"ending at {end_str} bpm"
        )


@dataclass
class WorkoutStats:
    """
    Results of a workout (interval) session analysis.

    All interval-level fields are populated after a successful run of
    :func:`analyze` with a valid ``effort_threshold_ms``.
    """

    # Detected structure
    intervals:             List[Interval]  = field(default_factory=list)
    recoveries:            List[Recovery]  = field(default_factory=list)
    num_intervals:         int             = 0

    # Warm-up / cool-down
    warmup_distance_m:     float           = 0.0
    warmup_duration_s:     float           = 0.0
    cooldown_distance_m:   float           = 0.0
    cooldown_duration_s:   float           = 0.0

    # Aggregate interval stats
    avg_interval_pace_min_km: Optional[float] = None
    avg_interval_hr:          Optional[float] = None
    avg_interval_power_w:     Optional[float] = None
    peak_interval_hr:         Optional[float] = None

    # Consistency across intervals
    pace_consistency_cv:   Optional[float] = None  # lower = more even pacing
    hr_consistency_cv:     Optional[float] = None

    # Detection parameters (stored for transparency / reproducibility)
    effort_threshold_ms:   float           = 3.5
    intervals_detected:    bool            = False

    def summary(self) -> str:
        lines = ["── Workout Analysis ────────────────────────────────────────────"]

        if not self.intervals_detected:
            lines.append(
                "  ⚠  No interval detection run.  "
                "Pass effort_threshold_ms= to analyze()."
            )
        else:
            lines.append(f"  Intervals detected : {self.num_intervals}")

        wu_m, wu_s = divmod(int(self.warmup_duration_s), 60)
        cd_m, cd_s = divmod(int(self.cooldown_duration_s), 60)

        if self.warmup_distance_m > 0:
            lines.append(
                f"  Warm-up            : {wu_m}:{wu_s:02d}  "
                f"{self.warmup_distance_m / 1000:.2f} km"
            )
        if self.cooldown_distance_m > 0:
            lines.append(
                f"  Cool-down          : {cd_m}:{cd_s:02d}  "
                f"{self.cooldown_distance_m / 1000:.2f} km"
            )

        if self.avg_interval_pace_min_km is not None:
            m = int(self.avg_interval_pace_min_km)
            s = int((self.avg_interval_pace_min_km - m) * 60)
            lines.append(f"  Avg interval pace  : {m}:{s:02d} /km")
        if self.avg_interval_hr is not None:
            lines.append(f"  Avg interval HR    : {self.avg_interval_hr:.0f} bpm")
        if self.peak_interval_hr is not None:
            lines.append(f"  Peak interval HR   : {self.peak_interval_hr:.0f} bpm")
        if self.avg_interval_power_w is not None:
            lines.append(f"  Avg interval power : {self.avg_interval_power_w:.0f} W")

        if self.pace_consistency_cv is not None:
            pacing_label = (
                "✓ very consistent" if self.pace_consistency_cv < 0.03 else
                "consistent"        if self.pace_consistency_cv < 0.07 else
                "⚠ variable pacing"
            )
            lines.append(
                f"  Pace consistency   : CV={self.pace_consistency_cv:.3f}  {pacing_label}"
            )
        if self.hr_consistency_cv is not None:
            lines.append(f"  HR consistency     : CV={self.hr_consistency_cv:.3f}")

        if self.intervals:
            lines += ["", "  Reps:"]
            for iv in self.intervals:
                lines.append(f"    {iv}")

        if self.recoveries:
            lines += ["", "  Recoveries:"]
            for rec in self.recoveries:
                lines.append(f"    {rec}")

        lines.append("────────────────────────────────────────────────────────────────")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(
    run: 'RunData',
    hr_stats: Optional['HRStats'] = None,
    pace_stats: Optional['PaceStats'] = None,
    config: Optional['Config'] = None,
    *,
    effort_threshold_ms: float = 3.5,
    smooth_window: int = 20,
    min_interval_s: float = 60.0,
    min_interval_m: float = 200.0,
) -> WorkoutStats:
    """
    Analyse a workout (interval) run.

    Parameters
    ----------
    run : RunData
        Parsed run from :func:`runtrackz.load`.
    hr_stats : HRStats, optional
        Pre-computed HR stats.  Passed through for summary data.
    pace_stats : PaceStats, optional
        Pre-computed pace stats.  Passed through for summary data.
    config : Config, optional
        Personal config (reserved for future threshold defaults).
    effort_threshold_ms : float
        Speed threshold in m/s above which a sample is classified as an
        "effort" (default 3.5 m/s ≈ 4:45 /km).  Adjust for your interval
        target pace.
    smooth_window : int
        Rolling-median window in samples applied to speed before thresholding.
        Larger values give smoother detection; default 20 s works well for
        per-second data.
    min_interval_s : float
        Minimum duration for a detected interval (seconds).  Shorter segments
        are discarded as acceleration artefacts.  Default 60 s.
    min_interval_m : float
        Minimum distance for a detected interval (metres).  Default 200 m.

    Returns
    -------
    WorkoutStats
    """
    df      = run.df.copy()
    elapsed = df['elapsed_s'].values.astype(float)
    dist_cum = df['distance_m'].values.astype(float)  # cumulative metres

    # ── 1. Smooth speed and threshold ──────────────────────────────────────
    speed_smooth = (
        df['speed_ms']
        .rolling(smooth_window, center=True, min_periods=1)
        .median()
        .values
    )
    is_effort = speed_smooth > effort_threshold_ms

    # ── 2. Detect contiguous effort segments ───────────────────────────────
    raw_segs: List[tuple[int, int]] = []   # (start_pos, end_pos) inclusive
    in_effort = False
    seg_start = 0

    for i, effort in enumerate(is_effort):
        if effort and not in_effort:
            seg_start = i
            in_effort = True
        elif not effort and in_effort:
            raw_segs.append((seg_start, i - 1))
            in_effort = False
    if in_effort:
        raw_segs.append((seg_start, len(is_effort) - 1))

    # ── 3. Filter short / trivial segments ─────────────────────────────────
    valid_segs: List[tuple[int, int]] = []
    for a, b in raw_segs:
        dur_s  = elapsed[b] - elapsed[a]
        dist_m = dist_cum[b] - dist_cum[a]
        if dur_s >= min_interval_s and dist_m >= min_interval_m:
            valid_segs.append((a, b))

    # ── 4. Build Interval objects ──────────────────────────────────────────
    intervals: List[Interval] = []
    for k, (a, b) in enumerate(valid_segs):
        seg = df.iloc[a : b + 1]
        dur_s  = float(elapsed[b] - elapsed[a])
        dist_m = float(dist_cum[b] - dist_cum[a])

        if dist_m > 0:
            pace = dur_s / (dist_m / 1000.0) / 60.0
        else:
            sp = seg['speed_ms'].dropna()
            pace = (1000 / 60) / sp.mean() if len(sp) > 0 and sp.mean() > 0 else float('nan')

        hr_s     = seg['heart_rate'].dropna()
        avg_hr   = float(hr_s.mean())   if len(hr_s) > 0 else None
        peak_hr  = float(hr_s.max())    if len(hr_s) > 0 else None

        pwr_s    = seg['power_w'].dropna() if 'power_w' in seg.columns else pd.Series(dtype=float)
        avg_pwr  = float(pwr_s.mean()) if len(pwr_s) > 0 else None

        intervals.append(Interval(
            index           = k + 1,
            start_s         = float(elapsed[a]),
            end_s           = float(elapsed[b]),
            duration_s      = dur_s,
            distance_m      = dist_m,
            avg_pace_min_km = pace,
            avg_hr          = avg_hr,
            avg_power_w     = avg_pwr,
            peak_hr         = peak_hr,
        ))

    # ── 5. Build Recovery objects ──────────────────────────────────────────
    recoveries: List[Recovery] = []
    for k in range(len(valid_segs) - 1):
        rec_a = valid_segs[k][1] + 1    # first sample after interval k
        rec_b = valid_segs[k + 1][0] - 1  # last sample before interval k+1

        if rec_a > rec_b:
            continue

        rec_df = df.iloc[rec_a : rec_b + 1]
        dur_s  = float(elapsed[rec_b] - elapsed[rec_a])

        hr_s = rec_df['heart_rate'].dropna()
        avg_hr = float(hr_s.mean()) if len(hr_s) > 0 else None

        window15 = max(1, min(15, len(rec_df) // 4))
        hr_at_start = float(hr_s.iloc[:window15].max())  if len(hr_s) >= window15 else avg_hr
        hr_at_end   = float(hr_s.iloc[-window15:].mean()) if len(hr_s) >= window15 else avg_hr
        hr_drop     = (hr_at_start - hr_at_end) if (hr_at_start and hr_at_end) else None

        recoveries.append(Recovery(
            index        = k + 1,
            start_s      = float(elapsed[rec_a]),
            end_s        = float(elapsed[rec_b]),
            duration_s   = dur_s,
            avg_hr       = avg_hr,
            hr_at_start  = hr_at_start,
            hr_at_end    = hr_at_end,
            hr_drop_bpm  = hr_drop,
        ))

    # ── 6. Warm-up and cool-down ───────────────────────────────────────────
    warmup_dist   = cooldown_dist   = 0.0
    warmup_dur    = cooldown_dur    = 0.0

    if valid_segs:
        wu_end   = valid_segs[0][0]
        cd_start = valid_segs[-1][1] + 1

        warmup_dist  = float(dist_cum[wu_end] - dist_cum[0])
        warmup_dur   = float(elapsed[wu_end]  - elapsed[0])

        if cd_start < len(elapsed):
            cooldown_dist = float(dist_cum[-1]        - dist_cum[cd_start])
            cooldown_dur  = float(elapsed[-1]         - elapsed[cd_start])

    # ── 7. Aggregate interval stats ────────────────────────────────────────
    stats = WorkoutStats(
        intervals           = intervals,
        recoveries          = recoveries,
        num_intervals       = len(intervals),
        warmup_distance_m   = warmup_dist,
        warmup_duration_s   = warmup_dur,
        cooldown_distance_m = cooldown_dist,
        cooldown_duration_s = cooldown_dur,
        effort_threshold_ms = effort_threshold_ms,
        intervals_detected  = True,
    )

    if intervals:
        paces = [iv.avg_pace_min_km for iv in intervals if not np.isnan(iv.avg_pace_min_km)]
        hrs   = [iv.avg_hr          for iv in intervals if iv.avg_hr is not None]
        pwrs  = [iv.avg_power_w     for iv in intervals if iv.avg_power_w is not None]

        if paces:
            stats.avg_interval_pace_min_km = float(np.mean(paces))
            if len(paces) > 1:
                stats.pace_consistency_cv = float(np.std(paces) / np.mean(paces))

        if hrs:
            stats.avg_interval_hr  = float(np.mean(hrs))
            stats.peak_interval_hr = float(max(iv.peak_hr for iv in intervals if iv.peak_hr))
            if len(hrs) > 1:
                stats.hr_consistency_cv = float(np.std(hrs) / np.mean(hrs))

        if pwrs:
            stats.avg_interval_power_w = float(np.mean(pwrs))

    elif pace_stats is not None:
        # Fallback: use full-run stats when no intervals detected
        stats.avg_interval_pace_min_km = pace_stats.avg_pace_min_km
        stats.intervals_detected = False

    if hr_stats is not None and stats.avg_interval_hr is None:
        stats.avg_interval_hr = hr_stats.avg_hr

    return stats
