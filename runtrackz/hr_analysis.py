"""
runtrackz.hr_analysis
~~~~~~~~~~~~~~~~~~~~~
Heart rate zone analysis and statistics.

Supported zone schemes:
- 'max_pct'   : classic % of max HR (Polar/Garmin 5-zone)
- 'hrr'       : Heart Rate Reserve (Karvonen method, requires resting HR)
- 'custom'    : user-supplied zone boundaries

Default 5-zone scheme (% of max HR):
    Zone 1  50–60%   Easy / recovery
    Zone 2  60–70%   Aerobic base / fat burning
    Zone 3  70–80%   Aerobic / tempo
    Zone 4  80–90%   Threshold / hard
    Zone 5  90–100%  Maximum / VO2max
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Default zone definitions
# ---------------------------------------------------------------------------

DEFAULT_ZONES = {
    1: (0.50, 0.60, "Z1 Recovery"),
    2: (0.60, 0.70, "Z2 Aerobic"),
    3: (0.70, 0.80, "Z3 Tempo"),
    4: (0.80, 0.90, "Z4 Threshold"),
    5: (0.90, 1.00, "Z5 Maximum"),
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HRZone:
    zone: int
    name: str
    lower_bpm: float
    upper_bpm: float
    time_s: float = 0.0
    time_pct: float = 0.0
    avg_hr: Optional[float] = None
    sample_count: int = 0

    @property
    def time_min(self) -> float:
        return self.time_s / 60.0

    def __str__(self) -> str:
        return (
            f"{self.name:18s} "
            f"{self.lower_bpm:.0f}–{self.upper_bpm:.0f} bpm  "
            f"{self.time_min:6.1f} min  ({self.time_pct:5.1f}%)"
        )


@dataclass
class HRStats:
    avg_hr: float
    max_hr: float
    min_hr: float
    time_above_z3_pct: float
    zones: Dict[int, HRZone]
    trimp: float               # Training Impulse (Bangsbo method)
    aerobic_decoupling_pct: float  # pace:HR drift (lower = better aerobic fitness)

    def summary(self) -> str:
        lines = [
            "=== Heart Rate Summary ===",
            f"  Avg HR : {self.avg_hr:.0f} bpm",
            f"  Max HR : {self.max_hr:.0f} bpm",
            f"  Min HR : {self.min_hr:.0f} bpm",
            f"  TRIMP  : {self.trimp:.0f}",
            f"  Aerobic decoupling: {self.aerobic_decoupling_pct:+.1f}%",
            f"  Time Z3+ (quality work): {self.time_above_z3_pct:.1f}%",
            "",
            "--- Zone breakdown ---",
        ]
        for z in sorted(self.zones.values(), key=lambda z: z.zone):
            lines.append(f"  {z}")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Zone breakdown as a tidy DataFrame."""
        rows = []
        for z in sorted(self.zones.values(), key=lambda z: z.zone):
            rows.append({
                'zone': z.zone,
                'name': z.name,
                'lower_bpm': z.lower_bpm,
                'upper_bpm': z.upper_bpm,
                'time_min': round(z.time_min, 2),
                'time_pct': round(z.time_pct, 1),
                'avg_hr': round(z.avg_hr, 1) if z.avg_hr else None,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Zone boundary helpers
# ---------------------------------------------------------------------------

def zones_from_max_hr(
    max_hr: int,
    scheme: dict = DEFAULT_ZONES,
) -> Dict[int, Tuple[float, float, str]]:
    """
    Compute absolute bpm zone boundaries from max HR.

    Returns
    -------
    dict  {zone_num: (lower_bpm, upper_bpm, label)}
    """
    result = {}
    for znum, (lo_pct, hi_pct, label) in scheme.items():
        result[znum] = (max_hr * lo_pct, max_hr * hi_pct, label)
    return result


def zones_from_hrr(
    max_hr: int,
    resting_hr: int,
    scheme: dict = DEFAULT_ZONES,
) -> Dict[int, Tuple[float, float, str]]:
    """
    Compute bpm zone boundaries using the Karvonen (HRR) method.

    HRR = max_hr - resting_hr
    Target HR = resting_hr + pct * HRR
    """
    hrr = max_hr - resting_hr
    result = {}
    for znum, (lo_pct, hi_pct, label) in scheme.items():
        lower = resting_hr + lo_pct * hrr
        upper = resting_hr + hi_pct * hrr
        result[znum] = (lower, upper, label)
    return result


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze(
    run: 'RunData',  # noqa: F821
    max_hr: Optional[int] = None,
    resting_hr: Optional[int] = None,
    method: str = 'max_pct',
    custom_zones: Optional[Dict[int, Tuple[float, float, str]]] = None,
    config: Optional['Config'] = None,  # noqa: F821
) -> HRStats:
    """
    Compute HR zone breakdown and statistics for a run.

    Parameters
    ----------
    run : RunData
        Parsed run from :func:`runtrackz.parser.load`.
    max_hr : int, optional
        Maximum heart rate in bpm. If None, uses config.max_hr or the
        observed max from the run (in that priority order).
    resting_hr : int, optional
        Resting heart rate. If None, uses config.resting_hr.
    method : str
        Zone calculation method: 'max_pct' (default) or 'hrr'.
        Overridden by config.zones.method if config is provided.
    custom_zones : dict, optional
        Custom zones as {zone_num: (lower_bpm, upper_bpm, label)}.
        Overrides all other zone settings if provided.
    config : Config, optional
        A loaded :class:`runtrackz.config.Config`. When provided, max_hr,
        resting_hr, method, and zone definitions are taken from it unless
        overridden by the explicit keyword arguments above.

    Returns
    -------
    HRStats
    """
    # Apply config defaults (explicit kwargs take priority over config)
    if config is not None:
        if max_hr is None:
            max_hr = config.max_hr
        if resting_hr is None:
            resting_hr = config.resting_hr
        if custom_zones is None and method == 'max_pct':
            # Use zones exactly as defined in config
            custom_zones = config.zones.as_dict()
            method = config.zones.method  # already encoded in zone boundaries
    df = run.df.copy()
    hr_col = 'heart_rate'

    if hr_col not in df.columns:
        raise ValueError("No heart_rate data found in this run.")

    hr = df[hr_col].dropna()
    if hr.empty:
        raise ValueError("Heart rate column is empty.")

    observed_max = int(hr.max())
    observed_avg = float(hr.mean())
    observed_min = int(hr.min())

    effective_max = max_hr if max_hr else observed_max

    # Build zone boundaries
    if custom_zones:
        zone_bounds = custom_zones
    elif method == 'hrr':
        if resting_hr is None:
            raise ValueError("resting_hr is required for method='hrr'.")
        zone_bounds = zones_from_hrr(effective_max, resting_hr)
    else:
        zone_bounds = zones_from_max_hr(effective_max)

    # Label each sample with a zone
    def _assign_zone(bpm: float) -> int:
        for znum, (lo, hi, _) in zone_bounds.items():
            if lo <= bpm <= hi:
                return znum
        if bpm < min(v[0] for v in zone_bounds.values()):
            return min(zone_bounds.keys())
        return max(zone_bounds.keys())

    df_hr = df[[hr_col]].dropna()
    df_hr = df_hr[df_hr[hr_col] > 0]
    df_hr['zone'] = df_hr[hr_col].apply(_assign_zone)

    total_seconds = len(df_hr)

    # Build HRZone objects
    zones: Dict[int, HRZone] = {}
    for znum, (lo, hi, label) in zone_bounds.items():
        zone_data = df_hr[df_hr['zone'] == znum]
        t = len(zone_data)
        zones[znum] = HRZone(
            zone=znum,
            name=label,
            lower_bpm=lo,
            upper_bpm=hi,
            time_s=float(t),
            time_pct=100.0 * t / total_seconds if total_seconds else 0.0,
            avg_hr=float(zone_data[hr_col].mean()) if not zone_data.empty else None,
            sample_count=t,
        )

    time_above_z3 = sum(
        z.time_s for zn, z in zones.items() if zn >= 3
    )
    time_above_z3_pct = 100.0 * time_above_z3 / total_seconds if total_seconds else 0.0

    # TRIMP (Training Impulse) — Bangsbo simplified
    # TRIMP = Σ (duration_min * HR_ratio * b_factor)
    # b_factor for men ≈ 0.64 * e^(1.92 * HR_ratio), for women ≈ 0.86 * e^(1.67 * HR_ratio)
    # Using gender-neutral simplification: TRIMP = sum(HR_ratio * duration_min)
    import math
    hr_max_eff = float(effective_max)
    hr_rest = float(resting_hr) if resting_hr else observed_min
    trimp = 0.0
    for _, row in df_hr.iterrows():
        hr_ratio = (row[hr_col] - hr_rest) / (hr_max_eff - hr_rest)
        hr_ratio = max(0.0, min(1.0, hr_ratio))
        b = 0.64 * math.exp(1.92 * hr_ratio)
        trimp += (1 / 60) * hr_ratio * b  # 1 second = 1/60 min

    # Aerobic decoupling: compare pace:HR ratio in first vs second half
    decoupling = _calc_aerobic_decoupling(df)

    return HRStats(
        avg_hr=observed_avg,
        max_hr=float(observed_max),
        min_hr=float(observed_min),
        time_above_z3_pct=time_above_z3_pct,
        zones=zones,
        trimp=trimp,
        aerobic_decoupling_pct=decoupling,
    )


def _calc_aerobic_decoupling(df: pd.DataFrame) -> float:
    """
    Aerobic decoupling (Pa:HR ratio drift).

    Compares the pace-to-HR ratio between the first and second halves.
    A value close to 0% means good aerobic fitness; >5% suggests cardiac drift.
    """
    if 'speed_ms' not in df.columns or 'heart_rate' not in df.columns:
        return 0.0

    clean = df[['speed_ms', 'heart_rate']].dropna()
    clean = clean[(clean['speed_ms'] > 0.5) & (clean['heart_rate'] > 60)]
    if len(clean) < 20:
        return 0.0

    mid = len(clean) // 2
    first_half = clean.iloc[:mid]
    second_half = clean.iloc[mid:]

    ratio_first = (first_half['speed_ms'] / first_half['heart_rate']).mean()
    ratio_second = (second_half['speed_ms'] / second_half['heart_rate']).mean()

    if ratio_first == 0:
        return 0.0
    return 100.0 * (ratio_first - ratio_second) / ratio_first
