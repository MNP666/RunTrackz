"""
runtrackz.pace_analysis
~~~~~~~~~~~~~~~~~~~~~~~
Pace, speed, splits, and running power analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Split:
    split_num: int
    distance_m: float        # target split distance (e.g. 1000 m)
    actual_distance_m: float
    elapsed_start_s: float
    elapsed_end_s: float
    duration_s: float
    pace_min_km: float
    avg_hr: Optional[float]
    avg_cadence: Optional[float]
    avg_power: Optional[float]
    elevation_gain_m: Optional[float]

    @property
    def pace_str(self) -> str:
        mins = int(self.pace_min_km)
        secs = int((self.pace_min_km - mins) * 60)
        return f"{mins}:{secs:02d} /km"

    def __str__(self) -> str:
        hr_str = f"{self.avg_hr:.0f}" if self.avg_hr else " -- "
        return (
            f"Split {self.split_num:3d} | "
            f"{self.actual_distance_m/1000:.2f} km | "
            f"{self.pace_str:10s} | "
            f"HR: {hr_str:5s} bpm"
        )


@dataclass
class PaceStats:
    total_distance_m: float
    total_time_s: float
    avg_pace_min_km: float
    best_pace_min_km: float
    avg_speed_kmh: float
    max_speed_kmh: float
    avg_cadence: Optional[float]
    avg_power: Optional[float]
    total_ascent_m: Optional[float]
    total_descent_m: Optional[float]
    splits: List[Split]

    @property
    def total_distance_km(self) -> float:
        return self.total_distance_m / 1000.0

    @property
    def total_time_str(self) -> str:
        h = int(self.total_time_s // 3600)
        m = int((self.total_time_s % 3600) // 60)
        s = int(self.total_time_s % 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    @property
    def avg_pace_str(self) -> str:
        mins = int(self.avg_pace_min_km)
        secs = int((self.avg_pace_min_km - mins) * 60)
        return f"{mins}:{secs:02d} /km"

    def summary(self) -> str:
        lines = [
            "=== Pace & Speed Summary ===",
            f"  Distance : {self.total_distance_km:.2f} km",
            f"  Time     : {self.total_time_str}",
            f"  Avg pace : {self.avg_pace_str}",
            f"  Avg speed: {self.avg_speed_kmh:.1f} km/h",
            f"  Max speed: {self.max_speed_kmh:.1f} km/h",
        ]
        if self.avg_cadence:
            lines.append(f"  Avg cadence: {self.avg_cadence:.0f} spm")
        if self.avg_power:
            lines.append(f"  Avg power  : {self.avg_power:.0f} W")
        if self.total_ascent_m is not None:
            lines.append(f"  Ascent     : {self.total_ascent_m:.0f} m")
            lines.append(f"  Descent    : {self.total_descent_m:.0f} m")
        if self.splits:
            lines += ["", "--- Splits ---"]
            for s in self.splits:
                lines.append(f"  {s}")
        return "\n".join(lines)

    def splits_dataframe(self) -> pd.DataFrame:
        rows = []
        for s in self.splits:
            rows.append({
                'split': s.split_num,
                'distance_km': round(s.actual_distance_m / 1000, 3),
                'duration_s': round(s.duration_s, 1),
                'pace_min_km': round(s.pace_min_km, 3),
                'pace_str': s.pace_str,
                'avg_hr': round(s.avg_hr, 1) if s.avg_hr else None,
                'avg_cadence': round(s.avg_cadence, 1) if s.avg_cadence else None,
                'avg_power': round(s.avg_power, 1) if s.avg_power else None,
                'elevation_gain_m': round(s.elevation_gain_m, 1) if s.elevation_gain_m else None,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(series: pd.Series, window: int = 5) -> pd.Series:
    """Rolling mean smoothing, forward-fill NaNs."""
    return series.rolling(window, min_periods=1, center=True).mean()


def _calc_elevation_gain(altitude: pd.Series, threshold: float = 0.5) -> tuple[float, float]:
    """
    Calculate cumulative ascent and descent from an altitude series.
    Only counts changes above `threshold` metres to reduce GPS noise.
    """
    diff = altitude.diff().dropna()
    gain = diff[diff > threshold].sum()
    loss = abs(diff[diff < -threshold].sum())
    return float(gain), float(loss)


def _pace_from_speed(speed_ms: float) -> float:
    """Convert m/s to min/km."""
    if speed_ms <= 0:
        return float('inf')
    return 1000.0 / speed_ms / 60.0


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze(
    run: 'RunData',  # noqa: F821
    split_distance_m: float = 1000.0,
    smooth_speed: bool = True,
    smooth_window: int = 5,
) -> PaceStats:
    """
    Compute pace statistics and per-km splits for a run.

    Parameters
    ----------
    run : RunData
        Parsed run from :func:`RunData.from_dataframe`.
    split_distance_m : float
        Distance per split in metres (default 1000 = per km).
    smooth_speed : bool
        Apply rolling average smoothing to speed before computing pace.
    smooth_window : int
        Smoothing window size in seconds.

    Returns
    -------
    PaceStats
    """
    df = run.df.copy()

    if 'speed_ms' not in df.columns:
        raise ValueError("No speed data found in this run.")

    # Smooth speed if requested
    speed = df['speed_ms'].fillna(0)
    if smooth_speed:
        speed = _smooth(speed, smooth_window)
    df['speed_ms_smooth'] = speed

    # Filter to moving periods (speed > 0.5 m/s ~ 1.8 km/h)
    moving = df[df['speed_ms_smooth'] > 0.5]

    total_distance_m = df['distance_m'].max() if 'distance_m' in df.columns else (
        moving['speed_ms_smooth'].sum()  # approximate: 1 sample ≈ 1 s
    )
    total_time_s = float(df['elapsed_s'].max()) if 'elapsed_s' in df.columns else len(df)

    # Average pace
    avg_speed_ms = float(moving['speed_ms_smooth'].mean()) if not moving.empty else 0.0
    avg_pace = _pace_from_speed(avg_speed_ms)
    best_pace = _pace_from_speed(float(speed[speed > 0.5].max())) if (speed > 0.5).any() else avg_pace
    avg_speed_kmh = avg_speed_ms * 3.6
    max_speed_kmh = float(speed.max()) * 3.6

    # Cadence (steps per min)
    avg_cadence = None
    if 'steps_per_min' in df.columns:
        cad = df['steps_per_min'].dropna()
        cad = cad[cad > 100]
        avg_cadence = float(cad.mean()) if not cad.empty else None

    # Power
    avg_power = None
    if 'power_w' in df.columns:
        pwr = df['power_w'].dropna()
        pwr = pwr[pwr > 0]
        avg_power = float(pwr.mean()) if not pwr.empty else None

    # Elevation
    total_ascent = total_descent = None
    if 'altitude_m' in df.columns:
        alt = df['altitude_m'].dropna()
        if not alt.empty:
            total_ascent, total_descent = _calc_elevation_gain(alt)

    # Splits
    splits = _compute_splits(df, split_distance_m)

    return PaceStats(
        total_distance_m=total_distance_m,
        total_time_s=total_time_s,
        avg_pace_min_km=avg_pace,
        best_pace_min_km=best_pace,
        avg_speed_kmh=avg_speed_kmh,
        max_speed_kmh=max_speed_kmh,
        avg_cadence=avg_cadence,
        avg_power=avg_power,
        total_ascent_m=total_ascent,
        total_descent_m=total_descent,
        splits=splits,
    )


def _compute_splits(df: pd.DataFrame, split_dist_m: float) -> List[Split]:
    """Generate even distance splits from the run DataFrame."""
    if 'distance_m' not in df.columns:
        return []

    dist = df['distance_m'].ffill()
    total_dist = dist.max()

    splits = []
    split_num = 1
    split_start_dist = 0.0

    while split_start_dist < total_dist - 1:
        split_end_dist = split_start_dist + split_dist_m
        mask = (dist >= split_start_dist) & (dist < split_end_dist)
        chunk = df[mask]

        if chunk.empty:
            split_start_dist = split_end_dist
            continue

        actual_dist = float(dist[mask].max() - dist[mask].min())
        elapsed_start = float(chunk['elapsed_s'].iloc[0]) if 'elapsed_s' in chunk.columns else 0.0
        elapsed_end = float(chunk['elapsed_s'].iloc[-1]) if 'elapsed_s' in chunk.columns else 0.0
        duration = elapsed_end - elapsed_start

        # Pace: use actual distance and time
        speed_col = 'speed_ms_smooth' if 'speed_ms_smooth' in chunk.columns else 'speed_ms'
        moving_chunk = chunk[chunk[speed_col] > 0.5] if speed_col in chunk.columns else chunk
        avg_spd = float(moving_chunk[speed_col].mean()) if not moving_chunk.empty and speed_col in moving_chunk else 0.0
        pace = _pace_from_speed(avg_spd)

        # HR
        avg_hr = None
        if 'heart_rate' in chunk.columns:
            hr_vals = chunk['heart_rate'].dropna()
            hr_vals = hr_vals[hr_vals > 40]
            avg_hr = float(hr_vals.mean()) if not hr_vals.empty else None

        # Cadence
        avg_cad = None
        if 'steps_per_min' in chunk.columns:
            cad_vals = chunk['steps_per_min'].dropna()
            cad_vals = cad_vals[cad_vals > 100]
            avg_cad = float(cad_vals.mean()) if not cad_vals.empty else None

        # Power
        avg_pwr = None
        if 'power_w' in chunk.columns:
            pwr_vals = chunk['power_w'].dropna()
            pwr_vals = pwr_vals[pwr_vals > 0]
            avg_pwr = float(pwr_vals.mean()) if not pwr_vals.empty else None

        # Elevation gain
        elev_gain = None
        if 'altitude_m' in chunk.columns:
            alt = chunk['altitude_m'].dropna()
            if not alt.empty:
                elev_gain, _ = _calc_elevation_gain(alt)

        splits.append(Split(
            split_num=split_num,
            distance_m=split_dist_m,
            actual_distance_m=actual_dist,
            elapsed_start_s=elapsed_start,
            elapsed_end_s=elapsed_end,
            duration_s=duration,
            pace_min_km=pace,
            avg_hr=avg_hr,
            avg_cadence=avg_cad,
            avg_power=avg_pwr,
            elevation_gain_m=elev_gain,
        ))

        split_num += 1
        split_start_dist = split_end_dist

    return splits


# ---------------------------------------------------------------------------
# Power-based metrics
# ---------------------------------------------------------------------------

def running_efficiency(run: 'RunData') -> Optional[float]:  # noqa: F821
    """
    Running Efficiency (RE) = speed (m/s) / power (W/kg).

    A higher value means you produce more speed per watt per kg of body weight.
    Returns None if power data is unavailable.
    """
    df = run.df
    if 'power_w' not in df.columns or 'speed_ms' not in df.columns:
        return None
    clean = df[['power_w', 'speed_ms']].dropna()
    clean = clean[(clean['power_w'] > 0) & (clean['speed_ms'] > 0.5)]
    if clean.empty:
        return None
    return float((clean['speed_ms'] / clean['power_w']).mean())
