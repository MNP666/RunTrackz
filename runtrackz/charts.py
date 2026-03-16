"""
runtrackz.charts
~~~~~~~~~~~~~~~~
Matplotlib-based visualizations for run data.

All functions return a matplotlib Figure object so they can be used
standalone (plt.show()) or embedded in a dashboard (Dash / Streamlit).

Example
-------
    import runtrackz
    run = runtrackz.load("my_run.fit")
    fig = runtrackz.charts.overview(run)
    fig.savefig("overview.png", dpi=150)
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Zone colours (Z1..Z5)
ZONE_COLORS = {
    1: "#5aadff",  # blue
    2: "#4ccc6e",  # green
    3: "#f0c040",  # yellow
    4: "#f07830",  # orange
    5: "#e03030",  # red
}

_DEFAULT_FIGSIZE = (14, 9)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_pace(val: float, _pos=None) -> str:
    if val <= 0 or val > 30:
        return ""
    mins = int(val)
    secs = int((val - mins) * 60)
    return f"{mins}:{secs:02d}"


def _format_elapsed(val: float, _pos=None) -> str:
    """Format seconds as mm:ss or h:mm:ss for axis labels."""
    val = int(val)
    h = val // 3600
    m = (val % 3600) // 60
    s = val % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _km_labels(ax, dist_col, elapsed_min_col, df):
    """Add vertical km markers to the x-axis (x-axis must be in minutes)."""
    if dist_col not in df.columns or elapsed_min_col not in df.columns:
        return
    for km in range(1, int(df[dist_col].max() / 1000) + 1):
        row = df[df[dist_col] >= km * 1000].first_valid_index()
        if row is not None:
            x_min = df.loc[row, elapsed_min_col]
            ax.axvline(x_min, color='gray', alpha=0.3, linewidth=0.8, linestyle='--')


# ---------------------------------------------------------------------------
# Individual chart functions
# ---------------------------------------------------------------------------

def heart_rate_over_time(
    run: 'RunData',  # noqa: F821
    hr_zones: Optional[Dict] = None,
    config: Optional['Config'] = None,  # noqa: F821
    figsize: tuple = (12, 4),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Line chart of heart rate vs elapsed time, coloured by zone.

    Parameters
    ----------
    run : RunData
    hr_zones : dict, optional
        Zone boundaries as returned by :func:`runtrackz.hr_analysis.zones_from_max_hr`.
        If provided, the background is shaded by zone.
    config : Config, optional
        When provided, uses config.zone_colors and draws a horizontal LT HR line.
    figsize : tuple
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = run.df.reset_index()
    if 'heart_rate' not in df.columns:
        raise ValueError("No heart_rate data available.")

    zone_colors = config.zone_colors if config else ZONE_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    x = df['elapsed_s'] / 60  # minutes
    ax.plot(x, df['heart_rate'], color='#e03030', linewidth=1.2, alpha=0.9, label='HR')

    # Shade zone bands
    if hr_zones:
        for znum, (lo, hi, label) in hr_zones.items():
            ax.axhspan(lo, hi, alpha=0.28, color=zone_colors.get(znum, '#aaaaaa'))
            ax.text(x.max() * 1.005, (lo + hi) / 2, f"Z{znum}",
                    va='center', ha='left', fontsize=7, color=zone_colors.get(znum, '#aaaaaa'))

    # Lactate threshold HR line
    if config and config.lactate_threshold.heart_rate:
        lt_hr = config.lactate_threshold.heart_rate
        ax.axhline(lt_hr, color='#cc0000', linestyle=':', linewidth=1.2,
                   label=f"LT HR ({lt_hr} bpm)")
        ax.legend(fontsize=8, loc='upper left')

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_title(title or f"Heart Rate — {run.df.index[0].strftime('%Y-%m-%d')}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def pace_over_distance(
    run: 'RunData',  # noqa: F821
    splits=None,
    config: Optional['Config'] = None,  # noqa: F821
    smooth_window: int = 15,
    figsize: tuple = (12, 4),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Pace chart (min/km) vs distance with per-km split markers.

    Parameters
    ----------
    run : RunData
    splits : list of Split, optional
        If provided, split average paces are shown as a bar overlay.
    config : Config, optional
        When provided, draws a horizontal LT pace reference line.
    smooth_window : int
        Rolling average window in seconds.
    figsize : tuple
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = run.df.reset_index()
    if 'pace_min_km' not in df.columns or 'distance_m' not in df.columns:
        raise ValueError("No pace/distance data available.")

    fig, ax = plt.subplots(figsize=figsize)

    # Smooth pace
    pace = df['pace_min_km'].copy()
    pace = pace[(pace > 2) & (pace < 20)]  # filter outliers
    pace_smooth = pace.rolling(smooth_window, min_periods=1, center=True).mean()

    dist_km = df['distance_m'] / 1000

    ax.plot(dist_km, pace_smooth, color='#1a7abf', linewidth=1.5, label='Pace (smoothed)')
    ax.fill_between(dist_km, pace_smooth, pace_smooth.max() * 1.02,
                    alpha=0.1, color='#1a7abf')

    # Split bar overlay
    if splits:
        for s in splits:
            mid_km = (s.elapsed_start_s + s.elapsed_end_s) / 2
            # find distance at midpoint
            mid_dist = s.split_num - 0.5
            bar_width = s.actual_distance_m / 1000 * 0.7
            bar = ax.bar(mid_dist, s.pace_min_km, width=bar_width,
                         alpha=0.25, color='#f07830', zorder=2)

    # Lactate threshold pace line
    if config and config.lactate_threshold.pace_min_km:
        lt_pace = config.lactate_threshold.pace_min_km
        ax.axhline(lt_pace, color='#cc6600', linestyle=':', linewidth=1.2,
                   label=f"LT pace ({_format_pace(lt_pace)})")
        ax.legend(fontsize=8, loc='upper right')

    # Y-axis: invert so faster = higher on chart
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_pace))
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Pace (min/km)")
    ax.set_title(title or f"Pace — {run.df.index[0].strftime('%Y-%m-%d')}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def hr_zone_bar(
    hr_stats: 'HRStats',  # noqa: F821
    config: Optional['Config'] = None,  # noqa: F821
    figsize: tuple = (8, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of time spent in each HR zone.

    Parameters
    ----------
    hr_stats : HRStats
        Result from :func:`runtrackz.hr_analysis.analyze`.
    config : Config, optional
        When provided, uses config.zone_colors.
    figsize : tuple
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    zone_colors = config.zone_colors if config else ZONE_COLORS
    zones = sorted(hr_stats.zones.values(), key=lambda z: z.zone)
    names = [z.name for z in zones]
    times = [z.time_min for z in zones]
    pcts = [z.time_pct for z in zones]
    colors = [zone_colors.get(z.zone, '#aaaaaa') for z in zones]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(names, times, color=colors, edgecolor='white', linewidth=0.5)

    # Annotate with percentage
    for bar, pct in zip(bars, pcts):
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va='center', ha='left', fontsize=9)

    ax.set_xlabel("Time (min)")
    ax.set_title(title or "Time in Heart Rate Zones")
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, max(times) * 1.2 if times else 10)
    fig.tight_layout()
    return fig


def splits_bar(
    pace_stats: 'PaceStats',  # noqa: F821
    config: Optional['Config'] = None,  # noqa: F821
    figsize: tuple = (12, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of per-km split paces with HR overlay.

    Parameters
    ----------
    pace_stats : PaceStats
        Result from :func:`runtrackz.pace_analysis.analyze`.
    config : Config, optional
        When provided, draws a horizontal LT pace reference line.
    figsize : tuple
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    splits = pace_stats.splits
    if not splits:
        raise ValueError("No splits data available.")

    fig, ax1 = plt.subplots(figsize=figsize)

    nums = [s.split_num for s in splits]
    paces = [s.pace_min_km for s in splits]
    avg_pace = pace_stats.avg_pace_min_km

    bar_colors = ['#4ccc6e' if p <= avg_pace else '#e03030' for p in paces]
    bars = ax1.bar(nums, paces, color=bar_colors, alpha=0.8, width=0.6)

    # Reference line for average pace
    ax1.axhline(avg_pace, color='gray', linestyle='--', linewidth=1,
                label=f"Avg pace ({_format_pace(avg_pace)})")

    # Lactate threshold pace line
    if config and config.lactate_threshold.pace_min_km:
        lt_pace = config.lactate_threshold.pace_min_km
        ax1.axhline(lt_pace, color='#cc6600', linestyle=':', linewidth=1.2,
                    label=f"LT pace ({_format_pace(lt_pace)})")

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_format_pace))
    ax1.invert_yaxis()
    ax1.set_xlabel("Split (km)")
    ax1.set_ylabel("Pace (min/km)")
    ax1.set_xticks(nums)

    # HR overlay
    hr_vals = [s.avg_hr for s in splits if s.avg_hr]
    if len(hr_vals) == len(splits):
        ax2 = ax1.twinx()
        ax2.plot(nums, [s.avg_hr for s in splits], 'o-',
                 color='#e03030', linewidth=1.5, markersize=4, label='Avg HR')
        ax2.set_ylabel("Heart Rate (bpm)", color='#e03030')
        ax2.tick_params(axis='y', labelcolor='#e03030')
        ax2.legend(loc='upper right', fontsize=8)

    ax1.set_title(title or f"Per-km Splits — {len(splits)} km run")
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, axis='y', alpha=0.3)

    # Label bars with pace
    for bar, pace in zip(bars, paces):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 _format_pace(pace),
                 ha='center', va='bottom', fontsize=7, rotation=0)

    fig.tight_layout()
    return fig


def overview(
    run: 'RunData',  # noqa: F821
    hr_stats: Optional['HRStats'] = None,  # noqa: F821
    pace_stats: Optional['PaceStats'] = None,  # noqa: F821
    config: Optional['Config'] = None,  # noqa: F821
    max_hr: Optional[int] = None,
    figsize: tuple = _DEFAULT_FIGSIZE,
) -> plt.Figure:
    """
    4-panel overview figure: HR, pace, elevation, and HR zone pie.

    Parameters
    ----------
    run : RunData
    hr_stats : HRStats, optional
        Pre-computed HR stats. Will be computed automatically if None.
    pace_stats : PaceStats, optional
        Pre-computed pace stats. Will be computed automatically if None.
    config : Config, optional
        When provided, drives zone colors, LT lines, and HR computation.
        Takes precedence over max_hr.
    max_hr : int, optional
        Used for automatic HR stats computation (ignored if config given).
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    from runtrackz import hr_analysis, pace_analysis  # avoid circular imports

    if hr_stats is None:
        hr_stats = hr_analysis.analyze(run, config=config, max_hr=max_hr)
    if pace_stats is None:
        pace_stats = pace_analysis.analyze(run)

    zone_colors = config.zone_colors if config else ZONE_COLORS

    df = run.df.reset_index()
    df['elapsed_min'] = df['elapsed_s'] / 60
    x = df['elapsed_min']

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    date_str = run.df.index[0].strftime('%Y-%m-%d %H:%M')
    dist_km = pace_stats.total_distance_km
    dur_str = pace_stats.total_time_str
    fig.suptitle(
        f"Run Overview  ·  {date_str}  ·  {dist_km:.2f} km  ·  {dur_str}",
        fontsize=13, fontweight='bold', y=0.98
    )

    # ── Panel 1: Heart rate ──────────────────────────────────────────────
    ax_hr = fig.add_subplot(gs[0, :2])
    if 'heart_rate' in df.columns:
        hr_zones = {zn: (z.lower_bpm, z.upper_bpm, z.name)
                    for zn, z in hr_stats.zones.items()}
        for znum, (lo, hi, _) in hr_zones.items():
            ax_hr.axhspan(lo, hi, alpha=0.28, color=zone_colors.get(znum, '#aaa'))
        ax_hr.plot(x, df['heart_rate'], color='k', linewidth=1.1, alpha=0.9)
        # Lactate threshold HR line
        if config and config.lactate_threshold.heart_rate:
            lt_hr = config.lactate_threshold.heart_rate
            ax_hr.axhline(lt_hr, color='#cc0000', linestyle=':', linewidth=1.2,
                          label=f"LT HR ({lt_hr} bpm)")
            ax_hr.legend(fontsize=7, loc='upper left')
        ax_hr.set_ylabel("HR (bpm)")
        ax_hr.set_title(f"Heart Rate  ·  avg {hr_stats.avg_hr:.0f}  max {hr_stats.max_hr:.0f} bpm")
        ax_hr.grid(True, alpha=0.3)
        # km markers (x-axis is in minutes)
        _km_labels(ax_hr, 'distance_m', 'elapsed_min', df)
        # Scale x-axis ticks to run duration
        duration_min = x.max()
        tick_interval = 5 if duration_min <= 35 else 10 if duration_min <= 90 else 15
        ax_hr.xaxis.set_major_locator(mticker.MultipleLocator(tick_interval))
        ax_hr.set_xlim(0, duration_min)
        ax_hr.set_xlabel("Time (min)")

    # ── Panel 2: HR Zone pie ─────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[0, 2])
    zone_times = [z.time_min for z in sorted(hr_stats.zones.values(), key=lambda z: z.zone)]
    pie_zone_colors = [zone_colors.get(z.zone, '#aaa') for z in sorted(hr_stats.zones.values(), key=lambda z: z.zone)]
    zone_labels = [f"Z{z.zone}" for z in sorted(hr_stats.zones.values(), key=lambda z: z.zone)]
    wedges, _, autotexts = ax_pie.pie(
        zone_times, labels=None, colors=pie_zone_colors,
        autopct=lambda p: f'{p:.0f}%' if p > 3 else '',
        startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 0.8},
    )
    ax_pie.legend(wedges, zone_labels, loc='lower center',
                  ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    ax_pie.set_title("HR Zones")

    # ── Panel 3: Pace ────────────────────────────────────────────────────
    ax_pace = fig.add_subplot(gs[1, :2])
    if 'pace_min_km' in df.columns and 'distance_m' in df.columns:
        pace = df['pace_min_km'].copy()
        valid_mask = (pace > 2) & (pace < 15)
        pace_valid = pace[valid_mask]
        dist_km_col = df['distance_m'][valid_mask] / 1000
        pace_smooth = pace_valid.rolling(15, min_periods=1, center=True).mean()
        ax_pace.plot(dist_km_col, pace_smooth, color='#1a7abf', linewidth=1.3)
        ax_pace.fill_between(dist_km_col, pace_smooth, pace_smooth.max() * 1.02,
                             alpha=0.1, color='#1a7abf')
        # Lactate threshold pace line
        if config and config.lactate_threshold.pace_min_km:
            lt_pace = config.lactate_threshold.pace_min_km
            ax_pace.axhline(lt_pace, color='#cc6600', linestyle=':', linewidth=1.2,
                            label=f"LT pace ({_format_pace(lt_pace)})")
            ax_pace.legend(fontsize=7, loc='upper right')
        ax_pace.invert_yaxis()
        ax_pace.yaxis.set_major_formatter(mticker.FuncFormatter(_format_pace))
        ax_pace.set_xlabel("Distance (km)")
        ax_pace.set_ylabel("Pace (min/km)")
        ax_pace.set_title(f"Pace  ·  avg {pace_stats.avg_pace_str}")
        ax_pace.grid(True, alpha=0.3)

    # ── Panel 4: Elevation ───────────────────────────────────────────────
    ax_elev = fig.add_subplot(gs[1, 2])
    if 'altitude_m' in df.columns:
        alt = df['altitude_m'].dropna()
        dist_km_elev = df['distance_m'].dropna() / 1000
        # align indices
        common_idx = alt.index.intersection(dist_km_elev.index)
        ax_elev.fill_between(dist_km_elev.loc[common_idx], alt.loc[common_idx],
                             alt.loc[common_idx].min(), alpha=0.5, color='#7a5230')
        ax_elev.plot(dist_km_elev.loc[common_idx], alt.loc[common_idx],
                     color='#7a5230', linewidth=1)
        ax_elev.set_xlabel("Distance (km)")
        ax_elev.set_ylabel("Altitude (m)")
        asc = pace_stats.total_ascent_m
        ax_elev.set_title(f"Elevation  ·  ↑{asc:.0f}m" if asc else "Elevation")
        ax_elev.grid(True, alpha=0.3)

    return fig
