"""
examples/intervals.py — interval workout analysis
--------------------------------------------------
Run from the project root:

    python examples/intervals.py path/to/your_intervals.fit

The script:
  1. Loads and validates the .fit file (checks it is a run)
  2. Runs HR and pace analysis
  3. Detects effort intervals based on a speed threshold
  4. Prints per-rep stats: duration, distance, pace, avg/peak HR, power
  5. Prints per-recovery stats: duration, HR drop, ending HR
  6. Saves a chart showing pace and HR over time with intervals highlighted

Threshold guidance
------------------
The default threshold of 3.5 m/s (≈ 4:45 /km) works for intervals run at
4:00–4:30 /km.  If your target interval pace is different, override it:

    # Intervals at 5:00 /km
    python examples/intervals.py my_run.fit --threshold 5:00

    # Or pass m/s directly in the script constants below
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')   # headless rendering — change to 'TkAgg' if you want a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import runtrackz

# ── Configuration ─────────────────────────────────────────────────────────────
# Default .fit file location: place your file here, or pass a path on the CLI
_DEFAULT_FIT = Path(__file__).parent / "intervals_example.fit"

# Speed threshold in m/s.  Samples faster than this are classified as an
# "effort" interval.  3.5 m/s = 12.6 km/h ≈ 4:45 /km.
# Tip: convert from a target pace: threshold = (1000 / 60) / pace_min_km
_EFFORT_THRESHOLD_MS: float = 3.5

# Minimum duration and distance for a segment to count as a real interval
_MIN_INTERVAL_S: float = 60.0
_MIN_INTERVAL_M: float = 200.0
# ─────────────────────────────────────────────────────────────────────────────


def _pace_str(pace_min_km: float) -> str:
    m = int(pace_min_km)
    s = int((pace_min_km - m) * 60)
    return f"{m}:{s:02d}"


def _parse_threshold_arg(arg: str) -> float:
    """Accept '3.5' (m/s) or '4:30' / '4.5' (min/km) and return m/s."""
    if ':' in arg:
        parts = arg.split(':')
        pace = int(parts[0]) + int(parts[1]) / 60.0
    else:
        pace = float(arg)
    if pace > 10:
        # Looks like min/km already — convert to m/s
        return (1000 / 60) / pace
    return pace   # already m/s


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def plot_intervals(run, stats, save_path: Path) -> None:
    """
    Three-panel chart:
      - Smoothed pace (min/km) with interval / recovery bands
      - Heart rate with the same bands
      - Running power (W) if Stryd data is present

    Interval bands are shaded red, recovery bands green, warmup/cooldown grey.
    """
    df = run.df.copy()
    t  = df['elapsed_s'] / 60.0   # x-axis in minutes

    pace_smooth = (
        df['pace_min_km']
        .rolling(30, center=True, min_periods=1)
        .median()
    )
    hr_smooth = (
        df['heart_rate']
        .rolling(10, center=True, min_periods=1)
        .mean()
    )
    has_power = 'power_w' in df.columns and df['power_w'].notna().sum() > 100
    pwr_smooth = (
        df['power_w'].rolling(10, center=True, min_periods=1).mean()
        if has_power else None
    )

    n_panels = 3 if has_power else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 3.5 * n_panels),
                              sharex=True, gridspec_kw={'hspace': 0.08})
    ax_pace, ax_hr = axes[0], axes[1]
    ax_pwr = axes[2] if has_power else None

    # ── Band colours ──
    C_INTERVAL = '#e05c5c'
    C_RECOVERY = '#4caf78'
    C_WARMUP   = '#aaaaaa'

    def _shade(ax, a, b, color, alpha=0.18):
        ax.axvspan(a / 60, b / 60, color=color, alpha=alpha, linewidth=0)

    t_max = float(t.max())

    # Warmup
    if stats.intervals:
        for ax in axes:
            _shade(ax, 0, stats.intervals[0].start_s, C_WARMUP)
    # Intervals
    for iv in stats.intervals:
        for ax in axes:
            _shade(ax, iv.start_s, iv.end_s, C_INTERVAL)
    # Recoveries
    for rec in stats.recoveries:
        for ax in axes:
            _shade(ax, rec.start_s, rec.end_s, C_RECOVERY)
    # Cool-down
    if stats.intervals:
        for ax in axes:
            _shade(ax, stats.intervals[-1].end_s, t_max * 60, C_WARMUP)

    # ── Pace panel ──
    ax_pace.plot(t, pace_smooth, color='#2563eb', linewidth=1.6, alpha=0.9)
    ax_pace.invert_yaxis()
    ax_pace.set_ylabel('Pace (min/km)', fontsize=10)

    # Annotate each rep with its pace
    for iv in stats.intervals:
        mid = (iv.start_s + iv.end_s) / 2 / 60
        y   = ax_pace.get_ylim()
        label_y = min(y) + (max(y) - min(y)) * 0.12
        ax_pace.text(
            mid, label_y,
            f"#{iv.index}\n{_pace_str(iv.avg_pace_min_km)}",
            ha='center', va='bottom', fontsize=8,
            color='#9b1c1c', fontweight='bold',
        )

    # Format pace y-axis ticks as MM:SS
    from matplotlib.ticker import FuncFormatter
    def pace_fmt(val, _pos):
        if val <= 0:
            return ''
        m = int(val)
        s = int((val - m) * 60)
        return f"{m}:{s:02d}"
    ax_pace.yaxis.set_major_formatter(FuncFormatter(pace_fmt))
    ax_pace.grid(axis='y', alpha=0.3, linestyle='--')

    # ── HR panel ──
    ax_hr.plot(t, hr_smooth, color='#dc2626', linewidth=1.5, alpha=0.85)
    ax_hr.set_ylabel('Heart Rate (bpm)', fontsize=10)
    ax_hr.grid(axis='y', alpha=0.3, linestyle='--')

    # ── Power panel (optional) ──
    if ax_pwr is not None:
        ax_pwr.plot(t, pwr_smooth, color='#7c3aed', linewidth=1.4, alpha=0.85)
        ax_pwr.set_ylabel('Power (W)', fontsize=10)
        ax_pwr.grid(axis='y', alpha=0.3, linestyle='--')
        ax_pwr.set_xlabel('Time (min)', fontsize=10)
    else:
        ax_hr.set_xlabel('Time (min)', fontsize=10)

    # ── Legend ──
    legend_patches = [
        mpatches.Patch(color=C_INTERVAL, alpha=0.45, label='Interval'),
        mpatches.Patch(color=C_RECOVERY, alpha=0.40, label='Recovery'),
        mpatches.Patch(color=C_WARMUP,   alpha=0.35, label='Warm-up / Cool-down'),
    ]
    ax_pace.legend(handles=legend_patches, loc='upper right', fontsize=8,
                   framealpha=0.85)

    # ── Title ──
    date_str = run.df.index[0].strftime('%Y-%m-%d')
    avg_pace = stats.avg_interval_pace_min_km
    title = (
        f"Interval Workout  —  {date_str}  |  "
        f"{stats.num_intervals} reps  |  "
        f"avg pace {_pace_str(avg_pace) if avg_pace else '?'} /km"
    )
    fig.suptitle(title, fontsize=11, y=0.995)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fit_path: Path, threshold_ms: float = _EFFORT_THRESHOLD_MS) -> None:

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\nLoading: {fit_path.name}")
    cfg = runtrackz.load_config()
    run = runtrackz.load(fit_path)
    print(f"  → {run}")

    if not run.is_run:
        print(f"\nFile is not a run (sport={run.sport!r}).  Exiting.")
        sys.exit(1)

    # ── HR and pace ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    hr   = runtrackz.hr_analysis.analyze(run, config=cfg)
    pace = runtrackz.pace_analysis.analyze(run)
    print(hr.summary())
    print()
    print(pace.summary())

    # ── Interval analysis ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    stats = runtrackz.workout_analysis.analyze(
        run, hr, pace,
        effort_threshold_ms=threshold_ms,
        min_interval_s=_MIN_INTERVAL_S,
        min_interval_m=_MIN_INTERVAL_M,
    )
    print(stats.summary())

    if stats.num_intervals == 0:
        print(
            f"\n  Tip: no intervals detected at threshold "
            f"{threshold_ms:.2f} m/s ({_pace_str((1000/60)/threshold_ms)} /km).\n"
            f"  Lower the threshold if your intervals are slower, or raise it\n"
            f"  if the warm-up pace is close to your interval pace.\n"
            f"  Use --threshold MM:SS on the command line, e.g. --threshold 5:00"
        )
        return

    # ── Chart ─────────────────────────────────────────────────────────────
    print("\nGenerating chart...")
    # Save chart in the same directory as this script so it's always writable,
    # regardless of where the .fit file lives.
    chart_name = fit_path.stem + '.intervals.png'
    chart_path = Path(__file__).parent / chart_name
    plot_intervals(run, stats, chart_path)

    print("\nDone.")


if __name__ == '__main__':
    argv = sys.argv[1:]

    # --threshold MM:SS or --threshold 3.5  (optional)
    threshold = _EFFORT_THRESHOLD_MS
    if '--threshold' in argv:
        idx = argv.index('--threshold')
        if idx + 1 < len(argv):
            threshold = _parse_threshold_arg(argv[idx + 1])
            argv = [a for i, a in enumerate(argv) if i not in (idx, idx + 1)]

    args = [a for a in argv if not a.startswith('--')]
    fit  = Path(args[0]) if args else _DEFAULT_FIT

    if not fit.exists():
        print(f"\nFile not found: {fit}")
        if args:
            print(f"Usage: python examples/intervals.py path/to/run.fit [--threshold 4:45]")
        else:
            print(
                f"\nPlace your intervals .fit file at:\n"
                f"    {_DEFAULT_FIT.relative_to(Path(__file__).parent.parent)}\n"
                f"Or pass the path explicitly:\n"
                f"    python examples/intervals.py path/to/run.fit"
            )
        sys.exit(1)

    main(fit, threshold_ms=threshold)
