"""
runtrackz.comparison
~~~~~~~~~~~~~~~~~~~~
Cross-session and cross-interval comparison utilities.

The core function :func:`align_intervals` takes a list of :class:`RunData`
objects (typically individual interval slices extracted with
:meth:`~RunData.slice_elapsed` or :meth:`~RunData.slice_km`) and resamples
them onto a common x-axis, returning a tidy long-format DataFrame ready
for plotting.

Typical workflow
----------------
Extract intervals from a workout, align them, and plot::

    import runtrackz
    from runtrackz import comparison

    run   = runtrackz.load("5x1km.fit")
    hr    = runtrackz.hr_analysis.analyze(run, config=cfg)
    pace  = runtrackz.pace_analysis.analyze(run)
    stats = runtrackz.workout_analysis.analyze(run, hr, pace)

    # Slice each detected interval into its own RunData
    slices = [run.slice_elapsed(iv.start_s, iv.end_s) for iv in stats.intervals]

    # Align to a common distance axis and compare pace + HR
    df = comparison.align_intervals(
        slices,
        normalize_by="distance_m",
        metrics=["pace_min_km", "heart_rate", "power_w"],
        labels=[f"Rep {iv.index}" for iv in stats.intervals],
    )

    # df is a tidy long DataFrame:
    #   x  |  metric        |  value  |  label  |  date
    #   0   |  pace_min_km  |  4.02   |  Rep 1  |  2026-03-03
    #   10  |  pace_min_km  |  4.05   |  Rep 1  |  2026-03-03
    #   ...

    # Plot in matplotlib
    import matplotlib.pyplot as plt
    for (label, metric), grp in df.groupby(["label", "metric"]):
        plt.plot(grp["x"], grp["value"], label=f"{label} {metric}")

Cross-session comparison
------------------------
You can also compare the same interval type across different training days::

    run1 = runtrackz.load("session_1.fit")
    run2 = runtrackz.load("session_2.fit")

    stats1 = runtrackz.workout_analysis.analyze(run1, ...)
    stats2 = runtrackz.workout_analysis.analyze(run2, ...)

    # Pick rep 3 from each session
    slices = [
        run1.slice_elapsed(stats1.intervals[2].start_s, stats1.intervals[2].end_s),
        run2.slice_elapsed(stats2.intervals[2].start_s, stats2.intervals[2].end_s),
    ]
    df = comparison.align_intervals(slices, labels=["Session A", "Session B"])
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from runtrackz.parser import RunData

#: Default metrics to include when the caller does not specify.
#: Columns absent from a given RunData are silently skipped.
DEFAULT_METRICS: list[str] = ["pace_min_km", "heart_rate", "power_w", "speed_ms"]


def align_intervals(
    runs: List["RunData"],
    normalize_by: str = "distance_m",
    metrics: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Resample multiple :class:`~runtrackz.RunData` objects onto a shared
    x-axis and return a tidy long-format DataFrame.

    Parameters
    ----------
    runs : list of RunData
        The runs (or slices) to compare.  Pass interval slices from
        :meth:`~RunData.slice_elapsed` / :meth:`~RunData.slice_km` for
        within-session rep comparison, or full runs from different days for
        cross-session comparison.
    normalize_by : str
        How to construct the common x-axis:

        ``"distance_m"`` *(default)*
            Metres from the start of each slice.  All intervals are
            truncated to the length of the shortest one.
        ``"elapsed_s"``
            Seconds from the start of each slice.  All intervals are
            truncated to the duration of the shortest one.
        ``"distance"``
            Normalised fraction 0 → 1 of each interval's total distance.
        ``"duration"``
            Normalised fraction 0 → 1 of each interval's total duration.

    metrics : list of str, optional
        Column names from ``run.df`` to include.  Defaults to
        ``["pace_min_km", "heart_rate", "power_w", "speed_ms"]``.
        Columns not present in a particular run are silently skipped.
    labels : list of str, optional
        Display labels for each run.  Defaults to ``"Rep 1"``, ``"Rep 2"``, …
    n_points : int
        Number of evenly-spaced points on the shared x-axis.  Default 200.

    Returns
    -------
    pd.DataFrame
        Tidy long-format DataFrame with columns:

        ``x``
            Position on the shared x-axis (unit depends on *normalize_by*).
        ``metric``
            Column name (e.g. ``"pace_min_km"``).
        ``value``
            Interpolated metric value at this x position.
        ``label``
            The label for this run.
        ``date``
            ISO date string of the first sample in the run.
        ``source_file``
            Basename of the original ``.fit`` file.

    Examples
    --------
    Compare 5 intervals from the same workout::

        slices = [run.slice_elapsed(iv.start_s, iv.end_s)
                  for iv in stats.intervals]

        df = comparison.align_intervals(
            slices,
            normalize_by="distance_m",
            metrics=["pace_min_km", "heart_rate"],
        )
    """
    if not runs:
        return pd.DataFrame(columns=["x", "metric", "value", "label", "date", "source_file"])

    if metrics is None:
        # Include only metrics that exist in at least one run
        metrics = [m for m in DEFAULT_METRICS if any(m in r.df.columns for r in runs)]

    if labels is None:
        labels = [f"Rep {i + 1}" for i in range(len(runs))]

    # ── Build the common x-axis ────────────────────────────────────────────
    x_grid = _build_x_grid(runs, normalize_by, n_points)

    # ── Resample each run onto the shared grid ─────────────────────────────
    records: list[dict] = []

    for run, label in zip(runs, labels):
        df   = run.df
        date = df.index[0].date().isoformat()
        src  = run.source_file.name

        x_raw = _x_values(df, normalize_by)

        for metric in metrics:
            if metric not in df.columns:
                continue
            y_raw = df[metric].values.astype(float)

            # Interpolate across the common grid (drop NaN first)
            valid = ~np.isnan(y_raw) & ~np.isnan(x_raw)
            if valid.sum() < 2:
                continue

            y_interp = np.interp(x_grid, x_raw[valid], y_raw[valid])

            for x_val, y_val in zip(x_grid, y_interp):
                records.append({
                    "x":           float(x_val),
                    "metric":      metric,
                    "value":       float(y_val),
                    "label":       label,
                    "date":        date,
                    "source_file": src,
                })

    return pd.DataFrame(records)


def summary_table(
    runs: List["RunData"],
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return a wide-format summary DataFrame with one row per run and key
    aggregate metrics as columns.

    Useful for a quick side-by-side comparison before plotting.

    Parameters
    ----------
    runs : list of RunData
        Runs or slices to summarise.
    labels : list of str, optional
        Row labels.  Defaults to ``"Rep 1"``, ``"Rep 2"``, …

    Returns
    -------
    pd.DataFrame
        Columns: ``label``, ``date``, ``duration_s``, ``distance_m``,
        ``avg_pace_min_km``, ``avg_hr``, ``avg_power_w``.
    """
    if labels is None:
        labels = [f"Rep {i + 1}" for i in range(len(runs))]

    rows = []
    for run, label in zip(runs, labels):
        df   = run.df
        sp   = df["speed_ms"].dropna()
        dur  = float(df["elapsed_s"].max() - df["elapsed_s"].min())
        dist = float(df["distance_m"].max() - df["distance_m"].min())
        avg_pace = float((1000 / 60) / sp.mean()) if len(sp) > 0 and sp.mean() > 0 else float("nan")

        hr_col  = df["heart_rate"].dropna()
        pwr_col = df["power_w"].dropna() if "power_w" in df.columns else pd.Series(dtype=float)

        rows.append({
            "label":            label,
            "date":             df.index[0].date().isoformat(),
            "duration_s":       round(dur, 1),
            "distance_m":       round(dist, 1),
            "avg_pace_min_km":  round(avg_pace, 3),
            "avg_hr":           round(float(hr_col.mean()), 1) if len(hr_col) > 0 else None,
            "avg_power_w":      round(float(pwr_col.mean()), 1) if len(pwr_col) > 0 else None,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _x_values(df: pd.DataFrame, normalize_by: str) -> np.ndarray:
    """Return the raw x-axis array (0-based) for a single run."""
    if normalize_by in ("distance_m", "distance"):
        col = df["distance_m"].values.astype(float)
    elif normalize_by in ("elapsed_s", "duration"):
        col = df["elapsed_s"].values.astype(float)
    else:
        raise ValueError(
            f"Unknown normalize_by={normalize_by!r}.  "
            "Choose 'distance_m', 'elapsed_s', 'distance', or 'duration'."
        )
    x = col - col[0]  # zero-base
    if normalize_by in ("distance", "duration"):
        x = x / x[-1] if x[-1] > 0 else x
    return x


def _build_x_grid(
    runs: List["RunData"],
    normalize_by: str,
    n_points: int,
) -> np.ndarray:
    """
    Build the common x-axis grid.

    For normalised axes (distance / duration) the grid always spans [0, 1].
    For absolute axes (distance_m / elapsed_s) the grid spans
    [0, min_length] so that all runs cover the full extent.
    """
    if normalize_by in ("distance", "duration"):
        return np.linspace(0.0, 1.0, n_points)

    x_maxes = []
    for run in runs:
        x = _x_values(run.df, normalize_by)
        if len(x) > 0:
            x_maxes.append(x[-1])

    if not x_maxes:
        return np.linspace(0.0, 1.0, n_points)

    return np.linspace(0.0, min(x_maxes), n_points)
