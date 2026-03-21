"""
runtrackz
~~~~~~~~~
A Python library for analysing running activity data.

RunTrackz is a pure analysis library.  It does not parse ``.fit`` files —
that is the responsibility of **FitTrackz**.  The entry point is
:meth:`RunData.from_dataframe`, which accepts a DataFrame produced by
FitTrackz (or any other source) and wraps it for use with the analysis
modules below.

Quick start (with FitTrackz as the parser)
------------------------------------------
    import sys
    sys.path.insert(0, "/path/to/FitTrackz")
    from analysis.utils import run_fit   # FitTrackz subprocess wrapper

    import runtrackz

    cfg    = runtrackz.load_config()
    df_raw = run_fit("my_run.fit", smoother="sma", param=10)

    # Map FitTrackz column names → RunTrackz schema
    df = df_raw.rename(columns={
        "smoothed_heart_rate": "heart_rate",
        "smoothed_speed":      "speed_ms",
        "distance_m":          "distance_m",
    })
    df.index = df_raw["time"]          # UTC DatetimeIndex from FitTrackz

    run = runtrackz.RunData.from_dataframe(
        df,
        session={"sport": "running"},
        source_file="my_run.fit",
        is_smoothed=True,
    )

    hr   = runtrackz.hr_analysis.analyze(run, config=cfg)
    pace = runtrackz.pace_analysis.analyze(run)
    print(hr.summary())
    print(pace.summary())

    fig = runtrackz.charts.overview(run, config=cfg)
    fig.savefig("overview.png", dpi=150)

Loading a previously saved Parquet file
----------------------------------------
    run = runtrackz.load_parquet("data/processed/21032026_run_01.parquet")

Slicing and comparison
-----------------------
    rep1 = run.slice_km(1.0, 2.0)
    rep2 = run.slice_elapsed(300, 600)

    from runtrackz import comparison
    df = comparison.align_intervals([rep1, rep2], normalize_by="distance_m",
                                    labels=["Rep 1", "Rep 2"])

Modules
-------
- runtrackz.models           : RunData container, DATAFRAME_SCHEMA, parquet helpers
- runtrackz.comparison       : align_intervals(), summary_table()
- runtrackz.hr_analysis      : HR zones, TRIMP, aerobic decoupling
- runtrackz.pace_analysis    : pace, splits, power, elevation
- runtrackz.charts           : matplotlib visualizations
- runtrackz.config           : YAML config loader
- runtrackz.run_type         : run type constants and heuristic classifier
- runtrackz.long_run_analysis  : pacing strategy, thirds, cardiac drift
- runtrackz.tempo_analysis     : HR drift, pace CV, time at threshold
- runtrackz.workout_analysis   : interval detection, pace consistency
- runtrackz.treadmill_analysis : GAP, gradient schedule, per-segment metrics
"""

from runtrackz.models import load_parquet, make_parquet_path, RunData, DATAFRAME_SCHEMA
from runtrackz.config import load_config, Config
from runtrackz import (
    hr_analysis, pace_analysis, charts, config,
    run_type, long_run_analysis, tempo_analysis, workout_analysis,
    treadmill_analysis, comparison,
)

__version__ = "0.1.0"
__all__ = [
    "load_parquet", "make_parquet_path", "RunData", "DATAFRAME_SCHEMA",
    "load_config", "Config",
    "hr_analysis", "pace_analysis", "charts", "config",
    "run_type", "long_run_analysis", "tempo_analysis", "workout_analysis",
    "treadmill_analysis", "comparison",
]
