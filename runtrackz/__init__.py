"""
runtrackz
~~~~~~~~~
A Python library for analysing running .fit files.

Quick start
-----------
    import runtrackz

    # Load personal config (reads runtrackz.yml from cwd or ~/.runtrackz.yml)
    cfg = runtrackz.load_config()

    run = runtrackz.load("my_run.fit")
    print(run)

    hr  = runtrackz.hr_analysis.analyze(run, config=cfg)
    print(hr.summary())

    pace = runtrackz.pace_analysis.analyze(run)
    print(pace.summary())

    fig = runtrackz.charts.overview(run, config=cfg)
    fig.savefig("overview.png", dpi=150)

External parser / FitTrackz integration
-----------------------------------------
    import runtrackz
    from runtrackz import DATAFRAME_SCHEMA

    # Build a DataFrame that conforms to the schema, then wrap it:
    run = runtrackz.RunData.from_dataframe(
        df,
        session={"sport": "running"},
        source_file="session_1.fit",
        is_smoothed=True,          # skip internal rolling-median when True
    )

Slicing and comparison
-----------------------
    # Slice by distance (km) or elapsed time (seconds) — both re-zero axes
    rep1 = run.slice_km(1.0, 2.0)
    rep2 = run.slice_elapsed(300, 600)

    # Align multiple slices onto a shared x-axis for comparison
    from runtrackz import comparison
    df = comparison.align_intervals([rep1, rep2], normalize_by="distance_m",
                                    labels=["Rep 1", "Rep 2"])

Modules
-------
- runtrackz.parser             : .fit file loading, RunData, DATAFRAME_SCHEMA
- runtrackz.comparison         : align_intervals(), summary_table()
- runtrackz.hr_analysis        : HR zones, TRIMP, aerobic decoupling
- runtrackz.pace_analysis      : pace, splits, power, elevation
- runtrackz.charts             : matplotlib visualizations
- runtrackz.config             : YAML config loader
- runtrackz.run_type           : run type constants and heuristic classifier
- runtrackz.long_run_analysis  : pacing strategy, thirds, cardiac drift
- runtrackz.tempo_analysis     : HR drift, pace CV, time at threshold
- runtrackz.workout_analysis   : interval detection, pace consistency
- runtrackz.treadmill_analysis : GAP, gradient schedule, per-segment metrics
"""

from runtrackz.parser import load, load_parquet, make_parquet_path, RunData, DATAFRAME_SCHEMA
from runtrackz.config import load_config, Config
from runtrackz import (
    hr_analysis, pace_analysis, charts, config, database,
    run_type, long_run_analysis, tempo_analysis, workout_analysis,
    treadmill_analysis, comparison,
)

__version__ = "0.1.0"
__all__ = [
    "load", "load_parquet", "make_parquet_path", "RunData", "DATAFRAME_SCHEMA",
    "load_config", "Config",
    "hr_analysis", "pace_analysis", "charts", "config",
    "database",
    "run_type", "long_run_analysis", "tempo_analysis", "workout_analysis",
    "treadmill_analysis", "comparison",
]
