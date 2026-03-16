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

Modules
-------
- runtrackz.parser       : .fit file loading (built-in pure-Python)
- runtrackz.hr_analysis  : HR zones, TRIMP, aerobic decoupling
- runtrackz.pace_analysis: pace, splits, power, elevation
- runtrackz.charts       : matplotlib visualizations
- runtrackz.config       : YAML config loader
"""

from runtrackz.parser import load, RunData
from runtrackz.config import load_config, Config
from runtrackz import hr_analysis, pace_analysis, charts, config

__version__ = "0.1.0"
__all__ = [
    "load", "RunData",
    "load_config", "Config",
    "hr_analysis", "pace_analysis", "charts", "config",
]
