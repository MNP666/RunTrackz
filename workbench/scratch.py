"""
workbench/scratch.py
--------------------
Quick-iteration script for exploring RunTrackz analysis on a real .fit file.

This file is NOT part of the RunTrackz library.  Use it to try out analysis
functions, tweak parameters, and inspect outputs without touching library code.

Usage
-----
    cd /path/to/RunTrackz
    python workbench/scratch.py /path/to/your_run.fit

Requirements
------------
- FitTrackz binary must be built:  cd /path/to/FitTrackz && cargo build --release
- Set FITTRACKZ_DIR below (or the FITTRACKZ_DIR env var) to the FitTrackz repo root.
"""

import os
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

# Adjust these to match your local setup, or set via environment variables.
FITTRACKZ_DIR = Path(os.getenv("FITTRACKZ_DIR", "../../FitTrackz")).resolve()
RUNTRACKZ_DIR = Path(__file__).parent.parent

# Make both libraries importable
sys.path.insert(0, str(RUNTRACKZ_DIR))
sys.path.insert(0, str(FITTRACKZ_DIR))

import pandas as pd
import runtrackz
from analysis.utils import run_fit   # FitTrackz Python subprocess wrapper


# ── Config ────────────────────────────────────────────────────────────────────

FIT_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else FITTRACKZ_DIR / "data/raw/long_run.fit"

# FitTrackz smoother settings — tweak and re-run to compare
SMOOTHER = "sma"   # "sma", "ema", or "none"
PARAM    = 10      # window size for sma, alpha for ema


# ── 1. Parse with FitTrackz ───────────────────────────────────────────────────

print(f"\nParsing: {FIT_FILE}")
print(f"Smoother: {SMOOTHER}  param={PARAM}")

df_raw = run_fit(
    fit_file=FIT_FILE,
    smoother=SMOOTHER,
    param=PARAM,
)
print(f"Raw output shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}\n")


# ── 2. Map FitTrackz columns → RunTrackz DATAFRAME_SCHEMA ────────────────────
#
# FitTrackz outputs:  raw_<channel>, smoothed_<channel>, timestamp, distance_m, time
# RunTrackz expects:  heart_rate, speed_ms, distance_m  (+ derived columns)
#
# Rename the smoothed channels we want to use.  Adjust the mapping below if
# you're requesting different channels from FitTrackz.

column_map = {
    "smoothed_heart_rate": "heart_rate",
    "smoothed_speed":      "speed_ms",
    # altitude, cadence, power — add here if channels were requested
}

df = df_raw.rename(columns=column_map).copy()

# Use the UTC datetime column from FitTrackz as the index
df.index = pd.to_datetime(df["time"], utc=True)
df = df.drop(columns=["time", "timestamp"], errors="ignore")

print("Mapped DataFrame head:")
print(df[["heart_rate", "speed_ms", "distance_m"]].head(5).to_string())
print()


# ── 3. Construct RunData ──────────────────────────────────────────────────────

run = runtrackz.RunData.from_dataframe(
    df,
    session={"sport": "running"},
    source_file=FIT_FILE,
    is_smoothed=True,
)
print(run)
print()


# ── 4. Core analysis ──────────────────────────────────────────────────────────

cfg = runtrackz.load_config()

print("=" * 60)
hr_stats = runtrackz.hr_analysis.analyze(run, config=cfg)
print(hr_stats.summary())

print()
print("=" * 60)
pace_stats = runtrackz.pace_analysis.analyze(run, split_distance_m=1000)
print(pace_stats.summary())

print("\nSplits:")
print(pace_stats.splits_dataframe().to_string(index=False))


# ── 5. Optional: run-type-specific analysis ───────────────────────────────────
#
# Uncomment the block that matches your activity type:

# -- Long run --
# from runtrackz import long_run_analysis
# lr = long_run_analysis.analyze(run, hr_stats=hr_stats, pace_stats=pace_stats, config=cfg)
# print(lr.summary())

# -- Tempo --
# from runtrackz import tempo_analysis
# tempo = tempo_analysis.analyze(run, hr_stats=hr_stats, pace_stats=pace_stats, config=cfg)
# print(tempo.summary())

# -- Intervals --
# from runtrackz import workout_analysis
# wo = workout_analysis.analyze(run, hr_stats=hr_stats, pace_stats=pace_stats, config=cfg)
# print(wo.summary())


# ── 6. Charts ─────────────────────────────────────────────────────────────────
#
# Uncomment to save charts alongside this script:

# fig = runtrackz.charts.overview(run, hr_stats=hr_stats, pace_stats=pace_stats, config=cfg)
# out = Path(__file__).parent / "overview.png"
# fig.savefig(out, dpi=150, bbox_inches="tight")
# print(f"\nSaved chart: {out}")

print("\nDone.")
