# RunTrackz

A Python library for analysing running data from `.fit` files — no third-party FIT parsing library required. Tested with **Coros** and **Garmin** wearables.

## Features

- **Pure-Python FIT parser** — reads `.fit` files using only the standard library and pandas; no fitdecode or similar dependency
- **External parser support** — `RunData.from_dataframe()` accepts a pre-built DataFrame from any source (e.g. FitTrackz) via a formal column contract (`DATAFRAME_SCHEMA`)
- **Activity type validation** — detects sport and sub-sport from the FIT session; `run.is_run` lets you filter non-running activities
- **Slicing API** — extract sub-intervals by distance (`slice_km`) or time (`slice_elapsed`) with re-zeroed axes
- **Interval comparison** — `comparison.align_intervals()` resamples multiple slices onto a shared x-axis for side-by-side plotting
- **Heart rate analysis** — zone breakdown (5 or 6 zones), TRIMP, aerobic decoupling
- **Pace & splits** — per-km splits, elevation gain/loss, running power
- **Interval workout analysis** — automatic rep/recovery detection, pace and HR consistency metrics
- **Long run analysis** — pacing strategy, thirds analysis, cardiac drift
- **Tempo run analysis** — HR drift, pace CV, time at lactate threshold
- **Treadmill analysis** — grade-adjusted pace (Minetti model), user-supplied gradient schedule, per-segment metrics
- **Matplotlib charts** — overview figure, HR zone bar, splits bar, pace over distance
- **Parquet storage** — saves each parsed run to a dated `.parquet` file in `data/processed/`
- **DuckDB database** — migration-based schema for longitudinal analysis
- **Batch processing** — `process_runs.py` processes every `.fit` file in `data/raw/` in one command
- **YAML config** — personal athlete settings: max HR, resting HR, lactate threshold, zone method, colour scheme

## Requirements

```
pandas
matplotlib
pyyaml
pyarrow    # for parquet storage
duckdb     # for the run database
```

```bash
pip install pandas matplotlib pyyaml pyarrow duckdb
```

## Project Structure

```
RunTrackz/
├── runtrackz/
│   ├── __init__.py            # Public API
│   ├── parser.py              # .fit parser, RunData, DATAFRAME_SCHEMA, parquet helpers
│   ├── comparison.py          # align_intervals(), summary_table()
│   ├── hr_analysis.py         # HR zones, TRIMP, aerobic decoupling
│   ├── pace_analysis.py       # Pace, splits, power, elevation
│   ├── workout_analysis.py    # Interval/rep detection, recovery analysis
│   ├── long_run_analysis.py   # Pacing strategy, thirds, cardiac drift
│   ├── tempo_analysis.py      # HR drift, pace CV, time at threshold
│   ├── treadmill_analysis.py  # GAP, gradient schedule, per-segment metrics
│   ├── run_type.py            # Run type constants and heuristic classifier
│   ├── charts.py              # Matplotlib visualisations
│   ├── config.py              # YAML config loader
│   └── database.py            # DuckDB run store with migrations
├── data/
│   ├── raw/                   # Drop .fit files here (git-ignored)
│   ├── processed/             # Parquet files written here (git-ignored)
│   └── database/              # runs.db lives here (git-ignored)
├── examples/
│   └── intervals.py           # Interval workout analysis demo
├── example.py                 # Single-file demo script
├── process_runs.py            # Batch processing script
├── runtrackz.yml              # Personal config (git-ignored)
└── README.md
```

## Quick Start

```python
import runtrackz

# Load your personal config (reads runtrackz.yml or ~/.runtrackz.yml)
cfg = runtrackz.load_config()

# Parse a .fit file
run = runtrackz.load("my_run.fit")
print(run)
# RunData(sport=running, date=2026-03-12, duration=64.5min, distance=12.35km, points=3843)

# Skip non-running activities before analysis
if not run.is_run:
    raise ValueError(f"Expected a run, got '{run.sport}'")

# Heart rate analysis
hr = runtrackz.hr_analysis.analyze(run, config=cfg)
print(hr.summary())

# Pace and splits
pace = runtrackz.pace_analysis.analyze(run)
print(pace.summary())

# Charts
fig = runtrackz.charts.overview(run, config=cfg)
fig.savefig("overview.png", dpi=150)

# Save parsed data as parquet
path = runtrackz.make_parquet_path(run, "data/processed")
run.save_parquet(path)     # e.g. data/processed/12032026_run_01.parquet
```

Or run the single-file demo:

```bash
python example.py path/to/your_run.fit
```

## Batch Processing

Drop your `.fit` files into `data/raw/` and run:

```bash
# Inspect what's there without writing anything
python process_runs.py --dry-run

# Process all new files
python process_runs.py

# Re-process everything (overwrites existing parquet files and database rows)
python process_runs.py --overwrite

# Tag the run type for all files in this batch
python process_runs.py --run-type tempo
```

Valid `--run-type` values: `run`, `long_run`, `tempo`, `intervals`, `treadmill`, `race`, `recovery`.

The script skips non-running activities automatically and prints a summary at the end:

```
Found 12 .fit file(s) in data/raw
[ 1/12] 12032026_run.fit … ok  → 12032026_run_01.parquet
[ 2/12] 09042025_run.fit … ok  → 09042025_run_01.parquet
[ 3/12] 08042025_bike.fit … skipped  (sport=cycling)
...
==================================================
Results for 12 file(s):
  Processed  : 10
  Not a run  : 2  (cycling, walking, etc.)
==================================================
```

## Configuration

Copy `runtrackz.yml` into your project root (or `~/.runtrackz.yml` for a global config) and edit it to match your physiology. The file is git-ignored so personal data is never committed.

```yaml
athlete:
  max_hr: 182
  resting_hr: 50

heart_rate_zones:
  method: max_pct        # max_pct | hrr | absolute
  boundaries: [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]  # 6 values = 5 zones
  names:
    - Z1 Recovery
    - Z2 Aerobic
    - Z3 Tempo
    - Z4 Threshold
    - Z5 Maximum

lactate_threshold:
  pace_min_km: 4.18      # 4:11 /km  (shown as reference line on charts)
  heart_rate: 166        # bpm

charts:
  color_scheme: default  # default | spectral | rainbow
```

**Zone methods:**

| Method | Description |
|---|---|
| `max_pct` | Boundaries as a fraction of `max_hr` (default) |
| `hrr` | Heart Rate Reserve / Karvonen method (uses `resting_hr` too) |
| `absolute` | Boundaries given directly in bpm |

**6-zone example** — add one more boundary value:

```yaml
heart_rate_zones:
  boundaries: [0.50, 0.60, 0.70, 0.80, 0.87, 0.93, 1.00]  # 7 values = 6 zones
```

Config is resolved in this order: explicit path → `./runtrackz.yml` → `~/.runtrackz.yml` → built-in defaults.

## API

### Loading a run

```python
run = runtrackz.load("run.fit")

run.df           # pandas DataFrame indexed by UTC-aware timestamp
run.session      # dict of FIT session summary fields
run.source_file  # Path to the original .fit file
run.is_smoothed  # True if the data was pre-smoothed by an external parser

# Activity type
run.sport        # e.g. 'running', 'cycling', 'walking'
run.sub_sport    # e.g. 'trail', 'treadmill', 'street', 'generic'
run.is_run       # True when sport == 'running'
```

### DataFrame schema (`DATAFRAME_SCHEMA`)

`runtrackz.DATAFRAME_SCHEMA` is a dict that defines the formal column contract for `run.df`. All analysis modules expect this layout.

**Required columns** — must be present in every `RunData`:

| Column | Unit | Description |
|---|---|---|
| `heart_rate` | bpm | Heart rate |
| `speed_ms` | m/s | Speed |
| `speed_kmh` | km/h | Speed |
| `pace_min_km` | min/km | Pace (decimal minutes) |
| `distance_m` | m | Cumulative distance |
| `elapsed_s` | s | Elapsed time from start |

**Optional columns** — present when the device records them:

| Column | Unit | Description |
|---|---|---|
| `altitude_m` | m | Altitude |
| `power_w` | W | Running power (Stryd etc.) |
| `cadence` | rpm | Steps per minute, one foot |
| `steps_per_min` | spm | Running cadence (cadence × 2) |
| `latitude` / `longitude` | degrees | GPS position |
| `vertical_oscillation_cm` | cm | Vertical oscillation |
| `stance_time_ms` | ms | Ground contact time |

The DataFrame index must be a **UTC-aware `DatetimeIndex`**.

### External parsers and `from_dataframe()`

If you parse `.fit` files with an external tool (e.g. the Rust-based **FitTrackz** library for fast smoothing), wrap the result in a `RunData` using `from_dataframe()` rather than going through the built-in parser:

```python
import runtrackz

# df is a DataFrame produced by FitTrackz (already smoothed and indexed by UTC timestamp)
run = runtrackz.RunData.from_dataframe(
    df,
    session={"sport": "running", "sub_sport": "generic"},
    source_file="session_2026-03-12.fit",   # str or Path; used for labelling
    is_smoothed=True,                        # tells analysis modules to skip internal smoothing
)

# All analysis modules work normally from here
hr   = runtrackz.hr_analysis.analyze(run, config=cfg)
pace = runtrackz.pace_analysis.analyze(run)
```

`from_dataframe()` will:

- Ensure the index is a UTC-aware `DatetimeIndex`
- Derive any computable required column that is missing (`elapsed_s` from the index, `speed_kmh`/`speed_ms` from one another, `pace_min_km` from speed)
- Validate that all required columns from `DATAFRAME_SCHEMA` are present after derivation
- Set the `is_smoothed` flag on the returned `RunData`

### `is_smoothed` flag

`run.is_smoothed` signals that the data was pre-smoothed before reaching RunTrackz. When `True`, analysis modules use `smooth_window=1` (no additional smoothing) instead of applying their default rolling median. The flag is persisted in Parquet metadata and restored on `load_parquet()`.

```python
run = runtrackz.RunData.from_dataframe(df, session, is_smoothed=True)
run.is_smoothed  # True

path = runtrackz.make_parquet_path(run, "data/processed")
run.save_parquet(path)

restored = runtrackz.load_parquet(path)
restored.is_smoothed  # True  ← flag round-trips through parquet
```

### Slicing

Extract a sub-interval from any `RunData`. Both methods return a new `RunData` with `distance_m` and `elapsed_s` **re-zeroed** to the start of the slice, so analysis modules always see a run that starts at 0.

```python
# Slice by cumulative distance (kilometres)
rep1 = run.slice_km(1.0, 2.0)     # 1 km to 2 km
rep2 = run.slice_km(3.0, 4.0)

# Slice by elapsed time (seconds)
warmup = run.slice_elapsed(0, 600)    # first 10 minutes
rep3   = run.slice_elapsed(1200, 1800)
```

The session dict of a slice includes `is_slice=True` plus `slice_start_km` / `slice_end_km` (distance slices) or `slice_start_s` / `slice_end_s` (time slices), so you can always trace a slice back to its source position.

### Interval comparison

`runtrackz.comparison.align_intervals()` resamples a list of `RunData` objects onto a common x-axis and returns a tidy long-format DataFrame ready for plotting.

```python
from runtrackz import comparison

# --- Within-session rep comparison ---
stats  = runtrackz.workout_analysis.analyze(run, hr, pace)
slices = [run.slice_elapsed(iv.start_s, iv.end_s) for iv in stats.intervals]

df = comparison.align_intervals(
    slices,
    normalize_by="distance_m",               # common x-axis in metres
    metrics=["pace_min_km", "heart_rate"],
    labels=[f"Rep {iv.index}" for iv in stats.intervals],
)

# df columns: x | metric | value | label | date | source_file
import matplotlib.pyplot as plt
for (label, metric), grp in df.groupby(["label", "metric"]):
    plt.plot(grp["x"], grp["value"], label=f"{label} – {metric}")

# --- Cross-session comparison ---
run1   = runtrackz.load("session_a.fit")
run2   = runtrackz.load("session_b.fit")
stats1 = runtrackz.workout_analysis.analyze(run1, ...)
stats2 = runtrackz.workout_analysis.analyze(run2, ...)

df = comparison.align_intervals(
    [
        run1.slice_elapsed(stats1.intervals[2].start_s, stats1.intervals[2].end_s),
        run2.slice_elapsed(stats2.intervals[2].start_s, stats2.intervals[2].end_s),
    ],
    labels=["Session A – Rep 3", "Session B – Rep 3"],
)
```

**`normalize_by` options:**

| Value | x-axis | Range |
|---|---|---|
| `"distance_m"` *(default)* | Metres from slice start | `[0, min_distance]` |
| `"elapsed_s"` | Seconds from slice start | `[0, min_duration]` |
| `"distance"` | Fraction of total distance | `[0, 1]` |
| `"duration"` | Fraction of total duration | `[0, 1]` |

For absolute axes the grid is truncated to the shortest run so every series covers the full extent. For normalised axes the grid always spans 0 → 1.

**Quick summary table:**

```python
summary = comparison.summary_table(slices, labels=[...])
# Columns: label | date | duration_s | distance_m | avg_pace_min_km | avg_hr | avg_power_w
```

### Parquet storage

```python
# Generate a dated filename, auto-incrementing if multiple runs on same day:
#   data/processed/12032026_run_01.parquet
#   data/processed/12032026_run_02.parquet  ← second run that day
path = runtrackz.make_parquet_path(run, "data/processed")
saved = run.save_parquet(path)

# Load back later — session dict, source_file, and is_smoothed are all restored
run = runtrackz.load_parquet("data/processed/12032026_run_01.parquet")
```

### Heart rate analysis

```python
hr = runtrackz.hr_analysis.analyze(
    run,
    config=cfg,         # uses cfg.max_hr, cfg.resting_hr, cfg.zones
    max_hr=182,         # override config value
    resting_hr=50,      # override config value
    method='max_pct',   # 'max_pct' | 'hrr' — ignored when config provided
)

hr.avg_hr                  # float, bpm
hr.max_hr                  # float, bpm
hr.trimp                   # float, Training Impulse (Bangsbo)
hr.aerobic_decoupling_pct  # float, pace:HR drift (%) — lower is better
hr.zones                   # dict {zone_num: HRZone}
hr.summary()               # formatted text summary
hr.to_dataframe()          # zone breakdown as DataFrame
```

### Pace analysis

```python
pace = runtrackz.pace_analysis.analyze(
    run,
    split_distance_m=1000,  # default: per km
    smooth_speed=True,
    smooth_window=5,
)

pace.total_distance_km   # float
pace.total_time_str      # e.g. "1:04:29"
pace.avg_pace_str        # e.g. "5:10 /km"
pace.avg_cadence         # float, steps per minute
pace.avg_power           # float, watts
pace.total_ascent_m      # float, metres
pace.splits              # list of Split objects
pace.splits_dataframe()  # splits as DataFrame
pace.summary()           # formatted text summary
```

### Interval workout analysis

```python
stats = runtrackz.workout_analysis.analyze(
    run,
    hr_stats=hr,                  # optional — enriches rep/recovery objects with HR
    pace_stats=pace,              # optional
    effort_threshold_ms=3.5,      # speed threshold in m/s (≈ 4:45 /km)
    min_interval_s=60.0,          # minimum rep duration
    min_interval_m=200.0,         # minimum rep distance
)

stats.num_intervals               # int
stats.intervals                   # list of Interval objects
stats.recoveries                  # list of Recovery objects
stats.avg_interval_pace_min_km    # float
stats.pace_consistency_cv         # coefficient of variation across reps (lower = more consistent)
stats.hr_consistency_cv           # HR consistency across reps
stats.summary()                   # formatted text summary

# Each Interval has: index, start_s, end_s, duration_s, distance_m,
#                    avg_pace_min_km, avg_hr, peak_hr, avg_power_w
# Each Recovery has: index, start_s, end_s, duration_s, avg_hr,
#                    hr_at_start, hr_at_end, hr_drop_bpm
```

**Threshold guidance:** the default of 3.5 m/s (~4:45 /km) suits intervals run at 4:00–4:30 /km. Override with `effort_threshold_ms` if your target pace differs. Use `examples/intervals.py` as a starting point:

```bash
python examples/intervals.py my_intervals.fit --threshold 4:30
```

### Long run analysis

```python
lr = runtrackz.long_run_analysis.analyze(run, hr_stats=hr)

lr.pacing_strategy        # e.g. 'negative split', 'positive split', 'even'
lr.thirds_pace            # list of three avg pace values (first, middle, last thirds)
lr.cardiac_drift_pct      # HR drift across the run (%)
lr.summary()
```

### Tempo run analysis

```python
tempo = runtrackz.tempo_analysis.analyze(run, hr_stats=hr, config=cfg)

tempo.time_at_threshold_s     # seconds spent at or above lactate threshold HR/pace
tempo.hr_drift_pct            # HR drift across the tempo effort
tempo.pace_cv                 # pace coefficient of variation
tempo.summary()
```

### Treadmill analysis

The treadmill module computes **grade-adjusted pace (GAP)** using the Minetti et al. (2002) metabolic cost model. Because wearables do not reliably record treadmill incline, you supply the gradient yourself as either a step schedule or a function.

```python
# Constant gradient
stats = runtrackz.treadmill_analysis.analyze(run, hr_stats=hr, pace_stats=pace,
                                              gradient=5.0)   # 5 % incline

# Step schedule: [(elapsed_seconds, gradient_pct), ...]
schedule = [(0, 0), (300, 3), (600, 5), (900, 8), (1200, 0)]
stats = runtrackz.treadmill_analysis.analyze(run, gradient=schedule)

# Callable: gradient_pct = f(elapsed_s)
import numpy as np
stats = runtrackz.treadmill_analysis.analyze(
    run, gradient=lambda t: 2 + 4 * np.sin(np.pi * t / 1800)
)

stats.avg_gap_min_km              # overall grade-adjusted pace
stats.flat_equivalent_distance_m  # distance equivalent on flat ground
stats.gap_factor                  # >1 means uphill (harder than flat)
stats.segments                    # list of GradientSegment objects
stats.summary()

# Reference table showing GAP effect at common gradients
print(runtrackz.treadmill_analysis.gap_table([0, 1, 3, 5, 8, 10]))
```

**GAP factors at common gradients** (Minetti 2002):

| Gradient | GAP factor | Effect on a 5:00 /km pace |
|---|---|---|
| 0 % | 1.000 | 5:00 /km |
| 1 % | 1.055 | 4:44 /km |
| 3 % | 1.174 | 4:16 /km |
| 5 % | 1.301 | 3:51 /km |
| 8 % | 1.509 | 3:19 /km |
| 10 % | 1.658 | 3:01 /km |

### Run types

```python
import runtrackz.run_type as rt

rt.ALL_TYPES   # ['run', 'long_run', 'tempo', 'intervals', 'treadmill', 'race', 'recovery']

# Heuristic classifier — returns one of the above strings based on duration,
# pace, HR, and sub_sport fields
detected = rt.classify(run, hr_stats=hr, pace_stats=pace)
```

### Charts

All chart functions return a `matplotlib.figure.Figure` and accept an optional `config` argument for zone colours and lactate threshold reference lines.

```python
fig = runtrackz.charts.overview(run, config=cfg)           # 4-panel overview
fig = runtrackz.charts.splits_bar(pace, config=cfg)        # per-km split bars
fig = runtrackz.charts.hr_zone_bar(hr, config=cfg)         # HR zone breakdown
fig = runtrackz.charts.heart_rate_over_time(run, config=cfg)
fig = runtrackz.charts.pace_over_distance(run, config=cfg)

fig.savefig("chart.png", dpi=150, bbox_inches="tight")
```

**Colour schemes** — set `charts.color_scheme` in `runtrackz.yml`:

| Scheme | Description |
|---|---|
| `default` | Custom blue → green → yellow → orange → red palette |
| `spectral` | Matplotlib Spectral colormap (cool → warm diverging) |
| `rainbow` | Matplotlib rainbow colormap (violet → red) |

## Database

RunTrackz stores processed runs in a local DuckDB file for longitudinal analysis.

```python
with runtrackz.database.open("data/database/runs.db") as db:
    db.insert_run(run, hr, pace, parquet_file=saved, run_type="intervals")

    # All runs as a DataFrame
    print(db.all_runs())

    # Arbitrary SQL
    df = db.query("SELECT run_date, distance_km, trimp FROM runs ORDER BY run_date")
    df = db.query("SELECT * FROM runs WHERE run_type = ?", ["intervals"])

    # Inspect the live schema and migration history
    print(db.describe_schema())
```

**Schema** (`runs` table, current):

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-incrementing primary key |
| `fit_file` | TEXT | Basename of the source `.fit` file |
| `runtrackz_version` | TEXT | Library version that processed the file |
| `run_date` | DATE | Local date of the run |
| `processed_at` | TIMESTAMPTZ | When the row was inserted |
| `distance_km` | DOUBLE | Total distance in km |
| `duration_s` | DOUBLE | Moving time in seconds |
| `trimp` | DOUBLE | Training Impulse (Bangsbo method) |
| `parquet_file` | TEXT | Basename of the corresponding `.parquet` file |
| `run_type` | TEXT | Run type tag (see run_type module) *(v2)* |
| `avg_hr` | DOUBLE | Average heart rate *(v2)* |

**Extending the schema**

The schema is managed as an ordered list of migrations in `database.py`. To add new columns or tables, append a new entry to `_MIGRATIONS` — existing databases are upgraded automatically on next `open()`:

```python
_MIGRATIONS = [
    ...existing entries...,
    (
        3,
        "Add notes column",
        "ALTER TABLE runs ADD COLUMN notes TEXT;",
    ),
]
```

## Dashboard integration (Panel + FitTrackz)

RunTrackz is designed to be the analysis engine for a Panel-based dashboard, while **FitTrackz** (a Rust module) handles fast pre-processing and smoothing of raw `.fit` files. The boundary rule is: *"would this be useful in a Jupyter notebook without a dashboard?"* — if yes, it belongs in RunTrackz; if it is pure UI state or callback logic, it belongs in the dashboard.

```
.fit file
   │
   ▼
FitTrackz (Rust)           ← fast binary parsing + noise smoothing
   │ DataFrame (smoothed)
   ▼
RunData.from_dataframe()   ← entry point into RunTrackz
   │ run.is_smoothed = True
   ▼
hr_analysis / pace_analysis / workout_analysis / ...
   │ stats objects
   ▼
comparison.align_intervals()   ← slice and compare regions of interest
   │ tidy DataFrame
   ▼
Panel dashboard            ← charts, widgets, selection UI
```

**Typical dashboard workflow:**

```python
import runtrackz
import panel as pn

# 1. Parse and smooth with FitTrackz (called from Panel callback or on startup)
#    df = fittrackz.load_and_smooth("session.fit")

# 2. Wrap in RunData — no parsing overhead, no re-smoothing
run = runtrackz.RunData.from_dataframe(df, session, is_smoothed=True)

# 3. Analyse
hr    = runtrackz.hr_analysis.analyze(run, config=cfg)
pace  = runtrackz.pace_analysis.analyze(run)
stats = runtrackz.workout_analysis.analyze(run, hr, pace)

# 4. Let the dashboard control region selection — RunTrackz handles the slicing
#    (e.g. user drags a range selector over km 3.2 → 4.3)
region = run.slice_km(3.2, 4.3)

# 5. Compare selected intervals
slices = [run.slice_elapsed(iv.start_s, iv.end_s) for iv in stats.intervals]
df_cmp = runtrackz.comparison.align_intervals(slices, normalize_by="distance_m")

# 6. Plot with matplotlib and embed in Panel
fig = runtrackz.charts.overview(run, config=cfg)
pn.pane.Matplotlib(fig)
```

All `charts.*` functions return standard `matplotlib.figure.Figure` objects and embed directly into Panel with `pn.pane.Matplotlib`.

## Supported Devices

Tested with **Coros** and **Garmin** wearables. The binary FIT protocol is standard across all manufacturers; the per-second record data (HR, cadence, speed, distance, altitude) uses the same field numbers on both devices. Some session-level summary fields differ between manufacturers but this does not affect analysis quality.

Fields parsed from record messages:

- Heart rate, cadence, speed, distance, power
- GPS position (latitude / longitude)
- Altitude — enhanced field (Coros: field 54/55; Garmin: field 73/78) with fallback to legacy field
- Running dynamics (vertical oscillation, stance time)

Activity type (sport / sub-sport) is decoded from the FIT session message and exposed as `run.sport`, `run.sub_sport`, and `run.is_run`. The full FIT sport and sub-sport enum tables are included in `parser.py`.
