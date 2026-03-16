# RunTrackz

A Python library for analysing running data from `.fit` files — no third-party FIT parsing library required. Tested with **Coros** and **Garmin** wearables.

## Features

- **Pure-Python FIT parser** — reads `.fit` files using only the standard library and pandas; no fitdecode or similar dependency
- **Activity type validation** — detects sport and sub-sport from the FIT session; `run.is_run` lets you filter non-running files before analysis
- **Heart rate analysis** — zone breakdown (5 or 6 zones), TRIMP, aerobic decoupling
- **Pace & splits** — per-km splits, elevation gain/loss, running power
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
│   ├── __init__.py        # Public API
│   ├── parser.py          # .fit binary parser + parquet helpers
│   ├── hr_analysis.py     # HR zones, TRIMP, aerobic decoupling
│   ├── pace_analysis.py   # Pace, splits, power, elevation
│   ├── charts.py          # Matplotlib visualisations
│   ├── config.py          # YAML config loader
│   └── database.py        # DuckDB run store with migrations
├── data/
│   ├── raw/               # Drop .fit files here (git-ignored)
│   ├── processed/         # Parquet files written here (git-ignored)
│   └── database/          # runs.db lives here (git-ignored)
├── example.py             # Single-file demo script
├── process_runs.py        # Batch processing script
├── runtrackz.yml          # Personal config (git-ignored)
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
```

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

run.df           # pandas DataFrame indexed by timestamp
run.session      # dict of FIT session summary fields
run.source_file  # Path to the original .fit file

# Activity type
run.sport        # e.g. 'running', 'cycling', 'walking'
run.sub_sport    # e.g. 'trail', 'treadmill', 'street', 'generic'
run.is_run       # True when sport == 'running'
```

**DataFrame columns:**

| Column | Unit | Description |
|---|---|---|
| `heart_rate` | bpm | Heart rate |
| `cadence` | rpm | Cadence (foot strikes per minute, one foot) |
| `steps_per_min` | spm | Running cadence (cadence × 2) |
| `speed_ms` | m/s | Speed |
| `speed_kmh` | km/h | Speed |
| `pace_min_km` | min/km | Pace (decimal) |
| `distance_m` | m | Cumulative distance |
| `altitude_m` | m | Altitude |
| `power_w` | W | Running power |
| `latitude` / `longitude` | degrees | GPS position |
| `vertical_oscillation_cm` | cm | Vertical oscillation |
| `stance_time_ms` | ms | Ground contact time |
| `elapsed_s` | s | Elapsed time from start |

### Parquet storage

```python
# Generate a dated filename, auto-incrementing if multiple runs on same day:
#   data/processed/12032026_run_01.parquet
#   data/processed/12032026_run_02.parquet  ← second run that day
path = runtrackz.make_parquet_path(run, "data/processed")
saved = run.save_parquet(path)

# Load back later
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
    db.insert_run(run, hr, pace, parquet_file=saved)

    # All runs as a DataFrame
    print(db.all_runs())

    # Arbitrary SQL
    df = db.query("SELECT run_date, distance_km, trimp FROM runs ORDER BY run_date")
    df = db.query("SELECT * FROM runs WHERE run_date >= ?", ["2026-01-01"])

    # Inspect the live schema and migration history
    print(db.describe_schema())
```

**Schema** (`runs` table, v1):

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

**Extending the schema**

The schema is managed as an ordered list of migrations in `database.py`. To add new columns or tables, append a new entry to `_MIGRATIONS` — existing databases are upgraded automatically on next `open()`:

```python
_MIGRATIONS = [
    ...existing entries...,
    (
        2,
        "Add run_type and avg_hr columns",
        """
        ALTER TABLE runs ADD COLUMN run_type TEXT DEFAULT 'run';
        ALTER TABLE runs ADD COLUMN avg_hr   DOUBLE;
        """,
    ),
]
```

## Dashboard Integration

All chart functions return standard `Figure` objects, so they embed directly into Streamlit or Dash:

```python
# Streamlit
import streamlit as st
st.pyplot(runtrackz.charts.overview(run, config=cfg))

# Dash — save to buffer and embed as image
import io
buf = io.BytesIO()
runtrackz.charts.overview(run, config=cfg).savefig(buf, format="png")
```

## Supported Devices

Tested with **Coros** and **Garmin** wearables. The binary FIT protocol is standard across all manufacturers; the per-second record data (HR, cadence, speed, distance, altitude) uses the same field numbers on both devices. Some session-level summary fields differ between manufacturers but this does not affect analysis quality.

Fields parsed from record messages:

- Heart rate, cadence, speed, distance, power
- GPS position (latitude / longitude)
- Altitude — enhanced field (Coros: field 54/55; Garmin: field 73/78) with fallback to legacy field
- Running dynamics (vertical oscillation, stance time)

Activity type (sport / sub-sport) is decoded from the FIT session message and exposed as `run.sport`, `run.sub_sport`, and `run.is_run`. The full FIT sport and sub-sport enum tables are included in `parser.py`.
