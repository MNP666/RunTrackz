# RunTrackz

A Python library for analysing running data from `.fit` files — no third-party FIT parsing library required. Built for Coros and Garmin wearables.

## Features

- **Pure-Python FIT parser** — reads `.fit` files using only the standard library and pandas
- **Heart rate analysis** — zone breakdown, TRIMP, aerobic decoupling
- **Pace & splits** — per-km splits, elevation gain/loss, running power
- **Matplotlib charts** — overview figure, HR zone bar, splits bar, pace over distance
- **YAML config file** — personal athlete settings (max HR, lactate threshold, zone definitions, colour scheme)

## Requirements

```
pandas
matplotlib
pyyaml
```

Install them with:

```bash
pip install pandas matplotlib pyyaml
```

## Project Structure

```
RunTrackz/
├── runtrackz/
│   ├── __init__.py        # Public API
│   ├── parser.py          # .fit binary parser
│   ├── hr_analysis.py     # HR zones, TRIMP, aerobic decoupling
│   ├── pace_analysis.py   # Pace, splits, power, elevation
│   ├── charts.py          # Matplotlib visualisations
│   └── config.py          # YAML config loader
├── example.py             # Runnable demo script
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
# RunData(date=2026-03-12, duration=64.5min, distance=12.35km, points=3843)

# Heart rate analysis
hr = runtrackz.hr_analysis.analyze(run, config=cfg)
print(hr.summary())

# Pace and splits
pace = runtrackz.pace_analysis.analyze(run)
print(pace.summary())

# Charts
fig = runtrackz.charts.overview(run, config=cfg)
fig.savefig("overview.png", dpi=150)
```

Or run the demo script directly:

```bash
python example.py path/to/your_run.fit
```

## Configuration

Copy `runtrackz.yml` into your project root (or `~/.runtrackz.yml` for a user-wide config) and edit it to match your physiology. The file is listed in `.gitignore` so personal data is never committed.

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
  color_scheme: default  # default | hsv | matplotlib
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

Config is located automatically in this order: explicit path → `./runtrackz.yml` → `~/.runtrackz.yml` → built-in defaults.

## API

### Loading a run

```python
run = runtrackz.load("run.fit")

run.df           # pandas DataFrame indexed by timestamp
run.session      # dict of FIT session summary fields
run.source_file  # Path to the original .fit file
```

**DataFrame columns:**

| Column | Unit | Description |
|---|---|---|
| `heart_rate` | bpm | Heart rate |
| `cadence` | rpm | Cadence (foot strikes per minute) |
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
# 4-panel overview: HR, pace, elevation, zone pie
fig = runtrackz.charts.overview(run, config=cfg)

# Per-km split bar chart with HR overlay
fig = runtrackz.charts.splits_bar(pace, config=cfg)

# Horizontal HR zone bar chart
fig = runtrackz.charts.hr_zone_bar(hr, config=cfg)

# HR vs time with zone shading
fig = runtrackz.charts.heart_rate_over_time(run, config=cfg)

# Pace vs distance with LT reference line
fig = runtrackz.charts.pace_over_distance(run, config=cfg)

fig.savefig("chart.png", dpi=150, bbox_inches="tight")
```

**Colour schemes** — set `charts.color_scheme` in `runtrackz.yml`:

| Scheme | Description |
|---|---|
| `default` | Blue → green → yellow → orange → red |
| `hsv` | Evenly-spaced HSV hues (scales to any number of zones) |
| `matplotlib` | Matplotlib default cycle (C0–C5 / tab10) |

## Dashboard Integration

All chart functions return standard `Figure` objects, so they drop directly into Streamlit or Dash:

```python
# Streamlit
import streamlit as st
st.pyplot(runtrackz.charts.overview(run, config=cfg))

# Dash
import dash_core_components as dcc
fig = runtrackz.charts.overview(run, config=cfg)
# convert to plotly or save/embed as image
```

## Supported Devices

Tested with **Coros** wearables. Should work with any Garmin-compatible `.fit` file that contains standard `record` messages (global message number 20). Fields parsed:

- Heart rate, cadence, speed, distance, power
- GPS position (latitude / longitude)
- Altitude (enhanced and standard)
- Running dynamics (vertical oscillation, stance time)
