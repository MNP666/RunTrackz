"""
example.py — runtrackz usage demonstration
-------------------------------------------
Run this script from the project root:

    python example.py path/to/your_run.fit [--overwrite] [--run-type TYPE]

It will print a summary, save overview charts, store the parsed DataFrame as a
Parquet file in data/processed/, and record the run in data/database/runs.db.
Re-running with the same file is safe — pass --overwrite to replace the entry.

Options:
    --overwrite       Replace existing database entry if the file was already
                      processed.
    --run-type TYPE   Tag this run with a type label, e.g. --run-type tempo.
                      Valid values: easy, long_run, tempo, workout, race.

Folder layout created automatically:
    data/
    ├── raw/          ← copy your .fit files here
    ├── processed/    ← parquet files land here
    └── database/     ← DuckDB file lives here
"""

import sys
from pathlib import Path

# Make sure runtrackz is importable from this directory
sys.path.insert(0, str(Path(__file__).parent))

import runtrackz

# Data directory layout (relative to this script)
_DATA_DIR = Path(__file__).parent / "data"
DB_PATH   = _DATA_DIR / "database" / "runs.db"


def main(fit_path: str, overwrite: bool = False, run_type: str | None = None):
    fit_path = Path(fit_path)

    # ── 1. Load config & run ─────────────────────────────────────────────
    print(f"\nLoading: {fit_path}")
    cfg = runtrackz.load_config()
    run = runtrackz.load(fit_path)
    print(f"  → {run}")
    print(f"\nDataFrame columns: {list(run.df.columns)}")
    print(run.df.head(3).to_string())

    # ── 2. Heart rate analysis ───────────────────────────────────────────
    print("\n" + "="*60)
    hr_stats = runtrackz.hr_analysis.analyze(run, config=cfg)
    print(hr_stats.summary())

    print("\nZone DataFrame:")
    print(hr_stats.to_dataframe().to_string(index=False))

    # ── 3. Pace analysis ─────────────────────────────────────────────────
    print("\n" + "="*60)
    pace_stats = runtrackz.pace_analysis.analyze(run, split_distance_m=1000)
    print(pace_stats.summary())

    print("\nSplits DataFrame:")
    print(pace_stats.splits_dataframe().to_string(index=False))

    # ── 4. Charts ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Generating charts...")

    fig = runtrackz.charts.overview(run, hr_stats=hr_stats, pace_stats=pace_stats, config=cfg)
    out = fit_path.with_suffix('.overview.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")

    fig2 = runtrackz.charts.splits_bar(pace_stats, config=cfg)
    out2 = fit_path.with_suffix('.splits.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out2}")

    fig3 = runtrackz.charts.hr_zone_bar(hr_stats, config=cfg)
    out3 = fit_path.with_suffix('.hr_zones.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out3}")

    # ── 5. Parquet ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    parquet_path = runtrackz.make_parquet_path(run, _DATA_DIR / "processed")
    saved = run.save_parquet(parquet_path)
    print(f"Saved parsed run to: {saved}")

    # ── 6. Database ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to database: {DB_PATH}")
    with runtrackz.database.open(DB_PATH) as db:
        try:
            row_id = db.insert_run(run, hr_stats, pace_stats,
                                   parquet_file=saved, run_type=run_type,
                                   overwrite=overwrite)
            print(f"  Inserted run id={row_id}")
        except Exception as e:
            print(f"  Skipped (already in database): {e}")
            print("  Tip: pass --overwrite to replace the existing entry.")

        print("\nAll runs in database:")
        print(db.all_runs().to_string(index=False))

    print("\nDone!")


if __name__ == '__main__':
    import runtrackz.run_type as _rt

    argv  = sys.argv[1:]
    flags = [a for a in argv if a.startswith('--')]
    args  = [a for a in argv if not a.startswith('--')]

    if not args:
        print("Usage: python example.py path/to/run.fit [--overwrite] [--run-type TYPE]")
        sys.exit(1)

    # --run-type VALUE
    run_type: str | None = None
    if '--run-type' in argv:
        idx = argv.index('--run-type')
        if idx + 1 >= len(argv) or argv[idx + 1].startswith('--'):
            print("Error: --run-type requires a value, e.g. --run-type tempo")
            sys.exit(1)
        run_type = argv[idx + 1]
        if run_type not in _rt.ALL_TYPES:
            print(f"Error: unknown run type '{run_type}'.  "
                  f"Valid values: {', '.join(_rt.ALL_TYPES)}")
            sys.exit(1)

    main(args[0], overwrite='--overwrite' in flags, run_type=run_type)
