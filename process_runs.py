"""
process_runs.py — batch process all .fit files in data/raw/
------------------------------------------------------------
Scans data/raw/ for .fit files, parses each one, saves a dated Parquet file
to data/processed/, and inserts a record into the DuckDB database.

Usage:
    python process_runs.py [--overwrite] [--dry-run] [--run-type TYPE]

Options:
    --overwrite       Replace existing database entries (and parquet files) for
                      runs that have already been processed.
    --dry-run         Parse and validate every file but do not write parquet
                      files or touch the database.  Useful for a first
                      inspection of a new batch of files.
    --run-type TYPE   Tag every run in this batch with the given run type, e.g.
                      --run-type tempo   or   --run-type long_run.
                      Valid values: easy, long_run, tempo, workout, race.
                      Leave unset to store NULL (you can update later via SQL).
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import runtrackz

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR     = Path(__file__).parent / "data"
RAW_DIR       = _DATA_DIR / "raw"
PROCESSED_DIR = _DATA_DIR / "processed"
DB_PATH       = _DATA_DIR / "database" / "runs.db"


def process_all(overwrite: bool = False, dry_run: bool = False, run_type: str | None = None) -> None:
    fit_files = sorted(
        p for p in RAW_DIR.iterdir()
        if p.suffix.lower() == ".fit"
    )

    if not fit_files:
        print(f"No .fit files found in {RAW_DIR}")
        return

    print(f"Found {len(fit_files)} .fit file(s) in {RAW_DIR}")
    if dry_run:
        print("DRY RUN — no files will be written or inserted.\n")
    if run_type:
        print(f"Run type tag  : {run_type}\n")

    cfg = runtrackz.load_config()

    # Counters
    n_processed  = 0
    n_duplicate  = 0
    n_not_run    = 0
    n_failed     = 0
    failed_files = []

    # Open the database once for the whole batch (no-op in dry-run)
    db_ctx = runtrackz.database.open(DB_PATH) if not dry_run else None
    if db_ctx is not None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for i, fit_path in enumerate(fit_files, 1):
            prefix = f"[{i:>{len(str(len(fit_files)))}}/{len(fit_files)}]"
            print(f"{prefix} {fit_path.name}", end=" … ", flush=True)

            try:
                run = runtrackz.load(fit_path)
            except Exception as exc:
                print(f"FAILED (parse error: {exc})")
                n_failed += 1
                failed_files.append((fit_path.name, str(exc)))
                continue

            if not run.is_run:
                print(f"skipped  (sport={run.sport})")
                n_not_run += 1
                continue

            try:
                hr_stats   = runtrackz.hr_analysis.analyze(run, config=cfg)
                pace_stats = runtrackz.pace_analysis.analyze(run)
            except Exception as exc:
                print(f"FAILED (analysis error: {exc})")
                n_failed += 1
                failed_files.append((fit_path.name, str(exc)))
                continue

            if dry_run:
                print(f"ok  {run}")
                n_processed += 1
                continue

            # ── Save parquet ──────────────────────────────────────────────
            parquet_path = runtrackz.make_parquet_path(run, PROCESSED_DIR)
            try:
                saved = run.save_parquet(parquet_path)
            except Exception as exc:
                print(f"FAILED (parquet write: {exc})")
                n_failed += 1
                failed_files.append((fit_path.name, str(exc)))
                continue

            # ── Insert into database ──────────────────────────────────────
            try:
                db_ctx.insert_run(
                    run, hr_stats, pace_stats,
                    parquet_file=saved,
                    run_type=run_type,
                    overwrite=overwrite,
                )
                print(f"ok  → {saved.name}")
                n_processed += 1
            except Exception as exc:
                msg = str(exc)
                # Detect duplicate constraint violation
                if "unique" in msg.lower() or "duplicate" in msg.lower():
                    print(f"duplicate (use --overwrite to replace)")
                    # Clean up the parquet we just wrote
                    saved.unlink(missing_ok=True)
                    n_duplicate += 1
                else:
                    print(f"FAILED (db error: {exc})")
                    saved.unlink(missing_ok=True)
                    n_failed += 1
                    failed_files.append((fit_path.name, msg))

    finally:
        if db_ctx is not None:
            db_ctx.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(fit_files)
    print()
    print("=" * 50)
    print(f"Results for {total} file(s):")
    print(f"  Processed  : {n_processed}")
    if n_duplicate:
        print(f"  Duplicate  : {n_duplicate}  (already in database)")
    if n_not_run:
        print(f"  Not a run  : {n_not_run}  (cycling, walking, etc.)")
    if n_failed:
        print(f"  Failed     : {n_failed}")
        for name, reason in failed_files:
            print(f"    • {name}: {reason}")
    print("=" * 50)

    if not dry_run and db_ctx is not None and n_processed > 0:
        print()
        with runtrackz.database.open(DB_PATH) as db:
            print("Database contents:")
            print(db.all_runs().to_string(index=False))


if __name__ == "__main__":
    import runtrackz.run_type as _rt

    argv      = sys.argv[1:]
    overwrite = "--overwrite" in argv
    dry_run   = "--dry-run"   in argv

    # --run-type VALUE  (value is the next positional token after the flag)
    run_type: str | None = None
    if "--run-type" in argv:
        idx = argv.index("--run-type")
        if idx + 1 >= len(argv) or argv[idx + 1].startswith("--"):
            print("Error: --run-type requires a value, e.g. --run-type tempo")
            sys.exit(1)
        run_type = argv[idx + 1]
        if run_type not in _rt.ALL_TYPES:
            print(f"Error: unknown run type '{run_type}'.  "
                  f"Valid values: {', '.join(_rt.ALL_TYPES)}")
            sys.exit(1)

    # Collect all flag tokens (including --run-type and its value)
    known_flags: set[str] = {"--overwrite", "--dry-run", "--run-type"}
    # Filter out --run-type's value token before checking for unknowns
    flag_tokens = [a for a in argv if a.startswith("--")]
    unknown     = set(flag_tokens) - known_flags
    if unknown:
        print(f"Unknown flag(s): {', '.join(sorted(unknown))}")
        print("Usage: python process_runs.py [--overwrite] [--dry-run] [--run-type TYPE]")
        sys.exit(1)

    process_all(overwrite=overwrite, dry_run=dry_run, run_type=run_type)
