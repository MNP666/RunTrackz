"""
runtrackz.database
~~~~~~~~~~~~~~~~~~
DuckDB-backed store for run metadata and parameters.

Install DuckDB:
    pip install duckdb

Quick start
-----------
    import runtrackz

    cfg  = runtrackz.load_config()
    run  = runtrackz.load("my_run.fit")
    hr   = runtrackz.hr_analysis.analyze(run, config=cfg)
    pace = runtrackz.pace_analysis.analyze(run)

    db = runtrackz.database.open("runs.db")
    db.insert_run(run, hr, pace)
    print(db.all_runs())

Schema versioning
-----------------
The schema is managed as a sequential list of migrations in _MIGRATIONS.
Each migration is applied exactly once and recorded in the `schema_versions`
table.  To add a new column or table in the future, append a new entry to
_MIGRATIONS — existing databases will be upgraded automatically on next open.

    _MIGRATIONS = [
        ...existing entries...,
        (2, "Add run_type column", \"\"\"
            ALTER TABLE runs ADD COLUMN run_type TEXT DEFAULT 'run';
        \"\"\"),
    ]
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Union, Optional

# ---------------------------------------------------------------------------
# Schema: ordered list of (version, description, sql) migrations
#
# Rules:
#   - Never edit or delete an existing entry (that would break existing DBs).
#   - To change the schema, append a new entry with the next version number.
#   - Each SQL block may contain multiple statements separated by semicolons.
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "Initial schema: metadata and core run parameters",
        """
        CREATE SEQUENCE IF NOT EXISTS runs_id_seq START 1;

        CREATE TABLE IF NOT EXISTS runs (
            -- Identity
            id               INTEGER  PRIMARY KEY DEFAULT nextval('runs_id_seq'),

            -- File metadata
            fit_file         TEXT     NOT NULL,   -- basename of the source .fit file
            runtrackz_version TEXT    NOT NULL,   -- library version that processed the file

            -- Timing
            run_date         DATE     NOT NULL,   -- local date of the run
            processed_at     TIMESTAMPTZ NOT NULL, -- when this row was inserted

            -- Core parameters
            distance_km      DOUBLE,              -- total distance in km
            duration_s       DOUBLE,              -- moving time in seconds
            trimp            DOUBLE,              -- Training Impulse (Bangsbo)

            -- Storage
            parquet_file     TEXT,                -- basename of the processed .parquet file

            -- Constraints
            UNIQUE (fit_file, run_date)           -- prevent duplicate imports
        );

        CREATE TABLE IF NOT EXISTS schema_versions (
            version      INTEGER PRIMARY KEY,
            description  TEXT    NOT NULL,
            applied_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
    ),
    (
        2,
        "Add run_type and avg_hr columns",
        """
        ALTER TABLE runs ADD COLUMN run_type TEXT;
        ALTER TABLE runs ADD COLUMN avg_hr   DOUBLE;
        """,
    ),
    # ── Future migrations go here ────────────────────────────────────────────
    # Example — uncomment and extend when ready:
    #
    # (
    #     3,
    #     "Add workouts table for interval sessions",
    #     """
    #     CREATE TABLE IF NOT EXISTS intervals (
    #         id        INTEGER PRIMARY KEY DEFAULT nextval('runs_id_seq'),
    #         run_id    INTEGER REFERENCES runs(id),
    #         lap_num   INTEGER NOT NULL,
    #         distance_m DOUBLE,
    #         duration_s DOUBLE,
    #         avg_pace_min_km DOUBLE,
    #         avg_hr   DOUBLE
    #     );
    #     """,
    # ),
]


# ---------------------------------------------------------------------------
# RunDatabase
# ---------------------------------------------------------------------------

class RunDatabase:
    """
    Manages a DuckDB database of processed runs.

    Parameters
    ----------
    path : str or Path
        Path to the DuckDB database file.  Use ``':memory:'`` for an
        in-memory database (useful for testing).
    """

    def __init__(self, path: Union[str, Path]):
        import duckdb  # imported here so the rest of the library works without it
        self.path = str(path)
        self._con = duckdb.connect(self.path)
        self._apply_migrations()

    # ── Schema management ─────────────────────────────────────────────────

    def _applied_versions(self) -> set[int]:
        """Return the set of migration version numbers already in the DB."""
        try:
            rows = self._con.execute(
                "SELECT version FROM schema_versions"
            ).fetchall()
            return {r[0] for r in rows}
        except Exception:
            return set()

    def _apply_migrations(self) -> None:
        """Run any migrations that have not yet been applied."""
        applied = self._applied_versions()
        for version, description, sql in _MIGRATIONS:
            if version in applied:
                continue
            # Execute each statement in the block
            for statement in _split_sql(sql):
                self._con.execute(statement)
            self._con.execute(
                "INSERT INTO schema_versions (version, description) VALUES (?, ?)",
                [version, description],
            )
        self._con.commit()

    def describe_schema(self) -> str:
        """
        Return a human-readable summary of the current database schema.

        Shows all tables, their columns, types, and any applied migrations.
        """
        lines = [f"Database: {self.path}", ""]

        # Tables and columns
        tables = self._con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()

        for (table,) in tables:
            lines.append(f"┌── {table}")
            cols = self._con.execute(
                "SELECT column_name, data_type, is_nullable, column_default "
                "FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
            for name, dtype, nullable, default in cols:
                null_str = "" if nullable == "YES" else " NOT NULL"
                default_str = f" DEFAULT {default}" if default else ""
                lines.append(f"│   {name:25s} {dtype}{null_str}{default_str}")
            lines.append("│")

        # Migration history
        lines.append("Applied migrations:")
        try:
            migrations = self._con.execute(
                "SELECT version, description, applied_at "
                "FROM schema_versions ORDER BY version"
            ).fetchall()
            for v, desc, applied_at in migrations:
                lines.append(f"  v{v}  {desc}  (applied {applied_at})")
        except Exception:
            lines.append("  (schema_versions table not yet created)")

        return "\n".join(lines)

    # ── Writing ───────────────────────────────────────────────────────────

    def insert_run(
        self,
        run: "RunData",          # noqa: F821
        hr_stats: "HRStats",     # noqa: F821
        pace_stats: "PaceStats", # noqa: F821
        parquet_file: "Optional[Union[str, Path]]" = None,  # noqa: F821
        run_type: Optional[str] = None,
        overwrite: bool = False,
    ) -> int:
        """
        Insert a processed run into the database.

        Parameters
        ----------
        run : RunData
            Parsed run from :func:`runtrackz.parser.load`.
        hr_stats : HRStats
            Result from :func:`runtrackz.hr_analysis.analyze`.
        pace_stats : PaceStats
            Result from :func:`runtrackz.pace_analysis.analyze`.
        parquet_file : str or Path, optional
            Path (or basename) of the ``.parquet`` file written for this run.
            Only the filename is stored, not the full path.
        run_type : str, optional
            Run type label, e.g. ``'easy'``, ``'tempo'``, ``'long_run'``,
            ``'workout'``.  Use the constants in :mod:`runtrackz.run_type`.
            Stored as-is; ``None`` when not provided.
        overwrite : bool
            If True, replace an existing row with the same fit_file + run_date.
            If False (default), raise an error on duplicate.

        Returns
        -------
        int
            The ``id`` of the inserted row.
        """
        from pathlib import Path as _Path
        from runtrackz import __version__

        fit_file     = run.source_file.name
        run_date     = run.df.index[0].date()
        processed_at = datetime.datetime.now(datetime.timezone.utc)
        parquet_name = _Path(parquet_file).name if parquet_file is not None else None

        if overwrite:
            self._con.execute(
                "DELETE FROM runs WHERE fit_file = ? AND run_date = ?",
                [fit_file, run_date],
            )

        self._con.execute(
            """
            INSERT INTO runs
                (fit_file, runtrackz_version, run_date, processed_at,
                 distance_km, duration_s, trimp, parquet_file,
                 run_type, avg_hr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                fit_file,
                __version__,
                run_date,
                processed_at,
                round(pace_stats.total_distance_km, 4),
                round(pace_stats.total_time_s, 1),
                round(hr_stats.trimp, 2),
                parquet_name,
                run_type,
                round(hr_stats.avg_hr, 1) if hr_stats.avg_hr is not None else None,
            ],
        )
        self._con.commit()

        row_id = self._con.execute(
            "SELECT id FROM runs WHERE fit_file = ? AND run_date = ?",
            [fit_file, run_date],
        ).fetchone()[0]
        return row_id

    # ── Reading ───────────────────────────────────────────────────────────

    def all_runs(self):
        """Return all runs as a pandas DataFrame."""
        import pandas as pd
        return self._con.execute("SELECT * FROM runs ORDER BY run_date").df()

    def query(self, sql: str, params: Optional[list] = None):
        """
        Execute an arbitrary SQL query and return the result as a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query string.  DuckDB syntax is supported.
        params : list, optional
            Positional parameters for parameterised queries (use ``?``).

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> db.query("SELECT run_date, distance_km, trimp FROM runs WHERE trimp > 80")
        >>> db.query("SELECT * FROM runs WHERE run_date >= ?", ["2026-01-01"])
        """
        result = self._con.execute(sql, params or [])
        return result.df()

    # ── Housekeeping ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        try:
            n = self._con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        except Exception:
            n = "?"
        return f"RunDatabase(path={self.path!r}, runs={n})"


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def open(path: Union[str, Path] = "runs.db") -> RunDatabase:
    """
    Open (or create) a RunTrackz database.

    Parameters
    ----------
    path : str or Path
        Path to the DuckDB file.  Defaults to ``runs.db`` in the current
        directory.  Pass ``':memory:'`` for a temporary in-memory database.

    Returns
    -------
    RunDatabase
    """
    return RunDatabase(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_sql(sql: str) -> list[str]:
    """Split a multi-statement SQL block on semicolons, dropping empty parts."""
    return [s.strip() for s in sql.split(";") if s.strip()]
