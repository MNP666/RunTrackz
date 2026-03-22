"""
runtrackz.models
~~~~~~~~~~~~~~~~
Core data container and DataFrame schema contract for RunTrackz.

``RunData`` is the input accepted by all analysis modules.  It wraps a
per-second DataFrame (conforming to :data:`DATAFRAME_SCHEMA`) and optional
session-level metadata.

The canonical way to construct a ``RunData`` is via
:meth:`RunData.from_dataframe`, which accepts output from an external parser
such as **FitTrackz**.  RunTrackz itself does not parse ``.fit`` files —
that responsibility belongs to FitTrackz.

Example
-------
    import sys, subprocess
    import pandas as pd
    import runtrackz

    # 1. Parse with FitTrackz (see workbench/scratch.py for a fuller example)
    df_raw = ...  # DataFrame from FitTrackz subprocess

    # 2. Map FitTrackz column names → RunTrackz schema
    df = df_raw.rename(columns={"smoothed_heart_rate": "heart_rate",
                                "smoothed_speed": "speed_ms"})
    df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # 3. Wrap and analyse
    run = runtrackz.RunData.from_dataframe(df, session={}, is_smoothed=True)
    hr  = runtrackz.hr_analysis.analyze(run)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Union

import pandas as pd


# ---------------------------------------------------------------------------
# Public DataFrame schema contract
# ---------------------------------------------------------------------------

#: Formal specification of the per-second DataFrame consumed by all RunTrackz
#: analysis modules.
#:
#: Any external parser (e.g. FitTrackz) that supplies data to RunTrackz must
#: produce a DataFrame whose columns match this contract.  Pass the result to
#: :meth:`RunData.from_dataframe` — it will derive missing-but-computable
#: columns automatically before validating.
#:
#: **Index**: UTC-aware :class:`pandas.DatetimeIndex`
#: (``dtype='datetime64[ns, UTC]'``).  Must be sorted ascending.
#:
#: Format: ``{column_name: {"unit", "description", "required"}}``
DATAFRAME_SCHEMA: dict[str, dict] = {
    # ── Required ──────────────────────────────────────────────────────────
    "heart_rate": {
        "unit": "bpm",
        "description": "Heart rate",
        "required": True,
    },
    "speed_ms": {
        "unit": "m/s",
        "description": "Speed",
        "required": True,
    },
    "speed_kmh": {
        "unit": "km/h",
        "description": "Speed (derived from speed_ms if absent)",
        "required": True,
    },
    "pace_min_km": {
        "unit": "min/km (decimal)",
        "description": "Pace (derived from speed_ms if absent)",
        "required": True,
    },
    "distance_m": {
        "unit": "m",
        "description": "Cumulative distance from activity start",
        "required": True,
    },
    "elapsed_s": {
        "unit": "s",
        "description": "Elapsed time from activity start (derived from index if absent)",
        "required": True,
    },
    # ── Optional ──────────────────────────────────────────────────────────
    "altitude_m": {
        "unit": "m",
        "description": "Altitude above sea level",
        "required": False,
    },
    "power_w": {
        "unit": "W",
        "description": "Running power (e.g. Stryd foot pod)",
        "required": False,
    },
    "cadence": {
        "unit": "rpm",
        "description": "Cadence — single-foot strikes per minute",
        "required": False,
    },
    "steps_per_min": {
        "unit": "spm",
        "description": "Running cadence (cadence × 2)",
        "required": False,
    },
    "latitude": {
        "unit": "degrees",
        "description": "GPS latitude",
        "required": False,
    },
    "longitude": {
        "unit": "degrees",
        "description": "GPS longitude",
        "required": False,
    },
    "vertical_oscillation_cm": {
        "unit": "cm",
        "description": "Vertical oscillation",
        "required": False,
    },
    "stance_time_ms": {
        "unit": "ms",
        "description": "Ground contact time",
        "required": False,
    },
    "stride_length_m": {
        "unit": "m",
        "description": "Stride length — one full gait cycle (left + right step). "
                       "Key indicator of running economy; used in long-run economy "
                       "analysis alongside cadence.",
        "required": False,
    },
}


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def make_parquet_path(run: 'RunData', directory: Union[str, Path]) -> Path:
    """
    Generate a dated parquet filename for a run, auto-incrementing the
    counter when more than one run already exists on the same date.

    Format: ``DDMMYYYY_run_NN.parquet``

    Examples
    --------
    First run on 12 March 2026:
        ``12032026_run_01.parquet``

    Second run recorded the same day:
        ``12032026_run_02.parquet``

    Parameters
    ----------
    run : RunData
        The parsed run whose date will be used.
    directory : str or Path
        Destination directory (e.g. ``data/processed``).

    Returns
    -------
    Path
        Full path to the next available parquet file.
    """
    directory = Path(directory)
    date_str = run.df.index[0].strftime('%d%m%Y')   # e.g. '12032026'
    existing = sorted(directory.glob(f'{date_str}_run_*.parquet'))
    if existing:
        last_idx = int(existing[-1].stem.rsplit('_', 1)[-1])
        idx = last_idx + 1
    else:
        idx = 1
    return directory / f'{date_str}_run_{idx:02d}.parquet'


def load_parquet(path: Union[str, Path]) -> 'RunData':
    """
    Load a :class:`RunData` from a Parquet file written by
    :meth:`RunData.save_parquet`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.parquet`` file.

    Returns
    -------
    RunData
    """
    return RunData.load_parquet(path)


# ---------------------------------------------------------------------------
# RunData
# ---------------------------------------------------------------------------

class RunData:
    """
    Container for a single run session.

    Attributes
    ----------
    df : pd.DataFrame
        Per-second data indexed by UTC timestamp.  Columns conform to
        :data:`DATAFRAME_SCHEMA`.
    session : dict
        Summary fields from the activity (total distance, avg HR, avg speed,
        total ascent, sport, etc.).  May be empty when constructed without
        session data.
    source_file : Path
        Path to the original ``.fit`` file (or ``Path('external')`` when
        constructed from an external parser).
    is_smoothed : bool
        ``True`` when the data has been pre-smoothed by an external tool
        (e.g. FitTrackz).  Analysis modules that apply their own internal
        rolling median honour this flag — pass ``smooth_window=1`` to those
        functions to skip their internal smoothing when data is already clean.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        session: dict,
        source_file: Path,
        is_smoothed: bool = False,
    ):
        self.df = df
        self.session = session
        self.source_file = source_file
        self.is_smoothed = is_smoothed

    # ── External-parser entry point ───────────────────────────────────────

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        session: dict,
        source_file: Union[str, Path, None] = None,
        is_smoothed: bool = False,
    ) -> 'RunData':
        """
        Construct a :class:`RunData` from a pre-built DataFrame.

        This is the primary integration point for external parsers such as
        **FitTrackz**.  The DataFrame must have a UTC-aware
        :class:`~pandas.DatetimeIndex` and conform to :data:`DATAFRAME_SCHEMA`.

        Missing-but-derivable columns are computed automatically:

        - ``elapsed_s`` — derived from the index if absent.
        - ``speed_kmh`` — derived from ``speed_ms`` if absent.
        - ``speed_ms`` — derived from ``speed_kmh`` if absent.
        - ``pace_min_km`` — derived from ``speed_ms`` if absent.

        Parameters
        ----------
        df : pd.DataFrame
            Per-second data.  See :data:`DATAFRAME_SCHEMA` for the full
            column contract.
        session : dict
            Activity-level summary (sport, total_distance, avg_hr, etc.).
            Can be empty — all analysis uses the per-second DataFrame.
        source_file : str or Path, optional
            Path to the original ``.fit`` file.  Used for display and
            parquet metadata only.  Defaults to ``Path('external')``.
        is_smoothed : bool
            Set to ``True`` when the data has already been smoothed by the
            caller (e.g. FitTrackz applied SMA or EMA smoothing).
            Analysis modules that do their own rolling-median pass will note
            this; pass ``smooth_window=1`` to those functions to skip their
            internal smoothing.

        Returns
        -------
        RunData

        Raises
        ------
        ValueError
            If the index is not a DatetimeIndex or required columns are
            missing and cannot be derived.

        Examples
        --------
        Typical usage from FitTrackz:

        .. code-block:: python

            import runtrackz

            df, session = load_from_fittrackz("my_run.fit", smooth=True)
            run = runtrackz.RunData.from_dataframe(
                df, session, source_file="my_run.fit", is_smoothed=True
            )
            hr = runtrackz.hr_analysis.analyze(run, config=cfg)
        """
        df = df.copy()

        # ── Index: ensure UTC-aware DatetimeIndex ─────────────────────────
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame index must be a pandas DatetimeIndex.  "
                f"Got {type(df.index).__name__!r}."
            )
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz).upper() != "UTC":
            df.index = df.index.tz_convert("UTC")

        df = df.sort_index()

        # ── Derive missing-but-computable columns ─────────────────────────
        if "elapsed_s" not in df.columns:
            df["elapsed_s"] = (df.index - df.index[0]).total_seconds()

        if "speed_kmh" not in df.columns and "speed_ms" in df.columns:
            df["speed_kmh"] = df["speed_ms"] * 3.6

        if "speed_ms" not in df.columns and "speed_kmh" in df.columns:
            df["speed_ms"] = df["speed_kmh"] / 3.6

        if "pace_min_km" not in df.columns and "speed_ms" in df.columns:
            df["pace_min_km"] = (1000.0 / df["speed_ms"] / 60.0).where(
                df["speed_ms"] > 0
            )

        # ── Validate required columns ─────────────────────────────────────
        missing = [
            col
            for col, spec in DATAFRAME_SCHEMA.items()
            if spec["required"] and col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {missing}.  "
                f"See runtrackz.DATAFRAME_SCHEMA for the full contract."
            )

        return cls(
            df=df,
            session=dict(session),
            source_file=Path(source_file) if source_file is not None else Path("external"),
            is_smoothed=is_smoothed,
        )

    # ── Slicing ───────────────────────────────────────────────────────────

    def slice_km(self, start_km: float, end_km: float) -> 'RunData':
        """
        Return a new :class:`RunData` containing only rows between
        *start_km* and *end_km* of cumulative distance.

        ``elapsed_s`` and ``distance_m`` are re-zeroed to the start of the
        slice so that all analysis modules work naturally on the result.
        """
        mask = (
            (self.df["distance_m"] >= start_km * 1000.0) &
            (self.df["distance_m"] <= end_km   * 1000.0)
        )
        new_df = self.df[mask].copy()
        if not new_df.empty:
            new_df["distance_m"] -= new_df["distance_m"].iloc[0]
            new_df["elapsed_s"]  -= new_df["elapsed_s"].iloc[0]
        sliced_session = {
            **self.session,
            "is_slice":       True,
            "slice_start_km": start_km,
            "slice_end_km":   end_km,
        }
        return RunData(
            df=new_df,
            session=sliced_session,
            source_file=self.source_file,
            is_smoothed=self.is_smoothed,
        )

    def slice_elapsed(self, start_s: float, end_s: float) -> 'RunData':
        """
        Return a new :class:`RunData` containing only rows between
        *start_s* and *end_s* elapsed seconds.

        ``elapsed_s`` and ``distance_m`` are re-zeroed to the start of the
        slice so that all analysis modules work naturally on the result.
        """
        mask = (
            (self.df["elapsed_s"] >= start_s) &
            (self.df["elapsed_s"] <= end_s)
        )
        new_df = self.df[mask].copy()
        if not new_df.empty:
            dist_offset = new_df["distance_m"].iloc[0]
            time_offset = new_df["elapsed_s"].iloc[0]
            new_df["distance_m"] -= dist_offset
            new_df["elapsed_s"]  -= time_offset
        sliced_session = {
            **self.session,
            "is_slice":     True,
            "slice_start_s": start_s,
            "slice_end_s":   end_s,
        }
        return RunData(
            df=new_df,
            session=sliced_session,
            source_file=self.source_file,
            is_smoothed=self.is_smoothed,
        )

    # ── Activity type helpers ─────────────────────────────────────────────

    @property
    def sport(self) -> str:
        """
        Human-readable sport name from the session metadata
        (e.g. ``'running'``, ``'cycling'``, ``'walking'``).

        Returns ``'unknown'`` if the field was not present.
        """
        return self.session.get('sport', 'unknown')

    @property
    def sub_sport(self) -> str:
        """
        Human-readable sub-sport name from the session metadata
        (e.g. ``'trail'``, ``'treadmill'``, ``'street'``, ``'track'``).

        Returns ``'generic'`` if the field was not present.
        """
        return self.session.get('sub_sport', 'generic')

    @property
    def is_run(self) -> bool:
        """``True`` when the activity sport is ``'running'``."""
        return self.sport == 'running'

    # ── Persistence ───────────────────────────────────────────────────────

    def save_parquet(self, path: Union[str, Path]) -> Path:
        """
        Save the run DataFrame and session metadata to a Parquet file.

        The session dict and original source file path are embedded in the
        Parquet schema metadata so the full ``RunData`` can be reconstructed
        with :meth:`load_parquet` without any sidecar files.

        Parameters
        ----------
        path : str or Path
            Destination path.  Parent directories are created automatically.

        Returns
        -------
        Path
            The resolved path that was written.
        """
        import json
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialise session: convert datetime/date objects to ISO strings
        session_out: dict = {}
        for k, v in self.session.items():
            if isinstance(v, (datetime.datetime, datetime.date)):
                session_out[k] = v.isoformat()
            else:
                session_out[k] = v

        table = pa.Table.from_pandas(self.df)
        existing_meta = table.schema.metadata or {}
        custom_meta = {
            b'runtrackz_session':     json.dumps(session_out).encode(),
            b'runtrackz_source_file': str(self.source_file).encode(),
            b'runtrackz_is_smoothed': str(self.is_smoothed).encode(),
        }
        table = table.replace_schema_metadata({**existing_meta, **custom_meta})
        pq.write_table(table, path)
        return path

    @classmethod
    def load_parquet(cls, path: Union[str, Path]) -> 'RunData':
        """
        Load a :class:`RunData` from a Parquet file written by
        :meth:`save_parquet`.

        Parameters
        ----------
        path : str or Path
            Path to the ``.parquet`` file.

        Returns
        -------
        RunData
        """
        import json
        import pyarrow.parquet as pq

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        table = pq.read_table(path)
        df = table.to_pandas()

        meta = table.schema.metadata or {}
        session: dict = {}
        source_file: Path = path
        is_smoothed: bool = False

        if b'runtrackz_session' in meta:
            session = json.loads(meta[b'runtrackz_session'])
        if b'runtrackz_source_file' in meta:
            source_file = Path(meta[b'runtrackz_source_file'].decode())
        if b'runtrackz_is_smoothed' in meta:
            is_smoothed = meta[b'runtrackz_is_smoothed'].decode().lower() == 'true'

        return cls(df=df, session=session, source_file=source_file, is_smoothed=is_smoothed)

    def __repr__(self) -> str:
        if self.df.empty:
            return "RunData(empty)"
        duration = self.df['elapsed_s'].max()
        dist = self.df['distance_m'].max() if 'distance_m' in self.df else 0
        sport_str = self.sport
        if self.sub_sport not in ('generic', 'unknown'):
            sport_str += f'/{self.sub_sport}'
        is_slice   = self.session.get('is_slice', False)
        smoothed   = ', smoothed' if self.is_smoothed else ''
        slice_note = ', slice' if is_slice else ''
        return (
            f"RunData("
            f"sport={sport_str}, "
            f"date={self.df.index[0].date()}, "
            f"duration={duration/60:.1f}min, "
            f"distance={dist/1000:.2f}km, "
            f"points={len(self.df)}"
            f"{smoothed}{slice_note})"
        )
