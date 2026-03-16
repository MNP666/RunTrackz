"""
runtrackz.parser
~~~~~~~~~~~~~~~~
Parses Garmin/Coros .fit files into a pandas DataFrame using a
built-in pure-Python implementation — no third-party FIT library required.

Only standard library modules (struct, datetime) and pandas are needed.
"""

from __future__ import annotations

import datetime
import struct
from pathlib import Path
from typing import Union

import pandas as pd

# ---------------------------------------------------------------------------
# FIT protocol constants
# ---------------------------------------------------------------------------

FIT_EPOCH = datetime.datetime(1989, 12, 31, 0, 0, 0, tzinfo=datetime.timezone.utc)

_BASE_TYPES = {
    0x00: ('B', 1),  # enum
    0x01: ('b', 1),  # sint8
    0x02: ('B', 1),  # uint8
    0x83: ('h', 2),  # sint16
    0x84: ('H', 2),  # uint16
    0x85: ('i', 4),  # sint32
    0x86: ('I', 4),  # uint32
    0x07: ('s', 1),  # string
    0x88: ('f', 4),  # float32
    0x89: ('d', 8),  # float64
    0x0A: ('B', 1),  # uint8z
    0x8B: ('H', 2),  # uint16z
    0x8C: ('I', 4),  # uint32z
    0x0D: ('B', 1),  # byte
    0x8E: ('q', 8),  # sint64
    0x8F: ('Q', 8),  # uint64
    0x90: ('Q', 8),  # uint64z
}

# Global message number -> message name
_GLOBAL_MESSAGES = {
    0: 'file_id',
    18: 'session',
    19: 'lap',
    20: 'record',
    21: 'event',
    23: 'device_info',
}

# Field number -> (name, scale, offset) for message 20 (record)
# scale/offset follow FIT SDK spec: value = raw / scale - offset
_RECORD_FIELD_MAP = {
    0:   ('position_lat',          11930465, 0),   # semicircles -> degrees
    1:   ('position_long',         11930465, 0),   # semicircles -> degrees
    2:   ('altitude',              5,        500), # m
    3:   ('heart_rate',            1,        0),   # bpm
    4:   ('cadence',               1,        0),   # rpm (steps/min = cadence*2)
    5:   ('distance',              100,      0),   # m
    6:   ('speed',                 1000,     0),   # m/s
    7:   ('power',                 1,        0),   # watts
    13:  ('temperature',           1,        0),   # °C
    29:  ('accumulated_power',     1,        0),   # watts
    32:  ('vertical_speed',        1000,     0),   # m/s
    39:  ('vertical_oscillation',  10,       0),   # cm
    40:  ('stance_time_percent',   100,      0),   # %
    41:  ('stance_time',           10,       0),   # ms
    53:  ('fractional_cadence',    128,      0),   # rpm
    54:  ('enhanced_altitude',     5,        500), # m  — Coros
    55:  ('enhanced_speed',        1000,     0),   # m/s — Coros
    73:  ('enhanced_speed_2',      1000,     0),   # m/s — Garmin (same scale as field 55)
    78:  ('enhanced_altitude_2',   5,        500), # m   — Garmin (same scale as field 54)
    253: ('timestamp',             1,        0),   # FIT epoch seconds
}

# Field number -> (name, scale, offset) for message 18 (session)
# Field positions validated against Coros .fit files.
# These differ from the standard Garmin FIT SDK profile for some fields;
# each mapping has been confirmed by cross-checking raw values against
# the same metrics computed from the per-second record messages.
_SESSION_FIELD_MAP = {
    2:   ('start_time',         1,    0),   # uint32, FIT epoch → datetime
    5:   ('sport',              1,    0),   # enum uint8 → decoded to string below
    6:   ('sub_sport',          1,    0),   # enum uint8 → decoded to string below
    7:   ('total_elapsed_time', 1000, 0),   # uint32, ms → s (wall-clock time)
    8:   ('total_timer_time',   1000, 0),   # uint32, ms → s (active time, excl. pauses)
    9:   ('total_distance',     100,  0),   # uint32, cm → m
    11:  ('total_calories',     1,    0),   # uint16, kcal
    14:  ('avg_speed',          1000, 0),   # uint16, mm/s → m/s
    15:  ('max_speed',          1000, 0),   # uint16, mm/s → m/s
    16:  ('avg_heart_rate',     1,    0),   # uint8, bpm
    17:  ('max_heart_rate',     1,    0),   # uint8, bpm
    18:  ('avg_cadence',        1,    0),   # uint8, rpm (half-cadence; ×2 = steps/min)
    19:  ('max_cadence',        1,    0),   # uint8, rpm (half-cadence; ×2 = steps/min)
    22:  ('total_ascent',       1,    0),   # uint16, m
    23:  ('total_descent',      1,    0),   # uint16, m
    253: ('timestamp',          1,    0),   # uint32, FIT epoch → datetime
}

# FIT SDK sport enum (global message 0 field 7 / session field 7)
_SPORT_NAMES: dict[int, str] = {
    0:  'generic',
    1:  'running',
    2:  'cycling',
    3:  'transition',
    4:  'fitness_equipment',
    5:  'swimming',
    6:  'basketball',
    7:  'soccer',
    8:  'tennis',
    9:  'american_football',
    10: 'training',
    11: 'walking',
    12: 'cross_country_skiing',
    13: 'alpine_skiing',
    14: 'snowboarding',
    15: 'rowing',
    16: 'mountaineering',
    17: 'hiking',
    18: 'multisport',
    19: 'paddling',
    20: 'flying',
    21: 'e_biking',
    22: 'motorcycling',
    23: 'boating',
    24: 'driving',
    25: 'golf',
    26: 'hang_gliding',
    27: 'horseback_riding',
    28: 'hunting',
    29: 'fishing',
    30: 'inline_skating',
    31: 'rock_climbing',
    32: 'sailing',
    33: 'ice_skating',
    34: 'sky_diving',
    35: 'snowshoeing',
    36: 'snowmobiling',
    37: 'stand_up_paddleboarding',
    38: 'surfing',
    39: 'wakeboarding',
    40: 'water_skiing',
    41: 'kayaking',
    42: 'rafting',
    43: 'windsurfing',
    44: 'kitesurfing',
    45: 'tactical',
    46: 'jumpmaster',
    47: 'boxing',
    48: 'floor_climbing',
    53: 'multisport',
    254: 'all',
}

# FIT SDK sub_sport enum (same field across all sports; common running values shown first)
_SUB_SPORT_NAMES: dict[int, str] = {
    0:  'generic',
    # Running sub-sports
    1:  'treadmill',
    2:  'street',
    3:  'trail',
    4:  'track',
    # Cycling sub-sports
    5:  'spin',
    6:  'indoor_cycling',
    7:  'road',
    8:  'mountain',
    9:  'downhill',
    10: 'recumbent',
    11: 'cyclocross',
    12: 'hand_cycling',
    13: 'track_cycling',
    14: 'indoor_rowing',
    15: 'elliptical',
    16: 'stair_climbing',
    17: 'lap_swimming',
    18: 'open_water',
    19: 'flexibility_training',
    20: 'strength_training',
    21: 'warm_up',
    22: 'match',
    23: 'exercise',
    24: 'challenge',
    25: 'indoor_skiing',
    26: 'cardio_training',
    27: 'indoor_walking',
    28: 'e_bike_fitness',
    29: 'bmx',
    30: 'casual_walking',
    31: 'speed_walking',
    32: 'bike_to_run_transition',
    33: 'run_to_bike_transition',
    34: 'swim_to_bike_transition',
    35: 'atv',
    36: 'motocross',
    37: 'backcountry',
    38: 'resort',
    39: 'rc_drone',
    40: 'wingsuit',
    41: 'whitewater',
    42: 'skate_skiing',
    43: 'yoga',
    44: 'pilates',
    45: 'indoor_running',
    46: 'gravel_cycling',
    47: 'e_bike_mountain',
    48: 'commuting',
    49: 'mixed_surface',
    50: 'navigate',
    51: 'track_me',
    52: 'map',
    53: 'single_gas_diving',
    54: 'multi_gas_diving',
    55: 'gauge_diving',
    56: 'apnea_diving',
    57: 'apnea_hunting',
    58: 'virtual_activity',
    59: 'obstacle',
    254: 'all',
}

# Invalid / null values per base type (FIT SDK)
_INVALID_VALUES = {
    0x00: 0xFF,
    0x01: 0x7F,
    0x02: 0xFF,
    0x83: 0x7FFF,
    0x84: 0xFFFF,
    0x85: 0x7FFFFFFF,
    0x86: 0xFFFFFFFF,
    0x88: float('nan'),
    0x89: float('nan'),
    0x0A: 0x00,
    0x8B: 0x0000,
    0x8C: 0x00000000,
    0x8E: 0x7FFFFFFFFFFFFFFF,
    0x8F: 0xFFFFFFFFFFFFFFFF,
    0x90: 0x0000000000000000,
}


# ---------------------------------------------------------------------------
# FIT binary parser
# ---------------------------------------------------------------------------

def _parse_fit(path: Path) -> tuple[list[dict], dict]:
    """
    Parse a .fit file using the built-in pure-Python implementation.
    Returns (record_rows, session_summary).
    """
    with open(path, 'rb') as fh:
        raw = fh.read()

    header_size = raw[0]
    magic = raw[8:12]
    if magic != b'.FIT':
        raise ValueError(f"Not a valid .fit file: {path}")

    pos = header_size
    end = len(raw) - 2  # exclude trailing CRC
    local_defs: dict = {}
    record_rows: list[dict] = []
    session_data: dict = {}

    while pos < end:
        if pos >= len(raw):
            break
        hb = raw[pos]; pos += 1

        if hb & 0x80:
            # Compressed timestamp message
            lt = (hb >> 5) & 0x03
            if lt not in local_defs:
                break
            d = local_defs[lt]
            row = _read_data_fields(raw, pos, d)
            pos += sum(f[1] for f in d['fields'])
            if d['global'] == 20:
                record_rows.append(row)
            continue

        is_def = bool(hb & 0x40)
        has_dev = bool(hb & 0x20)
        lt = hb & 0x0F

        if is_def:
            pos += 1  # reserved byte
            arch = raw[pos]; pos += 1
            endian = '>' if arch else '<'
            gmn = struct.unpack_from(endian + 'H', raw, pos)[0]; pos += 2
            nf = raw[pos]; pos += 1
            fields = []
            for _ in range(nf):
                fields.append((raw[pos], raw[pos + 1], raw[pos + 2]))
                pos += 3
            dev_fields = []
            if has_dev:
                nd = raw[pos]; pos += 1
                for _ in range(nd):
                    dev_fields.append((raw[pos], raw[pos + 1], raw[pos + 2]))
                    pos += 3
            local_defs[lt] = {
                'global': gmn, 'fields': fields,
                'dev_fields': dev_fields, 'endian': endian,
            }
        else:
            if lt not in local_defs:
                break
            d = local_defs[lt]
            row = _read_data_fields(raw, pos, d)
            pos += sum(f[1] for f in d['fields'])
            for (_, fs, _t) in d.get('dev_fields', []):
                pos += fs
            if d['global'] == 20:
                record_rows.append(row)
            elif d['global'] == 18:
                session_data = _decode_message(row, _SESSION_FIELD_MAP)

    # Remove session values that decoded to a null sentinel.
    # uint16 null (0xFFFF) with /1000 scale → 65.535; uint32 null with /1000 → 65535.535.
    # Any speed above 50 m/s (180 km/h) is physically impossible for a run/ride.
    for _key in ('avg_speed', 'max_speed'):
        if session_data.get(_key, 0) > 50.0:
            session_data.pop(_key, None)

    # Decode sport/sub_sport integers to human-readable strings.
    # Keep the raw integer as sport_id / sub_sport_id for exact comparisons.
    if 'sport' in session_data:
        sport_id = int(session_data['sport'])
        session_data['sport_id']  = sport_id
        session_data['sport']     = _SPORT_NAMES.get(sport_id, f'unknown_{sport_id}')
    if 'sub_sport' in session_data:
        sub_id = int(session_data['sub_sport'])
        session_data['sub_sport_id'] = sub_id
        session_data['sub_sport']    = _SUB_SPORT_NAMES.get(sub_id, f'unknown_{sub_id}')

    return record_rows, session_data


def _read_data_fields(raw: bytes, pos: int, defn: dict) -> dict:
    """Read raw field values from a data message."""
    row: dict = {}
    for (fnum, fsize, ftype) in defn['fields']:
        fdata = raw[pos:pos + fsize]
        pos += fsize
        if ftype not in _BASE_TYPES:
            continue
        fmt, base_size = _BASE_TYPES[ftype]
        if ftype == 0x07:
            row[fnum] = fdata.split(b'\x00')[0].decode('utf-8', errors='replace')
        elif base_size == fsize:
            row[fnum] = struct.unpack_from(defn['endian'] + fmt, fdata)[0]
    return row


def _decode_message(row: dict, field_map: dict) -> dict:
    """Apply scale/offset and rename fields using a field map."""
    out: dict = {}
    for fnum, raw_val in row.items():
        if fnum not in field_map:
            continue
        name, scale, offset = field_map[fnum]
        if isinstance(raw_val, (int, float)):
            out[name] = raw_val / scale - offset
        else:
            out[name] = raw_val
    return out


def _fit_ts_to_datetime(ts: float) -> datetime.datetime:
    """Convert FIT timestamp (seconds since 1989-12-31) to datetime."""
    return FIT_EPOCH + datetime.timedelta(seconds=ts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(path: Union[str, Path]) -> 'RunData':
    """
    Load a .fit file and return a :class:`RunData` object.

    Parameters
    ----------
    path : str or Path
        Path to the .fit file.

    Returns
    -------
    RunData
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw_records, session = _parse_fit(path)
    return RunData._from_raw(raw_records, session, path)


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


class RunData:
    """
    Container for a single run parsed from a .fit file.

    Attributes
    ----------
    df : pd.DataFrame
        Per-second data with columns: heart_rate, cadence, steps_per_min,
        speed_ms, speed_kmh, pace_min_km, distance_m, altitude_m, power_w,
        latitude, longitude, vertical_oscillation_cm, stance_time_ms,
        elapsed_s.
    session : dict
        Summary fields from the FIT session message (total distance, avg HR,
        avg speed, total ascent, etc.).
    source_file : Path
        Path of the original .fit file.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        session: dict,
        source_file: Path,
    ):
        self.df = df
        self.session = session
        self.source_file = source_file

    @classmethod
    def _from_raw(
        cls,
        raw_records: list[dict],
        session: dict,
        path: Path,
    ) -> 'RunData':
        df = cls._build_df(raw_records)
        return cls(df=df, session=session, source_file=path)

    @staticmethod
    def _build_df(raw_records: list[dict]) -> pd.DataFrame:
        """Build a clean DataFrame from raw field-number-keyed records."""
        rows = []
        for r in raw_records:
            row: dict = {}

            # Timestamp
            ts_raw = r.get(253)
            if ts_raw is not None and ts_raw != 0xFFFFFFFF:
                row['timestamp'] = _fit_ts_to_datetime(ts_raw)
            else:
                continue  # skip records without a timestamp

            # Heart rate
            hr = r.get(3)
            row['heart_rate'] = hr if (hr is not None and hr != 0xFF) else None

            # Cadence (steps per minute = cadence * 2 for running)
            cad = r.get(4)
            row['cadence'] = cad if (cad is not None and cad != 0xFF) else None
            if row['cadence'] is not None:
                row['steps_per_min'] = row['cadence'] * 2

            # Speed: field 6 = legacy uint16 (Coros); field 73 = enhanced uint32 (Garmin).
            # Both use scale ÷1000 → m/s.
            spd = r.get(6)
            if spd is None or spd == 0xFFFF:
                v = r.get(73)
                spd = v if (v is not None and v != 0xFFFFFFFF) else None
            if spd is not None:
                row['speed_ms'] = spd / 1000.0
                row['speed_kmh'] = row['speed_ms'] * 3.6
                row['pace_min_km'] = (1000.0 / row['speed_ms'] / 60.0) if row['speed_ms'] > 0 else None
            else:
                row['speed_ms'] = row['speed_kmh'] = row['pace_min_km'] = None

            # Distance
            dist = r.get(5)
            row['distance_m'] = dist / 100.0 if (dist is not None and dist != 0xFFFFFFFF) else None

            # Altitude: field 54 = enhanced uint32 (Coros), field 78 = enhanced uint32 (Garmin),
            # field 2 = legacy uint16 (any device).  All use scale ÷5, offset -500 → metres.
            alt = r.get(54) or r.get(78) or r.get(2)
            row['altitude_m'] = (alt / 5.0 - 500.0) if (alt is not None and alt not in (0xFFFF, 0xFFFFFFFF)) else None

            # Power
            pwr = r.get(7)
            row['power_w'] = pwr if (pwr is not None and pwr != 0xFFFF) else None

            # GPS
            lat_raw = r.get(0)
            lon_raw = r.get(1)
            row['latitude'] = lat_raw * (180.0 / 2**31) if (lat_raw is not None and lat_raw != 0x7FFFFFFF) else None
            row['longitude'] = lon_raw * (180.0 / 2**31) if (lon_raw is not None and lon_raw != 0x7FFFFFFF) else None

            # Running dynamics
            vo = r.get(39)
            row['vertical_oscillation_cm'] = vo / 10.0 if (vo is not None and vo != 0xFFFF) else None

            st = r.get(41)
            row['stance_time_ms'] = st / 10.0 if (st is not None and st != 0xFFFF) else None

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index('timestamp').sort_index()
            df['elapsed_s'] = (df.index - df.index[0]).total_seconds()
        return df

    # ── Activity type helpers ─────────────────────────────────────────────

    @property
    def sport(self) -> str:
        """
        Human-readable sport name from the FIT session message
        (e.g. ``'running'``, ``'cycling'``, ``'walking'``).

        Returns ``'unknown'`` if the field was not present in the file.
        """
        return self.session.get('sport', 'unknown')

    @property
    def sub_sport(self) -> str:
        """
        Human-readable sub-sport name from the FIT session message
        (e.g. ``'trail'``, ``'treadmill'``, ``'street'``, ``'track'``).

        Returns ``'generic'`` if the field was not present in the file.
        """
        return self.session.get('sub_sport', 'generic')

    @property
    def is_run(self) -> bool:
        """
        ``True`` when the activity is a running activity
        (sport == ``'running'``).

        Use this to filter out non-run files before analysis::

            run = runtrackz.load("my_activity.fit")
            if not run.is_run:
                raise ValueError(f"Expected a run, got '{run.sport}'")
        """
        return self.sport == 'running'

    def save_parquet(self, path: Union[str, Path]) -> Path:
        """
        Save the run DataFrame and session metadata to a Parquet file.

        The session dict and original source file path are embedded in the
        Parquet file's schema metadata so the full ``RunData`` can be
        reconstructed with :func:`load_parquet` without any sidecar files.

        Parameters
        ----------
        path : str or Path
            Destination path (e.g. ``data/processed/my_run.parquet``).
            Parent directories are created automatically.

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
        source_file: Path = path  # fallback if metadata is missing

        if b'runtrackz_session' in meta:
            session = json.loads(meta[b'runtrackz_session'])
        if b'runtrackz_source_file' in meta:
            source_file = Path(meta[b'runtrackz_source_file'].decode())

        return cls(df=df, session=session, source_file=source_file)

    def __repr__(self) -> str:
        if self.df.empty:
            return "RunData(empty)"
        duration = self.df['elapsed_s'].max()
        dist = self.df['distance_m'].max() if 'distance_m' in self.df else 0
        sport_str = self.sport
        if self.sub_sport not in ('generic', 'unknown'):
            sport_str += f'/{self.sub_sport}'
        return (
            f"RunData("
            f"sport={sport_str}, "
            f"date={self.df.index[0].date()}, "
            f"duration={duration/60:.1f}min, "
            f"distance={dist/1000:.2f}km, "
            f"points={len(self.df)})"
        )
