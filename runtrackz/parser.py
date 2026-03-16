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
    54:  ('enhanced_altitude',     5,        500), # m
    55:  ('enhanced_speed',        1000,     0),   # m/s
    253: ('timestamp',             1,        0),   # FIT epoch seconds
}

# Field number -> (name, scale, offset) for message 18 (session)
_SESSION_FIELD_MAP = {
    2:   ('start_time',            1,    0),
    7:   ('sport',                 1,    0),
    9:   ('total_elapsed_time',    1000, 0),   # s
    10:  ('total_timer_time',      1000, 0),   # s
    11:  ('total_distance',        100,  0),   # m
    13:  ('total_calories',        1,    0),   # kcal
    15:  ('avg_speed',             1000, 0),   # m/s
    16:  ('max_speed',             1000, 0),   # m/s
    17:  ('avg_heart_rate',        1,    0),   # bpm
    18:  ('max_heart_rate',        1,    0),   # bpm
    19:  ('avg_cadence',           1,    0),   # rpm
    20:  ('max_cadence',           1,    0),   # rpm
    21:  ('avg_power',             1,    0),   # watts
    22:  ('max_power',             1,    0),   # watts
    25:  ('total_ascent',          1,    0),   # m
    26:  ('total_descent',         1,    0),   # m
    29:  ('num_laps',              1,    0),
    253: ('timestamp',             1,    0),
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

            # Speed
            spd = r.get(6)
            if spd is not None and spd != 0xFFFF:
                row['speed_ms'] = spd / 1000.0
                row['speed_kmh'] = row['speed_ms'] * 3.6
                row['pace_min_km'] = (1000.0 / row['speed_ms'] / 60.0) if row['speed_ms'] > 0 else None
            else:
                row['speed_ms'] = row['speed_kmh'] = row['pace_min_km'] = None

            # Distance
            dist = r.get(5)
            row['distance_m'] = dist / 100.0 if (dist is not None and dist != 0xFFFFFFFF) else None

            # Altitude (prefer enhanced_altitude field 54, fall back to field 2)
            alt = r.get(54) or r.get(2)
            row['altitude_m'] = (alt / 5.0 - 500.0) if (alt is not None and alt != 0xFFFF) else None

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

    def __repr__(self) -> str:
        if self.df.empty:
            return "RunData(empty)"
        duration = self.df['elapsed_s'].max()
        dist = self.df['distance_m'].max() if 'distance_m' in self.df else 0
        return (
            f"RunData("
            f"date={self.df.index[0].date()}, "
            f"duration={duration/60:.1f}min, "
            f"distance={dist/1000:.2f}km, "
            f"points={len(self.df)})"
        )
