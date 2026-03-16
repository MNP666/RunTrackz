"""
runtrackz.config
~~~~~~~~~~~~~~~~
Loads and validates runtrackz.yml (or a custom path).

Config file is looked up in this order:
  1. Explicit path passed to load_config()
  2. ./runtrackz.yml  (working directory)
  3. ~/.runtrackz.yml (home directory)

If no file is found, built-in defaults are used so the library
always works out of the box.

Example runtrackz.yml
---------------------
See the bundled runtrackz.yml in your project root.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_ZONE_BOUNDARIES_PCT = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]  # 5 zones
_DEFAULT_ZONE_NAMES = [
    "Z1 Recovery",
    "Z2 Aerobic",
    "Z3 Tempo",
    "Z4 Threshold",
    "Z5 Maximum",
]

# Current custom palette (matches original ZONE_COLORS in charts.py)
_COLORS_DEFAULT = {
    1: "#5aadff",
    2: "#4ccc6e",
    3: "#f0c040",
    4: "#f07830",
    5: "#e03030",
    6: "#a020a0",  # extra zone if 6 zones
}

# Matplotlib default cycle (C0–C5)
_COLORS_MATPLOTLIB = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
    6: "#8c564b",
}


def _hsv_colors(n: int) -> Dict[int, str]:
    """Generate n evenly-spaced HSV colours as hex strings."""
    import colorsys
    result = {}
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.88)
        result[i + 1] = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
    return result


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ZoneConfig:
    """Heart-rate zone definitions resolved to absolute bpm values."""
    num_zones: int
    boundaries_bpm: List[float]   # len = num_zones + 1
    names: List[str]              # len = num_zones
    method: str                   # 'max_pct' | 'hrr' | 'absolute'

    def as_dict(self) -> Dict[int, Tuple[float, float, str]]:
        """Return {zone_num: (lower_bpm, upper_bpm, name)} for hr_analysis."""
        result = {}
        for i in range(self.num_zones):
            result[i + 1] = (
                self.boundaries_bpm[i],
                self.boundaries_bpm[i + 1],
                self.names[i],
            )
        return result


@dataclass
class LactateThreshold:
    pace_min_km: Optional[float] = None   # e.g. 4.75 → 4:45 /km
    heart_rate: Optional[int] = None      # bpm

    @property
    def pace_str(self) -> Optional[str]:
        if self.pace_min_km is None:
            return None
        mins = int(self.pace_min_km)
        secs = int((self.pace_min_km - mins) * 60)
        return f"{mins}:{secs:02d} /km"


@dataclass
class Config:
    """
    Resolved configuration for a runtrackz session.

    Attributes
    ----------
    max_hr : int
        Athlete's maximum heart rate in bpm.
    resting_hr : int
        Athlete's resting heart rate in bpm.
    zones : ZoneConfig
        Heart-rate zone definitions (resolved to bpm).
    lactate_threshold : LactateThreshold
        LT pace and/or HR.
    color_scheme : str
        One of 'default', 'hsv', 'matplotlib'.
    zone_colors : dict
        {zone_num: hex_color} resolved from color_scheme.
    source_file : Path or None
        The yml file that was loaded (None if using built-in defaults).
    """

    max_hr: int = 185
    resting_hr: int = 50
    zones: ZoneConfig = field(default_factory=lambda: _default_zones(185, 50))
    lactate_threshold: LactateThreshold = field(default_factory=LactateThreshold)
    color_scheme: str = "default"
    zone_colors: Dict[int, str] = field(default_factory=dict)
    source_file: Optional[Path] = None

    def __post_init__(self):
        if not self.zone_colors:
            self.zone_colors = _resolve_colors(self.color_scheme, self.zones.num_zones)

    def summary(self) -> str:
        lines = [
            f"runtrackz config  {'(from ' + str(self.source_file) + ')' if self.source_file else '(defaults)'}",
            f"  max HR       : {self.max_hr} bpm",
            f"  resting HR   : {self.resting_hr} bpm",
            f"  zone method  : {self.zones.method}",
            f"  num zones    : {self.zones.num_zones}",
            f"  color scheme : {self.color_scheme}",
        ]
        if self.lactate_threshold.pace_str:
            lines.append(f"  LT pace      : {self.lactate_threshold.pace_str}")
        if self.lactate_threshold.heart_rate:
            lines.append(f"  LT heart rate: {self.lactate_threshold.heart_rate} bpm")
        lines.append("")
        for znum, (lo, hi, name) in self.zones.as_dict().items():
            lines.append(f"  {name:18s}  {lo:.0f}–{hi:.0f} bpm  {self.zone_colors.get(znum,'')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_zones(max_hr: int, resting_hr: int) -> ZoneConfig:
    n = len(_DEFAULT_ZONE_BOUNDARIES_PCT) - 1  # 5
    bpm = [max_hr * p for p in _DEFAULT_ZONE_BOUNDARIES_PCT]
    return ZoneConfig(
        num_zones=n,
        boundaries_bpm=bpm,
        names=_DEFAULT_ZONE_NAMES[:n],
        method="max_pct",
    )


def _resolve_colors(scheme: str, n: int) -> Dict[int, str]:
    if scheme == "hsv":
        return _hsv_colors(n)
    elif scheme == "matplotlib":
        return {k: v for k, v in list(_COLORS_MATPLOTLIB.items())[:n]}
    else:  # default
        return {k: v for k, v in list(_COLORS_DEFAULT.items())[:n]}


def _build_zones(raw: dict, max_hr: int, resting_hr: int) -> ZoneConfig:
    """Parse the [heart_rate_zones] section of the YAML."""
    method = raw.get("method", "max_pct").lower()
    raw_boundaries = raw.get("boundaries", _DEFAULT_ZONE_BOUNDARIES_PCT)
    raw_names = raw.get("names", None)

    num_zones = len(raw_boundaries) - 1
    if num_zones < 1:
        raise ValueError(
            f"heart_rate_zones.boundaries must have at least 2 values "
            f"(got {len(raw_boundaries)}). "
            f"Provide N+1 boundary values for N zones."
        )

    # Resolve to absolute bpm
    if method == "absolute":
        boundaries_bpm = [float(b) for b in raw_boundaries]
    elif method == "hrr":
        hrr = max_hr - resting_hr
        boundaries_bpm = [resting_hr + float(p) * hrr for p in raw_boundaries]
    else:  # max_pct
        boundaries_bpm = [max_hr * float(p) for p in raw_boundaries]

    # Zone names: fall back to Z1..ZN if not provided
    default_names = [
        "Z1 Recovery", "Z2 Aerobic", "Z3 Tempo",
        "Z4 Threshold", "Z5 Maximum", "Z6 Peak",
    ]
    if raw_names and len(raw_names) >= num_zones:
        names = [str(n) for n in raw_names[:num_zones]]
    else:
        names = default_names[:num_zones]

    return ZoneConfig(
        num_zones=num_zones,
        boundaries_bpm=boundaries_bpm,
        names=names,
        method=method,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Optional[str | Path] = None) -> Config:
    """
    Load configuration from a YAML file.

    Search order (stops at first match):
    1. `path` argument (if given)
    2. `./runtrackz.yml` in the current working directory
    3. `~/.runtrackz.yml` in the user's home directory

    If no file is found, returns a Config with sensible defaults.

    Parameters
    ----------
    path : str or Path, optional
        Explicit path to a runtrackz.yml file.

    Returns
    -------
    Config
    """
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    candidates += [
        Path.cwd() / "runtrackz.yml",
        Path.home() / ".runtrackz.yml",
    ]

    yml_path: Optional[Path] = None
    for p in candidates:
        if p.exists():
            yml_path = p
            break

    if yml_path is None:
        cfg = Config()
        cfg.zones = _default_zones(cfg.max_hr, cfg.resting_hr)
        cfg.zone_colors = _resolve_colors(cfg.color_scheme, cfg.zones.num_zones)
        return cfg

    with open(yml_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    # ── athlete ──────────────────────────────────────────────────────────
    athlete = raw.get("athlete", {})
    max_hr = int(athlete.get("max_hr", 185))
    resting_hr = int(athlete.get("resting_hr", 50))

    # ── heart rate zones ─────────────────────────────────────────────────
    hr_raw = raw.get("heart_rate_zones", {})
    zones = _build_zones(hr_raw, max_hr, resting_hr)

    # ── lactate threshold ────────────────────────────────────────────────
    lt_raw = raw.get("lactate_threshold", {})
    lt = LactateThreshold(
        pace_min_km=float(lt_raw["pace_min_km"]) if "pace_min_km" in lt_raw else None,
        heart_rate=int(lt_raw["heart_rate"]) if "heart_rate" in lt_raw else None,
    )

    # ── charts ───────────────────────────────────────────────────────────
    charts_raw = raw.get("charts", {})
    color_scheme = str(charts_raw.get("color_scheme", "default")).lower()
    if color_scheme not in ("default", "hsv", "matplotlib"):
        raise ValueError(
            f"charts.color_scheme must be 'default', 'hsv', or 'matplotlib' "
            f"(got '{color_scheme}')."
        )

    zone_colors = _resolve_colors(color_scheme, zones.num_zones)

    return Config(
        max_hr=max_hr,
        resting_hr=resting_hr,
        zones=zones,
        lactate_threshold=lt,
        color_scheme=color_scheme,
        zone_colors=zone_colors,
        source_file=yml_path,
    )
