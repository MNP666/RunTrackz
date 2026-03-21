"""
runtrackz.run_type
~~~~~~~~~~~~~~~~~~
Run type constants and classification helpers.

Run types
---------
easy        Recovery or general aerobic run.
long_run    Long slow distance — typically the longest run of the week,
            at a comfortable aerobic pace.
tempo       Sustained effort at or near lactate threshold pace/HR for a
            significant portion of the run.
workout     Structured session with discrete intervals, speed work, or
            fartlek efforts — pace varies significantly.
race        Competitive event.
unknown     Not yet classified (default when no type is given).

Usage
-----
The run type is stored as a plain string in the database ``run_type`` column.
Use the constants here rather than string literals to avoid typos::

    import runtrackz.run_type as rt

    if run_type == rt.WORKOUT:
        stats = runtrackz.workout_analysis.analyze(run, hr, pace, cfg)

Automatic classification
------------------------
``classify()`` provides heuristic-based guesses.  The heuristics are stubs
for now — they will be calibrated once you have a labelled set of runs to
validate against.  Manual tagging via ``--run-type`` is the recommended
approach until then.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from runtrackz.models import RunData
    from runtrackz.hr_analysis import HRStats
    from runtrackz.pace_analysis import PaceStats
    from runtrackz.config import Config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EASY     = 'easy'
LONG_RUN = 'long_run'
TEMPO    = 'tempo'
WORKOUT  = 'workout'
RACE     = 'race'
UNKNOWN  = 'unknown'

ALL_TYPES = (EASY, LONG_RUN, TEMPO, WORKOUT, RACE, UNKNOWN)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(
    run: 'RunData',
    hr_stats: Optional['HRStats'] = None,
    pace_stats: Optional['PaceStats'] = None,
    config: Optional['Config'] = None,
) -> str:
    """
    Heuristic run type classification.

    Returns one of the module-level constants: ``EASY``, ``LONG_RUN``,
    ``TEMPO``, ``WORKOUT``, or ``UNKNOWN``.

    .. note::
        These heuristics are intentional stubs.  They will produce reasonable
        guesses for obvious cases, but should not be trusted until validated
        against a labelled dataset.  Manual tagging (``--run-type`` flag in
        ``process_runs.py``) is more reliable for now.

    Parameters
    ----------
    run : RunData
        Parsed run.
    hr_stats : HRStats, optional
        Pre-computed HR stats — avoids re-analysis if already available.
    pace_stats : PaceStats, optional
        Pre-computed pace stats — avoids re-analysis if already available.
    config : Config, optional
        Personal config, used for LT thresholds when available.

    Returns
    -------
    str
        One of the run type constants.
    """
    df = run.df

    # ── Derive basic metrics ─────────────────────────────────────────────
    distance_km = df['distance_m'].max() / 1000 if 'distance_m' in df else 0
    duration_min = df['elapsed_s'].max() / 60 if 'elapsed_s' in df else 0

    pace_col = df['pace_min_km'].dropna()
    pace_cv = (pace_col.std() / pace_col.mean()) if len(pace_col) > 10 else 0

    avg_hr = hr_stats.avg_hr if hr_stats is not None else df['heart_rate'].mean()
    max_hr_cfg = config.max_hr if config is not None else None

    # ── Heuristics (to be refined with real data) ────────────────────────

    # Workout: high pace variability suggests intervals or fartlek
    # TODO: replace with proper interval-detection once labelled data is available
    if pace_cv > 0.15:
        return WORKOUT

    # Long run: distance-based threshold (adjust to your own training context)
    # TODO: make the threshold configurable via runtrackz.yml
    if distance_km >= 16:
        return LONG_RUN

    # Tempo: sustained high HR relative to max, low pace variability
    # TODO: use LT HR from config once confirmed reliable
    if max_hr_cfg is not None and avg_hr is not None:
        hr_pct = avg_hr / max_hr_cfg
        if hr_pct >= 0.82 and pace_cv < 0.08:
            return TEMPO

    return UNKNOWN
