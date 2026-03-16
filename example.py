"""
example.py — runtrackz usage demonstration
-------------------------------------------
Run this script from the project root:

    python example.py path/to/your_run.fit

It will print a summary and save an overview chart as overview.png.
"""

import sys
from pathlib import Path

# Make sure runtrackz is importable from this directory
sys.path.insert(0, str(Path(__file__).parent))

import runtrackz

def main(fit_path: str):
    # ── 1. Load the run ──────────────────────────────────────────────────
    print(f"\nLoading: {fit_path}")
    run = runtrackz.load(fit_path)
    print(f"  → {run}")
    print(f"\nDataFrame columns: {list(run.df.columns)}")
    print(run.df.head(3).to_string())

    # ── 2. Heart rate analysis ───────────────────────────────────────────
    print("\n" + "="*60)
    # Set your actual max HR here; if None, the observed max is used
    hr_stats = runtrackz.hr_analysis.analyze(
        run,
        max_hr=185,        # adjust to your max HR
        resting_hr=50,     # needed only for method='hrr'
        method='max_pct',  # 'max_pct' or 'hrr'
    )
    print(hr_stats.summary())

    # Zone breakdown as DataFrame
    print("\nZone DataFrame:")
    print(hr_stats.to_dataframe().to_string(index=False))

    # ── 3. Pace analysis ─────────────────────────────────────────────────
    print("\n" + "="*60)
    pace_stats = runtrackz.pace_analysis.analyze(
        run,
        split_distance_m=1000,  # per km
    )
    print(pace_stats.summary())

    # Splits as DataFrame
    print("\nSplits DataFrame:")
    print(pace_stats.splits_dataframe().to_string(index=False))

    # ── 4. Charts ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Generating charts...")

    # 4a. Overview (4-panel)
    fig = runtrackz.charts.overview(run, hr_stats=hr_stats, pace_stats=pace_stats)
    out = Path(fit_path).with_suffix('.overview.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")

    # 4b. Splits bar chart
    fig2 = runtrackz.charts.splits_bar(pace_stats)
    out2 = Path(fit_path).with_suffix('.splits.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out2}")

    # 4c. HR zone bar
    fig3 = runtrackz.charts.hr_zone_bar(hr_stats)
    out3 = Path(fit_path).with_suffix('.hr_zones.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out3}")

    print("\nDone!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python example.py path/to/run.fit")
        sys.exit(1)
    main(sys.argv[1])
