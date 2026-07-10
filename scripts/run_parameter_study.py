"""Automated accuracy parameter study for the ArUco tracking pipeline.

Sweeps marker distance and tilt angle, measures pose error against known
ground truth at each condition, and writes a CSV report plus a summary plot.
Run: python scripts/run_parameter_study.py

Position error is a reliable, well-conditioned metric across this sweep.
Rotation error is not: single-camera planar-marker pose estimation is
fundamentally two-valued (a true pose and a mirror-flipped "ghost" pose can
have near-identical reprojection error), so `detect_pose`'s single answer
periodically locks onto the wrong one -- this is a real, known limitation of
monocular fiducial tracking, not a bug in this pipeline. This script reports
the resulting flip rate as a finding rather than hiding it; production
systems mitigate it with stereo cameras, non-planar marker constellations,
or temporal filtering (see README "Known limitations").
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nav_testbed.stats import compute_pose_errors, summarize_errors
from nav_testbed.tracking import default_camera, detect_pose, render_synthetic_marker

MARKER_LENGTH = 0.05
DISTANCES_M = [0.3, 0.6, 1.0, 1.5, 2.0]
TILTS_DEG = [0, 15, 30, 45, 60]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"


def run_condition(distance_m: float, tilt_deg: float) -> dict | None:
    camera = default_camera()
    rvec_gt = np.array([[np.radians(tilt_deg)], [0.0], [0.0]])
    tvec_gt = np.array([[0.0], [0.0], [distance_m]])

    image = render_synthetic_marker(0, MARKER_LENGTH, rvec_gt, tvec_gt, camera)
    poses = detect_pose(image, MARKER_LENGTH, camera)
    if 0 not in poses:
        return None

    rvec_est, tvec_est = poses[0]
    errors = compute_pose_errors([(rvec_est, tvec_est)], [(rvec_gt, tvec_gt)])
    summary = summarize_errors(errors)
    return {
        "distance_m": distance_m,
        "tilt_deg": tilt_deg,
        "position_error_mm": summary["position_error_mm"]["mean"],
        "rotation_error_deg": summary["rotation_error_deg"]["mean"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []
    for distance_m in DISTANCES_M:
        for tilt_deg in TILTS_DEG:
            result = run_condition(distance_m, tilt_deg)
            if result is None:
                print(f"[MISS] distance={distance_m}m tilt={tilt_deg}deg: marker not detected")
                continue
            rows.append(result)
            print(
                f"distance={distance_m:>4.1f}m tilt={tilt_deg:>3}deg  "
                f"pos_err={result['position_error_mm']:6.2f}mm  "
                f"rot_err={result['rotation_error_deg']:5.2f}deg"
            )

    csv_path = OUTPUT_DIR / "parameter_study.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    flips = sum(1 for r in rows if r["rotation_error_deg"] > 90.0)
    print(
        f"Orientation flip rate: {flips}/{len(rows)} conditions locked onto the "
        f"mirror-ambiguous pose (rotation_error_deg > 90) -- see module docstring."
    )

    _plot(rows)


def _plot(rows: list[dict]) -> None:
    fig, (ax_pos, ax_rot) = plt.subplots(1, 2, figsize=(11, 4.5))
    for tilt_deg in sorted({r["tilt_deg"] for r in rows}):
        subset = sorted((r for r in rows if r["tilt_deg"] == tilt_deg), key=lambda r: r["distance_m"])
        distances = [r["distance_m"] for r in subset]
        ax_pos.plot(distances, [r["position_error_mm"] for r in subset], marker="o", label=f"{tilt_deg} deg")
        ax_rot.plot(distances, [r["rotation_error_deg"] for r in subset], marker="o", label=f"{tilt_deg} deg")

    ax_pos.set_xlabel("Distance (m)")
    ax_pos.set_ylabel("Position error (mm)")
    ax_pos.set_title("Position accuracy vs. distance")
    ax_pos.legend(title="Tilt")
    ax_pos.grid(True, alpha=0.3)

    ax_rot.set_xlabel("Distance (m)")
    ax_rot.set_ylabel("Rotation error (deg)")
    ax_rot.set_title("Rotation accuracy vs. distance")
    ax_rot.legend(title="Tilt")
    ax_rot.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "parameter_study.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
