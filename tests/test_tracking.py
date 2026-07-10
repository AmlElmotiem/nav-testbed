import numpy as np
import pytest

from nav_testbed.tracking import (
    default_camera,
    detect_pose,
    detect_pose_candidates,
    render_synthetic_marker,
)

MARKER_LENGTH = 0.05  # 5 cm marker


@pytest.mark.parametrize("distance_m", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("tilt_deg", [0, 20, 45])
def test_position_recovered_within_tolerance(distance_m, tilt_deg):
    # Beyond ~1m the marker projects to <30px with this camera/marker size
    # and PnP accuracy degrades sharply (see scripts/run_parameter_study.py,
    # which sweeps out to 2m specifically to characterize that falloff).
    # This test only asserts the range where tracking is meant to be reliable.
    camera = default_camera()
    rvec_gt = np.array([[np.radians(tilt_deg)], [0.0], [0.0]])
    tvec_gt = np.array([[0.0], [0.0], [distance_m]])

    image = render_synthetic_marker(0, MARKER_LENGTH, rvec_gt, tvec_gt, camera)
    poses = detect_pose(image, MARKER_LENGTH, camera)

    assert 0 in poses, f"marker not detected at distance={distance_m}m tilt={tilt_deg}deg"
    _, tvec_est = poses[0]

    position_error_mm = np.linalg.norm(tvec_est - tvec_gt) * 1000.0
    # Error grows roughly with distance^2 as the marker shrinks in pixels;
    # tilt adds a smaller additional penalty from foreshortening.
    tolerance_mm = 5.0 + 45.0 * (distance_m**2) + 0.3 * tilt_deg
    assert position_error_mm < tolerance_mm


def test_pose_candidates_returns_two_ranked_solutions():
    """Coplanar 4-point pose estimation is fundamentally two-valued (a true
    pose and a mirror-flipped "ghost" pose can have near-identical
    reprojection error -- see detect_pose_candidates' docstring, and
    scripts/run_parameter_study.py's orientation-ambiguity notes for the
    empirical characterization). This only checks the structural contract
    (two ranked candidates come back for a detected marker); it doesn't
    assert which one is closer to ground truth, because which one wins is
    genuinely not reliably predictable from a single monocular view.
    """
    camera = default_camera()
    rvec_gt = np.array([[np.radians(30)], [0.0], [0.0]])
    tvec_gt = np.array([[0.0], [0.0], [0.5]])

    image = render_synthetic_marker(0, MARKER_LENGTH, rvec_gt, tvec_gt, camera)
    candidates = detect_pose_candidates(image, MARKER_LENGTH, camera)

    assert 0 in candidates
    assert len(candidates[0]) == 2
    errors = [err for _, _, err in candidates[0]]
    assert errors == sorted(errors)  # ranked ascending by reprojection error


def test_no_marker_detected_in_blank_image():
    camera = default_camera()
    blank = np.full((camera.height, camera.width), 255, dtype="uint8")
    assert detect_pose(blank, MARKER_LENGTH, camera) == {}
