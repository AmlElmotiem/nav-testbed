import numpy as np
import pytest

from nav_testbed.stats import compute_pose_errors, summarize_errors


def test_zero_error_for_identical_poses():
    rvec = np.array([[0.1], [0.2], [0.3]])
    tvec = np.array([[0.0], [0.0], [0.5]])
    errors = compute_pose_errors([(rvec, tvec)], [(rvec, tvec)])

    assert errors[0].position_error_mm == 0.0
    assert errors[0].rotation_error_deg < 1e-4  # floating-point noise only


def test_known_position_offset():
    rvec = np.zeros((3, 1))
    tvec_gt = np.array([[0.0], [0.0], [0.5]])
    tvec_est = np.array([[0.0], [0.0], [0.51]])  # 10 mm off in Z

    errors = compute_pose_errors([(rvec, tvec_est)], [(rvec, tvec_gt)])

    assert errors[0].position_error_mm == pytest.approx(10.0)


def test_summary_matches_manual_computation():
    errors = compute_pose_errors(
        [(np.zeros((3, 1)), np.array([[0.0], [0.0], [z]])) for z in (0.50, 0.51, 0.49)],
        [(np.zeros((3, 1)), np.array([[0.0], [0.0], [0.5]]))] * 3,
    )
    summary = summarize_errors(errors)

    pos = summary["position_error_mm"]
    assert pos["n"] == 3
    assert pos["mean"] == pytest.approx((0 + 10 + 10) / 3)
