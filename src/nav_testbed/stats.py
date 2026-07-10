"""Accuracy statistics for comparing estimated poses against ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PoseError:
    position_error_mm: float
    rotation_error_deg: float


def _rotation_angle_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    r_a, _ = cv2.Rodrigues(rvec_a)
    r_b, _ = cv2.Rodrigues(rvec_b)
    r_err = r_a @ r_b.T
    cos_angle = (np.trace(r_err) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_pose_errors(
    estimated: list[tuple[np.ndarray, np.ndarray]],
    ground_truth: list[tuple[np.ndarray, np.ndarray]],
) -> list[PoseError]:
    """Pairwise error between estimated (rvec, tvec) poses and ground truth,
    in the same order. tvec is assumed to be in meters."""
    if len(estimated) != len(ground_truth):
        raise ValueError("estimated and ground_truth must be the same length")

    errors = []
    for (rvec_est, tvec_est), (rvec_gt, tvec_gt) in zip(estimated, ground_truth):
        position_error_mm = float(np.linalg.norm(tvec_est - tvec_gt) * 1000.0)
        rotation_error_deg = _rotation_angle_deg(rvec_est, rvec_gt)
        errors.append(PoseError(position_error_mm, rotation_error_deg))
    return errors


def summarize_errors(errors: list[PoseError]) -> dict:
    """Mean, std, RMSE and 95% CI (normal approximation) for position and
    rotation error across a batch of measurements."""
    if not errors:
        raise ValueError("errors is empty")

    pos = np.array([e.position_error_mm for e in errors])
    rot = np.array([e.rotation_error_deg for e in errors])
    n = len(errors)

    def _summary(values: np.ndarray) -> dict:
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        rmse = float(np.sqrt(np.mean(values**2)))
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        return {
            "mean": mean,
            "std": std,
            "rmse": rmse,
            "ci95_low": mean - ci95,
            "ci95_high": mean + ci95,
            "n": n,
        }

    return {
        "position_error_mm": _summary(pos),
        "rotation_error_deg": _summary(rot),
    }
