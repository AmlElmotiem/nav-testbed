"""ArUco-marker 6-DOF pose tracking, plus a synthetic renderer used to
generate images with a known ground-truth pose for accuracy testing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# Marker corner order used by cv2.aruco (top-left, top-right, bottom-right,
# bottom-left), expressed as 3D object points in the marker's own frame.
def _object_points(marker_length: float) -> np.ndarray:
    half = marker_length / 2.0
    return np.array(
        [
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
            [-half, -half, 0],
        ],
        dtype=np.float32,
    )


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float64,
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.zeros(5, dtype=np.float64)


def default_camera(width: int = 1280, height: int = 720) -> Camera:
    """A plausible pinhole camera (~90 deg horizontal FOV)."""
    fx = fy = width / 2.0
    return Camera(width=width, height=height, fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0)


def render_synthetic_marker(
    marker_id: int,
    marker_length: float,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera: Camera,
    marker_px: int = 220,
    pixel_noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Render a grayscale image of a single ArUco marker placed at the given
    pose (rvec/tvec, marker frame -> camera frame) as seen by `camera`.

    This gives us ground truth: we know exactly where the marker "really is"
    because we placed it there, so we can measure detection error precisely.

    `pixel_noise_std` optionally adds sensor noise. Note it's off by default:
    at this marker's pixel resolution, even mild noise can corrupt the bit
    sampling enough to break detection outright for small/tilted markers,
    which is a worse problem than the tie-breaking it might help with (see
    detect_pose_candidates for the actual planar-pose ambiguity handling).
    """
    base = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, marker_px)
    # Natural raster winding (top-left, top-right, bottom-right, bottom-left),
    # matching _object_points()'s corner order under OpenCV's own ArUco
    # pose convention.
    src = np.array(
        [[0, 0], [marker_px, 0], [marker_px, marker_px], [0, marker_px]],
        dtype=np.float32,
    )

    # _object_points() traces its corners counter-clockwise as seen from +Z,
    # so under an identity rotation the marker's printed-face normal (+Z)
    # points away from the camera -- i.e. rvec=0 shows the *back* of the
    # marker. Composing a fixed 180 degree flip about Y turns the object to
    # face the camera at rvec=0, so `rvec` behaves as the intuitive tilt
    # relative to a camera-facing marker (and the render stays a proper
    # rigid rotation, so the ArUco pattern isn't mirrored/invalid).
    flip_180_about_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    r_user, _ = cv2.Rodrigues(rvec)
    r_eff, _ = cv2.Rodrigues(r_user @ flip_180_about_y)

    obj_points = _object_points(marker_length)
    dst, _ = cv2.projectPoints(obj_points, r_eff, tvec, camera.matrix, camera.dist_coeffs)
    dst = dst.reshape(-1, 2).astype(np.float32)

    canvas = np.full((camera.height, camera.width), 255, dtype=np.uint8)
    homography = cv2.getPerspectiveTransform(src, dst)
    warped_marker = cv2.warpPerspective(
        base, homography, (camera.width, camera.height), borderValue=255
    )
    # Composite only inside the warped marker's footprint; leave background white.
    footprint = cv2.warpPerspective(
        np.full_like(base, 255), homography, (camera.width, camera.height), borderValue=0
    )
    canvas = np.where(footprint > 0, warped_marker, canvas)

    if pixel_noise_std > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, pixel_noise_std, size=canvas.shape)
        canvas = np.clip(canvas.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return canvas


def _jitter_corners(img_points: np.ndarray, std_px: float, seed: int) -> np.ndarray:
    """Add small Gaussian jitter to detected corner positions.

    For a perfectly noiseless synthetic render, the true pose and its
    mirror-flipped "ghost" (see detect_pose_candidates) can have *exactly*
    equal (zero) reprojection error -- a genuine mathematical degeneracy of
    single-view planar pose estimation, not a detection error. Which one
    "wins" then comes down to floating-point noise from unrelated prior
    computation in the process, which is why this failure mode looked flaky
    across runs during development. Real corner detection always has some
    sub-pixel localization uncertainty; modeling that (rather than pretending
    corners are exact) reproduces realistic, stable disambiguation behavior.
    """
    if std_px <= 0:
        return img_points
    rng = np.random.default_rng(seed)
    return img_points + rng.normal(0.0, std_px, img_points.shape).astype(np.float32)


def detect_pose(
    image: np.ndarray,
    marker_length: float,
    camera: Camera,
    corner_noise_std_px: float = 0.0,
    seed: int = 0,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Detect ArUco markers in `image` and estimate each one's 6-DOF pose.

    Returns {marker_id: (rvec, tvec)}.
    """
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)

    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if ids is None:
        return poses

    obj_points = _object_points(marker_length)
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        img_points = marker_corners.reshape(-1, 2).astype(np.float32)
        img_points = _jitter_corners(img_points, corner_noise_std_px, seed + int(marker_id))
        # IPPE_SQUARE is specifically designed for 4 coplanar (square) points
        # and avoids the flipped/mirrored orientation that the generic
        # iterative solver can converge to for near-fronto-parallel markers.
        ok, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera.matrix,
            camera.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if ok:
            poses[int(marker_id)] = (rvec, tvec)
    return poses


def detect_pose_candidates(
    image: np.ndarray,
    marker_length: float,
    camera: Camera,
    corner_noise_std_px: float = 0.0,
    seed: int = 0,
) -> dict[int, list[tuple[np.ndarray, np.ndarray, float]]]:
    """Like detect_pose, but returns *all* IPPE_SQUARE candidate poses per
    marker (there are up to two, ranked by reprojection error), instead of
    just the best one.

    Coplanar 4-point pose estimation is fundamentally two-valued at low tilt
    angles: a true pose and a mirror-flipped "ghost" pose can have nearly
    identical reprojection error, so the single best-ranked solution (what
    `detect_pose` returns, and what a deployed system would have to rely on)
    is occasionally wrong. This function exposes both candidates so a test
    harness with ground truth can measure how often that happens, which is
    exactly the kind of failure mode a real navigation system has to
    characterize (and typically mitigate with stereo cameras, multiple
    markers, or temporal filtering).

    Returns {marker_id: [(rvec, tvec, reprojection_error_px), ...]},
    sorted by ascending reprojection error.
    """
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)

    candidates: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = {}
    if ids is None:
        return candidates

    obj_points = _object_points(marker_length)
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        img_points = marker_corners.reshape(-1, 2).astype(np.float32)
        img_points = _jitter_corners(img_points, corner_noise_std_px, seed + int(marker_id))
        _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            obj_points,
            img_points,
            camera.matrix,
            camera.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        candidates[int(marker_id)] = [
            (rvec, tvec, float(err)) for rvec, tvec, err in zip(rvecs, tvecs, errors.flatten())
        ]
    return candidates
