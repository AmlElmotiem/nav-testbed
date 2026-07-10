"""Live webcam test of the tracking pipeline against a real, printed marker.

Print docs/marker_to_print.png, measure the actual black square's side
length with a ruler (in meters), and set MARKER_LENGTH_M below to that
measurement -- accuracy depends on this being correct.

Run: python scripts/live_camera_demo.py
Press 'q' or Esc in the video window to quit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nav_testbed.tracking import Camera, detect_pose

MARKER_LENGTH_M = 0.10  # measure your printed marker's black square and update this
CAMERA_INDEX = 0


def guess_camera(width: int, height: int) -> Camera:
    """Rough camera intrinsics guess for a typical laptop webcam.

    Not calibrated -- good enough to see the pipeline work, not for
    trusting the exact millimeter numbers. See README for how a real
    system would calibrate this properly (checkerboard calibration).
    """
    fx = fy = width * 0.9
    return Camera(width=width, height=height, fx=fx, fy=fy, cx=width / 2, cy=height / 2)


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Could not read a frame from the camera")
        return
    height, width = frame.shape[:2]
    camera = guess_camera(width, height)
    print(f"Camera opened at {width}x{height}. Marker length assumed: {MARKER_LENGTH_M*100:.1f} cm")
    print("Press 'q' or Esc to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        poses = detect_pose(gray, MARKER_LENGTH_M, camera)

        if 0 in poses:
            rvec, tvec = poses[0]
            x, y, z = tvec.flatten()
            cv2.drawFrameAxes(frame, camera.matrix, camera.dist_coeffs, rvec, tvec, MARKER_LENGTH_M * 0.5)
            text = f"x={x*100:5.1f}cm  y={y*100:5.1f}cm  z={z*100:5.1f}cm"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "marker not detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("nav-testbed live camera demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
