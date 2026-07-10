# nav-testbed

An automated accuracy testing framework for optical surgical navigation tracking.

Surgical navigation systems track instruments in real time using fiducial
markers and report their position relative to patient anatomy. Before such a
system can be trusted in an OR, its tracking accuracy has to be validated
across many conditions (distance, viewing angle, occlusion, motion) with
repeatable, quantitative tests rather than one-off manual checks. This
project builds that kind of test harness around an ArUco-marker pose
tracker.

## What it does

- **Tracking module** (`nav_testbed.tracking`) — detects ArUco fiducial
  markers in an image and estimates their 6-DOF pose (rotation + translation)
  relative to the camera using OpenCV's marker detector and `solvePnP`.
- **Synthetic ground truth** — since real tracking hardware isn't always
  available, the same module can render a marker at a *known* pose. That
  gives an exact ground truth to test detection against, without needing a
  calibration rig.
- **Accuracy statistics** (`nav_testbed.stats`) — position error (mm),
  rotation error (deg), RMSE, and 95% confidence intervals over a batch of
  measurements.
- **Automated parameter study** (`scripts/run_parameter_study.py`) — sweeps
  marker distance and tilt angle, measures tracking error at each condition,
  and writes a CSV report plus a summary accuracy-vs-distance plot.
- **Test suite** (`tests/`) — pytest tests that render markers at known poses
  and assert the recovered *position* stays within a distance/angle-dependent
  tolerance, so accuracy regressions are caught automatically in CI. (Rotation
  is measured and reported, not hard-asserted — see Known limitations.)

## Known limitations

**Monocular planar-marker orientation is fundamentally ambiguous.** Recovering
6-DOF pose from a single flat marker in a single camera view is a two-valued
problem: a true pose and a mirror-flipped "ghost" pose can produce nearly
identical reprojection error, so a solver has no way to reliably tell them
apart from one image alone. `scripts/run_parameter_study.py` measures and
reports this directly — in a typical run roughly 40% of tilt/distance
conditions lock onto the wrong (flipped) orientation, while position stays
accurate throughout. `nav_testbed.tracking.detect_pose_candidates` exposes
both ranked candidate poses for exactly this reason.

This isn't a bug to "fix" in software; it's why real optical trackers avoid
relying on a single planar marker for orientation — e.g. NDI Polaris-style
tools use rigid, non-planar constellations of 3+ markers, and some setups use
stereo cameras or temporal filtering across frames instead. A production
version of this testbed would need one of those (see Roadmap).

## Why this design

Real surgical navigation testing (e.g. with an NDI Polaris or similar
optical tracker) validates accuracy the same way conceptually: place a
tracked object at known reference positions, measure what the system
reports, and quantify the error statistically across a range of conditions.
This project reproduces that workflow with a webcam-friendly ArUco tracker
so the full pipeline — rendering/measuring, statistics, automated
parameter sweeps, CI — can be built and tested without lab hardware.

## Getting started

```bash
pip install -e ".[dev]"
pytest -v
python scripts/run_parameter_study.py
```

The parameter study writes `results/parameter_study.csv` and
`results/parameter_study.png`.

## Roadmap

- Resolve the orientation ambiguity with a rigid multi-marker constellation
  or a second (stereo) camera, and add regression tests for orientation
  accuracy once it's well-conditioned
- Wrap the tracker as ROS2 nodes (pose publisher + TF2 frames) for
  integration with a robot-arm ground truth reference
- Use a robot arm to move a real marker to known poses for hardware-in-the-loop
  accuracy testing (not just synthetic ground truth)
- Add occlusion and motion-blur test conditions to the parameter study
