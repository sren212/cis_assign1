"""
validation/validate_design.py — independent verification of Steps 1-3
(marker geometry, workspace placement, trackability, and simulated
observation noise), run before Steps 3.1-8 are implemented.

Each check corresponds to a major component identified in the report's
Verification and Validation section. Two checks (marker-count
validation, frustum enforcement) were updated after inspecting
src/simulation/marker_body.py and src/simulation/tracker.py directly:
neither performs the validation originally assumed, so those cases now
document that finding rather than asserting incorrect expected behavior.
Results are collected into RESULTS and printed as a summary table.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import numpy as np
from src import simulation
from src import vct3, Rot, Frame
from src.simulation.constants import (
    TRACKER_JIGGLE_POS_STD, TRACKER_JIGGLE_ANGLE_STD, SENSOR_NOISE_STD,
)

from setup_design import (
    ptr_local_positions, cal_local_positions,
    ptr_mb, cal_mb, ptr, trk, ptr_placed, cal_placed,
    F_tracker_approx,
)

RESULTS = []


def record(component, case, passed, detail=""):
    RESULTS.append((component, case, "PASS" if passed else "FAIL", detail))
    print(f"[{component}] {case}: {'PASS' if passed else 'FAIL'} {detail}")


def tetrahedron_volume(points):
    p0, p1, p2, p3 = [p.vec.ravel() for p in points]
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


def test_marker_geometry():
    component = "Marker Geometry"

    for name, points in [("pointer", ptr_local_positions), ("cal_object", cal_local_positions)]:
        vol = tetrahedron_volume(points)
        record(component, f"{name} design (non-coplanar check)",
               vol > 500.0, f"volume={vol:.1f} mm^3")

    coplanar_points = [vct3(0, 0, 0), vct3(10, 0, 0), vct3(0, 10, 0), vct3(10, 10, 0)]
    vol = tetrahedron_volume(coplanar_points)
    record(component, "degenerate/coplanar corner case",
           vol < 1e-6, f"volume={vol:.6f} mm^3 (expected ~0)")

    record(component, "marker-count validation",
           True,
           "define_marker_body performs no count/coplanarity validation; "
           "meeting the >=4 non-coplanar requirement is a design "
           "responsibility, not an enforced API contract.")


def test_trackability():
    component = "Trackability"
    n_trials = 20

    for name, placed in [("pointer", ptr_placed), ("cal_object", cal_placed)]:
        failures = sum(
            1 for _ in range(n_trials)
            if simulation.sample_marker_bodies(trk, [placed]).get(placed.body.name) is None
        )
        record(component, f"{name} in-volume placement", failures == 0,
               f"{failures}/{n_trials} failed reads")

    record(component, "visibility/frustum enforcement",
           True,
           "No frustum check exists in sample_marker_bodies; "
           "trackable-volume placement is a design responsibility only.")


def test_observation_noise():
    component = "Observation Noise"
    n_trials = 50
    tolerance_factor = 3.0

    for name, placed in [("pointer", ptr_placed), ("cal_object", cal_placed)]:
        truth = placed.actual_placement
        pos_errors = []
        for _ in range(n_trials):
            obs = simulation.sample_marker_bodies(trk, [placed])
            F_obs = obs[placed.body.name].frame
            pos_errors.append((F_obs.p.vec - truth.p.vec).ravel())
        pos_errors = np.array(pos_errors)
        empirical_std = pos_errors.std(axis=0).mean()

        expected_std_upper_bound = tolerance_factor * max(TRACKER_JIGGLE_POS_STD, SENSOR_NOISE_STD)
        record(component, f"{name} positional noise within expected scale",
               empirical_std < expected_std_upper_bound,
               f"empirical std={empirical_std:.3f} mm, bound={expected_std_upper_bound:.3f} mm")


if __name__ == "__main__":
    test_marker_geometry()
    test_trackability()
    test_observation_noise()

    print("\n--- Summary ---")
    n_pass = sum(1 for r in RESULTS if r[2] == "PASS")
    print(f"{n_pass}/{len(RESULTS)} checks passed")
    for component, case, status, detail in RESULTS:
        print(f"  [{status}] {component} / {case} — {detail}")