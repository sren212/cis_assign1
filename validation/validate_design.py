"""
validate_design.py — independent sanity checks for Steps 1-3, runnable
before pivot-calibration logic (Steps 3.1/4-8) is implemented.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import numpy as np
from src import simulation
from src import vct3, Rot, Frame

from notebook_or_module import (  # replace with actual import of your Step1-3 cells
    ptr_local_positions, ptr_tip_approx,
    cal_local_positions,
    F_tracker_approx, F_cal_approx,
    ptr_mb, cal_mb, ptr, trk, ptr_placed, cal_placed,
)


def tetrahedron_volume(points):
    """Signed volume of the tetrahedron formed by 4 vct3 points; ~0 => coplanar."""
    p0, p1, p2, p3 = [p.vec.ravel() for p in points]
    return np.abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


def check_marker_geometry(name, points, min_volume_mm3=500.0):
    vol = tetrahedron_volume(points)
    spread = max(np.linalg.norm(p.vec.ravel() - q.vec.ravel())
                 for p in points for q in points)
    print(f"[{name}] tetrahedron volume = {vol:.1f} mm^3, max spread = {spread:.1f} mm")
    assert vol > min_volume_mm3, f"{name}: markers too close to coplanar"


def check_trackability(placed, trk, n_trials=20):
    """Repeatedly sample a placed body and confirm the tracker returns a
    valid reading each time (i.e. the body's true position stays inside
    the tracker's usable volume given its jiggle/noise)."""
    failures = 0
    for _ in range(n_trials):
        obs = simulation.sample_marker_bodies(trk, [placed])
        if obs.get(placed.body.name) is None:
            failures += 1
    print(f"[{placed.body.name}] {failures}/{n_trials} failed observations")
    assert failures == 0, "body is not reliably trackable at this placement"


def check_observation_noise(placed, trk, n_trials=50):
    """Compare empirical scatter of observed frames against the ground-truth
    actual_placement, as a rough check that noise magnitudes are sane."""
    truth = placed.actual_placement
    pos_errors, ang_errors = [], []
    for _ in range(n_trials):
        obs = simulation.sample_marker_bodies(trk, [placed])
        F_obs = obs[placed.body.name].frame
        pos_errors.append((F_obs.p.vec - truth.p.vec).ravel())
        R_err = F_obs.R.matrix @ truth.R.matrix.T
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        ang_errors.append(angle)
    pos_errors = np.array(pos_errors)
    print(f"[{placed.body.name}] pos noise std (mm): {pos_errors.std(axis=0)}")
    print(f"[{placed.body.name}] angular noise std (rad): {np.std(ang_errors):.4f}")


if __name__ == "__main__":
    check_marker_geometry("pointer", ptr_local_positions)
    check_marker_geometry("cal_object", cal_local_positions)

    check_trackability(cal_placed, trk)
    check_trackability(ptr_placed, trk)

    check_observation_noise(cal_placed, trk)
    check_observation_noise(ptr_placed, trk)

    print("Step 1-3 validation complete.")