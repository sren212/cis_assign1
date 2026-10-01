# assignment1/setup_design.py
"""Shared Step 1-3 setup: marker-body design and simulation wiring.
Imported by both the solution notebook and validation scripts to avoid
duplicating the design definitions in two places.
"""
from src import simulation
from src import vct3, Rot, Frame

ptr_local_positions = [
    vct3(20, 20, 20),
    vct3(20, -20, -20),
    vct3(-20, 20, -20),
    vct3(-20, -20, 20),
]
ptr_tip_approx = vct3(0, 0, -100)

cal_local_positions = [
    vct3(20, 20, 20),
    vct3(20, -20, -20),
    vct3(-20, 20, -20),
    vct3(-20, -20, 20),
]

F_tracker_approx = Frame(Rot(), vct3(0.0, 0.0, 0.0))
F_cal_approx = Frame(Rot.xyz(0.0, 0.0, 0.0), vct3(300.0, 300.0, 1400.0))

ptr_mb = simulation.define_marker_body("ptr_mb", ptr_local_positions)
cal_mb = simulation.define_marker_body("cal_mb", cal_local_positions)

cal_placed = simulation.place_marker_body(cal_mb, F_cal_approx)
ptr_placed = simulation.place_marker_body(ptr_mb, Frame.eye(), precision="precise")

ptr = simulation.define_pointer("ptr1", ptr_tip_approx, ptr_mb)
trk = simulation.create_tracker("trk1", F_tracker_approx)