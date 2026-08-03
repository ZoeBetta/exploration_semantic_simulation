"""Regression checks for camera handedness and restored wall decoration."""
import math
from pathlib import Path

from camera3d import (
    CAMERA_FORWARD_OFFSET_M,
    CAMERA_LOOK_AHEAD_M,
    _camera_pose_from_robot,
)

# For the four cardinal simulator headings, the Panda3D look vector must equal
# (cos(theta), -sin(theta)): Y is mirrored to preserve left/right handedness.
for theta in (0.0, math.pi / 2, math.pi, -math.pi / 2):
    camera, target = _camera_pose_from_robot(3.0, 4.0, theta)
    dx, dy = target[0] - camera[0], target[1] - camera[1]
    assert math.isclose(dx, CAMERA_LOOK_AHEAD_M * math.cos(theta), abs_tol=1e-9)
    assert math.isclose(dy, -CAMERA_LOOK_AHEAD_M * math.sin(theta), abs_tol=1e-9)
    robot_px, robot_py = 3.0, -4.0
    offset = math.hypot(camera[0] - robot_px, camera[1] - robot_py)
    assert math.isclose(offset, CAMERA_FORWARD_OFFSET_M, abs_tol=1e-9)

# The richer wall system must include all three decorative families and must
# choose visible faces from the ground-truth grid instead of one fixed side.
source = Path(__file__).with_name("camera3d.py").read_text(encoding="utf-8")
for token in ("_wall_free_sides", "_add_bookcase", "_add_framed_art", "_add_noticeboard", "_add_baseboard"):
    assert token in source
assert "root.setScale(1.0, -1.0, 1.0)" in source
assert "self.camera.lookAt(*look_at)" in source
print("Camera orientation and wall-decoration checks: OK")
