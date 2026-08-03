"""Regression checks for rich office content sent to Panda3D."""
import numpy as np
from building import generate_building, FREE, DOOR, STAIR_UP, STAIR_DOWN, CELL_SIZE
from camera3d import _serialise_scene, _render_frame, RENDER_H, RENDER_W

floors = generate_building(4, seed=12, layout_mode='office', include_objects=True, object_density=4.0)
scene = _serialise_scene(floors)
assert any(any(obj['kind'] == 'table' for obj in floor['objects']) for floor in scene)
assert any(any(obj['kind'] == 'chair' for obj in floor['objects']) for floor in scene)
assert any(any(obj['kind'] == 'debris' for obj in floor['objects']) for floor in scene)
assert any(floor['victims'] for floor in scene)

floor = floors[0]
ys, xs = np.nonzero(np.isin(floor.grid, [FREE, DOOR, STAIR_UP, STAIR_DOWN]))
frames = []
for theta in (0.0, np.pi / 2, np.pi, -np.pi / 2):
    frame = _render_frame(scene[0], (xs[len(xs)//2] + .5) * CELL_SIZE,
                          (ys[len(ys)//2] + .5) * CELL_SIZE, theta)
    assert frame.shape == (RENDER_H, RENDER_W, 3)
    assert frame.dtype == np.uint8
    frames.append(frame)
assert max(np.ptp(f.astype(np.int16)) for f in frames) > 80
print('Rich Panda3D office scene checks: OK')
