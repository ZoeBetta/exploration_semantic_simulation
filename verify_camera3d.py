"""Headless regression checks for Panda3D camera scene serialization."""
import numpy as np
from building import generate_building, FREE, DOOR, STAIR_UP, STAIR_DOWN, CELL_SIZE
from camera3d import _serialise_scene, _render_frame, OBJECT_HEIGHTS

floors = generate_building(4, seed=9, layout_mode='office', include_objects=True, object_density=2.0)
scene = _serialise_scene(floors)
assert len(scene) == 4
assert any(np.any(f['object_kind'] != '') for f in scene)
assert all('stair_dirs' in f and 'objects' in f and 'victims' in f for f in scene)

floor = floors[0]
ys, xs = np.nonzero(np.isin(floor.grid, [FREE, DOOR, STAIR_UP, STAIR_DOWN]))
assert len(xs)
x, y = int(xs[len(xs)//2]), int(ys[len(ys)//2])
frame = _render_frame(scene[0], (x + .5) * CELL_SIZE, (y + .5) * CELL_SIZE, 0.0)
assert frame.shape == (400, 640, 3)
assert frame.dtype == np.uint8
assert np.ptp(frame.astype(np.int16)) > 30

for item in floor.environment_objects:
    x0, y0, x1, y1 = item.region
    kind = scene[0]['object_kind'][y0, x0]
    assert kind == item.kind
    assert OBJECT_HEIGHTS[kind] < 2.0
    break
else:
    raise AssertionError('The deterministic building has no objects')

print('Panda3D camera serialization checks: OK')
