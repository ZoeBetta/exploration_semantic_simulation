"""Headless regression checks for v27 Panda3D scene additions."""
from pathlib import Path
import camera3d

source = Path(camera3d.__file__).read_text(encoding='utf-8')
required = [
    'def _add_tabletop_props',
    'tabletop_props',
    'monitor',
    'Pen holder',
    'ring binders',
    'def _add_coat_rack',
    'table-overturned',
    'chair-overturned',
    'kind == "down"',
    'actual opening in the slab',
]
for token in required:
    assert token in source, token

# The helper is intentionally pure and remains available without Panda3D.
pos, target = camera3d._camera_pose_from_robot(1.0, 2.0, 0.0)
assert target[0] > pos[0]
print('v27 visual regression checks: OK')
