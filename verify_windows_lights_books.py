"""Headless regression checks for v26 windows, lamps, and fallen books."""
from pathlib import Path
import numpy as np
from building import generate_building, WALL, DOOR, STAIR_UP, STAIR_DOWN
from camera3d import _serialise_scene, _merge_rectangles, _window_specs_for_wall

window_count = 0
for seed in range(24):
    floors = generate_building(4, seed=seed, layout_mode='office', include_objects=True, object_density=2.0)
    scene = _serialise_scene(floors)
    for floor in scene:
        structural = (floor['grid'] == WALL) & (floor['object_kind'] == '')
        for rectangle in _merge_rectangles(structural):
            specs = _window_specs_for_wall(floor, *rectangle)
            window_count += len(specs)
            for spec in specs:
                assert spec['side'] in {'north', 'south', 'west', 'east'}
                assert spec['width'] > spec['height']
                for x, y in spec['strip_cells']:
                    assert floor['grid'][y, x] not in (WALL, DOOR, STAIR_UP, STAIR_DOWN)
                    assert floor['object_kind'][y, x] == ''
                    assert (x, y) not in set(floor['victims'])
assert window_count > 0, 'No unobstructed perimeter windows were generated'

source = Path(__file__).with_name('camera3d.py').read_text(encoding='utf-8')
for token in (
    '_window_specs_for_wall', '_add_ceiling_lights', 'PointLight',
    '_add_scattered_books', 'central vertical mullion', 'handle_color',
    'reserved_window_cells',
):
    assert token in source, token
print(f'Windows, ceiling lights, and fallen books checks: OK ({window_count} windows)')
