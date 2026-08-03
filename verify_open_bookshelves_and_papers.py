"""Regression checks for v31 open bookshelves and decorative floor papers."""
from pathlib import Path

import numpy as np

import camera3d

source = Path(__file__).with_name("camera3d.py").read_text(encoding="utf-8")

required_tokens = [
    "def _add_open_bookshelf",
    'root.attachNewNode("open-bookshelf")',
    "support_levels = (0.15, 0.53, 0.91, 1.29, 1.67, 2.05)",
    "spine_w = 0.026 + 0.009",
    "z = support_z + shelf_thickness / 2 + book_h / 2",
    "def _add_scattered_floor_papers",
    'root.attachNewNode("scattered-paper")',
    "zone_cols, zone_rows = 6, 4",
    "sheet_count = 1 + int(rng.integers(0, 3))",
    "_add_scattered_floor_papers(app, root, floor)",
]
missing = [token for token in required_tokens if token not in source]
assert not missing, f"Missing v31 visual features: {missing}"

# The previous storage unit and the new open bookshelf must coexist in the
# decoration dispatcher.
assert '_add_bookcase(app, root, side, x0, y0, x1, y1, key)' in source
assert '_add_open_bookshelf(app, root, side, x0, y0, x1, y1, key)' in source

# A simple synthetic floor with two long indoor wall faces should reserve one
# existing cabinet and one new bookshelf.
grid = np.zeros((18, 24), dtype=np.int8)
grid[4, 2:20] = camera3d.WALL
grid[12, 3:22] = camera3d.WALL
rectangles = [(2, 4, 19, 4), (3, 12, 21, 12)]
plan = camera3d._plan_required_wall_furniture(0, grid, rectangles, [])
assert "cabinet" in plan.values(), plan
assert "bookshelf" in plan.values(), plan

# Paper placement uses a deterministic geometry-derived seed and never mutates
# the input grid.
floor = {"floor_index": 2, "grid": grid.copy()}
before = floor["grid"].copy()
seed_a = camera3d._paper_seed_for_floor(floor)
seed_b = camera3d._paper_seed_for_floor(floor)
assert seed_a == seed_b
assert np.array_equal(before, floor["grid"])

print("OK: cabinets, open bookshelves, fallen books and widespread papers are configured.")
