"""Static regression checks for v29 visual-only changes."""
from pathlib import Path

text = Path(__file__).with_name("camera3d.py").read_text(encoding="utf-8")
assert 'def _add_stair_arrow' in text
assert 'level_index = run_index if kind == "up" else (steps - 1 - run_index)' in text
assert 'down", floor["stair_down"], down_color, down_arrow' in text
assert 'tabletop-props' in text
assert 'rear_v' in text and 'front_v' in text and 'side_u' in text
assert 'down_dark' not in text
assert 'inner_wall' not in text
print("OK: scale simmetriche con frecce e accessori separati")
