"""Static regression checks for the v28 visual-only changes."""
from pathlib import Path

source = Path(__file__).with_name("camera3d.py").read_text(encoding="utf-8")

assert 'long_axis_is_x = sx >= sy' in source
assert 'monitor.setH(90)' in source
assert '_add_coat_rack' not in source
assert 'top_z = -0.018 - i * 0.065' in source
assert 'region_w*0.96, region_h*0.96' in source
assert 'down_color' in source and 'down_dark' in source
print("v28 monitor orientation, stair descent and coat-rack removal: OK")
