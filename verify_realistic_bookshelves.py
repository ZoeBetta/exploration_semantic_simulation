"""Lightweight source-level regression checks for the v30 bookcase layout."""
from pathlib import Path

source = Path(__file__).with_name("camera3d.py").read_text(encoding="utf-8")
required = [
    "shelf_levels = (0.34, 0.72, 1.10, 1.48, 1.86)",
    "z = support_z + shelf_thickness / 2 + book_h / 2",
    "while cursor < usable_width / 2 - 0.025",
    "book_w = 0.035 + 0.012",
    "room_sign",
]
missing = [token for token in required if token not in source]
assert not missing, f"Missing v30 bookcase features: {missing}"
assert source.count("_add_scattered_books(app, root, side, cx, cy, width, key)") >= 1
print("OK: realistic shelf-aligned book spines are present.")
