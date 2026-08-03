"""Headless checks for the SAR-specific camera decorations added in v25."""
from pathlib import Path
import camera3d

source = Path(camera3d.__file__).read_text(encoding='utf-8')
required = [
    'table-overturned', 'chair-overturned', 'recumbent-body',
    '_add_perimeter_windows', '_add_wall_plants_and_accessories',
    '_build_office_plant', 'CAMERA_BOB_AMPLITUDE_M',
    'CAMERA_BOB_FREQUENCY_HZ', 'CAMERA_BOB_MIN_SPEED_M_S',
]
for token in required:
    assert token in source, token

assert 0.0 < camera3d.CAMERA_BOB_AMPLITUDE_M <= 0.02
assert 1.0 <= camera3d.CAMERA_BOB_FREQUENCY_HZ <= 2.5
assert camera3d.CAMERA_FORWARD_OFFSET_M > 0.0
print('SAR visual details and camera gait checks: OK')
