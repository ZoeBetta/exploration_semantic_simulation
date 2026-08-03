"""Multi-floor building generator with two selectable topology families.

The geometric world remains an occupancy grid, while the robot pose, LiDAR
rays and local motion are continuous.  The initial configuration can choose:

``office``
    Several office-building families: single and double corridors, crossed
    corridors, racetrack/loop circulation, variable-size offices and merged
    meeting rooms. Each room has one or two open doors as appropriate.

``free``
    A less structured topology made of independent wall segments and open
    passages, similar to the earlier versions of the simulator.

Two non-combinatorial options enable or disable chairs, tables and rubble and
set their density from 1x to 4x.  These objects have explicit semantic metadata for rendering, but all
of their cells are written as ``WALL`` in the geometric grid: the LiDAR,
collision model, occupancy map and A* therefore treat them as ordinary,
non-traversable obstacles.

Cell codes:
    0 = free space
    1 = wall or non-traversable object
    2 = open door
    3 = stair going up
    4 = stair going down

All doors are exactly three cells wide. Office main corridors are five or six
cells wide. The free topology now uses a dense, irregular cross-corridor plan
with at least fourteen wall segments and ten rooms per floor. Chairs, tables
and debris have twice the linear footprint used by the previous version while
remaining ordinary non-traversable cells for sensing and planning.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# CODE-REVIEW NOTES
# Purpose: Environment generation: grid semantics, office topology, doors, stairs, objects, and deterministic random seeds.
# Coordinate convention: array indices are (row=y, column=x), while
# metric positions and GUI points are written as (x, y). Distances are
# metres unless a name explicitly ends in ``_cells`` or ``_px``.
# Reproducibility: stochastic operations must use the seeded RNG passed
# by the run; avoid module-global random calls in simulation logic.
# Separation of concerns: ground truth, robot-built maps, planner state,
# and rendering overlays are intentionally distinct representations.
# When modifying this file, preserve those boundaries and update the
# corresponding ``verify_*.py`` regression test.
# -----------------------------------------------------------------------------

from dataclasses import dataclass
import copy
from functools import lru_cache
import random
from typing import Iterable

import numpy as np
from scipy import ndimage

CELL_SIZE = 0.5
FLOOR_W = 44
FLOOR_H = 30
STAIR_SIZE = 2

DOOR_WIDTH_CELLS = 3
MIN_CORRIDOR_WIDTH_CELLS = 3
MAIN_CORRIDOR_MIN_WIDTH_CELLS = STAIR_SIZE + MIN_CORRIDOR_WIDTH_CELLS
MAIN_CORRIDOR_MAX_WIDTH_CELLS = MAIN_CORRIDOR_MIN_WIDTH_CELLS + 1

MIN_ROOM_WIDTH_CELLS = DOOR_WIDTH_CELLS + 3
MAX_ROOM_WIDTH_CELLS = 13
MIN_ROOM_DEPTH_CELLS = MIN_CORRIDOR_WIDTH_CELLS + 2
DOOR_STAIR_CLEARANCE_CELLS = 1
STAIR_CORE_TWO_PROBABILITY_DENOMINATOR = 3
MIN_ROOM_SPAN_CELLS = DOOR_WIDTH_CELLS

# Geometry retries are deterministic because they consume only the run-local
# seeded RNG.  They handle rare combinations of corridor template, meeting-room
# partitions and reserved stairs without ever retrying the same impossible
# building merely by changing the robot start position.
MAX_BUILDING_LAYOUT_ATTEMPTS = 48
MAX_FLOOR_LAYOUT_ATTEMPTS = 32

# Object dimensions are doubled in both linear directions with respect to v9.
OBJECT_LINEAR_SCALE = 2
OBJECT_DENSITY_MIN = 1.0
OBJECT_DENSITY_MAX = 4.0
OBJECT_ROOM_SAFE_CLEARANCE_CELLS = 1.45

# Dense free-topology guarantees. The cross-shaped corridor network is less
# regular than the office layout but deliberately contains many rooms/walls.
FREE_CORRIDOR_MIN_WIDTH_CELLS = MAIN_CORRIDOR_MIN_WIDTH_CELLS
FREE_CORRIDOR_MAX_WIDTH_CELLS = MAIN_CORRIDOR_MAX_WIDTH_CELLS
FREE_MIN_BLOCK_WIDTH_CELLS = 14
FREE_MIN_BLOCK_DEPTH_CELLS = 7
FREE_MIN_ROOM_WIDTH_CELLS = DOOR_WIDTH_CELLS + 2
FREE_MIN_WALL_REGIONS = 14
FREE_MIN_ROOM_COUNT = 10

FREE, WALL, DOOR, STAIR_UP, STAIR_DOWN = 0, 1, 2, 3, 4

LAYOUT_OFFICE = "office"
LAYOUT_FREE = "free"

OBJECT_CHAIR = "chair"
OBJECT_TABLE = "table"
OBJECT_DEBRIS = "debris"
OBJECT_KINDS = (OBJECT_CHAIR, OBJECT_TABLE, OBJECT_DEBRIS)


@dataclass(frozen=True)
class EnvironmentObject:
    """Static object used only for ground-truth visualization metadata.

    ``region`` is an inclusive cell rectangle ``(x0, y0, x1, y1)``.
    ``rotation_quarters`` is a clockwise top-view rotation in multiples of 90
    degrees.  It does not affect traversability; every region cell is WALL.
    """

    kind: str
    region: tuple[int, int, int, int]
    rotation_quarters: int = 0


class Floor:
    def __init__(self, index: int, n_floors: int):
        self.index = index
        self.n_floors = n_floors
        self.grid = np.zeros((FLOOR_H, FLOOR_W), dtype=np.int8)
        self.victims: list[tuple[int, int]] = []
        # Multiple stair cores are supported. Each core contains one UP and
        # one DOWN footprint placed side by side in a corridor. The singular
        # aliases are kept for backward compatibility and point to core 0.
        self.stair_up_regions: list[tuple[int, int, int, int]] = []
        self.stair_down_regions: list[tuple[int, int, int, int]] = []
        self.stair_core_ids: list[int] = []
        self.stair_up_region = None
        self.stair_down_region = None
        self.stair_core_count = 0
        # Unit corridor direction for each stair core. The arrow, heading and
        # exit pose all use this direction.
        self.stair_core_directions: list[tuple[int, int]] = []
        self.explorable_area_m2 = None
        self.fmap = None

        # Shared metadata.  Existing simulator modules only require grid,
        # victims and stairs, so these additions remain backward-compatible.
        self.layout_mode = None
        self.environment_objects: list[EnvironmentObject] = []
        self.wall_regions: list[tuple[int, int, int, int]] = []
        self.door_regions: list[tuple[int, int, int, int]] = []
        self.obstacle_regions: list[tuple[int, int, int, int]] = []

        # Transient object-packing caches. Object placement is monotonic: once
        # a footprint is rejected because it disconnects free space or removes
        # the planner-clear route to a room, adding further obstacles can never
        # make that same footprint valid later. Remembering rejected
        # footprints therefore preserves the generated result while avoiding
        # thousands of repeated global connectivity/clearance checks at 3x/4x
        # density. These fields are implementation details and are not used by
        # the simulator after building generation.
        self._rejected_object_footprints: set[tuple[int, int, int, int]] = set()
        self._object_placement_saturated = False

        # Office-only diagnostics.
        self.office_orientation = None
        self.office_variant = None
        self.corridor_regions: list[tuple[int, int, int, int]] = []
        self.room_regions: list[tuple[int, int, int, int]] = []
        self.room_door_regions: list[tuple[int, int, int, int]] = []


def normalize_layout_mode(value: str) -> str:
    """Return a canonical topology name, accepting Italian/English aliases."""
    normalized = str(value).strip().lower().replace("_", " ")
    if normalized in {"office", "ufficio", "uffici", "office like"}:
        return LAYOUT_OFFICE
    if normalized in {
        "free",
        "libera",
        "libero",
        "topologia libera",
        "free topology",
    }:
        return LAYOUT_FREE
    raise ValueError(
        f"Topologia non valida: {value!r}. Usare 'office' oppure 'free'."
    )


def normalize_object_density(value) -> float:
    """Validate the fixed clutter-density multiplier (1x to 4x)."""
    density = float(value)
    if not np.isfinite(density):
        raise ValueError("La densita degli oggetti deve essere finita")
    if not OBJECT_DENSITY_MIN <= density <= OBJECT_DENSITY_MAX:
        raise ValueError(
            f"Densita oggetti fuori intervallo: {density}. "
            f"Usare un valore tra {OBJECT_DENSITY_MIN:g} e "
            f"{OBJECT_DENSITY_MAX:g}."
        )
    return density


def _partition_axis(start: int, end: int, rng: random.Random):
    """Split an interval into variable offices separated by one-cell walls."""
    length = end - start + 1
    max_rooms = max(2, (length + 1) // (MIN_ROOM_WIDTH_CELLS + 1))
    min_rooms = max(3, max_rooms - 2) if max_rooms >= 3 else max_rooms
    n_rooms = rng.randint(min_rooms, max_rooms)

    usable_for_rooms = length - (n_rooms - 1)
    widths = [MIN_ROOM_WIDTH_CELLS] * n_rooms
    extra = usable_for_rooms - sum(widths)
    while extra > 0:
        candidates = [
            index for index, width in enumerate(widths)
            if width < MAX_ROOM_WIDTH_CELLS
        ]
        if not candidates:
            candidates = list(range(n_rooms))
        widths[rng.choice(candidates)] += 1
        extra -= 1

    if n_rooms >= 2 and len(set(widths)) == 1:
        donor = next((i for i, width in enumerate(widths)
                      if width > MIN_ROOM_WIDTH_CELLS), None)
        receiver = next((i for i, width in enumerate(widths)
                         if width < MAX_ROOM_WIDTH_CELLS and i != donor), None)
        if donor is not None and receiver is not None:
            widths[donor] -= 1
            widths[receiver] += 1

    rooms = []
    partitions = []
    cursor = start
    for index, width in enumerate(widths):
        room_start = cursor
        room_end = cursor + width - 1
        rooms.append((room_start, room_end))
        cursor = room_end + 1
        if index < n_rooms - 1:
            partitions.append(cursor)
            cursor += 1

    if cursor != end + 1:
        raise RuntimeError("Invalid office partition")
    return rooms, partitions


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return -1


def _choose_door_start(room_start, room_end, stair_regions, axis, rng):
    """Choose a three-cell opening, preferably away from a stair footprint."""
    first = room_start + 1
    last = room_end - DOOR_WIDTH_CELLS
    candidates = list(range(first, last + 1))
    if not candidates:
        raise RuntimeError("Room too small for the requested door width")

    def stair_interval(region):
        x0, y0, x1, y1 = region
        return (x0, x1) if axis == "x" else (y0, y1)

    safe = []
    for start in candidates:
        end = start + DOOR_WIDTH_CELLS - 1
        if all(
            _interval_gap(start, end, *stair_interval(region))
            >= DOOR_STAIR_CLEARANCE_CELLS
            for region in stair_regions
        ):
            safe.append(start)
    if safe:
        return rng.choice(safe)

    non_overlapping = []
    for start in candidates:
        end = start + DOOR_WIDTH_CELLS - 1
        if all(_interval_gap(start, end, *stair_interval(region)) >= 0
               for region in stair_regions):
            non_overlapping.append(start)
    if non_overlapping:
        return max(
            non_overlapping,
            key=lambda start: min(
                _interval_gap(
                    start,
                    start + DOOR_WIDTH_CELLS - 1,
                    *stair_interval(region),
                )
                for region in stair_regions
            ) if stair_regions else 999,
        )
    return rng.choice(candidates)


# =========================================================================
# Office topology
# =========================================================================
def _office_spine(rng: random.Random):
    orientation = rng.choice(("horizontal", "vertical"))
    width = rng.randint(
        MAIN_CORRIDOR_MIN_WIDTH_CELLS,
        MAIN_CORRIDOR_MAX_WIDTH_CELLS,
    )

    if orientation == "horizontal":
        nominal = (FLOOR_H - width) // 2
        lower = max(MIN_ROOM_DEPTH_CELLS + 2, nominal - 2)
        upper = min(
            FLOOR_H - width - MIN_ROOM_DEPTH_CELLS - 2,
            nominal + 2,
        )
        start = rng.randint(lower, upper)
        region = (1, start, FLOOR_W - 2, start + width - 1)
    else:
        nominal = (FLOOR_W - width) // 2
        lower = max(MIN_ROOM_DEPTH_CELLS + 2, nominal - 3)
        upper = min(
            FLOOR_W - width - MIN_ROOM_DEPTH_CELLS - 2,
            nominal + 3,
        )
        start = rng.randint(lower, upper)
        region = (start, 1, start + width - 1, FLOOR_H - 2)

    return orientation, region


def _office_stair_transition_regions(
    n_floors: int,
    orientation: str,
    corridor_region,
    rng: random.Random,
):
    """Create alternating service cores inside the office main corridor."""
    if n_floors <= 1:
        return []

    if orientation == "horizontal":
        _, corridor_y0, _, corridor_y1 = corridor_region
        stair_y0 = corridor_y0
        left_high = max(4, FLOOR_W // 3 - STAIR_SIZE - 1)
        left_x0 = rng.randint(4, left_high)
        right_low = max(
            2 * FLOOR_W // 3,
            left_x0 + STAIR_SIZE + MIN_CORRIDOR_WIDTH_CELLS + 3,
        )
        right_high = FLOOR_W - STAIR_SIZE - 5
        right_x0 = rng.randint(right_low, right_high)
        cores = [
            (left_x0, stair_y0,
             left_x0 + STAIR_SIZE - 1, stair_y0 + STAIR_SIZE - 1),
            (right_x0, stair_y0,
             right_x0 + STAIR_SIZE - 1, stair_y0 + STAIR_SIZE - 1),
        ]
        assert corridor_y1 - (stair_y0 + STAIR_SIZE - 1) >= MIN_CORRIDOR_WIDTH_CELLS
    else:
        corridor_x0, _, corridor_x1, _ = corridor_region
        stair_x0 = corridor_x0
        top_high = max(4, FLOOR_H // 3 - STAIR_SIZE - 1)
        top_y0 = rng.randint(4, top_high)
        bottom_low = max(
            2 * FLOOR_H // 3,
            top_y0 + STAIR_SIZE + MIN_CORRIDOR_WIDTH_CELLS + 3,
        )
        bottom_high = FLOOR_H - STAIR_SIZE - 5
        bottom_y0 = rng.randint(bottom_low, bottom_high)
        cores = [
            (stair_x0, top_y0,
             stair_x0 + STAIR_SIZE - 1, top_y0 + STAIR_SIZE - 1),
            (stair_x0, bottom_y0,
             stair_x0 + STAIR_SIZE - 1, bottom_y0 + STAIR_SIZE - 1),
        ]
        assert corridor_x1 - (stair_x0 + STAIR_SIZE - 1) >= MIN_CORRIDOR_WIDTH_CELLS

    return [cores[index % 2] for index in range(n_floors - 1)]


def _add_horizontal_room_band(
    grid,
    floor,
    wall_y,
    interior_y0,
    interior_y1,
    side,
    stair_regions,
    rng,
):
    grid[wall_y, 1:FLOOR_W - 1] = WALL
    floor.wall_regions.append((1, wall_y, FLOOR_W - 2, wall_y))

    room_spans, partitions = _partition_axis_with_meeting_rooms(1, FLOOR_W - 2, rng)
    for partition_x in partitions:
        y0 = min(interior_y0, wall_y)
        y1 = max(interior_y1, wall_y)
        grid[y0:y1 + 1, partition_x] = WALL
        floor.wall_regions.append((partition_x, y0, partition_x, y1))

    room_entries = []
    for room_x0, room_x1 in room_spans:
        room_region = (room_x0, interior_y0, room_x1, interior_y1)
        door_x0 = _choose_door_start(
            room_x0, room_x1, stair_regions, "x", rng)
        door_region = (
            door_x0,
            wall_y,
            door_x0 + DOOR_WIDTH_CELLS - 1,
            wall_y,
        )
        dx0, dy0, dx1, dy1 = door_region
        grid[dy0:dy1 + 1, dx0:dx1 + 1] = DOOR
        if not _room_is_wide_enough(room_region):
            _seal_narrow_room(grid, floor, room_region)
            continue
        floor.room_regions.append(room_region)
        floor.room_door_regions.append(door_region)
        floor.door_regions.append(door_region)
        room_entries.append((room_region, side))
    return room_entries


def _add_vertical_room_band(
    grid,
    floor,
    wall_x,
    interior_x0,
    interior_x1,
    side,
    stair_regions,
    rng,
):
    grid[1:FLOOR_H - 1, wall_x] = WALL
    floor.wall_regions.append((wall_x, 1, wall_x, FLOOR_H - 2))

    room_spans, partitions = _partition_axis_with_meeting_rooms(1, FLOOR_H - 2, rng)
    for partition_y in partitions:
        x0 = min(interior_x0, wall_x)
        x1 = max(interior_x1, wall_x)
        grid[partition_y, x0:x1 + 1] = WALL
        floor.wall_regions.append((x0, partition_y, x1, partition_y))

    room_entries = []
    for room_y0, room_y1 in room_spans:
        room_region = (interior_x0, room_y0, interior_x1, room_y1)
        door_y0 = _choose_door_start(
            room_y0, room_y1, stair_regions, "y", rng)
        door_region = (
            wall_x,
            door_y0,
            wall_x,
            door_y0 + DOOR_WIDTH_CELLS - 1,
        )
        dx0, dy0, dx1, dy1 = door_region
        grid[dy0:dy1 + 1, dx0:dx1 + 1] = DOOR
        if not _room_is_wide_enough(room_region):
            _seal_narrow_room(grid, floor, room_region)
            continue
        floor.room_regions.append(room_region)
        floor.room_door_regions.append(door_region)
        floor.door_regions.append(door_region)
        room_entries.append((room_region, side))
    return room_entries


def _build_office_layout(
    grid,
    floor,
    orientation,
    corridor_region,
    stair_regions,
    rng,
):
    floor.office_orientation = orientation
    floor.corridor_regions.append(corridor_region)

    if orientation == "horizontal":
        _, corridor_y0, _, corridor_y1 = corridor_region
        top_wall_y = corridor_y0 - 1
        bottom_wall_y = corridor_y1 + 1
        room_entries = []
        room_entries.extend(_add_horizontal_room_band(
            grid, floor, top_wall_y, 1, top_wall_y - 1,
            "north", stair_regions, rng,
        ))
        room_entries.extend(_add_horizontal_room_band(
            grid, floor, bottom_wall_y, bottom_wall_y + 1, FLOOR_H - 2,
            "south", stair_regions, rng,
        ))
    else:
        corridor_x0, _, corridor_x1, _ = corridor_region
        left_wall_x = corridor_x0 - 1
        right_wall_x = corridor_x1 + 1
        room_entries = []
        room_entries.extend(_add_vertical_room_band(
            grid, floor, left_wall_x, 1, left_wall_x - 1,
            "west", stair_regions, rng,
        ))
        room_entries.extend(_add_vertical_room_band(
            grid, floor, right_wall_x, right_wall_x + 1, FLOOR_W - 2,
            "east", stair_regions, rng,
        ))
    return room_entries



# =========================================================================
# Diverse office topology templates (v16)
# =========================================================================
OFFICE_VARIANTS = (
    "single_horizontal",
    "single_vertical",
    "cross",
    "dual_vertical",
    "dual_horizontal",
    "loop",
)
STAIR_DOOR_CLEARANCE_CELLS = 2


def _partition_axis_with_meeting_rooms(start: int, end: int, rng: random.Random):
    """Create ordinary offices plus occasional 2x/3x meeting rooms.

    A base partition is first generated using the same minimum room width as
    before.  Adjacent offices are then merged by deleting one or two internal
    partitions.  This preserves exact coverage of the interval and yields
    recognisably larger meeting rooms without changing door/corridor widths.
    """
    rooms, _ = _partition_axis(start, end, rng)
    if len(rooms) < 2:
        return rooms, []

    merged=[]
    i=0
    made_large=False
    while i < len(rooms):
        remaining=len(rooms)-i
        group=1
        if remaining >= 3 and rng.random() < 0.28:
            group=3
        elif remaining >= 2 and rng.random() < 0.42:
            group=2
        if group > 1:
            made_large=True
        merged.append((rooms[i][0], rooms[i+group-1][1]))
        i += group

    # Most office floors contain at least one meeting room.
    if not made_large and len(merged) >= 2 and rng.random() < 0.72:
        j=rng.randrange(len(merged)-1)
        merged[j:j+2]=[(merged[j][0], merged[j+1][1])]

    partitions=[merged[k][1]+1 for k in range(len(merged)-1)]
    return merged, partitions


def _room_is_wide_enough(region):
    x0, y0, x1, y1 = region
    return (x1 - x0 + 1 >= MIN_ROOM_SPAN_CELLS and
            y1 - y0 + 1 >= MIN_ROOM_SPAN_CELLS)


def _seal_narrow_room(grid, floor, region):
    x0, y0, x1, y1 = region
    grid[y0:y1 + 1, x0:x1 + 1] = WALL
    floor.wall_regions.append(region)


def _add_horizontal_room_band_range(
    grid, floor, wall_y, interior_y0, interior_y1, side,
    x_start, x_end, rng, doors_on_wall=True, second_door_wall_y=None,
    stair_regions=(),
):
    """Add a room band over a restricted horizontal interval."""
    if x_end - x_start + 1 < MIN_ROOM_WIDTH_CELLS:
        return []
    grid[wall_y, x_start:x_end+1] = WALL
    floor.wall_regions.append((x_start, wall_y, x_end, wall_y))
    if second_door_wall_y is not None:
        grid[second_door_wall_y, x_start:x_end+1] = WALL
        floor.wall_regions.append((x_start, second_door_wall_y, x_end, second_door_wall_y))

    spans, partitions = _partition_axis_with_meeting_rooms(x_start, x_end, rng)
    for px in partitions:
        y0=min(interior_y0, wall_y, second_door_wall_y if second_door_wall_y is not None else wall_y)
        y1=max(interior_y1, wall_y, second_door_wall_y if second_door_wall_y is not None else wall_y)
        grid[y0:y1+1, px]=WALL
        floor.wall_regions.append((px,y0,px,y1))

    entries=[]
    for x0,x1 in spans:
        room=(x0, interior_y0, x1, interior_y1)
        if not _room_is_wide_enough(room):
            _seal_narrow_room(grid, floor, room)
            continue
        door_x=_choose_door_start(x0, x1, stair_regions, "x", rng)
        door=(door_x,wall_y,door_x+DOOR_WIDTH_CELLS-1,wall_y)
        grid[wall_y,door_x:door_x+DOOR_WIDTH_CELLS]=DOOR
        floor.room_regions.append(room); floor.room_door_regions.append(door); floor.door_regions.append(door)
        if second_door_wall_y is not None:
            door2=(door_x,second_door_wall_y,door_x+DOOR_WIDTH_CELLS-1,second_door_wall_y)
            grid[second_door_wall_y,door_x:door_x+DOOR_WIDTH_CELLS]=DOOR
            floor.room_door_regions.append(door2); floor.door_regions.append(door2)
        entries.append((room,side))
    return entries


def _add_vertical_room_band_range(
    grid, floor, wall_x, interior_x0, interior_x1, side,
    y_start, y_end, rng, second_door_wall_x=None,
    stair_regions=(),
):
    """Add a room band over a restricted vertical interval."""
    if y_end - y_start + 1 < MIN_ROOM_WIDTH_CELLS:
        return []
    grid[y_start:y_end+1, wall_x]=WALL
    floor.wall_regions.append((wall_x,y_start,wall_x,y_end))
    if second_door_wall_x is not None:
        grid[y_start:y_end+1, second_door_wall_x]=WALL
        floor.wall_regions.append((second_door_wall_x,y_start,second_door_wall_x,y_end))

    spans, partitions = _partition_axis_with_meeting_rooms(y_start,y_end,rng)
    for py in partitions:
        x0=min(interior_x0,wall_x,second_door_wall_x if second_door_wall_x is not None else wall_x)
        x1=max(interior_x1,wall_x,second_door_wall_x if second_door_wall_x is not None else wall_x)
        grid[py,x0:x1+1]=WALL
        floor.wall_regions.append((x0,py,x1,py))

    entries=[]
    for y0,y1 in spans:
        room=(interior_x0,y0,interior_x1,y1)
        if not _room_is_wide_enough(room):
            _seal_narrow_room(grid, floor, room)
            continue
        door_y=_choose_door_start(y0, y1, stair_regions, "y", rng)
        door=(wall_x,door_y,wall_x,door_y+DOOR_WIDTH_CELLS-1)
        grid[door_y:door_y+DOOR_WIDTH_CELLS,wall_x]=DOOR
        floor.room_regions.append(room); floor.room_door_regions.append(door); floor.door_regions.append(door)
        if second_door_wall_x is not None:
            door2=(second_door_wall_x,door_y,second_door_wall_x,door_y+DOOR_WIDTH_CELLS-1)
            grid[door_y:door_y+DOOR_WIDTH_CELLS,second_door_wall_x]=DOOR
            floor.room_door_regions.append(door2); floor.door_regions.append(door2)
        entries.append((room,side))
    return entries


def _office_template(rng: random.Random):
    variant=rng.choice(OFFICE_VARIANTS)
    w=rng.randint(MAIN_CORRIDOR_MIN_WIDTH_CELLS,MAIN_CORRIDOR_MAX_WIDTH_CELLS)
    if variant == "single_horizontal":
        y=(FLOOR_H-w)//2 + rng.choice((-2,-1,0,1,2))
        corridors=[(1,y,FLOOR_W-2,y+w-1)]
    elif variant == "single_vertical":
        x=(FLOOR_W-w)//2 + rng.choice((-3,-2,-1,0,1,2,3))
        corridors=[(x,1,x+w-1,FLOOR_H-2)]
    elif variant == "cross":
        y=(FLOOR_H-w)//2 + rng.choice((-1,0,1))
        x=(FLOOR_W-w)//2 + rng.choice((-2,-1,0,1,2))
        corridors=[(1,y,FLOOR_W-2,y+w-1),(x,1,x+w-1,FLOOR_H-2)]
    elif variant == "dual_vertical":
        x1=rng.randint(8,11); x2=rng.randint(FLOOR_W-12,FLOOR_W-9)
        corridors=[(x1,1,x1+w-1,FLOOR_H-2),(x2,1,x2+w-1,FLOOR_H-2)]
    elif variant == "dual_horizontal":
        y1=rng.randint(6,8); y2=rng.randint(FLOOR_H-9,FLOOR_H-7)
        corridors=[(1,y1,FLOOR_W-2,y1+w-1),(1,y2,FLOOR_W-2,y2+w-1)]
    else:  # loop / racetrack
        x1=rng.randint(7,9); x2=rng.randint(FLOOR_W-w-5,FLOOR_W-w-3)
        y1=rng.randint(5,7); y2=rng.randint(FLOOR_H-w-5,FLOOR_H-w-3)
        corridors=[(x1,y1,x2+w-1,y1+w-1),(x1,y2,x2+w-1,y2+w-1),
                   (x1,y1,x1+w-1,y2+w-1),(x2,y1,x2+w-1,y2+w-1)]
    return variant,corridors


def _build_office_variant(grid, floor, variant, corridors, stair_regions, rng):
    """Build one of several office-like plans from a shared building template."""
    floor.office_variant=variant
    floor.office_orientation=variant
    floor.corridor_regions.extend(corridors)
    entries=[]

    if variant.startswith("single_"):
        orientation="horizontal" if variant.endswith("horizontal") else "vertical"
        entries.extend(_build_office_layout(grid, floor, orientation, corridors[0], stair_regions, rng))
        # Replace a few ordinary partitions with larger meeting-room spans is
        # already handled by the new helpers in the additional variants.
        return entries

    if variant == "cross":
        h,v=corridors; _,hy0,_,hy1=h; vx0,_,vx1,_=v
        # Four quadrant bands facing the horizontal corridor.
        for xa,xb in ((1,vx0-2),(vx1+2,FLOOR_W-2)):
            entries += _add_horizontal_room_band_range(
                grid, floor, hy0 - 1, 1, hy0 - 2, "north", xa, xb, rng,
                stair_regions=stair_regions,
            )
            entries += _add_horizontal_room_band_range(
                grid, floor, hy1 + 1, hy1 + 2, FLOOR_H - 2, "south",
                xa, xb, rng, stair_regions=stair_regions,
            )
    elif variant == "dual_vertical":
        left,right=corridors; lx0,_,lx1,_=left; rx0,_,rx1,_=right
        entries += _add_vertical_room_band_range(
            grid, floor, lx0 - 1, 1, lx0 - 2, "west", 1, FLOOR_H - 2, rng,
            stair_regions=stair_regions,
        )
        # Central rooms have a door onto both corridors.
        entries += _add_vertical_room_band_range(
            grid, floor, lx1 + 1, lx1 + 2, rx0 - 2, "west",
            1, FLOOR_H - 2, rng, second_door_wall_x=rx0 - 1,
            stair_regions=stair_regions,
        )
        entries += _add_vertical_room_band_range(
            grid, floor, rx1 + 1, rx1 + 2, FLOOR_W - 2, "east",
            1, FLOOR_H - 2, rng, stair_regions=stair_regions,
        )
    elif variant == "dual_horizontal":
        top,bottom=corridors; _,ty0,_,ty1=top; _,by0,_,by1=bottom
        entries += _add_horizontal_room_band_range(
            grid, floor, ty0 - 1, 1, ty0 - 2, "north", 1, FLOOR_W - 2, rng,
            stair_regions=stair_regions,
        )
        entries += _add_horizontal_room_band_range(
            grid, floor, ty1 + 1, ty1 + 2, by0 - 2, "north",
            1, FLOOR_W - 2, rng, second_door_wall_y=by0 - 1,
            stair_regions=stair_regions,
        )
        entries += _add_horizontal_room_band_range(
            grid, floor, by1 + 1, by1 + 2, FLOOR_H - 2, "south",
            1, FLOOR_W - 2, rng, stair_regions=stair_regions,
        )
    else:  # loop / racetrack
        top,bottom,left,right=corridors
        lx0,_,lx1,_=left; rx0,_,rx1,_=right; _,ty0,_,ty1=top; _,by0,_,by1=bottom
        entries += _add_horizontal_room_band_range(
            grid, floor, ty0 - 1, 1, ty0 - 2, "north", 1, FLOOR_W - 2, rng,
            stair_regions=stair_regions,
        )
        entries += _add_horizontal_room_band_range(
            grid, floor, by1 + 1, by1 + 2, FLOOR_H - 2, "south",
            1, FLOOR_W - 2, rng, stair_regions=stair_regions,
        )
        # Central core: one or two large meeting rooms, each with two access doors.
        if rx0-lx1 >= MIN_ROOM_WIDTH_CELLS+4 and by0-ty1 >= MIN_ROOM_WIDTH_CELLS+4:
            entries += _add_horizontal_room_band_range(
                grid, floor, ty1 + 1, ty1 + 2, by0 - 2, "north",
                lx1 + 1, rx0 - 1, rng, second_door_wall_y=by0 - 1,
                stair_regions=stair_regions,
            )
    return entries


def _stair_is_in_front_of_door(region, door, clearance=STAIR_DOOR_CLEARANCE_CELLS):
    """Return True only for the approach strip normal to a doorway.

    A horizontal door blocks a narrow vertical approach strip and a vertical
    door blocks a narrow horizontal strip.  This is deliberately less
    restrictive than an isotropic square buffer: a stair may be laterally near
    a door, but never directly in front of its opening.
    """
    sx0,sy0,sx1,sy1=region
    dx0,dy0,dx1,dy1=door
    if dy0 == dy1:  # horizontal doorway: approach is vertical
        lateral_overlap = not (sx1 < dx0 or sx0 > dx1)
        normal_gap = max(dy0-sy1-1, sy0-dy1-1, 0)
        return lateral_overlap and normal_gap <= clearance
    # vertical doorway: approach is horizontal
    lateral_overlap = not (sy1 < dy0 or sy0 > dy1)
    normal_gap = max(dx0-sx1-1, sx0-dx1-1, 0)
    return lateral_overlap and normal_gap <= clearance


def _regions_adjacent_pair(region_a, region_b):
    """True when two equal stair footprints share a complete side."""
    ax0, ay0, ax1, ay1 = region_a
    bx0, by0, bx1, by1 = region_b
    horizontal = ay0 == by0 and ay1 == by1 and (ax1 + 1 == bx0 or bx1 + 1 == ax0)
    vertical = ax0 == bx0 and ax1 == bx1 and (ay1 + 1 == by0 or by1 + 1 == ay0)
    return horizontal or vertical


def _corridor_direction_for_footprint(corridor_mask, footprint):
    """Return the corridor axis and sign offering the longest clear exit."""
    x0, y0, x1, y1 = footprint
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    def run(dx, dy):
        n = 0; x, y = cx, cy
        while True:
            x += dx; y += dy
            if not (0 <= x < FLOOR_W and 0 <= y < FLOOR_H): break
            if not corridor_mask[y, x]: break
            n += 1
        return n
    left, right = run(-1,0), run(1,0)
    up, down = run(0,-1), run(0,1)
    if left + right >= up + down:
        return (1,0) if right >= left else (-1,0)
    return (0,1) if down >= up else (0,-1)


def _candidate_stair_core_pairs(floors, corridors):
    """Enumerate adjacent UP/DOWN pairs fully contained in corridor space.

    The pair is placed against one side of a corridor so at least three free
    cells remain beside it. Both orientations are considered, which also works
    for crossed, dual and loop office plans.
    """
    all_doors = [door for floor in floors for door in floor.door_regions]
    corridor_mask = np.zeros((FLOOR_H, FLOOR_W), dtype=bool)
    for x0, y0, x1, y1 in corridors:
        corridor_mask[y0:y1 + 1, x0:x1 + 1] = True

    pairs = []
    # Side-by-side along x: total footprint 4 x 2 cells.
    for y in range(1, FLOOR_H - STAIR_SIZE):
        for x in range(1, FLOOR_W - 2 * STAIR_SIZE):
            up = (x, y, x + STAIR_SIZE - 1, y + STAIR_SIZE - 1)
            down = (x + STAIR_SIZE, y,
                    x + 2 * STAIR_SIZE - 1, y + STAIR_SIZE - 1)
            footprint = (x, y, x + 2 * STAIR_SIZE - 1, y + STAIR_SIZE - 1)
            ys = slice(y, y + STAIR_SIZE)
            xs = slice(x, x + 2 * STAIR_SIZE)
            if not np.all(corridor_mask[ys, xs]):
                continue
            if any(np.any(f.grid[ys, xs] != FREE) for f in floors):
                continue
            if any(_stair_is_in_front_of_door(footprint, door)
                   for door in all_doors):
                continue
            # Preserve at least three cells across the corridor next to stairs.
            local_clear = False
            for cy0, cy1 in ((y - MIN_CORRIDOR_WIDTH_CELLS, y - 1),
                             (y + STAIR_SIZE,
                              y + STAIR_SIZE + MIN_CORRIDOR_WIDTH_CELLS - 1)):
                if 0 <= cy0 <= cy1 < FLOOR_H:
                    if np.all(corridor_mask[cy0:cy1 + 1, xs]):
                        local_clear = True
            if local_clear:
                pairs.append((up, down, footprint, _corridor_direction_for_footprint(corridor_mask, footprint)))

    # Side-by-side along y: total footprint 2 x 4 cells.
    for y in range(1, FLOOR_H - 2 * STAIR_SIZE):
        for x in range(1, FLOOR_W - STAIR_SIZE):
            up = (x, y, x + STAIR_SIZE - 1, y + STAIR_SIZE - 1)
            down = (x, y + STAIR_SIZE,
                    x + STAIR_SIZE - 1, y + 2 * STAIR_SIZE - 1)
            footprint = (x, y, x + STAIR_SIZE - 1, y + 2 * STAIR_SIZE - 1)
            ys = slice(y, y + 2 * STAIR_SIZE)
            xs = slice(x, x + STAIR_SIZE)
            if not np.all(corridor_mask[ys, xs]):
                continue
            if any(np.any(f.grid[ys, xs] != FREE) for f in floors):
                continue
            if any(_stair_is_in_front_of_door(footprint, door)
                   for door in all_doors):
                continue
            local_clear = False
            for cx0, cx1 in ((x - MIN_CORRIDOR_WIDTH_CELLS, x - 1),
                             (x + STAIR_SIZE,
                              x + STAIR_SIZE + MIN_CORRIDOR_WIDTH_CELLS - 1)):
                if 0 <= cx0 <= cx1 < FLOOR_W:
                    if np.all(corridor_mask[ys, cx0:cx1 + 1]):
                        local_clear = True
            if local_clear:
                pairs.append((up, down, footprint, _corridor_direction_for_footprint(corridor_mask, footprint)))

    # Rank by distance from doors; farther is better.
    ranked = []
    for up, down, footprint, direction in pairs:
        cx = (footprint[0] + footprint[2]) / 2.0
        cy = (footprint[1] + footprint[3]) / 2.0
        min_d2 = min(
            (cx - (d[0] + d[2]) / 2.0) ** 2
            + (cy - (d[1] + d[3]) / 2.0) ** 2
            for d in all_doors
        ) if all_doors else 9999.0
        ranked.append((min_d2, up, down, footprint, direction))
    ranked.sort(reverse=True, key=lambda item: item[0])
    return ranked


def _select_stair_cores_after_rooms(floors, corridors, rng, two_cores=False):
    """Select one core in 2/3 of buildings and two cores in 1/3.

    Each core is a pair of adjacent UP/DOWN stair footprints. The same cores
    are reused on every floor and for every Fw x Opt condition of a run.
    """
    if len(floors) <= 1:
        return []
    requested = 2 if two_cores else 1
    candidates = _candidate_stair_core_pairs(floors, corridors)
    if not candidates:
        raise RuntimeError("No adjacent stair pair available in corridors")

    chosen = []
    for _ in range(requested):
        viable = []
        for item in candidates:
            footprint = item[3]
            if all(
                _interval_gap(footprint[0], footprint[2], prev[0], prev[2])
                >= MIN_CORRIDOR_WIDTH_CELLS
                or _interval_gap(footprint[1], footprint[3], prev[1], prev[3])
                >= MIN_CORRIDOR_WIDTH_CELLS
                for _, _, prev, _ in chosen
            ):
                viable.append(item)
        if not viable:
            # Very compact layouts may not admit a second safe core. To keep
            # the exact 1/3 policy, regenerate by signalling failure upstream.
            if requested == 2:
                raise RuntimeError("Layout cannot host two separated stair cores")
            viable = candidates
        top = viable[:max(1, min(12, len(viable)))]
        _, up, down, footprint, direction = rng.choice(top)
        chosen.append((up, down, footprint, direction))
    return [(up, down, direction) for up, down, _, direction in chosen]


def _combined_stair_footprint(up_region, down_region):
    """Return the inclusive rectangle occupied by one adjacent UP/DOWN pair."""
    return (
        min(up_region[0], down_region[0]),
        min(up_region[1], down_region[1]),
        max(up_region[2], down_region[2]),
        max(up_region[3], down_region[3]),
    )


def _stair_reservation_regions(stair_cores):
    """Flatten stair cores into regions used while doors are generated.

    The critical fix for tall buildings is to reserve the stair locations
    *before* floor-specific room doors are sampled.  Previously, the code first
    generated all doors independently and then searched for one stair location
    that was clear on every floor.  The union of door approach strips becomes
    increasingly dense as the number of floors grows, so six-floor buildings
    could have no common candidate even though every individual floor had ample
    corridor space.
    """
    regions = []
    for up_region, down_region, _ in stair_cores:
        regions.extend((up_region, down_region))
    return regions


def _stair_cores_fit_floor(floor, stair_cores):
    """Check that reserved cores remain free and are not in front of doors."""
    for up_region, down_region, _ in stair_cores:
        for region in (up_region, down_region):
            x0, y0, x1, y1 = region
            if np.any(floor.grid[y0:y1 + 1, x0:x1 + 1] != FREE):
                return False
        footprint = _combined_stair_footprint(up_region, down_region)
        if any(_stair_is_in_front_of_door(footprint, door)
               for door in floor.door_regions):
            return False
    return True


def _stair_core_pair_is_separated(footprint_a, footprint_b):
    """Keep two independent stair cores apart by a usable corridor interval."""
    return (
        _interval_gap(footprint_a[0], footprint_a[2],
                      footprint_b[0], footprint_b[2])
        >= MIN_CORRIDOR_WIDTH_CELLS
        or
        _interval_gap(footprint_a[1], footprint_a[3],
                      footprint_b[1], footprint_b[3])
        >= MIN_CORRIDOR_WIDTH_CELLS
    )


def _select_stair_cores_from_corridors(corridors, rng, two_cores=False):
    """Reserve one or two adjacent stair pairs from the corridor template.

    This selector intentionally runs before rooms, doors and clutter are
    generated.  It therefore depends only on the shared circulation skeleton,
    which is identical on all floors.  Door placement on every floor then uses
    these regions as exclusions.  The result scales to many floors without
    taking the ever-growing union of independently sampled door positions.
    """
    requested = 2 if two_cores else 1
    candidates = _candidate_stair_core_pairs([], corridors)
    if not candidates:
        raise RuntimeError(
            "No adjacent stair pair fits the corridor template"
        )

    if requested == 1:
        top = candidates[:max(1, min(24, len(candidates)))]
        _, up, down, _, direction = rng.choice(top)
        return [(up, down, direction)]

    # Select both cores jointly.  The previous greedy selection could choose a
    # good first core that left no legal second core, even though another pair
    # of candidates would have worked.  Joint enumeration removes that failure
    # mode while preserving deterministic seeded randomness.
    combinations = []
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            if not _stair_core_pair_is_separated(first[3], second[3]):
                continue
            c1x = (first[3][0] + first[3][2]) / 2.0
            c1y = (first[3][1] + first[3][3]) / 2.0
            c2x = (second[3][0] + second[3][2]) / 2.0
            c2y = (second[3][1] + second[3][3]) / 2.0
            separation = abs(c1x - c2x) + abs(c1y - c2y)
            score = first[0] + second[0] + separation
            combinations.append((score, first, second))
    if not combinations:
        raise RuntimeError(
            "Corridor template cannot host two separated stair cores"
        )
    combinations.sort(reverse=True, key=lambda item: item[0])
    top = combinations[:max(1, min(64, len(combinations)))]
    _, first, second = rng.choice(top)
    return [
        (first[1], first[2], first[4]),
        (second[1], second[2], second[4]),
    ]


# =========================================================================
# Object placement shared by both topology families
# =========================================================================
def _traversable_connected(grid: np.ndarray) -> bool:
    """Check that static clutter does not isolate a free pocket.

    This test is called many times while dense furniture is packed. Using
    SciPy's compiled connected-component labelling avoids the Python set/stack
    overhead of the former flood fill without changing the four-neighbour
    connectivity rule.
    """
    traversable = np.isin(grid, (FREE, DOOR, STAIR_UP, STAIR_DOWN))
    if not np.any(traversable):
        return True
    structure = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=np.uint8,
    )
    _labels, component_count = ndimage.label(
        traversable, structure=structure
    )
    return int(component_count) <= 1



def _all_rooms_keep_clearance_route(grid: np.ndarray, floor: Floor) -> bool:
    """Keep at least one A*-clear cell in every room and one shared component.

    The global planner rejects intermediate cells whose distance from a known
    obstacle is below 1.45 cells.  Central clutter must therefore not leave a
    room geometrically connected but unreachable under the actual planner
    clearance rule.
    """
    if not floor.room_regions:
        return True
    wall_mask = grid == WALL
    padded = np.pad(~wall_mask, 1, mode="constant", constant_values=False)
    clearance = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    traversable = np.isin(grid, (FREE, DOOR, STAIR_UP, STAIR_DOWN))
    safe = traversable & (clearance >= OBJECT_ROOM_SAFE_CLEARANCE_CELLS)
    ys, xs = np.where(safe)
    if len(xs) == 0:
        return False

    # Eight-neighbour movement is allowed only when a diagonal does not cut
    # across an unsafe orthogonal corner, exactly as in the planner. A boolean
    # array is substantially faster than a Python set for this fixed 44x30 map.
    start_x, start_y = int(xs[0]), int(ys[0])
    seen = np.zeros_like(safe, dtype=bool)
    seen[start_y, start_x] = True
    stack = [(start_x, start_y)]
    while stack:
        x, y = stack.pop()
        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < FLOOR_W and 0 <= ny < FLOOR_H):
                continue
            if not safe[ny, nx] or seen[ny, nx]:
                continue
            if dx and dy and (not safe[y, nx] or not safe[ny, x]):
                continue
            seen[ny, nx] = True
            stack.append((nx, ny))

    for x0, y0, x1, y1 in floor.room_regions:
        if not np.any(seen[y0:y1 + 1, x0:x1 + 1]):
            return False
    return True

def _register_object(
    grid: np.ndarray,
    floor: Floor,
    kind: str,
    region: tuple[int, int, int, int],
    rotation_quarters: int = 0,
) -> bool:
    """Try one static-object footprint without repeating known failures.

    The validity checks depend only on the occupied cells, not on whether the
    object is rendered as a chair, table, or rubble. Because placement only
    turns FREE cells into WALL cells, a footprint rejected at one point can
    never become valid after additional clutter is added. Caching that result
    is therefore exact, deterministic, and especially important for 3x/4x
    packing where the old code could evaluate the same impossible footprint
    thousands of times.
    """
    if kind not in OBJECT_KINDS:
        raise ValueError(f"Unknown environment object: {kind}")
    region = tuple(map(int, region))
    rejected = floor._rejected_object_footprints
    if region in rejected:
        return False

    x0, y0, x1, y1 = region
    if x0 < 1 or y0 < 1 or x1 >= FLOOR_W - 1 or y1 >= FLOOR_H - 1:
        rejected.add(region)
        return False
    if not np.all(grid[y0:y1 + 1, x0:x1 + 1] == FREE):
        # Occupancy can only increase during packing, so overlap cannot resolve
        # later either. Remember it to avoid revisiting the same rectangle.
        rejected.add(region)
        return False

    previous = grid[y0:y1 + 1, x0:x1 + 1].copy()
    grid[y0:y1 + 1, x0:x1 + 1] = WALL
    if (not _traversable_connected(grid) or
            not _all_rooms_keep_clearance_route(grid, floor)):
        grid[y0:y1 + 1, x0:x1 + 1] = previous
        rejected.add(region)
        return False

    floor.environment_objects.append(
        EnvironmentObject(kind, region, int(rotation_quarters) % 4)
    )
    floor.obstacle_regions.append(region)
    return True


def _object_shapes(kind, room_w, room_h):
    """Return fitting (width, height, rotation) alternatives."""
    if kind == OBJECT_CHAIR:
        candidates = [(2, 2, 0)]
    elif kind == OBJECT_TABLE:
        candidates = [
            (4, 2, 0), (6, 2, 0),
            (2, 4, 1), (2, 6, 1),
        ]
    else:
        candidates = [
            (2, 2, 0), (4, 2, 0), (2, 4, 1), (4, 4, 0),
        ]
    return [shape for shape in candidates
            if shape[0] <= room_w and shape[1] <= room_h]


def _object_region(room_region, side, kind, rng, placement_mode):
    """Propose an object either against a wall or in the room centre."""
    x0, y0, x1, y1 = room_region
    room_w = x1 - x0 + 1
    room_h = y1 - y0 + 1
    shapes = _object_shapes(kind, room_w, room_h)
    if not shapes:
        return None
    width, height, base_rotation = rng.choice(shapes)

    if placement_mode == "center":
        # Centre with one-cell jitter.  This deliberately puts desks, chairs
        # and rubble in the central circulation area in a visible fraction of
        # rooms, while the connectivity test prevents complete obstruction.
        nominal_x = int(round((x0 + x1 - width + 1) / 2.0))
        nominal_y = int(round((y0 + y1 - height + 1) / 2.0))
        tx0 = min(max(nominal_x + rng.choice((-1, 0, 0, 1)), x0), x1 - width + 1)
        ty0 = min(max(nominal_y + rng.choice((-1, 0, 0, 1)), y0), y1 - height + 1)
        rotation = base_rotation if kind != OBJECT_CHAIR else rng.randrange(4)
        return (tx0, ty0, tx0 + width - 1, ty0 + height - 1), rotation

    if side in ("north", "south"):
        # Prefer the long horizontal table alternative along horizontal walls.
        horizontal = [shape for shape in shapes if shape[0] >= shape[1]]
        if horizontal:
            width, height, base_rotation = rng.choice(horizontal)
        tx0 = rng.randint(x0, x1 - width + 1)
        ty0 = y0 if side == "north" else y1 - height + 1
    else:
        vertical = [shape for shape in shapes if shape[1] >= shape[0]]
        if vertical:
            width, height, base_rotation = rng.choice(vertical)
        tx0 = x0 if side == "west" else x1 - width + 1
        ty0 = rng.randint(y0, y1 - height + 1)

    rotation = base_rotation
    if kind == OBJECT_CHAIR:
        rotation = {
            "north": 0, "east": 1, "south": 2, "west": 3,
        }[side]
    elif kind == OBJECT_DEBRIS:
        rotation = rng.randrange(4)
    return (tx0, ty0, tx0 + width - 1, ty0 + height - 1), rotation


def _place_one_office_object(
    grid,
    floor,
    room_entries,
    kind,
    rng,
    placement_mode=None,
) -> bool:
    entries = list(room_entries)
    rng.shuffle(entries)
    modes = ([placement_mode] if placement_mode is not None
             else ["center", "wall"] if rng.random() < 0.38
             else ["wall", "center"])
    for room_region, side in entries:
        for mode in modes:
            for _ in range(16):
                proposal = _object_region(
                    room_region, side, kind, rng, mode
                )
                if proposal is None:
                    break
                region, rotation = proposal
                if _register_object(grid, floor, kind, region, rotation):
                    return True
    return False


def _place_exhaustive_small_object(grid, floor, room_entries, rng) -> bool:
    """Try every still-plausible 2x2 footprint once, then declare saturation.

    At high density the random proposer eventually runs out of legal room. The
    former fallback was invoked after every failed random proposal and rescanned
    the complete floor, repeatedly running expensive global route checks on the
    same rectangles. This bounded pass skips cached failures and, when no 2x2
    footprint remains, records that no larger object can fit either.
    """
    if floor._object_placement_saturated:
        return False

    entries = list(room_entries)
    rng.shuffle(entries)
    kinds = [OBJECT_CHAIR, OBJECT_DEBRIS]
    rng.shuffle(kinds)
    rejected = floor._rejected_object_footprints

    for room_region, _side in entries:
        x0, y0, x1, y1 = room_region
        positions = [
            (x, y)
            for y in range(y0, y1)
            for x in range(x0, x1)
            if (x, y, x + 1, y + 1) not in rejected
        ]
        # Central positions are tried before the periphery.
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        positions.sort(key=lambda p: (p[0] + 0.5 - cx) ** 2
                       + (p[1] + 0.5 - cy) ** 2)
        for x, y in positions:
            region = (x, y, x + 1, y + 1)
            # Semantics and rotation do not affect occupancy. Try the first
            # shuffled visual kind only; if this footprint fails, trying the
            # second kind would repeat the exact same geometry test.
            if _register_object(
                grid, floor, kinds[0], region, rng.randrange(4)
            ):
                return True

    floor._object_placement_saturated = True
    return False


def _place_office_objects(grid, floor, room_entries, rng, density=1.0):
    """Place 1x..4x clutter, explicitly including room-centre objects."""
    density = normalize_object_density(density)
    base_target = max(6, min(12, len(room_entries) + 2))
    target_total = int(round(base_target * density))

    # Guarantee all semantic classes and guarantee central clutter.  At 1x at
    # least two objects are central; the number grows with the density.
    for kind in OBJECT_KINDS:
        _place_one_office_object(
            grid, floor, room_entries, kind, rng,
            placement_mode="center",
        )
    central_target = min(
        len(room_entries),
        max(2, int(round(2.0 * density))),
    )
    central_count = sum(
        1 for obj in floor.environment_objects
        if any(
            abs((obj.region[0] + obj.region[2]) / 2.0
                - (room[0] + room[2]) / 2.0) <= 1.5
            and abs((obj.region[1] + obj.region[3]) / 2.0
                    - (room[1] + room[3]) / 2.0) <= 1.5
            for room, _side in room_entries
        )
    )
    while central_count < central_target:
        kind = rng.choice((OBJECT_CHAIR, OBJECT_DEBRIS, OBJECT_TABLE))
        if not _place_one_office_object(
            grid, floor, room_entries, kind, rng, placement_mode="center"
        ):
            break
        central_count += 1

    weighted_kinds = (
        OBJECT_TABLE,
        OBJECT_CHAIR, OBJECT_CHAIR, OBJECT_CHAIR, OBJECT_CHAIR,
        OBJECT_DEBRIS, OBJECT_DEBRIS,
    )
    attempts = 0
    # Random proposals preserve the visual distribution used by earlier
    # versions. A bounded exhaustive pass is used only after a random failure.
    # Once that pass proves that no 2x2 footprint remains, continuing to sample
    # larger objects is pointless and used to be the source of multi-minute
    # pauses at building boundaries (notably seed 24 at density 3x).
    max_attempts = int(600 + 700 * density)
    while len(floor.environment_objects) < target_total and attempts < max_attempts:
        attempts += 1
        kind = rng.choice(weighted_kinds)
        if _place_one_office_object(
            grid, floor, room_entries, kind, rng
        ):
            continue
        if density >= 2.0:
            if _place_exhaustive_small_object(
                    grid, floor, room_entries, rng):
                continue
            break


# =========================================================================
# Dense irregular free topology
# =========================================================================
def _free_cross_spine(rng: random.Random):
    """Return two wide crossing corridors with randomized offsets.

    The cross is intentionally less regular than the office layout: its
    horizontal and vertical axes move independently, producing four unequal
    room blocks. Both corridors remain at least five cells wide so a two-cell
    stair footprint still leaves a three-cell passage beside it.
    """
    h_width = rng.randint(
        FREE_CORRIDOR_MIN_WIDTH_CELLS,
        FREE_CORRIDOR_MAX_WIDTH_CELLS,
    )
    v_width = rng.randint(
        FREE_CORRIDOR_MIN_WIDTH_CELLS,
        FREE_CORRIDOR_MAX_WIDTH_CELLS,
    )

    h_low = FREE_MIN_BLOCK_DEPTH_CELLS + 2
    h_high = FLOOR_H - h_width - FREE_MIN_BLOCK_DEPTH_CELLS - 2
    if h_low > h_high:
        raise RuntimeError("Floor too short for dense free topology")
    hy0 = rng.randint(h_low, h_high)
    hy1 = hy0 + h_width - 1

    v_low = FREE_MIN_BLOCK_WIDTH_CELLS + 2
    v_high = FLOOR_W - v_width - FREE_MIN_BLOCK_WIDTH_CELLS - 2
    if v_low > v_high:
        raise RuntimeError("Floor too narrow for dense free topology")
    vx0 = rng.randint(v_low, v_high)
    vx1 = vx0 + v_width - 1

    horizontal = (1, hy0, FLOOR_W - 2, hy1)
    vertical = (vx0, 1, vx1, FLOOR_H - 2)
    return horizontal, vertical


def _free_stair_transition_regions(
    n_floors: int,
    horizontal_corridor,
    vertical_corridor,
    rng: random.Random,
):
    """Choose alternating stair cores in the four arms of the corridor cross."""
    if n_floors <= 1:
        return []

    _, hy0, _, hy1 = horizontal_corridor
    vx0, _, vx1, _ = vertical_corridor
    candidates = []

    # On a horizontal arm, place the 2x2 stair at the upper edge, preserving
    # at least three traversable rows below it.
    left_low = 4
    left_high = vx0 - STAIR_SIZE - 3
    if left_low <= left_high:
        x0 = rng.randint(left_low, left_high)
        candidates.append((x0, hy0, x0 + STAIR_SIZE - 1, hy0 + STAIR_SIZE - 1))

    right_low = vx1 + 3
    right_high = FLOOR_W - STAIR_SIZE - 5
    if right_low <= right_high:
        x0 = rng.randint(right_low, right_high)
        candidates.append((x0, hy0, x0 + STAIR_SIZE - 1, hy0 + STAIR_SIZE - 1))

    # On a vertical arm, place the stair at the left edge, preserving at least
    # three traversable columns on its right.
    top_low = 4
    top_high = hy0 - STAIR_SIZE - 3
    if top_low <= top_high:
        y0 = rng.randint(top_low, top_high)
        candidates.append((vx0, y0, vx0 + STAIR_SIZE - 1, y0 + STAIR_SIZE - 1))

    bottom_low = hy1 + 3
    bottom_high = FLOOR_H - STAIR_SIZE - 5
    if bottom_low <= bottom_high:
        y0 = rng.randint(bottom_low, bottom_high)
        candidates.append((vx0, y0, vx0 + STAIR_SIZE - 1, y0 + STAIR_SIZE - 1))

    if len(candidates) < 2:
        raise RuntimeError("Not enough distinct stair cores in free topology")

    rng.shuffle(candidates)
    # Adjacent floor transitions always use different footprints. For very tall
    # buildings the sequence wraps, but never repeats on consecutive links.
    return [candidates[index % len(candidates)]
            for index in range(n_floors - 1)]


def _write_wall(grid, floor, region):
    x0, y0, x1, y1 = region
    grid[y0:y1 + 1, x0:x1 + 1] = WALL
    floor.wall_regions.append(region)


def _open_door(grid, floor, region):
    x0, y0, x1, y1 = region
    grid[y0:y1 + 1, x0:x1 + 1] = DOOR
    floor.door_regions.append(region)
    floor.room_door_regions.append(region)


def _partition_span_count(start, end, count, minimum, rng):
    """Partition an inclusive interval into ``count`` variable room spans."""
    length = end - start + 1
    required = count * minimum + (count - 1)
    if length < required:
        raise ValueError("Span too short for requested room count")

    widths = [minimum] * count
    extra = length - required
    while extra > 0:
        widths[rng.randrange(count)] += 1
        extra -= 1

    spans = []
    partitions = []
    cursor = start
    for index, width in enumerate(widths):
        spans.append((cursor, cursor + width - 1))
        cursor += width
        if index < count - 1:
            partitions.append(cursor)
            cursor += 1
    if cursor != end + 1:
        raise RuntimeError("Invalid dense-free partition")
    return spans, partitions


def _corridor_door_for_side(block, side, stair_regions, rng):
    x0, y0, x1, y1 = block
    if side in ("north", "south"):
        start = _choose_door_start(x0, x1, stair_regions, "x", rng)
        wall_y = y0 - 1 if side == "north" else y1 + 1
        return (start, wall_y, start + DOOR_WIDTH_CELLS - 1, wall_y)
    start = _choose_door_start(y0, y1, stair_regions, "y", rng)
    wall_x = x0 - 1 if side == "west" else x1 + 1
    return (wall_x, start, wall_x, start + DOOR_WIDTH_CELLS - 1)


def _build_free_cross_layout(
    grid,
    floor,
    horizontal_corridor,
    vertical_corridor,
    stair_regions,
    rng,
):
    """Build a dense irregular room/corridor topology.

    The two crossing corridors form four unequal blocks. Every block has two
    corridor-facing wall segments, one or two three-cell doors, and is split
    into two or three connected rooms. The resulting floor always contains at
    least fourteen wall segments, ten rooms and two broad corridors -- more
    than double the structural density of the old 4..7-wall free generator.
    """
    _, hy0, _, hy1 = horizontal_corridor
    vx0, _, vx1, _ = vertical_corridor

    # Store the four arms plus the central crossing as separate corridor
    # regions. Their union is the visible cross-shaped circulation network.
    floor.corridor_regions.extend([
        (1, hy0, vx0 - 1, hy1),
        (vx1 + 1, hy0, FLOOR_W - 2, hy1),
        (vx0, 1, vx1, hy0 - 1),
        (vx0, hy1 + 1, vx1, FLOOR_H - 2),
        (vx0, hy0, vx1, hy1),
    ])

    blocks = [
        # block, corridor-facing sides, preferred object side
        ((1, 1, vx0 - 2, hy0 - 2), ("south", "east"), "north"),
        ((vx1 + 2, 1, FLOOR_W - 2, hy0 - 2), ("south", "west"), "north"),
        ((1, hy1 + 2, vx0 - 2, FLOOR_H - 2), ("north", "east"), "south"),
        ((vx1 + 2, hy1 + 2, FLOOR_W - 2, FLOOR_H - 2), ("north", "west"), "south"),
    ]

    room_entries = []
    for block, corridor_sides, object_side in blocks:
        x0, y0, x1, y1 = block
        if x1 < x0 or y1 < y0:
            raise RuntimeError("Invalid room block in free topology")

        # Two wall segments bound each block against the crossing corridors.
        for side in corridor_sides:
            if side == "north":
                region = (x0, y0 - 1, x1, y0 - 1)
            elif side == "south":
                region = (x0, y1 + 1, x1, y1 + 1)
            elif side == "west":
                region = (x0 - 1, y0, x0 - 1, y1)
            else:
                region = (x1 + 1, y0, x1 + 1, y1)
            _write_wall(grid, floor, region)

        # The horizontal arm borders the full row of rooms.  A separate
        # three-cell opening is created for every room below, so the
        # clearance-aware planner never has to squeeze around a partition
        # intersection to enter a room.  An additional vertical-arm entrance
        # keeps the topology less regular and offers loop closures.
        horizontal_side = next(
            side for side in corridor_sides
            if side in ("north", "south")
        )
        vertical_side = next(
            side for side in corridor_sides
            if side in ("east", "west")
        )
        if rng.random() < 0.75:
            _open_door(
                grid,
                floor,
                _corridor_door_for_side(
                    block, vertical_side, stair_regions, rng
                ),
            )

        block_width = x1 - x0 + 1
        room_count = (3 if block_width >=
                      3 * FREE_MIN_ROOM_WIDTH_CELLS + 2 else 2)
        spans, partitions = _partition_span_count(
            x0,
            x1,
            room_count,
            FREE_MIN_ROOM_WIDTH_CELLS,
            rng,
        )

        # Vertical partitions create a varied sequence of rooms. Every
        # partition contains a three-cell opening, so all rooms remain connected
        # to the block's corridor entrance.
        for partition_x in partitions:
            wall_region = (partition_x, y0, partition_x, y1)
            _write_wall(grid, floor, wall_region)
            door_y0 = _choose_door_start(y0, y1, [], "y", rng)
            _open_door(
                grid,
                floor,
                (partition_x, door_y0,
                 partition_x, door_y0 + DOOR_WIDTH_CELLS - 1),
            )

        for room_x0, room_x1 in spans:
            room_region = (room_x0, y0, room_x1, y1)
            if not _room_is_wide_enough(room_region):
                _seal_narrow_room(grid, floor, room_region)
                continue
            floor.room_regions.append(room_region)
            room_entries.append((room_region, object_side))
            _open_door(
                grid,
                floor,
                _corridor_door_for_side(
                    room_region,
                    horizontal_side,
                    stair_regions,
                    rng,
                ),
            )

    if len(floor.wall_regions) < FREE_MIN_WALL_REGIONS:
        raise RuntimeError("Dense free topology did not create enough walls")
    if len(floor.room_regions) < FREE_MIN_ROOM_COUNT:
        raise RuntimeError("Dense free topology did not create enough rooms")
    if not _traversable_connected(grid):
        raise RuntimeError("Dense free topology is not connected")
    return room_entries

# =========================================================================
# Shared finalization
# =========================================================================
def _place_victims(grid, floor, rng):
    candidates = []
    if floor.room_regions:
        for x0, y0, x1, y1 in floor.room_regions:
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if grid[y, x] == FREE:
                        candidates.append((x, y))
    else:
        ys, xs = np.where(grid == FREE)
        candidates = list(zip(xs.tolist(), ys.tolist()))
    rng.shuffle(candidates)

    n_victims = rng.randint(1, 2)
    for vx, vy in candidates:
        if any(abs(vx - px) < 3 and abs(vy - py) < 3
               for px, py in floor.victims):
            continue
        floor.victims.append((vx, vy))
        if len(floor.victims) >= n_victims:
            break


def _write_stairs(floors, stair_cores):
    """Write adjacent stair pairs and their corridor-aligned directions."""
    for index, floor in enumerate(floors):
        floor.stair_up_regions = []
        floor.stair_down_regions = []
        floor.stair_core_ids = list(range(len(stair_cores)))
        floor.stair_core_count = len(stair_cores)
        floor.stair_core_directions = [direction for _, _, direction in stair_cores]
        for core_id, (up_region, down_region, direction) in enumerate(stair_cores):
            if index < len(floors) - 1:
                x0, y0, x1, y1 = up_region
                floor.grid[y0:y1 + 1, x0:x1 + 1] = STAIR_UP
                floor.stair_up_regions.append(up_region)
            if index > 0:
                x0, y0, x1, y1 = down_region
                floor.grid[y0:y1 + 1, x0:x1 + 1] = STAIR_DOWN
                floor.stair_down_regions.append(down_region)
        floor.stair_up_region = (floor.stair_up_regions[0]
                                 if floor.stair_up_regions else None)
        floor.stair_down_region = (floor.stair_down_regions[0]
                                   if floor.stair_down_regions else None)


def _validate_room_spans(floors):
    for floor in floors:
        for region in floor.room_regions:
            x0, y0, x1, y1 = region
            if (x1 - x0 + 1 < MIN_ROOM_SPAN_CELLS or
                    y1 - y0 + 1 < MIN_ROOM_SPAN_CELLS):
                raise RuntimeError(
                    f"Room narrower than {MIN_ROOM_SPAN_CELLS} cells: {region}"
                )

def _finalize_floor(floor, rng):
    traversable = np.isin(
        floor.grid,
        [FREE, DOOR, STAIR_UP, STAIR_DOWN],
    )
    floor.explorable_area_m2 = (
        float(np.sum(traversable)) * CELL_SIZE ** 2
    )
    _place_victims(floor.grid, floor, rng)

    # Packing caches are useful only while clutter is being inserted. Clearing
    # them keeps cached pristine building templates compact and prevents a long
    # batch from retaining thousands of rejected candidate rectangles.
    floor._rejected_object_footprints.clear()
    floor._object_placement_saturated = False


def _new_floor_shell(index: int, n_floors: int, layout_mode: str):
    """Create a clean floor with only the immutable perimeter walls."""
    floor = Floor(index, n_floors)
    floor.layout_mode = layout_mode
    floor.grid[:, :] = FREE
    floor.grid[0, :] = WALL
    floor.grid[-1, :] = WALL
    floor.grid[:, 0] = WALL
    floor.grid[:, -1] = WALL
    return floor


def _generate_building_uncached(
    n_floors: int,
    seed=None,
    layout_mode: str = LAYOUT_OFFICE,
    include_objects: bool = True,
    object_density: float = 1.0,
):
    """Generate one deterministic multi-floor building.

    Stair cores are reserved from the shared corridor template before any
    floor-specific doors are sampled.  This is essential for tall buildings:
    searching for a location only after six independent sets of doors had been
    generated made the legal intersection disappear for some seeds.

    If a particular room subdivision cannot respect the reservation, only that
    floor is regenerated.  If the whole template is unusually restrictive, a
    new template is sampled from the same seeded RNG.  Therefore the outcome is
    still perfectly reproducible and remains identical for every Fw x Opt
    condition belonging to the same run.
    """
    if n_floors < 1:
        raise ValueError("n_floors must be >= 1")

    layout_mode = normalize_layout_mode(layout_mode)
    object_density = normalize_object_density(object_density)
    rng = random.Random(seed)
    two_cores = bool(int(seed or 0) % STAIR_CORE_TWO_PROBABILITY_DENOMINATOR == 0)
    last_error = None

    for layout_attempt in range(MAX_BUILDING_LAYOUT_ATTEMPTS):
        try:
            if layout_mode == LAYOUT_OFFICE:
                office_variant, corridor_regions = _office_template(rng)
                stair_cores = (
                    _select_stair_cores_from_corridors(
                        corridor_regions, rng, two_cores=two_cores
                    )
                    if n_floors > 1 else []
                )
                stair_regions = _stair_reservation_regions(stair_cores)
                floors = []
                room_entries_by_floor = []

                for floor_index in range(n_floors):
                    floor_error = None
                    for _ in range(MAX_FLOOR_LAYOUT_ATTEMPTS):
                        floor = _new_floor_shell(
                            floor_index, n_floors, layout_mode
                        )
                        try:
                            room_entries = _build_office_variant(
                                floor.grid, floor, office_variant,
                                corridor_regions, stair_regions, rng,
                            )
                        except RuntimeError as exc:
                            floor_error = exc
                            continue
                        if not _stair_cores_fit_floor(floor, stair_cores):
                            floor_error = RuntimeError(
                                "room doors conflict with reserved stair core"
                            )
                            continue
                        floors.append(floor)
                        room_entries_by_floor.append(room_entries)
                        break
                    else:
                        raise RuntimeError(
                            f"floor {floor_index + 1} could not preserve stair "
                            f"reservations: {floor_error}"
                        )
            else:
                horizontal_corridor, vertical_corridor = _free_cross_spine(rng)
                corridor_regions = [horizontal_corridor, vertical_corridor]
                stair_cores = (
                    _select_stair_cores_from_corridors(
                        corridor_regions, rng, two_cores=two_cores
                    )
                    if n_floors > 1 else []
                )
                stair_regions = _stair_reservation_regions(stair_cores)
                floors = []
                room_entries_by_floor = []

                for floor_index in range(n_floors):
                    floor_error = None
                    for _ in range(MAX_FLOOR_LAYOUT_ATTEMPTS):
                        floor = _new_floor_shell(
                            floor_index, n_floors, layout_mode
                        )
                        try:
                            room_entries = _build_free_cross_layout(
                                floor.grid, floor, horizontal_corridor,
                                vertical_corridor, stair_regions, rng,
                            )
                        except (RuntimeError, ValueError) as exc:
                            floor_error = exc
                            continue
                        if not _stair_cores_fit_floor(floor, stair_cores):
                            floor_error = RuntimeError(
                                "free-layout doors conflict with reserved stairs"
                            )
                            continue
                        floors.append(floor)
                        room_entries_by_floor.append(room_entries)
                        break
                    else:
                        raise RuntimeError(
                            f"floor {floor_index + 1} could not preserve stair "
                            f"reservations: {floor_error}"
                        )

            # Furniture is generated only after a valid common stair layout has
            # been established.  Objects are room-local, but the post-check is
            # intentionally retained as a guard against future placement code.
            if include_objects:
                for floor, room_entries in zip(
                        floors, room_entries_by_floor):
                    _place_office_objects(
                        floor.grid, floor, room_entries, rng, object_density
                    )

            if any(not _stair_cores_fit_floor(floor, stair_cores)
                   for floor in floors):
                raise RuntimeError(
                    "static objects or room geometry occupied a reserved stair core"
                )

            _validate_room_spans(floors)
            _write_stairs(floors, stair_cores)
            for floor in floors:
                _finalize_floor(floor, rng)
            return floors

        except (RuntimeError, ValueError) as exc:
            last_error = exc
            # Continue with a new template/subdivision drawn from the same RNG.
            # No unseeded randomness is introduced by this recovery path.
            continue

    raise RuntimeError(
        "Unable to generate a valid multi-floor building after "
        f"{MAX_BUILDING_LAYOUT_ATTEMPTS} deterministic layout attempts: "
        f"{last_error}"
    )


@lru_cache(maxsize=12)
def _cached_building_template(
    n_floors: int,
    seed: int,
    layout_mode: str,
    include_objects: bool,
    object_density: float,
):
    """Return one pristine deterministic template for repeated conditions.

    A single experimental building is reconstructed many times: once for each
    start diagnostic and once for every Fw x Opt episode.  Geometry depends on
    none of those planner parameters.  Caching the pristine pre-map template
    avoids packing the same dense furniture repeatedly while ``deepcopy`` in
    :func:`generate_building` guarantees that maps and episode state can never
    leak between conditions.  The small LRU retains only nearby runs.
    """
    return _generate_building_uncached(
        n_floors=n_floors,
        seed=seed,
        layout_mode=layout_mode,
        include_objects=include_objects,
        object_density=object_density,
    )


def clear_building_cache() -> None:
    """Clear deterministic templates, mainly for tests and very long batches."""
    _cached_building_template.cache_clear()


def generate_building(
    n_floors: int,
    seed=None,
    layout_mode: str = LAYOUT_OFFICE,
    include_objects: bool = True,
    object_density: float = 1.0,
):
    """Return a fresh building, reusing deterministic geometry when possible.

    ``seed=None`` intentionally remains uncached because ``random.Random(None)``
    requests a new non-deterministic building.  Integer-seeded experiment runs
    receive an independent deep copy of a cached pristine template.
    """
    layout_mode = normalize_layout_mode(layout_mode)
    object_density = normalize_object_density(object_density)
    if seed is None:
        return _generate_building_uncached(
            n_floors=n_floors,
            seed=None,
            layout_mode=layout_mode,
            include_objects=bool(include_objects),
            object_density=object_density,
        )
    template = _cached_building_template(
        int(n_floors),
        int(seed),
        layout_mode,
        bool(include_objects),
        float(object_density),
    )
    return copy.deepcopy(template)


def is_blocking(cell_value):
    return cell_value == WALL


def is_traversable(cell_value):
    return cell_value in (FREE, DOOR, STAIR_UP, STAIR_DOWN)
