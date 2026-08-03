"""GPU-accelerated first-person camera for the SAR simulator.

The main simulator remains Pygame based.  This module owns a second process in
which Panda3D renders the same immutable ground-truth building.  The parent
sends only the robot pose, so opening the camera does not alter planning,
physics, mapping, random seeds, or experimental results.

Panda3D is imported only in the child process.  Consequently the normal 2-D
simulator can still be imported and inspected on a machine where Panda3D has
not yet been installed; attempting to open the 3-D window then exits the child
cleanly and prints an actionable dependency message.
"""
from __future__ import annotations

from i18n import tr

import math
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# Ground-truth values shared with building.py.  They are kept local to avoid
# importing the entire simulator inside the rendering process.
FREE, WALL, DOOR, STAIR_UP, STAIR_DOWN = 0, 1, 2, 3, 4
CELL_SIZE = 0.5
CAMERA_HEIGHT_M = 0.72
# The front cameras are mounted slightly ahead of the robot centre.  Keeping
# this offset explicit prevents the viewpoint from appearing to originate from
# the robot body or, after a turn, from its rear.
CAMERA_FORWARD_OFFSET_M = 0.28
CAMERA_LOOK_AHEAD_M = 4.0
CAMERA_LOOK_DOWN_M = 0.04
CAMERA_FOV_DEG = 74.0
WINDOW_W = 1280
WINDOW_H = 720
TARGET_FPS = 60

# Very small first-person vertical motion that suggests the gait of a
# quadruped without producing an uncomfortable handheld-camera effect.
CAMERA_BOB_AMPLITUDE_M = 0.018
CAMERA_BOB_FREQUENCY_HZ = 1.75
CAMERA_BOB_MIN_SPEED_M_S = 0.04
CAMERA_BOB_SMOOTHING = 0.12

WALL_HEIGHT_M = 2.70
TABLE_HEIGHT_M = 0.78
CHAIR_SEAT_HEIGHT_M = 0.47
CHAIR_BACK_HEIGHT_M = 0.92
DEBRIS_MAX_HEIGHT_M = 0.42
PERSON_HEIGHT_M = 1.70


@dataclass
class CameraHandle:
    """Parent-side lifecycle wrapper for the external Panda3D process."""

    process: mp.Process | None = None
    connection: Any | None = None

    @property
    def active(self) -> bool:
        return bool(self.process is not None and self.process.is_alive())

    def open(self, floors, initial_pose):
        """Start the renderer and transfer static scene data exactly once."""
        if self.active:
            return
        self.close()
        scene = _serialise_scene(floors)
        parent_conn, child_conn = mp.Pipe(duplex=True)
        process = mp.get_context("spawn").Process(
            target=_camera_process,
            args=(child_conn, scene, tuple(initial_pose)),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self.process = process
        self.connection = parent_conn

    def update_pose(self, floor_index, x, y, theta):
        """Send a pose without blocking the deterministic simulation loop."""
        if not self.active or self.connection is None:
            return
        try:
            self.connection.send(("pose", int(floor_index), float(x), float(y), float(theta)))
        except (BrokenPipeError, EOFError, OSError):
            self.close()

    def close(self):
        """Request graceful shutdown and terminate only as a final fallback."""
        if self.connection is not None:
            try:
                self.connection.send(("close",))
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                self.connection.close()
            except OSError:
                pass
        if self.process is not None:
            self.process.join(timeout=0.8)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.5)
        self.process = None
        self.connection = None


def _serialise_scene(floors):
    """Convert simulator objects to small, deterministic, pickle-safe records."""
    result = []
    for floor_index, floor in enumerate(floors):
        grid = np.asarray(floor.grid, dtype=np.int8).copy()
        object_kind = np.full(grid.shape, "", dtype="U8")
        objects = []
        for item in getattr(floor, "environment_objects", []):
            x0, y0, x1, y1 = map(int, item.region)
            object_kind[y0:y1 + 1, x0:x1 + 1] = item.kind
            objects.append({
                "kind": str(item.kind),
                "region": (x0, y0, x1, y1),
                "rotation": int(getattr(item, "rotation_quarters", 0)) % 4,
            })
        result.append({
            "floor_index": floor_index,
            "grid": grid,
            "object_kind": object_kind,
            "objects": objects,
            "victims": [tuple(map(int, victim)) for victim in getattr(floor, "victims", [])],
            "stair_dirs": [tuple(map(int, d)) for d in (getattr(floor, "stair_core_directions", []) or [])],
            "stair_up": [tuple(map(int, r)) for r in (getattr(floor, "stair_up_regions", []) or [])],
            "stair_down": [tuple(map(int, r)) for r in (getattr(floor, "stair_down_regions", []) or [])],
        })
    return result



def _camera_pose_from_robot(x, y, theta):
    """Map a simulator pose to Panda3D camera and look-at coordinates.

    This pure helper is intentionally independent of Panda3D so the coordinate
    convention can be regression-tested on machines without a graphics stack.
    """
    forward_x = math.cos(float(theta))
    forward_y = -math.sin(float(theta))
    camera_x = float(x) + CAMERA_FORWARD_OFFSET_M * forward_x
    camera_y = -float(y) + CAMERA_FORWARD_OFFSET_M * forward_y
    camera_z = CAMERA_HEIGHT_M
    target_x = camera_x + CAMERA_LOOK_AHEAD_M * forward_x
    target_y = camera_y + CAMERA_LOOK_AHEAD_M * forward_y
    target_z = camera_z - CAMERA_LOOK_DOWN_M
    return (camera_x, camera_y, camera_z), (target_x, target_y, target_z)

def _camera_process(connection, scene, initial_pose):
    """Entry point executed in the child process."""
    try:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", f"win-size {WINDOW_W} {WINDOW_H}")
        loadPrcFileData("", f"window-title {tr('camera_title')}")
        loadPrcFileData("", "sync-video true")
        loadPrcFileData("", "show-frame-rate-meter true")
        loadPrcFileData("", "framebuffer-multisample true")
        loadPrcFileData("", "multisamples 4")
        loadPrcFileData("", "texture-minfilter linear-mipmap-linear")
        loadPrcFileData("", "texture-magfilter linear")
        loadPrcFileData("", "audio-library-name null")
        from direct.showbase.ShowBase import ShowBase
    except ImportError:
        print("La vista 3D richiede Panda3D. Installare con: python -m pip install panda3d")
        try:
            connection.close()
        except OSError:
            pass
        return

    class SARCameraApp(ShowBase):
        def __init__(self):
            super().__init__(windowType="onscreen")
            self.disableMouse()
            self.connection = connection
            self.scene_data = scene
            self.pose = tuple(initial_pose)
            self.floor_nodes = []
            # Camera gait state is visual only.  It is estimated from successive
            # poses received from the simulator and never feeds back into the
            # robot dynamics, planner, mapping, or experimental measurements.
            self._previous_pose = tuple(initial_pose)
            self._previous_pose_time = time.perf_counter()
            self._camera_bob_phase = 0.0
            self._camera_bob_height = 0.0
            self._estimated_speed = 0.0
            self._configure_camera()
            self._configure_lighting()
            self._build_world()
            self.accept("escape", self.userExit)
            self.taskMgr.add(self._update_task, "sar-camera-update", sort=-50)

        def _configure_camera(self):
            self.camLens.setFov(CAMERA_FOV_DEG)
            self.camLens.setNearFar(0.03, 80.0)
            self.setBackgroundColor(0.12, 0.16, 0.20, 1.0)

        def _configure_lighting(self):
            from panda3d.core import AmbientLight, DirectionalLight, Vec4
            ambient = AmbientLight("office-ambient")
            ambient.setColor(Vec4(0.52, 0.54, 0.58, 1.0))
            self.render.setLight(self.render.attachNewNode(ambient))
            key = DirectionalLight("office-key")
            key.setColor(Vec4(0.86, 0.84, 0.78, 1.0))
            key_np = self.render.attachNewNode(key)
            key_np.setHpr(-35, -58, 0)
            self.render.setLight(key_np)
            fill = DirectionalLight("office-fill")
            fill.setColor(Vec4(0.28, 0.31, 0.36, 1.0))
            fill_np = self.render.attachNewNode(fill)
            fill_np.setHpr(140, -25, 0)
            self.render.setLight(fill_np)
            self.render.setShaderAuto()

        def _build_world(self):
            for floor in self.scene_data:
                root = self.render.attachNewNode(f"floor-{floor['floor_index']}")
                # Simulator coordinates use +Y downward on the 2-D map, while
                # Panda3D is a right-handed system.  Mirroring the whole floor
                # on Y preserves the simulator's left/right convention.  The
                # previous direct mapping was the source of the occasional
                # apparent rear camera and left/right reversal.
                root.setScale(1.0, -1.0, 1.0)
                root.setTwoSided(True)
                root.hide()
                _build_floor_scene(self, root, floor)
                self.floor_nodes.append(root)
            self._show_floor(int(self.pose[0]))

        def _show_floor(self, index):
            for i, node in enumerate(self.floor_nodes):
                if i == index:
                    node.show()
                else:
                    node.hide()

        def _update_task(self, task):
            from direct.task import Task
            try:
                while self.connection.poll():
                    message = self.connection.recv()
                    if not message or message[0] == "close":
                        self.userExit()
                        return Task.done
                    if message[0] == "pose":
                        old_floor = int(self.pose[0])
                        old_pose = self.pose
                        new_pose = tuple(message[1:])
                        now = time.perf_counter()
                        elapsed = max(1e-4, now - self._previous_pose_time)
                        if int(new_pose[0]) == int(old_pose[0]):
                            distance = math.hypot(float(new_pose[1]) - float(old_pose[1]),
                                                  float(new_pose[2]) - float(old_pose[2]))
                            raw_speed = distance / elapsed
                        else:
                            # A floor transition is teleportation in the visual
                            # process and must not be interpreted as a fast step.
                            raw_speed = 0.0
                        self._estimated_speed = 0.72 * self._estimated_speed + 0.28 * raw_speed
                        self._previous_pose = old_pose
                        self._previous_pose_time = now
                        self.pose = new_pose
                        if int(self.pose[0]) != old_floor:
                            self._show_floor(int(self.pose[0]))
            except (EOFError, OSError):
                self.userExit()
                return Task.done

            floor_index, x, y, theta = self.pose

            # Do not derive the camera orientation from Panda3D Euler-angle
            # conventions.  Instead, transform the simulator's forward vector
            # into Panda3D coordinates and use lookAt().  This makes the camera
            # unambiguously point along the robot's nose for every heading.
            #
            # Simulator: +X right, +Y down.
            # Panda3D:    +X right, +Y forward, +Z up.
            # Therefore (x, y) -> (x, -y) and (cos t, sin t) ->
            # (cos t, -sin t).  This also preserves which side is left/right.
            camera_pos, look_at = _camera_pose_from_robot(x, y, theta)

            # A quadruped's body rises and falls gently as diagonal leg pairs
            # alternate.  The oscillation is activated only while translating,
            # fades smoothly at rest, and is deliberately kept below 2 cm.
            dt = max(0.0, min(0.05, globalClock.getDt()))
            moving = self._estimated_speed >= CAMERA_BOB_MIN_SPEED_M_S
            target_amplitude = CAMERA_BOB_AMPLITUDE_M if moving else 0.0
            blend = 1.0 - math.pow(1.0 - CAMERA_BOB_SMOOTHING, max(1.0, dt * 60.0))
            current_amplitude = abs(self._camera_bob_height)
            amplitude = current_amplitude + (target_amplitude - current_amplitude) * blend
            if moving:
                cadence_scale = max(0.75, min(1.45, self._estimated_speed / 0.55))
                self._camera_bob_phase += 2.0 * math.pi * CAMERA_BOB_FREQUENCY_HZ * cadence_scale * dt
                self._camera_bob_height = amplitude * math.sin(self._camera_bob_phase)
            else:
                self._camera_bob_height *= max(0.0, 1.0 - 7.0 * dt)

            camera_pos = (camera_pos[0], camera_pos[1], camera_pos[2] + self._camera_bob_height)
            look_at = (look_at[0], look_at[1], look_at[2] + self._camera_bob_height * 0.45)
            self.camera.setPos(*camera_pos)
            self.camera.lookAt(*look_at)
            return Task.cont

        def userExit(self):
            try:
                self.connection.close()
            except OSError:
                pass
            super().userExit()

    SARCameraApp().run()


def _build_floor_scene(app, root, floor):
    """Create one immutable floor scene, resident in GPU memory."""
    from panda3d.core import CardMaker, PNMImage, Texture, TextureStage, TransparencyAttrib

    grid = floor["grid"]
    height, width = grid.shape
    world_w, world_h = width * CELL_SIZE, height * CELL_SIZE

    # Procedural carpet texture.  A tiny texture repeats over the floor and is
    # uploaded only once; no per-frame texture generation occurs.
    image = PNMImage(64, 64, 3)
    for py in range(64):
        for px in range(64):
            checker = ((px // 8) + (py // 8)) % 2
            noise = ((px * 17 + py * 29 + floor["floor_index"] * 13) % 11) / 255.0
            base = 0.35 + 0.025 * checker + noise
            image.setXel(px, py, base, base * 1.03, base * 1.08)
    texture = Texture(f"carpet-{floor['floor_index']}")
    texture.load(image)
    texture.setWrapU(Texture.WM_repeat)
    texture.setWrapV(Texture.WM_repeat)

    card = CardMaker("floor-card")
    card.setFrame(0, world_w, 0, world_h)
    floor_np = root.attachNewNode(card.generate())
    floor_np.setP(-90)
    floor_np.setPos(0, 0, 0)
    floor_np.setTexture(texture)
    floor_np.setTexScale(TextureStage.getDefault(), max(1, width / 6), max(1, height / 6))

    # Ceiling gives the camera a credible enclosed-office view.  It has no
    # physical effect and does not alter the simulator's collision geometry.
    ceiling = _make_box(app, root, world_w / 2, world_h / 2, WALL_HEIGHT_M + 0.04,
                        world_w, world_h, 0.08, (0.77, 0.78, 0.80, 1.0))
    ceiling.setTwoSided(True)

    structural = (grid == WALL) & (floor["object_kind"] == "")
    wall_rectangles = _merge_rectangles(structural)

    # Window positions are selected before any wall decoration is created.
    # This lets every other decorative subsystem reserve the wall and the
    # floor strip in front of a window, preventing bookcases, pictures, plants,
    # bins, tables, chairs, debris, or victims from visually blocking it.
    window_specs = []
    for rectangle in wall_rectangles:
        window_specs.extend(_window_specs_for_wall(floor, *rectangle))
    floor["window_specs"] = window_specs

    # Choose two distinct long indoor wall faces, when available, so each
    # floor visibly contains both the pre-existing tall storage unit and the
    # new open bookshelf.  The plan is derived only from geometry and floor
    # index; it never consumes the simulator's random-number generators.
    wall_decor_plan = _plan_required_wall_furniture(
        floor["floor_index"], grid, wall_rectangles, window_specs
    )

    for x0, y0, x1, y1 in wall_rectangles:
        _make_box_for_region(app, root, (x0, y0, x1, y1), WALL_HEIGHT_M,
                             (0.74, 0.75, 0.73, 1.0), z0=0.0)
        wall_windows = [spec for spec in window_specs if spec["wall"] == (x0, y0, x1, y1)]
        if wall_windows:
            _add_perimeter_windows(app, root, wall_windows)
        else:
            _add_wall_decorations(
                app, root, floor["floor_index"], grid, x0, y0, x1, y1,
                forced_styles=wall_decor_plan,
            )

    # Ceiling fixtures are visual-only.  Some are illuminated and a
    # deterministic minority are dark, suggesting partial electrical failure.
    _add_ceiling_lights(app, root, floor)

    # Door cells stay open.  A thin green threshold makes openings legible.
    door_mask = grid == DOOR
    for x0, y0, x1, y1 in _merge_rectangles(door_mask):
        _make_box_for_region(app, root, (x0, y0, x1, y1), 0.025,
                             (0.26, 0.52, 0.24, 1.0), z0=0.002)

    for obj in floor["objects"]:
        if obj["kind"] == "table":
            _build_table(app, root, obj)
        elif obj["kind"] == "chair":
            _build_chair(app, root, obj)
        elif obj["kind"] == "debris":
            _build_debris(app, root, obj, floor["floor_index"])

    _build_stairs(app, root, floor)
    for victim_index, victim in enumerate(floor["victims"]):
        _build_person(app, root, victim, floor["floor_index"], victim_index)

    # Plants and small office accessories are superficial scene dressing.
    # They are deliberately absent from floor.grid, so they do not become
    # collision obstacles or alter LiDAR, occupancy mapping, or A*.
    _add_wall_plants_and_accessories(app, root, floor)

    # Paper sheets are thin visual decals distributed throughout rooms and
    # corridors.  They are deliberately generated only in the Panda3D scene:
    # no cell is marked occupied, so LiDAR, collision checking, mapping and A*
    # remain exactly as in v30.
    _add_scattered_floor_papers(app, root, floor)


def _merge_rectangles(mask):
    """Greedily merge True cells into rectangles to reduce draw calls."""
    mask = np.asarray(mask, dtype=bool)
    used = np.zeros_like(mask)
    h, w = mask.shape
    rectangles = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or used[y, x]:
                continue
            x1 = x
            while x1 + 1 < w and mask[y, x1 + 1] and not used[y, x1 + 1]:
                x1 += 1
            y1 = y
            while y1 + 1 < h and np.all(mask[y1 + 1, x:x1 + 1] & ~used[y1 + 1, x:x1 + 1]):
                y1 += 1
            used[y:y1 + 1, x:x1 + 1] = True
            rectangles.append((x, y, x1, y1))
    return rectangles


def _unit_cube(app):
    """Return a shared Panda3D cube model, generated once per process."""
    if hasattr(app, "_sar_unit_cube"):
        return app._sar_unit_cube
    from panda3d.core import Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter
    fmt = GeomVertexFormat.getV3n3()
    data = GeomVertexData("unit-cube", fmt, Geom.UH_static)
    vertex = GeomVertexWriter(data, "vertex")
    normal = GeomVertexWriter(data, "normal")
    faces = [
        ((1, 0, 0), [(0.5,-0.5,-0.5),(0.5,0.5,-0.5),(0.5,0.5,0.5),(0.5,-0.5,0.5)]),
        ((-1,0,0), [(-0.5,0.5,-0.5),(-0.5,-0.5,-0.5),(-0.5,-0.5,0.5),(-0.5,0.5,0.5)]),
        ((0,1,0), [(-0.5,0.5,-0.5),(0.5,0.5,-0.5),(0.5,0.5,0.5),(-0.5,0.5,0.5)]),
        ((0,-1,0), [(0.5,-0.5,-0.5),(-0.5,-0.5,-0.5),(-0.5,-0.5,0.5),(0.5,-0.5,0.5)]),
        ((0,0,1), [(-0.5,-0.5,0.5),(0.5,-0.5,0.5),(0.5,0.5,0.5),(-0.5,0.5,0.5)]),
        ((0,0,-1), [(-0.5,0.5,-0.5),(0.5,0.5,-0.5),(0.5,-0.5,-0.5),(-0.5,-0.5,-0.5)]),
    ]
    prim = GeomTriangles(Geom.UH_static)
    index = 0
    for n, verts in faces:
        for v in verts:
            vertex.addData3(*v); normal.addData3(*n)
        prim.addVertices(index, index+1, index+2)
        prim.addVertices(index, index+2, index+3)
        index += 4
    geom = Geom(data); geom.addPrimitive(prim)
    node = GeomNode("unit-cube"); node.addGeom(geom)
    app._sar_unit_cube = app.render.attachNewNode(node)
    app._sar_unit_cube.hide()
    return app._sar_unit_cube


def _make_box(app, parent, cx, cy, cz, sx, sy, sz, color):
    cube = _unit_cube(app).copyTo(parent)
    cube.show()
    cube.setPos(cx, cy, cz)
    cube.setScale(max(0.001, sx), max(0.001, sy), max(0.001, sz))
    cube.setColor(*color)
    return cube


def _make_box_for_region(app, parent, region, height, color, z0=0.0, inset=0.0):
    x0, y0, x1, y1 = region
    sx = max(0.02, (x1 - x0 + 1) * CELL_SIZE - 2 * inset)
    sy = max(0.02, (y1 - y0 + 1) * CELL_SIZE - 2 * inset)
    cx = (x0 + x1 + 1) * CELL_SIZE / 2
    cy = (y0 + y1 + 1) * CELL_SIZE / 2
    return _make_box(app, parent, cx, cy, z0 + height / 2, sx, sy, height, color)


def _wall_free_sides(grid, x0, y0, x1, y1):
    """Return wall faces that border usable indoor space.

    Decorations used to be attached mechanically to the negative side of a
    merged wall rectangle.  On many layouts that side was either outdoors or
    buried inside a thick wall, which is why pictures and shelves seemed to
    disappear in the Panda3D version.  This helper inspects the ground-truth
    grid and returns only faces adjacent to non-wall cells.
    """
    h, w = grid.shape
    sides = []
    horizontal = (x1 - x0) >= (y1 - y0)
    if horizontal:
        if y0 > 0 and np.any(grid[y0 - 1, x0:x1 + 1] != WALL):
            sides.append("north")
        if y1 + 1 < h and np.any(grid[y1 + 1, x0:x1 + 1] != WALL):
            sides.append("south")
    else:
        if x0 > 0 and np.any(grid[y0:y1 + 1, x0 - 1] != WALL):
            sides.append("west")
        if x1 + 1 < w and np.any(grid[y0:y1 + 1, x1 + 1] != WALL):
            sides.append("east")
    return sides


def _plan_required_wall_furniture(floor_index, grid, wall_rectangles, window_specs):
    """Reserve wall faces for one cabinet and one open bookshelf per floor.

    Long wall faces that contain windows are excluded.  Remaining candidates
    are sorted by a coordinate-derived key, making the result deterministic.
    The existing deep storage unit is preserved under the ``cabinet`` style;
    the new ``bookshelf`` style is assigned to a different face whenever the
    layout offers at least two suitable faces.
    """
    window_walls = {tuple(spec["wall"]) for spec in window_specs}
    candidates = []
    for rectangle in wall_rectangles:
        x0, y0, x1, y1 = rectangle
        if tuple(rectangle) in window_walls:
            continue
        length_cells = max(x1 - x0 + 1, y1 - y0 + 1)
        if length_cells < 7:
            continue
        for side_index, side in enumerate(_wall_free_sides(grid, x0, y0, x1, y1)):
            key = (floor_index * 73856093 + x0 * 19349663 + y0 * 83492791
                   + x1 * 31 + y1 * 17 + side_index * 2654435761) & 0xffffffff
            candidates.append((key, tuple(rectangle), side))

    candidates.sort(key=lambda item: item[0])
    plan = {}
    if candidates:
        _, rectangle, side = candidates[0]
        plan[(rectangle, side)] = "cabinet"
    if len(candidates) >= 2:
        _, rectangle, side = candidates[1]
        plan[(rectangle, side)] = "bookshelf"
    elif candidates:
        # Extremely sparse layouts can expose only one long wall face.  In
        # that rare case the open bookshelf takes precedence, while cabinets
        # may still be produced by the normal deterministic decoration rule.
        _, rectangle, side = candidates[0]
        plan[(rectangle, side)] = "bookshelf"
    return plan


def _add_wall_decorations(app, root, floor_index, grid, x0, y0, x1, y1,
                          forced_styles=None):
    """Populate visible wall faces with grouped, deterministic office decor.

    The original tall storage/book cabinet remains available unchanged.  A
    second, visually distinct open bookshelf has explicit horizontal shelves
    and dense rows of coloured spines.  Neither object enters the simulator's
    occupancy or collision structures.
    """
    length_cells = max(x1 - x0 + 1, y1 - y0 + 1)
    sides = _wall_free_sides(grid, x0, y0, x1, y1)
    if not sides:
        return

    for side in sides:
        _add_baseboard(app, root, side, x0, y0, x1, y1)

    if length_cells < 4:
        return

    forced_styles = forced_styles or {}
    rectangle = (x0, y0, x1, y1)
    for side_index, side in enumerate(sides):
        key = (floor_index * 73856093 + x0 * 19349663 + y0 * 83492791
               + x1 * 31 + y1 * 17 + side_index * 2654435761) & 0xffffffff
        style = forced_styles.get((rectangle, side))
        if style == "cabinet":
            _add_bookcase(app, root, side, x0, y0, x1, y1, key)
        elif style == "bookshelf":
            _add_open_bookshelf(app, root, side, x0, y0, x1, y1, key)
        elif length_cells >= 7 and key % 8 in (0, 1):
            _add_bookcase(app, root, side, x0, y0, x1, y1, key)
        elif length_cells >= 7 and key % 8 in (2, 3):
            _add_open_bookshelf(app, root, side, x0, y0, x1, y1, key)
        elif key % 8 == 4:
            _add_noticeboard(app, root, side, x0, y0, x1, y1, key)
        else:
            _add_framed_art(app, root, side, x0, y0, x1, y1, key)



def _wall_face_pose(side, x0, y0, x1, y1, depth):
    """Return centre and dimensions for a thin object attached to a wall."""
    if side in ("north", "south"):
        cx = (x0 + x1 + 1) * CELL_SIZE / 2
        boundary = (y0 if side == "north" else y1 + 1) * CELL_SIZE
        cy = boundary + (-depth / 2 if side == "north" else depth / 2)
        return cx, cy, (x1 - x0 + 1) * CELL_SIZE, depth, True
    cy = (y0 + y1 + 1) * CELL_SIZE / 2
    boundary = (x0 if side == "west" else x1 + 1) * CELL_SIZE
    cx = boundary + (-depth / 2 if side == "west" else depth / 2)
    return cx, cy, depth, (y1 - y0 + 1) * CELL_SIZE, False


def _add_baseboard(app, root, side, x0, y0, x1, y1):
    cx, cy, sx, sy, _ = _wall_face_pose(side, x0, y0, x1, y1, 0.035)
    _make_box(app, root, cx, cy, 0.075, sx * 0.98, sy, 0.15,
              (0.20, 0.16, 0.12, 1.0))


def _add_bookcase(app, root, side, x0, y0, x1, y1, key):
    """Create a wall bookcase with dense rows of realistic visible spines.

    Only the room-facing spines are visually prominent.  Every volume rests on
    the shelf immediately below it, rather than floating at an arbitrary fixed
    height.  Width, height and colour vary slightly, while the books remain
    tightly aligned like a normal office library.
    """
    cx, cy, run_x, run_y, horizontal = _wall_face_pose(side, x0, y0, x1, y1, 0.18)
    run = run_x if horizontal else run_y
    width = min(2.25, max(1.05, run * 0.62))
    sx, sy = (width, 0.18) if horizontal else (0.18, width)

    # Cabinet back and structural boards.
    _make_box(app, root, cx, cy, 1.08, sx, sy, 1.95, (0.24, 0.14, 0.075, 1))
    shelf_levels = (0.34, 0.72, 1.10, 1.48, 1.86)
    shelf_thickness = 0.035
    for shelf_z in shelf_levels:
        _make_box(app, root, cx, cy, shelf_z, sx,
                  sy + (0.018 if horizontal else 0), shelf_thickness,
                  (0.10, 0.055, 0.025, 1))

    # Narrow vertical side boards make the rows read as a real bookcase.
    if horizontal:
        for edge_x in (cx - width / 2 + 0.025, cx + width / 2 - 0.025):
            _make_box(app, root, edge_x, cy, 1.08, 0.05, sy + 0.025, 1.95,
                      (0.11, 0.060, 0.028, 1))
    else:
        for edge_y in (cy - width / 2 + 0.025, cy + width / 2 - 0.025):
            _make_box(app, root, cx, edge_y, 1.08, sx + 0.025, 0.05, 1.95,
                      (0.11, 0.060, 0.028, 1))

    # The front face of every book is placed just inside the room-facing edge.
    room_sign = -1 if side in ("north", "west") else 1
    palette = [
        (0.13, 0.25, 0.45, 1), (0.50, 0.13, 0.12, 1),
        (0.12, 0.39, 0.20, 1), (0.61, 0.43, 0.10, 1),
        (0.34, 0.17, 0.48, 1), (0.08, 0.34, 0.42, 1),
        (0.62, 0.25, 0.08, 1), (0.38, 0.38, 0.40, 1),
    ]

    usable_width = width - 0.13
    for shelf_index, support_z in enumerate(shelf_levels[:-1]):
        # Deterministic widths are accumulated until the row is full.  This
        # produces many adjacent spines without overlaps or unrealistic gaps.
        cursor = -usable_width / 2
        book_index = 0
        while cursor < usable_width / 2 - 0.025:
            selector = (key + shelf_index * 101 + book_index * 37) & 0xffffffff
            book_w = 0.035 + 0.012 * (selector % 5)  # 3.5--8.3 cm spine
            remaining = usable_width / 2 - cursor
            book_w = min(book_w, remaining)
            if book_w < 0.022:
                break
            book_h = 0.245 + 0.018 * ((selector >> 3) % 6)
            book_depth = 0.125 + 0.008 * ((selector >> 7) % 4)
            gap = 0.004 + 0.002 * ((selector >> 11) % 2)
            along = cursor + book_w / 2
            z = support_z + shelf_thickness / 2 + book_h / 2
            color = palette[(selector >> 5) % len(palette)]

            if horizontal:
                bx = cx + along
                by = cy + room_sign * (sy / 2 - book_depth / 2 - 0.006)
                book = _make_box(app, root, bx, by, z,
                                 book_w, book_depth, book_h, color)
            else:
                bx = cx + room_sign * (sx / 2 - book_depth / 2 - 0.006)
                by = cy + along
                book = _make_box(app, root, bx, by, z,
                                 book_depth, book_w, book_h, color)

            # A very thin pale band on some spines suggests labels or title
            # strips while keeping only the spine visible from the room.
            if selector % 4 == 0:
                band_h = 0.012
                band_z = z + book_h * 0.22
                if horizontal:
                    _make_box(app, root, bx,
                              cy + room_sign * (sy / 2 + 0.001), band_z,
                              max(0.012, book_w * 0.72), 0.006, band_h,
                              (0.82, 0.78, 0.66, 1))
                else:
                    _make_box(app, root,
                              cx + room_sign * (sx / 2 + 0.001), by, band_z,
                              0.006, max(0.012, book_w * 0.72), band_h,
                              (0.82, 0.78, 0.66, 1))

            cursor += book_w + gap
            book_index += 1

    # This pre-existing deep unit is intentionally preserved.  Fallen books
    # are generated only next to the new open bookshelf, where their origin is
    # visually unambiguous.


def _add_open_bookshelf(app, root, side, x0, y0, x1, y1, key):
    """Create an unmistakable open library with visible horizontal shelves.

    The frame has no drawer fronts.  Five long shelves remain clearly visible
    between dense rows of books.  Only the narrow coloured spines face the
    room; covers and page blocks extend backward toward the wall.  Every book
    rests exactly on the supporting shelf below it.
    """
    cx, cy, run_x, run_y, horizontal = _wall_face_pose(side, x0, y0, x1, y1, 0.30)
    run = run_x if horizontal else run_y
    width = min(2.45, max(1.35, run * 0.68))
    depth = 0.30
    sx, sy = (width, depth) if horizontal else (depth, width)
    room_sign = -1 if side in ("north", "west") else 1

    shelf_root = root.attachNewNode("open-bookshelf")
    # A thin back panel is recessed against the wall, leaving the shelves and
    # book rows readable from the room rather than resembling closed drawers.
    if horizontal:
        back_y = cy - room_sign * (depth / 2 - 0.018)
        _make_box(app, shelf_root, cx, back_y, 1.10, width, 0.035, 2.08,
                  (0.16, 0.085, 0.040, 1))
    else:
        back_x = cx - room_sign * (depth / 2 - 0.018)
        _make_box(app, shelf_root, back_x, cy, 1.10, 0.035, width, 2.08,
                  (0.16, 0.085, 0.040, 1))

    # Uprights, top and base create an open frame.  The six support levels
    # define five independent rows of books.
    support_levels = (0.15, 0.53, 0.91, 1.29, 1.67, 2.05)
    board = (0.095, 0.050, 0.024, 1)
    shelf_thickness = 0.040
    for support_z in support_levels:
        _make_box(app, shelf_root, cx, cy, support_z, sx, sy,
                  shelf_thickness, board)
    if horizontal:
        for edge_x in (cx - width / 2 + 0.030, cx + width / 2 - 0.030):
            _make_box(app, shelf_root, edge_x, cy, 1.10, 0.060, depth, 2.10, board)
    else:
        for edge_y in (cy - width / 2 + 0.030, cy + width / 2 - 0.030):
            _make_box(app, shelf_root, cx, edge_y, 1.10, depth, 0.060, 2.10, board)

    palette = [
        (0.10, 0.24, 0.48, 1), (0.58, 0.12, 0.11, 1),
        (0.10, 0.42, 0.20, 1), (0.72, 0.47, 0.08, 1),
        (0.38, 0.17, 0.55, 1), (0.05, 0.38, 0.45, 1),
        (0.69, 0.26, 0.06, 1), (0.30, 0.31, 0.34, 1),
        (0.68, 0.65, 0.55, 1), (0.20, 0.48, 0.38, 1),
    ]
    usable_width = width - 0.16

    for row_index, support_z in enumerate(support_levels[:-1]):
        cursor = -usable_width / 2
        book_index = 0
        # Dense packing creates the requested continuous band of adjacent
        # coloured spines, with only occasional millimetric gaps.
        while cursor < usable_width / 2 - 0.020:
            selector = (key + row_index * 1009 + book_index * 53) & 0xffffffff
            spine_w = 0.026 + 0.009 * (selector % 5)
            remaining = usable_width / 2 - cursor
            spine_w = min(spine_w, remaining)
            if spine_w < 0.018:
                break
            book_h = 0.255 + 0.014 * ((selector >> 4) % 7)
            book_depth = 0.205 + 0.008 * ((selector >> 9) % 5)
            gap = 0.002 + 0.001 * ((selector >> 13) % 3)
            along = cursor + spine_w / 2
            z = support_z + shelf_thickness / 2 + book_h / 2
            color = palette[(selector >> 6) % len(palette)]

            if horizontal:
                bx = cx + along
                by = cy + room_sign * (depth / 2 - book_depth / 2 - 0.010)
                _make_box(app, shelf_root, bx, by, z,
                          spine_w, book_depth, book_h, color)
                # Small title label on the room-facing spine only.
                if selector % 5 == 0:
                    _make_box(app, shelf_root, bx,
                              cy + room_sign * (depth / 2 + 0.003),
                              z + book_h * 0.18,
                              max(0.012, spine_w * 0.72), 0.006, 0.014,
                              (0.86, 0.82, 0.70, 1))
            else:
                bx = cx + room_sign * (depth / 2 - book_depth / 2 - 0.010)
                by = cy + along
                _make_box(app, shelf_root, bx, by, z,
                          book_depth, spine_w, book_h, color)
                if selector % 5 == 0:
                    _make_box(app, shelf_root,
                              cx + room_sign * (depth / 2 + 0.003), by,
                              z + book_h * 0.18,
                              0.006, max(0.012, spine_w * 0.72), 0.014,
                              (0.86, 0.82, 0.70, 1))
            cursor += spine_w + gap
            book_index += 1

    # Fallen volumes occur only next to open libraries, linking the disorder on
    # the floor to a plausible source while remaining non-colliding decoration.
    _add_scattered_books(app, root, side, cx, cy, width, key)


def _add_scattered_books(app, root, side, cx, cy, width, key):
    """Place thin, non-colliding books on the carpet near a bookcase."""
    palette = [(0.18,0.31,0.52,1), (0.57,0.19,0.15,1), (0.20,0.48,0.27,1),
               (0.62,0.48,0.16,1), (0.40,0.24,0.53,1)]
    horizontal = side in ("north", "south")
    room_sign = -1 if side in ("north", "west") else 1
    count = 3 + (key % 4)
    for i in range(count):
        along = (((key >> (i % 16)) + i * 37) % 1000) / 999.0 - 0.5
        along *= min(width * 0.82, 1.65)
        away = 0.27 + 0.13 * (((key + i * 97) % 17) / 16.0)
        if horizontal:
            bx, by = cx + along, cy + room_sign * away
        else:
            bx, by = cx + room_sign * away, cy + along
        book = _make_box(app, root, bx, by, 0.022, 0.22, 0.15, 0.035,
                         palette[(key + i) % len(palette)])
        book.setH((key + i * 53) % 180)
        book.setR(((key >> 5) + i * 7) % 9 - 4)


def _add_framed_art(app, root, side, x0, y0, x1, y1, key):
    width = (0.58, 0.88, 1.18)[key % 3]
    height = (0.42, 0.60, 0.74)[(key // 3) % 3]
    palette = [(0.16,0.43,0.60,1), (0.67,0.29,0.18,1), (0.27,0.55,0.32,1),
               (0.65,0.51,0.16,1), (0.44,0.29,0.61,1)]
    art = palette[key % len(palette)]
    cx, cy, _, _, horizontal = _wall_face_pose(side, x0, y0, x1, y1, 0.045)
    if horizontal:
        _make_box(app, root, cx, cy, 1.58, width + 0.09, 0.045, height + 0.09,
                  (0.10,0.065,0.035,1))
        room_sign = -1 if side == "north" else 1
        _make_box(app, root, cx, cy + room_sign * 0.030, 1.58,
                  width, 0.018, height, art)
        # Two smaller colour fields make the picture look textured rather than
        # like a single flat rectangle.
        _make_box(app, root, cx - width*0.18, cy + room_sign*0.042, 1.66,
                  width*0.28, 0.009, height*0.34, (0.86,0.77,0.46,1))
    else:
        _make_box(app, root, cx, cy, 1.58, 0.045, width + 0.09, height + 0.09,
                  (0.10,0.065,0.035,1))
        room_sign = -1 if side == "west" else 1
        _make_box(app, root, cx + room_sign * 0.030, cy, 1.58,
                  0.018, width, height, art)
        _make_box(app, root, cx + room_sign*0.042, cy - width*0.18, 1.66,
                  0.009, width*0.28, height*0.34, (0.86,0.77,0.46,1))



def _add_noticeboard(app, root, side, x0, y0, x1, y1, key):
    """Add a cork noticeboard with several pinned sheets."""
    width, height = 1.18, 0.68
    cx, cy, _, _, horizontal = _wall_face_pose(side, x0, y0, x1, y1, 0.040)
    cork = (0.55, 0.34, 0.16, 1)
    paper = [(0.92,0.91,0.84,1), (0.78,0.88,0.94,1), (0.95,0.84,0.72,1)]
    if horizontal:
        room_sign = -1 if side == "north" else 1
        _make_box(app, root, cx, cy, 1.48, width+0.08, 0.04, height+0.08, (0.16,0.10,0.05,1))
        _make_box(app, root, cx, cy+room_sign*0.027, 1.48, width, 0.014, height, cork)
        for i, dx in enumerate((-0.34, 0.0, 0.31)):
            _make_box(app, root, cx+dx, cy+room_sign*0.040, 1.48 + (i-1)*0.07,
                      0.24, 0.008, 0.31, paper[(key+i)%3])
    else:
        room_sign = -1 if side == "west" else 1
        _make_box(app, root, cx, cy, 1.48, 0.04, width+0.08, height+0.08, (0.16,0.10,0.05,1))
        _make_box(app, root, cx+room_sign*0.027, cy, 1.48, 0.014, width, height, cork)
        for i, dy in enumerate((-0.34, 0.0, 0.31)):
            _make_box(app, root, cx+room_sign*0.040, cy+dy, 1.48 + (i-1)*0.07,
                      0.008, 0.24, 0.31, paper[(key+i)%3])

def _build_table(app, root, obj):
    """Build a desk; a deterministic minority is overturned for SAR realism."""
    x0, y0, x1, y1 = obj["region"]
    cx = (x0 + x1 + 1) * CELL_SIZE / 2
    cy = (y0 + y1 + 1) * CELL_SIZE / 2
    sx = (x1 - x0 + 1) * CELL_SIZE * 0.92
    sy = (y1 - y0 + 1) * CELL_SIZE * 0.88
    key = (x0 * 92821 + y0 * 68917 + x1 * 31337 + y1 * 101) & 0xffffffff
    overturned = (key % 9 == 0)

    table = root.attachNewNode("table-overturned" if overturned else "table")
    table.setPos(cx, cy, 0.0)
    if overturned:
        # Rotate the complete rigid object rather than deforming its footprint.
        # A small lift prevents the tabletop from clipping through the carpet.
        table.setZ(0.07)
        table.setR(88 if (key // 9) % 2 == 0 else -88)
        table.setH((key // 17) % 4 * 90)

    _make_box(app, table, 0, 0, TABLE_HEIGHT_M - 0.045, sx, sy, 0.09, (0.48,0.27,0.12,1))
    _make_box(app, table, 0, 0, TABLE_HEIGHT_M + 0.012, sx*0.96, sy*0.94, 0.028, (0.64,0.39,0.18,1))
    leg = 0.075
    for dx in (-sx/2 + 0.12, sx/2 - 0.12):
        for dy in (-sy/2 + 0.12, sy/2 - 0.12):
            _make_box(app, table, dx, dy, (TABLE_HEIGHT_M-0.09)/2,
                      leg, leg, TABLE_HEIGHT_M-0.09, (0.12,0.12,0.13,1))

    # Every desk carries at least one recognisable office prop.  Props are
    # children of the desk node: if the desk is overturned by the SAR visual
    # dressing, its monitor, folders and pen cup rotate with it instead of
    # floating in their original world position.
    _add_tabletop_props(app, table, sx, sy, key)


def _add_tabletop_props(app, table, sx, sy, key):
    """Place desk accessories in distinct, non-overlapping work zones.

    The desk is normalised so that its local X axis is always the longest
    dimension.  Monitor, files and pen holder are then assigned separate
    regions: rear-centre, front-left and front-right.  This prevents the visual
    clutter produced by stacking several props at nearly the same coordinates,
    while preserving deterministic combinations and the correct orientation on
    both horizontal and vertical tables.
    """
    top_z = TABLE_HEIGHT_M + 0.055
    mode = key % 7

    # Normalised accessory frame: local X follows the table's long dimension.
    # Rotating this single node keeps every prop correctly arranged on tables
    # whose long dimension lies along simulator Y.
    props = table.attachNewNode("tabletop-props")
    long_size = max(sx, sy)
    short_size = min(sx, sy)
    if sy > sx:
        props.setH(90)

    # Keep all accessory centres well inside the tabletop.  The values are
    # deliberately conservative so even the smallest generated desk has clear
    # gaps between monitor, files and pen cup.
    rear_v = min(short_size * 0.25, 0.22)
    front_v = -min(short_size * 0.25, 0.22)
    side_u = min(long_size * 0.30, max(0.26, long_size / 2 - 0.28))

    # An off computer display, centred along the rear edge.  Its screen plane
    # spans the table's long axis and therefore faces one of the long sides.
    if mode in (0, 1, 3, 5, 6):
        monitor = props.attachNewNode("monitor-off")
        monitor.setPos(0.0, rear_v, 0.0)
        _make_box(app, monitor, 0, 0, top_z + 0.34, 0.58, 0.075, 0.36,
                  (0.055, 0.060, 0.065, 1))
        _make_box(app, monitor, 0, -0.041, top_z + 0.34, 0.51, 0.012, 0.29,
                  (0.012, 0.016, 0.020, 1))
        _make_box(app, monitor, 0, 0, top_z + 0.13, 0.055, 0.055, 0.20,
                  (0.13, 0.14, 0.15, 1))
        _make_box(app, monitor, 0, 0, top_z + 0.025, 0.31, 0.19, 0.035,
                  (0.10, 0.11, 0.12, 1))

    # Pen holder occupies the front-right zone, separated from both monitor and
    # files. Cylinders are approximated by narrow boxes for renderer efficiency.
    if mode in (1, 2, 4, 5, 6):
        px, py = side_u, front_v
        _make_box(app, props, px, py, top_z + 0.07, 0.13, 0.13, 0.14,
                  (0.16, 0.18, 0.20, 1))
        pen_colors = ((0.10,0.22,0.65,1), (0.70,0.12,0.10,1),
                      (0.08,0.45,0.20,1), (0.12,0.12,0.13,1))
        for i, dx in enumerate((-0.040, -0.012, 0.018, 0.043)):
            pen = _make_box(app, props, px + dx, py + (i%2)*0.025 - 0.012,
                            top_z + 0.19 + 0.018*(i%2), 0.018, 0.018, 0.25,
                            pen_colors[(key+i) % len(pen_colors)])
            pen.setP(-5 + i * 3)

    # Binders and loose documents occupy the front-left zone.  Their maximum
    # footprint is kept below the gap to the central monitor and opposite cup.
    if mode in (0, 2, 3, 4, 6):
        fx, fy = -side_u, front_v
        file_colors = ((0.20,0.38,0.62,1), (0.66,0.18,0.14,1),
                       (0.24,0.50,0.28,1), (0.67,0.51,0.15,1))
        count = 2 + (key % 3)
        for i in range(count):
            _make_box(app, props, fx + i*0.045, fy, top_z + 0.045 + i*0.016,
                      0.24, 0.18, 0.055, file_colors[(key+i)%len(file_colors)])
        _make_box(app, props, fx + 0.03, fy - 0.010, top_z + 0.075 + count*0.016,
                  0.22, 0.16, 0.018, (0.91,0.89,0.80,1))


def _build_chair(app, root, obj):
    """Build an office chair, occasionally lying on its side."""
    x0, y0, x1, y1 = obj["region"]
    cx = (x0 + x1 + 1) * CELL_SIZE / 2
    cy = (y0 + y1 + 1) * CELL_SIZE / 2
    sx = (x1 - x0 + 1) * CELL_SIZE * 0.62
    sy = (y1 - y0 + 1) * CELL_SIZE * 0.62
    rotation = int(obj.get("rotation", 0)) % 4
    key = (x0 * 31337 + y0 * 7919 + x1 * 1013 + y1 * 37) & 0xffffffff
    overturned = (key % 7 == 0)

    chair = root.attachNewNode("chair-overturned" if overturned else "chair")
    chair.setPos(cx, cy, 0.0)
    chair.setH(rotation * 90)
    if overturned:
        chair.setZ(0.04)
        chair.setR(88 if (key // 7) % 2 == 0 else -88)

    _make_box(app, chair, 0, 0, CHAIR_SEAT_HEIGHT_M, sx, sy, 0.10, (0.10,0.31,0.42,1))
    _make_box(app, chair, 0, sy/2 - 0.035, (CHAIR_SEAT_HEIGHT_M+CHAIR_BACK_HEIGHT_M)/2,
              sx, 0.07, CHAIR_BACK_HEIGHT_M-CHAIR_SEAT_HEIGHT_M, (0.08,0.25,0.34,1))
    for dx in (-sx/2+0.05, sx/2-0.05):
        for dy in (-sy/2+0.05, sy/2-0.05):
            _make_box(app, chair, dx, dy, CHAIR_SEAT_HEIGHT_M/2, 0.045, 0.045,
                      CHAIR_SEAT_HEIGHT_M, (0.08,0.08,0.09,1))


def _build_debris(app, root, obj, floor_index):
    x0, y0, x1, y1 = obj["region"]
    width_cells, height_cells = x1-x0+1, y1-y0+1
    count = max(5, width_cells * height_cells * 2)
    for i in range(count):
        seed = (floor_index+1)*1000003 + x0*9176 + y0*131 + i*7919
        ux = ((seed * 1103515245 + 12345) & 0xffff) / 65535.0
        uy = ((seed * 214013 + 2531011) & 0xffff) / 65535.0
        size = 0.07 + 0.11 * (((seed >> 8) & 255) / 255.0)
        px = (x0 + 0.12 + ux * max(0.1, width_cells-0.24)) * CELL_SIZE
        py = (y0 + 0.12 + uy * max(0.1, height_cells-0.24)) * CELL_SIZE
        pz = size * (0.55 + 0.35*((seed >> 16)&3))
        rock = _make_box(app, root, px, py, pz/2, size*1.35, size, pz,
                         (0.32+0.04*(i%3), 0.28+0.03*(i%2), 0.23, 1))
        rock.setH((seed % 360))
        rock.setP(((seed >> 6) % 20) - 10)


def _add_stair_arrow(app, root, centre_x, centre_y, region_w, region_h,
                     direction, top_z, color, name):
    """Draw a clear floor-plan arrow along the stair run.

    The arrow points in the semantic travel direction supplied by the building:
    toward the highest tread for an upward flight and toward the lowest tread
    for a downward flight.  It is composed of a shaft and two diagonal head
    strokes so it remains legible from the robot camera without textures.
    """
    dx, dy = direction
    length = max(region_w, region_h) * 0.58
    shaft_len = length * 0.62
    width = min(region_w, region_h) * 0.10
    arrow = root.attachNewNode(name)
    arrow.setPos(centre_x, centre_y, top_z)
    # Local +X is the arrow direction. Panda heading rotates +X in the XY plane.
    arrow.setH(math.degrees(math.atan2(dy, dx)))
    _make_box(app, arrow, -length*0.10, 0, 0, shaft_len, width, 0.018, color)
    head_x = length * 0.31
    head_len = length * 0.28
    for sign in (-1, 1):
        stroke = _make_box(app, arrow, head_x - head_len*0.30,
                           sign * head_len*0.18, 0,
                           head_len, width, 0.018, color)
        stroke.setH(-sign * 38)


def _build_stairs(app, root, floor):
    """Render raised, colour-coded stair symbols with mirrored tread order.

    Both stair types are intentionally overlaid on the floor, avoiding an
    artificial floor opening.  Upward flights use orange treads whose height
    increases in the arrow direction.  Downward flights use the same raised
    geometry in blue, but the order is mirrored, so their height decreases in
    the arrow direction.  A contrasting arrow explicitly points toward the
    highest tread for ascent and toward the lowest tread for descent.

    This function is purely visual; stair selection and inter-floor transition
    logic remain unchanged in the simulator.
    """
    up_color = (0.95, 0.39, 0.08, 1)
    down_color = (0.12, 0.48, 0.78, 1)
    up_arrow = (1.00, 0.88, 0.62, 1)
    down_arrow = (0.78, 0.93, 1.00, 1)

    for kind, regions, color, arrow_color in (
            ("up", floor["stair_up"], up_color, up_arrow),
            ("down", floor["stair_down"], down_color, down_arrow)):
        for region_index, region in enumerate(regions):
            x0, y0, x1, y1 = region
            direction = (floor["stair_dirs"][min(region_index, len(floor["stair_dirs"])-1)]
                         if floor["stair_dirs"] else (1, 0))
            horizontal = abs(direction[0]) > abs(direction[1])
            positive = (direction[0] > 0) if horizontal else (direction[1] > 0)
            steps = 7
            region_w = (x1-x0+1)*CELL_SIZE
            region_h = (y1-y0+1)*CELL_SIZE
            centre_x = (x0+x1+1)*CELL_SIZE/2
            centre_y = (y0+y1+1)*CELL_SIZE/2
            max_top = 0.045 + (steps - 1) * 0.042

            for run_index in range(steps):
                # run_index grows in the semantic arrow direction regardless of
                # whether the corridor itself points along a positive or negative
                # world axis.
                ordered_i = run_index if positive else (steps - 1 - run_index)
                t = (ordered_i + 0.5) / steps
                # Up rises along the arrow; down is its exact mirror and falls
                # along the arrow while still being drawn above the floor.
                level_index = run_index if kind == "up" else (steps - 1 - run_index)
                height = 0.045 + level_index * 0.042
                z = height / 2
                if horizontal:
                    sx = region_w/steps * 0.92
                    sy = region_h * 0.82
                    cx = x0*CELL_SIZE + t*region_w
                    cy = centre_y
                else:
                    sx = region_w * 0.82
                    sy = region_h/steps * 0.92
                    cx = centre_x
                    cy = y0*CELL_SIZE + t*region_h
                _make_box(app, root, cx, cy, z, sx, sy, height, color)

            # Both arrows follow the corridor direction. Because the blue tread
            # order is mirrored, this points to the lowest blue tread and to the
            # highest orange tread exactly as requested.
            _add_stair_arrow(app, root, centre_x, centre_y, region_w, region_h,
                             direction, max_top + 0.028, arrow_color,
                             f"stair-{kind}-arrow-{region_index}")


def _build_person(app, root, victim, floor_index, victim_index):
    """Render victims in standing, seated, and recumbent SAR-relevant poses."""
    gx, gy = victim
    cx, cy = (gx + 0.5) * CELL_SIZE, (gy + 0.5) * CELL_SIZE
    person = root.attachNewNode(f"victim-{victim_index}")
    person.setPos(cx, cy, 0)
    key = (floor_index * 73 + victim_index * 131)
    person.setH(key % 360)
    pose = (floor_index + victim_index) % 3

    skin = (0.72,0.52,0.38,1)
    shirt = ((0.58,0.12,0.12,1), (0.16,0.35,0.58,1), (0.31,0.48,0.22,1))[key % 3]
    trousers = (0.13,0.18,0.25,1)

    if pose == 0:  # standing
        _make_box(app, person, 0, 0, 1.13, 0.38, 0.22, 0.64, shirt)
        _make_box(app, person, 0, 0, 1.58, 0.25, 0.22, 0.25, skin)
        for dx in (-0.105, 0.105):
            _make_box(app, person, dx, 0, 0.42, 0.13, 0.16, 0.82, trousers)
        for dx in (-0.25, 0.25):
            arm = _make_box(app, person, dx, 0, 1.13, 0.11, 0.12, 0.62, skin)
            arm.setR(8 if dx > 0 else -8)

    elif pose == 1:  # seated on the floor / leaning back
        _make_box(app, person, 0, 0, 0.70, 0.40, 0.25, 0.62, shirt)
        _make_box(app, person, 0, -0.02, 1.09, 0.25, 0.22, 0.25, skin)
        for dx in (-0.13, 0.13):
            thigh = _make_box(app, person, dx, -0.22, 0.34, 0.14, 0.48, 0.16, trousers)
            thigh.setP(-12)
            _make_box(app, person, dx, -0.46, 0.18, 0.13, 0.38, 0.14, trousers)
        for dx in (-0.27, 0.27):
            arm = _make_box(app, person, dx, 0.02, 0.69, 0.11, 0.12, 0.54, skin)
            arm.setR(22 if dx > 0 else -22)

    else:  # lying down
        body = person.attachNewNode("recumbent-body")
        body.setPos(0, 0, 0.13)
        body.setP(90)
        _make_box(app, body, 0, 0, 0.52, 0.38, 0.22, 0.70, shirt)
        _make_box(app, body, 0, 0, 0.98, 0.25, 0.22, 0.25, skin)
        for dx in (-0.105, 0.105):
            _make_box(app, body, dx, 0, 0.04, 0.13, 0.16, 0.76, trousers)
        for dx in (-0.26, 0.26):
            arm = _make_box(app, body, dx, 0, 0.52, 0.11, 0.12, 0.58, skin)
            arm.setR(16 if dx > 0 else -16)


def _is_perimeter_wall(grid, x0, y0, x1, y1):
    """True when a merged structural rectangle touches the outer wall band."""
    h, w = grid.shape
    return x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1


def _window_specs_for_wall(floor, x0, y0, x1, y1):
    """Return flush window placements only where the interior wall is clear.

    A candidate is accepted only when a two-cell-deep strip on the indoor side
    contains neither structural walls nor simulated obstacles, stairs, doors,
    or victims.  This is stricter than a visual overlap check and guarantees
    that a generated window belongs to a genuinely unobstructed wall section.
    """
    grid = floor["grid"]
    object_kind = floor["object_kind"]
    if not _is_perimeter_wall(grid, x0, y0, x1, y1):
        return []
    length_cells = max(x1 - x0 + 1, y1 - y0 + 1)
    if length_cells < 8:
        return []
    sides = _wall_free_sides(grid, x0, y0, x1, y1)
    if not sides:
        return []

    # A perimeter wall has one indoor face.  Select that face explicitly.
    side = sides[0]
    horizontal = side in ("north", "south")
    run_cells = (x1 - x0 + 1) if horizontal else (y1 - y0 + 1)
    key = (floor["floor_index"] * 104729 + x0 * 13007 + y0 * 7919 + x1 * 97 + y1) & 0xffffffff
    if key % 4 == 0:
        return []

    victim_cells = {tuple(v) for v in floor.get("victims", [])}
    specs = []
    # A window occupies three cells along the wall. Candidate centres retain a
    # two-cell buffer from corners and from one another.
    candidate_offsets = list(range(2, run_cells - 2, 5))
    for local_centre in candidate_offsets[:3]:
        half = 1
        if horizontal:
            ax0, ax1 = x0 + local_centre - half, x0 + local_centre + half
            if side == "north":
                strip = [(xx, yy) for yy in range(max(0, y0 - 2), y0) for xx in range(ax0, ax1 + 1)]
            else:
                strip = [(xx, yy) for yy in range(y1 + 1, min(grid.shape[0], y1 + 3)) for xx in range(ax0, ax1 + 1)]
            centre = ((ax0 + ax1 + 1) * CELL_SIZE / 2, (y0 if side == "north" else y1 + 1) * CELL_SIZE)
        else:
            ay0, ay1 = y0 + local_centre - half, y0 + local_centre + half
            if side == "west":
                strip = [(xx, yy) for xx in range(max(0, x0 - 2), x0) for yy in range(ay0, ay1 + 1)]
            else:
                strip = [(xx, yy) for xx in range(x1 + 1, min(grid.shape[1], x1 + 3)) for yy in range(ay0, ay1 + 1)]
            centre = ((x0 if side == "west" else x1 + 1) * CELL_SIZE, (ay0 + ay1 + 1) * CELL_SIZE / 2)

        if not strip:
            continue
        clear = True
        for xx, yy in strip:
            if not (0 <= xx < grid.shape[1] and 0 <= yy < grid.shape[0]):
                clear = False; break
            if grid[yy, xx] in (WALL, DOOR, STAIR_UP, STAIR_DOWN):
                clear = False; break
            if object_kind[yy, xx] != "" or (xx, yy) in victim_cells:
                clear = False; break
        if clear:
            specs.append({
                "wall": (x0, y0, x1, y1), "side": side,
                "centre": centre, "width": 1.32, "height": 1.02,
                "strip_cells": tuple(strip), "key": key + local_centre,
            })
    return specs


def _add_perimeter_windows(app, root, specs):
    """Build flush, recognisable office windows with mullion and handle."""
    frame = (0.12, 0.13, 0.14, 1)
    glass = (0.20, 0.62, 0.91, 1)
    sky = (0.34, 0.72, 0.96, 1)
    handle_color = (0.62, 0.64, 0.66, 1)
    for spec in specs:
        side = spec["side"]
        wx, wy = spec["centre"]
        width, height = spec["width"], spec["height"]
        horizontal = side in ("north", "south")
        room_sign = -1 if side in ("north", "west") else 1
        z = 1.55

        # The glass lies directly on the wall plane; only the frame projects
        # 8 mm into the room. This removes the previous detached-panel effect.
        if horizontal:
            glass_y = wy + room_sign * 0.004
            frame_y = wy + room_sign * 0.008
            _make_box(app, root, wx, glass_y, z, width, 0.008, height, sky)
            # Four frame bars plus a central vertical mullion.
            for dx in (-width/2, width/2):
                _make_box(app, root, wx+dx, frame_y, z, 0.055, 0.022, height+0.08, frame)
            for dz in (-height/2, height/2):
                _make_box(app, root, wx, frame_y, z+dz, width+0.08, 0.022, 0.055, frame)
            _make_box(app, root, wx, frame_y, z, 0.045, 0.024, height, frame)
            # Subtle sky reflection and a handle on the right sash.
            _make_box(app, root, wx-width*0.20, frame_y+room_sign*0.006, z+0.18,
                      width*0.28, 0.006, 0.055, (0.78,0.91,0.99,1))
            _make_box(app, root, wx+width*0.18, frame_y+room_sign*0.020, z,
                      0.035, 0.035, 0.22, handle_color)
            _make_box(app, root, wx+width*0.23, frame_y+room_sign*0.028, z,
                      0.12, 0.028, 0.035, handle_color)
        else:
            glass_x = wx + room_sign * 0.004
            frame_x = wx + room_sign * 0.008
            _make_box(app, root, glass_x, wy, z, 0.008, width, height, sky)
            for dy in (-width/2, width/2):
                _make_box(app, root, frame_x, wy+dy, z, 0.022, 0.055, height+0.08, frame)
            for dz in (-height/2, height/2):
                _make_box(app, root, frame_x, wy, z+dz, 0.022, width+0.08, 0.055, frame)
            _make_box(app, root, frame_x, wy, z, 0.024, 0.045, height, frame)
            _make_box(app, root, frame_x+room_sign*0.006, wy-width*0.20, z+0.18,
                      0.006, width*0.28, 0.055, (0.78,0.91,0.99,1))
            _make_box(app, root, frame_x+room_sign*0.020, wy+width*0.18, z,
                      0.035, 0.035, 0.22, handle_color)
            _make_box(app, root, frame_x+room_sign*0.028, wy+width*0.23, z,
                      0.028, 0.12, 0.035, handle_color)


def _add_ceiling_lights(app, root, floor):
    """Add deterministic ceiling panels, some dark after the disaster."""
    from panda3d.core import PointLight, Vec4
    grid = floor["grid"]
    h, w = grid.shape
    lit_count = 0
    for gy in range(3, h - 3, 7):
        for gx in range(3, w - 3, 8):
            # Fixtures only above non-wall indoor areas and not above stairs.
            if grid[gy, gx] in (WALL, STAIR_UP, STAIR_DOWN):
                continue
            key = floor["floor_index"] * 1009 + gx * 97 + gy * 193
            switched_off = (key % 5 == 0)
            cx, cy = (gx + 0.5) * CELL_SIZE, (gy + 0.5) * CELL_SIZE
            casing = _make_box(app, root, cx, cy, WALL_HEIGHT_M - 0.075,
                               1.05, 0.34, 0.055, (0.66,0.67,0.68,1))
            panel_color = (0.15,0.16,0.17,1) if switched_off else (0.96,0.95,0.82,1)
            panel = _make_box(app, root, cx, cy, WALL_HEIGHT_M - 0.115,
                              0.94, 0.25, 0.025, panel_color)
            if not switched_off:
                # Emission keeps the luminaire visibly bright even in shadow.
                panel.setColorScale(1.12, 1.10, 0.92, 1.0)
                # Limit real point lights to preserve the GPU performance gain.
                if lit_count < 6:
                    light = PointLight(f"ceiling-light-{floor['floor_index']}-{gx}-{gy}")
                    light.setColor(Vec4(0.28, 0.27, 0.22, 1.0))
                    light.setAttenuation((1.0, 0.10, 0.035))
                    light_np = root.attachNewNode(light)
                    light_np.setPos(cx, cy, WALL_HEIGHT_M - 0.22)
                    root.setLight(light_np)
                    lit_count += 1


def _add_wall_plants_and_accessories(app, root, floor):
    """Place non-colliding plants and small office props next to walls."""
    grid = floor["grid"]
    h, w = grid.shape
    reserved_window_cells = {
        cell for spec in floor.get("window_specs", []) for cell in spec.get("strip_cells", ())
    }
    candidates = []
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            if (x, y) in reserved_window_cells:
                continue
            if grid[y, x] == WALL:
                continue
            near_wall = (grid[y, x-1] == WALL or grid[y, x+1] == WALL or
                         grid[y-1, x] == WALL or grid[y+1, x] == WALL)
            if near_wall and ((x * 37 + y * 61 + floor["floor_index"] * 17) % 29 == 0):
                candidates.append((x, y))
    for index, (x, y) in enumerate(candidates[:8]):
        cx, cy = (x + 0.5) * CELL_SIZE, (y + 0.5) * CELL_SIZE
        if index % 3 != 2:
            _build_office_plant(app, root, cx, cy, index)
        else:
            # A narrow waste-paper bin or archive box adds variety while
            # remaining decorative and absent from collision geometry.
            _make_box(app, root, cx, cy, 0.18, 0.26, 0.26, 0.36, (0.18,0.20,0.21,1))
            _make_box(app, root, cx, cy, 0.37, 0.24, 0.24, 0.02, (0.08,0.09,0.10,1))



def _paper_seed_for_floor(floor):
    """Return a deterministic visual seed derived from the floor geometry."""
    grid = np.asarray(floor["grid"], dtype=np.int64)
    yy, xx = np.indices(grid.shape, dtype=np.int64)
    checksum = int(np.sum((grid + 1) * (xx + 11) * 131 + (grid + 3) * (yy + 7) * 197))
    return (checksum ^ ((int(floor["floor_index"]) + 1) * 0x9E3779B1)) & 0xffffffff


def _add_scattered_floor_papers(app, root, floor):
    """Scatter many overlapping A4-like sheets over free indoor floor cells.

    The floor is partitioned into zones and each zone contributes several
    clusters, ensuring papers appear throughout rooms and corridors rather than
    gathering in one corner.  Each cluster contains one to three sheets; upper
    sheets receive a slightly higher Z coordinate to avoid z-fighting and make
    overlaps visible.  These nodes are never written back to ``floor.grid``.
    """
    grid = np.asarray(floor["grid"])
    height, width = grid.shape
    seed = _paper_seed_for_floor(floor)
    rng = np.random.default_rng(seed)
    paper_palette = (
        (0.96, 0.95, 0.89, 1),
        (0.90, 0.93, 0.96, 1),
        (0.95, 0.91, 0.83, 1),
        (0.88, 0.92, 0.84, 1),
    )

    # Six by four zones provide broad coverage on the standard 44 x 30 floor.
    zone_cols, zone_rows = 6, 4
    cluster_index = 0
    for zone_y in range(zone_rows):
        y0 = zone_y * height // zone_rows
        y1 = (zone_y + 1) * height // zone_rows
        for zone_x in range(zone_cols):
            x0 = zone_x * width // zone_cols
            x1 = (zone_x + 1) * width // zone_cols
            local = np.argwhere(grid[y0:y1, x0:x1] == FREE)
            if len(local) == 0:
                continue

            # Two to four clusters per populated zone produce roughly 70--120
            # sheets per floor, depending on overlap multiplicity.
            cluster_count = 2 + int(rng.integers(0, 3))
            picks = rng.integers(0, len(local), size=cluster_count)
            for pick in picks:
                local_y, local_x = map(int, local[int(pick)])
                gx, gy = x0 + local_x, y0 + local_y
                base_x = (gx + 0.20 + 0.60 * float(rng.random())) * CELL_SIZE
                base_y = (gy + 0.20 + 0.60 * float(rng.random())) * CELL_SIZE
                base_heading = float(rng.uniform(0.0, 180.0))
                sheet_count = 1 + int(rng.integers(0, 3))

                for layer in range(sheet_count):
                    # Higher layers are offset by only a few centimetres, so
                    # they visibly overlap the sheet below instead of forming
                    # separate piles.
                    offset_x = float(rng.uniform(-0.045, 0.045)) if layer else 0.0
                    offset_y = float(rng.uniform(-0.045, 0.045)) if layer else 0.0
                    sheet = root.attachNewNode("scattered-paper")
                    sheet.setPos(base_x + offset_x, base_y + offset_y,
                                 0.006 + layer * 0.004)
                    sheet.setH(base_heading + float(rng.uniform(-13.0, 13.0)))
                    paper_color = paper_palette[(cluster_index + layer) % len(paper_palette)]
                    _make_box(app, sheet, 0, 0, 0,
                              0.210, 0.297, 0.004, paper_color)

                    # Sparse printed lines make individual sheets legible while
                    # limiting the number of additional GPU nodes.
                    if (cluster_index + layer) % 5 == 0:
                        ink = (0.35, 0.39, 0.43, 1)
                        for line_index in range(3):
                            _make_box(app, sheet, -0.018,
                                      0.060 - line_index * 0.038, 0.003,
                                      0.135 - line_index * 0.012,
                                      0.006, 0.002, ink)
                cluster_index += 1


def _build_office_plant(app, root, cx, cy, key):
    """Low-poly potted plant; purely visual and attached near a wall."""
    _make_box(app, root, cx, cy, 0.16, 0.30, 0.30, 0.32, (0.43,0.20,0.08,1))
    _make_box(app, root, cx, cy, 0.42, 0.055, 0.055, 0.48, (0.22,0.34,0.12,1))
    greens = [(0.16,0.45,0.18,1), (0.24,0.57,0.24,1), (0.12,0.37,0.16,1)]
    for i in range(7):
        angle = 2 * math.pi * i / 7 + key * 0.21
        leaf = _make_box(app, root, cx + 0.15*math.cos(angle), cy + 0.15*math.sin(angle),
                         0.58 + 0.07*(i%3), 0.10, 0.28, 0.055, greens[i%len(greens)])
        leaf.setH(math.degrees(angle))
        leaf.setP(-18 + 9*(i%3))


# Backward-compatible lightweight helper used by tests and external scripts.
# It no longer performs software rendering; it validates that scene data can be
# projected to a deterministic diagnostic image without importing Panda3D.
def _render_frame(floor, x, y, theta, width=640, height=400):
    """Return a small top-down diagnostic frame for headless regression tests."""
    grid = np.asarray(floor["grid"])
    palette = np.array([
        [224,224,224], [55,55,55], [120,185,105], [245,145,45], [80,160,225]
    ], dtype=np.uint8)
    safe = np.clip(grid, 0, len(palette)-1)
    image = palette[safe]
    # Nearest-neighbour scaling implemented with index arrays avoids optional
    # image packages in test environments.
    yy = (np.arange(height) * image.shape[0] / height).astype(int)
    xx = (np.arange(width) * image.shape[1] / width).astype(int)
    frame = image[yy[:,None], xx[None,:]].copy()
    px = int(np.clip(x / (grid.shape[1]*CELL_SIZE) * width, 0, width-1))
    py = int(np.clip(y / (grid.shape[0]*CELL_SIZE) * height, 0, height-1))
    frame[max(0,py-3):min(height,py+4), max(0,px-3):min(width,px+4)] = (245,195,45)
    ex = int(px + 12*math.cos(theta)); ey = int(py + 12*math.sin(theta))
    if 0 <= ex < width and 0 <= ey < height:
        frame[max(0,ey-1):min(height,ey+2), max(0,ex-1):min(width,ex+2)] = (220,40,40)
    return frame


RENDER_W = 640
RENDER_H = 400
OBJECT_HEIGHTS = {"table": TABLE_HEIGHT_M, "chair": CHAIR_BACK_HEIGHT_M, "debris": DEBRIS_MAX_HEIGHT_M}
