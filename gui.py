"""Configuration, smooth Pygame live viewer and final result table.

The configuration and final table remain Matplotlib based.  The live viewer is
Pygame based, as in the first continuous simulator, because redrawing complete
Matplotlib axes at every physical step made a mathematically continuous pose
look jerky.  Physics, sensing, planning and rendering now run at independent
rates.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# CODE-REVIEW NOTES
# Purpose: Tk configuration, Pygame rendering, legend, controls, and result presentation; no simulation work runs while editing fields.
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

import math
import time
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RadioButtons, CheckButtons
import numpy as np

from building import (
    FLOOR_W,
    FLOOR_H,
    CELL_SIZE,
    FREE as GT_FREE,
    WALL,
    DOOR,
    STAIR_UP,
    STAIR_DOWN,
    LAYOUT_OFFICE,
    LAYOUT_FREE,
    OBJECT_CHAIR,
    OBJECT_TABLE,
    OBJECT_DEBRIS,
    OBJECT_DENSITY_MIN,
    OBJECT_DENSITY_MAX,
)
from geometry import cell_center_world
from robot import (
    FREE,
    SEMANTIC_THRESHOLD,
    FRONTIER_SCALE,
    PIXEL_OCCUPANCY_RESOLUTION_M,
)
from simulation import (
    PHYSICS_DT_S,
    RunState,
    ACTIVE_TARGET_HYSTERESIS_WEIGHT_DEFAULT,
    TARGET_SWITCH_MARGIN_DEFAULT,
)
from camera3d import CameraHandle
from i18n import tr, yes_no, localize_analysis_text

from planner import (
    IW_DEFAULT,
    DW_DEFAULT,
    VW_DEFAULT,
    CW_DEFAULT,
    OW_DEFAULT,
)

try:
    import pygame
except ImportError:  # pragma: no cover - handled with a clear runtime message
    pygame = None


WORLD_W_M = FLOOR_W * CELL_SIZE
WORLD_H_M = FLOOR_H * CELL_SIZE

# Live-view timing.  The simulation uses a fixed 100 Hz physical step; the
# display is refreshed independently at 60 FPS.
RENDER_FPS = 60
MAX_REAL_FRAME_DT_S = 0.05
MAX_ACCUMULATED_SIM_S = 0.25
MAX_PHYSICS_STEPS_PER_FRAME = 40
MAP_SURFACE_REFRESH_S = 0.10

# Max Turbo executes fixed simulation steps in tight CPU-bound batches.  The
# batch duration keeps the Pygame event queue responsive while avoiding the
# 60 FPS limiter and all expensive map/legend rendering.
MAX_TURBO_BATCH_WALL_S = 0.050
MAX_TURBO_EVENT_POLL_STEPS = 256
MAX_TURBO_STATUS_REFRESH_S = 0.10

# Palette shared by the two map panels.
BACKGROUND = (235, 238, 243)
PANEL_BG = (250, 250, 250)
TEXT = (28, 32, 40)
MUTED_TEXT = (82, 88, 98)
BORDER = (118, 124, 134)
GT_COLORS = np.array([
    (250, 250, 250),  # free
    (58, 58, 58),     # wall / obstacle
    (143, 191, 111),  # open door
    (255, 157, 59),   # stairs up
    (111, 179, 232),  # stairs down
], dtype=np.uint8)
UNKNOWN_COLOR = np.array((212, 212, 212), dtype=np.float32)
MAP_FREE_COLOR = np.array((250, 250, 250), dtype=np.float32)
MAP_OCC_COLOR = np.array((58, 58, 58), dtype=np.float32)
SEMANTIC_COLOR = np.array((227, 74, 51), dtype=np.float32)

ROBOT_YELLOW = (242, 193, 78)
ROBOT_EDGE = (126, 96, 0)
ROBOT_FOOT = (74, 56, 0)
LASER_COLOR = (48, 167, 201)
FRONTIER_GREEN = (34, 164, 71)
CHOSEN_RED = (220, 45, 48)
UNREACHABLE_GRAY = (180, 180, 180)
PATH_RED = (210, 45, 48)
TRAJECTORY_BLUE = (49, 95, 155)
SEMANTIC_CYAN = (0, 220, 230)
VICTIM_RED = (220, 35, 45)

# Ground-truth object palette.  Object cells are still WALL in the simulation;
# these lighter fills and top-view icons only distinguish their semantic type.
TABLE_CELL_COLOR = np.array((222, 194, 148), dtype=np.uint8)
CHAIR_CELL_COLOR = np.array((145, 190, 207), dtype=np.uint8)
DEBRIS_CELL_COLOR = np.array((176, 151, 122), dtype=np.uint8)
OBJECT_CELL_COLORS = {
    OBJECT_TABLE: TABLE_CELL_COLOR,
    OBJECT_CHAIR: CHAIR_CELL_COLOR,
    OBJECT_DEBRIS: DEBRIS_CELL_COLOR,
}
TABLE_TOP = (177, 125, 72)
TABLE_EDGE = (91, 61, 35)
CHAIR_TOP = (64, 129, 157)
CHAIR_EDGE = (27, 72, 92)
DEBRIS_TOP = (116, 94, 72)
DEBRIS_EDGE = (67, 55, 45)
DEBRIS_STONE_COLORS = (
    (132, 113, 94),
    (104, 91, 78),
    (157, 137, 114),
)

# Canonical top-view pile for one 2x2-cell debris tile.  Every larger debris
# region repeats these same small/large stones instead of stretching one icon.
# Values are (u, v, nominal_radius, angle_radians, colour_index).
_DEBRIS_TILE_PATTERN = (
    (0.25, 0.31, 0.165, 0.20, 0),  # large
    (0.70, 0.66, 0.145, 1.05, 1),  # large
    (0.72, 0.24, 0.085, 2.20, 2),  # small
    (0.30, 0.76, 0.075, 1.65, 1),  # small
    (0.49, 0.49, 0.060, 0.55, 2),  # small
)
DEBRIS_STONE_ASPECT = 1.28


def debris_stone_layout(box_width, box_height, region_cells=(2, 2),
                        rotation_quarters=0):
    """Return deterministic, non-stretched top-view stone descriptors.

    The canonical tile covers two by two map cells.  A 4x2 debris region gets
    two copies, a 2x4 region gets two copies and a 4x4 region gets four copies.
    Each descriptor is ``(cx, cy, rx, ry, angle, colour_index)`` in pixels
    relative to the supplied box.  ``rx / ry`` is constant for every box, so
    rectangular debris areas add stones without changing their aspect ratio.
    """
    cells_w = max(1, int(region_cells[0]))
    cells_h = max(1, int(region_cells[1]))
    tile_cols = max(1, int(math.ceil(cells_w / 2.0)))
    tile_rows = max(1, int(math.ceil(cells_h / 2.0)))
    slot_w = float(box_width) / tile_cols
    slot_h = float(box_height) / tile_rows
    tile_size = max(1.0, min(slot_w, slot_h))
    quarter_turn = int(rotation_quarters) % 4 * (math.pi / 2.0)

    stones = []
    for row in range(tile_rows):
        for column in range(tile_cols):
            origin_x = column * slot_w + 0.5 * (slot_w - tile_size)
            origin_y = row * slot_h + 0.5 * (slot_h - tile_size)
            # Alternate a quarter turn between neighbouring tiles while keeping
            # the exact same stone glyphs and dimensions.
            tile_turn = quarter_turn + ((row + column) % 4) * (math.pi / 2.0)
            ct = math.cos(tile_turn)
            st = math.sin(tile_turn)
            for u, v, radius, angle, colour_index in _DEBRIS_TILE_PATTERN:
                local_x = (u - 0.5) * tile_size
                local_y = (v - 0.5) * tile_size
                rotated_x = local_x * ct - local_y * st
                rotated_y = local_x * st + local_y * ct
                cx = origin_x + 0.5 * tile_size + rotated_x
                cy = origin_y + 0.5 * tile_size + rotated_y
                base = max(1.4, radius * tile_size)
                rx = base * DEBRIS_STONE_ASPECT
                ry = base
                stones.append((
                    cx, cy, rx, ry,
                    angle + tile_turn,
                    int(colour_index) % len(DEBRIS_STONE_COLORS),
                ))
    return stones


# =========================================================================
# Configuration dialog
# =========================================================================
def run_config_dialog():
    """Display the bilingual experiment-configuration dialog."""
    import tkinter as tk
    from tkinter import ttk

    result = {}
    root = tk.Tk()
    root.title(tr("config_window_title"))
    root.geometry("1040x760")
    root.minsize(940, 680)

    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except tk.TclError:
        pass
    style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
    style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
    style.configure("TLabel", font=("Segoe UI", 9))
    style.configure("TEntry", padding=3)

    outer = ttk.Frame(root, padding=14)
    outer.pack(fill="both", expand=True)
    ttk.Label(
        outer,
        text=tr("config_title"),
        style="Title.TLabel",
    ).pack(anchor="w", pady=(0, 12))

    columns = ttk.Frame(outer)
    columns.pack(fill="both", expand=True)
    columns.columnconfigure(0, weight=1, uniform="config")
    columns.columnconfigure(1, weight=1, uniform="config")

    batch = ttk.LabelFrame(columns, text=tr("batch_parameters"),
                           style="Section.TLabelframe", padding=12)
    weights = ttk.LabelFrame(columns, text=tr("fixed_weights"),
                             style="Section.TLabelframe", padding=12)
    batch.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
    weights.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

    batch_defaults = [
        ("n_floors", tr("n_floors"), "4"),
        ("Tr", tr("run_duration"), "600"),
        ("Ts", tr("floor_change_cost"), "60"),
        ("n_runs", tr("n_buildings"), "5"),
        ("base_seed", tr("base_seed"), "0"),
        ("Fw_values", tr("fw_values"), "0; 0.3; 1"),
        ("Opt_values", tr("opt_values"), "ON; OFF"),
    ]
    weight_defaults = [
        ("Iw", tr("iw"), f"{IW_DEFAULT:g}"),
        ("Dw", tr("dw"), f"{DW_DEFAULT:g}"),
        ("Vw", tr("vw"), f"{VW_DEFAULT:g}"),
        ("Cw", tr("cw"), f"{CW_DEFAULT:g}"),
        ("Ow", tr("ow"), f"{OW_DEFAULT:g}"),
        ("persistence_weight", tr("wp"),
         f"{ACTIVE_TARGET_HYSTERESIS_WEIGHT_DEFAULT:g}"),
        ("target_switch_margin", tr("ms"),
         f"{TARGET_SWITCH_MARGIN_DEFAULT:g}"),
    ]

    variables = {}

    def add_entries(parent, definitions):
        parent.columnconfigure(1, weight=1)
        for row, (key, label, default) in enumerate(definitions):
            ttk.Label(parent, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=5)
            var = tk.StringVar(value=default)
            variables[key] = var
            entry = ttk.Entry(parent, textvariable=var, width=22)
            entry.grid(row=row, column=1, sticky="ew", pady=5)

    add_entries(batch, batch_defaults)
    add_entries(weights, weight_defaults)

    environment = ttk.LabelFrame(
        outer, text=tr("environment_fixed"),
        style="Section.TLabelframe", padding=12)
    environment.pack(fill="x", pady=(14, 0))
    environment.columnconfigure(1, weight=1)
    environment.columnconfigure(3, weight=1)

    layout_var = tk.StringVar(value=LAYOUT_OFFICE)
    ttk.Label(environment, text=tr("topology")).grid(
        row=0, column=0, sticky="nw", padx=(0, 12), pady=4)
    layout_box = ttk.Frame(environment)
    layout_box.grid(row=0, column=1, sticky="w", pady=4)
    ttk.Radiobutton(layout_box, text=tr("office_floors"),
                    variable=layout_var, value=LAYOUT_OFFICE).pack(
                        side="left", padx=(0, 16))
    ttk.Radiobutton(layout_box, text=tr("free_topology"),
                    variable=layout_var, value=LAYOUT_FREE).pack(side="left")

    include_objects_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        environment, text=tr("add_objects"),
        variable=include_objects_var,
    ).grid(row=0, column=2, columnspan=2, sticky="w", padx=(30, 0), pady=4)

    ttk.Label(environment, text=tr("object_density")).grid(
        row=1, column=0, sticky="w", padx=(0, 12), pady=6)
    density_var = tk.StringVar(value="1")
    ttk.Entry(environment, textvariable=density_var, width=12).grid(
        row=1, column=1, sticky="w", pady=6)
    ttk.Label(
        environment,
        text=tr("combination_note"),
        foreground="#555b66",
    ).grid(row=1, column=2, columnspan=2, sticky="w", padx=(30, 0), pady=6)

    status_var = tk.StringVar(value="")
    ttk.Label(outer, textvariable=status_var, foreground="crimson").pack(
        fill="x", pady=(10, 2))

    button_row = ttk.Frame(outer)
    button_row.pack(fill="x", pady=(6, 0))

    def split_values(text):
        return [part.strip() for part in text.replace(";", ",").split(",")
                if part.strip()]

    def nonnegative(key, name):
        value = float(variables[key].get().strip())
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(tr("finite_nonnegative", name=name))
        return value

    def on_start():
        try:
            n_floors = int(variables["n_floors"].get().strip())
            Tr = float(variables["Tr"].get().strip())
            Ts = float(variables["Ts"].get().strip())
            n_runs = int(variables["n_runs"].get().strip())
            base_seed = int(variables["base_seed"].get().strip())
            Fw_values = [float(v) for v in split_values(
                variables["Fw_values"].get())]
            if any(not math.isfinite(v) or v < 0 for v in Fw_values):
                raise ValueError(tr("fw_finite"))
            Opt_values = []
            for item in split_values(variables["Opt_values"].get()):
                upper = item.upper()
                if upper not in {"ON", "OFF"}:
                    raise ValueError(tr("invalid_opt", value=item))
                parsed = upper == "ON"
                if parsed not in Opt_values:
                    Opt_values.append(parsed)
            if n_floors < 2:
                raise ValueError(tr("at_least_two_floors"))
            if Tr <= 0 or Ts < 0 or not math.isfinite(Tr) or not math.isfinite(Ts):
                raise ValueError(tr("invalid_times"))
            if n_runs < 1 or not Fw_values or not Opt_values:
                raise ValueError(tr("invalid_batch"))
            density = float(density_var.get().strip())
            if not OBJECT_DENSITY_MIN <= density <= OBJECT_DENSITY_MAX:
                raise ValueError(tr(
                    "density_range",
                    minimum=OBJECT_DENSITY_MIN,
                    maximum=OBJECT_DENSITY_MAX,
                ))

            result.update(
                n_floors=n_floors, Tr=Tr, Ts=Ts, n_runs=n_runs,
                base_seed=base_seed, Fw_values=Fw_values,
                Opt_values=Opt_values,
                Iw=nonnegative("Iw", "Iw"),
                Dw=nonnegative("Dw", "Dw"),
                Vw=nonnegative("Vw", "Vw"),
                Cw=nonnegative("Cw", "Cw"),
                Ow=nonnegative("Ow", "Ow"),
                persistence_weight=nonnegative("persistence_weight", "Wp"),
                target_switch_margin=nonnegative("target_switch_margin", "Ms"),
                layout_mode=layout_var.get(),
                include_objects=bool(include_objects_var.get()),
                object_density=density,
            )
            root.destroy()
        except Exception as error:
            status_var.set(tr("input_error", error=error))

    ttk.Button(button_row, text=tr("cancel"), command=root.destroy).pack(
        side="right", padx=(8, 0))
    ttk.Button(button_row, text=tr("start_simulation"), command=on_start).pack(
        side="right")
    root.bind("<Return>", lambda _event: on_start())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return result


# =========================================================================
# Pygame live viewer
# =========================================================================
class RunViewer:
    """Smooth fixed-step live viewer with session-persistent playback speed.

    The renderer never decides how far the robot moves.  It only accumulates
    wall-clock time, asks ``RunState`` to execute 10 ms physical steps and then
    interpolates the last two poses for display.  A slow frame is capped rather
    than converted into a large position jump.  The selected 1x/2x/4x/8x time
    scale is stored at class level and therefore survives the creation of the
    viewer for the following experimental episode.
    """

    _persistent_time_scale = 1.0
    _allowed_time_scales = (1.0, 2.0, 4.0, 8.0)

    # Max Turbo is session-persistent so that a long unattended batch remains
    # accelerated when the current episode ends and the next one starts.  The
    # last fully rendered frame is also retained while the shared Pygame window
    # stays open; only the Run/Episode status strip is refreshed.
    _persistent_max_turbo = False
    _persistent_frozen_frame = None
    # User preference, distinct from the lifetime of the Panda3D process.
    # Each RunViewer owns a fresh CameraHandle because a new building requires
    # a new static scene, but this class-level flag makes the *enabled state*
    # survive all following episodes and buildings in the same batch.
    _persistent_camera_enabled = False
    _restore_camera_after_turbo = False
    # The display remains open between buildings.  These fields let the start
    # diagnostic service the operating-system event queue and publish a live
    # heartbeat instead of leaving the window apparently frozen.
    _batch_abort_requested = False
    _transition_last_draw_s = 0.0
    _transition_fonts = None

    def __init__(
        self,
        run_state: RunState,
        run_label: str,
        speed_ticks_per_frame: int = 1,
        initial_time_scale: float | None = None,
    ):
        self.rs = run_state
        self.run_label = run_label
        # ``speed_ticks_per_frame`` is retained only for compatibility with
        # older callers.  It must not reset the playback speed between runs.
        del speed_ticks_per_frame
        if initial_time_scale is None:
            initial_time_scale = type(self)._persistent_time_scale
        self.time_scale = self._normalise_time_scale(initial_time_scale)
        type(self)._persistent_time_scale = self.time_scale
        self.paused = False
        self._gt_surface_cache: dict[tuple[int, int, int], "pygame.Surface"] = {}
        self._map_surface_cache = None
        self._map_surface_key = None
        self._map_surface_age_s = math.inf
        # Optional first-person camera process. It is created only when the
        # user presses the button, so disabled mode has no rendering overhead.
        self._camera3d = CameraHandle()
        self._camera3d_last_send_s = 0.0
        self.max_turbo = bool(type(self)._persistent_max_turbo)
        self._turbo_last_status_update_s = 0.0

    @staticmethod
    def _camera_button_rect(window_w: int):
        """Return the clickable camera toggle area in the main viewer header."""
        assert pygame is not None
        return pygame.Rect(window_w - 230, 70, 212, 30)

    @staticmethod
    def _max_turbo_button_rect(window_w: int):
        """Clickable Max Turbo area, immediately left of the 3-D button."""
        assert pygame is not None
        return pygame.Rect(window_w - 390, 70, 150, 30)

    @classmethod
    def reset_abort_request(cls):
        """Clear a previous close/ESC request before a new batch starts."""
        cls._batch_abort_requested = False

    @classmethod
    def service_batch_transition(
        cls,
        title: str,
        detail: str,
        progress: float | None = None,
        force: bool = False,
    ) -> bool:
        """Keep the persistent viewer responsive between two buildings.

        The initial-pose diagnostic runs outside :meth:`run`, before the next
        ``RunViewer`` instance exists.  Version 32 therefore stopped polling
        Pygame for several seconds at every building boundary.  On Windows the
        still-open window could be marked as non-responsive, while the terminal
        remained silent until all Fw/Opt probes had completed.

        This class-level heartbeat consumes window events, lets the user stop
        Max Turbo or close the experiment, and redraws only a compact status
        strip.  It never advances the simulation and consequently cannot change
        any experimental result.
        """
        if pygame is None or not pygame.get_init():
            return not cls._batch_abort_requested
        screen = pygame.display.get_surface()
        if screen is None:
            return not cls._batch_abort_requested

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cls._batch_abort_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                cls._batch_abort_requested = True
            elif (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and
                  cls._max_turbo_button_rect(screen.get_width()).collidepoint(
                      event.pos)):
                # There is no active episode here, so toggling only changes the
                # persistent mode used by the next viewer.  The last complete
                # frame is retained when entering Turbo and released when
                # leaving it.
                cls._persistent_max_turbo = not cls._persistent_max_turbo
                if cls._persistent_max_turbo:
                    cls._persistent_frozen_frame = screen.copy()
                else:
                    cls._persistent_frozen_frame = None

        if cls._batch_abort_requested:
            return False

        now = time.perf_counter()
        if (not force and
                now - cls._transition_last_draw_s < MAX_TURBO_STATUS_REFRESH_S):
            return True
        cls._transition_last_draw_s = now

        if cls._persistent_max_turbo:
            frozen = cls._persistent_frozen_frame
            if frozen is not None and frozen.get_size() == screen.get_size():
                screen.blit(frozen, (0, 0))
            elif frozen is None:
                screen.fill(BACKGROUND)

        strip = pygame.Rect(0, 0, screen.get_width(), 132)
        veil = pygame.Surface(strip.size, pygame.SRCALPHA)
        veil.fill((20, 24, 32, 235) if cls._persistent_max_turbo
                  else (30, 36, 48, 220))
        screen.blit(veil, strip.topleft)

        if cls._transition_fonts is None:
            cls._transition_fonts = (
                pygame.font.SysFont("segoeui", 20, bold=True),
                pygame.font.SysFont("segoeui", 13),
                pygame.font.SysFont("segoeui", 13),
            )
        title_font, detail_font, button_font = cls._transition_fonts
        heading = (
            f"MAX TURBO   |   {title}" if cls._persistent_max_turbo
            else title
        )
        heading_surface = title_font.render(
            heading, True, (255, 255, 255))
        screen.blit(
            heading_surface,
            heading_surface.get_rect(center=(screen.get_width() // 2, 27)),
        )

        # Keep long diagnostic descriptions inside the window.  Truncation is
        # visual only; the terminal receives the complete text.
        max_detail_width = max(200, screen.get_width() - 430)
        visible_detail = detail
        while len(visible_detail) > 8:
            candidate = detail_font.render(
                visible_detail, True, (222, 227, 236))
            if candidate.get_width() <= max_detail_width:
                break
            visible_detail = visible_detail[:-2]
        if visible_detail != detail:
            visible_detail = visible_detail.rstrip() + "..."
        detail_surface = detail_font.render(
            visible_detail, True, (222, 227, 236))
        screen.blit(
            detail_surface,
            detail_surface.get_rect(center=(screen.get_width() // 2, 56)),
        )

        if progress is not None:
            fraction = max(0.0, min(1.0, float(progress)))
            bar = pygame.Rect(28, 82, max(80, screen.get_width() - 440), 14)
            pygame.draw.rect(screen, (69, 76, 91), bar, border_radius=5)
            fill = bar.copy()
            fill.width = int(round(bar.width * fraction))
            if fill.width > 0:
                pygame.draw.rect(screen, (76, 177, 111), fill, border_radius=5)
            pygame.draw.rect(screen, (218, 224, 235), bar, 1, border_radius=5)
            pct = detail_font.render(
                f"{fraction * 100.0:5.1f}%", True, (240, 243, 248))
            screen.blit(pct, (bar.right + 10, bar.top - 2))

        button_rect = cls._max_turbo_button_rect(screen.get_width())
        fill_color = ((196, 82, 38) if cls._persistent_max_turbo
                      else (71, 105, 173))
        pygame.draw.rect(screen, fill_color, button_rect, border_radius=6)
        pygame.draw.rect(
            screen, (255, 255, 255), button_rect, 1, border_radius=6)
        button_label = button_font.render(
            tr("stop_max_turbo") if cls._persistent_max_turbo else tr("max_turbo"),
            True, (255, 255, 255),
        )
        screen.blit(
            button_label, button_label.get_rect(center=button_rect.center))
        pygame.display.update(strip)
        return True

    def _open_persistent_camera_if_requested(self, pose):
        """Open the current building in 3-D when the session preference is ON.

        The Panda3D process is intentionally recreated for every episode so it
        receives the exact static geometry belonging to the current RunState.
        The class-level preference is not cleared when that process is closed;
        consequently the view reopens automatically after a condition or a
        whole building changes.
        """
        if (type(self)._persistent_camera_enabled and
                not self.max_turbo and not self._camera3d.active):
            self._camera3d.open(
                self.rs.floors,
                (self.rs.robot.floor, pose[0], pose[1], pose[2]),
            )

    def _toggle_camera3d(self, pose):
        """Toggle the session-persistent first-person camera preference."""
        enabled = not type(self)._persistent_camera_enabled
        type(self)._persistent_camera_enabled = enabled
        type(self)._restore_camera_after_turbo = enabled and self.max_turbo
        if enabled:
            self._open_persistent_camera_if_requested(pose)
        else:
            self._camera3d.close()

    def _draw_camera_button(self, screen, font, pose):
        """Draw the 3-D toggle using the persistent requested state.

        Green means that the external process is currently alive.  Amber means
        that the preference is enabled but the process is temporarily suspended
        (normally because Max Turbo is active).  Grey means disabled.
        """
        assert pygame is not None
        del pose  # Kept in the signature for compatibility with old callers.
        rect = self._camera_button_rect(screen.get_width())
        requested = bool(type(self)._persistent_camera_enabled)
        active = self._camera3d.active
        if active:
            fill = (53, 139, 92)
        elif requested:
            fill = (176, 125, 46)
        else:
            fill = (77, 91, 112)
        pygame.draw.rect(screen, fill, rect, border_radius=6)
        pygame.draw.rect(screen, (255, 255, 255), rect, 1, border_radius=6)
        label = font.render(
            tr("disable_3d") if requested else tr("enable_3d"),
            True, (255, 255, 255),
        )
        screen.blit(label, label.get_rect(center=rect.center))

    def _draw_max_turbo_button(self, screen, font):
        """Draw the toggle beside the 3-D camera control."""
        assert pygame is not None
        rect = self._max_turbo_button_rect(screen.get_width())
        fill = (196, 82, 38) if self.max_turbo else (71, 105, 173)
        pygame.draw.rect(screen, fill, rect, border_radius=6)
        pygame.draw.rect(screen, (255, 255, 255), rect, 1, border_radius=6)
        label = font.render(
            tr("stop_max_turbo") if self.max_turbo else tr("max_turbo"),
            True, (255, 255, 255),
        )
        screen.blit(label, label.get_rect(center=rect.center))

    def _set_max_turbo(self, enabled: bool, screen, pose):
        """Enter or leave unthrottled headless simulation mode.

        Entering stores the latest complete frame, suspends the expensive 3-D
        renderer, and preserves the selected 1x/2x/4x/8x multiplier.  Leaving
        restarts the 3-D view only if it was open before acceleration.
        """
        enabled = bool(enabled)
        if enabled == self.max_turbo:
            return
        if enabled:
            type(self)._persistent_frozen_frame = screen.copy()
            # Suspending the renderer must not disable the user's preference.
            # The preference is restored automatically after Turbo and also in
            # every subsequent RunViewer, including a newly generated building.
            type(self)._restore_camera_after_turbo = bool(
                type(self)._persistent_camera_enabled)
            if self._camera3d.active:
                self._camera3d.close()
            self.max_turbo = True
            type(self)._persistent_max_turbo = True
            self._turbo_last_status_update_s = 0.0
        else:
            self.max_turbo = False
            type(self)._persistent_max_turbo = False
            type(self)._persistent_frozen_frame = None
            if type(self)._persistent_camera_enabled:
                self._open_persistent_camera_if_requested(pose)
            type(self)._restore_camera_after_turbo = False

    def _draw_max_turbo_status(self, screen, fonts, force=False):
        """Refresh only the Run/Episode banner while the video stays frozen."""
        assert pygame is not None
        now = time.perf_counter()
        if not force and now - self._turbo_last_status_update_s < MAX_TURBO_STATUS_REFRESH_S:
            return
        self._turbo_last_status_update_s = now

        frozen = type(self)._persistent_frozen_frame
        if frozen is not None and frozen.get_size() == screen.get_size():
            screen.blit(frozen, (0, 0))
        elif frozen is None:
            screen.fill(BACKGROUND)

        title_font, _info_font, small_font, _legend_title_font = fonts
        strip = pygame.Rect(0, 0, screen.get_width(), 108)
        veil = pygame.Surface(strip.size, pygame.SRCALPHA)
        veil.fill((20, 24, 32, 232))
        screen.blit(veil, strip.topleft)

        status = title_font.render(
            f"MAX TURBO   |   {self.run_label}",
            True, (255, 255, 255),
        )
        screen.blit(status, status.get_rect(center=(screen.get_width() // 2, 30)))
        note = small_font.render(
            tr("turbo_note"),
            True, (220, 225, 235),
        )
        screen.blit(note, note.get_rect(center=(screen.get_width() // 2, 57)))
        self._draw_max_turbo_button(screen, small_font)
        pygame.display.update(strip)

    @classmethod
    def _normalise_time_scale(cls, value):
        """Return the closest supported playback multiplier."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 1.0
        return min(cls._allowed_time_scales, key=lambda item: abs(item - numeric))

    @classmethod
    def persistent_time_scale(cls):
        """Playback multiplier that will be used by the next viewer."""
        return float(cls._persistent_time_scale)

    def _set_time_scale(self, value):
        self.time_scale = self._normalise_time_scale(value)
        type(self)._persistent_time_scale = self.time_scale
        return self.time_scale

    # ------------------------------------------------------------------ layout
    @staticmethod
    def _window_size():
        assert pygame is not None
        desktop_sizes = pygame.display.get_desktop_sizes()
        if desktop_sizes:
            desktop_w, desktop_h = desktop_sizes[0]
        else:
            desktop_w, desktop_h = 1600, 900
        return (
            min(1500, max(820, desktop_w - 40)),
            min(920, max(650, desktop_h - 70)),
        )

    @staticmethod
    def _layout(window_w: int, window_h: int):
        assert pygame is not None
        margin = 18
        gap = 22
        header_h = 112
        legend_h = 205
        available_h = window_h - header_h - legend_h - margin
        available_w = window_w - 2 * margin - gap
        panel_w = available_w // 2
        panel_h = int(round(panel_w * WORLD_H_M / WORLD_W_M))
        if panel_h > available_h:
            panel_h = available_h
            panel_w = int(round(panel_h * WORLD_W_M / WORLD_H_M))
        total_w = 2 * panel_w + gap
        left_x = (window_w - total_w) // 2
        panel_y = header_h
        left = pygame.Rect(left_x, panel_y, panel_w, panel_h)
        right = pygame.Rect(left.right + gap, panel_y, panel_w, panel_h)
        legend_top = panel_y + panel_h + 31
        legend = pygame.Rect(
            margin,
            legend_top,
            window_w - 2 * margin,
            window_h - legend_top - 8,
        )
        return left, right, legend

    # ------------------------------------------------------------------ mapping
    @staticmethod
    def _world_to_panel(x_m: float, y_m: float, rect):
        return (
            int(round(rect.left + x_m / WORLD_W_M * rect.width)),
            int(round(rect.top + y_m / WORLD_H_M * rect.height)),
        )

    def _ground_truth_surface(self, floor_index: int, size):
        assert pygame is not None
        key = (floor_index, int(size[0]), int(size[1]))
        cached = self._gt_surface_cache.get(key)
        if cached is not None:
            return cached

        floor = self.rs.floors[floor_index]
        clipped = np.clip(floor.grid, 0, len(GT_COLORS) - 1)
        rgb = GT_COLORS[clipped].copy()
        for item in floor.environment_objects:
            x0, y0, x1, y1 = item.region
            rgb[y0:y1 + 1, x0:x1 + 1] = OBJECT_CELL_COLORS[item.kind]

        raw = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        surface = pygame.transform.scale(raw, size)
        panel_rect = pygame.Rect(0, 0, int(size[0]), int(size[1]))
        self._draw_environment_objects(surface, panel_rect, floor)
        self._draw_floor_stair_icons(surface, panel_rect, floor)
        self._gt_surface_cache[key] = surface
        return surface

    def _constructed_map_surface(self, floor_index: int, fmap, size):
        """Return the 5 cm probabilistic occupancy map shown on the right.

        The planner still uses the paper-compatible 0.5 m map.  This surface,
        instead, is generated from ``pixel_occ_log_odds`` and
        ``pixel_occ_observed``: unknown pixels remain grey, visible free space
        tends toward white, and measured obstacle surfaces tend toward black.
        No complete coarse cell is revealed merely because one laser return
        touched it.

        Rebuilding and scaling this image at 60 FPS would be wasted work
        because the LiDAR itself updates at 10 Hz.  Caching leaves the renderer
        free to update the continuous robot pose every frame.
        """
        assert pygame is not None
        key = (floor_index, int(size[0]), int(size[1]))
        if (self._map_surface_cache is not None
                and self._map_surface_key == key
                and self._map_surface_age_s < MAP_SURFACE_REFRESH_S):
            return self._map_surface_cache

        probability = fmap.pixel_occupancy_probability().astype(
            np.float32, copy=False
        )
        observed = fmap.pixel_occ_observed
        height, width = probability.shape
        rgb = np.empty((height, width, 3), dtype=np.float32)
        rgb[:, :] = UNKNOWN_COLOR

        # A probability of 0.5 has zero confidence and therefore retains the
        # unknown grey.  Increasing confidence interpolates toward white for
        # free space or toward dark grey for occupied space.  This preserves a
        # familiar occupancy-grid appearance after even a single scan without
        # pretending that unseen pixels are known.
        free_mask = observed & (probability < 0.5)
        occupied_mask = observed & (probability >= 0.5)
        # The colour transfer is intentionally steeper than the probability
        # scale: one valid laser observation must already be distinguishable
        # from unknown grey, while repeated observations still increase the
        # contrast as the log-odds accumulate.
        free_confidence = np.clip(
            (0.5 - probability[free_mask]) / 0.15, 0.0, 1.0
        )[:, None]
        occupied_confidence = np.clip(
            (probability[occupied_mask] - 0.5) / 0.20, 0.0, 1.0
        )[:, None]
        if free_confidence.size:
            rgb[free_mask] = (
                UNKNOWN_COLOR * (1.0 - free_confidence)
                + MAP_FREE_COLOR * free_confidence
            )
        if occupied_confidence.size:
            rgb[occupied_mask] = (
                UNKNOWN_COLOR * (1.0 - occupied_confidence)
                + MAP_OCC_COLOR * occupied_confidence
            )

        semantic_strength_coarse = np.clip(
            (fmap.semantic.astype(np.float32) - 0.55) / 0.45,
            0.0,
            1.0,
        )
        semantic_strength = np.repeat(
            np.repeat(semantic_strength_coarse, FRONTIER_SCALE, axis=0),
            FRONTIER_SCALE,
            axis=1,
        )
        semantic_strength = semantic_strength[:height, :width]
        alpha = (0.68 * semantic_strength)[..., None]
        rgb = rgb * (1.0 - alpha) + SEMANTIC_COLOR * alpha
        raw = pygame.surfarray.make_surface(
            np.transpose(np.clip(rgb, 0, 255).astype(np.uint8), (1, 0, 2))
        )
        self._map_surface_cache = pygame.transform.scale(raw, size)
        self._map_surface_key = key
        self._map_surface_age_s = 0.0
        return self._map_surface_cache

    # ------------------------------------------------------------------ symbols
    @staticmethod
    def _draw_star(surface, center, radius, color, outline=(20, 20, 20)):
        assert pygame is not None
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            current_radius = radius if index % 2 == 0 else radius * 0.43
            points.append(
                (
                    center[0] + current_radius * math.cos(angle),
                    center[1] + current_radius * math.sin(angle),
                )
            )
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, outline, points, 1)

    @staticmethod
    def _draw_dashed_line(surface, color, start, end, width=2, dash=8, gap=5):
        assert pygame is not None
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        position = 0.0
        while position < length:
            segment_end = min(length, position + dash)
            a = (start[0] + ux * position, start[1] + uy * position)
            b = (start[0] + ux * segment_end, start[1] + uy * segment_end)
            pygame.draw.line(surface, color, a, b, width)
            position += dash + gap

    @classmethod
    def _draw_dashed_polyline(cls, surface, color, points, width=2):
        for first, second in zip(points, points[1:]):
            cls._draw_dashed_line(surface, color, first, second, width=width)

    @staticmethod
    def _draw_spot(surface, rect, x_m: float, y_m: float, theta: float):
        assert pygame is not None
        scale = min(rect.width / WORLD_W_M, rect.height / WORLD_H_M)
        cx = rect.left + x_m / WORLD_W_M * rect.width
        cy = rect.top + y_m / WORLD_H_M * rect.height
        ct, st = math.cos(theta), math.sin(theta)

        def transform(local_x, local_y):
            return (
                cx + scale * (local_x * ct - local_y * st),
                cy + scale * (local_x * st + local_y * ct),
            )

        # Four articulated-looking legs and feet.
        for local_x, local_y in [
            (-0.29, -0.20),
            (-0.29, 0.20),
            (0.29, -0.20),
            (0.29, 0.20),
        ]:
            hip = transform(local_x, local_y * 0.65)
            foot = transform(local_x, local_y)
            pygame.draw.line(surface, ROBOT_EDGE, hip, foot, max(1, int(scale * 0.05)))
            pygame.draw.circle(
                surface,
                ROBOT_FOOT,
                (int(foot[0]), int(foot[1])),
                max(2, int(scale * 0.055)),
            )

        corners = [
            transform(-0.42, -0.20),
            transform(0.42, -0.20),
            transform(0.42, 0.20),
            transform(-0.42, 0.20),
        ]
        pygame.draw.polygon(surface, ROBOT_YELLOW, corners)
        pygame.draw.polygon(surface, ROBOT_EDGE, corners, max(1, int(scale * 0.035)))
        head = transform(0.49, 0.0)
        pygame.draw.circle(
            surface,
            ROBOT_YELLOW,
            (int(head[0]), int(head[1])),
            max(3, int(scale * 0.13)),
        )
        pygame.draw.circle(
            surface,
            ROBOT_EDGE,
            (int(head[0]), int(head[1])),
            max(3, int(scale * 0.13)),
            max(1, int(scale * 0.025)),
        )

    @staticmethod
    def _draw_victim(surface, center, font):
        assert pygame is not None
        pygame.draw.circle(surface, (35, 35, 35), center, 9)
        pygame.draw.circle(surface, VICTIM_RED, center, 8)
        glyph = font.render("P", True, (255, 255, 255))
        surface.blit(glyph, glyph.get_rect(center=center))


    @staticmethod
    def _cell_region_to_panel_rect(region, rect):
        """Convert an inclusive cell rectangle to panel pixel coordinates."""
        assert pygame is not None
        x0, y0, x1, y1 = region
        left = rect.left + int(round(x0 / FLOOR_W * rect.width))
        top = rect.top + int(round(y0 / FLOOR_H * rect.height))
        right = rect.left + int(round((x1 + 1) / FLOOR_W * rect.width))
        bottom = rect.top + int(round((y1 + 1) / FLOOR_H * rect.height))
        return pygame.Rect(
            left,
            top,
            max(4, right - left),
            max(4, bottom - top),
        )

    @staticmethod
    def _draw_table_icon(surface, box):
        assert pygame is not None
        pad = max(1, min(box.width, box.height) // 7)
        top = box.inflate(-2 * pad, -2 * pad)
        if top.width < 4 or top.height < 4:
            top = box.copy()
        pygame.draw.rect(surface, TABLE_TOP, top, border_radius=max(1, pad))
        pygame.draw.rect(surface, TABLE_EDGE, top, max(1, pad // 2),
                         border_radius=max(1, pad))
        leg_radius = max(1, min(top.width, top.height) // 10)
        for point in (
            (top.left + leg_radius + 1, top.top + leg_radius + 1),
            (top.right - leg_radius - 2, top.top + leg_radius + 1),
            (top.left + leg_radius + 1, top.bottom - leg_radius - 2),
            (top.right - leg_radius - 2, top.bottom - leg_radius - 2),
        ):
            pygame.draw.circle(surface, TABLE_EDGE, point, leg_radius)

    @staticmethod
    def _draw_chair_icon(surface, box, rotation_quarters=0):
        assert pygame is not None
        inset_x = max(2, box.width // 5)
        inset_y = max(2, box.height // 5)
        seat = box.inflate(-2 * inset_x, -2 * inset_y)
        if seat.width < 4 or seat.height < 4:
            seat = box.inflate(-2, -2)
        pygame.draw.rect(surface, CHAIR_TOP, seat,
                         border_radius=max(1, min(seat.width, seat.height) // 4))
        pygame.draw.rect(surface, CHAIR_EDGE, seat, 1,
                         border_radius=max(1, min(seat.width, seat.height) // 4))

        thickness = max(2, min(box.width, box.height) // 6)
        rotation_quarters %= 4
        if rotation_quarters == 0:
            back = pygame.Rect(seat.left, box.top + 1, seat.width, thickness)
        elif rotation_quarters == 1:
            back = pygame.Rect(box.right - thickness - 1, seat.top,
                               thickness, seat.height)
        elif rotation_quarters == 2:
            back = pygame.Rect(seat.left, box.bottom - thickness - 1,
                               seat.width, thickness)
        else:
            back = pygame.Rect(box.left + 1, seat.top, thickness, seat.height)
        pygame.draw.rect(surface, CHAIR_EDGE, back, border_radius=1)

    @staticmethod
    def _draw_top_view_stone(surface, center, rx, ry, angle, fill):
        """Draw one irregular stone from above without anisotropic stretching."""
        assert pygame is not None
        radial_factors = (1.00, 0.88, 1.04, 0.84, 0.97,
                          0.90, 1.02, 0.86, 0.96, 0.91)
        points = []
        ct = math.cos(angle)
        st = math.sin(angle)
        for index, factor in enumerate(radial_factors):
            phase = 2.0 * math.pi * index / len(radial_factors)
            local_x = math.cos(phase) * rx * factor
            local_y = math.sin(phase) * ry * factor
            points.append((
                int(round(center[0] + local_x * ct - local_y * st)),
                int(round(center[1] + local_x * st + local_y * ct)),
            ))
        # A one-pixel shadow and a small highlight make the symbol readable as
        # a top-view stone rather than as a flat triangular fragment.
        shadow = [(x + 1, y + 1) for x, y in points]
        pygame.draw.polygon(surface, (74, 63, 54), shadow)
        pygame.draw.polygon(surface, fill, points)
        pygame.draw.polygon(surface, DEBRIS_EDGE, points, 1)
        highlight_end = (
            int(round(center[0] - 0.30 * rx * math.cos(angle)
                      + 0.18 * ry * math.sin(angle))),
            int(round(center[1] - 0.30 * rx * math.sin(angle)
                      - 0.18 * ry * math.cos(angle))),
        )
        pygame.draw.line(
            surface,
            (190, 172, 149),
            (int(round(center[0])), int(round(center[1]))),
            highlight_end,
            1,
        )

    @classmethod
    def _draw_debris_icon(cls, surface, box, rotation_quarters=0,
                          region_cells=(2, 2)):
        """Draw rubble as repeated small and large top-view stones.

        The stone glyphs retain a fixed aspect ratio.  Larger debris regions
        contain more copies of the same canonical 2x2-cell pile instead of
        stretching a single icon to fill the bounding rectangle.
        """
        assert pygame is not None
        for cx, cy, rx, ry, angle, colour_index in debris_stone_layout(
                box.width,
                box.height,
                region_cells=region_cells,
                rotation_quarters=rotation_quarters):
            cls._draw_top_view_stone(
                surface,
                (box.left + cx, box.top + cy),
                rx,
                ry,
                angle,
                DEBRIS_STONE_COLORS[colour_index],
            )

    @staticmethod
    def _draw_stair_icon(surface, box, going_up=True, direction=(0, -1)):
        """Draw steps and an arrow aligned with the corridor direction."""
        assert pygame is not None
        # Inset the glyph inside its free corridor footprint. This keeps a
        # visible gap from every wall even when the underlying stair cells are
        # adjacent to the corridor boundary in the coarse ground-truth grid.
        inset = max(1, int(round(0.12 * min(box.width, box.height))))
        if box.width > 2 * inset + 2 and box.height > 2 * inset + 2:
            box = box.inflate(-2 * inset, -2 * inset)
        fill = tuple(GT_COLORS[STAIR_UP] if going_up else GT_COLORS[STAIR_DOWN])
        outline = (112, 72, 28) if going_up else (42, 91, 130)
        radius = max(1, min(box.width, box.height) // 8)
        pygame.draw.rect(surface, fill, box, border_radius=radius)
        pygame.draw.rect(surface, outline, box,
                         max(1, min(box.width, box.height) // 12),
                         border_radius=radius)

        dx, dy = direction
        horizontal = abs(dx) >= abs(dy)
        n_steps = max(4, min(7, (box.width if horizontal else box.height) // 4))
        if horizontal:
            for i in range(1, n_steps):
                x = box.left + int(round(i * box.width / n_steps))
                pygame.draw.line(surface, (245, 245, 245),
                                 (x, box.top + 2), (x, box.bottom - 3), 1)
        else:
            for i in range(1, n_steps):
                y = box.top + int(round(i * box.height / n_steps))
                pygame.draw.line(surface, (245, 245, 245),
                                 (box.left + 2, y), (box.right - 3, y), 1)

        # Both UP and DOWN glyphs point in the physical corridor exit direction;
        # colour still distinguishes the destination level.
        length = max(5, int(0.36 * (box.width if horizontal else box.height)))
        cx, cy = box.center
        ux, uy = (1 if dx > 0 else -1 if dx < 0 else 0,
                  1 if dy > 0 else -1 if dy < 0 else 0)
        start = (cx - ux * length, cy - uy * length)
        end = (cx + ux * length, cy + uy * length)
        pygame.draw.line(surface, (255, 255, 255), start, end,
                         max(2, min(box.width, box.height) // 10))
        px, py = -uy, ux
        head = [end,
                (end[0] - ux * 6 + px * 4, end[1] - uy * 6 + py * 4),
                (end[0] - ux * 6 - px * 4, end[1] - uy * 6 - py * 4)]
        pygame.draw.polygon(surface, (255, 255, 255), head)

    @classmethod
    def _draw_floor_stair_icons(cls, surface, rect, floor):
        up_regions = list(getattr(floor, "stair_up_regions", []) or [])
        down_regions = list(getattr(floor, "stair_down_regions", []) or [])
        directions = list(getattr(floor, "stair_core_directions", []) or [])
        if not up_regions and floor.stair_up_region is not None:
            up_regions = [floor.stair_up_region]
        if not down_regions and floor.stair_down_region is not None:
            down_regions = [floor.stair_down_region]
        for index, region in enumerate(up_regions):
            direction = directions[index] if index < len(directions) else (0, -1)
            cls._draw_stair_icon(
                surface, cls._cell_region_to_panel_rect(region, rect),
                True, direction)
        for index, region in enumerate(down_regions):
            direction = directions[index] if index < len(directions) else (0, -1)
            cls._draw_stair_icon(
                surface, cls._cell_region_to_panel_rect(region, rect),
                False, direction)

    @classmethod
    def _draw_object_icon(cls, surface, box, kind, rotation_quarters=0,
                          region_cells=None):
        if kind == OBJECT_TABLE:
            cls._draw_table_icon(surface, box)
        elif kind == OBJECT_CHAIR:
            cls._draw_chair_icon(surface, box, rotation_quarters)
        elif kind == OBJECT_DEBRIS:
            cls._draw_debris_icon(
                surface,
                box,
                rotation_quarters,
                region_cells=region_cells or (2, 2),
            )

    @classmethod
    def _draw_environment_objects(cls, surface, rect, floor):
        for item in floor.environment_objects:
            box = cls._cell_region_to_panel_rect(item.region, rect)
            cells_w = item.region[2] - item.region[0] + 1
            cells_h = item.region[3] - item.region[1] + 1
            cls._draw_object_icon(
                surface,
                box,
                item.kind,
                item.rotation_quarters,
                region_cells=(cells_w, cells_h),
            )

    # ---------------------------------------------------------------- overlays
    def _history_points(self, floor_index: int, rect):
        history = [
            (x, y)
            for floor, x, y in self.rs.pose_history
            if floor == floor_index
        ]
        if len(history) > 1800:
            stride = int(math.ceil(len(history) / 1800))
            history = history[::stride]
        return [self._world_to_panel(x, y, rect) for x, y in history]

    def _draw_ground_truth(self, screen, rect, pose, small_font):
        assert pygame is not None
        rs = self.rs
        floor = rs.floors[rs.robot.floor]
        screen.blit(self._ground_truth_surface(floor.index, rect.size), rect.topleft)

        history = self._history_points(floor.index, rect)
        if len(history) >= 2:
            pygame.draw.lines(screen, TRAJECTORY_BLUE, False, history, 2)

        # Sensor scan is continuous; only every tenth beam is drawn to keep the
        # view readable and the renderer light.  The rays start from the exact
        # continuous pose at which the last scan was acquired, rather than from
        # the current interpolated render pose.
        scan_floor, scan_x, scan_y = rs.last_scan_pose
        if scan_floor == rs.robot.floor:
            start = self._world_to_panel(scan_x, scan_y, rect)
            for index, beam in enumerate(rs.last_laser_scan):
                if index % 10 != 0:
                    continue
                end = self._world_to_panel(
                    beam.endpoint_x_m, beam.endpoint_y_m, rect
                )
                pygame.draw.line(screen, LASER_COLOR, start, end, 1)

        for victim_x, victim_y in floor.victims:
            vx, vy = cell_center_world(victim_x, victim_y)
            self._draw_victim(
                screen,
                self._world_to_panel(vx, vy, rect),
                small_font,
            )

        for frontier in rs.current_frontiers:
            chosen = rs.chosen_frontier is frontier
            reachable = frontier.score is not None and frontier.score > -math.inf
            color = CHOSEN_RED if chosen else (
                FRONTIER_GREEN if reachable else UNREACHABLE_GRAY
            )
            width = 4 if chosen else 2
            if frontier.kind == "standard" and frontier.segments:
                if getattr(frontier, "polylines", None):
                    for polyline in frontier.polylines:
                        points = [
                            self._world_to_panel(x, y, rect)
                            for x, y in polyline
                        ]
                        if len(points) >= 2:
                            pygame.draw.aalines(
                                screen, color, False, points
                            )
                            if width > 1:
                                pygame.draw.lines(
                                    screen, color, False, points, width
                                )
                else:
                    for x1, y1, x2, y2 in frontier.segments:
                        pygame.draw.line(
                            screen, color,
                            self._world_to_panel(x1, y1, rect),
                            self._world_to_panel(x2, y2, rect), width,
                        )
                pygame.draw.circle(
                    screen,
                    color,
                    self._world_to_panel(frontier.x, frontier.y, rect),
                    5 if chosen else 3,
                )
            else:
                self._draw_star(
                    screen,
                    self._world_to_panel(frontier.x, frontier.y, rect),
                    11 if chosen else 8,
                    color,
                )

        chosen = rs.chosen_frontier
        if chosen is not None and chosen.path:
            path_points = [
                self._world_to_panel(x, y, rect) for x, y in chosen.path
            ]
            self._draw_dashed_polyline(screen, PATH_RED, path_points, width=2)

        self._draw_spot(screen, rect, *pose)
        pygame.draw.rect(screen, BORDER, rect, 1)

    def _draw_constructed_map(self, screen, rect, pose):
        assert pygame is not None
        rs = self.rs
        floor = rs.floors[rs.robot.floor]
        fmap = floor.fmap
        screen.blit(self._constructed_map_surface(floor.index, fmap, rect.size), rect.topleft)

        history = self._history_points(floor.index, rect)
        if len(history) >= 2:
            pygame.draw.lines(screen, TRAJECTORY_BLUE, False, history, 2)

        ys, xs = np.where(fmap.semantic >= SEMANTIC_THRESHOLD)
        for cell_x, cell_y in zip(xs.tolist(), ys.tolist()):
            center = self._world_to_panel(*cell_center_world(cell_x, cell_y), rect)
            pygame.draw.line(
                screen,
                SEMANTIC_CYAN,
                (center[0] - 4, center[1] - 4),
                (center[0] + 4, center[1] + 4),
                2,
            )
            pygame.draw.line(
                screen,
                SEMANTIC_CYAN,
                (center[0] - 4, center[1] + 4),
                (center[0] + 4, center[1] - 4),
                2,
            )

        self._draw_spot(screen, rect, *pose)
        pygame.draw.rect(screen, BORDER, rect, 1)

    # ------------------------------------------------------------------ legend
    @staticmethod
    def _legend_groups():
        """Return semantically grouped and localised legend entries."""
        return [
            (
                tr("legend_real_environment"),
                [
                    (tr("legend_free_space"), "square", tuple(GT_COLORS[GT_FREE])),
                    (tr("legend_wall"), "square", tuple(GT_COLORS[WALL])),
                    (tr("legend_open_door"), "square", tuple(GT_COLORS[DOOR])),
                    (tr("legend_stairs_up"), "stair_up", tuple(GT_COLORS[STAIR_UP])),
                    (tr("legend_stairs_down"), "stair_down", tuple(GT_COLORS[STAIR_DOWN])),
                    (tr("legend_table"), OBJECT_TABLE, TABLE_TOP),
                    (tr("legend_chair"), OBJECT_CHAIR, CHAIR_TOP),
                    (tr("legend_debris"), OBJECT_DEBRIS, DEBRIS_TOP),
                    (tr("legend_victim"), "victim", VICTIM_RED),
                ],
            ),
            (
                tr("legend_robot_sensors"),
                [
                    (tr("legend_robot"), "robot", ROBOT_YELLOW),
                    (tr("legend_lidar"), "line", LASER_COLOR),
                    (tr("legend_semantic"), "square", tuple(SEMANTIC_COLOR.astype(int))),
                    (tr("legend_detection"), "cross", SEMANTIC_CYAN),
                ],
            ),
            (
                tr("legend_planning"),
                [
                    (tr("legend_candidate"), "line", FRONTIER_GREEN),
                    (tr("legend_chosen"), "thick_line", CHOSEN_RED),
                    (tr("legend_stair_frontier"), "star", FRONTIER_GREEN),
                    (tr("legend_unreachable"), "line", UNREACHABLE_GRAY),
                    (tr("legend_path"), "dashed", PATH_RED),
                    (tr("legend_trajectory"), "line", TRAJECTORY_BLUE),
                ],
            ),
            (
                tr("legend_internal_map"),
                [
                    (tr("legend_unknown"), "square", tuple(UNKNOWN_COLOR.astype(int))),
                    (tr("legend_free"), "square", tuple(MAP_FREE_COLOR.astype(int))),
                    (tr("legend_occupied"), "square", tuple(MAP_OCC_COLOR.astype(int))),
                ],
            ),
        ]

    def _draw_legend_sample(self, screen, font, label, kind, color, x, y):
        """Draw one legend symbol using the same visual vocabulary as the map."""
        sample_center = (x + 13, y)
        if kind in (OBJECT_TABLE, OBJECT_CHAIR, OBJECT_DEBRIS):
            sample_box = pygame.Rect(sample_center[0] - 11, y - 9, 22, 18)
            self._draw_object_icon(screen, sample_box, kind, 0)
        elif kind in ("stair_up", "stair_down"):
            sample_box = pygame.Rect(sample_center[0] - 10, y - 9, 20, 18)
            self._draw_stair_icon(screen, sample_box, kind == "stair_up")
        elif kind == "square":
            box = pygame.Rect(sample_center[0] - 9, y - 8, 18, 16)
            pygame.draw.rect(screen, color, box)
            pygame.draw.rect(screen, (80, 80, 80), box, 1)
        elif kind in ("line", "thick_line"):
            pygame.draw.line(screen, color,
                             (sample_center[0] - 11, y),
                             (sample_center[0] + 11, y),
                             4 if kind == "thick_line" else 2)
        elif kind == "dashed":
            self._draw_dashed_line(screen, color,
                                   (sample_center[0] - 12, y),
                                   (sample_center[0] + 12, y),
                                   width=2, dash=5, gap=3)
        elif kind == "star":
            self._draw_star(screen, sample_center, 9, color)
        elif kind == "cross":
            pygame.draw.line(screen, color,
                             (sample_center[0] - 7, y - 7),
                             (sample_center[0] + 7, y + 7), 2)
            pygame.draw.line(screen, color,
                             (sample_center[0] - 7, y + 7),
                             (sample_center[0] + 7, y - 7), 2)
        elif kind == "victim":
            pygame.draw.circle(screen, color, sample_center, 8)
            glyph = font.render("P", True, (255, 255, 255))
            screen.blit(glyph, glyph.get_rect(center=sample_center))
        elif kind == "robot":
            body = pygame.Rect(sample_center[0] - 11, y - 6, 22, 12)
            pygame.draw.ellipse(screen, ROBOT_YELLOW, body)
            pygame.draw.ellipse(screen, ROBOT_EDGE, body, 1)

        label_surface = font.render(label, True, TEXT)
        screen.blit(label_surface,
                    (x + 32, y - label_surface.get_height() // 2))

    def _draw_legend(self, screen, rect, title_font, font):
        """Render four semantic legend columns with explicit group headings."""
        assert pygame is not None
        pygame.draw.rect(screen, (248, 249, 251), rect, border_radius=6)
        pygame.draw.rect(screen, BORDER, rect, 1, border_radius=6)
        title = title_font.render("Legenda per categorie", True, TEXT)
        screen.blit(title, (rect.left + 10, rect.top + 7))

        groups = self._legend_groups()
        col_w = rect.width / len(groups)
        usable_top = rect.top + 34
        for column, (heading, entries) in enumerate(groups):
            x = int(rect.left + column * col_w + 10)
            heading_surface = font.render(heading, True, TEXT)
            screen.blit(heading_surface, (x, usable_top))
            pygame.draw.line(screen, BORDER,
                             (x, usable_top + heading_surface.get_height() + 2),
                             (int(rect.left + (column + 1) * col_w - 10),
                              usable_top + heading_surface.get_height() + 2), 1)
            available = rect.bottom - (usable_top + heading_surface.get_height() + 8)
            row_h = max(20, available / max(1, len(entries)))
            for row, (label, kind, color) in enumerate(entries):
                y = int(usable_top + heading_surface.get_height() + 10
                        + row * row_h + row_h / 2)
                self._draw_legend_sample(screen, font, label, kind, color, x, y)

    # ------------------------------------------------------------------ frame
    def _draw_frame(self, screen, left_rect, right_rect, legend_rect, fonts, alpha):
        assert pygame is not None
        title_font, info_font, small_font, legend_title_font = fonts
        screen.fill(BACKGROUND)

        pose = self.rs.interpolated_pose(alpha)
        remaining = max(0.0, self.rs.remaining_time)
        minutes = int(remaining // 60)
        seconds = remaining - minutes * 60
        v, omega = self.rs.last_command

        layout_label = (
            tr("office") if self.rs.cfg.layout_mode == LAYOUT_OFFICE
            else tr("free")
        )
        objects_label = yes_no(self.rs.cfg.include_objects)
        header = title_font.render(
            f"{self.run_label}   |   Fw={self.rs.cfg.Fw:g}   "
            f"Opt={'ON' if self.rs.cfg.Opt else 'OFF'}",
            True,
            TEXT,
        )
        screen.blit(header, (18, 9))
        timer = title_font.render(
            tr("time_remaining", minutes=minutes, seconds=seconds),
            True,
            CHOSEN_RED if remaining < 30 else TEXT,
        )
        screen.blit(timer, (screen.get_width() - timer.get_width() - 18, 9))

        status_text = (
            f"{layout_label}, {tr('objects')}={objects_label}, "
            f"{tr('density')}={self.rs.cfg.object_density:g}x  |  "
            f"{tr('floor')} {self.rs.robot.floor + 1}/{self.rs.cfg.n_floors}  |  "
            f"{tr('visited')} {len(self.rs.visited_floors)}  |  "
            f"{tr('changes')} {self.rs.floor_changes}  |  "
            f"x={pose[0]:.2f} m  y={pose[1]:.2f} m  |  "
            f"v={v:.2f} m/s  omega={omega:.2f} rad/s  |  "
            f"Htarget={self.rs.last_target_continuity:.2f}  "
            f"bonus={self.rs.last_target_hysteresis_bonus:.1f}  |  "
            f"{tr('render')} {RENDER_FPS} FPS, "
            f"{tr('physics')} {1 / PHYSICS_DT_S:.0f} Hz, "
            f"{tr('time_scale')} {self.time_scale:g}x"
        )
        status = info_font.render(status_text, True, MUTED_TEXT)
        screen.blit(status, (18, 42))
        controls_text = tr("controls")
        controls = small_font.render(
            controls_text + (f"   [{tr('paused')}]" if self.paused else ""),
            True,
            CHOSEN_RED if self.paused else MUTED_TEXT,
        )
        screen.blit(controls, (18, 64))
        self._draw_max_turbo_button(screen, small_font)
        self._draw_camera_button(screen, small_font, pose)

        left_title = info_font.render(
            tr("ground_truth_title", floor=self.rs.robot.floor + 1),
            True,
            TEXT,
        )
        right_title = info_font.render(
            tr(
                "map_title",
                floor=self.rs.robot.floor + 1,
                resolution=PIXEL_OCCUPANCY_RESOLUTION_M * 100,
            ),
            True,
            TEXT,
        )
        screen.blit(left_title, (left_rect.left, left_rect.top - 24))
        screen.blit(right_title, (right_rect.left, right_rect.top - 24))

        self._draw_ground_truth(screen, left_rect, pose, small_font)
        self._draw_constructed_map(screen, right_rect, pose)

        coverage_parts = []
        for floor in self.rs.floors:
            explored = float(np.sum(floor.fmap.occ == FREE)) * CELL_SIZE ** 2
            coverage = 100.0 * explored / max(floor.explorable_area_m2, 1e-9)
            coverage_parts.append(f"P{floor.index + 1}: {coverage:5.1f}%")
        coverage = small_font.render(
            f"{tr('coverage')}: " + "   ".join(coverage_parts),
            True,
            TEXT,
        )
        coverage_rect = coverage.get_rect(
            centerx=right_rect.centerx,
            top=right_rect.bottom + 4,
        )
        # The legend begins lower; keep coverage directly under the maps.
        screen.blit(coverage, coverage_rect)

        self._draw_legend(
            screen,
            legend_rect,
            legend_title_font,
            small_font,
        )

    # ------------------------------------------------------------------- loop
    @classmethod
    def shutdown_display(cls):
        """Close the shared Pygame display after the complete experiment."""
        if pygame is not None:
            pygame.display.quit()
            pygame.quit()
        cls._persistent_frozen_frame = None
        cls._persistent_max_turbo = False
        cls._persistent_camera_enabled = False
        cls._restore_camera_after_turbo = False
        cls._batch_abort_requested = False
        cls._transition_last_draw_s = 0.0
        cls._transition_fonts = None

    def _handle_event(self, event, screen, window_w):
        """Handle one viewer event and return ``(running, accumulator_reset)``."""
        running = True
        reset_accumulator = False
        if event.type == pygame.QUIT:
            self.rs.finished = True
            return False, True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pose_now = self.rs.interpolated_pose(1.0)
            if self._max_turbo_button_rect(window_w).collidepoint(event.pos):
                self._set_max_turbo(not self.max_turbo, screen, pose_now)
                reset_accumulator = True
            elif (not self.max_turbo and
                  self._camera_button_rect(window_w).collidepoint(event.pos)):
                self._toggle_camera3d(pose_now)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.rs.finished = True
                running = False
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
                reset_accumulator = True
            elif event.key == pygame.K_1 and not self.max_turbo:
                self._set_time_scale(1.0)
            elif event.key == pygame.K_2 and not self.max_turbo:
                self._set_time_scale(2.0)
            elif event.key == pygame.K_3 and not self.max_turbo:
                self._set_time_scale(4.0)
            elif event.key == pygame.K_4 and not self.max_turbo:
                self._set_time_scale(8.0)
        return running, reset_accumulator

    def run(self):
        if pygame is None:
            raise RuntimeError(
                tr("pygame_missing")
            )

        # Keep one display alive for the whole batch.  This is essential in Max
        # Turbo: when an episode ends, the same frozen image remains visible
        # and only the Run/Episode banner changes for the next condition.
        if not pygame.get_init():
            pygame.init()
        window_w, window_h = self._window_size()
        screen = pygame.display.get_surface()
        if screen is None or screen.get_size() != (window_w, window_h):
            screen = pygame.display.set_mode((window_w, window_h))
        pygame.display.set_caption(tr("viewer_caption"))
        clock = pygame.time.Clock()
        left_rect, right_rect, legend_rect = self._layout(window_w, window_h)

        fonts = (
            pygame.font.SysFont("segoeui", 20, bold=True),
            pygame.font.SysFont("segoeui", 15),
            pygame.font.SysFont("segoeui", 13),
            pygame.font.SysFont("segoeui", 14, bold=True),
        )

        accumulator = 0.0
        alive = True
        running = True

        if self.max_turbo:
            self._draw_max_turbo_status(screen, fonts, force=True)
        else:
            # Recreate the camera scene for this RunState whenever the user had
            # enabled 3-D in any previous episode/building.
            pose_now = self.rs.interpolated_pose(1.0)
            self._open_persistent_camera_if_requested(pose_now)
            type(self)._restore_camera_after_turbo = False

        while running and alive and not self.rs.finished:
            # In ordinary mode the 60 FPS clock controls only rendering.  Max
            # Turbo deliberately avoids clock.tick(), the real-time accumulator
            # and the per-frame physics cap, so the CPU spends nearly all its
            # time advancing deterministic 10 ms simulation steps.
            if self.max_turbo and not self.paused:
                for event in pygame.event.get():
                    running_now, reset = self._handle_event(event, screen, window_w)
                    running = running and running_now
                    if reset:
                        accumulator = 0.0
                if not running or not self.max_turbo or self.paused:
                    continue

                batch_start = time.perf_counter()
                steps = 0
                while (
                    running and alive and not self.rs.finished and
                    self.max_turbo and not self.paused
                ):
                    alive = self.rs.step(PHYSICS_DT_S)
                    steps += 1

                    # Poll input periodically even when a single episode can be
                    # simulated much faster than real time.
                    if steps % MAX_TURBO_EVENT_POLL_STEPS == 0:
                        for event in pygame.event.get():
                            running_now, reset = self._handle_event(
                                event, screen, window_w)
                            running = running and running_now
                            if reset:
                                accumulator = 0.0
                        if not running or not self.max_turbo or self.paused:
                            break

                    if time.perf_counter() - batch_start >= MAX_TURBO_BATCH_WALL_S:
                        break

                self._draw_max_turbo_status(screen, fonts)
                continue

            # Paused Max Turbo is intentionally throttled so it does not consume
            # a full CPU core while waiting for user input.
            if self.max_turbo and self.paused:
                clock.tick(30)
                for event in pygame.event.get():
                    running_now, reset = self._handle_event(event, screen, window_w)
                    running = running and running_now
                    if reset:
                        accumulator = 0.0
                if self.max_turbo:
                    self._draw_max_turbo_status(screen, fonts)
                continue

            real_dt = min(clock.tick(RENDER_FPS) / 1000.0, MAX_REAL_FRAME_DT_S)
            self._map_surface_age_s += real_dt
            for event in pygame.event.get():
                running_now, reset = self._handle_event(event, screen, window_w)
                running = running and running_now
                if reset:
                    accumulator = 0.0

            if not self.paused and running:
                accumulator = min(
                    accumulator + real_dt * self.time_scale,
                    MAX_ACCUMULATED_SIM_S,
                )
                steps = 0
                while (
                    accumulator >= PHYSICS_DT_S
                    and steps < MAX_PHYSICS_STEPS_PER_FRAME
                    and alive
                ):
                    alive = self.rs.step(PHYSICS_DT_S)
                    accumulator -= PHYSICS_DT_S
                    steps += 1

            alpha = (max(0.0, min(1.0, accumulator / PHYSICS_DT_S))
                     if PHYSICS_DT_S > 0 else 1.0)
            render_pose = self.rs.interpolated_pose(alpha)
            if self._camera3d.active:
                self._camera3d_last_send_s += real_dt
                if self._camera3d_last_send_s >= 1.0 / 30.0:
                    self._camera3d.update_pose(
                        self.rs.robot.floor,
                        render_pose[0], render_pose[1], render_pose[2],
                    )
                    self._camera3d_last_send_s = 0.0
            self._draw_frame(
                screen,
                left_rect,
                right_rect,
                legend_rect,
                fonts,
                alpha,
            )
            pygame.display.flip()

        self._camera3d.close()
        return self.time_scale


# =========================================================================
# Results tables and aggregate statistics
# =========================================================================
def summarize_results_by_condition(results):
    """Aggregate A_f, A_f* and A_tot for every (Fw, Opt) condition.

    Means are arithmetic means over all random buildings belonging to the same
    condition.  Standard deviations are sample standard deviations (N-1); for
    a condition containing one episode the reported standard deviation is 0.
    """
    groups = {}
    for result in results:
        key = (float(result["Fw"]), bool(result["Opt"]))
        groups.setdefault(key, []).append(result)

    summary = []
    for (fw, opt), group in sorted(
            groups.items(), key=lambda item: (item[0][0], item[0][1])):
        row = {"Fw": fw, "Opt": opt, "N": len(group)}
        for metric, prefix in (
            ("Af", "Af"),
            ("Af_star", "Af_star"),
            ("Atot", "Atot"),
        ):
            values = np.asarray(
                [float(item[metric]) for item in group],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                mean = float("nan")
                std = float("nan")
            else:
                mean = float(np.mean(values))
                std = (float(np.std(values, ddof=1))
                       if values.size > 1 else 0.0)
            row[f"{prefix}_mean"] = mean
            row[f"{prefix}_std"] = std
        summary.append(row)
    return summary


def _fixed_experiment_summary(results):
    if not results:
        return ""
    first = results[0]
    layout_summary = (
        tr("office") if first.get("layout_mode") == LAYOUT_OFFICE
        else tr("free")
    )
    return (
        f"{tr('environment')}: {layout_summary}; {tr('objects')}="
        f"{yes_no(bool(first.get('include_objects')))}, "
        f"{tr('object_density_short')}={first.get('object_density', 1.0):g}x | "
        f"{tr('fixed_weights_label')}: Iw={first.get('Iw', 0):g}, "
        f"Dw={first.get('Dw', 0):g}, Vw={first.get('Vw', 0):g}, "
        f"Cw={first.get('Cw', 0):g}, Ow={first.get('Ow', 0):g}, "
        f"Wp={first.get('Wp', 0):g}, Ms={first.get('Ms', 0):g}"
    )


def _style_table(table, header_color="#4a6fa5", font_size=8,
                 vertical_scale=1.35):
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, vertical_scale)
    column_count = max(column for row, column in table.get_celld().keys()) + 1
    for column in range(column_count):
        table[0, column].set_facecolor(header_color)
        table[0, column].set_text_props(color="white", fontweight="bold")


def _relationship_model_prediction(model, x_values):
    """Evaluate one fitted time-area model on an array of times."""
    x_values = np.asarray(x_values, dtype=float)
    intercept = float(model.get("intercept", math.nan))
    coefficient_1 = float(model.get("coefficient_1", math.nan))
    name = model.get("model")
    if name == "linear":
        return intercept + coefficient_1 * x_values
    if name == "quadratic":
        coefficient_2 = float(model.get("coefficient_2", math.nan))
        return intercept + coefficient_1 * x_values + coefficient_2 * x_values ** 2
    if name == "logarithmic":
        return intercept + coefficient_1 * np.log1p(np.maximum(x_values, 0.0))
    if name == "square_root":
        return intercept + coefficient_1 * np.sqrt(np.maximum(x_values, 0.0))
    if name == "saturating_exponential":
        tau = float(model.get("tau", math.nan))
        if not math.isfinite(tau) or tau <= 0.0:
            return np.full_like(x_values, np.nan)
        return intercept + coefficient_1 * (1.0 - np.exp(-x_values / tau))
    return np.full_like(x_values, np.nan)


def _show_time_area_relationship(results, analysis):
    """Open a statistical scatter plot for effective time versus total area."""
    if not analysis:
        return
    summaries = analysis.get("time_area_relationship", [])
    models = analysis.get("time_area_models", [])
    overall = next(
        (row for row in summaries if row.get("scope") == "Complessiva"), None)
    if not overall or overall.get("relationship") == "non valutabile":
        return

    points = []
    for result in results:
        try:
            x_value = float(result["texpl"])
            y_value = float(result["Atot"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x_value) and math.isfinite(y_value):
            points.append((x_value, y_value, bool(result.get("Opt"))))
    if len(points) < 2:
        return

    selected = next(
        (
            row for row in models
            if row.get("scope") == "Complessiva"
            and row.get("model") == overall.get("best_model")
        ),
        None,
    )
    linear = next(
        (
            row for row in models
            if row.get("scope") == "Complessiva"
            and row.get("model") == "linear"
        ),
        None,
    )
    if selected is None or linear is None:
        return

    figure, axis = plt.subplots(figsize=(11.8, 7.4))
    try:
        figure.canvas.manager.set_window_title(
            tr("relationship_window")
        )
    except Exception:
        pass

    off_x = [x for x, _y, opt in points if not opt]
    off_y = [y for _x, y, opt in points if not opt]
    on_x = [x for x, _y, opt in points if opt]
    on_y = [y for _x, y, opt in points if opt]
    if off_x:
        axis.scatter(off_x, off_y, alpha=0.75, label="Opt=OFF")
    if on_x:
        axis.scatter(on_x, on_y, alpha=0.75, marker="s", label="Opt=ON")

    all_x = np.asarray([point[0] for point in points], dtype=float)
    x_grid = np.linspace(float(np.min(all_x)), float(np.max(all_x)), 300)
    linear_y = _relationship_model_prediction(linear, x_grid)
    axis.plot(
        x_grid, linear_y, linestyle="--",
        label=f"{tr('linear')} (R²={overall['linear_r2']:.3f})",
    )
    if selected.get("model") != "linear":
        selected_y = _relationship_model_prediction(selected, x_grid)
        axis.plot(
            x_grid, selected_y, linewidth=2.2,
            label=(f"{tr('selected')}: {localize_analysis_text(overall['best_model_label'])} "
                   f"(R²={overall['best_model_r2']:.3f})"),
        )

    axis.set_xlabel(tr("effective_time_axis"))
    axis.set_ylabel(tr("total_area_axis"))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    axis.set_title(
        tr(
            "relationship_title",
            relationship=localize_analysis_text(overall['relationship']),
            strength=localize_analysis_text(overall['relationship_strength']),
            evidence=localize_analysis_text(overall['nonlinearity_evidence']),
        ),
        fontweight="bold",
    )
    detail = (
        f"N={overall['N']} | Pearson r={overall['pearson_r']:.3f}, "
        f"p={overall['pearson_p']:.3g}\n"
        f"{tr('slope')}={overall['linear_slope']:.4f} percentage points/s, "
        f"p={overall['linear_slope_p']:.3g}\n"
        f"{tr('delta_aicc')}={overall['delta_aicc_vs_linear']:.3f}"
    )
    axis.text(
        0.02, 0.98, detail, transform=axis.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.86},
    )
    figure.tight_layout()


def show_results_table(results, condition_summary=None, analysis=None):
    """Display aggregate statistics and the unchanged per-episode results.

    The aggregate window is intentionally separate and compact.  Detailed
    episode rows are paginated so a large batch no longer creates a single
    impossibly tall Matplotlib figure.
    """
    if condition_summary is None:
        condition_summary = summarize_results_by_condition(results)

    def fmt_number(value, digits=2):
        if isinstance(value, (float, np.floating)) and math.isnan(value):
            return "-"
        return f"{float(value):.{digits}f}"

    # --------------------------- compact condition-level summary
    summary_columns = [
        "Fw", "Opt", "N",
        tr("mean_af"), tr("sd_af"),
        tr("mean_afstar"), tr("sd_afstar"),
        tr("mean_atot"), tr("sd_atot"),
    ]
    summary_rows = []
    for item in condition_summary:
        summary_rows.append([
            f"{item['Fw']:g}",
            "ON" if item["Opt"] else "OFF",
            item["N"],
            fmt_number(item["Af_mean"]),
            fmt_number(item["Af_std"]),
            fmt_number(item["Af_star_mean"]),
            fmt_number(item["Af_star_std"]),
            fmt_number(item["Atot_mean"]),
            fmt_number(item["Atot_std"]),
        ])

    summary_height = max(4.8, 1.8 + 0.48 * max(1, len(summary_rows)))
    summary_fig, summary_axis = plt.subplots(figsize=(13.8, summary_height))
    summary_axis.axis("off")
    try:
        summary_fig.canvas.manager.set_window_title(
            tr("condition_statistics")
        )
    except Exception:
        pass
    summary_table = summary_axis.table(
        cellText=summary_rows,
        colLabels=summary_columns,
        loc="center",
        cellLoc="center",
    )
    _style_table(summary_table, font_size=9, vertical_scale=1.55)
    summary_axis.set_title(
        tr(
            "condition_summary_title",
            summary=_fixed_experiment_summary(results),
        ),
        fontsize=11.5,
        fontweight="bold",
        pad=25,
    )
    summary_fig.tight_layout()

    # Statistical relationship requested for the paper analysis.  The graph
    # is separate from the tables so points and fitted curves remain readable.
    _show_time_area_relationship(results, analysis)

    # --------------------------- detailed rows, preserved but paginated
    columns = [
        "Run", "Fw", "Opt", tr("layout"), tr("objects_column"), tr("density_column"),
        "Af (%)", "Af* (%)", "Atot (%)", "Vf", "Cf", "t_expl (s)",
        "TP", "FP", "FN", "TN", "SAR-Sens", "SAR-Spec",
        "SAR-BalAcc", "SAR-MCC",
    ]

    def fmt_metric(value):
        return (
            "-"
            if isinstance(value, float) and math.isnan(value)
            else f"{value:.3f}"
        )

    all_rows = []
    for result in results:
        all_rows.append([
            result["run"],
            f"{result['Fw']:g}",
            "ON" if result["Opt"] else "OFF",
            tr("office") if result.get("layout_mode") == LAYOUT_OFFICE else tr("free"),
            yes_no(bool(result.get("include_objects"))),
            f"{result.get('object_density', 1.0):g}x",
            f"{result['Af']:.1f}",
            f"{result['Af_star']:.1f}",
            f"{result['Atot']:.1f}",
            result["Vf"],
            result["Cf"],
            f"{result['texpl']:.0f}",
            result["TP"],
            result["FP"],
            result["FN"],
            f"{result['TN']:.1f}",
            fmt_metric(result["SAR_Sensitivity"]),
            fmt_metric(result["SAR_Specificity"]),
            fmt_metric(result["SAR_BalancedAccuracy"]),
            fmt_metric(result["SAR_MCC"]),
        ])

    page_size = 36
    page_count = max(1, int(math.ceil(len(all_rows) / page_size)))
    if not all_rows:
        all_rows = [["-" for _ in columns]]

    for page_index in range(page_count):
        page_rows = all_rows[
            page_index * page_size:(page_index + 1) * page_size
        ]
        figure_height = max(6.2, 2.4 + 0.31 * len(page_rows))
        fig, axis = plt.subplots(figsize=(17.8, figure_height))
        axis.axis("off")
        try:
            fig.canvas.manager.set_window_title(
                tr("episode_results_window", page=page_index + 1, pages=page_count)
            )
        except Exception:
            pass
        table = axis.table(
            cellText=page_rows,
            colLabels=columns,
            loc="center",
            cellLoc="center",
        )
        _style_table(table, font_size=7.4, vertical_scale=1.28)
        axis.set_title(
            tr(
                "episode_results_title",
                page=page_index + 1,
                pages=page_count,
                summary=_fixed_experiment_summary(results),
            ),
            fontsize=10.5,
            fontweight="bold",
            pad=22,
        )
        fig.tight_layout()

    plt.show()

