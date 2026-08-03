"""Continuous-motion multi-floor exploration episode and paper metrics.

The simulation uses a fixed-step continuous unicycle model. Sensor updates and
frontier planning use independent lower-frequency clocks. Rendering is handled
separately by the GUI, which can interpolate between consecutive physical
states without changing the occupancy-grid model used by the paper.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# CODE-REVIEW NOTES
# Purpose: One complete episode: multi-rate sensing, planning, control, stair transitions, recovery, and metrics.
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
import random
import numpy as np
from scipy import ndimage

import building
from building import generate_building, CELL_SIZE, FLOOR_W, FLOOR_H
from geometry import (cell_center_world, normalize_angle,
                      path_segment_is_traversable, point_in_region_world,
                      region_center_world)
from robot import (Robot, FloorMap, lidar_scan, camera_scan, semantic_decay,
                   FREE, SEMANTIC_THRESHOLD)
import planner
from planner import (detect_standard_frontiers, detect_stair_frontiers,
                     score_frontiers, best_frontier,
                     continuous_velocity_command)

# Fixed physical integration step. The live viewer renders at 60 FPS and uses
# an accumulator, so physics and graphics are no longer coupled one-to-one.
PHYSICS_DT_S = 0.01
CONTROL_DT_S = PHYSICS_DT_S  # compatibility with older callers
PLANNER_UPDATE_PERIOD_S = 0.75
LIDAR_UPDATE_PERIOD_S = 0.10
CAMERA_UPDATE_PERIOD_S = 0.50
SEMANTIC_DECAY_PERIOD_S = 2.0
STAIR_COOLDOWN_S = 15.0
# Initial spawn must not be near a stair frontier. The selected free cell is
# at least this far from every stair footprint on the starting floor.
START_MIN_STAIR_DISTANCE_M = 4.0
START_MIN_WALL_CLEARANCE_M = 1.0

# Velocity commands are filtered through acceleration limits. This prevents
# instantaneous starts, stops and rotations while retaining unicycle dynamics.
MAX_LINEAR_ACCEL_M_S2 = 0.90
MAX_ANGULAR_ACCEL_RAD_S2 = 4.00

# Soft selection hysteresis. This is not an absolute goal lock: another
# frontier can replace the current one immediately if its raw reward is large
# enough to overcome the continuity bonus. The default is exposed in the
# initial dialog as Wp (persistence weight).
ACTIVE_TARGET_HYSTERESIS_WEIGHT_DEFAULT = 24.0
TARGET_SWITCH_MARGIN_DEFAULT = 1.0
# Backward-compatible aliases for older scripts.
ACTIVE_TARGET_HYSTERESIS_WEIGHT = ACTIVE_TARGET_HYSTERESIS_WEIGHT_DEFAULT
TARGET_SWITCH_MARGIN = TARGET_SWITCH_MARGIN_DEFAULT
TARGET_MATCH_SIGMA_M = 2.25
TARGET_MATCH_MAX_DISTANCE_M = 5.0
TARGET_MIN_CONTINUITY = 0.12
TARGET_REFERENCE_SMOOTHING = 0.35
ARRIVAL_THRESHOLD_M = 1.5 * CELL_SIZE

# Recovery if no meaningful translational progress is made over a time window.
STUCK_PROGRESS_M = 0.04
STUCK_TRIGGER_S = 2.0
RECOVERY_ROTATION_S = 1.2
# If every detected frontier is temporarily rejected, the robot performs an
# active 360-degree scan instead of silently commanding zero velocity.  The
# shorter planner period makes the map/frontier set recover promptly.
PLANNER_STARVATION_REPLAN_S = 0.20
PLANNER_STARVATION_TURN_RAD_S = 0.55
RELAXED_PATH_CLEARANCE_CELLS = 0.75
VALID_PATH_GRACE_REFRESHES = 2

# SAR metrics.
VISIBILITY_RADIUS_M = 1.5
ELEMENTARY_AREA_M2 = (3 * CELL_SIZE) ** 2


def _approach(current: float, target: float, max_delta: float) -> float:
    """Move a scalar toward a target without exceeding ``max_delta``."""
    if target > current:
        return min(target, current + max_delta)
    return max(target, current - max_delta)


class RunConfig:
    """Configuration of one episode.

    Only ``Fw`` and ``Opt`` vary combinatorially.  All other planner weights
    are scalar settings chosen once in the initial dialog and reused for every
    episode of the experiment.
    """

    def __init__(
        self,
        n_floors,
        Tr,
        Ts,
        Fw,
        Opt,
        seed,
        Iw=planner.IW_DEFAULT,
        Dw=planner.DW_DEFAULT,
        Vw=planner.VW_DEFAULT,
        Cw=planner.CW_DEFAULT,
        Ow=planner.OW_DEFAULT,
        persistence_weight=ACTIVE_TARGET_HYSTERESIS_WEIGHT_DEFAULT,
        target_switch_margin=TARGET_SWITCH_MARGIN_DEFAULT,
        layout_mode=building.LAYOUT_OFFICE,
        include_objects=True,
        object_density=1.0,
        start_attempt=0,
    ):
        self.n_floors = n_floors
        self.Tr = Tr
        self.Ts = Ts
        self.Fw = Fw
        self.Opt = Opt
        self.seed = seed
        # Fixed for the whole batch. Only Fw and Opt are combinatorial.
        self.Iw = float(Iw)
        self.Dw = float(Dw)
        self.Vw = float(Vw)
        self.Cw = float(Cw)
        self.Ow = float(Ow)
        self.persistence_weight = float(persistence_weight)
        self.target_switch_margin = float(target_switch_margin)
        # Fixed environment options; they do not add experimental combinations.
        self.layout_mode = building.normalize_layout_mode(layout_mode)
        self.include_objects = bool(include_objects)
        self.object_density = building.normalize_object_density(
            object_density
        )
        self.start_attempt = int(start_attempt)
        if self.start_attempt < 0:
            raise ValueError("start_attempt must be non-negative")


class RunState:
    """Complete state of one episode, read directly by the GUI."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.floors = generate_building(
            cfg.n_floors,
            seed=cfg.seed,
            layout_mode=cfg.layout_mode,
            include_objects=cfg.include_objects,
            object_density=cfg.object_density,
        )
        for floor in self.floors:
            floor.fmap = FloorMap()

        # The building seed stays fixed for all Fw x Opt combinations of an
        # environment.  start_attempt changes only the initial pose selector,
        # allowing main.py to reject a blocked start and retry the same
        # building from a different random free position.
        rng = random.Random(cfg.seed * 7919 + 13 + cfg.start_attempt * 104729)
        start_options = []
        for candidate_floor in self.floors:
            positions = self._safe_start_positions(candidate_floor)
            if positions:
                start_options.append((candidate_floor.index, positions))
        if not start_options:
            raise RuntimeError(
                "No valid start position satisfies stair and wall clearance"
            )
        start_floor, positions = rng.choice(start_options)
        sx, sy = rng.choice(positions)
        self.robot = Robot(start_floor, sx, sy,
                           theta=rng.uniform(-math.pi, math.pi))
        self.initial_stair_distance_m = self._position_stair_distance_m(
            self.floors[start_floor], sx, sy
        )
        self.previous_pose = (self.robot.floor, self.robot.x,
                              self.robot.y, self.robot.theta)
        self.start_floor = start_floor

        self.remaining_time = float(cfg.Tr)
        self.elapsed_ticks = 0
        self.elapsed_time = 0.0
        self.visited_floors = {start_floor}
        self.floor_changes = 0
        self.stair_cooldown = 0.0
        self.detection_rng = random.Random(cfg.seed * 104729 + 1)
        self.lidar_rng = random.Random(cfg.seed * 130363 + 5)
        self.move_rng = random.Random(cfg.seed * 65537 + 3)

        self.current_frontiers = []
        self.chosen_frontier = None
        self.finished = False
        self.last_laser_scan = []
        self.last_scan_pose = (start_floor, self.robot.x, self.robot.y)
        self.last_desired_command = (0.0, 0.0)
        self.last_command = (0.0, 0.0)
        self.pose_history = [(self.robot.floor, self.robot.x, self.robot.y)]

        # Persistent frontier target.  We remember the selected curve and
        # goal cell because a frontier component is rebuilt from scratch at
        # every planner update and its centroid can move substantially.
        self.active_target_xy = None
        self.active_target_goal_cell = None
        self.active_target_kind = None
        self.active_target_floor = None
        self.active_target_stair_id = None
        self.active_target_segments = []
        self.seconds_since_switch = 999.0
        self.target_switch_count = 0
        self.last_target_continuity = 0.0
        self.last_target_hysteresis_bonus = 0.0
        self.last_target_switch_reason = "initial"

        # Continuous local-controller recovery state.
        self.stuck_seconds = 0.0
        self.recovery_remaining = 0.0
        self.progress_anchor = (self.robot.x, self.robot.y)
        self.progress_window_elapsed = 0.0

        # Planner-starvation diagnostics and recovery.  These counters make a
        # transient empty frontier set observable and prevent it from freezing
        # the robot with no displayed path.
        self.planner_starved = False
        self.planner_starvation_reason = ""
        self.planner_starvation_refreshes = 0
        self.relaxed_frontier_recovery_used = False

        # Multi-rate simulation clocks. Negative/zero values force an update
        # at the first rendered frame.
        self._lidar_due = 0.0
        self._camera_due = 0.0
        self._planner_due = 0.0
        self._decay_due = SEMANTIC_DECAY_PERIOD_S

    @staticmethod
    def _point_region_distance_m(x_m, y_m, region):
        if region is None:
            return math.inf
        x0, y0, x1, y1 = region
        left = x0 * CELL_SIZE
        right = (x1 + 1) * CELL_SIZE
        top = y0 * CELL_SIZE
        bottom = (y1 + 1) * CELL_SIZE
        dx = max(left - x_m, 0.0, x_m - right)
        dy = max(top - y_m, 0.0, y_m - bottom)
        return math.hypot(dx, dy)

    @classmethod
    def _position_stair_distance_m(cls, floor, x_m, y_m):
        stairs = list(getattr(floor, 'stair_up_regions', []) or [])
        stairs += list(getattr(floor, 'stair_down_regions', []) or [])
        if not stairs:
            stairs = [region for region in
                      (floor.stair_up_region, floor.stair_down_region)
                      if region is not None]
        if not stairs:
            return math.inf
        return min(cls._point_region_distance_m(x_m, y_m, region)
                   for region in stairs)

    @classmethod
    def _safe_start_positions(cls, floor):
        """FREE cell centres sufficiently far from stairs and all obstacles."""
        blocking = floor.grid == building.WALL
        padded = np.pad(~blocking, 1, mode='constant', constant_values=False)
        wall_clearance_m = (
            ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * CELL_SIZE
        )

        positions = []
        ys, xs = np.where(floor.grid == building.FREE)
        for x_cell, y_cell in zip(xs.tolist(), ys.tolist()):
            if wall_clearance_m[y_cell, x_cell] < START_MIN_WALL_CLEARANCE_M:
                continue
            x_m, y_m = cell_center_world(x_cell, y_cell)
            if (cls._position_stair_distance_m(floor, x_m, y_m) <
                    START_MIN_STAIR_DISTANCE_M):
                continue
            positions.append((x_m, y_m))
        return positions

    @classmethod
    def _random_free_position(cls, floor, rng):
        """Backward-compatible safe start selector used by older scripts."""
        positions = cls._safe_start_positions(floor)
        if not positions:
            raise RuntimeError(
                "No FREE start satisfies the requested stair/wall clearance"
            )
        return rng.choice(positions)

    def step(self, dt: float = CONTROL_DT_S):
        """Advance the simulation by one small continuous control interval.

        Sensing and frontier planning retain their own lower update rates,
        while the unicycle pose and the visible countdown are updated every
        ``dt`` seconds.
        """
        if self.finished or self.remaining_time <= 0.0:
            self.finished = True
            return False

        dt = max(1e-4, min(float(dt), 0.25))
        floor = self.floors[self.robot.floor]
        fmap = floor.fmap

        # 1) Multi-rate sensing.
        if self._lidar_due <= 0.0 or not self.last_laser_scan:
            self.last_laser_scan = lidar_scan(
                self.robot, floor, fmap, self.lidar_rng)
            self.last_scan_pose = (
                self.robot.floor, self.robot.x, self.robot.y)
            self._lidar_due += LIDAR_UPDATE_PERIOD_S

        if self._camera_due <= 0.0:
            camera_scan(self.robot, floor, fmap, self.detection_rng)
            self._camera_due += CAMERA_UPDATE_PERIOD_S

        if self._decay_due <= 0.0:
            semantic_decay(fmap, self.robot, floor)
            self._decay_due += SEMANTIC_DECAY_PERIOD_S

        # 2) Frontier geometry, paper reward and A* path are refreshed at a
        # lower frequency. Between refreshes the continuous controller follows
        # the currently stored metric polyline.
        if self._planner_due <= 0.0:
            self._refresh_frontiers_and_plan(floor, fmap)
            self._planner_due += PLANNER_UPDATE_PERIOD_S

        # 3) One fixed-step continuous unicycle integration.
        chosen = self.chosen_frontier
        old_floor = self.robot.floor
        old_x, old_y, old_theta = (
            self.robot.x, self.robot.y, self.robot.theta)
        self.previous_pose = (old_floor, old_x, old_y, old_theta)
        floor_changed = False

        if chosen is None or not chosen.path or len(chosen.path) < 2:
            # Do not remain motionless when the planner has no usable path.
            # Rotating in place is safe, exposes new surfaces to the 180-degree
            # LiDAR, and usually recreates a valid geometric frontier within a
            # few sensing cycles.  The red A* path reappears as soon as a
            # reachable candidate is found.
            desired_v = 0.0
            desired_omega = (PLANNER_STARVATION_TURN_RAD_S
                             if self.planner_starved else 0.0)
        else:
            desired_v, desired_omega = continuous_velocity_command(
                self.robot, chosen.path, self.last_laser_scan,
                rotate_only=self.recovery_remaining > 0.0)

        self.last_desired_command = (desired_v, desired_omega)
        v = _approach(
            self.robot.linear_velocity, desired_v,
            MAX_LINEAR_ACCEL_M_S2 * dt)
        omega = _approach(
            self.robot.angular_velocity, desired_omega,
            MAX_ANGULAR_ACCEL_RAD_S2 * dt)
        self.last_command = (v, omega)
        self.robot.linear_velocity = v
        self.robot.angular_velocity = omega

        # Exact integration of a constant-velocity unicycle over one short
        # interval. For nearly zero angular velocity it reduces to a straight
        # line; otherwise it follows the corresponding circular arc.
        new_theta = normalize_angle(old_theta + omega * dt)
        if abs(omega) < 1e-8:
            new_x = old_x + v * math.cos(old_theta) * dt
            new_y = old_y + v * math.sin(old_theta) * dt
        else:
            radius = v / omega
            new_x = old_x + radius * (math.sin(new_theta) - math.sin(old_theta))
            new_y = old_y - radius * (math.cos(new_theta) - math.cos(old_theta))

        if path_segment_is_traversable(
                floor.grid, old_x, old_y, new_x, new_y):
            self.robot.x = new_x
            self.robot.y = new_y
            self.robot.theta = new_theta
        else:
            # The point robot may rotate but never cross a wall.
            self.robot.theta = new_theta
            self.robot.linear_velocity = 0.0
            self.last_command = (0.0, omega)
            self.recovery_remaining = max(
                self.recovery_remaining, RECOVERY_ROTATION_S)

        if self.recovery_remaining > 0.0:
            self.recovery_remaining = max(0.0, self.recovery_remaining - dt)

        if self._check_stair_transition(floor):
            floor_changed = True

        # Store a continuous trajectory for visual verification.
        last_floor, last_x, last_y = self.pose_history[-1]
        if (last_floor != self.robot.floor or
                math.hypot(self.robot.x - last_x, self.robot.y - last_y) >= 0.02):
            self.pose_history.append((self.robot.floor, self.robot.x, self.robot.y))
            if len(self.pose_history) > 6000:
                self.pose_history = self.pose_history[-6000:]

        # Progress is checked over a time window rather than per rendered
        # frame, otherwise a slow continuous robot would be falsely considered
        # stuck at every 50 ms step.
        self.progress_window_elapsed += dt
        if self.progress_window_elapsed >= STUCK_TRIGGER_S:
            px, py = self.progress_anchor
            progress = math.hypot(self.robot.x - px, self.robot.y - py)
            if chosen is not None and progress < STUCK_PROGRESS_M and not floor_changed:
                self.recovery_remaining = max(
                    self.recovery_remaining, RECOVERY_ROTATION_S)
            self.progress_anchor = (self.robot.x, self.robot.y)
            self.progress_window_elapsed = 0.0

        self.stair_cooldown = max(0.0, self.stair_cooldown - dt)
        self.seconds_since_switch += dt

        # 4) Continuous time budget. Stair costs are subtracted immediately by
        # _check_stair_transition.
        self.elapsed_time += dt
        self.elapsed_ticks = int(self.elapsed_time)
        self.remaining_time -= dt

        self._lidar_due -= dt
        self._camera_due -= dt
        self._planner_due -= dt
        self._decay_due -= dt

        if self.remaining_time <= 0.0:
            self.finished = True
            return False
        return True

    def interpolated_pose(self, alpha: float = 1.0):
        """Return a render pose interpolated between two physical states.

        The interpolation deliberately lags by at most one 10 ms physics step,
        which removes visible quantisation without altering the simulation.
        """
        alpha = max(0.0, min(1.0, float(alpha)))
        prev_floor, px, py, ptheta = self.previous_pose
        if prev_floor != self.robot.floor:
            return (self.robot.x, self.robot.y, self.robot.theta)
        dtheta = normalize_angle(self.robot.theta - ptheta)
        return (
            px + alpha * (self.robot.x - px),
            py + alpha * (self.robot.y - py),
            normalize_angle(ptheta + alpha * dtheta),
        )

    def _refresh_frontiers_and_plan(self, floor, fmap):
        """Refresh candidates and choose one with soft persistence.

        Eq. 2/Eq. 5 still provide the raw reward.  The only additional term is
        a continuity bonus for the frontier associated with the previous
        target.  The bonus is geometric and decays smoothly, so it prevents
        oscillation without imposing a hard "keep until reached" rule.
        """
        curve_weight = self.cfg.Cw if self.cfg.Opt else 0.0
        orientation_weight = self.cfg.Ow if self.cfg.Opt else 0.0

        # First pass: preserve the normal, conservative behaviour exactly.
        standard = detect_standard_frontiers(fmap)
        stairs = ([] if self.stair_cooldown > 0.0
                  else detect_stair_frontiers(fmap, floor))
        all_frontiers = standard + stairs
        score_frontiers(
            all_frontiers, self.robot, fmap, floor, self.floors,
            self.cfg.Fw, curve_weight, orientation_weight,
            Iw=self.cfg.Iw, Dw=self.cfg.Dw, Vw=self.cfg.Vw,
        )
        candidate = self._best_not_reached(all_frontiers)
        self.relaxed_frontier_recovery_used = False

        # Recovery pass: a fine frontier can disappear after wall-adjacency
        # filtering, or all its coarse approach cells can be rejected by the
        # conservative clearance band.  Only when the strict pass has no
        # reachable candidate do we reconsider wall-adjacent free frontier
        # pixels and use a still-safe but narrower A* clearance.  Occupied and
        # unknown cells remain non-traversable.
        if candidate is None:
            relaxed_standard = detect_standard_frontiers(fmap, relaxed=True)
            relaxed_frontiers = relaxed_standard + stairs
            score_frontiers(
                relaxed_frontiers, self.robot, fmap, floor, self.floors,
                self.cfg.Fw, curve_weight, orientation_weight,
                Iw=self.cfg.Iw, Dw=self.cfg.Dw, Vw=self.cfg.Vw,
                hard_clearance_cells=RELAXED_PATH_CLEARANCE_CELLS,
            )
            relaxed_candidate = self._best_not_reached(relaxed_frontiers)
            if relaxed_candidate is not None:
                all_frontiers = relaxed_frontiers
                candidate = relaxed_candidate
                self.relaxed_frontier_recovery_used = True

        chosen = self._select_target(all_frontiers, candidate)

        # A brief grace period preserves a previously valid path if one single
        # frontier refresh is empty because of sensor discretisation.  It is
        # deliberately short: after two refreshes the robot switches to active
        # scanning rather than pursuing a stale target indefinitely.
        previous = self.chosen_frontier
        if (chosen is None and previous is not None and previous.path
                and len(previous.path) >= 2
                and self.planner_starvation_refreshes < VALID_PATH_GRACE_REFRESHES
                and not self._frontier_reached(previous)):
            chosen = previous

        self.current_frontiers = all_frontiers
        self.chosen_frontier = chosen
        self.planner_starved = (chosen is None or not chosen.path
                                or len(chosen.path) < 2)
        if self.planner_starved:
            self.planner_starvation_refreshes += 1
            if not all_frontiers:
                self.planner_starvation_reason = "no_frontiers_detected"
            elif not any(f.path for f in all_frontiers):
                self.planner_starvation_reason = "frontiers_unreachable"
            else:
                self.planner_starvation_reason = "no_eligible_frontier"
            # Re-run the planner quickly while the robot rotates and scans.
            # ``step`` adds the normal planner period immediately after this
            # method returns.  Store an offset so the resulting countdown is
            # the shorter starvation period rather than 0.75 s.
            self._planner_due = (PLANNER_STARVATION_REPLAN_S
                                 - PLANNER_UPDATE_PERIOD_S)
        else:
            self.planner_starvation_refreshes = 0
            self.planner_starvation_reason = ""

    # ------------------------------------------------ target persistence
    @staticmethod
    def _distance_to_frontier_goal(robot, frontier):
        if frontier.path:
            target_x, target_y = frontier.path[-1]
        elif frontier.goal_cell is not None:
            target_x, target_y = cell_center_world(*frontier.goal_cell)
        else:
            target_x, target_y = frontier.x, frontier.y
        return math.hypot(target_x - robot.x, target_y - robot.y)

    def _frontier_reached(self, frontier):
        # A stair frontier is completed only by the explicit floor transition.
        # Its footprint is one metre wide, so the generic proximity threshold
        # could otherwise discard the stair target before the robot enters it.
        if frontier.kind == 'stair':
            return False
        return (self._distance_to_frontier_goal(self.robot, frontier)
                <= ARRIVAL_THRESHOLD_M)

    def _best_not_reached(self, frontiers):
        eligible = [frontier for frontier in frontiers
                    if frontier.score is not None
                    and frontier.score > -math.inf
                    and not self._frontier_reached(frontier)]
        return best_frontier(eligible)

    def _clear_active_target(self, reason):
        self.active_target_xy = None
        self.active_target_goal_cell = None
        self.active_target_kind = None
        self.active_target_floor = None
        self.active_target_stair_id = None
        self.active_target_segments = []
        self.last_target_continuity = 0.0
        self.last_target_hysteresis_bonus = 0.0
        self.last_target_switch_reason = reason

    def _continuity_with_active_target(self, frontier):
        if self.active_target_xy is None:
            return 0.0
        if frontier.kind != self.active_target_kind:
            return 0.0
        if frontier.target_floor != self.active_target_floor:
            return 0.0
        if getattr(frontier, 'stair_id', None) != self.active_target_stair_id:
            return 0.0

        if frontier.kind == 'stair':
            return 1.0

        anchor_x, anchor_y = self.active_target_xy
        curve_distance = planner.frontier_distance_to_point(
            frontier, anchor_x, anchor_y)
        overlap = planner.frontier_segment_overlap(
            frontier, self.active_target_segments)

        goal_distance = math.inf
        if (self.active_target_goal_cell is not None
                and frontier.goal_cell is not None):
            old_x, old_y = cell_center_world(*self.active_target_goal_cell)
            new_x, new_y = cell_center_world(*frontier.goal_cell)
            goal_distance = math.hypot(new_x - old_x, new_y - old_y)

        match_distance = min(curve_distance, goal_distance)
        if match_distance > TARGET_MATCH_MAX_DISTANCE_M and overlap <= 0.0:
            return 0.0

        distance_gain = math.exp(
            -0.5 * (match_distance / TARGET_MATCH_SIGMA_M) ** 2)

        old_bearing = math.atan2(anchor_y - self.robot.y,
                                 anchor_x - self.robot.x)
        new_bearing = math.atan2(frontier.y - self.robot.y,
                                 frontier.x - self.robot.x)
        direction_gain = max(
            0.0,
            math.cos(normalize_angle(new_bearing - old_bearing)),
        )

        # Exact segment overlap dominates. If the frontier advances and exact
        # overlap disappears, nearby geometry in the same direction retains a
        # progressively smaller association.
        return max(
            overlap,
            distance_gain * (0.40 + 0.60 * direction_gain),
        )

    def _match_active_frontier(self, frontiers):
        best_match = None
        best_continuity = 0.0
        for frontier in frontiers:
            if frontier.score is None or frontier.score == -math.inf:
                continue
            continuity = self._continuity_with_active_target(frontier)
            if continuity > best_continuity:
                best_continuity = continuity
                best_match = frontier
        if best_continuity < TARGET_MIN_CONTINUITY:
            return None, 0.0
        return best_match, best_continuity

    def _select_target(self, all_frontiers, candidate):
        had_active = self.active_target_xy is not None
        active, continuity = self._match_active_frontier(all_frontiers)

        if active is not None and self._frontier_reached(active):
            self._clear_active_target("reached")
            had_active = False
            active = None
            continuity = 0.0
            candidate = self._best_not_reached(all_frontiers)

        bonus = (self.cfg.persistence_weight * continuity
                 if active is not None else 0.0)
        self.last_target_continuity = continuity
        self.last_target_hysteresis_bonus = bonus

        if active is None:
            chosen = candidate
            reason = "initial_or_lost"
        elif candidate is None or candidate is active:
            chosen = active
            reason = "keep"
        elif candidate.score > (active.score + bonus
                                + self.cfg.target_switch_margin):
            chosen = candidate
            reason = "superior_reward"
        else:
            chosen = active
            reason = "hysteresis_keep"

        if chosen is None:
            self._clear_active_target(reason)
            return None

        continuing = chosen is active and had_active
        if continuing:
            alpha = TARGET_REFERENCE_SMOOTHING
            old_x, old_y = self.active_target_xy
            self.active_target_xy = (
                old_x + alpha * (chosen.x - old_x),
                old_y + alpha * (chosen.y - old_y),
            )
        else:
            if had_active:
                self.target_switch_count += 1
            self.seconds_since_switch = 0.0
            self.active_target_xy = (chosen.x, chosen.y)
            self.active_target_kind = chosen.kind
            self.active_target_floor = chosen.target_floor
            self.active_target_stair_id = getattr(chosen, 'stair_id', None)

        self.active_target_goal_cell = chosen.goal_cell
        self.active_target_segments = list(chosen.segments)
        self.last_target_switch_reason = reason
        return chosen

    # ---------------------------------------------------------- stairs
    def _check_stair_transition(self, floor):
        """Change floor only while explicitly pursuing that stair frontier.

        Merely crossing a stair footprint is no longer sufficient.  This is
        important on middle floors, where a path toward a normal frontier can
        legitimately pass over either stair area without changing level.
        """
        if self.stair_cooldown > 0.0:
            return False

        chosen = self.chosen_frontier
        if chosen is None or chosen.kind != 'stair':
            return False

        target = chosen.target_floor
        if target is None or not 0 <= target < self.cfg.n_floors:
            return False

        # Only the footprint corresponding to the selected destination is
        # active.  Standing on the other staircase of an intermediate floor
        # must not trigger an unintended transition.
        stair_id = getattr(chosen, 'stair_id', None)
        if stair_id is None:
            stair_id = 0
        if target == self.robot.floor + 1:
            regions = getattr(floor, 'stair_up_regions', []) or []
        elif target == self.robot.floor - 1:
            regions = getattr(floor, 'stair_down_regions', []) or []
        else:
            return False
        if not regions:
            fallback = (floor.stair_up_region if target > self.robot.floor
                        else floor.stair_down_region)
            regions = [fallback] if fallback is not None else []
        if not 0 <= stair_id < len(regions):
            return False
        selected_region = regions[stair_id]

        if selected_region is None or not point_in_region_world(
                self.robot.x, self.robot.y, selected_region):
            return False

        old_floor_index = self.robot.floor
        self.robot.floor = target
        self.robot.linear_velocity = 0.0
        self.robot.angular_velocity = 0.0
        self.visited_floors.add(target)
        self.floor_changes += 1
        self.remaining_time -= self.cfg.Ts
        self.stair_cooldown = STAIR_COOLDOWN_S

        # Place the robot at the centre of the matching footprint on the new
        # floor.  This is the same stairwell geometry and is guaranteed free.
        destination_floor = self.floors[target]
        destination_regions = (
            getattr(destination_floor, 'stair_down_regions', [])
            if target > old_floor_index
            else getattr(destination_floor, 'stair_up_regions', [])
        ) or []
        if not destination_regions:
            fallback = (destination_floor.stair_down_region
                        if target > old_floor_index
                        else destination_floor.stair_up_region)
            destination_regions = [fallback] if fallback is not None else []
        destination_region = (destination_regions[stair_id]
                              if 0 <= stair_id < len(destination_regions)
                              else None)
        if destination_region is not None:
            cx, cy = region_center_world(destination_region)
            directions = (getattr(destination_floor, 'stair_core_directions', [])
                          or getattr(floor, 'stair_core_directions', []) or [])
            dx, dy = (directions[stair_id]
                      if 0 <= stair_id < len(directions) else (1, 0))
            # Spawn just beyond the selected stair footprint, in the exact
            # direction shown by its arrow. Candidate stair cores have a full
            # free-cell ring, so this exit pose is guaranteed to be clear.
            x0, y0, x1, y1 = destination_region
            half_w_m = 0.5 * (x1 - x0 + 1) * CELL_SIZE
            half_h_m = 0.5 * (y1 - y0 + 1) * CELL_SIZE
            offset = ((half_w_m if dx else half_h_m) + 0.60 * CELL_SIZE)
            self.robot.x = cx + dx * offset
            self.robot.y = cy + dy * offset
            self.robot.theta = math.atan2(dy, dx)

        self._clear_active_target("floor_change")
        self.seconds_since_switch = 999.0
        self.recovery_remaining = 0.0
        self.last_laser_scan = []
        self.last_scan_pose = (
            self.robot.floor, self.robot.x, self.robot.y)
        self.progress_anchor = (self.robot.x, self.robot.y)
        self.progress_window_elapsed = 0.0
        self._lidar_due = 0.0
        self._camera_due = 0.0
        self._planner_due = 0.0
        self.previous_pose = (self.robot.floor, self.robot.x,
                              self.robot.y, self.robot.theta)
        return True

    # ---------------------------------------------------------- metrics
    def compute_metrics(self):
        per_floor_pct = {}
        for floor in self.floors:
            explored = float(np.sum(floor.fmap.occ == FREE)) * CELL_SIZE ** 2
            per_floor_pct[floor.index] = (
                100.0 * explored / floor.explorable_area_m2)

        visited = sorted(self.visited_floors)
        Af = (float(np.mean([per_floor_pct[i] for i in visited]))
              if visited else 0.0)
        Af_star = per_floor_pct[self.start_floor]

        total_explorable = sum(f.explorable_area_m2 for f in self.floors)
        total_explored = sum(
            float(np.sum(f.fmap.occ == FREE)) * CELL_SIZE ** 2
            for f in self.floors)
        Atot = 100.0 * total_explored / total_explorable

        metrics = {
            'Af': Af,
            'Af_star': Af_star,
            'Atot': Atot,
            'Vf': len(self.visited_floors),
            'Cf': self.floor_changes,
            'texpl': self.elapsed_time,
            'per_floor_pct': per_floor_pct,
        }
        metrics.update(self._compute_sar_metrics())
        return metrics

    def _compute_sar_metrics(self):
        TP = FP = FN = 0
        TN = 0.0
        visibility_cells = VISIBILITY_RADIUS_M / CELL_SIZE

        for floor in self.floors:
            if floor.index not in self.visited_floors:
                continue
            semantic_mask = floor.fmap.semantic >= SEMANTIC_THRESHOLD

            matched = np.zeros_like(semantic_mask)
            floor_TP = 0
            for victim_x, victim_y in floor.victims:
                y0 = max(0, int(victim_y - visibility_cells))
                y1 = min(FLOOR_H, int(victim_y + visibility_cells) + 1)
                x0 = max(0, int(victim_x - visibility_cells))
                x1 = min(FLOOR_W, int(victim_x + visibility_cells) + 1)
                found = False
                ys, xs = np.where(semantic_mask[y0:y1, x0:x1])
                for local_y, local_x in zip(ys, xs):
                    global_y = y0 + local_y
                    global_x = x0 + local_x
                    if math.hypot(global_x - victim_x,
                                  global_y - victim_y) <= visibility_cells:
                        found = True
                        matched[global_y, global_x] = True
                if found:
                    floor_TP += 1
                    TP += 1
                else:
                    FN += 1
            floor_FN = len(floor.victims) - floor_TP

            labelled, count = ndimage.label(
                semantic_mask & (~matched), structure=np.ones((3, 3)))
            floor_FP = 0
            for label in range(1, count + 1):
                ys, xs = np.where(labelled == label)
                centre_x, centre_y = float(np.mean(xs)), float(np.mean(ys))
                if not any(math.hypot(centre_x - vx, centre_y - vy)
                           <= visibility_cells
                           for vx, vy in floor.victims):
                    floor_FP += 1
                    FP += 1

            explored_area = (float(np.sum(floor.fmap.occ == FREE))
                             * CELL_SIZE ** 2)
            units = explored_area / ELEMENTARY_AREA_M2
            TN += max(0.0, units - floor_TP - floor_FP - floor_FN)

        sensitivity = TP / (TP + FN) if TP + FN > 0 else float('nan')
        specificity = TN / (TN + FP) if TN + FP > 0 else float('nan')
        balanced = np.nanmean([sensitivity, specificity])
        denominator = math.sqrt(max(
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN), 1e-9))
        mcc = ((TP * TN - FP * FN) / denominator
               if denominator > 0 else float('nan'))

        return {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': round(TN, 1),
            'SAR_Sensitivity': sensitivity,
            'SAR_Specificity': specificity,
            'SAR_BalancedAccuracy': float(balanced),
            'SAR_MCC': mcc,
        }
