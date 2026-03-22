from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from shared_field_visuals import FieldPalette, RenderFlags, draw_field_matplotlib
from shared_sim_core import (
    ALLIANCE_ZONE_DEPTH_M,
    BALL_RADIUS_M,
    BALL_STATE_CONTACTED,
    BALL_STATE_CONTROLLED,
    BALL_STATE_FREE,
    BALL_STATE_INDEXED,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    PathSpec,
    RunResult,
    SimParams,
    Snapshot,
    Ball,
    apply_motion_losses,
    apply_surface_regions,
    build_field_obstacles,
    build_field_surface_regions,
    build_spatial_hash,
    carried_ball_pose,
    clamp,
    generate_trajectory,
    load_pathspec_json,
    make_initial_balls,
    neighbor_cells,
    norm,
    resolve_ball_ball,
    resolve_ball_field_obstacles,
    resolve_ball_robot,
    resolve_ball_wall,
    robot_body_collides,
    robot_body_in_region,
    robot_pose_from_trajectory,
    rotmat,
    save_params_json,
    update_controlled_ball,
)


ROBOT_COLORS = {
    "red": "tab:red",
    "blue": "tab:blue",
    "neutral": "tab:green",
}


@dataclass
class MultiRobotSpec:
    name: str
    alliance: str
    path: PathSpec
    priority: float = 1.0
    intake_enabled: bool = True
    color: str | None = None


@dataclass
class RobotStepState:
    robot_index: int
    spec: MultiRobotSpec
    traj_total_time: float
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    theta: float = 0.0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    target_theta: float = 0.0
    target_vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    pos_offset: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    indexed_count: int = 0
    first_intake_time: float | None = None
    robot_trace: list[np.ndarray] = field(default_factory=list)
    robot_theta_trace: list[float] = field(default_factory=list)
    collision_trace: list[np.ndarray] = field(default_factory=list)
    bump_trace: list[np.ndarray] = field(default_factory=list)
    trench_trace: list[np.ndarray] = field(default_factory=list)
    capture_trace: list[np.ndarray] = field(default_factory=list)
    contact_trace: list[np.ndarray] = field(default_factory=list)
    robot_robot_trace: list[np.ndarray] = field(default_factory=list)
    illegal_contact_trace: list[np.ndarray] = field(default_factory=list)


@dataclass
class MultiRobotSnapshot:
    t: float
    ball_positions: np.ndarray
    active_mask: np.ndarray
    indexed_mask: np.ndarray
    owner_index: np.ndarray
    robot_positions: np.ndarray
    robot_thetas: np.ndarray
    robot_names: list[str]
    indexed_counts: np.ndarray


@dataclass
class MultiRobotRunResult:
    robot_results: list[RunResult]
    final_positions: np.ndarray
    active_mask: np.ndarray
    indexed_mask: np.ndarray
    indexed_owner: np.ndarray
    owner_by_ball: np.ndarray
    snapshots: list[MultiRobotSnapshot]
    total_time: float
    robot_robot_collision_samples: int
    illegal_opponent_contact_samples: int
    robot_robot_collision_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    illegal_opponent_contact_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))


def normalize_alliance_name(alliance: str) -> str:
    out = alliance.strip().lower()
    if out not in {"red", "blue", "neutral"}:
        raise ValueError(f"Unsupported alliance '{alliance}'")
    return out


def _robot_half_extents(params: SimParams) -> np.ndarray:
    return np.array([params.robot_length / 2.0, params.robot_width / 2.0], dtype=float)


def _robot_corners(pos: np.ndarray, theta: float, params: SimParams) -> np.ndarray:
    hx, hy = _robot_half_extents(params)
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=float)
    return (local @ rotmat(theta).T) + pos


def _project_obb_radius(half_extents: np.ndarray, theta: float, axis: np.ndarray) -> float:
    axis = axis / max(1e-12, float(np.linalg.norm(axis)))
    R = rotmat(theta)
    a0 = R[:, 0]
    a1 = R[:, 1]
    return abs(np.dot(a0 * half_extents[0], axis)) + abs(np.dot(a1 * half_extents[1], axis))


def _obb_mtv(c1: np.ndarray, t1: float, c2: np.ndarray, t2: float, half_extents: np.ndarray) -> np.ndarray | None:
    axes = [rotmat(t1)[:, 0], rotmat(t1)[:, 1], rotmat(t2)[:, 0], rotmat(t2)[:, 1]]
    d = c2 - c1
    best_axis: np.ndarray | None = None
    best_overlap: float | None = None
    for axis in axes:
        axis = axis / max(1e-12, float(np.linalg.norm(axis)))
        r1 = _project_obb_radius(half_extents, t1, axis)
        r2 = _project_obb_radius(half_extents, t2, axis)
        center_sep = abs(float(np.dot(d, axis)))
        overlap = (r1 + r2) - center_sep
        if overlap <= 0.0:
            return None
        if best_overlap is None or overlap < best_overlap:
            best_overlap = overlap
            sign = 1.0 if float(np.dot(d, axis)) >= 0.0 else -1.0
            best_axis = sign * axis
    if best_axis is None or best_overlap is None:
        return None
    return best_axis * (best_overlap + 1e-6)


def _robot_fully_across_center_line(pos: np.ndarray, theta: float, params: SimParams, alliance: str) -> bool:
    corners = _robot_corners(pos, theta, params)
    if alliance == "red":
        return float(np.min(corners[:, 0])) > 0.0
    if alliance == "blue":
        return float(np.max(corners[:, 0])) < 0.0
    return False


def _clamp_robot_inside_field(pos: np.ndarray, params: SimParams) -> np.ndarray:
    radius = 0.5 * math.hypot(params.robot_length, params.robot_width)
    return np.array(
        [
            clamp(pos[0], -FIELD_LENGTH_M / 2.0 + radius, FIELD_LENGTH_M / 2.0 - radius),
            clamp(pos[1], -FIELD_WIDTH_M / 2.0 + radius, FIELD_WIDTH_M / 2.0 - radius),
        ],
        dtype=float,
    )


def _choose_contact_claim(ball: Ball, states: list[RobotStepState], available_slots: list[int], params: SimParams) -> tuple[int, int] | None:
    if ball.state != BALL_STATE_FREE:
        return None

    candidates: list[tuple[float, int, int]] = []
    for idx, state in enumerate(states):
        if available_slots[idx] <= 0 or not state.spec.intake_enabled:
            continue

        local = rotmat(state.theta).T @ (ball.pos - state.pos)
        front_x = params.robot_length / 2.0
        in_front = (front_x + BALL_RADIUS_M) <= local[0] <= (front_x + params.intake_depth - BALL_RADIUS_M)
        centered = abs(local[1]) <= (params.intake_half_width - BALL_RADIUS_M)
        if not (in_front and centered):
            continue

        front_normal = rotmat(state.theta) @ np.array([1.0, 0.0], dtype=float)
        side_normal = rotmat(state.theta) @ np.array([0.0, 1.0], dtype=float)
        rel_vel = ball.vel - state.vel
        rel_normal_speed = abs(float(np.dot(rel_vel, front_normal)))
        side_speed = abs(float(np.dot(rel_vel, side_normal)))
        side_allowance = params.intake_capture_speed * (1.0 - params.side_entry_rejection_factor)
        if rel_normal_speed > params.intake_capture_speed or side_speed > params.intake_capture_speed + side_allowance:
            continue

        slot_idx = max(0, params.max_intaked_balls - available_slots[idx])
        target = carried_ball_pose(state.pos, state.theta, params, slot_idx)
        score = norm(ball.pos - target)
        candidates.append((score, idx, slot_idx))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, winner_idx, slot_idx = candidates[0]
    return winner_idx, slot_idx


def _claim_ball(ball: Ball, slot_idx: int, owner_state: RobotStepState, params: SimParams) -> bool:
    ball.capture_pos = ball.pos.copy()
    ball.captured_slot = slot_idx
    if params.intake_settle_time <= 0.0:
        if params.intake_index_time <= 0.0:
            ball.state = BALL_STATE_INDEXED
            ball.timer = 0.0
            ball.vel[:] = 0.0
            return True
        ball.state = BALL_STATE_CONTROLLED
        ball.timer = params.intake_index_time
    else:
        ball.state = BALL_STATE_CONTACTED
        ball.timer = params.intake_settle_time

    ball.pos[:] = carried_ball_pose(owner_state.pos, owner_state.theta, params, slot_idx)
    ball.vel[:] = owner_state.vel
    return False


def _make_single_robot_result(
    state: RobotStepState,
    balls: list[Ball],
    owner_by_ball: np.ndarray,
    indexed_owner: np.ndarray,
    snapshots: list[Snapshot],
    total_time: float,
    seed: int,
) -> RunResult:
    robot_idx = state.robot_index
    final_positions = np.array([b.pos.copy() for b in balls], dtype=float)
    active_mask = np.array([b.active for b in balls], dtype=bool)
    indexed_mask = np.array([bool(b.state == BALL_STATE_INDEXED and indexed_owner[i] == robot_idx) for i, b in enumerate(balls)], dtype=bool)
    indexed_positions = np.array(
        [
            (b.capture_pos.copy() if (indexed_mask[i] and b.capture_pos is not None) else b.pos.copy())
            for i, b in enumerate(balls)
        ],
        dtype=float,
    )
    contacted_owned = np.array([bool(b.state == BALL_STATE_CONTACTED and owner_by_ball[i] == robot_idx) for i, b in enumerate(balls)], dtype=bool)
    controlled_owned = np.array([bool(b.state == BALL_STATE_CONTROLLED and owner_by_ball[i] == robot_idx) for i, b in enumerate(balls)], dtype=bool)

    return RunResult(
        final_positions=final_positions,
        active_mask=active_mask,
        indexed_mask=indexed_mask,
        indexed_positions=indexed_positions,
        snapshots=snapshots,
        captured_count=state.indexed_count,
        path_name=state.spec.name,
        seed=seed,
        total_time=total_time,
        collision_samples=len(state.collision_trace),
        bump_region_samples=len(state.bump_trace),
        trench_region_samples=len(state.trench_trace),
        first_intake_time=state.first_intake_time,
        robot_trace=np.array(state.robot_trace, dtype=float) if state.robot_trace else np.empty((0, 2), dtype=float),
        robot_theta_trace=np.array(state.robot_theta_trace, dtype=float) if state.robot_theta_trace else np.empty((0,), dtype=float),
        collision_trace=np.array(state.collision_trace, dtype=float) if state.collision_trace else np.empty((0, 2), dtype=float),
        bump_trace=np.array(state.bump_trace, dtype=float) if state.bump_trace else np.empty((0, 2), dtype=float),
        trench_trace=np.array(state.trench_trace, dtype=float) if state.trench_trace else np.empty((0, 2), dtype=float),
        capture_trace=np.array(state.capture_trace, dtype=float) if state.capture_trace else np.empty((0, 2), dtype=float),
        contact_trace=np.array(state.contact_trace, dtype=float) if state.contact_trace else np.empty((0, 2), dtype=float),
        contacted_count=int(np.sum(contacted_owned)),
        controlled_count=int(np.sum(controlled_owned)),
        free_count=int(np.sum(active_mask)),
    )


def simulate_multi_robot_auto(
    robot_specs: Iterable[MultiRobotSpec],
    params: SimParams,
    seed: int = 0,
    collect_snapshots: bool = False,
    opponent_contact_rule: str = "warn",
    offset_decay: float = 0.82,
    collision_iterations: int = 3,
    sim_duration_s: float | None = 20.0,
) -> MultiRobotRunResult:
    specs = [
        MultiRobotSpec(
            name=spec.name,
            alliance=normalize_alliance_name(spec.alliance),
            path=spec.path,
            priority=float(spec.priority),
            intake_enabled=bool(spec.intake_enabled),
            color=spec.color or ROBOT_COLORS.get(normalize_alliance_name(spec.alliance)),
        )
        for spec in robot_specs
    ]
    if not specs:
        raise ValueError("At least one robot spec is required.")

    balls = make_initial_balls(params, seed)
    owner_by_ball = np.full(len(balls), -1, dtype=int)
    indexed_owner = np.full(len(balls), -1, dtype=int)
    field_obstacles = build_field_obstacles(params)
    surface_regions = build_field_surface_regions(params)
    half_extents = _robot_half_extents(params)
    cell_size = 2.5 * (2.0 * BALL_RADIUS_M)

    trajectories = [generate_trajectory(spec.path) for spec in specs]
    max_traj_time = max(traj.total_time for traj in trajectories)
    total_time = max(max_traj_time + params.total_time_pad, sim_duration_s or 0.0)
    n_steps = int(math.ceil(total_time / params.physics_dt))

    states: list[RobotStepState] = [
        RobotStepState(robot_index=i, spec=spec, traj_total_time=traj.total_time)
        for i, (spec, traj) in enumerate(zip(specs, trajectories))
    ]

    snapshots: list[MultiRobotSnapshot] = []
    robot_robot_collision_points: list[np.ndarray] = []
    illegal_opponent_contact_points: list[np.ndarray] = []

    for step in range(n_steps + 1):
        t = min(total_time, step * params.physics_dt)
        prev_offsets = [state.pos_offset.copy() for state in states]

        for state, traj in zip(states, trajectories):
            drive_t = min(t, traj.total_time)
            target_pos, target_theta, target_vel = robot_pose_from_trajectory(traj, drive_t)
            state.target_pos = target_pos.copy()
            state.target_theta = float(target_theta)
            state.target_vel = target_vel.copy()
            state.pos = _clamp_robot_inside_field(target_pos + state.pos_offset, params)
            state.theta = float(target_theta)
            state.vel = target_vel.copy()

        for _ in range(max(1, collision_iterations)):
            any_overlap = False
            for i in range(len(states)):
                for j in range(i + 1, len(states)):
                    mtv = _obb_mtv(states[i].pos, states[i].theta, states[j].pos, states[j].theta, half_extents)
                    if mtv is None:
                        continue
                    any_overlap = True
                    inv_pi = 1.0 / max(1e-6, states[i].spec.priority)
                    inv_pj = 1.0 / max(1e-6, states[j].spec.priority)
                    weight_sum = inv_pi + inv_pj
                    move_i = -(inv_pi / weight_sum) * mtv
                    move_j = (inv_pj / weight_sum) * mtv
                    states[i].pos = _clamp_robot_inside_field(states[i].pos + move_i, params)
                    states[j].pos = _clamp_robot_inside_field(states[j].pos + move_j, params)
                    mid = 0.5 * (states[i].pos + states[j].pos)
                    robot_robot_collision_points.append(mid.copy())
                    states[i].robot_robot_trace.append(mid.copy())
                    states[j].robot_robot_trace.append(mid.copy())

                    opposing = states[i].spec.alliance != states[j].spec.alliance
                    illegal = opposing and (
                        _robot_fully_across_center_line(states[i].pos, states[i].theta, params, states[i].spec.alliance)
                        or _robot_fully_across_center_line(states[j].pos, states[j].theta, params, states[j].spec.alliance)
                    )
                    if illegal:
                        illegal_opponent_contact_points.append(mid.copy())
                        states[i].illegal_contact_trace.append(mid.copy())
                        states[j].illegal_contact_trace.append(mid.copy())
                        if opponent_contact_rule == "prevent":
                            states[i].pos = _clamp_robot_inside_field(states[i].target_pos + prev_offsets[i], params)
                            states[j].pos = _clamp_robot_inside_field(states[j].target_pos + prev_offsets[j], params)
            if not any_overlap:
                break

        for state, prev_offset in zip(states, prev_offsets):
            new_offset = state.pos - state.target_pos
            state.pos_offset = offset_decay * new_offset
            state.vel = state.target_vel + (state.pos_offset - prev_offset) / max(1e-9, params.physics_dt)
            state.robot_trace.append(state.pos.copy())
            state.robot_theta_trace.append(float(state.theta))

        for ball in balls:
            if ball.active:
                ball.pos += ball.vel * params.physics_dt

        available_slots = []
        for idx in range(len(states)):
            occupied = sum(1 for b_idx, b in enumerate(balls) if owner_by_ball[b_idx] == idx and b.state in (BALL_STATE_CONTACTED, BALL_STATE_CONTROLLED))
            available_slots.append(max(0, params.max_intaked_balls - occupied))

        if params.intake_enabled:
            for b_idx, ball in enumerate(balls):
                claim = _choose_contact_claim(ball, states, available_slots, params)
                if claim is None:
                    continue
                owner_idx, slot_idx = claim
                owner_state = states[owner_idx]
                owner_by_ball[b_idx] = owner_idx
                did_index = _claim_ball(ball, slot_idx, owner_state, params)
                if did_index:
                    indexed_owner[b_idx] = owner_idx
                    owner_state.indexed_count += 1
                    owner_state.capture_trace.append(owner_state.pos.copy())
                else:
                    owner_state.contact_trace.append(owner_state.pos.copy())
                available_slots[owner_idx] = max(0, available_slots[owner_idx] - 1)
                if owner_state.first_intake_time is None:
                    owner_state.first_intake_time = t

        for b_idx, ball in enumerate(balls):
            owner_idx = int(owner_by_ball[b_idx])
            if owner_idx < 0:
                continue
            owner_state = states[owner_idx]
            did_index = update_controlled_ball(ball, owner_state.pos, owner_state.theta, owner_state.vel, params, params.physics_dt)
            if did_index:
                indexed_owner[b_idx] = owner_idx
                owner_state.indexed_count += 1
                owner_state.capture_trace.append(owner_state.pos.copy())
            elif ball.state == BALL_STATE_FREE:
                owner_by_ball[b_idx] = -1

        for ball in balls:
            if not ball.simulated:
                continue
            for state in states:
                resolve_ball_robot(ball, state.pos, state.theta, state.vel, params)
            resolve_ball_field_obstacles(ball, field_obstacles, params.physics_dt)
            apply_surface_regions(ball, surface_regions, params, params.physics_dt)

        grid = build_spatial_hash(balls, cell_size)
        checked: set[tuple[int, int]] = set()
        for (cx, cy), idxs in grid.items():
            nearby: list[int] = []
            for nc in neighbor_cells(cx, cy):
                nearby.extend(grid.get(nc, []))
            for i in idxs:
                for j in nearby:
                    if j <= i:
                        continue
                    key = (i, j)
                    if key in checked:
                        continue
                    checked.add(key)
                    resolve_ball_ball(balls[i], balls[j], params)

        for ball in balls:
            if ball.simulated:
                resolve_ball_wall(ball, -FIELD_LENGTH_M / 2.0, FIELD_LENGTH_M / 2.0, -FIELD_WIDTH_M / 2.0, FIELD_WIDTH_M / 2.0, params)
                apply_motion_losses(ball, params.physics_dt, params)

        for state in states:
            if robot_body_collides(state.pos, state.theta, params, field_obstacles):
                state.collision_trace.append(state.pos.copy())
            for reg in surface_regions:
                if robot_body_in_region(state.pos, state.theta, params, reg):
                    state.bump_trace.append(state.pos.copy())
            for obs in field_obstacles:
                if "trench" in obs.tags:
                    reach = 0.5 * max(params.robot_length, params.robot_width) + float(np.linalg.norm(obs.half_extents))
                    if float(np.linalg.norm(state.pos - obs.center)) < reach:
                        state.trench_trace.append(state.pos.copy())

        if collect_snapshots and step % max(1, params.snapshot_stride) == 0:
            snapshots.append(
                MultiRobotSnapshot(
                    t=t,
                    ball_positions=np.array([b.pos.copy() for b in balls], dtype=float),
                    active_mask=np.array([b.active for b in balls], dtype=bool),
                    indexed_mask=np.array([b.indexed for b in balls], dtype=bool),
                    owner_index=owner_by_ball.copy(),
                    robot_positions=np.array([state.pos.copy() for state in states], dtype=float),
                    robot_thetas=np.array([state.theta for state in states], dtype=float),
                    robot_names=[state.spec.name for state in states],
                    indexed_counts=np.array([state.indexed_count for state in states], dtype=int),
                )
            )

    final_positions = np.array([b.pos.copy() for b in balls], dtype=float)
    active_mask = np.array([b.active for b in balls], dtype=bool)
    indexed_mask = np.array([b.indexed for b in balls], dtype=bool)

    per_robot_snapshots: list[list[Snapshot]] = [[] for _ in states]
    for snap in snapshots:
        for idx in range(len(states)):
            per_robot_snapshots[idx].append(
                Snapshot(
                    t=snap.t,
                    ball_positions=snap.ball_positions.copy(),
                    active_mask=snap.active_mask.copy(),
                    indexed_mask=np.logical_and(snap.indexed_mask, snap.owner_index == idx),
                    robot_pos=snap.robot_positions[idx].copy(),
                    robot_theta=float(snap.robot_thetas[idx]),
                    captured_count=int(snap.indexed_counts[idx]),
                )
            )

    robot_results = [
        _make_single_robot_result(
            state=state,
            balls=balls,
            owner_by_ball=owner_by_ball,
            indexed_owner=indexed_owner,
            snapshots=per_robot_snapshots[idx],
            total_time=total_time,
            seed=seed,
        )
        for idx, state in enumerate(states)
    ]

    for spec, robot_result in zip(specs, robot_results):
        robot_result._robot_color = spec.color or ROBOT_COLORS.get(spec.alliance, "tab:green")  # type: ignore[attr-defined]
        robot_result.path_name = spec.name

    return MultiRobotRunResult(
        robot_results=robot_results,
        final_positions=final_positions,
        active_mask=active_mask,
        indexed_mask=indexed_mask,
        indexed_owner=indexed_owner.copy(),
        owner_by_ball=owner_by_ball.copy(),
        snapshots=snapshots,
        total_time=total_time,
        robot_robot_collision_samples=len(robot_robot_collision_points),
        illegal_opponent_contact_samples=len(illegal_opponent_contact_points),
        robot_robot_collision_points=np.array(robot_robot_collision_points, dtype=float) if robot_robot_collision_points else np.empty((0, 2), dtype=float),
        illegal_opponent_contact_points=np.array(illegal_opponent_contact_points, dtype=float) if illegal_opponent_contact_points else np.empty((0, 2), dtype=float),
    )


def count_active_balls_by_zone(points: np.ndarray) -> tuple[int, int, int]:
    red_max_x = -FIELD_LENGTH_M / 2.0 + ALLIANCE_ZONE_DEPTH_M
    blue_min_x = FIELD_LENGTH_M / 2.0 - ALLIANCE_ZONE_DEPTH_M
    red = int(np.sum(points[:, 0] < red_max_x))
    blue = int(np.sum(points[:, 0] > blue_min_x))
    neutral = int(len(points) - red - blue)
    return red, neutral, blue


def draw_robot_outline(ax, pos: np.ndarray, theta: float, params: SimParams, color: str, label: str | None = None, alpha: float = 0.82) -> None:
    corners = _robot_corners(pos, theta, params)
    patch = mpatches.Polygon(corners, closed=True, fill=True, alpha=alpha, facecolor=color, edgecolor="black", linewidth=1.2, label=label)
    ax.add_patch(patch)
    heading = pos + rotmat(theta) @ np.array([params.robot_length / 2.0, 0.0], dtype=float)
    ax.plot([pos[0], heading[0]], [pos[1], heading[1]], linestyle="--", linewidth=1.1, color="black")


def save_multi_robot_summary_plot(result: MultiRobotRunResult, specs: list[MultiRobotSpec], params: SimParams, out_png: str | Path) -> None:
    fig = plt.figure(figsize=(15.5, 9.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])
    ax_field = fig.add_subplot(gs[:, 0])
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 1])

    draw_field_matplotlib(ax_field, palette=FieldPalette(), params=params, flags=RenderFlags(show_underlay=False, show_labels=False, show_apriltags=False))
    active_pts = result.final_positions[result.active_mask]
    indexed_pts = result.final_positions[result.indexed_mask]
    if len(active_pts):
        ax_field.scatter(active_pts[:, 0], active_pts[:, 1], s=10, alpha=0.75, label="free balls")
    if len(indexed_pts):
        ax_field.scatter(indexed_pts[:, 0], indexed_pts[:, 1], s=18, marker="x", alpha=0.95, label="indexed balls")
    if len(result.robot_robot_collision_points):
        ax_field.scatter(result.robot_robot_collision_points[:, 0], result.robot_robot_collision_points[:, 1], s=20, marker="s", alpha=0.75, label="robot-robot contacts")
    if len(result.illegal_opponent_contact_points):
        ax_field.scatter(result.illegal_opponent_contact_points[:, 0], result.illegal_opponent_contact_points[:, 1], s=32, marker="*", alpha=0.95, label="illegal opponent contacts")

    for spec, robot_result in zip(specs, result.robot_results):
        color = spec.color or ROBOT_COLORS.get(spec.alliance, "tab:green")
        if len(robot_result.robot_trace):
            ax_field.plot(robot_result.robot_trace[:, 0], robot_result.robot_trace[:, 1], linewidth=2.0, color=color, label=f"{spec.name} trace")
            draw_robot_outline(ax_field, robot_result.robot_trace[-1], robot_result.robot_theta_trace[-1], params, color=color, label=spec.name)

    ax_field.set_title("Multi-robot final state")
    ax_field.legend(loc="upper right", fontsize=8)

    draw_field_matplotlib(ax_trace, palette=FieldPalette(), params=params, flags=RenderFlags(show_underlay=False, show_labels=False, show_apriltags=False))
    for spec, robot_result in zip(specs, result.robot_results):
        color = spec.color or ROBOT_COLORS.get(spec.alliance, "tab:green")
        if len(robot_result.robot_trace):
            ax_trace.plot(robot_result.robot_trace[:, 0], robot_result.robot_trace[:, 1], linewidth=2.0, color=color, label=spec.name)
        if len(robot_result.contact_trace):
            ax_trace.scatter(robot_result.contact_trace[:, 0], robot_result.contact_trace[:, 1], s=14, marker=".", color=color, alpha=0.7)
        if len(robot_result.capture_trace):
            ax_trace.scatter(robot_result.capture_trace[:, 0], robot_result.capture_trace[:, 1], s=22, marker="o", color=color, alpha=0.9)
    ax_trace.set_title("Robot traces, contacts, and captures")
    ax_trace.legend(loc="upper right", fontsize=8)

    red_zone, neutral_zone, blue_zone = count_active_balls_by_zone(active_pts) if len(active_pts) else (0, 0, 0)
    lines = [
        f"sim total time: {result.total_time:.2f} s",
        f"total balls: {len(result.final_positions)}",
        f"free balls: {int(np.sum(result.active_mask))}",
        f"indexed balls: {int(np.sum(result.indexed_mask))}",
        f"red zone active balls: {red_zone}",
        f"neutral zone active balls: {neutral_zone}",
        f"blue zone active balls: {blue_zone}",
        f"robot-robot contact samples: {result.robot_robot_collision_samples}",
        f"illegal opponent contact samples: {result.illegal_opponent_contact_samples}",
        "",
    ]
    for spec, robot_result in zip(specs, result.robot_results):
        lines.extend(
            [
                f"{spec.name} [{spec.alliance}]",
                f"  path file: {spec.path.name}",
                f"  captured/indexed: {robot_result.captured_count}",
                f"  intake contacts: {len(robot_result.contact_trace)}",
                f"  captures: {len(robot_result.capture_trace)}",
                f"  first intake time: {robot_result.first_intake_time if robot_result.first_intake_time is not None else 'n/a'}",
                f"  obstacle collisions: {robot_result.collision_samples}",
                "",
            ]
        )

    ax_text.axis("off")
    ax_text.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)




def intake_box_polygon(robot_pos: np.ndarray, robot_theta: float, params: SimParams) -> np.ndarray:
    front_x = params.robot_length / 2.0
    local = np.array(
        [
            [front_x, -params.intake_half_width],
            [front_x + params.intake_depth, -params.intake_half_width],
            [front_x + params.intake_depth, params.intake_half_width],
            [front_x, params.intake_half_width],
            [front_x, -params.intake_half_width],
        ],
        dtype=float,
    )
    return (rotmat(robot_theta) @ local.T).T + robot_pos[None, :]


def save_multi_robot_animation(result: MultiRobotRunResult, specs: list[MultiRobotSpec], params: SimParams, out_base: str | Path) -> None:
    if not result.snapshots:
        return

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    draw_field_matplotlib(ax, palette=FieldPalette(), params=params, flags=RenderFlags(show_underlay=False, show_labels=False, show_apriltags=False))
    ax.set_title('Multi-robot autonomous animation')

    scat_free = ax.scatter([], [], s=12, label='free balls')
    scat_held = ax.scatter([], [], s=18, label='held/contacted')
    scat_indexed = ax.scatter([], [], s=18, marker='x', label='indexed')

    robot_patches = []
    heading_lines = []
    intake_lines = []
    robot_labels = []
    for spec in specs:
        color = spec.color or ROBOT_COLORS.get(spec.alliance, 'tab:green')
        patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, fill=True, alpha=0.82, facecolor=color, edgecolor='black', linewidth=1.2)
        ax.add_patch(patch)
        robot_patches.append(patch)
        heading_line, = ax.plot([], [], linestyle='--', linewidth=1.0, color='black')
        intake_line, = ax.plot([], [], linewidth=1.4, linestyle='--', color=color)
        label = ax.text(0.0, 0.0, spec.name, fontsize=8, ha='center', va='bottom', color='black')
        heading_lines.append(heading_line)
        intake_lines.append(intake_line)
        robot_labels.append(label)

    collision_scat = ax.scatter([], [], s=24, marker='s', label='robot-robot contacts')
    illegal_scat = ax.scatter([], [], s=38, marker='*', label='illegal opponent contacts')
    info = ax.text(0.01, 0.99, '', transform=ax.transAxes, va='top', family='monospace')
    ax.legend(loc='upper right', fontsize=8)

    def init():
        empty = np.zeros((0, 2))
        scat_free.set_offsets(empty)
        scat_held.set_offsets(empty)
        scat_indexed.set_offsets(empty)
        collision_scat.set_offsets(empty)
        illegal_scat.set_offsets(empty)
        for patch in robot_patches:
            patch.set_xy(np.zeros((4, 2)))
        for heading_line, intake_line, label in zip(heading_lines, intake_lines, robot_labels):
            heading_line.set_data([], [])
            intake_line.set_data([], [])
            label.set_position((0.0, 0.0))
        info.set_text('')
        return [scat_free, scat_held, scat_indexed, collision_scat, illegal_scat, info, *robot_patches, *heading_lines, *intake_lines, *robot_labels]

    def update(frame_idx):
        snap = result.snapshots[frame_idx]
        pts = snap.ball_positions
        active = snap.active_mask
        indexed = snap.indexed_mask
        held = ~(active | indexed)
        scat_free.set_offsets(pts[active] if active.any() else np.zeros((0, 2)))
        scat_held.set_offsets(pts[held] if held.any() else np.zeros((0, 2)))
        scat_indexed.set_offsets(pts[indexed] if indexed.any() else np.zeros((0, 2)))

        if len(result.robot_robot_collision_points):
            rr_pts = result.robot_robot_collision_points[result.robot_robot_collision_points[:, 0] == result.robot_robot_collision_points[:, 0]]
        else:
            rr_pts = np.zeros((0, 2))
        if len(result.illegal_opponent_contact_points):
            il_pts = result.illegal_opponent_contact_points[result.illegal_opponent_contact_points[:, 0] == result.illegal_opponent_contact_points[:, 0]]
        else:
            il_pts = np.zeros((0, 2))

        # Show only recent contact points up to current snapshot time.
        collision_scat.set_offsets(rr_pts if len(rr_pts) else np.zeros((0, 2)))
        illegal_scat.set_offsets(il_pts if len(il_pts) else np.zeros((0, 2)))

        for idx, spec in enumerate(specs):
            pos = snap.robot_positions[idx]
            theta = float(snap.robot_thetas[idx])
            corners = _robot_corners(pos, theta, params)
            robot_patches[idx].set_xy(corners)
            heading = pos + rotmat(theta) @ np.array([params.robot_length / 2.0, 0.0], dtype=float)
            heading_lines[idx].set_data([pos[0], heading[0]], [pos[1], heading[1]])
            intake_poly = intake_box_polygon(pos, theta, params)
            intake_lines[idx].set_data(intake_poly[:, 0], intake_poly[:, 1])
            robot_labels[idx].set_position((pos[0], pos[1] + params.robot_width * 0.65))
            robot_labels[idx].set_text(f"{spec.name} ({snap.indexed_counts[idx]})")

        per_robot = ' | '.join(f"{spec.name}:{int(snap.indexed_counts[idx])}" for idx, spec in enumerate(specs))
        held_count = int(held.sum())
        info.set_text(
            f"t = {snap.t:.2f} s\n"
            f"indexed total = {int(indexed.sum())}\n"
            f"held = {held_count}\n"
            f"per robot = {per_robot}"
        )
        return [scat_free, scat_held, scat_indexed, collision_scat, illegal_scat, info, *robot_patches, *heading_lines, *intake_lines, *robot_labels]

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(result.snapshots),
        interval=max(20, int(params.physics_dt * params.snapshot_stride * 1000)),
        blit=False,
        repeat=False,
    )
    out_base = Path(out_base)
    try:
        gif_path = out_base.with_suffix('.gif')
        anim.save(gif_path, writer='pillow', fps=20)
        print(f'Saved animation: {gif_path}')
    except Exception as e:
        print('Animation save unavailable:', e)
    plt.close(fig)


def list_path_files() -> list[Path]:
    return sorted(list(Path("paths").glob("*.json")) + list(Path("paths").glob("*.path")))


def prompt_int(prompt: str, default: int = 0, minimum: int = 0, maximum: int | None = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
        if value < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be at most {maximum}.")
            continue
        return value


def prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def prompt_choice_index(files: list[Path], prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter one of the listed indices.")
            continue
        if 0 <= idx < len(files):
            return idx
        print("Index out of range.")


def prompt_robot_specs() -> list[MultiRobotSpec]:
    files = list_path_files()
    if not files:
        raise FileNotFoundError("No saved paths found in ./paths. Save some in path_editor.py first.")

    print("\nSaved paths:")
    for idx, path_file in enumerate(files):
        print(f"[{idx}] {path_file.name}")

    red_count = prompt_int("How many red alliance robots?", default=1, minimum=0, maximum=3)
    blue_count = prompt_int("How many blue alliance robots?", default=1, minimum=0, maximum=3)
    if red_count + blue_count <= 0:
        raise ValueError("You need at least one robot.")

    specs: list[MultiRobotSpec] = []
    for alliance, count in (("red", red_count), ("blue", blue_count)):
        for slot in range(count):
            print(f"\nChoose path for {alliance.title()} {slot + 1}:")
            idx = prompt_choice_index(files, "Path index: ")
            path = load_pathspec_json(files[idx])
            default_name = f"{alliance.title()} {slot + 1}"
            custom_name = input(f"Robot label [{default_name}]: ").strip() or default_name
            priority = prompt_float("Collision priority (higher moves less)", default=1.0)
            specs.append(
                MultiRobotSpec(
                    name=custom_name,
                    alliance=alliance,
                    path=path,
                    priority=priority,
                    intake_enabled=True,
                    color=ROBOT_COLORS.get(alliance),
                )
            )
    return specs


def save_run_bundle(result: MultiRobotRunResult, specs: list[MultiRobotSpec], params: SimParams, out_dir: Path, save_animation: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_params_json(params, out_dir / "params.json")
    save_multi_robot_summary_plot(result, specs, params, out_dir / 'multi_robot_summary.png')
    if save_animation:
        save_multi_robot_animation(result, specs, params, out_dir / 'multi_robot_animation.gif')

    config_rows = [
        {
            "name": spec.name,
            "alliance": spec.alliance,
            "path_name": spec.path.name,
            "priority": spec.priority,
        }
        for spec in specs
    ]
    (out_dir / "robot_config.json").write_text(json.dumps(config_rows, indent=2))

    summary_lines = [
        f"total_time={result.total_time:.3f}",
        f"robot_robot_collision_samples={result.robot_robot_collision_samples}",
        f"illegal_opponent_contact_samples={result.illegal_opponent_contact_samples}",
    ]
    for spec, robot_result in zip(specs, result.robot_results):
        summary_lines.extend(
            [
                f"{spec.name}_alliance={spec.alliance}",
                f"{spec.name}_path={spec.path.name}",
                f"{spec.name}_captured={robot_result.captured_count}",
                f"{spec.name}_contacts={len(robot_result.contact_trace)}",
            ]
        )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")


def main() -> None:
    print("REBUILT multi-robot autonomous runner")
    print("This uses the same saved path files as path_editor.py (./paths/*.json and *.path).")

    specs = prompt_robot_specs()

    params = SimParams()
    params.n_balls_target = prompt_int("Ball count", default=params.n_balls_target, minimum=1)
    params.physics_dt = prompt_float("Physics dt (s)", default=params.physics_dt)
    params.max_intaked_balls = prompt_int("Max intaked balls per robot", default=params.max_intaked_balls, minimum=1)
    seed = prompt_int("Random seed", default=0, minimum=0)
    sim_duration_s = prompt_float("Simulation duration in seconds", default=20.0)
    rule_mode = input("Opponent contact mode (ignore/warn/prevent) [warn]: ").strip().lower() or "warn"
    if rule_mode not in {"ignore", "warn", "prevent"}:
        print("Unrecognized mode, using 'warn'.")
        rule_mode = "warn"

    out_dir = Path('outputs') / 'multi_robot_auto'
    make_animation = (input('Save animation gif? (y/n) [y]: ').strip().lower() or 'y') not in {'n', 'no'}
    result = simulate_multi_robot_auto(
        specs,
        params,
        seed=seed,
        collect_snapshots=make_animation,
        opponent_contact_rule=rule_mode,
        sim_duration_s=sim_duration_s,
    )
    save_run_bundle(result, specs, params, out_dir, save_animation=make_animation)

    print("\nRun complete.")
    print(f"Saved summary plot to: {out_dir / 'multi_robot_summary.png'}")
    if make_animation:
        print(f"Saved animation to:    {out_dir / 'multi_robot_animation.gif'}")
    print(f"Saved params to:        {out_dir / 'params.json'}")
    print(f"Saved config to:        {out_dir / 'robot_config.json'}")
    print(f"Saved summary text to:  {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
