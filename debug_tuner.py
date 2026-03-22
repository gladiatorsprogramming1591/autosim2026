
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from shared_field_visuals import FieldPalette, RenderFlags, draw_field_matplotlib
from shared_sim_core import (
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    ALLIANCE_ZONE_DEPTH_M,
    SimParams,
    alliance_hub_centers,
    build_apriltag_metadata,
    build_field_obstacles,
    build_field_reference_points,
    build_field_surface_regions,
    default_paths,
    load_pathspec_json,
    path_diagnostics,
    rotmat,
    save_numpy_dict,
    save_params_json,
    simulate_path,
    BALL_RADIUS_M
)


def list_path_files() -> list[Path]:
    return sorted(list(Path("paths").glob("*.json")) + list(Path("paths").glob("*.path")))


def choose_path():
    files = list_path_files()
    if files:
        print("Saved paths:")
        for idx, f in enumerate(files):
            print(f"[{idx}] {f.name}")
        try:
            idx = int(input("Choose path index (blank for 0): ") or "0")
            return load_pathspec_json(files[idx])
        except Exception:
            pass
    return default_paths()[0]


def print_geometry_summary(params: SimParams):
    obstacles = build_field_obstacles(params)
    regions = build_field_surface_regions(params)
    refs = build_field_reference_points(params)
    centers = alliance_hub_centers(params)
    tags = build_apriltag_metadata(params)

    print("\nGeometry summary")
    print("----------------")
    print(f"hub centers: red={centers['red']}, blue={centers['blue']}")
    print(f"obstacle count: {len(obstacles)}")
    print(f"surface region count: {len(regions)}")
    print(f"reference point count: {len(refs)}")
    print(f"AprilTag count: {len(tags)}")

    print("\nObstacles")
    for obs in obstacles:
        print(f"- {obs.name:18s} center=({obs.center[0]:6.3f}, {obs.center[1]:6.3f})  size=({2*obs.half_extents[0]:5.3f}, {2*obs.half_extents[1]:5.3f})  tags={obs.tags}")

    print("\nSurface regions")
    for reg in regions:
        print(f"- {reg.name:18s} center=({reg.center[0]:6.3f}, {reg.center[1]:6.3f})  size=({2*reg.half_extents[0]:5.3f}, {2*reg.half_extents[1]:5.3f})  tags={reg.tags}")

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


def intake_hold_point(robot_pos: np.ndarray, robot_theta: float, params: SimParams) -> np.ndarray:
    # This matches the "front of robot" hold interpretation:
    local = np.array(
        [params.robot_length / 2.0 + BALL_RADIUS_M + params.intake_hold_offset, 0.0],
        dtype=float,
    )
    return robot_pos + rotmat(robot_theta) @ local

def count_balls_by_zone(points: np.ndarray) -> tuple[int, int, int]:
    red_max_x = -FIELD_LENGTH_M / 2.0 + ALLIANCE_ZONE_DEPTH_M
    blue_min_x = FIELD_LENGTH_M / 2.0 - ALLIANCE_ZONE_DEPTH_M

    red = int(np.sum(points[:, 0] < red_max_x))
    blue = int(np.sum(points[:, 0] > blue_min_x))
    neutral = int(len(points) - red - blue)

    return red, blue, neutral

def plot_results_page(result, path, params: SimParams, out_png: Path):
    diag = path_diagnostics(path, params)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    draw_field_matplotlib(ax1, palette=FieldPalette(), params=params, flags=RenderFlags(show_underlay=False, show_labels=False, show_apriltags=True))
    pts = result.final_positions
    indexed_pts = result.indexed_positions
    active = result.active_mask
    indexed = result.indexed_mask
    held = ~(active | indexed)
    red_zone_balls, blue_zone_balls, neutral_zone_balls = count_balls_by_zone(pts[active])

    if active.any():
        ax1.scatter(pts[active, 0], pts[active, 1], s=11, label="free balls")
    if held.any():
        ax1.scatter(pts[held, 0], pts[held, 1], s=16, label="held/contacted balls")
    if indexed.any():
        ax1.scatter(indexed_pts[indexed, 0], indexed_pts[indexed, 1], s=20, marker="x", label="indexed balls")
    if len(result.contact_trace):
        ax1.scatter(result.contact_trace[:, 0], result.contact_trace[:, 1], s=16, marker="+", label="intake contacts")
    if len(result.capture_trace):
        ax1.scatter(result.capture_trace[:, 0], result.capture_trace[:, 1], s=14, marker="o", label="captures")
    ax1.set_title("Final state")
    ax1.legend(loc="upper right")
    # show intake geometry at the end pose
    if len(result.robot_trace):
        robot_pos = result.robot_trace[-1]
        robot_theta = 0.0
        if hasattr(result, "robot_theta_trace") and len(result.robot_theta_trace):
            robot_theta = result.robot_theta_trace[-1]
        intake_poly = intake_box_polygon(robot_pos, robot_theta, params)
        ax1.plot(intake_poly[:, 0], intake_poly[:, 1], linewidth=1.5, linestyle="--")
        hp = intake_hold_point(robot_pos, robot_theta, params)
        ax1.scatter([hp[0]], [hp[1]], s=24, marker="o")    

    ax2 = fig.add_subplot(gs[0, 1])
    draw_field_matplotlib(ax2, palette=FieldPalette(), params=params, flags=RenderFlags(show_underlay=False, show_labels=False, show_apriltags=False))
    if len(diag["robot_trace"]):
        rt = diag["robot_trace"]
        ax2.plot(rt[:, 0], rt[:, 1], linewidth=2.0, label="robot trace")
        # draw a few intake boxes along the path so the intake footprint is visible
    sample_count = min(10, len(rt)) if len(diag["robot_trace"]) else 0
    if sample_count > 0:
        idxs = np.linspace(0, len(rt) - 1, sample_count, dtype=int)
        for idx in idxs:
            if idx < len(diag["robot_theta_trace"]):
                poly = intake_box_polygon(rt[idx], diag["robot_theta_trace"][idx], params)
                ax2.plot(poly[:, 0], poly[:, 1], linewidth=1.0)
                hp = intake_hold_point(rt[idx], diag["robot_theta_trace"][idx], params)
                ax2.scatter([hp[0]], [hp[1]], s=12, marker="o")
    if len(diag["collision_trace"]):
        ax2.scatter(diag["collision_trace"][:, 0], diag["collision_trace"][:, 1], s=14, label="collision samples")
    if len(diag["bump_trace"]):
        ax2.scatter(diag["bump_trace"][:, 0], diag["bump_trace"][:, 1], s=14, label="bump samples")
    if len(diag["trench_trace"]):
        ax2.scatter(diag["trench_trace"][:, 0], diag["trench_trace"][:, 1], s=14, label="trench samples")
    ax2.set_title("Path diagnostics")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(gs[1, 0])
    if active.any():
        h = ax3.hist2d(
            pts[active, 0],
            pts[active, 1],
            bins=[params.heatmap_bins_x, params.heatmap_bins_y],
            range=[[-FIELD_LENGTH_M / 2.0, FIELD_LENGTH_M / 2.0], [-FIELD_WIDTH_M / 2.0, FIELD_WIDTH_M / 2.0]],
            cmap="inferno",
        )
        fig.colorbar(h[3], ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title("Final active-ball heatmap")
    ax3.set_aspect("equal")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    lines = [
        f"path: {result.path_name}",
        f"seed: {result.seed}",
    #    f"source format: {path.metadata.get('source_format', 'native')}",
        f"sim total time: {result.total_time:.2f} s",
        f"total balls: {len(result.final_positions)}",
        f"red alliance zone balls: {red_zone_balls}",
        f"neutral zone balls: {neutral_zone_balls}",
        f"blue alliance zone balls: {blue_zone_balls}",
        f"free balls: {result.free_count}",
    #    f"contacted balls: {result.contacted_count}",
    #    f"controlled balls: {result.controlled_count}",
        f"captured/indexed count: {result.captured_count}",
        f"collision samples: {result.collision_samples}",
    #    f"bump samples: {result.bump_region_samples}",
    #    f"trench samples: {result.trench_region_samples}",
    #    f"first intake time: {result.first_intake_time if result.first_intake_time is not None else 'n/a'}",
    ]
    ax4.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_animation(result, params: SimParams, out_base: Path):
    if not result.snapshots:
        return
    fig, ax = plt.subplots(figsize=(13, 6.0))
    draw_field_matplotlib(ax, palette=FieldPalette(), params=params, flags=RenderFlags(show_underlay=False, show_labels=False, show_apriltags=False))
    ax.set_title(f"Debug run animation: {result.path_name}")

    scat_free = ax.scatter([], [], s=12, label="free")
    scat_held = ax.scatter([], [], s=18, label="held/contacted")
    scat_indexed = ax.scatter([], [], s=18, marker="x", label="indexed")
    robot_line, = ax.plot([], [], linewidth=2.0, color="cyan")
    intake_line, = ax.plot([], [], linewidth=1.5, linestyle="--", label="intake box")
    hold_point_plot = ax.scatter([], [], s=24, marker="o", label="hold target")
    info = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", family="monospace")
    ax.legend(loc="upper right")
    def init():
        scat_free.set_offsets(np.zeros((0, 2)))
        scat_held.set_offsets(np.zeros((0, 2)))
        scat_indexed.set_offsets(np.zeros((0, 2)))
        robot_line.set_data([], [])
        intake_line.set_data([], [])
        hold_point_plot.set_offsets(np.zeros((0, 2)))
        info.set_text("")
        return scat_free, scat_held, scat_indexed, robot_line, intake_line, hold_point_plot, info

    def update(i):
        snap = result.snapshots[i]
        pts = snap.ball_positions
        free = snap.active_mask
        indexed = snap.indexed_mask
        held = ~(free | indexed)
        scat_free.set_offsets(pts[free] if free.any() else np.zeros((0, 2)))
        scat_held.set_offsets(pts[held] if held.any() else np.zeros((0, 2)))
        scat_indexed.set_offsets(pts[indexed] if indexed.any() else np.zeros((0, 2)))

        hx, hy = params.robot_length / 2.0, params.robot_width / 2.0
        local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy], [hx, hy]])
        R = rotmat(snap.robot_theta)
        world = (R @ local.T).T + snap.robot_pos[None, :]
        robot_line.set_data(world[:, 0], world[:, 1])
        intake_poly = intake_box_polygon(snap.robot_pos, snap.robot_theta, params)
        intake_line.set_data(intake_poly[:, 0], intake_poly[:, 1])

        hp = intake_hold_point(snap.robot_pos, snap.robot_theta, params)
        hold_point_plot.set_offsets(np.array([[hp[0], hp[1]]], dtype=float))
        info.set_text(
            f"t = {snap.t:.2f} s\n"
            f"indexed = {snap.captured_count}\n"
            f"held = {int(held.sum())}"
        )
        return scat_free, scat_held, scat_indexed, robot_line, intake_line, hold_point_plot, info

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(result.snapshots),
        interval=max(20, int(params.physics_dt * params.snapshot_stride * 1000)),
        blit=False,
        repeat=False,
    )
    try:
        gif_path = out_base.with_suffix(".gif")
        anim.save(gif_path, writer="pillow", fps=20)
        print(f"Saved animation: {gif_path}")
    except Exception as e:
        print("Animation save unavailable:", e)
    plt.close(fig)


def main():
    out_dir = Path("outputs/debug_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = SimParams()
    path = choose_path()
    for i in range (1):
        seed = i
        print("Running " + f"'{path.name}' with seed {i}")
        #print_geometry_summary(params)
        t0 = time.time()
        result = simulate_path(path, params, seed=seed, collect_snapshots=True)
        elapsed = time.time() - t0

        save_params_json(params, out_dir / f"params_{i}.json")
        save_numpy_dict(
            out_dir / f"run_data_{i}.npz",
            final_positions=result.final_positions,
            active_mask=result.active_mask,
            indexed_mask=result.indexed_mask,
            indexed_positions=result.indexed_positions,
            robot_trace=result.robot_trace,
            collision_trace=result.collision_trace,
            bump_trace=result.bump_trace,
            trench_trace=result.trench_trace,
            capture_trace=result.capture_trace,
            contact_trace=result.contact_trace,
        )
        plot_results_page(result, path, params, out_dir / f"results_page_{i}.png")
        save_animation(result, params, out_dir / f"animation_{i}.gif")
        (out_dir / f"runtime_{i}.txt").write_text(f"{elapsed:.3f} seconds\n")
        print(f"Runtime: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
