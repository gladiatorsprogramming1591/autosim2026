
"""
batch_path_heatmaps.py

Batch runner for all saved JSON / PathPlanner paths in ./paths, else defaults.
Produces per-path heatmaps for final active-ball positions plus robot-diagnostic maps for
contact locations, capture locations, bump samples, trench samples, and collision samples.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared_field_visuals import FieldPalette, RenderFlags, draw_field_matplotlib
from shared_sim_core import (
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    SimParams,
    default_paths,
    load_pathspec_json,
    save_numpy_dict,
    save_params_json,
    simulate_path,
)


def load_paths():
    files = sorted(list(Path("paths").glob("*.json")) + list(Path("paths").glob("*.path")))
    return [load_pathspec_json(f) for f in files] if files else default_paths()


def summarize_points(pts: np.ndarray):
    if len(pts) == 0:
        return {"mean_x": np.nan, "std_x": np.nan, "mean_abs_y": np.nan, "right_half_pct": np.nan}
    return {
        "mean_x": float(np.mean(pts[:, 0])),
        "std_x": float(np.std(pts[:, 0])),
        "mean_abs_y": float(np.mean(np.abs(pts[:, 1]))),
        "right_half_pct": float(100.0 * np.mean(pts[:, 0] > 0.0)),
    }


def save_trace_heatmap(title: str, pts: np.ndarray, params: SimParams, out_png: Path):
    fig, ax = plt.subplots(figsize=(14, 6.0))
    draw_field_matplotlib(ax, palette=FieldPalette(), params=params, flags=RenderFlags(show_underlay=False, show_labels=False, show_apriltags=False))
    if len(pts):
        h = ax.hist2d(
            pts[:, 0],
            pts[:, 1],
            bins=[params.heatmap_bins_x, params.heatmap_bins_y],
            range=[[-FIELD_LENGTH_M / 2.0, FIELD_LENGTH_M / 2.0], [-FIELD_WIDTH_M / 2.0, FIELD_WIDTH_M / 2.0]],
            cmap="inferno",
        )
        fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_metric_bar(rows, key, title, out_png: Path):
    labels = [row["path"] for row in rows]
    values = [0.0 if row[key] is None or (isinstance(row[key], float) and np.isnan(row[key])) else row[key] for row in rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(key.replace("_", " "))
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    out_dir = Path("outputs/batch_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = SimParams(
        physics_dt=1.0 / 180.0,
        n_balls_target=300,
        intake_enabled=True,
        max_intaked_balls=40,
    )
    n_trials_per_path = 12
    paths = load_paths()
    save_params_json(params, out_dir / "params.json")

    total_jobs = len(paths) * n_trials_per_path
    job = 0
    rows = []
    t0 = time.time()

    for p_idx, path in enumerate(paths):
        active_runs = []
        finals = []
        masks = []
        contact_pts = []
        capture_pts = []
        bump_pts = []
        trench_pts = []
        collision_pts = []
        first_intake_times = []
        collision_samples = []
        bump_samples = []
        trench_samples = []
        captured_counts = []

        for trial in range(n_trials_per_path):
            job += 1
            seed = params.random_seed_base + 1000 * p_idx + trial
            print(f"[{job}/{total_jobs}] {path.name} trial {trial + 1}/{n_trials_per_path}")
            result = simulate_path(path, params, seed=seed, collect_snapshots=False)
            finals.append(result.final_positions)
            masks.append(result.active_mask)
            if result.active_mask.any():
                active_runs.append(result.final_positions[result.active_mask])
            if len(result.contact_trace):
                contact_pts.append(result.contact_trace)
            if len(result.capture_trace):
                capture_pts.append(result.capture_trace)
            if len(result.bump_trace):
                bump_pts.append(result.bump_trace)
            if len(result.trench_trace):
                trench_pts.append(result.trench_trace)
            if len(result.collision_trace):
                collision_pts.append(result.collision_trace)
            if result.first_intake_time is not None:
                first_intake_times.append(result.first_intake_time)
            collision_samples.append(result.collision_samples)
            bump_samples.append(result.bump_region_samples)
            trench_samples.append(result.trench_region_samples)
            captured_counts.append(result.captured_count)

        finals_arr = np.stack(finals, axis=0)
        masks_arr = np.stack(masks, axis=0)
        stem = path.name.replace(" ", "_")
        save_numpy_dict(out_dir / f"{stem}_raw.npz", final_positions=finals_arr, active_mask=masks_arr)

        pts = np.vstack(active_runs) if active_runs else np.empty((0, 2))
        contact_arr = np.vstack(contact_pts) if contact_pts else np.empty((0, 2))
        capture_arr = np.vstack(capture_pts) if capture_pts else np.empty((0, 2))
        bump_arr = np.vstack(bump_pts) if bump_pts else np.empty((0, 2))
        trench_arr = np.vstack(trench_pts) if trench_pts else np.empty((0, 2))
        collision_arr = np.vstack(collision_pts) if collision_pts else np.empty((0, 2))

        save_trace_heatmap(f"Final active-ball heatmap: {path.name}", pts, params, out_dir / f"{stem}_final_positions_heatmap.png")
        save_trace_heatmap(f"Intake contact density: {path.name}", contact_arr, params, out_dir / f"{stem}_contact_density_heatmap.png")
        save_trace_heatmap(f"Capture density: {path.name}", capture_arr, params, out_dir / f"{stem}_capture_density_heatmap.png")
        save_trace_heatmap(f"Bump crossings: {path.name}", bump_arr, params, out_dir / f"{stem}_bump_crossings_heatmap.png")
        save_trace_heatmap(f"Trench crossings: {path.name}", trench_arr, params, out_dir / f"{stem}_trench_crossings_heatmap.png")
        save_trace_heatmap(f"Collision density: {path.name}", collision_arr, params, out_dir / f"{stem}_collision_density_heatmap.png")

        metrics = summarize_points(pts)
        metrics.update(
            {
                "path": path.name,
                "source_format": path.metadata.get("source_format", "native"),
                "total_active_points": int(len(pts)),
                "mean_collision_samples": float(np.mean(collision_samples)) if collision_samples else np.nan,
                "mean_bump_samples": float(np.mean(bump_samples)) if bump_samples else np.nan,
                "mean_trench_samples": float(np.mean(trench_samples)) if trench_samples else np.nan,
                "mean_captured_count": float(np.mean(captured_counts)) if captured_counts else np.nan,
                "std_captured_count": float(np.std(captured_counts)) if captured_counts else np.nan,
                "mean_first_intake_time": float(np.mean(first_intake_times)) if first_intake_times else np.nan,
                "capture_events_total": int(len(capture_arr)),
                "contact_events_total": int(len(contact_arr)),
            }
        )
        rows.append(metrics)

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "source_format",
                "total_active_points",
                "mean_x",
                "std_x",
                "mean_abs_y",
                "right_half_pct",
                "mean_collision_samples",
                "mean_bump_samples",
                "mean_trench_samples",
                "mean_captured_count",
                "std_captured_count",
                "mean_first_intake_time",
                "capture_events_total",
                "contact_events_total",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    save_metric_bar(rows, "mean_first_intake_time", "Time to first intake", out_dir / "time_to_first_intake.png")
    save_metric_bar(rows, "mean_captured_count", "Average captured/indexed balls", out_dir / "capture_summary.png")
    save_metric_bar(rows, "std_captured_count", "Capture count consistency", out_dir / "capture_consistency.png")
    save_metric_bar(rows, "mean_collision_samples", "Average collision samples", out_dir / "collision_summary.png")
    save_metric_bar(rows, "mean_bump_samples", "Average bump samples", out_dir / "bump_summary.png")
    save_metric_bar(rows, "mean_trench_samples", "Average trench samples", out_dir / "trench_summary.png")

    elapsed = time.time() - t0
    (out_dir / "runtime.txt").write_text(f"{elapsed:.3f} seconds\n")
    print(f"Completed batch run in {elapsed:.2f} s")
    print(f"Outputs saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
