
"""
path_editor.py


Numeric-first REBUILT path editor.
Controls:
- Left click empty space: add waypoint
- Left click waypoint: select
- Drag selected waypoint: move
- Right drag waypoint: set heading directly
- F: toggle follow-heading tangent mode
- Q/E: rotate selected heading
- Up/Down: max velocity
- Left/Right: max acceleration
- I: insert after selected
- Delete/Backspace: delete selected
- C: toggle spline
- S: save native JSON
- P: export PathPlanner .path
- L: load/cycle saved paths (.json and .path)
- R/N: rename/new
- T: cycle render mode (numeric / underlay / both)
- G: toggle snap-to-reference anchors
- H: toggle labels
- Tab: cycle selection
- Esc: quit
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pygame

from shared_sim_core import (
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    PathSpec,
    PathWaypoint,
    SimParams,
    build_field_obstacles,
    build_field_reference_points,
    build_field_surface_regions,
    generate_trajectory,
    load_pathspec_json,
    point_in_obb,
    robot_body_collides,
    robot_body_in_region,
    robot_pose_from_trajectory,
    save_pathplanner_path,
    save_pathspec_json,
)
from shared_field_visuals import FieldPalette, RenderFlags, draw_field


class Viewport:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def world_to_screen(self, pt: Tuple[float, float], rect: pygame.Rect):
        x, y = pt
        sx = rect.left + (x - self.xmin) / (self.xmax - self.xmin) * rect.width
        sy = rect.bottom - (y - self.ymin) / (self.ymax - self.ymin) * rect.height
        return int(sx), int(sy)

    def screen_to_world(self, pt: Tuple[int, int], rect: pygame.Rect):
        sx, sy = pt
        x = self.xmin + (sx - rect.left) / rect.width * (self.xmax - self.xmin)
        y = self.ymin + (rect.bottom - sy) / rect.height * (self.ymax - self.ymin)
        return float(x), float(y)


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("REBUILT Path Editor")
        self.screen = pygame.display.set_mode((1520, 960), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.small = pygame.font.SysFont("consolas", 16)
        self.palette = FieldPalette()
        self.params = SimParams()

        self.viewport = Viewport(-FIELD_LENGTH_M / 2.0, FIELD_LENGTH_M / 2.0, -FIELD_WIDTH_M / 2.0, FIELD_WIDTH_M / 2.0)
        self.paths_dir = Path("paths")
        self.path = self.default_path()
        self.selected_idx: Optional[int] = 0
        self.dragging = False
        self.heading_drag = False
        self.follow_heading = False
        self.message = "Ready"
        self.last_loaded: Optional[Path] = None
        self.render_mode = 0  # 0 numeric, 1 underlay, 2 both
        self.snap_enabled = True
        self.show_labels = True
        self.constraint_mode = 0  # 0 linear, 1 angular

    def list_saved_paths(self) -> list[Path]:
        files = list(self.paths_dir.glob("*.json")) + list(self.paths_dir.glob("*.path"))
        return sorted(files)

    def default_path(self) -> PathSpec:
        return PathSpec(
            name="My Path",
            max_velocity=3.5,
            max_acceleration=2.5,
            use_spline=True,
            samples_per_seg=28,
            metadata={
                "max_angular_velocity_deg_s": 540.0,
                "max_angular_acceleration_deg_s2": 720.0,
            },
            waypoints=[
                PathWaypoint(-5.8, 0.0, 0.0),
                PathWaypoint(-2.2, 0.0, 0.0),
                PathWaypoint(0.8, 0.0, 0.0),
                PathWaypoint(3.2, 0.0, 0.0),
            ],
        )

    def field_rect(self):
        W, H = self.screen.get_size()
        return pygame.Rect(18, 130, W - 36, H - 148)

    def current_flags(self) -> RenderFlags:
        return RenderFlags(
            show_numeric_geometry=self.render_mode in (0, 2),
            show_underlay=self.render_mode in (1, 2),
            show_labels=self.show_labels,
            show_centers=False,
            show_clearance_boxes=True,
            show_apriltags=True,
            underlay_alpha=90,
        )

    def set_message(self, msg: str):
        self.message = msg
        print(msg)

    def find_waypoint_at(self, mouse_pos, rect, r=11):
        for i, wp in enumerate(self.path.waypoints):
            sx, sy = self.viewport.world_to_screen((wp.x, wp.y), rect)
            if (sx - mouse_pos[0]) ** 2 + (sy - mouse_pos[1]) ** 2 <= r ** 2:
                return i
        return None

    def apply_follow_heading(self):
        if not self.follow_heading or len(self.path.waypoints) < 2:
            return
        for i, wp in enumerate(self.path.waypoints):
            if i < len(self.path.waypoints) - 1:
                nxt = self.path.waypoints[i + 1]
                dx, dy = nxt.x - wp.x, nxt.y - wp.y
            else:
                prv = self.path.waypoints[i - 1]
                dx, dy = wp.x - prv.x, wp.y - prv.y
            if abs(dx) + abs(dy) > 1e-9:
                wp.theta_deg = math.degrees(math.atan2(dy, dx))

    def snap_world_point(self, world_pos: Tuple[float, float], threshold: float = 0.18) -> Tuple[float, float]:
        if not self.snap_enabled:
            return world_pos
        pt = np.array(world_pos, dtype=float)
        refs = build_field_reference_points(self.params)
        if not refs:
            return world_pos
        dists = [np.linalg.norm(ref.pos - pt) for ref in refs]
        idx = int(np.argmin(dists))
        if dists[idx] <= threshold:
            ref = refs[idx]
            self.set_message(f"Snapped to {ref.name}")
            return float(ref.pos[0]), float(ref.pos[1])
        return world_pos

    def add_waypoint(self, world_pos):
        world_pos = self.snap_world_point(world_pos)
        new_wp = PathWaypoint(world_pos[0], world_pos[1], 0.0)
        if self.selected_idx is None or self.selected_idx == len(self.path.waypoints) - 1:
            self.path.waypoints.append(new_wp)
            self.selected_idx = len(self.path.waypoints) - 1
        else:
            self.path.waypoints.insert(self.selected_idx + 1, new_wp)
            self.selected_idx += 1
        self.apply_follow_heading()
        self.set_message("Added waypoint")

    def insert_after_selected(self):
        if self.selected_idx is None or not self.path.waypoints:
            return
        cur = self.path.waypoints[self.selected_idx]
        if self.selected_idx == len(self.path.waypoints) - 1:
            new_wp = PathWaypoint(cur.x + 0.8, cur.y, cur.theta_deg)
            self.path.waypoints.append(new_wp)
            self.selected_idx = len(self.path.waypoints) - 1
        else:
            nxt = self.path.waypoints[self.selected_idx + 1]
            new_wp = PathWaypoint(0.5 * (cur.x + nxt.x), 0.5 * (cur.y + nxt.y), cur.theta_deg)
            self.path.waypoints.insert(self.selected_idx + 1, new_wp)
            self.selected_idx += 1
        self.apply_follow_heading()
        self.set_message("Inserted waypoint")

    def delete_selected(self):
        if self.selected_idx is None or not self.path.waypoints:
            return
        del self.path.waypoints[self.selected_idx]
        if self.path.waypoints:
            self.selected_idx = min(self.selected_idx, len(self.path.waypoints) - 1)
        else:
            self.selected_idx = None
        self.set_message("Deleted waypoint")

    def _safe_stem(self) -> str:
        return "".join(c if c.isalnum() or c in ("_", "-", " ") else "_" for c in self.path.name).strip().replace(" ", "_") or "unnamed_path"

    def save_current(self):
        self.paths_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.paths_dir / f"{self._safe_stem()}.json"
        save_pathspec_json(self.path, out_path)
        self.last_loaded = out_path
        self.set_message(f"Saved native path: {out_path.name}")

    def export_pathplanner(self):
        self.paths_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.paths_dir / f"{self._safe_stem()}.path"
        save_pathplanner_path(self.path, out_path)
        self.last_loaded = out_path
        self.set_message(f"Exported PathPlanner path: {out_path.name}")

    def load_interactive(self):
        files = self.list_saved_paths()
        if not files:
            self.set_message("No saved paths")
            return

        if self.last_loaded is not None and self.last_loaded in files:
            try:
                idx = files.index(self.last_loaded)
                chosen = files[(idx + 1) % len(files)]
            except Exception:
                chosen = files[0]
        else:
            chosen = files[0]

        self.path = load_pathspec_json(chosen)
        self.selected_idx = 0 if self.path.waypoints else None
        self.last_loaded = chosen
        self.set_message(f"Loaded {chosen.name} (press L to cycle)")

    def rename(self):
        try:
            name = input("Enter path name: ").strip()
        except EOFError:
            name = ""
        if name:
            self.path.name = name
            self.set_message(f"Renamed to {name}")

    def draw_robot_ghost(self, x, y, theta, rect, color=(210, 110, 240)):
        L, W = self.params.robot_length, self.params.robot_width
        hx, hy = L / 2.0, W / 2.0
        local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy], [hx, hy]])
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        world = (R @ local.T).T + np.array([x, y])
        pts = [self.viewport.world_to_screen((p[0], p[1]), rect) for p in world]
        pygame.draw.lines(self.screen, color, False, pts, 1)

    def trajectory_diagnostics(self):
        try:
            traj = generate_trajectory(self.path)
        except Exception:
            return None, [], [], [], []

        obstacles = build_field_obstacles(self.params)
        regions = build_field_surface_regions(self.params)
        samples = []
        collisions = []
        surfaces = []
        invalid_points = []
        if len(traj.xy_samples) == 0:
            return traj, samples, collisions, surfaces, invalid_points

        dense_n = max(20, 3 * len(traj.xy_samples))
        times = np.linspace(0.0, traj.total_time if traj.total_time > 1e-6 else 1e-3, dense_n)
        for t in times:
            pos, theta, _ = robot_pose_from_trajectory(traj, float(t))
            coll = robot_body_collides(pos, theta, self.params, obstacles)
            surf = any(robot_body_in_region(pos, theta, self.params, reg) for reg in regions)
            trench = any("trench" in obs.tags and robot_body_collides(pos, theta, self.params, [obs]) for obs in obstacles)
            samples.append((pos, theta))
            collisions.append(coll)
            surfaces.append(surf or trench)
        for wp in self.path.waypoints:
            pt = np.array([wp.x, wp.y], dtype=float)
            inside = False
            for obs in obstacles:
                if point_in_obb(pt, obs.center, obs.half_extents, obs.theta_rad):
                    inside = True
                    break
            invalid_points.append(inside)
        return traj, samples, collisions, surfaces, invalid_points

    def draw_path(self, rect):
        diag = self.trajectory_diagnostics()
        if diag is None:
            return
        traj, samples, collisions, surfaces, invalid_points = diag

        if len(self.path.waypoints) > 1:
            control_pts = [self.viewport.world_to_screen((wp.x, wp.y), rect) for wp in self.path.waypoints]
            pygame.draw.lines(self.screen, (255, 210, 95), False, control_pts, 2)

        if samples:
            traj_pts = [self.viewport.world_to_screen((p[0][0], p[0][1]), rect) for p in samples]
            for i in range(len(traj_pts) - 1):
                color = (110, 235, 130)
                if collisions[i] or collisions[i + 1]:
                    color = (255, 85, 85)
                elif surfaces[i] or surfaces[i + 1]:
                    color = (245, 215, 75)
                pygame.draw.line(self.screen, color, traj_pts[i], traj_pts[i + 1], 3)

            sample_idx = np.linspace(0, len(samples) - 1, min(10, len(samples)), dtype=int)
            for idx in sample_idx:
                pos, theta = samples[idx]
                ghost_color = (255, 85, 85) if collisions[idx] else ((245, 215, 75) if surfaces[idx] else (210, 110, 240))
                self.draw_robot_ghost(pos[0], pos[1], theta, rect, color=ghost_color)

        for i, wp in enumerate(self.path.waypoints):
            sx, sy = self.viewport.world_to_screen((wp.x, wp.y), rect)
            bad = invalid_points[i] if i < len(invalid_points) else False
            if bad:
                col = (255, 70, 70)
            elif i == self.selected_idx:
                col = (255, 255, 100)
            else:
                col = (255, 140, 70)
            pygame.draw.circle(self.screen, col, (sx, sy), 8)
            pygame.draw.circle(self.screen, (20, 20, 20), (sx, sy), 8, 2)
            th = math.radians(wp.theta_deg)
            ex = sx + 38 * math.cos(th)
            ey = sy - 38 * math.sin(th)
            pygame.draw.line(self.screen, (120, 220, 255), (sx, sy), (int(ex), int(ey)), 3)
            pygame.draw.circle(self.screen, (120, 220, 255), (int(ex), int(ey)), 4)
            self.screen.blit(self.small.render(str(i), True, (240, 242, 246)), (sx + 10, sy - 16))

    def draw_ui(self):
        modes = ["numeric", "underlay", "both"]
        source_format = self.path.metadata.get("source_format", "native")
        ang_vel = float(self.path.metadata.get("max_angular_velocity_deg_s", 540.0))
        ang_accel = float(self.path.metadata.get("max_angular_acceleration_deg_s2", 720.0))
        mode_name = "linear" if self.constraint_mode == 0 else "angular"

        self.screen.blit(self.font.render(f"Path: {self.path.name}", True, (240, 242, 246)), (18, 14))
        self.screen.blit(
            self.small.render(
                f"Waypoints: {len(self.path.waypoints)}   max_vel: {self.path.max_velocity:.2f} m/s   max_accel: {self.path.max_acceleration:.2f} m/s^2   "
                f"ang_vel: {ang_vel:.1f} deg/s   ang_accel: {ang_accel:.1f} deg/s^2   edit: {mode_name}   "
                f"spline: {'ON' if self.path.use_spline else 'OFF'}   follow heading: {'ON' if self.follow_heading else 'OFF'}   "
                f"render: {modes[self.render_mode]}   snap: {'ON' if self.snap_enabled else 'OFF'}   src: {source_format}",
                True,
                (180, 185, 195),
            ),
            (18, 46),
        )
        self.screen.blit(self.small.render(self.message, True, (110, 235, 130)), (18, 74))
        help_text = "LMB add/select+drag | RMB drag heading | F follow | M toggle linear/angular edits | arrows edit active constraints | Q/E heading | I insert | Del delete | C spline | S save JSON | P export .path | L load | T render | G snap | H labels | R rename | N new"
        self.screen.blit(self.small.render(help_text, True, (160, 165, 175)), (18, 98))


    def handle_key(self, event):
        if event.key == pygame.K_ESCAPE:
            return False
        elif event.key == pygame.K_q and self.selected_idx is not None:
            self.path.waypoints[self.selected_idx].theta_deg += 5.0
        elif event.key == pygame.K_e and self.selected_idx is not None:
            self.path.waypoints[self.selected_idx].theta_deg -= 5.0
        elif event.key == pygame.K_m:
            self.constraint_mode = 1 - self.constraint_mode
            self.set_message(f"Constraint edit mode: {'linear' if self.constraint_mode == 0 else 'angular'}")

        elif event.key == pygame.K_UP:
            if self.constraint_mode == 0:
                self.path.max_velocity += 0.1
            else:
                cur = float(self.path.metadata.get("max_angular_velocity_deg_s", 540.0))
                self.path.metadata["max_angular_velocity_deg_s"] = cur + 15.0

        elif event.key == pygame.K_DOWN:
            if self.constraint_mode == 0:
                self.path.max_velocity = max(0.1, self.path.max_velocity - 0.1)
            else:
                cur = float(self.path.metadata.get("max_angular_velocity_deg_s", 540.0))
                self.path.metadata["max_angular_velocity_deg_s"] = max(15.0, cur - 15.0)

        elif event.key == pygame.K_RIGHT:
            if self.constraint_mode == 0:
                self.path.max_acceleration += 0.1
            else:
                cur = float(self.path.metadata.get("max_angular_acceleration_deg_s2", 720.0))
                self.path.metadata["max_angular_acceleration_deg_s2"] = cur + 30.0

        elif event.key == pygame.K_LEFT:
            if self.constraint_mode == 0:
                self.path.max_acceleration = max(0.1, self.path.max_acceleration - 0.1)
            else:
                cur = float(self.path.metadata.get("max_angular_acceleration_deg_s2", 720.0))
                self.path.metadata["max_angular_acceleration_deg_s2"] = max(30.0, cur - 30.0)
        elif event.key == pygame.K_f:
            self.follow_heading = not self.follow_heading
            self.apply_follow_heading()
            self.set_message(f"Follow heading {'enabled' if self.follow_heading else 'disabled'}")
        elif event.key == pygame.K_i:
            self.insert_after_selected()
        elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            self.delete_selected()
        elif event.key == pygame.K_c:
            self.path.use_spline = not self.path.use_spline
            self.set_message(f"Spline {'enabled' if self.path.use_spline else 'disabled'}")
        elif event.key == pygame.K_s:
            self.save_current()
        elif event.key == pygame.K_p:
            self.export_pathplanner()
        elif event.key == pygame.K_l:
            self.load_interactive()
        elif event.key == pygame.K_r:
            self.rename()
        elif event.key == pygame.K_n:
            self.path = self.default_path()
            self.selected_idx = 0 if self.path.waypoints else None
            self.set_message("Started new path")
        elif event.key == pygame.K_TAB:
            if self.path.waypoints:
                self.selected_idx = 0 if self.selected_idx is None else (self.selected_idx + 1) % len(self.path.waypoints)
        elif event.key == pygame.K_t:
            self.render_mode = (self.render_mode + 1) % 3
        elif event.key == pygame.K_g:
            self.snap_enabled = not self.snap_enabled
            self.set_message(f"Snap {'enabled' if self.snap_enabled else 'disabled'}")
        elif event.key == pygame.K_h:
            self.show_labels = not self.show_labels
            self.set_message(f"Labels {'enabled' if self.show_labels else 'disabled'}")
        return True

    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            rect = self.field_rect()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_key(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if rect.collidepoint(event.pos):
                        if event.button == 1:
                            hit = self.find_waypoint_at(event.pos, rect)
                            if hit is not None:
                                self.selected_idx = hit
                                self.dragging = True
                            else:
                                self.add_waypoint(self.viewport.screen_to_world(event.pos, rect))
                                self.dragging = True
                        elif event.button == 3:
                            hit = self.find_waypoint_at(event.pos, rect)
                            if hit is not None:
                                self.selected_idx = hit
                                self.heading_drag = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                    elif event.button == 3:
                        self.heading_drag = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging and self.selected_idx is not None and rect.collidepoint(event.pos):
                        x, y = self.snap_world_point(self.viewport.screen_to_world(event.pos, rect))
                        self.path.waypoints[self.selected_idx].x = x
                        self.path.waypoints[self.selected_idx].y = y
                        self.apply_follow_heading()
                    elif self.heading_drag and self.selected_idx is not None:
                        sx, sy = self.viewport.world_to_screen((self.path.waypoints[self.selected_idx].x, self.path.waypoints[self.selected_idx].y), rect)
                        dx, dy = event.pos[0] - sx, sy - event.pos[1]
                        if abs(dx) + abs(dy) > 2:
                            self.path.waypoints[self.selected_idx].theta_deg = math.degrees(math.atan2(dy, dx))

            self.screen.fill((18, 20, 24))
            draw_field(self.screen, self.viewport, rect, self.palette, font=self.small, params=self.params, flags=self.current_flags())
            self.draw_path(rect)
            self.draw_ui()
            pygame.display.flip()
        pygame.quit()


def main():
    App().run()


if __name__ == "__main__":
    main()
