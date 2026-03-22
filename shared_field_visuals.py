from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
try:
    import pygame
except ModuleNotFoundError:
    pygame = None

from shared_sim_core import (
    ALLIANCE_ZONE_DEPTH_M,
    APRILTAG_PANEL_M,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    HUMAN_START_LINE_FROM_WALL_M,
    PACK_HEIGHT_M,
    PACK_WIDTH_M,
    SimParams,
    build_apriltag_metadata,
    build_field_obstacles,
    build_field_reference_points,
    build_field_surface_regions,
    rotmat,
)

RectF = Tuple[float, float, float, float]


@dataclass(frozen=True)
class FieldPalette:
    bg: Tuple[int, int, int] = (23, 25, 29)
    field_fill: Tuple[int, int, int] = (70, 70, 72)
    field_edge: Tuple[int, int, int] = (240, 242, 246)
    zone_line: Tuple[int, int, int] = (162, 168, 180)
    center_line: Tuple[int, int, int] = (250, 250, 252)
    pack_outline: Tuple[int, int, int] = (245, 214, 66)
    human_line: Tuple[int, int, int] = (205, 208, 215)
    red_zone: Tuple[int, int, int] = (120, 60, 58)
    neutral_zone: Tuple[int, int, int] = (110, 108, 62)
    blue_zone: Tuple[int, int, int] = (56, 69, 122)
    bump_fill_red: Tuple[int, int, int] = (215, 40, 32)
    bump_fill_blue: Tuple[int, int, int] = (25, 100, 210)
    bump_outline: Tuple[int, int, int] = (235, 235, 235)
    obstacle_outline: Tuple[int, int, int] = (30, 30, 32)
    hub_fill_red: Tuple[int, int, int] = (196, 77, 77)
    hub_fill_blue: Tuple[int, int, int] = (104, 151, 214)
    trench_fill: Tuple[int, int, int] = (38, 38, 42)
    trench_opening: Tuple[int, int, int] = (95, 95, 100)
    depot_fill: Tuple[int, int, int] = (240, 200, 72)
    tag_fill: Tuple[int, int, int] = (240, 240, 240)
    tag_edge: Tuple[int, int, int] = (25, 25, 25)
    reference: Tuple[int, int, int] = (200, 220, 250)


@dataclass(frozen=True)
class RenderFlags:
    show_underlay: bool = False
    show_numeric_geometry: bool = True
    show_labels: bool = True
    show_centers: bool = False
    show_clearance_boxes: bool = False
    show_apriltags: bool = True
    underlay_alpha: int = 75


def field_rects() -> Dict[str, RectF]:
    xmin = -FIELD_LENGTH_M / 2.0
    ymin = -FIELD_WIDTH_M / 2.0
    xmax = FIELD_LENGTH_M / 2.0
    return {
        "field": (xmin, ymin, FIELD_LENGTH_M, FIELD_WIDTH_M),
        "red_zone": (xmin, ymin, ALLIANCE_ZONE_DEPTH_M, FIELD_WIDTH_M),
        "neutral_zone": (xmin + ALLIANCE_ZONE_DEPTH_M, ymin, FIELD_LENGTH_M - 2.0 * ALLIANCE_ZONE_DEPTH_M, FIELD_WIDTH_M),
        "blue_zone": (xmax - ALLIANCE_ZONE_DEPTH_M, ymin, ALLIANCE_ZONE_DEPTH_M, FIELD_WIDTH_M),
        "pack": (-PACK_WIDTH_M / 2.0, -PACK_HEIGHT_M / 2.0, PACK_WIDTH_M, PACK_HEIGHT_M),
    }


def world_rect_to_screen(r: RectF, viewport, screen_rect: pygame.Rect) -> pygame.Rect:
    x, y, w, h = r
    p0 = viewport.world_to_screen((x, y), screen_rect)
    p1 = viewport.world_to_screen((x + w, y + h), screen_rect)
    sx0, sy0 = p0
    sx1, sy1 = p1
    return pygame.Rect(min(sx0, sx1), min(sy0, sy1), abs(sx1 - sx0), abs(sy1 - sy0))


def _world_poly_to_screen(points_world, viewport, screen_rect):
    return [viewport.world_to_screen((x, y), screen_rect) for x, y in points_world]


def _obb_corners(center, half_extents, theta):
    hx, hy = half_extents
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=float)
    R = rotmat(theta)
    return (local @ R.T) + center


def _candidate_underlay_paths() -> list[Path]:
    base = Path(__file__).resolve().parent
    return [
        base / "rebuilt_field_underlay.png",
        base / "rebuilt_field_underlay_clean.png",
        base / "field_reference_topdown.png",
        base / "field_underlay.png",
        base / "field.png",
    ]


@lru_cache(maxsize=1)
def _load_background_surface() -> Optional[pygame.Surface]:
    for path in _candidate_underlay_paths():
        if path.exists():
            try:
                return pygame.image.load(str(path)).convert_alpha()
            except Exception:
                try:
                    return pygame.image.load(str(path))
                except Exception:
                    continue
    return None


@lru_cache(maxsize=1)
def _load_background_array() -> Optional[np.ndarray]:
    for path in _candidate_underlay_paths():
        if path.exists():
            try:
                return mpimg.imread(str(path))
            except Exception:
                continue
    return None


@lru_cache(maxsize=16)
def _scaled_background(size: Tuple[int, int], alpha: int) -> Optional[pygame.Surface]:
    src = _load_background_surface()
    if src is None or size[0] <= 1 or size[1] <= 1:
        return None
    out = pygame.transform.smoothscale(src, size)
    if out.get_alpha() is None:
        out = out.convert_alpha()
    out.set_alpha(alpha)
    return out


def _draw_world_rect_outline(surface, viewport, screen_rect, rect_world: RectF, color, width=2):
    pygame.draw.rect(surface, color, world_rect_to_screen(rect_world, viewport, screen_rect), width)


def _draw_world_vline(surface, viewport, screen_rect, x: float, color, width: int):
    p0 = viewport.world_to_screen((x, -FIELD_WIDTH_M / 2.0), screen_rect)
    p1 = viewport.world_to_screen((x, FIELD_WIDTH_M / 2.0), screen_rect)
    pygame.draw.line(surface, color, p0, p1, width)


def _draw_world_hline(surface, viewport, screen_rect, y: float, color, width: int):
    p0 = viewport.world_to_screen((-FIELD_LENGTH_M / 2.0, y), screen_rect)
    p1 = viewport.world_to_screen((FIELD_LENGTH_M / 2.0, y), screen_rect)
    pygame.draw.line(surface, color, p0, p1, width)


def draw_zones(surface, viewport, screen_rect, palette: FieldPalette):
    rects = field_rects()
    pygame.draw.rect(surface, palette.red_zone, world_rect_to_screen(rects["red_zone"], viewport, screen_rect))
    pygame.draw.rect(surface, palette.neutral_zone, world_rect_to_screen(rects["neutral_zone"], viewport, screen_rect))
    pygame.draw.rect(surface, palette.blue_zone, world_rect_to_screen(rects["blue_zone"], viewport, screen_rect))
    pygame.draw.rect(surface, palette.field_fill, world_rect_to_screen(rects["field"], viewport, screen_rect), 0)
    # re-overlay zones so carpet stays contained while keeping field fill around missing art
    pygame.draw.rect(surface, palette.red_zone, world_rect_to_screen(rects["red_zone"], viewport, screen_rect))
    pygame.draw.rect(surface, palette.neutral_zone, world_rect_to_screen(rects["neutral_zone"], viewport, screen_rect))
    pygame.draw.rect(surface, palette.blue_zone, world_rect_to_screen(rects["blue_zone"], viewport, screen_rect))


def draw_obstacles(surface, viewport, screen_rect, params: SimParams, palette: FieldPalette, flags: RenderFlags):
    for obs in build_field_obstacles(params):
        corners = _world_poly_to_screen(_obb_corners(obs.center, obs.half_extents, obs.theta_rad), viewport, screen_rect)
        fill = palette.trench_fill
        if "hub" in obs.tags:
            fill = palette.hub_fill_red if "red" in obs.tags else palette.hub_fill_blue
        elif "depot" in obs.tags:
            fill = palette.depot_fill
        pygame.draw.polygon(surface, fill, corners)
        pygame.draw.polygon(surface, palette.obstacle_outline, corners, 2)
        if flags.show_clearance_boxes and obs.min_clear_height_m > 0.0:
            inner = obs.half_extents.copy()
            inner[0] = min(inner[0], 0.5 * obs.min_clear_height_m)
            icorners = _world_poly_to_screen(_obb_corners(obs.center, inner, obs.theta_rad), viewport, screen_rect)
            pygame.draw.polygon(surface, palette.trench_opening, icorners, 1)


def draw_bump_regions(surface, viewport, screen_rect, params: SimParams, palette: FieldPalette):
    for region in build_field_surface_regions(params):
        corners = _world_poly_to_screen(_obb_corners(region.center, region.half_extents, region.theta_rad), viewport, screen_rect)
        fill = palette.bump_fill_red if "red" in region.tags else palette.bump_fill_blue
        poly_surf = pygame.Surface((screen_rect.width, screen_rect.height), pygame.SRCALPHA)
        shifted = [(x - screen_rect.left, y - screen_rect.top) for x, y in corners]
        pygame.draw.polygon(poly_surf, (*fill, 170), shifted)
        surface.blit(poly_surf, screen_rect.topleft)
        pygame.draw.polygon(surface, palette.bump_outline, corners, 2)
        # slope arrow in the x direction, because the bump crest runs along the long axis.
        start = region.center + np.array([-0.22, 0.0], dtype=float)
        end = region.center + np.array([0.22, 0.0], dtype=float)
        a = viewport.world_to_screen(tuple(start), screen_rect)
        b = viewport.world_to_screen(tuple(end), screen_rect)
        pygame.draw.line(surface, palette.bump_outline, a, b, 2)
        pygame.draw.circle(surface, palette.bump_outline, b, 4)


def draw_pack(surface, viewport, screen_rect, palette: FieldPalette):
    _draw_world_rect_outline(surface, viewport, screen_rect, field_rects()["pack"], palette.pack_outline, 2)


def draw_apriltags_pygame(surface, viewport, screen_rect, params: SimParams, palette: FieldPalette):
    half = APRILTAG_PANEL_M / 2.0
    for tag in build_apriltag_metadata(params):
        rect = (tag.center[0] - half, tag.center[1] - half, APRILTAG_PANEL_M, APRILTAG_PANEL_M)
        pygame.draw.rect(surface, palette.tag_fill, world_rect_to_screen(rect, viewport, screen_rect))
        pygame.draw.rect(surface, palette.tag_edge, world_rect_to_screen(rect, viewport, screen_rect), 1)


def draw_reference_points(surface, viewport, screen_rect, params: SimParams, palette: FieldPalette):
    for ref in build_field_reference_points(params):
        sx, sy = viewport.world_to_screen(tuple(ref.pos), screen_rect)
        pygame.draw.circle(surface, palette.reference, (sx, sy), 3)


def draw_labels(surface, viewport, screen_rect, params: SimParams, font, palette: FieldPalette):
    if font is None:
        return
    for ref in build_field_reference_points(params):
        sx, sy = viewport.world_to_screen(tuple(ref.pos), screen_rect)
        txt = font.render(ref.name.replace("_center", ""), True, palette.field_edge)
        surface.blit(txt, (sx + 6, sy - 12))


def draw_field(surface, viewport, screen_rect, palette: Optional[FieldPalette] = None, font=None, params: Optional[SimParams] = None, flags: Optional[RenderFlags] = None):
    palette = palette or FieldPalette()
    params = params or SimParams()
    flags = flags or RenderFlags()
    surface.fill(palette.bg)

    if flags.show_numeric_geometry:
        draw_zones(surface, viewport, screen_rect, palette)

    if flags.show_underlay:
        field_screen = world_rect_to_screen(field_rects()["field"], viewport, screen_rect)
        img = _scaled_background((field_screen.width, field_screen.height), flags.underlay_alpha)
        if img is not None:
            surface.blit(img, field_screen.topleft)

    if flags.show_numeric_geometry:
        draw_pack(surface, viewport, screen_rect, palette)
        draw_bump_regions(surface, viewport, screen_rect, params, palette)
        draw_obstacles(surface, viewport, screen_rect, params, palette, flags)

        rects = field_rects()
        _draw_world_rect_outline(surface, viewport, screen_rect, rects["field"], palette.field_edge, 2)
        xmin = -FIELD_LENGTH_M / 2.0
        xmax = FIELD_LENGTH_M / 2.0
        _draw_world_vline(surface, viewport, screen_rect, xmin + ALLIANCE_ZONE_DEPTH_M, palette.zone_line, 1)
        _draw_world_vline(surface, viewport, screen_rect, xmax - ALLIANCE_ZONE_DEPTH_M, palette.zone_line, 1)
        _draw_world_vline(surface, viewport, screen_rect, 0.0, palette.center_line, 1)
        _draw_world_vline(surface, viewport, screen_rect, xmin + HUMAN_START_LINE_FROM_WALL_M, palette.human_line, 1)
        _draw_world_vline(surface, viewport, screen_rect, xmax - HUMAN_START_LINE_FROM_WALL_M, palette.human_line, 1)

    if flags.show_apriltags:
        draw_apriltags_pygame(surface, viewport, screen_rect, params, palette)
    if flags.show_centers:
        draw_reference_points(surface, viewport, screen_rect, params, palette)
    if flags.show_labels:
        draw_labels(surface, viewport, screen_rect, params, font, palette)


def _patch(ax, poly_xy, fill, edge, lw=1.2, alpha=1.0, z=4):
    ax.add_patch(mpatches.Polygon(poly_xy, closed=True, facecolor=fill, edgecolor=edge, linewidth=lw, alpha=alpha, zorder=z))


def draw_field_matplotlib(ax, palette: Optional[FieldPalette] = None, params: Optional[SimParams] = None, flags: Optional[RenderFlags] = None):
    palette = palette or FieldPalette()
    params = params or SimParams()
    flags = flags or RenderFlags()
    rects = field_rects()
    xmin, ymin, fw, fh = rects["field"]
    xmax = xmin + fw
    ymax = ymin + fh

    if flags.show_numeric_geometry:
        for key, color in [("red_zone", palette.red_zone), ("neutral_zone", palette.neutral_zone), ("blue_zone", palette.blue_zone)]:
            x, y, w, h = rects[key]
            ax.add_patch(mpatches.Rectangle((x, y), w, h, facecolor=tuple(c / 255.0 for c in color), edgecolor="none", zorder=0))

    if flags.show_underlay:
        arr = _load_background_array()
        if arr is not None:
            ax.imshow(arr, extent=[xmin, xmax, ymin, ymax], origin="upper", aspect="auto", alpha=flags.underlay_alpha / 255.0, zorder=1)

    if flags.show_numeric_geometry:
        x, y, w, h = rects["pack"]
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=tuple(c / 255.0 for c in palette.pack_outline), linewidth=1.0, zorder=3))
        for reg in build_field_surface_regions(params):
            corners = _obb_corners(reg.center, reg.half_extents, reg.theta_rad)
            fill = palette.bump_fill_red if "red" in reg.tags else palette.bump_fill_blue
            _patch(ax, corners, tuple(c / 255.0 for c in fill), tuple(c / 255.0 for c in palette.bump_outline), alpha=0.65, z=4)
        for obs in build_field_obstacles(params):
            corners = _obb_corners(obs.center, obs.half_extents, obs.theta_rad)
            fill = palette.trench_fill
            if "hub" in obs.tags:
                fill = palette.hub_fill_red if "red" in obs.tags else palette.hub_fill_blue
            elif "depot" in obs.tags:
                fill = palette.depot_fill
            _patch(ax, corners, tuple(c / 255.0 for c in fill), tuple(c / 255.0 for c in palette.obstacle_outline), z=5)
        for tag in build_apriltag_metadata(params) if flags.show_apriltags else []:
            half = APRILTAG_PANEL_M / 2.0
            ax.add_patch(mpatches.Rectangle((tag.center[0] - half, tag.center[1] - half), APRILTAG_PANEL_M, APRILTAG_PANEL_M, facecolor="white", edgecolor="black", linewidth=0.5, zorder=6))
        ax.add_patch(mpatches.Rectangle((xmin, ymin), fw, fh, fill=False, edgecolor=tuple(c / 255.0 for c in palette.field_edge), linewidth=1.1, zorder=7))
        ax.axvline(xmin + ALLIANCE_ZONE_DEPTH_M, color=tuple(c / 255.0 for c in palette.zone_line), linewidth=0.8, zorder=2)
        ax.axvline(xmax - ALLIANCE_ZONE_DEPTH_M, color=tuple(c / 255.0 for c in palette.zone_line), linewidth=0.8, zorder=2)
        ax.axvline(0.0, color=tuple(c / 255.0 for c in palette.center_line), linewidth=0.8, zorder=2)
        ax.axvline(xmin + HUMAN_START_LINE_FROM_WALL_M, color=tuple(c / 255.0 for c in palette.human_line), linewidth=0.8, zorder=2)
        ax.axvline(xmax - HUMAN_START_LINE_FROM_WALL_M, color=tuple(c / 255.0 for c in palette.human_line), linewidth=0.8, zorder=2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("Field x (m)")
    ax.set_ylabel("Field y (m)")
