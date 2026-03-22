
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

AUTO_CONTROL_DISTANCE_FACTOR = 0.4


def is_pathplanner_payload(data: dict[str, Any]) -> bool:
    return isinstance(data, dict) and "waypoints" in data and "globalConstraints" in data and "goalEndState" in data


def _translation_from_json(node: dict[str, Any] | None) -> tuple[float, float] | None:
    if node is None:
        return None
    return float(node["x"]), float(node["y"])


def _translation_to_json(pt: tuple[float, float] | None) -> dict[str, float] | None:
    if pt is None:
        return None
    return {"x": float(pt[0]), "y": float(pt[1])}


def _heading_from_controls(
    prev_control: tuple[float, float] | None,
    anchor: tuple[float, float],
    next_control: tuple[float, float] | None,
    prev_anchor: tuple[float, float] | None,
    next_anchor: tuple[float, float] | None,
) -> float:
    candidates: list[tuple[float, float]] = []
    if next_control is not None:
        candidates.append((next_control[0] - anchor[0], next_control[1] - anchor[1]))
    if prev_control is not None:
        candidates.append((anchor[0] - prev_control[0], anchor[1] - prev_control[1]))
    if next_anchor is not None:
        candidates.append((next_anchor[0] - anchor[0], next_anchor[1] - anchor[1]))
    if prev_anchor is not None:
        candidates.append((anchor[0] - prev_anchor[0], anchor[1] - prev_anchor[1]))
    for dx, dy in candidates:
        if abs(dx) + abs(dy) > 1e-9:
            return math.degrees(math.atan2(dy, dx))
    return 0.0


def load_pathplanner_dict(data: dict[str, Any]) -> dict[str, Any]:
    waypoints = []
    raw_waypoints = data.get("waypoints", [])
    for i, wp in enumerate(raw_waypoints):
        anchor = _translation_from_json(wp["anchor"])
        prev_control = _translation_from_json(wp.get("prevControl"))
        next_control = _translation_from_json(wp.get("nextControl"))
        prev_anchor = _translation_from_json(raw_waypoints[i - 1]["anchor"]) if i > 0 else None
        next_anchor = _translation_from_json(raw_waypoints[i + 1]["anchor"]) if i < len(raw_waypoints) - 1 else None
        theta_deg = _heading_from_controls(prev_control, anchor, next_control, prev_anchor, next_anchor)
        waypoints.append({"x": anchor[0], "y": anchor[1], "theta_deg": theta_deg})

    constraints = data.get("globalConstraints", {})
    metadata = {
        "source_format": "pathplanner",
        "pathplanner_version": data.get("version"),
        "goal_end_state": data.get("goalEndState"),
        "ideal_starting_state": data.get("idealStartingState"),
        "rotation_targets": data.get("rotationTargets", []),
        "point_towards_zones": data.get("pointTowardsZones", []),
        "constraint_zones": data.get("constraintZones", []),
        "event_markers": data.get("eventMarkers", []),
        "reversed": bool(data.get("reversed", False)),
        "folder": data.get("folder"),
        "preview_starting_state": data.get("previewStartingState"),
        "use_default_constraints": bool(data.get("useDefaultConstraints", False)),
        "prevent_flipping": bool(data.get("preventFlipping", False)),
        "alliance_locked": bool(data.get("allianceLocked", False)),
        "max_angular_velocity_deg_s": float(constraints.get("maxAngularVelocity", 540.0)),
        "max_angular_acceleration_deg_s2": float(constraints.get("maxAngularAcceleration", 720.0)),
    }
    return {
        "name": data.get("name") or Path(str(data.get("pathName", "Imported Path"))).stem,
        "waypoints": waypoints,
        "max_velocity": float(constraints.get("maxVelocity", 3.5)),
        "max_acceleration": float(constraints.get("maxAcceleration", 2.5)),
        "use_spline": True,
        "samples_per_seg": 28,
        "metadata": metadata,
    }


def _auto_control_points(
    anchor: tuple[float, float],
    heading_deg: float,
    prev_anchor: tuple[float, float] | None,
    next_anchor: tuple[float, float] | None,
) -> dict[str, Any]:
    theta = math.radians(float(heading_deg))
    dx = math.cos(theta)
    dy = math.sin(theta)
    prev_control = None
    next_control = None
    if prev_anchor is not None:
        d = math.dist(anchor, prev_anchor) * AUTO_CONTROL_DISTANCE_FACTOR
        prev_control = (anchor[0] - d * dx, anchor[1] - d * dy)
    if next_anchor is not None:
        d = math.dist(anchor, next_anchor) * AUTO_CONTROL_DISTANCE_FACTOR
        next_control = (anchor[0] + d * dx, anchor[1] + d * dy)
    return {
        "prevControl": _translation_to_json(prev_control),
        "anchor": _translation_to_json(anchor),
        "nextControl": _translation_to_json(next_control),
    }


def export_pathplanner_dict(pathspec_dict: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(pathspec_dict.get("metadata") or {})
    waypoints_in = pathspec_dict["waypoints"]
    anchors = [(float(wp["x"]), float(wp["y"])) for wp in waypoints_in]
    headings = [float(wp.get("theta_deg", 0.0)) for wp in waypoints_in]
    waypoints = []
    for i, anchor in enumerate(anchors):
        prev_anchor = anchors[i - 1] if i > 0 else None
        next_anchor = anchors[i + 1] if i < len(anchors) - 1 else None
        waypoints.append(_auto_control_points(anchor, headings[i], prev_anchor, next_anchor))

    max_velocity = float(pathspec_dict.get("max_velocity", 3.5))
    max_acceleration = float(pathspec_dict.get("max_acceleration", 2.5))
    goal_end_state = metadata.get("goal_end_state") or {"velocity": 0.0, "rotation": headings[-1] if headings else 0.0}
    ideal_starting_state = metadata.get("ideal_starting_state")
    if ideal_starting_state is None and headings:
        ideal_starting_state = {"velocity": 0.0, "rotation": headings[0]}

    return {
        "version": str(metadata.get("pathplanner_version", "2025.0")),
        "name": pathspec_dict.get("name", "Exported Path"),
        "waypoints": waypoints,
        "globalConstraints": {
            "maxVelocity": max_velocity,
            "maxAcceleration": max_acceleration,
            "maxAngularVelocity": float(metadata.get("max_angular_velocity_deg_s", 540.0)),
            "maxAngularAcceleration": float(metadata.get("max_angular_acceleration_deg_s2", 720.0)),
            "nominalVoltage": float(metadata.get("nominal_voltage", 12.0)),
            "unlimited": False,
        },
        "idealStartingState": ideal_starting_state,
        "goalEndState": goal_end_state,
        "rotationTargets": list(metadata.get("rotation_targets", [])),
        "pointTowardsZones": list(metadata.get("point_towards_zones", [])),
        "constraintZones": list(metadata.get("constraint_zones", [])),
        "eventMarkers": list(metadata.get("event_markers", [])),
        "reversed": bool(metadata.get("reversed", False)),
        "folder": metadata.get("folder"),
        "previewStartingState": metadata.get("preview_starting_state"),
        "useDefaultConstraints": bool(metadata.get("use_default_constraints", False)),
        "preventFlipping": bool(metadata.get("prevent_flipping", False)),
        "allianceLocked": bool(metadata.get("alliance_locked", False)),
    }


def load_pathplanner_file(path_file: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path_file).read_text())
    return load_pathplanner_dict(data)


def save_pathplanner_file(pathspec_dict: dict[str, Any], out_path: str | Path) -> None:
    payload = export_pathplanner_dict(pathspec_dict)
    Path(out_path).write_text(json.dumps(payload, indent=2))
