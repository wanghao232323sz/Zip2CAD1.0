from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import trimesh


@dataclass
class LoadedScene:
    vertices_world_m: np.ndarray
    faces: np.ndarray
    face_instance: np.ndarray
    instance_names: list[str]
    source_unit_name: str
    source_unit_meter: float
    up_axis: str
    geometry_count: int
    scene_node_count: int


def read_collada_asset(path: Path) -> tuple[str, float, str]:
    """Read COLLADA unit metadata without loading the full XML tree."""
    unit_name = "meter"
    unit_meter = 1.0
    up_axis = "Z_UP"

    for _event, element in ElementTree.iterparse(path, events=("end",)):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "unit":
            unit_name = element.attrib.get("name", unit_name)
            unit_meter = float(element.attrib.get("meter", unit_meter))
        elif tag == "up_axis" and element.text:
            up_axis = element.text.strip()
        elif tag == "asset":
            break
        element.clear()
    return unit_name, unit_meter, up_axis


def _axis_transform(up_axis: str) -> np.ndarray:
    if up_axis == "Z_UP":
        return np.eye(4)
    if up_axis == "Y_UP":
        # Rotate +90 degrees about X: source Y becomes local Z.
        return trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    if up_axis == "X_UP":
        # Rotate -90 degrees about Y: source X becomes local Z.
        return trimesh.transformations.rotation_matrix(-np.pi / 2.0, [0.0, 1.0, 0.0])
    raise ValueError(f"Unsupported COLLADA up axis: {up_axis}")


def load_collada_scene(path: Path) -> LoadedScene:
    """Load every instantiated COLLADA mesh and retain face provenance.

    Trimesh/pycollada returns source coordinates in the COLLADA model unit. We
    apply the scene graph transforms first and then multiply by the explicit
    `<unit meter="...">` conversion factor.
    """
    unit_name, unit_meter, up_axis = read_collada_asset(path)
    scene = trimesh.load_scene(path, process=False)
    axis_transform = _axis_transform(up_axis)

    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    provenance_parts: list[np.ndarray] = []
    instance_names: list[str] = []
    vertex_offset = 0

    for instance_id, node_name in enumerate(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if len(geometry.vertices) == 0 or len(geometry.faces) == 0:
            continue

        world_transform = axis_transform @ np.asarray(transform, dtype=np.float64)
        vertices = trimesh.transform_points(
            np.asarray(geometry.vertices, dtype=np.float64),
            world_transform,
        )
        vertices *= unit_meter
        faces = np.asarray(geometry.faces, dtype=np.int64) + vertex_offset

        vertices_parts.append(vertices)
        faces_parts.append(faces)
        provenance_parts.append(
            np.full(len(faces), len(instance_names), dtype=np.int32)
        )
        instance_names.append(f"{node_name}|{geometry_name}")
        vertex_offset += len(vertices)

    if not vertices_parts:
        raise ValueError(f"No triangle geometry found in COLLADA scene: {path}")

    return LoadedScene(
        vertices_world_m=np.vstack(vertices_parts),
        faces=np.vstack(faces_parts),
        face_instance=np.concatenate(provenance_parts),
        instance_names=instance_names,
        source_unit_name=unit_name,
        source_unit_meter=unit_meter,
        up_axis=up_axis,
        geometry_count=len(scene.geometry),
        scene_node_count=len(scene.graph.nodes),
    )

