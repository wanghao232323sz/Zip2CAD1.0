"""Read the deliberately small v1 COLLADA subset without vertex deduplication.

General-purpose mesh loaders can merge coincident vertices even with process=False.
Exchange vertex/face references must be checked against the original source arrays.
"""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from zip2cad.dae_scene import LoadedScene


def load_exchange_geometry(path: Path) -> LoadedScene:
    document = ET.parse(path)
    root = document.getroot()
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    unit = root.find("c:asset/c:unit", ns)
    axis = root.find("c:asset/c:up_axis", ns)
    if unit is None or axis is None or float(unit.get("meter", "nan")) != 1 or axis.text.strip() != "Z_UP":
        raise ValueError("Exchange DAE must explicitly declare metre units and Z_UP")
    geometries = root.findall("c:library_geometries/c:geometry", ns)
    instances = root.findall(".//c:instance_geometry", ns)
    nodes = root.findall("c:library_visual_scenes/c:visual_scene/c:node", ns)
    if len(geometries) != 1 or len(instances) != 1 or len(nodes) != 1:
        raise ValueError("Exchange DAE must contain exactly one flat geometry instance")
    if any(node.tag.rsplit('}', 1)[-1] != 'instance_geometry' for node in nodes[0]):
        raise ValueError("World-coordinate exchange mesh cannot have scene transforms or nested nodes")
    geometry = geometries[0]
    if instances[0].get("url") != f"#{geometry.get('id')}":
        raise ValueError("Geometry instance reference mismatch")
    mesh = geometry.find("c:mesh", ns)
    triangles = mesh.findall("c:triangles", ns) if mesh is not None else []
    if len(triangles) != 1:
        raise ValueError("Expected one triangles element")
    inputs = triangles[0].findall("c:input", ns)
    if len(inputs) != 1 or inputs[0].get("semantic") != "VERTEX" or inputs[0].get("offset") != "0":
        raise ValueError("Expected only offset-zero VERTEX input")
    vertex = mesh.find("c:vertices", ns)
    if vertex is None or inputs[0].get("source") != f"#{vertex.get('id')}":
        raise ValueError("Vertex source mismatch")
    position_inputs = vertex.findall("c:input", ns)
    if len(position_inputs) != 1 or position_inputs[0].get("semantic") != "POSITION":
        raise ValueError("Expected only POSITION vertex input")
    sources = mesh.findall("c:source", ns)
    source = next((item for item in sources if f"#{item.get('id')}" == position_inputs[0].get("source")), None)
    if source is None:
        raise ValueError("Missing position source")
    array = source.find("c:float_array", ns)
    accessor = source.find("c:technique_common/c:accessor", ns)
    if array is None or accessor is None or accessor.get("stride") != "3" or accessor.get("source") != f"#{array.get('id')}":
        raise ValueError("Invalid position accessor")
    coordinates = np.asarray((array.text or "").split(), dtype=np.float64)
    if len(coordinates) != int(array.get("count")) or len(coordinates) != int(accessor.get("count")) * 3:
        raise ValueError("Position count mismatch")
    vertices = coordinates.reshape(-1, 3)
    indices = triangles[0].find("c:p", ns)
    if indices is None:
        raise ValueError("Missing triangle indices")
    faces = np.asarray((indices.text or "").split(), dtype=np.int64).reshape(-1, 3)
    if len(faces) != int(triangles[0].get("count")) or not len(faces) or not len(vertices):
        raise ValueError("Triangle count mismatch or empty mesh")
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Non-finite coordinates or out-of-range indices")
    return LoadedScene(vertices, faces, np.zeros(len(faces), dtype=np.int32),
                       [geometry.get("id")], unit.get("name", "meter"), 1.0, "Z_UP", 1, 1)
