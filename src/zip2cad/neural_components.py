"""Finite solids from neural surface proposals, preserving unobserved grid gaps.

Neural predictions are hypotheses. CAD assembly requires independent model,
observation, novelty and geometry gates; a valid solid alone is not truth.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, triangulate


def polygon_solid(polygon, origin, basis, normal, thickness):
    polygon = orient(polygon, sign=1.)
    rings = [np.asarray(polygon.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in polygon.interiors]
    xy = np.vstack(rings)
    lookup = {tuple(np.round(p, 8)): i for i, p in enumerate(xy)}
    caps = []
    area = 0.
    for triangle in triangulate(polygon):
        if not polygon.covers(triangle):
            continue
        coords = np.asarray(orient(triangle, sign=1.).exterior.coords)[:3]
        caps.append([lookup[tuple(np.round(p, 8))] for p in coords]); area += triangle.area
    if not caps or abs(area - polygon.area) > max(1e-7, polygon.area * 1e-7):
        raise ValueError("Unreliable polygon triangulation; do not fill unknown regions")
    n = len(xy)
    faces = list(caps) + [[c+n, b+n, a+n] for a, b, c in caps]
    top_mask = [True] * len(caps) + [False] * len(caps)
    offset = 0
    for ring in rings:
        for i in range(len(ring)):
            a, b = offset+i, offset+(i+1) % len(ring)
            faces.extend([[a, a+n, b+n], [a, b+n, b]])
            top_mask.extend([False, False])
        offset += len(ring)
    top = np.asarray(origin) + xy @ np.asarray(basis).T
    vertices = np.vstack([top, top - np.asarray(normal) * thickness])
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("Proposal is not a consistently oriented solid")
    return dict(vertices_world_CAD_m=vertices.tolist(), triangles=faces, top_triangle_mask=top_mask,
                top_color=[170, 170, 166], side_color=[230, 230, 225],
                volume_CAD_units_cubed=float(mesh.volume), closed_solid_verified=True)


def extract_proposal_solids(points, cell_m=.8, plane_tolerance_m=.12, thickness_m=.12,
                            minimum_points=24, maximum_planes=16, seed=20261015):
    points = np.asarray(points, float)
    if not np.isfinite(points).all() or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Finite Nx3 predictions required")
    if min(cell_m, plane_tolerance_m, thickness_m) <= 0:
        raise ValueError("Positive geometry scales required")
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(points))
    components, rows = [], []
    for _ in range(maximum_planes):
        if len(remaining) < minimum_points:
            break
        best = np.empty(0, int)
        for __ in range(160):
            tri = points[rng.choice(remaining, 3, replace=False)]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            length = np.linalg.norm(normal)
            if length < 1e-8:
                continue
            normal /= length
            ids = remaining[abs((points[remaining] - tri[0]) @ normal) < plane_tolerance_m]
            if len(ids) > len(best):
                best = ids
        if len(best) < minimum_points:
            break
        remaining = remaining[~np.isin(remaining, best)]
        origin = points[best].mean(0)
        _, _, vh = np.linalg.svd(points[best] - origin, full_matrices=False)
        normal = vh[-1]
        if normal[np.argmax(abs(normal))] < 0:
            normal = -normal
        u = vh[0]; v = np.cross(normal, u); basis = np.c_[u, v]
        xy = (points[best] - origin) @ basis
        cells = np.floor(xy / cell_m).astype(int)
        unique = np.unique(cells, axis=0)
        # Union exactly occupied cells: no convex hull and no gap closing.
        occupied = unary_union([box(x*cell_m, y*cell_m, (x+1)*cell_m, (y+1)*cell_m) for x, y in unique])
        polygons = [occupied] if occupied.geom_type == "Polygon" else list(occupied.geoms)
        for polygon in polygons:
            if polygon.area < 1.:
                continue
            from shapely import contains_xy
            inside = contains_xy(polygon.buffer(1e-8), xy[:, 0], xy[:, 1])
            ids = best[inside]
            if len(ids) < minimum_points:
                continue
            name = f"neural_surface_{len(rows):03d}"
            row = dict(component_id=name, proposal_indices=ids.tolist(), area_m2=float(polygon.area),
                       plane_origin=origin.tolist(), plane_normal=normal.tolist(),
                       hole_count=len(polygon.interiors), source="neural_prediction_not_observation",
                       grid_boundary_resolution_m=cell_m, thickness_is_engineering_prior=True,
                       physical_instance_verified=False, native_edit_approved=False)
            try:
                part = polygon_solid(polygon, origin, basis, normal, thickness_m)
                part["component_id"] = name
                components.append(part)
                row["geometry_valid"] = True
            except ValueError as error:
                row.update(geometry_valid=False, geometry_rejection=str(error))
            rows.append(row)
    assigned = np.zeros(len(points), bool)
    for row in rows:
        assigned[row["proposal_indices"]] = True
    return dict(components=components, candidate_rows=rows,
                unassigned_point_indices=np.flatnonzero(~assigned).tolist())


def candidate_contacts(components, maximum_gap_m=.05):
    """Proximity is only a contact candidate; do not infer a verified connection."""
    edges = []
    for i, left in enumerate(components):
        p = np.asarray(left["vertices_world_CAD_m"])
        for right in components[i+1:]:
            q = np.asarray(right["vertices_world_CAD_m"])
            gap = np.maximum(np.maximum(p.min(0)-q.max(0), q.min(0)-p.max(0)), 0)
            lower_bound = float(np.linalg.norm(gap))
            if lower_bound <= maximum_gap_m:
                edges.append(dict(left=left["component_id"], right=right["component_id"],
                                  bounding_box_gap_lower_bound_m=lower_bound,
                                  relation="possible_contact_or_overlap", verified_connection=False,
                                  automatic_bridge_allowed=False))
    return edges
